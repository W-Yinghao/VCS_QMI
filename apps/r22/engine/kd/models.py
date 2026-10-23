"""Pinned upstream CIFAR architectures and a separately named smoke model."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

PINS = {
    "mdistiller": "a08d46f10d6102bd6e3f258ca5ac880b020ea259",
    "repdistiller": "b84f547c5db6a35318d4671d7d5c4de74c822403",
}


def verify_upstream(root, backend):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing upstream checkout {root}; run the explicit bootstrap script first.")
    try:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError("A Git checkout is required for pinned source verification") from e
    if sha != PINS[backend] or dirty:
        raise ValueError(f"Expected clean {backend} {PINS[backend]}, got {sha}, tracked_changes={bool(dirty)}")
    return sha


def load_upstream(root, backend):
    verify_upstream(root, backend)
    relative = "mdistiller/models/cifar" if backend == "mdistiller" else "models"
    folder = Path(root).resolve() / relative
    module_name = "_kd_pinned_" + backend
    if module_name in sys.modules:
        existing = Path(sys.modules[module_name].__file__).resolve()
        if existing != folder / "__init__.py":
            raise RuntimeError("Cannot use two distinct checkouts of the same backend in one process")
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, folder / "__init__.py", submodule_search_locations=[str(folder)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise RuntimeError(f"Cannot import pinned {backend} model definitions; no fallback model is used: {e}") from e
    return module


class TinyModel(nn.Module):
    """CPU execution fixture only, never a published benchmark architecture."""
    def __init__(self, width=8, num_classes=4):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(3, width, 3, padding=1, bias=False), nn.BatchNorm2d(width),
                                   nn.ReLU(), nn.AvgPool2d(4), nn.Conv2d(width, width * 2, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(width * 2), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(width * 2, num_classes)

    def forward(self, x):
        feat = self.trunk(x).flatten(1)
        return self.fc(feat), feat


class ModelAdapter(nn.Module):
    def __init__(self, net, backend):
        super().__init__()
        self.net, self.backend = net, backend

    def forward(self, x):
        if self.backend == "repdistiller":
            features, logits = self.net(x, is_feat=True)
            feat = features[-1]
        elif self.backend == "mdistiller":
            logits, features = self.net(x)
            if "pooled_feat" not in features:
                raise KeyError("Pinned model must provide pooled_feat; do not silently choose a different layer")
            feat = features["pooled_feat"]
        else:
            return self.net(x)
        if feat.ndim != 2:
            raise ValueError(f"Expected pre-classifier pooled feature [B,D], got {tuple(feat.shape)}")
        return logits, feat


def create_model(model_cfg, num_classes, seed):
    backend, name = model_cfg["backend"], model_cfg["name"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if backend == "tiny":
            if name not in ("tiny_teacher", "tiny_student"):
                raise ValueError("Tiny backend accepts explicit smoke architecture names only")
            net = TinyModel(width=12 if name == "tiny_teacher" else 6, num_classes=num_classes)
        elif backend in PINS:
            module = load_upstream(model_cfg["source_root"], backend)
            registry = module.cifar_model_dict if backend == "mdistiller" else module.model_dict
            if name not in registry:
                raise ValueError(f"Unknown pinned architecture {name}; choices: {sorted(registry)}")
            constructor = registry[name][0] if backend == "mdistiller" else registry[name]
            net = constructor(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported backend {backend}")
    return ModelAdapter(net, backend)


def feature_dim(model):
    previous = model.training
    model.eval()
    with torch.no_grad():
        _, f = model(torch.zeros(2, 3, 32, 32))
    model.train(previous)
    return f.shape[1]


def load_model_checkpoint(model, path):
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        raise RuntimeError("Checkpoint must be weights-only loadable. Export trusted weights plus JSON metadata separately; unsafe pickle fallback is disabled.") from e
    if "student" in blob:
        state = blob["student"]
    elif "model" in blob:
        state = blob["model"]
    elif "state_dict" in blob:
        state = blob["state_dict"]
    elif all(torch.is_tensor(v) for v in blob.values()):
        state = blob
    else:
        raise ValueError("Unrecognized checkpoint structure")
    # Official checkpoints usually contain raw network keys, ours adapter keys.
    state = {k.removeprefix("module."): v for k, v in state.items()}
    if all(k.startswith("net.") for k in state):
        model.load_state_dict(state, strict=True)
    else:
        model.net.load_state_dict(state, strict=True)
    return blob
