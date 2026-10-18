"""Guarded CoMIR native host recipe with resumable optimizer and RNG state.

Uses the already installed, content-pinned official DenseUNet. It never calls the
V9 writer or reuses old training/latents. A=SHG, B=BF RGB, 23 x 32 x 32 recipe.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from v10 import adapter as A


@dataclass(frozen=True)
class CoMIRConfig:
    seed: int
    method: str = "CoMIR_native"
    epochs: int = 23
    steps_per_epoch: int = 32
    batch_size: int = 32
    learning_rate: float = 0.01
    temperature: float = 0.5
    weight_decay: float = 1e-5
    momentum: float = 0.9
    clip_norm: float = 1.0
    regularization_l1: float = 1e-4
    regularization_l2: float = 1e-4


def verify_official_source():
    pin = json.loads((A.REPO / "v10/supplement/COMIR_SOURCE.json").read_text())
    root = Path(pin["root"]).resolve()
    if root != A.P.COMIR.resolve():
        raise RuntimeError("Official CoMIR root differs from the pinned host")
    for name, expected in pin["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Official CoMIR source changed: " + name)
    # P.comir_models imports a generic 'models' package. Refuse namespace shadowing.
    for name, module in tuple(sys.modules.items()):
        if name == "models" or name.startswith("models."):
            filename = getattr(module, "__file__", None)
            if filename is None or not Path(filename).resolve().is_relative_to(root):
                raise RuntimeError("CoMIR model namespace already belongs to another package")
    return pin


class CoMIRTrainer:
    def __init__(self, config, guard, device):
        self.cfg, self.device = config, torch.device(device)
        self.source = verify_official_source()
        self.fit = A.fit_cores(guard, 1.0, config.seed)
        if len(self.fit) != 32:
            raise RuntimeError("BF CoMIR requires exactly the frozen FIT32")
        self.images = {}
        for core in self.fit:
            guard.check(core, {"FIT"}, "V10 CoMIR full FIT training")
            self.images[core] = A.P.load_img(guard.r6_root, core, ("shg_log_n", "bf_rgb_n"))
        self.moving_encoder, self.reference_encoder, self.architecture = A.P.comir_models(
            torch, config.seed, self.device)
        self.optimizers = [torch.optim.SGD(model.parameters(), lr=config.learning_rate,
                                          weight_decay=config.weight_decay, momentum=config.momentum)
                           for model in self.modules()]
        self.g = np.random.default_rng(3000 + config.seed)
        self.n_steps = 0

    def modules(self):
        return self.moving_encoder, self.reference_encoder

    def rng_state(self):
        return {"sample_rng": self.g.bit_generator.state}

    def restore_rng(self, state):
        self.g.bit_generator.state = state["sample_rng"]

    def draw(self):
        g, big = self.g, 182
        xa, xb = [], []
        for core in g.choice(self.fit, self.cfg.batch_size, replace=True):
            moving, reference = self.images[core]
            y, x = g.integers(0, A.P.SIZE - big, 2)
            joined = np.concatenate((moving[y:y + big, x:x + big, None],
                                     reference[y:y + big, x:x + big]), -1)
            sample = torch.as_tensor(joined, device=self.device, dtype=torch.float32).permute(2, 0, 1)
            if g.random() < 0.5:
                sample = sample.flip(-1)
            angle = math.radians(g.uniform(-180, 180))
            c, s = math.cos(angle), math.sin(angle)
            theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]],
                                 device=self.device, dtype=torch.float32)[None]
            grid = F.affine_grid(theta, (1, 4, big, big), align_corners=False)
            sample = F.grid_sample(sample[None], grid, mode="bilinear",
                                   padding_mode="reflection", align_corners=False)[0]
            start = (big - 128) // 2
            sample = sample[:, start:start + 128, start:start + 128]
            xa.append(sample[:1]); xb.append(sample[1:])
        return torch.stack(xa), torch.stack(xb)

    def step(self):
        for model in self.modules():
            model.train()
        xa, xb = self.draw()
        batch = self.cfg.batch_size
        ka, kb = self.g.integers(4, size=batch), self.g.integers(4, size=batch)

        def rotate(values, rotations):
            return torch.stack([torch.rot90(values[i], int(rotations[i]), (1, 2))
                                for i in range(len(values))])

        la = rotate(self.moving_encoder(rotate(xa, ka)), -ka)
        lb = rotate(self.reference_encoder(rotate(xb, kb)), -kb)
        points = torch.cat((la, lb)).reshape(2 * batch, -1)
        similarity = -torch.cdist(points, points).square() / points.shape[1] / self.cfg.temperature
        ids = torch.arange(2 * batch, device=self.device)
        positives = similarity[ids, (ids + batch) % (2 * batch)]
        negatives = similarity.masked_fill(torch.eye(2 * batch, device=self.device, dtype=torch.bool),
                                          -float("inf"))
        contrastive = (-positives + torch.logsumexp(negatives, 1)).mean()
        size = la.numel() + lb.numel()
        loss = (contrastive + self.cfg.regularization_l1 * (1 + la.abs().sum() + lb.abs().sum()) / size
                + self.cfg.regularization_l2 * (1 + la.square().sum() + lb.square().sum()) / size)
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norms = [float(torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.clip_norm))
                 for model in self.modules()]
        for optimizer in self.optimizers:
            optimizer.step()
        self.n_steps += 1
        return {"step": self.n_steps, "epoch": (self.n_steps - 1) // self.cfg.steps_per_epoch + 1,
                "loss": float(loss.detach()), "contrastive_loss": float(contrastive.detach()),
                "grad_norm_preclip": max(norms), "training_objective": "CoMIR_native"}
