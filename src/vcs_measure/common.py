"""Read-only loading of a finished SSL run and frozen multi-layer features (P109)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from vcs_ssl.checkpoint import load_checkpoint
from vcs_ssl.config import load_resolved
from vcs_ssl.data.splits import load_manifest
from vcs_ssl.models import build_models

OUTPUT_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs")
LAYERS = ("layer3", "h", "z")          # + "logits" (clean-trained linear classifier on h, added by the audit)


def run_status(run: str, root: Path = OUTPUT_ROOT) -> tuple[str, int | None]:
    try:
        s = json.load(open(root / run / "status.json"))
        return s.get("status"), s.get("completed_epoch")
    except FileNotFoundError:
        return "MISSING", None


def load_run(run: str, checkpoint: str = "epoch_800.pt", device: torch.device | str = "cpu", root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Encoder, projector, (critic) of a COMPLETED run in eval mode, gradients off; the run directory is only read."""
    rd = root / run
    status, ep = run_status(run, root)
    if status != "COMPLETED":
        raise RuntimeError(f"{run}: status {status} (epoch {ep}); only COMPLETED runs are measured")
    cfg = load_resolved(rd / "config.resolved.yaml"); man = load_manifest(rd / "manifest.json")
    ck = load_checkpoint(rd / "checkpoints" / checkpoint)
    with torch.random.fork_rng(devices=[]):
        built = build_models(cfg, seed=int(cfg["run"]["seed"]), device=device)
    enc, proj, crit = built["encoder"], built["projector"], built["critic"]
    enc.load_state_dict(ck["encoder_state"]); proj.load_state_dict(ck["projector_state"])
    if crit is not None and ck.get("critic_state") is not None:
        crit.load_state_dict(ck["critic_state"])
    for m in (enc, proj, crit):
        if m is not None:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
    return {"run": run, "cfg": cfg, "manifest": man, "encoder": enc, "projector": proj, "critic": crit, "epoch": ck.get("epoch"),
            "eps": float(cfg["model"]["normalization"]["eps"])}


@torch.no_grad()
def layer_features(enc, proj, x: torch.Tensor, eps: float = 1e-8) -> dict[str, torch.Tensor]:
    """layer3 = global-average-pooled torchvision layer3 output [B, 256]; h = encoder output [B, 512]; z = L2(projector(h)) [B, 128]."""
    a = enc.relu(enc.bn1(enc.conv1(x))); a = enc.maxpool(a)
    a = enc.layer1(a); a = enc.layer2(a); l3 = enc.layer3(a); l4 = enc.layer4(l3)
    h = torch.flatten(enc.avgpool(l4), 1)
    z = F.normalize(proj(h), dim=1, eps=eps)
    return {"layer3": l3.mean(dim=(2, 3)), "h": h, "z": z}


def split_base_ids(uids: np.ndarray, sizes: dict[str, int], seed: int) -> dict[str, np.ndarray]:
    """Disjoint base-image splits (sampling unit = base image ID), fixed permutation."""
    rng = np.random.default_rng(seed); perm = rng.permutation(np.asarray(uids))
    out, i = {}, 0
    for k, n in sizes.items():
        out[k] = np.sort(perm[i:i + n]); i += n
    assert i <= len(perm), "split sizes exceed the pool"
    return out


def run_eval_summary(run: str, root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Final frozen linear / kNN and the run's own held-out training-critic J at the last epoch (selection split; recorded by training)."""
    st, ep = run_status(run, root)
    try:
        d = json.load(open(root / run / "evaluations" / f"evaluation_epoch_{int(ep):03d}.json"))
    except Exception:
        return {"status": st}
    return {"status": st, "epoch": ep, "linear_val_top1_pct": d.get("linear_val_top1_pct"), "knn_val_top1_pct": d.get("knn_val_top1_pct"),
            "train_critic_heldout_J": d.get("heldout_J")}
