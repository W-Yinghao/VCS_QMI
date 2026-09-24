"""Derive the frozen 800-epoch VCS-only configs from the frozen 200-epoch confirmation configs (explicit overrides only).

Stage P8_long800 (owner request 2026-09-24): same recipe, VCS only, seeds {0,1,2}, 800 epochs, warm-up 10 epochs unchanged,
checkpoints at 100/200/400/600/800, kNN/spectrum/critic monitor at 0/20/50/100/200/400/600/800.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
OVERRIDES = {
    ("run", "stage"): "P8_long800",
    ("train", "epochs"): 800,
    ("logging", "checkpoint_epochs"): [100, 200, 400, 600, 800],
    ("evaluation", "knn_epochs"): [0, 20, 50, 100, 200, 400, 600, 800],
    ("evaluation", "linear_epochs_of_pretrain"): [800],
}
out, src = {}, {}
for seed in (0, 1, 2):
    base_p = ROOT / "configs" / f"cifar10_confirm200_vcs_seed{seed}.yaml"
    src[base_p.name] = hashlib.sha256(base_p.read_bytes()).hexdigest()
    c = yaml.safe_load(base_p.read_text())
    for path, val in OVERRIDES.items():
        d = c
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = val
    assert c["run"]["seed"] == seed and c["train"]["warmup_epochs"] == 10 and c["run"]["method"] == "vcs_qmi"
    p = ROOT / "configs" / f"cifar10_long800_vcs_seed{seed}.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
    out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT / "configs" / "LONG800_SHA256.json").write_text(json.dumps({"derived_from": src, "overrides": {".".join(k): v for k, v in OVERRIDES.items()}, "files": out}, indent=2))
print(json.dumps(out, indent=2))
