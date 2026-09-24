"""Derive the frozen K-study configs (VCS only) from the frozen 200-epoch VCS confirmation configs: the ONLY change is pairing.k.

Stage P10_kstudy200 (owner request 2026-09-24): K in {8, 64}, seeds {0,1,2}, 200 epochs, warm-up 10; comparator = P5 VCS runs (K=1).
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
out, src = {}, {}
for k in (8, 64):
    for seed in (0, 1, 2):
        base_p = ROOT / "configs" / f"cifar10_confirm200_vcs_seed{seed}.yaml"
        src[base_p.name] = hashlib.sha256(base_p.read_bytes()).hexdigest()
        c = yaml.safe_load(base_p.read_text())
        c["run"]["stage"] = "P10_kstudy200"
        c["pairing"]["k"] = k
        assert c["run"]["seed"] == seed and c["train"]["epochs"] == 200 and c["train"]["warmup_epochs"] == 10 and c["run"]["method"] == "vcs_qmi"
        p = ROOT / "configs" / f"cifar10_k{k}_vcs_seed{seed}.yaml"
        p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
        out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT / "configs" / "KSTUDY_SHA256.json").write_text(json.dumps({"derived_from": src, "overrides": {"run.stage": "P10_kstudy200", "pairing.k": [8, 64]}, "files": out}, indent=2))
print(json.dumps(out, indent=2))
