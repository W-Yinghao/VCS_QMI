"""Wave I (stage P31_vcs_scale_init): seeds 1/2 of the a5_learn run (81.90, new best) and a sweep of the cosine critic's initial scale a0
(learnable) on the neg-detach cosine base.  Source: configs/cifar10_hpG_a5_learn_vcs_seed0.yaml (a0 = 5)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "configs" / "cifar10_hpG_a5_learn_vcs_seed0.yaml"
STAGE = "P31_vcs_scale_init"
UNITS = [  # (label, seed, a0)
    ("a5_learn", 1, 5.0), ("a5_learn", 2, 5.0),
    ("a2_learn", 0, 2.0), ("a10_learn", 0, 10.0), ("a20_learn", 0, 20.0),
]
files, units = {}, []
for label, seed, a0 in UNITS:
    c = yaml.safe_load(SRC.read_text()); c["run"]["stage"] = STAGE; c["run"]["seed"] = seed
    c["model"]["critic"]["cosine_scale_init"] = a0
    p = ROOT / "configs" / f"cifar10_hpI_{label}_vcs_seed{seed}.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8"); files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    units.append({"label": label, "seed": seed, "cosine_scale_init": a0, "config": f"configs/{p.name}", "run_id": f"P31_vcs_{label}_seed{seed}"})
(ROOT / "configs" / "HPARAM_I_SHA256.json").write_text(json.dumps({"source": SRC.name, "stage": STAGE, "units": units, "files": files}, indent=2))
(ROOT / "slurm" / "hparamI_units.txt").write_text("".join(f"{u['run_id']} {u['config']} epoch_200.pt\n" for u in units))
print("\n".join(f"{u['run_id']} a0={u['cosine_scale_init']} {files[Path(u['config']).name][:12]}" for u in units))
