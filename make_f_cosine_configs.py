"""Wave F: single factors re-based on the cosine-critic + K=8 config (P18_vcs_crit_cosine_seed0 = 78.32). Stage P24_vcs_cosine_base."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "configs" / "cifar10_hpC_crit_cosine_vcs_seed0.yaml"

def setp(c, path, val):
    d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = val

LONG = [(("train", "epochs"), 800), (("logging", "checkpoint_epochs"), [100, 200, 400, 600, 800]),
        (("evaluation", "knn_epochs"), [0, 20, 50, 100, 200, 400, 600, 800]), (("evaluation", "linear_epochs_of_pretrain"), [800])]
UNITS = [
    ("cos_k64", [(("pairing", "k"), 64)]),
    ("cos_k255", [(("pairing", "k"), 255)]),
    ("cos_clr10", [(("optimizer", "critic_lr_multiplier"), 10.0)]),
    ("cos_scale10", [(("model", "critic", "cosine_scale_init"), 10.0)]),
    ("cos_on_h", [(("model", "critic", "feature_source"), "h_l2")]),
    ("cos_negdetach", [(("pairing", "negative_detach"), True)]),
    ("cos_ema0.99", [(("train", "target_branch"), "ema_0.99")]),
    ("cos_ema0.996", [(("train", "target_branch"), "ema_0.996")]),
    ("cos_stopgrad", [(("train", "target_branch"), "stopgrad")]),
    ("cos_sg_pred", [(("train", "target_branch"), "stopgrad"), (("model", "projector", "predictor"), True)]),
    ("cos_ema0.99_pred", [(("train", "target_branch"), "ema_0.99"), (("model", "projector", "predictor"), True)]),
    ("cos_proj_depth3", [(("model", "projector", "depth"), 3)]),
    ("cos_800ep", LONG),
]
out, units = {}, []
for label, changes in UNITS:
    c = yaml.safe_load(BASE.read_text())
    c["run"]["stage"] = "P24_vcs_cosine_base"
    for path, val in changes:
        setp(c, path, val)
    assert c["model"]["critic"]["input"] == "cosine" and c["run"]["seed"] == 0
    p = ROOT / "configs" / f"cifar10_hpF_{label}_vcs_seed0.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
    out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    final = "epoch_800.pt" if c["train"]["epochs"] == 800 else "epoch_200.pt"
    units.append({"label": label, "changes": [(".".join(pth), v) for pth, v in changes], "config": f"configs/{p.name}", "run_id": f"P24_vcs_{label}_seed0", "final_ckpt": final})
(ROOT / "configs" / "HPARAM_F_COSINE_SHA256.json").write_text(json.dumps({"derived_from": {BASE.name: hashlib.sha256(BASE.read_bytes()).hexdigest()},
    "baseline_run": "P18_vcs_crit_cosine_seed0", "units": units, "files": out}, indent=2))
(ROOT / "slurm" / "hparamF_cosine_units.txt").write_text("".join(f"{u['run_id']} {u['config']} {u['final_ckpt']}\n" for u in units))
print("\n".join(f"{u['run_id']}: {u['changes'] if len(u['changes'])<4 else '800-epoch'}" for u in units))
