"""Wave K (stage P35_vcs_a5_base): first units on the confirmed new base — cosine critic, K = 8, negative detach, a0 = 5 (learnable):
81.90 / 81.06 / 81.72 on seeds 0/1/2 (mean 81.56) vs 80.59 for a0 = 1.  Source: configs/cifar10_hpG_a5_learn_vcs_seed0.yaml."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "configs" / "cifar10_hpG_a5_learn_vcs_seed0.yaml"
STAGE = "P35_vcs_a5_base"
def setp(c, path, val):
    d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = val
UNITS = [
    ("a5_views4", [(("views", "count"), 4)], "epoch_200.pt"),
    ("a5_800ep", [(("train", "epochs"), 800), (("logging", "checkpoint_epochs"), [100, 200, 400, 600, 800]),
                  (("evaluation", "knn_epochs"), [0, 20, 50, 100, 200, 400, 600, 800]), (("evaluation", "linear_epochs_of_pretrain"), [800])], "epoch_800.pt"),
]
files, units = {}, []
for label, changes, final in UNITS:
    c = yaml.safe_load(SRC.read_text()); assert c["model"]["critic"]["cosine_scale_init"] == 5.0 and c["pairing"]["negative_detach"]
    c["run"]["stage"] = STAGE
    for path, val in changes:
        setp(c, path, val)
    p = ROOT / "configs" / f"cifar10_hpK_{label}_vcs_seed0.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8"); files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    units.append({"label": label, "changes": [(".".join(pth), v) for pth, v in changes], "config": f"configs/{p.name}", "run_id": f"P35_vcs_{label}_seed0", "final_ckpt": final})
(ROOT / "configs" / "HPARAM_K_SHA256.json").write_text(json.dumps({"source": SRC.name, "stage": STAGE, "units": units, "files": files}, indent=2))
(ROOT / "slurm" / "hparamK_units.txt").write_text("".join(f"{u['run_id']} {u['config']} {u['final_ckpt']}\n" for u in units))
print("\n".join(f"{u['run_id']} {files[Path(u['config']).name][:12]} {u['changes']}" for u in units))
