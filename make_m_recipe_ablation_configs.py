"""Wave M (stage P39_vcs_recipe_ablation): factor ablations *inside* the current recipe (cosine critic, K = 8, negative detach, a0 = 5, 4 views)
at the 1x compute budget, plus an optimizer re-check of the a0 = 5 base.
  4-view units, 100 epochs (= compute of the 2-view 200-epoch base; compared with P37 a5_views4_100ep):
    a5_views4_k1_100ep        K = 1            -> do negatives still matter once detach + 4 views carry the signal?
    a5_views4_nodetach_100ep  detach off       -> does the detach gain persist under 4 views?
    a5_views4_b128_100ep      batch 128        -> same compute, twice the optimizer steps (= steps of the 2-view 200-ep base)
  2-view units, 200 epochs (compared with the a0 = 5 base 81.56 +- 0.44):
    a5_lr2e-3, a5_lr5e-4      lr x2 / x0.5     -> optimizer re-check on the new objective operating point
Source: configs/cifar10_hpK_a5_views4_vcs_seed0.yaml (4-view) and configs/cifar10_hpG_a5_learn_vcs_seed0.yaml (2-view)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
STAGE = "P39_vcs_recipe_ablation"
V4 = ROOT / "configs" / "cifar10_hpK_a5_views4_vcs_seed0.yaml"
V2 = ROOT / "configs" / "cifar10_hpG_a5_learn_vcs_seed0.yaml"
def setp(c, path, val):
    d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = val
SHORT = {("train", "epochs"): 100, ("logging", "checkpoint_epochs"): [20, 50, 100], ("evaluation", "knn_epochs"): [0, 10, 20, 50, 100], ("evaluation", "linear_epochs_of_pretrain"): [100]}
UNITS = [  # label, source, changes, final
    ("a5_views4_k1_100ep", V4, {**SHORT, ("pairing", "k"): 1}, "epoch_100.pt"),
    ("a5_views4_nodetach_100ep", V4, {**SHORT, ("pairing", "negative_detach"): False}, "epoch_100.pt"),
    ("a5_views4_b128_100ep", V4, {**SHORT, ("train", "batch_size_images"): 128}, "epoch_100.pt"),
    ("a5_lr2e-3", V2, {("optimizer", "lr"): 0.002}, "epoch_200.pt"),
    ("a5_lr5e-4", V2, {("optimizer", "lr"): 0.0005}, "epoch_200.pt"),
]
files, units = {}, []
for label, src, changes, final in UNITS:
    c = yaml.safe_load(src.read_text()); assert c["model"]["critic"]["cosine_scale_init"] == 5.0 and c["pairing"]["negative_detach"]
    c["run"]["stage"] = STAGE
    for path, val in changes.items():
        setp(c, path, val)
    p = ROOT / "configs" / f"cifar10_hpM_{label}_vcs_seed0.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8"); files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    units.append({"label": label, "source": src.name, "changes": {".".join(k): v for k, v in changes.items()}, "config": f"configs/{p.name}", "run_id": f"P39_vcs_{label}_seed0", "final_ckpt": final})
(ROOT / "configs" / "HPARAM_M_SHA256.json").write_text(json.dumps({"stage": STAGE, "units": units, "files": files}, indent=2))
(ROOT / "slurm" / "hparamM_units.txt").write_text("".join(f"{u['run_id']} {u['config']} {u['final_ckpt']}\n" for u in units))
print("\n".join(f"{u['run_id']} {files[Path(u['config']).name][:12]} {u['changes']}" for u in units))
