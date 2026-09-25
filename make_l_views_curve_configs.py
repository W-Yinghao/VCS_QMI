"""Wave L (stage P37_vcs_views_curve): views-at-fixed-compute curve on the a0 = 5 base (cosine critic, K = 8, negative detach, a0 = 5).
Encoder compute is held at the 200-epoch 2-view budget (8.96 M image-forwards) or at twice it:
    a5_views4_100ep  (4 views ×100 ep = 1×), a5_views8_50ep (8 views × 50 ep = 1×), a5_views8_100ep (8 views × 100 ep = 2×).
Source: configs/cifar10_hpK_a5_views4_vcs_seed0.yaml (a5 + 4 views, 200 epochs)."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "configs" / "cifar10_hpK_a5_views4_vcs_seed0.yaml"
STAGE = "P37_vcs_views_curve"
UNITS = [  # label, views, epochs, checkpoints, knn epochs
    ("a5_views4_100ep", 4, 100, [20, 50, 100], [0, 10, 20, 50, 100]),
    ("a5_views8_50ep", 8, 50, [10, 20, 50], [0, 10, 20, 50]),
    ("a5_views8_100ep", 8, 100, [20, 50, 100], [0, 10, 20, 50, 100]),
]
files, units = {}, []
for label, nv, ep, cks, kes in UNITS:
    c = yaml.safe_load(SRC.read_text()); assert c["model"]["critic"]["cosine_scale_init"] == 5.0 and c["pairing"]["negative_detach"] and c["views"]["count"] == 4
    c["run"]["stage"] = STAGE; c["views"]["count"] = nv; c["train"]["epochs"] = ep
    c["logging"]["checkpoint_epochs"] = cks; c["evaluation"]["knn_epochs"] = kes; c["evaluation"]["linear_epochs_of_pretrain"] = [ep]
    p = ROOT / "configs" / f"cifar10_hpL_{label}_vcs_seed0.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8"); files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    units.append({"label": label, "views": nv, "epochs": ep, "config": f"configs/{p.name}", "run_id": f"P37_vcs_{label}_seed0", "final_ckpt": f"epoch_{ep:03d}.pt"})
(ROOT / "configs" / "HPARAM_L_SHA256.json").write_text(json.dumps({"source": SRC.name, "stage": STAGE, "units": units, "files": files}, indent=2))
(ROOT / "slurm" / "hparamL_units.txt").write_text("".join(f"{u['run_id']} {u['config']} {u['final_ckpt']}\n" for u in units))
print("\n".join(f"{u['run_id']} views={u['views']} ep={u['epochs']} {files[Path(u['config']).name][:12]}" for u in units))
