"""Wave J (stage P33_vcs_views_compute): equal-compute control for the 4-view unit — 4 views for 100 epochs (2 encoder forwards per image per
step, so 100 epochs = the encoder compute of the 200-epoch 2-view base).  Source: configs/cifar10_hpH_views4_vcs_seed0.yaml."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "configs" / "cifar10_hpH_views4_vcs_seed0.yaml"
STAGE = "P33_vcs_views_compute"
c = yaml.safe_load(SRC.read_text()); assert c["views"]["count"] == 4 and c["train"]["epochs"] == 200
c["run"]["stage"] = STAGE
c["train"]["epochs"] = 100
c["logging"]["checkpoint_epochs"] = [20, 50, 100]
c["evaluation"]["knn_epochs"] = [0, 10, 20, 50, 100]
c["evaluation"]["linear_epochs_of_pretrain"] = [100]
p = ROOT / "configs" / "cifar10_hpJ_views4_100ep_vcs_seed0.yaml"
p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
sha = hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT / "configs" / "HPARAM_J_SHA256.json").write_text(json.dumps({"source": SRC.name, "stage": STAGE, "units": [{"label": "views4_100ep", "config": f"configs/{p.name}", "run_id": "P33_vcs_views4_100ep_seed0", "final_ckpt": "epoch_100.pt"}], "files": {p.name: sha}}, indent=2))
(ROOT / "slurm" / "hparamJ_units.txt").write_text(f"P33_vcs_views4_100ep_seed0 configs/{p.name} epoch_100.pt\n")
print(p.name, sha[:12])
