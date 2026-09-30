"""P97 — 16x compute controls: tuned SimCLR and VICReg (P41 4-view 800-epoch recipes) extended to 1600 epochs exactly as the VCS 16x run
(P43_vcs_a5_views4_1600ep_seed0, config cifar10_hpO_a5_views4_1600ep_vcs_seed0.yaml) was extended from its 800-epoch recipe: epochs 800 -> 1600, kNN
epochs {0, 50, 100, 200, 400, 800, 1200, 1600}, checkpoints {200, 400, 800, 1200, 1600}; everything else unchanged.  Seed 0 (one seed, matching the VCS 16x cell)."""
import hashlib, json, sys
from pathlib import Path
import yaml
R = Path(__file__).resolve().parents[1]
ref800 = yaml.safe_load(open(R / "configs/cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml"))
ref1600 = yaml.safe_load(open(R / "configs/cifar10_hpO_a5_views4_1600ep_vcs_seed0.yaml"))
out = {}
for m in ("simclr", "vicreg"):
    c = yaml.safe_load(open(R / f"configs/cifar10_hpN_{m}_views4_800ep_seed0.yaml"))
    assert c["train"]["epochs"] == 800
    c["run"]["stage"] = "P97_controls_16x"
    c["train"]["epochs"] = 1600
    c["evaluation"]["knn_epochs"] = ref1600["evaluation"]["knn_epochs"]
    c["evaluation"]["linear_epochs_of_pretrain"] = ref1600["evaluation"]["linear_epochs_of_pretrain"]
    c["logging"]["checkpoint_epochs"] = ref1600["logging"]["checkpoint_epochs"]
    p = R / f"configs/cifar10_hpW_{m}_views4_1600ep_seed0.yaml"
    txt = yaml.safe_dump(c, sort_keys=False)
    if "--write" in sys.argv:
        p.write_text(txt)
    out[p.name] = hashlib.sha256(txt.encode()).hexdigest()
    print(p.name, out[p.name][:16], "knn", c["evaluation"]["knn_epochs"], "ckpt", c["logging"]["checkpoint_epochs"])
if "--write" in sys.argv:
    (R / "configs/P97_SHA256.json").write_text(json.dumps(out, indent=1))
