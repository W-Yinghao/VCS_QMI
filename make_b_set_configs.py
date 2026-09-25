"""B-group 'set' batch for VCS (owner: 再提交一批), derived from cifar10_confirm200_vcs_seed0.yaml.  Stage P16_vcs_hparamB_set.

Targets the saturation / weak-signal hypothesis (P9 report): harder positives (augmentation), more pairs per step (batch size), matrix
weight decay, and the first combinations of the two factors that helped (K=8, critic LR x3/x10), including one 800-epoch combination.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "configs" / "cifar10_confirm200_vcs_seed0.yaml"

def setp(c, path, val):
    d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = val

UNITS = [  # label, list of (path, value)
    ("aug_crop008", [(("views", "random_resized_crop", "scale"), [0.08, 1.0])]),
    ("aug_cj08", [(("views", "color_jitter", "brightness"), 0.8), (("views", "color_jitter", "contrast"), 0.8),
                  (("views", "color_jitter", "saturation"), 0.8), (("views", "color_jitter", "hue"), 0.2)]),
    ("aug_blur05", [(("views", "gaussian_blur_p"), 0.5)]),
    ("aug_strong", [(("views", "random_resized_crop", "scale"), [0.08, 1.0]), (("views", "color_jitter", "brightness"), 0.8),
                    (("views", "color_jitter", "contrast"), 0.8), (("views", "color_jitter", "saturation"), 0.8),
                    (("views", "color_jitter", "hue"), 0.2), (("views", "gaussian_blur_p"), 0.5)]),
    ("aug_weak", [(("views", "random_resized_crop", "scale"), [0.5, 1.0]), (("views", "color_jitter", "brightness"), 0.2),
                  (("views", "color_jitter", "contrast"), 0.2), (("views", "color_jitter", "saturation"), 0.2), (("views", "color_jitter", "hue"), 0.05)]),
    ("b128", [(("train", "batch_size_images"), 128)]),
    ("b512", [(("train", "batch_size_images"), 512)]),
    ("b1024", [(("train", "batch_size_images"), 1024)]),
    ("wd1e-5", [(("optimizer", "matrix_weight_decay_encoder_projector"), 1e-5)]),
    ("wd5e-4", [(("optimizer", "matrix_weight_decay_encoder_projector"), 5e-4)]),
    ("k8_clr3", [(("pairing", "k"), 8), (("optimizer", "critic_lr_multiplier"), 3.0)]),
    ("k8_clr10", [(("pairing", "k"), 8), (("optimizer", "critic_lr_multiplier"), 10.0)]),
    ("k8_clr10_800ep", [(("pairing", "k"), 8), (("optimizer", "critic_lr_multiplier"), 10.0), (("train", "epochs"), 800),
                        (("logging", "checkpoint_epochs"), [100, 200, 400, 600, 800]),
                        (("evaluation", "knn_epochs"), [0, 20, 50, 100, 200, 400, 600, 800]), (("evaluation", "linear_epochs_of_pretrain"), [800])]),
    ("k8_aug_strong", [(("pairing", "k"), 8), (("views", "random_resized_crop", "scale"), [0.08, 1.0]), (("views", "color_jitter", "brightness"), 0.8),
                       (("views", "color_jitter", "contrast"), 0.8), (("views", "color_jitter", "saturation"), 0.8),
                       (("views", "color_jitter", "hue"), 0.2), (("views", "gaussian_blur_p"), 0.5)]),
]
out, units = {}, []
for label, changes in UNITS:
    c = yaml.safe_load(BASE.read_text())
    c["run"]["stage"] = "P16_vcs_hparamB_set"
    for path, val in changes:
        setp(c, path, val)
    assert c["run"]["seed"] == 0 and c["run"]["method"] == "vcs_qmi"
    p = ROOT / "configs" / f"cifar10_hpB_{label}_vcs_seed0.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
    out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    final = "epoch_800.pt" if c["train"]["epochs"] == 800 else "epoch_200.pt"
    units.append({"label": label, "changes": [(".".join(pth), v) for pth, v in changes], "config": f"configs/{p.name}", "run_id": f"P16_vcs_{label}_seed0", "final_ckpt": final})
(ROOT / "configs" / "HPARAM_B_SET_SHA256.json").write_text(json.dumps({"derived_from": {BASE.name: hashlib.sha256(BASE.read_bytes()).hexdigest()},
    "baseline_run": "P5_vcs_seed0 (200ep) / P8_vcs800_seed0 (800ep unit)", "units": units, "files": out}, indent=2))
(ROOT / "slurm" / "hparamB_set_units.txt").write_text("".join(f"{u['run_id']} {u['config']} {u['final_ckpt']}\n" for u in units))
print("\n".join(f"{u['run_id']}: {u['changes']}" for u in units))
