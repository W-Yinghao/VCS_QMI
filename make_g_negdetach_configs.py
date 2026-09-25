"""Wave G: single factors on the new best base (cosine critic, K=8, negative_detach = P24_vcs_cos_negdetach_seed0, 80.48), plus seeds and 800 ep.
Stage P26_vcs_negdetach_base. Candidates follow the external review (N1-N6) and this session's results."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "configs" / "cifar10_hpF_cos_negdetach_vcs_seed0.yaml"

def setp(c, path, val):
    d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = val

LONG = [(("train", "epochs"), 800), (("logging", "checkpoint_epochs"), [100, 200, 400, 600, 800]),
        (("evaluation", "knn_epochs"), [0, 20, 50, 100, 200, 400, 600, 800]), (("evaluation", "linear_epochs_of_pretrain"), [800])]
UNITS = [
    ("base_seed1", 1, []),
    ("base_seed2", 2, []),
    ("b0_calib", 0, [(("model", "critic", "cosine_bias_calibrate"), True)]),                                   # N1
    ("a5_learn", 0, [(("model", "critic", "cosine_scale_init"), 5.0)]),                                        # N2
    ("a5_fixed", 0, [(("model", "critic", "cosine_scale_init"), 5.0), (("model", "critic", "cosine_scale_fixed"), True)]),  # N3
    ("proj_outBN", 0, [(("model", "projector", "output_batchnorm"), True), (("model", "projector", "output_linear_bias"), False)]),  # N4
    ("interact_only", 0, [(("model", "critic", "input"), "interact_only"), (("model", "critic", "hidden_dims"), [256, 256])]),  # N5
    ("shared_metric", 0, [(("model", "critic", "input"), "shared_metric")]),                                   # N6
    ("k255", 0, [(("pairing", "k"), 255)]),
    ("base_800ep", 0, LONG),
]
out, units = {}, []
for label, seed, changes in UNITS:
    c = yaml.safe_load(BASE.read_text())
    c["run"]["stage"] = "P26_vcs_negdetach_base"; c["run"]["seed"] = seed
    for path, val in changes:
        setp(c, path, val)
    assert c["pairing"]["negative_detach"] is True and c["model"]["critic"]["input"] in ("cosine", "interact_only", "shared_metric")
    p = ROOT / "configs" / f"cifar10_hpG_{label}_vcs_seed{seed}.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
    out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    final = "epoch_800.pt" if c["train"]["epochs"] == 800 else "epoch_200.pt"
    units.append({"label": label, "seed": seed, "changes": [(".".join(pth), v) for pth, v in changes], "config": f"configs/{p.name}", "run_id": f"P26_vcs_{label}_seed{seed}", "final_ckpt": final})
(ROOT / "configs" / "HPARAM_G_NEGDETACH_SHA256.json").write_text(json.dumps({"derived_from": {BASE.name: hashlib.sha256(BASE.read_bytes()).hexdigest()},
    "baseline_run": "P24_vcs_cos_negdetach_seed0", "units": units, "files": out}, indent=2))
(ROOT / "slurm" / "hparamG_negdetach_units.txt").write_text("".join(f"{u['run_id']} {u['config']} {u['final_ckpt']}\n" for u in units))
print("\n".join(f"{u['run_id']}: {u['changes'] if len(u['changes'])<4 else '800-epoch'}" for u in units))
