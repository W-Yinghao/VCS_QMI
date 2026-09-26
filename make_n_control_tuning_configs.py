"""Wave N (stage P41_control_tuning) — NOT RUN until the owner confirms (see reports/P41_CONTROL_TUNING_PREREG_DRAFT_20260926.md).
Equal-budget tuning of the two controls with the knobs that mattered for VCS (views, batch/steps, schedule) plus each control's own knob."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
STAGE = "P41_control_tuning"
def setp(c, path, val):
    d = c
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = val
LONG = {("train", "epochs"): 800, ("logging", "checkpoint_epochs"): [100, 200, 400, 600, 800], ("evaluation", "knn_epochs"): [0, 20, 50, 100, 200, 400, 600, 800], ("evaluation", "linear_epochs_of_pretrain"): [800]}
SHORT = {("train", "epochs"): 100, ("logging", "checkpoint_epochs"): [20, 50, 100], ("evaluation", "knn_epochs"): [0, 10, 20, 50, 100], ("evaluation", "linear_epochs_of_pretrain"): [100]}
UNITS = {
    "simclr": {"src": "cifar10_confirm200_simclr_seed0.yaml", "units": [
        ("tau0.1", {("objective", "simclr_temperature"): 0.1}), ("tau0.5", {("objective", "simclr_temperature"): 0.5}),
        ("views4", {("views", "count"): 4}), ("views4_b128_100ep", {("views", "count"): 4, ("train", "batch_size_images"): 128, **SHORT}),
        ("b128", {("train", "batch_size_images"): 128}), ("800ep", LONG)]},
    "vicreg": {"src": "cifar10_confirm200_vicreg_seed0.yaml", "units": [
        ("cov0.1", {("objective", "vicreg_weights", "covariance"): 0.1}), ("w10_10_1", {("objective", "vicreg_weights", "invariance"): 10, ("objective", "vicreg_weights", "variance"): 10}),
        ("views4", {("views", "count"): 4}), ("views4_b128_100ep", {("views", "count"): 4, ("train", "batch_size_images"): 128, **SHORT}),
        ("b128", {("train", "batch_size_images"): 128}), ("800ep", LONG)]},
}
if __name__ == "__main__":
    if "--confirm" not in sys.argv:
        print("dry run: pass --confirm after the owner's go to write configs"); sys.exit(0)
    files, units = {}, []
    for ctrl, spec in UNITS.items():
        src = ROOT / "configs" / spec["src"]
        for label, changes in spec["units"]:
            c = yaml.safe_load(src.read_text()); c["run"]["stage"] = STAGE; c["run"]["control_tuning"] = True
            for path, val in changes.items():
                setp(c, path, val)
            p = ROOT / "configs" / f"cifar10_hpN_{ctrl}_{label}_seed0.yaml"
            p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8"); files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
            final = f"epoch_{c['train']['epochs']:03d}.pt"
            units.append({"control": ctrl, "label": label, "changes": {".".join(k): v for k, v in changes.items()}, "config": f"configs/{p.name}", "run_id": f"P41_{ctrl}_{label}_seed0", "final_ckpt": final})
    (ROOT / "configs" / "HPARAM_N_SHA256.json").write_text(json.dumps({"stage": STAGE, "units": units, "files": files}, indent=2))
    (ROOT / "slurm" / "hparamN_units.txt").write_text("".join(f"{u['run_id']} {u['config']} {u['final_ckpt']}\n" for u in units))
    print("\n".join(f"{u['run_id']} {u['changes']}" for u in units))
