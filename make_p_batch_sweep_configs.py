"""Wave-2 C-S1 (stage P61_batchsweep): from-scratch batch sweep on CIFAR-10 — does the separately-averaged VCS objective degrade less than
SimCLR / VICReg when the batch shrinks?  Owner 2026-09-27 ("全部都补充实验"): these are SSL-type runs, authorised for this unit only.
Design (reports/SECOND_APP_WAVE2_PLAN_20260927.md, row C): 2 views, 200 epochs, seed 0, B in {32, 64, 128, 256}; VCS = final 2-view recipe
(cosine critic, a0 = 5, negative detach, K = min(8, B-1)); SimCLR / VICReg = frozen P5 recipes; lr = frozen lr (1e-3 at B = 256, all three
methods) x B/256 (linear rule, declared for all); equal-tuning check: lr x 2 at B = 32 for every method.  Nothing else changes."""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
STAGE = "P61_batchsweep"
BASES = {"vcs": ("cifar10_hpG_a5_learn_vcs_seed0.yaml", "vcs_qmi"), "simclr": ("cifar10_confirm200_simclr_seed0.yaml", "simclr_matched"),
         "vicreg": ("cifar10_confirm200_vicreg_seed0.yaml", "vicreg_matched_128")}
BATCHES = (32, 64, 128, 256)
# wall-clock estimate (hours, H100 / RTX6000PRO class; A100 about x2): 2-view B256 ~17 s/epoch; smaller batches pay per-step overhead
EST_H = {32: 3.5, 64: 2.0, 128: 1.3, 256: 1.0}
units, files = [], {}
for method, (fname, mname) in BASES.items():
    base = yaml.safe_load((ROOT / "configs" / fname).read_text())
    assert base["run"]["method"] == mname and base["views"]["count"] == 2 and base["train"]["epochs"] == 200
    assert base["train"]["batch_size_images"] == 256 and abs(base["optimizer"]["lr"] - 1e-3) < 1e-12 and base["train"]["warmup_epochs"] == 10
    if method == "vcs":
        assert base["model"]["critic"]["input"] == "cosine" and base["model"]["critic"]["cosine_scale_init"] == 5.0 and base["pairing"]["negative_detach"] and base["pairing"]["k"] == 8
    for B in BATCHES:
        variants = [("", 1.0)] + ([("_lr2x", 2.0)] if B == 32 else [])
        for suffix, mult in variants:
            c = copy.deepcopy(base)
            c["run"]["stage"] = STAGE
            c["train"]["batch_size_images"] = B
            c["optimizer"]["lr"] = float(base["optimizer"]["lr"]) * B / 256 * mult
            if method == "vcs":
                c["pairing"]["k"] = min(8, B - 1)
            label = f"{method}_B{B}{suffix}"
            p = ROOT / "configs" / f"cifar10_hpP_{label}_seed0.yaml"
            p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
            files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
            units.append({"label": label, "method": mname, "batch": B, "lr": c["optimizer"]["lr"], "lr_rule": f"1e-3 x B/256{' x 2 (equal-tuning check)' if mult != 1.0 else ''}",
                          "K": c["pairing"]["k"] if method == "vcs" else None, "epochs": 200, "views": 2, "steps_per_epoch": 45000 // B,
                          "config": f"configs/{p.name}", "run_id": f"P61_{label}_seed0", "final_ckpt": "epoch_200.pt", "est_hours_h100": EST_H[B], "chain_links": 1})
(ROOT / "configs" / "HPARAM_P_SHA256.json").write_text(json.dumps({"stage": STAGE, "bases": {k: v[0] for k, v in BASES.items()}, "units": units, "files": files}, indent=2))
print("\n".join(f"{u['run_id']:28s} B={u['batch']:3d} lr={u['lr']:.3g} K={u['K']} steps/ep={u['steps_per_epoch']} ~{u['est_hours_h100']}h {files[Path(u['config']).name][:12]}" for u in units))
