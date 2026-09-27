"""Wave-2 C-T (stage P63_queue): streaming / asymmetric negatives — a FIFO queue of 4096 detached features from previous steps is the
negative pool (VCS: K = 8 queue partners per anchor in the product term; SimCLR: MoCo-style logits over the queue, no momentum encoder;
identical treatment).  B in {16, 32}, 2 views, 200 epochs, seed 0, lr = 1e-3 x B/256 (same linear rule as P61).  From-scratch SSL-type runs,
authorised by the owner's instruction of 2026-09-27 for this unit only (reports/SECOND_APP_WAVE2_PLAN_20260927.md, row C)."""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
STAGE = "P63_queue"
BASES = {"vcs": ("cifar10_hpG_a5_learn_vcs_seed0.yaml", "vcs_qmi"), "simclr": ("cifar10_confirm200_simclr_seed0.yaml", "simclr_matched")}
BATCHES = (16, 32)
QUEUE = 4096
EST_H = {16: 6.5, 32: 3.5}
units, files = [], {}
for method, (fname, mname) in BASES.items():
    base = yaml.safe_load((ROOT / "configs" / fname).read_text())
    assert base["run"]["method"] == mname and base["views"]["count"] == 2 and base["train"]["batch_size_images"] == 256 and abs(base["optimizer"]["lr"] - 1e-3) < 1e-12
    for B in BATCHES:
        c = copy.deepcopy(base)
        c["run"]["stage"] = STAGE
        c["train"]["batch_size_images"] = B
        c["optimizer"]["lr"] = float(base["optimizer"]["lr"]) * B / 256
        c["pairing"].update({"negative_source": "queue", "queue": True, "queue_size": QUEUE})
        if method == "vcs":
            c["pairing"]["k"] = 8
        label = f"{method}_B{B}_queue{QUEUE}"
        p = ROOT / "configs" / f"cifar10_hpQ_{label}_seed0.yaml"
        p.write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True), encoding="utf-8")
        files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        units.append({"label": label, "method": mname, "batch": B, "lr": c["optimizer"]["lr"], "lr_rule": "1e-3 x B/256", "K": c["pairing"]["k"] if method == "vcs" else None,
                      "queue_size": QUEUE, "epochs": 200, "views": 2, "steps_per_epoch": 45000 // B, "config": f"configs/{p.name}", "run_id": f"P63_{label}_seed0",
                      "final_ckpt": "epoch_200.pt", "est_hours_h100": EST_H[B], "chain_links": 1,
                      "reference_B256": f"P61_{method}_B256_seed0 (cyclic / in-batch negatives, HPARAM_P)"})
(ROOT / "configs" / "HPARAM_Q_SHA256.json").write_text(json.dumps({"stage": STAGE, "bases": {k: v[0] for k, v in BASES.items()}, "units": units, "files": files}, indent=2))
print("\n".join(f"{u['run_id']:34s} B={u['batch']:3d} lr={u['lr']:.3g} K={u['K']} queue={u['queue_size']} steps/ep={u['steps_per_epoch']} ~{u['est_hours_h100']}h {files[Path(u['config']).name][:12]}" for u in units))
