"""P115 diagnostics, job 1 + job 2: the 2 x 2 augmentation factor cells (standard / crop-only / jitter-only / both strong) for A-P3 and SimCLR at
checkpoints 100 / 400 / 800 (epoch 20 where it exists).  Read-only; reads reports/P115_diag_results.json.  Descriptive (one seed per cell)."""
import json
D = json.load(open("reports/P115_diag_results.json"))["runs"]
CELLS = {"A-P3": [("standard", "P107_AP3_views4_800ep_seed0"), ("crop-only", "P115_AP3_croponly_views4_800ep_seed0"),
                  ("jitter-only", "P115_AP3_jitteronly_views4_800ep_seed0"), ("both", "P107_AP3_augstrong_views4_800ep_seed0")],
         "SimCLR": [("standard", "P41_simclr_views4_800ep_seed0"), ("crop-only", "P115_simclr_croponly_views4_800ep_seed0"),
                    ("jitter-only", "P115_simclr_jitteronly_views4_800ep_seed0"), ("both", "P89_simclr_views4_800ep_augstrong_seed0")]}
COLS = [("s_pos_mean", "{:.3f}"), ("s_pos_q05", "{:.3f}"), ("frac_pos_below_kappa", "{:.3f}"), ("lowiou_frac_pos", "{:.3f}"), ("lowiou_share_of_P_grad", "{:.3f}"),
        ("iou_lt0.1_share_of_P_grad", "{:.3f}"), ("viewpair_cancellation", "{:.3f}"), ("grad_encoder_P_over_Q", "{:.1f}"), ("h_eff_rank", "{:.1f}"), ("z_eff_rank", "{:.1f}")]
out = ["| method | cell | epoch | linear / kNN @800 | " + " | ".join(c for c, _ in COLS) + " |", "|" + "---|" * (4 + len(COLS))]
for m, cells in CELLS.items():
    for cell, run in cells:
        r = D[run]
        for ep in ("20", "100", "400", "800"):
            c = r["checkpoints"].get(ep)
            if not c or c.get("checkpoint") is None: continue
            if ep == "20" and str(c.get("checkpoint")) == "epoch_100.pt": continue   # SimCLR / recipe runs: 20 -> 100 duplicate
            s = c["scalars"]
            vals = [(f.format(s[k]) if isinstance(s.get(k), (int, float)) else "—") for k, f in COLS]
            acc = f"{r['linear_800']:.2f} / {r['knn_800']:.2f}" if ep == "800" else ""
            out.append(f"| {m} | {cell} | {ep} | {acc} | " + " | ".join(vals) + " |")
open("reports/P115_diag_factor_table.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
