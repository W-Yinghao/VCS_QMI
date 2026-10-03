"""P115 diagnostics aggregate (job 1 = eight existing std / strong runs; job 2 = the P115 crop-only / jitter-only runs when present).

Reads outputs/P115_diag_results/<run>.json (written by scripts/p115_diag.py) and the runs' epoch-800 evaluations; writes reports/P115_diag_results.{md,json}:
per run x checkpoint scalars (distributions, P / Q gradient split, crop-IoU groups, multi-view cancellation, h geometry) and std-vs-strong contrasts
(strong − std) within each method at each common checkpoint.  Descriptive only (frozen prereg reports/P115_DIAG_PREREG_FROZEN_20261003.md).

    python scripts/p115_diag_aggregate.py [--in-dir outputs/P115_diag_results] [--out reports/P115_diag_results]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

OUT_ROOT = Path("/home/infres/yinwang/CS_QMI/outputs")
# method -> (standard run, strong run); job-2 cells are added when their files exist
CELLS = {
    "A-P3": ("P107_AP3_views4_800ep_seed0", "P107_AP3_augstrong_views4_800ep_seed0"),
    "G2": ("P104_G2_views4_800ep_seed0", "P111_G2_augstrong_views4_800ep_seed0"),
    "SimCLR": ("P41_simclr_views4_800ep_seed0", "P89_simclr_views4_800ep_augstrong_seed0"),
    "recipe VCS": ("P35_vcs_a5_views4_800ep_seed0", "P43_vcs_a5_views4_800ep_augstrong_seed0"),
}
JOB2 = {"A-P3 crop-only": "P115_AP3_croponly_views4_800ep_seed0", "A-P3 jitter-only": "P115_AP3_jitteronly_views4_800ep_seed0",
        "SimCLR crop-only": "P115_simclr_croponly_views4_800ep_seed0", "SimCLR jitter-only": "P115_simclr_jitteronly_views4_800ep_seed0"}
# 3-seed mean change of linear-val under strong augmentation (P107 layer-2, P111, P89 / P90 reports) — context only
ACC_CHANGE_3SEED = {"A-P3": -0.78, "G2": -1.25, "SimCLR": +1.17, "recipe VCS": +0.65}


def acc(run: str):
    p = OUT_ROOT / run / "evaluations" / "evaluation_epoch_800.json"
    if not p.is_file():
        return None, None
    d = json.load(open(p))
    return d.get("linear_val_top1_pct"), d.get("knn_val_top1_pct")


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def ratio(a, b):
    return a / b if (a is not None and b not in (None, 0)) else None


def scalars(m: dict, geo: dict) -> dict:
    """Flat scalar record for one checkpoint (mean over the diagnostic batches)."""
    dist, grad, tp, vp = m.get("dist", {}), m.get("grad", {}), m.get("token_pull", {}), m.get("view_pairs", {})
    r = {
        "s_pos_mean": g(dist, "s_pos", "mean"), "s_pos_sd": g(dist, "s_pos", "sd"), "s_pos_q05": (g(dist, "s_pos", "q05_25_50_75_95") or [None])[0],
        "s_neg_mean": g(dist, "s_neg", "mean"), "s_neg_sd": g(dist, "s_neg", "sd"), "s_neg_q95": (g(dist, "s_neg", "q05_25_50_75_95") or [None] * 5)[-1],
        "T_pos_mean": g(dist, "T_pos", "mean"), "T_neg_mean": g(dist, "T_neg", "mean"),
        "residual_pos_mean": g(dist, "residual_pos", "mean"), "residual_neg_mean": g(dist, "residual_neg", "mean"),
        "one_minus_T2_pos_mean": g(dist, "one_minus_T2_pos", "mean"), "one_minus_T2_neg_mean": g(dist, "one_minus_T2_neg", "mean"),
        "a": g(dist, "critic", "a"), "b": g(dist, "critic", "b"), "kappa": g(dist, "critic", "kappa"),
        "frac_pos_below_kappa": dist.get("frac_pos_below_kappa"), "frac_neg_above_kappa": dist.get("frac_neg_above_kappa"),
        "loss": m.get("loss"), "Lp": m.get("Lp"), "Lq": m.get("Lq"), "n_pos": m.get("n_pos"),
        "iou_mean": g(m, "crop_overlap", "iou", "mean"), "iou_q05": (g(m, "crop_overlap", "iou", "q05_25_50_75_95") or [None])[0],
        "viewpair_mean_offdiag_cos": vp.get("mean_offdiag_cos"), "viewpair_cancellation": vp.get("cancellation_ratio"),
        "token_cancel_all": tp.get("cancellation_ratio_mean"), "token_cancel_strongcrop": tp.get("cancellation_ratio_tokens_with_partner_iou_lt_0.1"),
        "token_cancel_other": tp.get("cancellation_ratio_other_tokens"), "n_tokens_strongcrop": tp.get("n_tokens_with_partner_iou_lt_0.1"),
        "autograd_check_rel_err": tp.get("autograd_check_rel_err"),
        "h_norm_mean": g(geo, "h", "norm", "mean"), "h_mean_vector_norm": g(geo, "h", "mean_vector_norm"), "h_eff_rank": g(geo, "h", "effective_rank"),
        "h_top1_spectrum": (g(geo, "h", "spectrum_top16_normalised") or [None])[0],
        "z_eff_rank": g(geo, "z", "effective_rank"), "z_mean_vector_norm": g(geo, "z", "mean_vector_norm"),
    }
    for part in ("z", "h", "encoder", "projector"):
        P, Q = g(grad, part, "P"), g(grad, part, "Q")
        r[f"grad_{part}_P"], r[f"grad_{part}_Q"] = P, Q
        r[f"grad_{part}_P_over_Q"] = ratio(P, Q)
        r[f"grad_{part}_cos_PQ"] = g(grad, part, "cos_P_Q")
    groups = {}
    for kind in ("iou", "inter_area"):
        rows = []
        for grp in g(m, "crop_groups", kind) or []:
            rows.append({k: grp.get(k) for k in ("bin", "n_pos", "frac_pos", "s_pos_mean", "T_pos_mean", "frac_below_kappa", "loss_share",
                                                  "share_of_P_grad", "cos_with_P_grad", "grad_norm")})
        groups[kind] = rows
    iou = groups["iou"]
    if len(iou) >= 2:  # low-IoU = the two lowest bins, IoU < 0.25
        r["lowiou_frac_pos"] = sum(x["frac_pos"] or 0 for x in iou[:2])
        r["lowiou_share_of_P_grad"] = sum(x["share_of_P_grad"] or 0 for x in iou[:2])
        r["lowiou_loss_share"] = sum(x["loss_share"] or 0 for x in iou[:2])
        r["iou_lt0.1_share_of_P_grad"] = iou[0]["share_of_P_grad"]
        r["iou_lt0.1_frac_pos"] = iou[0]["frac_pos"]
        r["iou_lt0.1_s_pos_mean"] = iou[0]["s_pos_mean"]
        r["iou_lt0.1_frac_below_kappa"] = iou[0]["frac_below_kappa"]
        # gradient share per unit of pair share (> 1: these pairs pull harder than their count)
        r["lowiou_share_ratio"] = ratio(r["lowiou_share_of_P_grad"], r["lowiou_frac_pos"])
    return r, groups


def load_run(path: Path) -> dict:
    d = json.load(open(path))
    cks = {}
    for mp in d["checkpoint_map"]:
        if mp.get("duplicate"):
            continue
        c = d["checkpoints"][mp["checkpoint"]]
        r, groups = scalars(c["mean_over_batches"], c.get("geometry", {}))
        cks[int(mp["epoch"])] = {"checkpoint": mp["checkpoint"], "requested": mp["requested"], "note": mp["note"], "scalars": r, "groups": groups,
                                 "seconds": c.get("seconds")}
    lin, knn = acc(d["run"])
    return {"run": d["run"], "method": d["method"], "pair_scope": d.get("pair_scope"), "negative_detach": d.get("negative_detach"),
            "augmentation": d.get("augmentation"), "base_uids_sha256": d.get("base_uids_sha256"), "checkpoint_map": d["checkpoint_map"],
            "linear_800": lin, "knn_800": knn, "checkpoints": cks,
            "anomaly_epoch_field_none": all(d["checkpoints"][k].get("epoch") is None for k in d["checkpoints"])}


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    return f"{x:.{nd}f}"


def fs(x, nd=3):
    return "—" if x is None else f"{x:+.{nd}f}"


TRAJ_COLS = [("s_pos_mean", 3), ("s_pos_q05", 3), ("s_neg_mean", 3), ("T_pos_mean", 3), ("T_neg_mean", 3), ("kappa", 3), ("frac_pos_below_kappa", 4),
             ("grad_encoder_P_over_Q", 1), ("grad_encoder_cos_PQ", 3), ("grad_z_P_over_Q", 1), ("grad_z_cos_PQ", 3),
             ("viewpair_cancellation", 3), ("token_cancel_all", 3), ("token_cancel_strongcrop", 3), ("h_norm_mean", 2), ("h_eff_rank", 1)]
CONTRAST_KEYS = ["s_pos_mean", "s_pos_q05", "s_neg_mean", "T_pos_mean", "T_neg_mean", "frac_pos_below_kappa", "grad_encoder_P", "grad_encoder_Q",
                 "grad_encoder_P_over_Q", "grad_encoder_cos_PQ", "grad_z_P_over_Q", "grad_z_cos_PQ", "grad_h_P_over_Q", "viewpair_mean_offdiag_cos",
                 "viewpair_cancellation", "token_cancel_all", "token_cancel_strongcrop", "token_cancel_other", "lowiou_frac_pos", "lowiou_share_of_P_grad",
                 "lowiou_loss_share", "lowiou_share_ratio", "iou_lt0.1_share_of_P_grad", "iou_lt0.1_s_pos_mean", "h_norm_mean", "h_eff_rank", "z_eff_rank"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(OUT_ROOT / "P115_diag_results")); ap.add_argument("--out", default="reports/P115_diag_results")
    a = ap.parse_args()
    ind = Path(a.in_dir)
    runs, anomalies = {}, []
    wanted = [r for pair in CELLS.values() for r in pair] + list(JOB2.values())
    for r in wanted:
        p = ind / f"{r}.json"
        if p.is_file():
            runs[r] = load_run(p)
            if runs[r]["anomaly_epoch_field_none"]:
                anomalies.append(f"{r}: per-checkpoint 'epoch' field is None (epochs taken from checkpoint_map)")
        elif r in JOB2.values():
            pass
        else:
            anomalies.append(f"missing diagnostic output for {r}")
    contrasts = {}
    for meth, (std, strong) in CELLS.items():
        if std not in runs or strong not in runs:
            continue
        common = sorted(set(runs[std]["checkpoints"]) & set(runs[strong]["checkpoints"]))
        contrasts[meth] = {}
        for ep in common:
            s0, s1 = runs[std]["checkpoints"][ep]["scalars"], runs[strong]["checkpoints"][ep]["scalars"]
            contrasts[meth][ep] = {k: (None if s0.get(k) is None or s1.get(k) is None else s1[k] - s0[k]) for k in CONTRAST_KEYS}
            contrasts[meth][ep]["_std"] = {k: s0.get(k) for k in CONTRAST_KEYS}
            contrasts[meth][ep]["_strong"] = {k: s1.get(k) for k in CONTRAST_KEYS}
    # crop-box sanity: std runs share boxes, strong runs share boxes (same base images, same RNG, same augmentation family)
    for label, members in (("standard", [CELLS[m][0] for m in CELLS]), ("strong", [CELLS[m][1] for m in CELLS])):
        means = {r: runs[r]["checkpoints"][max(runs[r]["checkpoints"])]["scalars"]["iou_mean"] for r in members if r in runs}
        if means and max(means.values()) - min(means.values()) > 1e-6:
            anomalies.append(f"{label}-aug runs do not share crop-IoU means: {means}")
    # self-cosine on the view-pair cosine matrix diagonal (should be 1)
    for r, R in runs.items():
        p = ind / f"{r}.json"; d = json.load(open(p))
        for k, c in d["checkpoints"].items():
            cm = g(c, "mean_over_batches", "view_pairs", "cos_matrix")
            if cm:
                dmin = min(cm[i][i] for i in range(len(cm)))
                asym = max(abs(cm[i][j] - cm[j][i]) for i in range(len(cm)) for j in range(len(cm)))
                if dmin < 0.999 or asym > 1e-3:
                    anomalies.append(f"{r} {k}: view-pair cos_matrix diagonal min {dmin:.4f}, max asymmetry {asym:.4f} (expected 1 / 0)")
    out = {"runs": runs, "contrasts": contrasts, "acc_change_3seed": ACC_CHANGE_3SEED, "anomalies": anomalies,
           "note": "descriptive only; one seed per cell; strong − std at common checkpoints; SimCLR / recipe earliest checkpoint = epoch 100"}
    Path(a.out + ".json").write_text(json.dumps(out, indent=1, default=float))

    L = ["# P115 diagnostics — results (job 1: eight existing std / strong runs, seed 0)", "",
         "Descriptive tables from `outputs/P115_diag_results/` (frozen prereg `P115_DIAG_PREREG_FROZEN_20261003.md`).  One seed per cell; mean over the 2",
         "diagnostic batches (B 256, same FIT base images and diagnostic RNG for every run).  Ratios > 1 in `P/Q` = the positive term's gradient is larger.",
         "Cancellation ratios = ‖Σg‖ / Σ‖g‖ (1 = fully aligned, smaller = more cancellation).  Low-IoU = positive pairs with crop IoU < 0.25 (bins [0, 0.1) + [0.1, 0.25)).", "",
         "## 1. Runs and checkpoint maps", "",
         "| method | aug | run | objective / pairing / routing | crop scale min, jitter | linear / kNN @800 | checkpoints measured (requested → used) |",
         "|---|---|---|---|---|---|---|"]
    for meth, (std, strong) in CELLS.items():
        for lab, r in (("std", std), ("strong", strong)):
            if r not in runs:
                continue
            R = runs[r]; aug = R["augmentation"] or {}
            sc = g(aug, "random_resized_crop", "scale"); cj = g(aug, "color_jitter") or {}
            cmap = "; ".join(f"{m['requested']}→{m['checkpoint'].replace('.pt','')}" + (" (dup)" if m.get("duplicate") else "") for m in R["checkpoint_map"])
            L.append(f"| {meth} | {lab} | `{r}` | {R['method']} / {R['pair_scope']} / {'detach' if R['negative_detach'] else 'full'} | "
                     f"{sc[0] if sc else '—'}, {cj.get('brightness')}/{cj.get('contrast')}/{cj.get('saturation')}/{cj.get('hue')} | "
                     f"{fmt(R['linear_800'], 2)} / {fmt(R['knn_800'], 2)} | {cmap} |")
    L += ["", "3-seed mean linear change std → strong (P107 layer 2, P111, P89): " + ", ".join(f"{k} {v:+.2f}" for k, v in ACC_CHANGE_3SEED.items()) + ".", "",
          "## 2. Trajectories per run (epoch = checkpoint epoch)", ""]
    hdr = "| method | aug | epoch | " + " | ".join(c for c, _ in TRAJ_COLS) + " |"
    L += [hdr, "|---|---|---|" + "---|" * len(TRAJ_COLS)]
    for meth, (std, strong) in CELLS.items():
        for lab, r in (("std", std), ("strong", strong)):
            if r not in runs:
                continue
            for ep, ck in sorted(runs[r]["checkpoints"].items()):
                s = ck["scalars"]
                L.append(f"| {meth} | {lab} | {ep} | " + " | ".join(fmt(s.get(c), nd) for c, nd in TRAJ_COLS) + " |")
    L += ["", "## 3. Crop-IoU groups of positive pairs (share of the total positive-term encoder gradient; shares sum to 1)", "",
          "| method | aug | epoch | IoU bin | frac of positives | mean s | mean T | frac below κ | loss share | share of P-grad | cos with P-grad |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for meth, (std, strong) in CELLS.items():
        for lab, r in (("std", std), ("strong", strong)):
            if r not in runs:
                continue
            for ep, ck in sorted(runs[r]["checkpoints"].items()):
                for grp in ck["groups"]["iou"]:
                    b = grp["bin"]
                    L.append(f"| {meth} | {lab} | {ep} | [{b[0]:g}, {b[1]:g}) | {fmt(grp['frac_pos'])} | {fmt(grp['s_pos_mean'])} | {fmt(grp['T_pos_mean'])} | "
                             f"{fmt(grp['frac_below_kappa'])} | {fmt(grp['loss_share'])} | {fmt(grp['share_of_P_grad'])} | {fmt(grp['cos_with_P_grad'])} |")
    L += ["", "## 4. Std-vs-strong contrasts (strong − std) at common checkpoints", "",
          "| method (3-seed Δacc) | epoch | Δ s_pos | Δ s_pos q05 | Δ s_neg | Δ frac pos<κ | enc P/Q std→strong | Δ enc cos(P,Q) | z P/Q std→strong | "
          "low-IoU frac std→strong | low-IoU P-grad share std→strong | IoU<0.1 P-grad share std→strong | token cancel strong-crop std→strong | "
          "view-pair cancel std→strong | Δ h eff rank |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for meth, eps in contrasts.items():
        for ep, c in sorted(eps.items()):
            s0, s1 = c["_std"], c["_strong"]
            L.append(f"| {meth} ({ACC_CHANGE_3SEED[meth]:+.2f}) | {ep} | {fs(c['s_pos_mean'])} | {fs(c['s_pos_q05'])} | {fs(c['s_neg_mean'])} | {fs(c['frac_pos_below_kappa'], 4)} | "
                     f"{fmt(s0['grad_encoder_P_over_Q'], 1)} → {fmt(s1['grad_encoder_P_over_Q'], 1)} | {fs(c['grad_encoder_cos_PQ'])} | "
                     f"{fmt(s0['grad_z_P_over_Q'], 1)} → {fmt(s1['grad_z_P_over_Q'], 1)} | {fmt(s0['lowiou_frac_pos'])} → {fmt(s1['lowiou_frac_pos'])} | "
                     f"{fmt(s0['lowiou_share_of_P_grad'])} → {fmt(s1['lowiou_share_of_P_grad'])} | {fmt(s0['iou_lt0.1_share_of_P_grad'])} → {fmt(s1['iou_lt0.1_share_of_P_grad'])} | "
                     f"{fmt(s0['token_cancel_strongcrop'])} → {fmt(s1['token_cancel_strongcrop'])} | {fmt(s0['viewpair_cancellation'])} → {fmt(s1['viewpair_cancellation'])} | "
                     f"{fs(c['h_eff_rank'], 1)} |")
    if any(r in runs for r in JOB2.values()):
        L += ["", "## 5. Job 2 (crop-only / jitter-only) — present"]
    else:
        L += ["", "## 5. Job 2 (P115 crop-only / jitter-only runs) — pending (orchestrator unit P115_diag_job2)"]
    L += ["", "## Anomalies", ""] + ([f"- {x}" for x in anomalies] if anomalies else ["- none"])
    Path(a.out + ".md").write_text("\n".join(L) + "\n")
    print(f"-> {a.out}.md / .json  ({len(runs)} runs, {sum(len(R['checkpoints']) for R in runs.values())} checkpoints, {len(anomalies)} anomaly lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
