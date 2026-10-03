"""P123 (v6 V6-GRAD) aggregation — read-only on outputs/P123_grad_results/<run>.json; writes reports/P123_v6_grad_results.{md,json}.

Follows the frozen prereg (reports/P123_V6_GRAD_PREREG_FROZEN_20261003.md): descriptive only; actual and counterfactual-readonly kept in separate tables;
counterfactual numbers are instantaneous derivatives on fixed checkpoints (never training results); scalar action, vector norms and projection
contributions are reported as distinct quantities; one seed per cell; batch spread = sd over the 4 batches.

    python scripts/p123_aggregate.py [--in-dir outputs/P123_grad_results] [--out reports/P123_v6_grad_results]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

LOW_BINS = ((0.0, 0.1), (0.1, 0.25))      # "low IoU" = IoU < 0.25 (sum of the two lowest bins; projection contributions are additive)
SCORERS = ["affine_a1_k0.25", "affine_a1_k0.5", "affine_a1_k0.75", "affine_a2_k0.25", "affine_a2_k0.5", "affine_a2_k0.75",
           "affine_a3_k0.25", "affine_a3_k0.5", "affine_a3_k0.75", "curv_a2_k0.5_lam-0.25", "curv_a2_k0.5_lam+0.25"]
FAMILY = {"P107_AP3_views4_800ep_seed0": ("A-P3", "cifar10", "std"), "P107_AP3_augstrong_views4_800ep_seed0": ("A-P3", "cifar10", "strong"),
          "P104_G2_views4_800ep_seed0": ("G2", "cifar10", "std"), "P111_G2_augstrong_views4_800ep_seed0": ("G2", "cifar10", "strong"),
          "P41_simclr_views4_800ep_seed0": ("SimCLR", "cifar10", "std"), "P89_simclr_views4_800ep_augstrong_seed0": ("SimCLR", "cifar10", "strong"),
          "P35_vcs_a5_views4_800ep_seed0": ("recipe VCS", "cifar10", "std"), "P43_vcs_a5_views4_800ep_augstrong_seed0": ("recipe VCS", "cifar10", "strong"),
          "P107_AP3_c100_views4_800ep_seed0": ("A-P3", "cifar100", "std"), "P91_c100_vcs_a5_views4_800ep_seed0": ("recipe VCS", "cifar100", "std"),
          "P91_c100_simclr_views4_800ep_seed0": ("SimCLR", "cifar100", "std")}


def low_share(groups, key):
    return float(sum(g[key] for g in groups if tuple(g["bin"]) in LOW_BINS)) if groups and key in groups[0] else None


def block_summary(b):
    """Summary of one actual / counterfactual block (mean over batches already done by the runner)."""
    out = {}
    if "scorer" in b:
        out["scorer"] = b["scorer"]
    if "loss_kind" in b:
        out["loss_kind"] = b["loss_kind"]
    for k in ("Lp", "Lq", "loss"):
        if k in b:
            out[k] = b[k]
    if "peaks" in b:
        p = b["peaks"]
        out["peaks"] = {"actual_zero": p.get("actual_zero"), "pos_s_peak": p["pos"].get("s_peak_in_range"), "pos_max": p["pos"].get("max_abs_dA_ds_in_range"),
                        "neg_s_peak": p["neg"].get("s_peak_in_range"), "neg_max": p["neg"].get("max_abs_dA_ds_in_range"),
                        "pos_closed_form_in_range": p["pos"].get("closed_form_in_range"), "neg_closed_form_in_range": p["neg"].get("closed_form_in_range")}
    for side in ("pos", "neg"):
        if side not in b:
            continue
        s = b[side]; out[side] = {}
        for q in ("s", "f", "T", "c_minus_T", "fprime", "abs_dA_ds"):
            if q in s:
                out[side][q] = {"mean": s[q]["mean"], "sd": s[q]["sd"], "q05": s[q]["q05_25_50_75_95"][0], "q50": s[q]["q05_25_50_75_95"][2],
                                "q95": s[q]["q05_25_50_75_95"][4]}
        for q in ("scalar_total_abs_action", "mass_near_peak"):
            if q in s:
                out[side][q] = s[q]
    if "grad_vector_norms" in b:
        out["grad_vector_norms"] = b["grad_vector_norms"]
    g = b.get("iou_groups_pos")
    if g:
        out["iou_groups_pos"] = g
        out["low_iou_enc_proj_share"] = low_share(g, "enc_proj_contribution")
        out["low_iou_h_proj_share"] = low_share(g, "h_proj_contribution")
        out["low_iou_scalar_action_share"] = low_share(g, "scalar_action_share")
        out["low_iou_n_pos_frac"] = float(sum(x["n_pos"] for x in g if tuple(x["bin"]) in LOW_BINS) / sum(x["n_pos"] for x in g))
    if "semantic_view_isolated" in b:
        out["semantic_view_isolated"] = b["semantic_view_isolated"]
    return out


def batch_spread(batches, which, path):
    vals = []
    for bt in batches:
        b = bt[which] if which == "actual" else bt["counterfactual"].get(path[0])
        if b is None:
            continue
        v = path_get(b, path if which == "actual" else path[1:])
        if v is not None:
            vals.append(v)
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else None


def path_get(b, path):
    if path[0] == "low_enc":
        g = b.get("iou_groups_pos"); return low_share(g, "enc_proj_contribution") if g else None
    if path[0] == "low_h":
        g = b.get("iou_groups_pos"); return low_share(g, "h_proj_contribution") if g else None
    o = b
    for k in path:
        if not isinstance(o, dict) or k not in o:
            return None
        o = o[k]
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="/home/infres/yinwang/CS_QMI/outputs/P123_grad_results")
    ap.add_argument("--out", default="reports/P123_v6_grad_results"); a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.in_dir, "*.json")))
    R, anomalies = {}, []
    for f in files:
        d = json.load(open(f)); run = d["run"]
        fam, ds, aug = FAMILY.get(run, (run, d.get("dataset"), "?"))
        rec = {"family": fam, "dataset": ds, "aug": aug, "method": d.get("method"), "pair_scope": d.get("pair_scope"), "negative_detach": d.get("negative_detach"),
               "checkpoint_map": d["checkpoint_map"], "checkpoints": {}}
        for ck, c in d["checkpoints"].items():
            m = c["mean_over_batches"]; bs = c["batches"]
            if c.get("ckpt_epoch_field") is None:
                pass  # epoch taken from checkpoint_map / true_epoch
            act = block_summary(m["actual"])
            act["spread"] = {"low_iou_enc_proj_share_sd": batch_spread(bs, "actual", ["low_enc"]),
                             "enc_P_norm_sd": batch_spread(bs, "actual", ["grad_vector_norms", "enc", "P"]),
                             "pos_scalar_action_sd": batch_spread(bs, "actual", ["pos", "scalar_total_abs_action"]),
                             "pos_mass_near_peak_sd": batch_spread(bs, "actual", ["pos", "mass_near_peak"])}
            chk = [bt["actual"].get("check_vs_training_loss") for bt in bs if bt["actual"].get("check_vs_training_loss")]
            act["max_check_vs_training_loss_rel_err"] = max((max(x.values()) for x in chk), default=None)
            cf = {}
            for sc in SCORERS:
                if sc not in m["counterfactual"]:
                    anomalies.append(f"{run} {ck}: counterfactual {sc} missing"); continue
                cf[sc] = block_summary(m["counterfactual"][sc])
                cf[sc]["spread"] = {"low_iou_enc_proj_share_sd": batch_spread(bs, "cf", [sc, "low_enc"]),
                                    "pos_mass_near_peak_sd": batch_spread(bs, "cf", [sc, "pos", "mass_near_peak"])}
            g = c["geometry_eval"]
            rec["checkpoints"][ck] = {"true_epoch": c["true_epoch"], "n_batches": len(bs), "pairing": m["pairing"], "actual": act, "counterfactual": cf,
                                      "geometry_eval": {k: {"effective_rank": g[k]["effective_rank"], "norm_mean": g[k]["norm"]["mean"],
                                                            "mean_vector_norm": g[k]["mean_vector_norm"], "top1_spectrum": g[k]["spectrum_top16_normalised"][0]}
                                                        for k in ("h", "z")},
                                      "seconds": c["seconds"]}
            # identity check: for fixed-(2, -1) VCS runs the counterfactual affine (2, 0.5) equals the actual block
            sa = m["actual"].get("scorer", {})
            if sa.get("kind") == "affine" and abs(sa.get("a", 0) - 2.0) < 1e-9 and abs(sa.get("kappa", 0) - 0.5) < 1e-9:
                d1 = abs(m["actual"]["Lp"] - m["counterfactual"]["affine_a2_k0.5"]["Lp"]) + abs(m["actual"]["Lq"] - m["counterfactual"]["affine_a2_k0.5"]["Lq"])
                rec["checkpoints"][ck]["identity_actual_eq_cf_a2k05_abs_diff"] = d1
                if d1 > 1e-9:
                    anomalies.append(f"{run} {ck}: actual != counterfactual affine (2, 0.5) (|ΔL| {d1:.2e})")
            if len(bs) != 4:
                anomalies.append(f"{run} {ck}: {len(bs)} batches (expected 4)")
        R[run] = rec
    # ---------------------------------------------------------------- contrasts
    C = {"a_std_vs_strong": [], "b_counterfactual": [], "c_c10_vs_c100": []}
    def actual_key(rec, ck):
        c = rec["checkpoints"].get(ck);
        if c is None:
            return None
        a_ = c["actual"]; gv = a_["grad_vector_norms"]
        return {"low_enc": a_.get("low_iou_enc_proj_share"), "low_h": a_.get("low_iou_h_proj_share"), "low_n_frac": a_.get("low_iou_n_pos_frac"),
                "enc_PQ_ratio": gv["enc"]["P"] / gv["enc"]["Q"], "enc_cos_PQ": gv["enc"]["cos_P_Q"], "h_PQ_ratio": gv["h"]["P"] / gv["h"]["Q"],
                "pos_s_mean": a_["pos"]["s"]["mean"], "neg_s_mean": a_["neg"]["s"]["mean"],
                "pos_T_mean": a_["pos"].get("T", {}).get("mean"), "pos_mass_near_peak": a_["pos"].get("mass_near_peak"),
                "pos_scalar_action": a_["pos"].get("scalar_total_abs_action"), "neg_scalar_action": a_["neg"].get("scalar_total_abs_action"),
                "h_eff_rank": c["geometry_eval"]["h"]["effective_rank"], "z_eff_rank": c["geometry_eval"]["z"]["effective_rank"],
                "sem_same_class_frac": (a_.get("semantic_view_isolated") or {}).get("frac_same_class_neg"),
                "sem_action_share": (a_.get("semantic_view_isolated") or {}).get("scalar_action_share_same_class")}
    pairs = [("A-P3", "P107_AP3_views4_800ep_seed0", "P107_AP3_augstrong_views4_800ep_seed0"), ("G2", "P104_G2_views4_800ep_seed0", "P111_G2_augstrong_views4_800ep_seed0"),
             ("SimCLR", "P41_simclr_views4_800ep_seed0", "P89_simclr_views4_800ep_augstrong_seed0"), ("recipe VCS", "P35_vcs_a5_views4_800ep_seed0", "P43_vcs_a5_views4_800ep_augstrong_seed0")]
    for fam, rs, rt in pairs:
        for ck in ("epoch_100.pt", "epoch_400.pt", "epoch_800.pt"):
            ks, kt = actual_key(R[rs], ck), actual_key(R[rt], ck)
            if ks and kt:
                C["a_std_vs_strong"].append({"family": fam, "checkpoint": ck, "std": ks, "strong": kt,
                                             "delta": {k: (kt[k] - ks[k]) if (ks[k] is not None and kt[k] is not None) else None for k in ks}})
    for run, rec in R.items():
        c = rec["checkpoints"].get("epoch_800.pt")
        if c is None:
            continue
        rows = []
        for sc in SCORERS:
            b = c["counterfactual"].get(sc)
            if not b:
                continue
            rows.append({"scorer": sc, "a": b["scorer"]["a"], "kappa": b["scorer"]["kappa"], "lam": b["scorer"]["lam"],
                         "zero": b["peaks"]["actual_zero"], "pos_s_peak": b["peaks"]["pos_s_peak"], "neg_s_peak": b["peaks"]["neg_s_peak"],
                         "pos_s_q50": b["pos"]["s"]["q50"], "pos_mass_near_peak": b["pos"]["mass_near_peak"], "neg_mass_near_peak": b["neg"]["mass_near_peak"],
                         "pos_mean_abs_dA": b["pos"]["abs_dA_ds"]["mean"], "neg_mean_abs_dA": b["neg"]["abs_dA_ds"]["mean"],
                         "pos_T_mean": b["pos"]["T"]["mean"], "neg_T_mean": b["neg"]["T"]["mean"],
                         "enc_P": b["grad_vector_norms"]["enc"]["P"], "enc_Q": b["grad_vector_norms"]["enc"]["Q"], "enc_cos_PQ": b["grad_vector_norms"]["enc"]["cos_P_Q"],
                         "low_enc": b.get("low_iou_enc_proj_share"), "low_h": b.get("low_iou_h_proj_share"), "low_action": b.get("low_iou_scalar_action_share"),
                         "low_enc_sd": b["spread"]["low_iou_enc_proj_share_sd"], "sem_action_share": (b.get("semantic_view_isolated") or {}).get("scalar_action_share_same_class")})
        C["b_counterfactual"].append({"run": run, "family": rec["family"], "dataset": rec["dataset"], "aug": rec["aug"], "pair_scope": c["pairing"]["scope"],
                                      "simclr_counterfactual_pairing": c["pairing"].get("simclr_counterfactual_pairing"), "rows": rows})
    for fam, r10, r100 in (("A-P3", "P107_AP3_views4_800ep_seed0", "P107_AP3_c100_views4_800ep_seed0"), ("recipe VCS", "P35_vcs_a5_views4_800ep_seed0", "P91_c100_vcs_a5_views4_800ep_seed0"),
                           ("SimCLR", "P41_simclr_views4_800ep_seed0", "P91_c100_simclr_views4_800ep_seed0")):
        for ck in ("epoch_100.pt", "epoch_400.pt", "epoch_800.pt"):
            k10, k100 = actual_key(R[r10], ck), actual_key(R[r100], ck)
            if k10 and k100:
                C["c_c10_vs_c100"].append({"family": fam, "checkpoint": ck, "c10": k10, "c100": k100})
    out = {"in_dir": a.in_dir, "n_files": len(files), "runs": R, "contrasts": C, "anomalies": anomalies}
    Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    # ---------------------------------------------------------------- markdown
    f2 = lambda x, n=3: ("—" if x is None else f"{x:.{n}f}")
    L = ["# P123 — v6 V6-GRAD results (actual vs counterfactual-readonly; one seed per cell; mean over 4 batches)", "",
         "Counterfactual blocks are instantaneous derivatives on fixed checkpoints with only f replaced — never training results.  "
         "Low IoU = positives with true crop IoU < 0.25 (bins [0, 0.1) + [0.1, 0.25)).  Projection contribution ⟨g_b, g_P⟩/‖g_P‖² (sums to 1 over bins).", "",
         "## 1. Checkpoint coverage", "", "| run | family | data | aug | pair scope | checkpoints (requested → true epoch) |", "|---|---|---|---|---|---|"]
    for run, rec in R.items():
        mp = "; ".join(f"{x['requested']}→{x['epoch']}{' (dup)' if x['duplicate'] else ''}" for x in rec["checkpoint_map"])
        scope = (f"{rec['pair_scope']}, neg detach {rec['negative_detach']}" if rec["method"] == "vcs_qmi"
                 else "NT-Xent (counterfactual: all view tokens, full gradient)")
        L.append(f"| {run} | {rec['family']} | {rec['dataset']} | {rec['aug']} | {scope} | {mp} |")
    L += ["", "## 2. Actual (each checkpoint's own loss / pairing / scorer)", "",
          "| run | ep | scorer (a, κ) | pos s mean | neg s mean | pos T | pos mass near peak | pos scalar action | neg scalar action | ‖g_P‖ enc | ‖g_Q‖ enc | cos(P,Q) enc | low-IoU enc share (sd) | low-IoU h share | low-IoU pos frac | h eff rank |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, rec in R.items():
        for ck, c in rec["checkpoints"].items():
            a_ = c["actual"]; gv = a_["grad_vector_norms"]; sc = a_.get("scorer")
            scs = f"({sc['a']:.2f}, {sc['kappa']:.3f})" if sc else "NT-Xent"
            L.append(f"| {run} | {c['true_epoch']} | {scs} | {f2(a_['pos']['s']['mean'])} | {f2(a_['neg']['s']['mean'])} | {f2(a_['pos'].get('T', {}).get('mean'))} | "
                     f"{f2(a_['pos'].get('mass_near_peak'))} | {f2(a_['pos'].get('scalar_total_abs_action'))} | {f2(a_['neg'].get('scalar_total_abs_action'))} | "
                     f"{gv['enc']['P']:.4f} | {gv['enc']['Q']:.4f} | {gv['enc']['cos_P_Q']:+.3f} | {f2(a_.get('low_iou_enc_proj_share'))} ({f2(a_['spread']['low_iou_enc_proj_share_sd'])}) | "
                     f"{f2(a_.get('low_iou_h_proj_share'))} | {f2(a_.get('low_iou_n_pos_frac'))} | {c['geometry_eval']['h']['effective_rank']:.1f} |")
    L += ["", "### 2b. Actual: IoU-group projection contributions at epoch 800 (enc / h), with ‖g_b‖ (enc)", "",
          "| run | IoU [0,0.1) n / enc / h / ‖g‖ | [0.1,0.25) | [0.25,0.5) | [0.5,1] |", "|---|---|---|---|---|"]
    for run, rec in R.items():
        c = rec["checkpoints"].get("epoch_800.pt")
        if not c:
            continue
        cells = [f"{g['n_pos']:.0f} / {g['enc_proj_contribution']:+.3f} / {g['h_proj_contribution']:+.3f} / {g['enc_norm']:.4f}" for g in c["actual"]["iou_groups_pos"]]
        L.append(f"| {run} | " + " | ".join(cells) + " |")
    L += ["", "### 2c. Isolated semantic view (labels read only; never used to filter or weight), actual, epoch 800", "",
          "| run | same-class neg fraction | their share of neg scalar action | their h projection contribution |", "|---|---|---|---|"]
    for run, rec in R.items():
        c = rec["checkpoints"].get("epoch_800.pt"); sv = c and c["actual"].get("semantic_view_isolated")
        if sv:
            L.append(f"| {run} | {sv['frac_same_class_neg']:.3f} | {sv['scalar_action_share_same_class']:.3f} | {sv['h_proj_contribution_same_class']:+.3f} |")
    L += ["", "## 3. Counterfactual-readonly at epoch 800 (same checkpoint, images, pairs, routing; only f replaced)", ""]
    for blk in C["b_counterfactual"]:
        note = " (SimCLR checkpoint: counterfactual pairing = all view tokens, full gradient)" if blk["simclr_counterfactual_pairing"] else f" (pairing {blk['pair_scope']})"
        L += [f"### {blk['run']}{note}", "",
              "| scorer | zero | pos peak s | pos s median | pos mass near peak | neg mass near peak | mean \\|∂A/∂s\\| pos / neg | pos T / neg T | ‖g_P‖ / ‖g_Q‖ enc | cos(P,Q) | low-IoU enc share (sd) | low-IoU h share | low-IoU scalar-action share |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in blk["rows"]:
            L.append(f"| {r['scorer']} | {r['zero']:.3f} | {r['pos_s_peak']:.3f} | {r['pos_s_q50']:.3f} | {r['pos_mass_near_peak']:.4f} | {r['neg_mass_near_peak']:.4f} | "
                     f"{r['pos_mean_abs_dA']:.3f} / {r['neg_mean_abs_dA']:.3f} | {r['pos_T_mean']:+.3f} / {r['neg_T_mean']:+.3f} | {r['enc_P']:.4f} / {r['enc_Q']:.4f} | "
                     f"{r['enc_cos_PQ']:+.3f} | {f2(r['low_enc'])} ({f2(r['low_enc_sd'])}) | {f2(r['low_h'])} | {f2(r['low_action'])} |")
        L.append("")
    L += ["## 4. Contrast (a): standard vs strong augmentation within each method (actual)", "",
          "| family | ep | low-IoU enc share std → strong | low-IoU h share | low-IoU pos frac | ‖g_P‖/‖g_Q‖ enc | cos(P,Q) enc | pos s mean | neg s mean | pos mass near peak | h eff rank |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in C["a_std_vs_strong"]:
        s, t = x["std"], x["strong"]; ep = x["checkpoint"].split("_")[1].split(".")[0].lstrip("0")
        arrow = lambda k, n=3: f"{f2(s[k], n)} → {f2(t[k], n)}"
        L.append(f"| {x['family']} | {ep} | {arrow('low_enc')} | {arrow('low_h')} | {arrow('low_n_frac')} | {arrow('enc_PQ_ratio', 1)} | {arrow('enc_cos_PQ')} | "
                 f"{arrow('pos_s_mean')} | {arrow('neg_s_mean')} | {arrow('pos_mass_near_peak', 4)} | {arrow('h_eff_rank', 1)} |")
    L += ["", "## 5. Contrast (c): CIFAR-10 vs CIFAR-100 for the same method (actual; normalised quantities only — shares, ratios, cosines, ranks)", "",
          "| family | ep | low-IoU enc share C10 / C100 | low-IoU h share | ‖g_P‖/‖g_Q‖ enc | cos(P,Q) enc | pos s mean | neg s mean | pos T | h eff rank | same-class neg frac | their neg action share |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in C["c_c10_vs_c100"]:
        p, q = x["c10"], x["c100"]; ep = x["checkpoint"].split("_")[1].split(".")[0].lstrip("0")
        sl = lambda k, n=3: f"{f2(p[k], n)} / {f2(q[k], n)}"
        L.append(f"| {x['family']} | {ep} | {sl('low_enc')} | {sl('low_h')} | {sl('enc_PQ_ratio', 1)} | {sl('enc_cos_PQ')} | {sl('pos_s_mean')} | {sl('neg_s_mean')} | "
                 f"{sl('pos_T_mean')} | {sl('h_eff_rank', 1)} | {sl('sem_same_class_frac')} | {sl('sem_action_share')} |")
    L += ["", "## 6. Identities and anomalies", "",
          "- Actual block vs the P115 training-loss decomposition: max relative error over all VCS batches "
          + f"{max((c['actual']['max_check_vs_training_loss_rel_err'] or 0) for r in R.values() for c in r['checkpoints'].values()):.1e} (SimCLR blocks have no VCS check).",
          "- Fixed-(2, −1) runs: counterfactual affine (2, 0.5) equals the actual block (|ΔLp| + |ΔLq| max "
          + f"{max((c.get('identity_actual_eq_cf_a2k05_abs_diff') or 0) for r in R.values() for c in r['checkpoints'].values()):.1e}).",
          "- Anomalies: " + ("; ".join(anomalies) if anomalies else "none detected by the aggregator.")]
    Path(a.out + ".md").write_text("\n".join(L) + "\n")
    print(f"-> {a.out}.md / .json  ({len(files)} runs, anomalies {len(anomalies)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
