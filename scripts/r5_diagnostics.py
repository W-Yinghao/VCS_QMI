"""r5 R5-20A / B / C diagnostics on the saved score matrices of full fine-tuning checkpoints (package r5 docs/04 §3–5, docs/03 §4–7,
machine/diagnostic_design.json).  Read-only: cosines from scripts/r4_vl0_scores.py (sources r4vl1, r4vl2, zeroshot, r5dist) and each run's
CAL-selected critic (a, b); nothing is fitted.  Exploratory (DEV has been reused); the image is the independent unit.

Models, each with its own training law: R4-VL1 VCS / matched JS / set softmax (ν uniform); R4-VL2 VCS(ν) / JS(ν) (ν ∝ detector score); r5
distribution softmax (ν uniform; when present).  Per query (π = 1; conditional p_r = A_r / k, q_r = ν_r): f = a cos + b, T = tanh f; deployment
score f (uniform law) or 2f + log ν (ν law); candidate probability ρ = softmax(2f) or softmax(2f + log ν).  Score-level gradients (not encoder
gradients), with P / Q components kept apart:
    g_V = [p(T − 1) + q(T + 1)](1 − T²)    g_L = p(T − 1) + q(T + 1)    g_E = 2(ρ_s − ρ_A)    g_D = 2(ρ_s − p)      (ρ_s = softmax(2f))
A  hard distractors: invalid candidates ranked by the model's own deployment score (self) or by zero-shot SigLIP 2 B/16 cosine (shared, the same
   for every model); top 1 / 5: q-mass and its share of all invalid q-mass, |g| share (NA for a zero denominator), Σ|g_f|, Σ|g_cos| = a Σ|g_f|,
   signed Σ g_f, R4 error types and IoU band of the top-1.  Per query: valid-set P / Q / net gradient, invalid net gradient, posterior residual
   T − η (η = −1 on invalid), saturation (invalid T ≥ 0.95 and its |g| share; |T| ≥ 0.95 over all), near-overlap (IoU 0.3–0.5) vs low (< 0.3)
   q / |g| shares, training weight π_w / 32.  T-based fields of softmax models are flagged: the loss does not identify their offset b.
B  multiplicity: k = #valid (bins 1 / 2–3 / ≥ 4; k = 0 = uncovered), m = #real proposals (1–15 / 16–31 / ≥ 32); per bin image-macro accuracy
   (uncovered queries count as errors), coverage, counts, mean valid IoU, h = Σ_A ρ, KL(U_A ‖ ρ_A) (0 at k = 1, checked), max ρ_A, entropy of ρ_A,
   IoU of the top-scoring valid box; m-standardised accuracy per k bin (auxiliary; NA without support).
C  nested candidates: queries with ≥ 31 invalid and ≥ 1 valid; target = max-IoU valid box (ties: detector score desc, proposal index asc); invalid
   counts 1, 3, 7, 15, 31 from the r5 reference nested_invalid_ids (SHA-256 of query id and repeat, prefix-nested, 32 repeats) and all; hit = the
   target's deployment score ≥ every selected invalid score; repeats → query → image → images.
Statistics: per seed; seed mean; image bootstrap with ONE index matrix per dataset shared by all models, seeds, bins and contrasts; paired
contrasts = seed-averaged per-image differences.  Every missing input is listed, never filled.
Writes per-model / per-bin summaries (no per-query dump) to reports/R5/{gradient_diagnostics.jsonl, gradient_summary.md, multiplicity_diagnostics.jsonl,
support_counts.json, nested_query_manifest.json, nested_curves.json, r5_diagnostics_inputs.json} (smoke: reports/R5/smoke/); missing inputs are listed.
    python scripts/r5_diagnostics.py [--datasets refcocog refcocoplus] [--role DEV] [--parts A B C] [--boot 1000] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r5" / "package_r5_20261011" / "reference"))
from vl1_22_tableC import iou_xyxy  # noqa: E402
from r4_vl0_scores import meta_path, score_path  # noqa: E402
from r4_vl0_analyse import ETYPES, geometry  # noqa: E402
from r5_reference import nested_invalid_ids  # noqa: E402

ROOT = Path("/home/infres/yinwang/CS_QMI/outputs"); OUT = Path(__file__).resolve().parents[1] / "reports" / "R5"; BB, SEEDS = "siglip2_b16", (0, 1, 2)
MODELS = (("r4vl1", "vcs", "uniform", "vcs", "VCS"), ("r4vl1", "js", "uniform", "balanced_logistic", "JS"), ("r4vl1", "softmax", "uniform", "set_softmax", "set softmax"),
          ("r4vl2", "vcs", "score", "vcs", "VCS(nu)"), ("r4vl2", "js", "score", "balanced_logistic", "JS(nu)"),
          ("r5dist", "distribution_softmax", "uniform", "distribution_softmax", "distribution softmax"),
          ("r5qnorm", "vcs_qnorm", "uniform", "vcs_qnorm", "VCS(qnorm)"), ("r5qnorm", "js_qnorm", "uniform", "balanced_logistic_qnorm", "JS(qnorm)"))   # R5-QN
CONTRASTS = (("r4vl2:vcs", "r4vl1:vcs"), ("r4vl2:js", "r4vl1:js"), ("r4vl1:vcs", "r4vl1:softmax"), ("r4vl2:vcs", "r4vl1:softmax"),
             ("r5dist:distribution_softmax", "r4vl1:softmax"), ("r4vl1:vcs", "r5dist:distribution_softmax"), ("r4vl1:js", "r5dist:distribution_softmax"),
             ("r5qnorm:vcs_qnorm", "r4vl1:softmax"), ("r5qnorm:vcs_qnorm", "r4vl1:vcs"), ("r5qnorm:js_qnorm", "r4vl1:js"), ("r5qnorm:js_qnorm", "r4vl1:softmax"))
warnings.filterwarnings("ignore", message="Mean of empty slice"); warnings.filterwarnings("ignore", message="invalid value encountered")
K_BINS, M_BINS, NESTED, REPEATS, TOPS, SAT = (("1", 1, 1), ("2-3", 2, 3), ("4+", 4, None)), (("1-15", 1, 15), ("16-31", 16, 31), ("32+", 32, None)), (1, 3, 7, 15, 31), 32, (1, 5), 0.95


def key(m):
    return f"{m[0]}:{m[1]}"


def run_json(ds, m, s):
    sub = {"r4vl1": ("R4_VL1", ""), "r4vl2": ("R4_VL2", "_nuscore"), "r5dist": ("R5_DIST", ""), "r5qnorm": ("R5_QNORM", "")}[m[0]]
    return ROOT / sub[0] / f"{ds}_{BB}_{m[1]}{sub[1]}_s{s}.json"


def in_bin(x, lo, hi):
    return x >= lo and (hi is None or x <= hi)


def load_scores(ds, role, source, objective, seed, meta, missing, allow_prefix=False):
    """Cosine matrices aligned with the meta (the image order is checked; smoke runs may use an exact prefix of the images)."""
    f = score_path(ds, BB, objective, seed, role, source)
    if not f.exists():
        missing.append(str(f)); return None
    d = torch.load(f, weights_only=False); ids, sc = list(d.get("image_ids", [])), d["scores"]
    if allow_prefix and len(sc) > len(meta):
        ids, sc = ids[:len(meta)], sc[:len(meta)]
    if len(sc) != len(meta) or (ids and ids != [m["image_id"] for m in meta]):
        raise ValueError(f"{f}: image order differs from the meta")
    for i, (mm, s) in enumerate(zip(meta, sc)):
        if tuple(s["cos_prop"].shape) != (len(mm["prop_boxes"]), len(mm["target"])):
            raise ValueError(f"{f}: image {i} score shape {tuple(s['cos_prop'].shape)} vs meta ({len(mm['prop_boxes'])}, {len(mm['target'])})")
    return [s["cos_prop"].numpy().astype(np.float64) for s in sc]


def critic(ds, m, s, missing):
    f = run_json(ds, m, s)
    if not f.exists():
        missing.append(str(f)); return None
    a, b = json.load(open(f))["selected"]["critic_a_b"]; return float(a), float(b)


def softmax_cols(x):
    z = x - x.max(0, keepdims=True); e = np.exp(z); return e / e.sum(0, keepdims=True)


def model_arrays(cp, det, law, loss, ab):
    """Per image [n_p, n_q]: f, T, deployment score, own candidate probability ρ (law), softmax(2f), log ν."""
    a, b = ab; f = a * cp + b; T = np.tanh(f); n_p = cp.shape[0]
    logn = np.log(det / det.sum()) if law == "score" else np.full(n_p, -math.log(n_p))
    if loss.endswith("_qnorm") and n_p:                                                  # R5-QN: the trained critic is g = f − ½ log Σ ν exp(2f), per query
        z = 2 * f + logn[:, None]; zm = z.max(0, keepdims=True); T = np.tanh(f - 0.5 * (zm + np.log(np.exp(z - zm).sum(0, keepdims=True))))
    deploy = f if law == "uniform" else 2 * f + logn[:, None]
    rho = softmax_cols(2 * f + (logn[:, None] if law == "score" else 0.0)); rho_s = softmax_cols(2 * f)
    return f, T, deploy, rho, rho_s, logn


def score_gradients(loss, p, q, T, rho_s, valid):
    """π = 1 score-level gradients of one query column (arrays over candidates): (P component, Q component, net)."""
    if loss == "vcs":
        gp, gq = p * (T - 1) * (1 - T * T), q * (T + 1) * (1 - T * T)
    elif loss == "balanced_logistic":
        gp, gq = p * (T - 1), q * (T + 1)
    elif loss == "set_softmax":
        h = (rho_s * valid).sum(); gp, gq = -2 * rho_s * valid / h, 2 * rho_s
    elif loss == "distribution_softmax":
        gp, gq = -2 * p, 2 * rho_s
    elif loss in ("vcs_qnorm", "balanced_logistic_qnorm"):                                 # through the normalizer: ∂L/∂f = ∂L/∂g − ρ̃ Σ ∂L/∂g, ρ̃ = softmax(2f) (uniform ν)
        gp_g, gq_g, _ = score_gradients(loss[:-6], p, q, T, rho_s, valid)                # T = tanh(g) from model_arrays
        gp, gq = gp_g - rho_s * gp_g.sum(), gq_g - rho_s * gq_g.sum()
    else:
        raise ValueError(loss)
    return gp, gq, gp + gq


def kl_uniform(rho, valid):
    """KL(U_A ‖ ρ_A) with ρ_A = ρ restricted to the valid set and renormalised; 0 when k = 1."""
    k = int(valid.sum()); h = rho[valid].sum(); return float(-math.log(k) - np.log(rho[valid]).mean() + math.log(h)), float(h)


def frac(num, den):
    return num / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------------------------------------------- per-dataset data
def dataset(ds, role, smoke):
    meta = torch.load(meta_path(ds, role), weights_only=False)
    if smoke:
        meta = meta[:150]
    G = geometry(meta); return meta, G


def boot_matrix(n_img, B, ds_index):
    return np.random.default_rng(np.random.SeedSequence([20261011, 520, ds_index])).integers(0, n_img, (B, n_img))


def boot_stats(per_img, idx):
    """Image-macro (nanmean over images defining the quantity) and its bootstrap percentile interval with the shared index matrix."""
    x = np.asarray(per_img, float); ok = np.isfinite(x)
    if not ok.any():
        return {"mean": None, "ci95": None, "n_images": 0}
    xs = np.where(ok, x, 0.0)[idx]; ws = ok[idx].astype(float); bm = xs.sum(1) / np.maximum(ws.sum(1), 1)
    bm = bm[ws.sum(1) > 0]
    return {"mean": float(x[ok].mean()), "ci95": [float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))] if len(bm) else None, "n_images": int(ok.sum())}


def macro_mean(per_img):
    x = np.asarray(per_img, float); ok = np.isfinite(x); return float(x[ok].mean()) if ok.any() else None


def image_means(rows, field):
    """rows: per image list of per-query dicts -> per-image nanmean of a field (nan when undefined)."""
    out = []
    for r in rows:
        v = [q[field] for q in r if field in q and q[field] is not None and np.isfinite(q[field])]
        out.append(float(np.mean(v)) if v else np.nan)
    return np.array(out)


# ---------------------------------------------------------------------------------------------------------------- A: gradients
def part_a_image(mm, g, cp, zs, m, ab, n_cov):
    det = np.asarray(mm["det_score"], float); n_p = cp.shape[0]; rows = []
    if n_p == 0:
        return rows, {"no_valid": len(mm["target"]), "no_invalid": 0}
    f, T, deploy, rho, rho_s, logn = model_arrays(cp, det, m[2], m[3], ab); nu = np.exp(logn); counts = {"no_valid": 0, "no_invalid": 0}
    iou = g["iou"]
    for j in range(len(mm["target"])):
        valid = iou[j] >= 0.5; inv = ~valid; k = int(valid.sum())
        if k == 0:
            counts["no_valid"] += 1; continue
        if not inv.any():
            counts["no_invalid"] += 1; continue
        p = valid / k; q = nu.copy(); Tj = T[:, j]; gp, gq, gnet = score_gradients(m[3], p, q, Tj, rho_s[:, j], valid); ag = np.abs(gnet)
        eta_v = (p[valid] - q[valid]) / (p[valid] + q[valid])
        r = {"valid_gP": float(gp[valid].sum()), "valid_gQ": float(gq[valid].sum()), "valid_g": float(gnet[valid].sum()), "invalid_g": float(gnet[inv].sum()),
             "resid_valid": float((Tj[valid] - eta_v).mean()), "resid_invalid": float((Tj[inv] + 1).mean()),
             "wrong_sat_frac": float((Tj[inv] >= SAT).mean()), "wrong_sat_grad_share": frac(ag[inv & (Tj >= SAT)].sum(), ag[inv].sum()),
             "bounded_sat_frac": float((np.abs(Tj) >= SAT).mean()), "train_weight": 1.0 / max(1, n_cov) / 32}
        near = inv & (iou[j] >= 0.3); low = inv & (iou[j] < 0.3)
        r.update(near_q_share=frac(q[near].sum(), q[inv].sum()), near_grad_share=frac(ag[near].sum(), ag[inv].sum()), near_count=int(near.sum()), low_count=int(low.sum()))
        inv_ids = np.flatnonzero(inv)
        for order_name, sc in (("self", deploy[:, j]), ("shared", None if zs is None else zs[:, j])):
            if sc is None:
                continue
            order = inv_ids[np.argsort(-sc[inv_ids], kind="stable")]
            for t in TOPS:
                ch = order[:t]; pre = f"{order_name}_top{t}_"
                r.update({pre + "q_mass": float(q[ch].sum()), pre + "q_frac": frac(q[ch].sum(), q[inv].sum()), pre + "grad_frac": frac(ag[ch].sum(), ag[inv].sum()),
                          pre + "abs_f_grad": float(ag[ch].sum()), pre + "abs_cos_grad": float(ab[0] * ag[ch].sum()), pre + "signed_f_grad": float(gnet[ch].sum()),
                          pre + "selected": int(len(ch))})
                if t == 1:
                    c0 = int(ch[0]); et = g["etype"][j][c0]
                    for e in ETYPES:
                        r[pre + "type_" + e] = float(et == e)
                    r[pre + "near_overlap"] = float(0.3 <= iou[j, c0] < 0.5); r[pre + "T"] = float(Tj[c0])
        rows.append(r)
    return rows, counts


# ---------------------------------------------------------------------------------------------------------------- B: multiplicity
def part_b_image(mm, g, cp, m, ab):
    det = np.asarray(mm["det_score"], float); n_p = cp.shape[0] if cp is not None else 0; rows = []
    for j in range(len(mm["target"])):
        cat = int(mm["ref_cat"][mm["target"][j]])
        if n_p == 0:
            rows.append({"k": 0, "m": 0, "acc": 0.0, "cat": cat}); continue
        f, T, deploy, rho, rho_s, logn = model_arrays(cp[:, j:j + 1], det, m[2], m[3], ab)
        valid = g["iou"][j] >= 0.5; k = int(valid.sum()); top = int(np.argmax(deploy[:, 0]))
        r = {"k": k, "m": n_p, "acc": float(valid[top]), "cat": cat}
        if k:
            kl, h = kl_uniform(rho[:, 0], valid); rA = rho[valid, 0] / h; vi = np.flatnonzero(valid)
            r.update(kl=kl, h=h, max_rhoA=float(rA.max()), ent_rhoA=float(-(rA * np.log(rA)).sum()), valid_iou=float(g["iou"][j][valid].mean()),
                     top_valid_iou=float(g["iou"][j][vi[np.argmax(deploy[vi, 0])]]))
        rows.append(r)
    return rows


def bin_of(r, kind):
    if kind == "k":
        return "0" if r["k"] == 0 else next(n for n, lo, hi in K_BINS if in_bin(r["k"], lo, hi))
    return "0" if r["m"] == 0 else next(n for n, lo, hi in M_BINS if in_bin(r["m"], lo, hi))


def per_image_bin(rows_img, kind, b, field):
    out = []
    for rows in rows_img:
        v = [r[field] for r in rows if bin_of(r, kind) == b and field in r]
        out.append(float(np.mean(v)) if v else np.nan)
    return np.array(out)


# ---------------------------------------------------------------------------------------------------------------- C: nested
def nested_manifest(meta, G):
    """Eligible queries (>= 31 invalid, >= 1 valid), the fixed target and the 32 nested invalid orders (shared by every model)."""
    items = []
    for i, (mm, g) in enumerate(zip(meta, G)):
        det = np.asarray(mm["det_score"], float); n_p = len(mm["prop_boxes"])
        if n_p == 0:
            continue
        for j in range(len(mm["target"])):
            valid = g["iou"][j] >= 0.5; inv = np.flatnonzero(~valid)
            if not valid.any() or len(inv) < max(NESTED):
                continue
            vi = np.flatnonzero(valid); tgt = int(min(vi, key=lambda r: (-g["iou"][j][r], -det[r], r)))
            qid = f"{mm['image_id']}:{j}"; ids = [f"{r:03d}" for r in inv]
            orders = np.array([[int(x) for x in nested_invalid_ids(qid, ids, rep, NESTED)["all"][:max(NESTED)]] for rep in range(REPEATS)], dtype=np.int16)
            items.append({"image": i, "query": j, "query_id": qid, "target": tgt, "n_invalid": int(len(inv)), "orders": orders, "invalid": inv,
                          "cat": int(mm["ref_cat"][mm["target"][j]])})
    return items


def nested_hits(items, deploy_by_image):
    """Per eligible query: hit rate at each count (mean over repeats) and at 'all'."""
    out = []
    for it in items:
        d = deploy_by_image[it["image"]][:, it["query"]]; dt = d[it["target"]]
        cm = np.maximum.accumulate(d[it["orders"]], axis=1)                                        # [32, 31] running max of the nested prefixes
        h = {str(c): float((dt >= cm[:, c - 1]).mean()) for c in NESTED}; h["all"] = float(dt >= d[it["invalid"]].max()); out.append(h)
    return out


def per_image_from_queries(items, vals, n_img, field):
    acc = [[] for _ in range(n_img)]
    for it, v in zip(items, vals):
        acc[it["image"]].append(v[field])
    return np.array([float(np.mean(a)) if a else np.nan for a in acc])


# ---------------------------------------------------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--datasets", nargs="+", default=["refcocog", "refcocoplus"]); ap.add_argument("--role", default="DEV", choices=["DEV", "CAL"])
    ap.add_argument("--parts", nargs="+", default=["A", "B", "C"]); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    out = OUT / "smoke" if a.smoke else OUT; out.mkdir(parents=True, exist_ok=True); B = 50 if a.smoke else a.boot
    inputs = {"role": a.role, "missing": [], "used": [], "epoch_anchors": "missing: only CAL-selected checkpoints are saved (no epoch 1 / 5 / 10 states)"}
    gl, ml, summ_md = [], [], ["# r5 R5-20A — hard-distractor reference mass and score-level gradients (exploratory, DEV)", "",
                               "Score-level derivatives of each model's own training loss at its CAL-selected state; not encoder gradients.  Image-macro over queries with >= 1 valid "
                               "and >= 1 invalid candidate; seed mean (per-seed values in gradient_diagnostics.jsonl).  T-based fields of softmax models are not identified (b unconstrained).", ""]
    support, nested_out, manifest_out = {}, {}, {}
    for di, ds in enumerate(a.datasets):
        meta, G = dataset(ds, a.role, a.smoke); n_img = len(meta); idx = boot_matrix(n_img, B, di)
        zs = load_scores(ds, a.role, "zeroshot", "zeroshot", 0, meta, inputs["missing"], a.smoke)
        n_cov = [int(sum((g["iou"][j] >= 0.5).any() for j in range(len(mm["target"])))) if len(mm["prop_boxes"]) else 0 for mm, g in zip(meta, G)]
        items = nested_manifest(meta, G) if "C" in a.parts else []
        per_model_img = {}                                                                        # key -> part -> field -> [seed][image]
        for m in MODELS:
            for s in SEEDS:
                cps = load_scores(ds, a.role, m[0], m[1], s, meta, inputs["missing"], a.smoke); ab = critic(ds, m, s, inputs["missing"])
                if cps is None or ab is None:
                    continue
                inputs["used"].append(f"{ds}/{a.role}/{key(m)}/s{s}"); d = per_model_img.setdefault(key(m), {"A": {}, "B": {}, "C": {}})
                if "A" in a.parts:
                    rows, cnt = [], {"no_valid": 0, "no_invalid": 0}
                    for i, (mm, g, cp) in enumerate(zip(meta, G, cps)):
                        r, c = part_a_image(mm, g, cp, None if zs is None else zs[i], m, ab, n_cov[i]); rows.append(r)
                        cnt = {k2: cnt[k2] + c[k2] for k2 in cnt}
                    fields = sorted({f2 for r in rows for q in r for f2 in q})
                    for f2 in fields:
                        d["A"].setdefault(f2, {})[s] = image_means(rows, f2)
                    gl.append({"dataset": ds, "model": key(m), "label": m[4], "seed": s, "critic_a_b": ab, "query_counts": {"eligible": int(sum(len(r) for r in rows)), **cnt},
                               **{f2: macro_mean(image_means(rows, f2)) for f2 in fields}})
                if "B" in a.parts:
                    rows = [part_b_image(mm, g, cp, m, ab) for mm, g, cp in zip(meta, G, cps)]
                    kc = [r["kl"] for rr in rows for r in rr if r.get("k") == 1 and "kl" in r]
                    if kc and max(abs(x) for x in kc) > 1e-9:
                        raise AssertionError(f"KL at k = 1 is not 0 ({max(abs(x) for x in kc)})")
                    for kind, bins in (("k", ("0",) + tuple(n for n, _, _ in K_BINS)), ("m", ("0",) + tuple(n for n, _, _ in M_BINS))):
                        for b in bins:
                            for f2 in ("acc", "kl", "h", "max_rhoA", "ent_rhoA", "valid_iou", "top_valid_iou"):
                                d["B"].setdefault(f"{kind}={b}:{f2}", {})[s] = per_image_bin(rows, kind, b, f2)
                    for kb, _, _ in K_BINS:                                                    # k x m cells for the auxiliary standardisation
                        for mb, _, _ in M_BINS:
                            d["B"].setdefault(f"km={kb}|{mb}:acc", {})[s] = np.array([np.mean(v) if (v := [r["acc"] for r in rr if bin_of(r, "k") == kb and bin_of(r, "m") == mb]) else np.nan for rr in rows])
                    if ds not in support:
                        flat = [r for rr in rows for r in rr]; cross = {}
                        for r in flat:
                            cross.setdefault(bin_of(r, "k"), {}).setdefault(bin_of(r, "m"), 0); cross[bin_of(r, "k")][bin_of(r, "m")] += 1
                        cats = {}
                        for r in flat:
                            cats.setdefault(bin_of(r, "k"), {}).setdefault(r["cat"], 0); cats[bin_of(r, "k")][r["cat"]] += 1
                        support[ds] = {"queries": len(flat), "images": n_img, "k_by_m_queries": cross,
                                       "images_per_k_bin": {b: int(sum(any(bin_of(r, "k") == b for r in rr) for rr in rows)) for b in cross},
                                       "top_categories_per_k_bin": {b: sorted(c.items(), key=lambda kv: -kv[1])[:10] for b, c in cats.items()}}
                if "C" in a.parts and items:
                    det_all = [np.asarray(mm["det_score"], float) for mm in meta]
                    dep = [model_arrays(cp, det, m[2], m[3], ab)[2] if cp.shape[0] else cp for cp, det in zip(cps, det_all)]
                    hits = nested_hits(items, dep)
                    for c in [str(x) for x in NESTED] + ["all"]:
                        d["C"].setdefault(c, {})[s] = per_image_from_queries(items, hits, n_img, c)
        # seed-mean summaries, bootstrap, contrasts
        for k_, d in per_model_img.items():
            for part in ("A", "B", "C"):
                for f2, seeds in d[part].items():
                    mean_img = np.nanmean(np.stack(list(seeds.values())), 0)
                    rec = {"dataset": ds, "model": k_, "part": part, "field": f2, "seeds": sorted(seeds), "per_seed": {str(sd): macro_mean(x) for sd, x in sorted(seeds.items())},
                           **boot_stats(mean_img, idx)}
                    (ml if part == "B" else gl if part == "A" else nested_out.setdefault(ds, [])).append(rec)
        for x, y in CONTRASTS:
            if x in per_model_img and y in per_model_img:
                for part, fields in (("A", ("shared_top1_q_frac", "shared_top1_grad_frac", "self_top1_q_frac", "self_top1_grad_frac", "shared_top5_grad_frac")),
                                     ("B", tuple(f"k={b}:acc" for b in ("1", "2-3", "4+")) + tuple(f"m={b}:acc" for b in ("1-15", "16-31", "32+"))),
                                     ("C", tuple(str(c) for c in NESTED) + ("all",))):
                    for f2 in fields:
                        sx, sy = per_model_img[x][part].get(f2), per_model_img[y][part].get(f2); common = sorted(set(sx or {}) & set(sy or {}))
                        if not common:
                            continue
                        diff = np.nanmean(np.stack([sx[sd] - sy[sd] for sd in common]), 0)                  # paired by seed id
                        rec = {"dataset": ds, "contrast": f"{x} - {y}", "part": part, "field": f2, "seeds_paired": common, **boot_stats(diff, idx)}
                        (gl if part == "A" else ml if part == "B" else nested_out.setdefault(ds, [])).append(rec)
        for k_, d in per_model_img.items():                                                     # between-bin difference with the same resamples
            for kind, lo_b, hi_b in (("k", "1", "4+"), ("m", "1-15", "32+")):
                sx, sy = d["B"].get(f"{kind}={hi_b}:acc"), d["B"].get(f"{kind}={lo_b}:acc")
                if sx and sy:
                    X, Y = np.nanmean(np.stack(list(sx.values())), 0), np.nanmean(np.stack(list(sy.values())), 0); okx, oky = np.isfinite(X), np.isfinite(Y)
                    bx = (np.where(okx, X, 0)[idx].sum(1) / np.maximum(okx[idx].sum(1), 1)) - (np.where(oky, Y, 0)[idx].sum(1) / np.maximum(oky[idx].sum(1), 1))
                    ml.append({"dataset": ds, "model": k_, "part": "B", "field": f"{kind}:{hi_b}-minus-{lo_b}:acc", "mean": float(np.nanmean(X) - np.nanmean(Y)),
                               "ci95": [float(np.quantile(bx, 0.025)), float(np.quantile(bx, 0.975))]})
        if "B" in a.parts and per_model_img:                                                     # auxiliary: k-bin accuracy standardised to the covered m distribution
            cnt = support.get(ds, {}).get("k_by_m_queries", {}); mbs = [n for n, _, _ in M_BINS]
            w = np.array([sum(cnt.get(kb, {}).get(mb, 0) for kb, _, _ in K_BINS) for mb in mbs], float); w = w / w.sum() if w.sum() else w
            for k_, d in per_model_img.items():
                for kb, _, _ in K_BINS:
                    cells = [d["B"].get(f"km={kb}|{mb}:acc") for mb in mbs]
                    if any(c is None for c in cells) or any(cnt.get(kb, {}).get(mb, 0) == 0 for mb in mbs):
                        ml.append({"dataset": ds, "model": k_, "part": "B", "field": f"k={kb}:acc_m_standardised", "mean": None, "note": "NA: a k x m cell lacks support"}); continue
                    cm = [macro_mean(np.nanmean(np.stack(list(c.values())), 0)) for c in cells]
                    ml.append({"dataset": ds, "model": k_, "part": "B", "field": f"k={kb}:acc_m_standardised", "weights_m": dict(zip(mbs, w.tolist())), "cell_acc": dict(zip(mbs, cm)),
                               "mean": None if any(x is None for x in cm) else float(np.dot(w, cm)), "note": "auxiliary, image-macro within each k x m cell; association only"})
        if "C" in a.parts:
            cats = {}
            for it in items:
                cats[it["cat"]] = cats.get(it["cat"], 0) + 1
            covq = sum(n_cov); manifest_out[ds] = {"eligible_queries": len(items), "eligible_images": len({it["image"] for it in items}), "covered_queries": int(covq),
                                                   "fraction_of_covered_queries": frac(len(items), covq), "n_invalid_quantiles": np.quantile([it["n_invalid"] for it in items], [0, .25, .5, .75, 1]).tolist() if items else None,
                                                   "top_categories": sorted(cats.items(), key=lambda kv: -kv[1])[:15],
                                                   "queries": [{"query_id": it["query_id"], "target": it["target"], "n_invalid": it["n_invalid"]} for it in items]}
        # markdown summary for A
        rows_md = [r for r in gl if r.get("dataset") == ds and r.get("part") == "A" and "model" in r and r["field"] in
                   ("shared_top1_q_frac", "shared_top1_grad_frac", "self_top1_q_frac", "self_top1_grad_frac", "shared_top5_grad_frac", "wrong_sat_frac", "resid_invalid")]
        if rows_md:
            summ_md += [f"## {ds}", "", "| model | shared top-1 q share | shared top-1 |g| share | self top-1 q share | self top-1 |g| share | shared top-5 |g| share | invalid T ≥ 0.95 | invalid T + 1 |",
                        "|---|---|---|---|---|---|---|---|"]
            for m in MODELS:
                v = {r["field"]: r for r in rows_md if r["model"] == key(m)}
                if v:
                    cell = lambda f2: "—" if f2 not in v or v[f2]["mean"] is None else f"{v[f2]['mean']:.3f}"
                    summ_md.append(f"| {m[4]} | " + " | ".join(cell(f2) for f2 in ("shared_top1_q_frac", "shared_top1_grad_frac", "self_top1_q_frac", "self_top1_grad_frac",
                                                                                   "shared_top5_grad_frac", "wrong_sat_frac", "resid_invalid")) + " |")
            summ_md.append("")
    with open(out / "gradient_diagnostics.jsonl", "w") as fh:
        for r in gl:
            fh.write(json.dumps(r) + "\n")
    with open(out / "multiplicity_diagnostics.jsonl", "w") as fh:
        for r in ml:
            fh.write(json.dumps(r) + "\n")
    json.dump(support, open(out / "support_counts.json", "w"), indent=1); json.dump(manifest_out, open(out / "nested_query_manifest.json", "w"))
    json.dump(nested_out, open(out / "nested_curves.json", "w"), indent=1); json.dump(inputs, open(out / "r5_diagnostics_inputs.json", "w"), indent=1)
    if inputs["missing"]:
        summ_md += ["Missing inputs (not filled):", ""] + [f"- {x}" for x in sorted(set(inputs["missing"]))]
    (out / "gradient_summary.md").write_text("\n".join(summ_md) + "\n"); print("\n".join(summ_md[:40]))
    print(f"[r5_diagnostics] used {len(inputs['used'])} model-seeds; missing {len(set(inputs['missing']))} inputs; wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
