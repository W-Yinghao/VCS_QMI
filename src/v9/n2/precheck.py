"""V9-N2 precheck (pre-registered gate before any formal unit; owner 2026-10-15).  One table per (dataset, rho, seed) shared stage:

1. teacher score distributions on A-critic-select, clean (P) vs re-paired (Q): quantiles, AUROC, overlap; the formal block is REFUSED when the
   problem is degenerate — AUROC > AUROC_MAX (nothing to weight) or < AUROC_MIN (teacher uninformative); thresholds live in the protocol;
2. every weight method's final weight distribution on U (quantiles, ESS, total mass, shares at 0 / 1) and the pairwise differences between methods
   (mean |w_i − w_j|, Spearman rank correlation, share of pairs with identical weights to 1e-6);
3. the per-sample loss coefficients after the fixed-pool normaliser (v_i / mean_v_full: anchors v = 1, U weights) — two methods are merged as
   "implementation-identical" ONLY if their coefficient vectors are equal to 1e-12 on every pool row AND their data order and RNG state are provably
   equal (same seed → the trainer derives both from the seed alone, which the manifest records), otherwise both run;
4. the anchor roles used by teacher / critic-fit / select / reference (counts, hashes) and their isolation from val / test (ids disjoint when id files
   exist; otherwise "separate official split files" with their hashes).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .roles import ANCHOR_ROLES
from .weights import REF

AUROC_MAX, AUROC_MIN = 0.999, 0.55          # protocol thresholds (V9_N2_PROTOCOL_DRAFT.yaml: precheck_gate)
IDENTICAL_TOL, COEF_TOL = 1e-6, 1e-12


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC with tie-averaging (P scores vs Q scores)."""
    x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(x, kind="mergesort"); ranks = np.empty(len(x)); ranks[order] = np.arange(1, len(x) + 1)
    xs = x[order]; i = 0
    while i < len(xs):                                           # tie averaging
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def overlap_coefficient(pos: np.ndarray, neg: np.ndarray, bins: int = 50) -> float:
    lo, hi = float(min(pos.min(), neg.min())), float(max(pos.max(), neg.max())); edges = np.linspace(lo, hi, bins + 1)
    hp, _ = np.histogram(pos, edges); hq, _ = np.histogram(neg, edges)
    return float(np.minimum(hp / max(len(pos), 1), hq / max(len(neg), 1)).sum())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a, kind="mergesort")); rb = np.argsort(np.argsort(b, kind="mergesort"))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _q(x: np.ndarray) -> dict:
    return {f"q{int(p)}": float(np.quantile(x, p / 100)) for p in (5, 25, 50, 75, 95)} | {"mean": float(np.mean(x)), "min": float(np.min(x)), "max": float(np.max(x))}


def teacher_separability(s_P_sel: np.ndarray, s_Q_sel: np.ndarray) -> dict:
    a = auroc(np.asarray(s_P_sel, float), np.asarray(s_Q_sel, float))
    out = {"n_P": int(len(s_P_sel)), "n_Q": int(len(s_Q_sel)), "P": _q(np.asarray(s_P_sel, float)), "Q": _q(np.asarray(s_Q_sel, float)), "auroc": a,
           "overlap_coefficient": overlap_coefficient(np.asarray(s_P_sel, float), np.asarray(s_Q_sel, float)),
           "share_P_below_Q_median": float(np.mean(np.asarray(s_P_sel) < np.median(s_Q_sel))), "thresholds": {"auroc_max": AUROC_MAX, "auroc_min": AUROC_MIN}}
    out["degenerate"] = bool(a > AUROC_MAX or a < AUROC_MIN); out["reason"] = ("auroc above max: pairs already separable, nothing to weight" if a > AUROC_MAX else "auroc below min: teacher uninformative" if a < AUROC_MIN else "")
    return out


def weight_tables(weights_by_method: dict[str, np.ndarray]) -> dict:
    names = sorted(weights_by_method); per = {}
    for m in names:
        w = np.asarray(weights_by_method[m], float)
        per[m] = {**_q(w), "ess": REF.effective_sample_size(w) if w.sum() > 0 else 0.0, "total_mass": float(w.sum()), "share_at_0": float(np.mean(w <= 1e-12)), "share_at_1": float(np.mean(w >= 1 - 1e-12)), "n": int(len(w))}
    pair = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            wa, wb = np.asarray(weights_by_method[a], float), np.asarray(weights_by_method[b], float)
            pair[f"{a}|{b}"] = {"mean_abs_diff": float(np.mean(np.abs(wa - wb))), "spearman": spearman(wa, wb), "share_identical_1e-6": float(np.mean(np.abs(wa - wb) <= IDENTICAL_TOL)), "max_abs_diff": float(np.max(np.abs(wa - wb)))}
    return {"per_method": per, "pairwise": pair}


def loss_coefficients(pool_rows: list[dict], weights_by_method: dict[str, dict], seed: int) -> dict:
    """Coefficient vector c_i = v_i / mean_v_full per method over the pool (anchors v = 1).  Merge rule: identical to COEF_TOL on every row AND the data
    order / RNG state derive from the same seed (they do: the trainer seeds everything from `seed` and draws each epoch's permutation from
    (seed, 9020, epoch) — recorded here as the provable condition)."""
    names = sorted(weights_by_method); coef = {}
    for m in names:
        v = np.array([1.0 if r["role"] != "U" else float(weights_by_method[m][r["pair_id"]]) for r in pool_rows]); coef[m] = v / float(v.mean())
    groups, merged = [], {}
    for m in names:
        for g in groups:
            if np.max(np.abs(coef[m] - coef[g[0]])) <= COEF_TOL:
                g.append(m); break
        else:
            groups.append([m])
    for g in groups:
        for m in g[1:]:
            merged[m] = {"implementation_identical_to": g[0], "condition": "coefficients equal to 1e-12 on every pool row; data order and RNG state derive from the same seed (trainer: seed_everything(seed); epoch permutation rng([seed, 9020, epoch]))", "seed": seed}
    summary = {m: {"n_pool": int(len(coef[m])), "mean": float(coef[m].mean()), "min": float(coef[m].min()), "max": float(coef[m].max()), "sha256": hashlib.sha256(np.round(coef[m], 12).tobytes()).hexdigest()[:16]} for m in names}
    return {"coefficients": summary, "groups": groups, "merged": merged, "rule": "merge only if identical coefficients AND same seed-derived order/RNG; otherwise both methods run"}


def role_isolation(roles: dict, split_files: dict | None) -> dict:
    """roles: roles.json; split_files: {'train_ids': path, 'dev_ids': path, 'test_ids': path, 'dev_caps': path, 'test_caps': path} when available."""
    out = {"counts": roles["counts"], "n_U": roles["n_U"], "hashes": roles["hashes"], "anchor_fraction": roles["anchor_fraction"], "split": roles["split"], "usage": {
        "teacher_warmup": "teacher training only", "critic_fit": "critic fitting (P and within-role Q)", "critic_select": "lr / start selection of the critic + the precheck separability table",
        "reference_moments": "a = E_P T, b = E_Q T", "U": "rho point estimate (80 % of U images) + independent interval batch (20 %); student training with weights; anchors join the student with v = 1"}}
    if split_files:
        ids = {}
        for k in ("train_ids", "dev_ids", "test_ids"):
            p = split_files.get(k)
            if p and Path(p).exists():
                ids[k] = [l.strip() for l in Path(p).read_text().splitlines() if l.strip()]
        role_imgs = set(i for r in ANCHOR_ROLES for i in roles["A"][r]) | set(roles["U"])
        if "train_ids" in ids:
            train_set = set(ids["train_ids"]); mapped = set(ids["train_ids"][int(i)] if str(i).isdigit() and int(i) < len(ids["train_ids"]) else str(i) for i in role_imgs)
            out["train_ids"] = {"n": len(train_set), "roles_within_train": bool(mapped <= train_set)}
            for k in ("dev_ids", "test_ids"):
                if k in ids:
                    out[k] = {"n": len(ids[k]), "overlap_with_roles": len(mapped & set(ids[k])), "disjoint": len(mapped & set(ids[k])) == 0}
        for k in ("dev_caps", "test_caps"):
            p = split_files.get(k)
            if p and Path(p).exists():
                out[f"{k}_sha256"] = hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
        out["note"] = "dev / test are separate official split files; role images are indices into the train split only"
    return out


def write_precheck(out_dir: str | Path, dataset: str, rho: float, seed: int, sep: dict, wt: dict, coef: dict, iso: dict, noise_manifest: dict) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rec = {"dataset": dataset, "rho": rho, "seed": seed, "teacher_separability": sep, "weights": wt, "loss_coefficients": coef, "roles": iso, "noise_manifest": noise_manifest,
           "formal_block_allowed": not sep["degenerate"], "refusal_reason": sep["reason"]}
    (out / f"precheck_{dataset}_rho{rho:g}_s{seed}.json").write_text(json.dumps(rec, indent=1, default=str))
    L = [f"# V9-N2 precheck — {dataset}, rho {rho:g}, seed {seed} — formal block {'ALLOWED' if rec['formal_block_allowed'] else 'REFUSED: ' + sep['reason']}", "",
         f"## 1. Teacher scores on A-critic-select (P clean n={sep['n_P']}, Q re-paired n={sep['n_Q']}): AUROC {sep['auroc']:.4f} (thresholds > {AUROC_MAX} / < {AUROC_MIN} refuse), overlap {sep['overlap_coefficient']:.3f}, share of P below Q median {sep['share_P_below_Q_median']:.3f}", "",
         "| set | q5 | q25 | q50 | q75 | q95 | mean |", "|---|---|---|---|---|---|---|"]
    for k in ("P", "Q"):
        q = sep[k]; L.append(f"| {k} | {q['q5']:.4f} | {q['q25']:.4f} | {q['q50']:.4f} | {q['q75']:.4f} | {q['q95']:.4f} | {q['mean']:.4f} |")
    L += ["", "## 2. Weights on U per method", "", "| method | n | q5 | q50 | q95 | mean | ESS | total mass | share 0 | share 1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, q in wt["per_method"].items():
        L.append(f"| {m} | {q['n']} | {q['q5']:.3f} | {q['q50']:.3f} | {q['q95']:.3f} | {q['mean']:.3f} | {q['ess']:.1f} | {q['total_mass']:.1f} | {q['share_at_0']:.3f} | {q['share_at_1']:.3f} |")
    L += ["", "| pair | mean abs diff | Spearman | share identical (1e-6) | max abs diff |", "|---|---|---|---|---|"]
    for k, q in wt["pairwise"].items():
        L.append(f"| {k} | {q['mean_abs_diff']:.4f} | {q['spearman']:.3f} | {q['share_identical_1e-6']:.3f} | {q['max_abs_diff']:.4f} |")
    L += ["", "## 3. Loss coefficients after the fixed-pool normaliser (c_i = v_i / mean_v_full over the pool)", "", "| method | n pool | mean | min | max | sha256 |", "|---|---|---|---|---|---|"]
    for m, q in coef["coefficients"].items():
        L.append(f"| {m} | {q['n_pool']} | {q['mean']:.6f} | {q['min']:.6f} | {q['max']:.6f} | {q['sha256']} |")
    L += ["", f"Identical groups: {coef['groups']}; merged (run once): {list(coef['merged']) or 'none'} — rule: {coef['rule']}", "",
          "## 4. Roles and isolation", "", f"counts {iso['counts']}, U {iso['n_U']}, anchor fraction {iso['anchor_fraction']}, split {iso['split']}", "",
          "\n".join(f"- {k}: {v}" for k, v in iso["usage"].items()), ""]
    for k in ("train_ids", "dev_ids", "test_ids", "note", "dev_caps_sha256", "test_caps_sha256"):
        if k in iso:
            L.append(f"- {k}: {iso[k]}")
    L += ["", f"Noise: design rho {noise_manifest['design_rho']}, realized {noise_manifest['realized_rho']:.4f}, mapping sha256 {noise_manifest['mapping_sha256'][:16]}, A mismatched {noise_manifest['n_A_mismatched']}"]
    (out / f"precheck_{dataset}_rho{rho:g}_s{seed}.md").write_text("\n".join(L) + "\n")
    return rec
