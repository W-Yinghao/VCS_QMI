"""Signed pooled correlations and prompt-level candidate selection metrics."""
from collections import defaultdict
import numpy as np


def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    rank = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank[order[i:j]] = (i + j - 1) / 2.0 + 1
        i = j
    return rank


def correlation(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    x, y = x - x.mean(), y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / den) if den > 0 else None


def metrics(rows, tie_epsilon_mos=.05, prediction_tie_epsilon=1e-12):
    if not rows:
        raise ValueError("No predictions to evaluate")
    datasets = {r.get("dataset", "") for r in rows}
    if len(datasets) != 1:
        raise ValueError("Evaluate one dataset at a time; do not pool different MOS populations")
    y = np.asarray([float(r["mos"]) for r in rows])
    p = np.asarray([float(r["prediction"]) for r in rows])
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite label/prediction")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["prompt_group"]].append(i)
    acc, within_srcc, regrets, selected, oracle = [], [], [], [], []
    successes = total_pairs = 0
    for idx in groups.values():
        if len(idx) < 2:
            continue
        yy, pp = y[idx], p[idx]
        pair_values = []
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if abs(yy[i] - yy[j]) <= tie_epsilon_mos:
                    continue
                pd = pp[i] - pp[j]
                score = .5 if abs(pd) <= prediction_tie_epsilon else float(pd * (yy[i] - yy[j]) > 0)
                pair_values.append(score)
        if pair_values:
            acc.append(float(np.mean(pair_values)))
            successes += sum(pair_values)
            total_pairs += len(pair_values)
        sr = correlation(rankdata(yy), rankdata(pp))
        if sr is not None:
            within_srcc.append(sr)
        top = pp >= pp.max() - prediction_tie_epsilon
        # Expected MOS under a uniform tie-break; no annotation-dependent choice.
        picked = float(yy[top].mean())
        selected.append(picked / 5.)
        oracle.append(float(yy.max()) / 5.)
        regrets.append((float(yy.max()) - picked) / 5.)
    return {
        "n_images": len(rows), "n_prompt_groups": len(groups),
        "srcc": correlation(rankdata(y), rankdata(p)), "plcc": correlation(y, p),
        "rmse_normalized": float(np.sqrt(np.mean((p - y / 5.) ** 2))),
        "within_prompt_pairwise_macro": float(np.mean(acc)) if acc else None,
        "within_prompt_pairwise_micro": successes / total_pairs if total_pairs else None,
        "eligible_pair_groups": len(acc), "eligible_pairs": total_pairs,
        "within_prompt_srcc_macro": float(np.mean(within_srcc)) if within_srcc else None,
        "eligible_within_srcc_groups": len(within_srcc),
        "normalized_regret": float(np.mean(regrets)) if regrets else None,
        "selected_mos_normalized": float(np.mean(selected)) if selected else None,
        "oracle_mos_normalized": float(np.mean(oracle)) if oracle else None,
        "selection_groups": len(regrets), "label_tie_epsilon_mos": tie_epsilon_mos,
        "note": "Raw signed correlations; no abs(), TEST-fitted mapping, or per-prompt z-score. Prediction scale is MOS/5."
    }


def paired_bootstrap(a, b, repetitions=1000, seed=20261019, tie_epsilon_mos=.05):
    """Pair both methods, resample prompt groups, retain all candidates in group."""
    aa = {r["sample_id"]: r for r in a}
    bb = {r["sample_id"]: r for r in b}
    if aa.keys() != bb.keys() or len(aa) != len(a) or len(bb) != len(b):
        raise ValueError("Paired comparison requires identical unique sample id sets")
    for sid in aa:
        if aa[sid]["prompt_group"] != bb[sid]["prompt_group"] or float(aa[sid]["mos"]) != float(bb[sid]["mos"]):
            raise ValueError("Methods disagree on label / group")
    group_ids = sorted({r["prompt_group"] for r in a})
    by_group = {g: [sid for sid, r in aa.items() if r["prompt_group"] == g] for g in group_ids}
    keys = ["srcc", "plcc", "within_prompt_pairwise_macro", "normalized_regret", "selected_mos_normalized"]
    samples = {k: [] for k in keys}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        ar, br = [], []
        for draw, gi in enumerate(rng.integers(0, len(group_ids), len(group_ids))):
            for sid in by_group[group_ids[gi]]:
                ar.append({**aa[sid], "prompt_group": str(draw)})
                br.append({**bb[sid], "prompt_group": str(draw)})
        ma, mb = metrics(ar, tie_epsilon_mos), metrics(br, tie_epsilon_mos)
        for key in keys:
            if ma[key] is not None and mb[key] is not None:
                samples[key].append(ma[key] - mb[key])
    ma, mb = metrics(a, tie_epsilon_mos), metrics(b, tie_epsilon_mos)
    output = {}
    for key in keys:
        vals = samples[key]
        output[key] = {"a_minus_b": None if ma[key] is None or mb[key] is None else ma[key] - mb[key],
                       "ci95": [float(v) for v in np.quantile(vals, [.025, .975])] if vals else None,
                       "valid_resamples": len(vals), "higher_is_better": key != "normalized_regret"}
    return {"scope": "paired prompt bootstrap for one seed/dataset, not independent training seed uncertainty",
            "repetitions": repetitions, "seed": seed, "differences": output}
