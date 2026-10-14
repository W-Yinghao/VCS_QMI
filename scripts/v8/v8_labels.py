"""V8 (utility-aligned CS) — raw overall-vote histograms and the four threshold events (package docs/03 §1, docs/06 §1).

    python -I scripts/v8/v8_labels.py audit --data /home/infres/yinwang/CS_QMI/outputs/V7/data --out reports/V8_UTILITY/V8_01_label_audit

Per image with n valid votes H in {1..5}: hist_h = count(H = h) / n, tail_k = sum_{h >= k} hist_h (k = 2..5), so 1 + sum_k tail_k = vote mean exactly.
Natural sampling D: uniform prompt, uniform valid image of the prompt, uniform valid rater of the image (3- and 6-vote images weigh the same).
pi_k = E_D[tail_k] on FIT only.  overall_valid = at least one valid vote (V7 manifest `overall_votes`); the element filters (bad decomposition,
meaningless) do not remove an image whose overall label is valid (docs/06 §2).  Reads the V7 data manifest only (FIT / CAL / DEV roles); no
GenAI or EvalMuse-test label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROLES = ("FIT", "CAL_SELECT", "CAL_PROB", "DEV")


def histogram(votes) -> np.ndarray:
    v = np.asarray(votes)
    if v.ndim != 1 or len(v) == 0 or np.any(v != np.round(v)) or np.any((v < 1) | (v > 5)):
        raise ValueError(f"need a nonempty list of integer votes in 1..5, got {votes!r}")
    return np.bincount(v.astype(int), minlength=6)[1:].astype(float) / len(v)


def tails(h: np.ndarray) -> np.ndarray:
    return np.flip(np.cumsum(np.flip(h, -1), -1), -1)[..., 1:]


def load(data_dir) -> list[dict]:
    rows = []
    for line in open(Path(data_dir) / "manifest_images.jsonl"):
        r = json.loads(line)
        if r.get("role") in ROLES:
            rows.append(r)
    return rows


def records(rows: list[dict]) -> dict:
    """{role: [{image_id, prompt_id, votes, hist[5], tail[4], mean, n, w}]} for overall-valid images; w = prompt -> image natural weight."""
    out = {}
    for role in ROLES:
        rr = [r for r in rows if r["role"] == role and r.get("overall_votes")]
        per = Counter(r["prompt_id"] for r in rr); n_i, n_p = len(rr), len(per); recs = []
        for r in rr:
            h = histogram(r["overall_votes"]); t = tails(h)
            recs.append({"image_id": r["image_id"], "prompt_id": r["prompt_id"], "votes": list(r["overall_votes"]), "hist": h.tolist(), "tail": t.tolist(),
                         "mean": float(np.mean(r["overall_votes"])), "n": len(r["overall_votes"]), "w": n_i / (n_p * per[r["prompt_id"]])})
        out[role] = recs
    return out


def fit_priors(recs: list[dict]) -> np.ndarray:
    w = np.array([r["w"] for r in recs]); T = np.array([r["tail"] for r in recs]); return (w[:, None] * T).sum(0) / w.sum()


def cmd_audit(a):
    rows = load(a.data); R = records(rows); res = {"source": {"manifest": str(Path(a.data) / "manifest_images.jsonl"),
           "sha256": hashlib.sha256((Path(a.data) / "manifest_images.jsonl").read_bytes()).hexdigest()}, "roles": {}}
    allv = Counter(len(r.get("overall_votes") or []) for r in rows); res["vote_count_all_roles"] = dict(sorted(allv.items()))
    for role, recs in R.items():
        H = np.array([r["hist"] for r in recs]); T = np.array([r["tail"] for r in recs]); M = np.array([r["mean"] for r in recs])
        chk = {"hist_sums_to_1": bool(np.allclose(H.sum(1), 1)), "tails_nonincreasing": bool(np.all(np.diff(T, axis=1) <= 1e-12)),
               "one_plus_tail_sum_equals_mean": float(np.abs(1 + T.sum(1) - M).max())}
        w = np.array([r["w"] for r in recs]); nat_hist = (w[:, None] * H).sum(0) / w.sum()
        by_prompt = defaultdict(list)
        for r in recs:
            by_prompt[r["prompt_id"]].append(r["mean"])
        res["roles"][role] = {"n_images": len(recs), "n_prompts": len(by_prompt), "votes_per_image": dict(sorted(Counter(r["n"] for r in recs).items())),
                              "natural_hist": nat_hist.round(5).tolist(), "natural_tail": fit_priors(recs).round(5).tolist(), "checks": chk,
                              "images_per_prompt": dict(sorted(Counter(len(v) for v in by_prompt.values()).items())),
                              "excluded_no_valid_vote": sum(1 for r in rows if r["role"] == role and not r.get("overall_votes"))}
    pi = fit_priors(R["FIT"]); res["pi_FIT"] = pi.tolist(); res["degenerate_thresholds"] = [k + 2 for k, p in enumerate(pi) if p <= 0 or p >= 1]
    res["fit_positive_mass_per_threshold"] = {f"H>={k + 2}": {"pi": round(float(p), 5), "images_with_any_positive_vote": int(sum(1 for r in R["FIT"] if r["tail"][k] > 0))}
                                              for k, p in enumerate(pi)}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "source"}, indent=1)[:3000]); return 0


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("audit"); s.add_argument("--data", required=True); s.add_argument("--out", required=True)
    a = ap.parse_args(); return {"audit": cmd_audit}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
