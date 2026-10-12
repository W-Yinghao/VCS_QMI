"""r7 R7-30 D1-30 secondary (frozen D1_30.secondary_endpoints 'selector risk calibration on DEV masks'): every frozen D1-20 unit's predicted
risk r(K) for the D1-10 DEV binary questions × their 12 D1-10 masks (3 mechanisms × 4 budgets), against the realised teacher
h = (C − a)² − (C − b)² from the sealed D1-10 DEV records — read only here, after the D1-30 freeze, through the D1-10 analyser's coverage
guard.  Per unit: Spearman(r, h), least-squares slope / intercept of h on r, per-budget and per-mechanism mean r vs mean h; image-grouped
bootstrap CI of the Spearman (2,000 reps).  GPU (host vision + selector forwards; no generation).
    python scripts/r7_d1_30_devrisk.py --out O --protocol P --units U ...                 (after the freeze; DEV)
    python scripts/r7_d1_30_devrisk.py --out O --records R --units U ... --selectors-dir D --pca P2 --images-from-fixture F   (smoke only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_30_data as DD  # noqa: E402
import r7_d1_30_run as RUN  # noqa: E402
import r7_d1_masks as MK  # noqa: E402
import r7_vl_common as V  # noqa: E402


def T(pair):
    y, n = pair
    return 2.0 * y / max(y + n, 1e-12) - 1.0


def masks_and_h(r, unb64, digest):
    orders = {"random": MK.order_random(r["qid"])[0], "grid": MK.order_grid(r["qid"])[0], "qsim": unb64(r["qsim_order"])}
    for mech in ("random", "grid"):
        if digest(orders[mech]) != r["order_digest"][mech]:
            raise RuntimeError(f"ordering digest mismatch {r['qid']} / {mech}")
    tb = T(r["full"]); out = []
    for mech in MK.MECHANISMS:
        for k in MK.BUDGETS:
            ta = T(r["masked"][mech][str(k)]); out.append((mech, k, MK.kept(orders[mech], k), (r["C"] - ta) ** 2 - (r["C"] - tb) ** 2))
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a, kind="stable"), kind="stable"), np.argsort(np.argsort(b, kind="stable"), kind="stable")
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else None


def summarize(r, h, k, mech, groups, reps=2000):
    r, h, k, mech, groups = (np.asarray(x) for x in (r, h, k, mech, groups)); r, h = r.astype(float), h.astype(float)
    A = np.vstack([r, np.ones_like(r)]).T; slope, icpt = np.linalg.lstsq(A, h, rcond=None)[0]
    ug, inv = np.unique(groups, return_inverse=True); mem = [np.flatnonzero(inv == j) for j in range(len(ug))]; rng = np.random.default_rng(0); bs = []
    for _ in range(reps):
        idx = np.concatenate([mem[j] for j in rng.integers(0, len(ug), len(ug))]); v = spearman(r[idx], h[idx])
        if v is not None:
            bs.append(v)
    return {"n_pairs": int(len(r)), "n_images": int(len(ug)), "spearman_r_h": spearman(r, h), "spearman_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None,
            "slope_h_on_r": float(slope), "intercept": float(icpt),
            "by_budget": {str(b): {"mean_r": float(r[k == b].mean()), "mean_h": float(h[k == b].mean())} for b in MK.BUDGETS},
            "by_mechanism": {m: {"mean_r": float(r[mech == m].mean()), "mean_h": float(h[mech == m].mean()), "spearman": spearman(r[mech == m], h[mech == m])} for m in MK.MECHANISMS}}


def main(argv=None, host=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--protocol"); ap.add_argument("--units", nargs="+")
    ap.add_argument("--lr-choice"); ap.add_argument("--records"); ap.add_argument("--images-from-fixture"); ap.add_argument("--selectors-dir", default=str(RUN.D1 / "selectors"))
    ap.add_argument("--pca", default=str(RUN.D1 / "cache" / "pca.pt")); ap.add_argument("--reps", type=int, default=2000); a = ap.parse_args(argv)
    if not a.units:                                                                    # every D1-20 unit (S1-S3 × 3 seeds, A1 / A2 × 3 seeds)
        main_u, abl_u = RUN.unit_names(json.loads(Path(a.lr_choice).read_text())); a.units = [u for v in list(main_u.values()) + list(abl_u.values()) for u in v]
    import torch
    import r7_d1_teacher as TE
    if a.records:                                                                      # smoke: probe records on the R7-20 fixture
        rows = V.read_jsonl(Path(a.records)); cov = {"smoke_records": len(rows)}
        files = {json.loads(x)["image_id"]: json.loads(x)["image_file"] for x in Path(a.images_from_fixture).read_text().splitlines() if x.strip()}
        from PIL import Image
        open_img = lambda i: Image.open(files[i]).convert("RGB")  # noqa: E731
        load_questions(Path(a.images_from_fixture))
    else:
        DD.require_freeze(Path(a.protocol))
        import r7_d1_analyze as AN
        rows, cov = AN.load_role(RUN.D1 / "teacher", RUN.D1 / "data", "DEV"); src = TE.ImageSource(); open_img = src.open
        load_questions(RUN.D1 / "data" / "d1_manifest.jsonl")
    host = host or TE.LlavaPrefill(Path(a.out)); bank = RUN.SelectorBank(Path(a.selectors_dir), Path(a.pca), a.units)
    by_img = {}
    for r in rows:
        by_img.setdefault(r["image_id"], []).append(r)
    res = {u: {"r": [], "h": [], "k": [], "m": [], "g": []} for u in a.units}
    for i in sorted(by_img):
        f = host.encode([open_img(i)])[0]
        for r in by_img[i]:
            mh = masks_and_h(r, TE.unb64, TE.digest); st = bank.prepare(f[None], RUN.question_embeddings(host, [question_of(r)])); M = len(mh)
            keep = torch.zeros(M, RUN.N_VIS, dtype=torch.bool, device="cuda")
            for j, (_, _, kk, _) in enumerate(mh):
                keep[j, torch.as_tensor(kk, device="cuda")] = True
            for u in a.units:
                m = bank.models[u]
                with torch.no_grad():
                    _, pb = m.encode(st["q"], st["pad"], st["v"], torch.zeros(1, RUN.N_VIS, device="cuda"))
                    pa = m.encode(st["q"].expand(M, -1, -1), st["pad"].expand(M, -1), st["v"].expand(M, -1, -1), bank.SEL.hard_bias(keep))[1]
                    rr = m.readout(pb.expand(M, -1), pa).float().cpu().numpy()
                d = res[u]; d["r"] += rr.tolist(); d["h"] += [x[3] for x in mh]; d["k"] += [x[1] for x in mh]; d["m"] += [x[0] for x in mh]; d["g"] += [i] * M
    out = {"coverage": cov, "units": {u: summarize(v["r"], v["h"], v["k"], v["m"], v["g"], a.reps) for u, v in res.items()}, "env": V.env_info()}
    Path(a.out).mkdir(parents=True, exist_ok=True); name = "d1_30_devrisk_smoke.json" if a.records else "d1_30_devrisk.json"
    (Path(a.out) / name).write_text(V.jdump(out, indent=1)); print(V.jdump(out, indent=1)[:3000])
    return 0


QUESTIONS = {}


def load_questions(path: Path):
    """The raw question text per qid (D1-10 records carry the qid only): the frozen manifest at D1-30, the R7-20 fixture in the smoke."""
    for x in Path(path).read_text().splitlines():
        if x.strip():
            j = json.loads(x); QUESTIONS[j["qid"]] = j["question"]


def question_of(r):
    return QUESTIONS[r["qid"]]


if __name__ == "__main__":
    sys.exit(main())
