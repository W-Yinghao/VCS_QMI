"""V9-N1-Image (package §04.1 N1-Image, §04.2–04.6): proportion / reliability quantification on the existing COCO CLIP cache.

    python scripts/v9/n1_image.py --out <dir> [--seeds 901,902,903] [--rhos 0,0.1,0.3,0.5] [--sizes 128,512,2048] [--repeats 30] [--smoke]

Source: `outputs/P49_precheck_A/features/SRC-FIT.pt` — frozen open_clip ViT-B/32 (laion2b) image / caption features of 20 000 COCO train2017
images (5 captions each; image-disjoint roles of P49).  This cache was observed by earlier units (P49/P50/P57/P58/P71/P72/P79/P80/P159): N1-Image is
EXPLORATION and code continuity, not a new independent confirmation (package §07.1).
Common teacher score: the SGR teacher of §05 does not exist yet, so the single pairing similarity is the frozen-CLIP cosine s = cos(img, txt)
(disclosed; it is the same score for every method, §04.2).
Roles per seed (images shuffled by the seed; one caption index per image drawn per seed): anchor A = 10 % of the 20 000 images (2 000) partitioned
by image 40 / 30 / 15 / 15 → teacher-warmup 800 (UNUSED here: no teacher is trained), critic-fit 600, critic-select 300, reference-moments 300; the
reference-moments role is split into P-reference images (100), Q-anchor images (100) and Q-donor images (100) so that every reference cluster has
disjoint units.  U = the remaining 18 000 images.  P pair = (image, its drawn caption).  Q in the fit / select roles = one non-zero cyclic shift of
the captions inside the role (same-image rejection ⇒ a conditional Q, disclosed; fitting only).  Q in the reference-moments role = (Q-anchor image,
caption of a distinct Q-donor image), each donor used once.
Batches: for each design rho ∈ {0, .1, .3, .5} and size n ∈ {128, 512, 2048}, 30 random batches of n U images (fresh per repeat); round(rho n) of
them get the caption of a donor U image OUTSIDE the batch (distinct donors), the rest keep their own caption.  The injection labels go only to the
offline scorer (Brier / log loss / AUROC of the weights); realized rho = replaced fraction.  Batch clusters = (image, donor) with disjoint units,
so the finite-sample set's cluster condition can be checked on real ids; "30 repeats" reuse the same U pool (dependent repeats, §04.5).
Outputs <dir>/seed<seed>_rho<rho>_n<n>.json (skip-if-done) + <dir>/seed<seed>_panel.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v9 import quant as Q  # noqa: E402
from v9 import reference_core as RC  # noqa: E402

CACHE = Path("/home/infres/yinwang/CS_QMI/outputs/P49_precheck_A")
ANCHOR_FRACTION, PARTITION = 0.10, {"teacher_warmup": 0.40, "critic_fit": 0.30, "critic_select": 0.15, "reference_moments": 0.15}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cache():
    d = torch.load(CACHE / "features" / "SRC-FIT.pt", map_location="cpu", weights_only=False)
    img = torch.nn.functional.normalize(d["img"].float(), dim=1).numpy(); txt = torch.nn.functional.normalize(d["txt"].float(), dim=2).numpy()
    return {"img": img, "txt": txt, "ids": d["image_ids"].numpy(), "n": len(img)}


def roles_for_seed(n_images: int, seed: int) -> dict:
    rng = np.random.default_rng([seed, 9])
    perm = rng.permutation(n_images); n_a = int(round(ANCHOR_FRACTION * n_images)); A, U = perm[:n_a], perm[n_a:]
    sizes = {k: int(round(v * n_a)) for k, v in PARTITION.items()}; sizes["reference_moments"] = n_a - sum(v for k, v in sizes.items() if k != "reference_moments")
    roles, o = {}, 0
    for k in ("teacher_warmup", "critic_fit", "critic_select", "reference_moments"):
        roles[k] = A[o:o + sizes[k]]; o += sizes[k]
    rm = roles["reference_moments"]; third = len(rm) // 3
    roles["ref_p"], roles["ref_q_anchor"], roles["ref_q_donor"] = rm[:third], rm[third:2 * third], rm[2 * third:3 * third]
    roles["U"] = U; caps = rng.integers(0, 5, size=n_images)
    return {"roles": roles, "caption_index": caps, "sizes": {k: int(len(v)) for k, v in roles.items()}}


def score(C, img_idx, txt_img_idx, caps):
    return np.einsum("ij,ij->i", C["img"][img_idx], C["txt"][txt_img_idx, caps[txt_img_idx]])


def pq_fit_role(C, idx, caps, rng):
    """P pairs (i, cap_i) and Q = one non-zero cyclic shift of the captions inside the role (conditional Q, fitting only)."""
    sp = score(C, idx, idx, caps); shift = int(rng.integers(1, len(idx))); sq = score(C, idx, np.roll(idx, shift), caps); return sp, sq, shift


def reference_rows(C, R, caps, Tfun):
    P = R["ref_p"]; A, D = R["ref_q_anchor"], R["ref_q_donor"]; m = min(len(A), len(D)); A, D = A[:m], D[:m]
    sp = score(C, P, P, caps); sq = score(C, A, D, caps)
    rows_p = [{"role": "P", "cluster_id": f"p{int(i)}", "units": [f"img{int(i)}"], "score": float(t)} for i, t in zip(P, Tfun(sp))]
    rows_q = [{"role": "Q", "cluster_id": f"q{int(i)}", "units": [f"img{int(i)}", f"img{int(j)}"], "score": float(t)} for i, j, t in zip(A, D, Tfun(sq))]
    return sp, sq, rows_p, rows_q


def make_batch(C, U, caps, n, rho, rng):
    """n U images; round(rho n) get the caption of a distinct donor U image outside the batch.  Returns scores, labels, clusters."""
    pick = rng.choice(U, size=n, replace=False); k = int(round(rho * n)); rest = np.setdiff1d(U, pick); donors = rng.choice(rest, size=k, replace=False)
    txt_idx = pick.copy(); txt_idx[:k] = donors; is_p = np.ones(n, bool); is_p[:k] = False
    s = score(C, pick, txt_idx, caps)
    units = [[f"img{int(i)}"] + ([f"img{int(j)}"] if not p else []) for i, j, p in zip(pick, txt_idx, is_p)]
    return s, is_p, pick, txt_idx, units


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", default="901,902,903"); ap.add_argument("--rhos", default="0,0.1,0.3,0.5")
    ap.add_argument("--sizes", default="128,512,2048"); ap.add_argument("--repeats", type=int, default=30); ap.add_argument("--tuning-seed", type=int, default=917)
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]; rhos = [float(x) for x in a.rhos.split(",")]; sizes = [int(x) for x in a.sizes.split(",")]; reps = a.repeats
    if a.smoke:
        seeds, rhos, sizes, reps = seeds[:1], rhos[:2], [128], (a.repeats if a.repeats != 30 else 2)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); C = load_cache()
    prov = {"cache": str(CACHE / "features" / "SRC-FIT.pt"), "sha256": sha256(CACHE / "features" / "SRC-FIT.pt"), "manifest_sha256": sha256(CACHE / "manifest.json"),
            "splits_sha256": sha256(CACHE / "splits.json"), "n_images": C["n"], "observed_before": "P49/P50/P57/P58/P71/P72/P79/P80/P159 (exploration, not a new confirmation)",
            "teacher": "frozen open_clip ViT-B/32 cosine (no SGR teacher exists yet; disclosed)"}
    for seed in seeds:
        R = roles_for_seed(C["n"], seed); roles, caps = R["roles"], R["caption_index"]; rng = np.random.default_rng([seed, 1])
        t0 = time.time(); sp_fit, sq_fit, sh_fit = pq_fit_role(C, roles["critic_fit"], caps, rng); sp_sel, sq_sel, sh_sel = pq_fit_role(C, roles["critic_select"], caps, rng)
        panel = Q.fit_panel(sp_fit, sq_fit, sp_sel, sq_sel, seed=seed, adam_sensitivity=not a.smoke)
        sp_ref, sq_ref, rows_p, rows_q = reference_rows(C, roles, caps, panel.vcs.T); _, _, rows_p_b, rows_q_b = reference_rows(C, roles, caps, panel.bce.T)
        pj = {"seed": seed, "sizes": R["sizes"], "shifts": {"fit": sh_fit, "select": sh_sel}, "panel": Q.panel_to_json(panel), "fit_seconds": time.time() - t0,
              "reference_moments": {"a_vcs": float(np.mean(panel.vcs.T(sp_ref))), "b_vcs": float(np.mean(panel.vcs.T(sq_ref))), "a_raw": float(np.mean(sp_ref)), "b_raw": float(np.mean(sq_ref)),
                                    "n_p": len(rows_p), "n_q": len(rows_q)}, "cosine": {"fit_P_median": float(np.median(sp_fit)), "fit_Q_median": float(np.median(sq_fit))}, "provenance": prov,
              "ids_sha256": {k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest()[:16] for k, v in roles.items()}}
        (out / f"seed{seed}_panel.json").write_text(json.dumps(pj, indent=1)); print(f"[seed {seed}] panel fitted in {pj['fit_seconds']:.1f}s; J_select {panel.info['J_select']}", flush=True)
        for rho in rhos:
            for n in sizes:
                f = out / f"seed{seed}_rho{rho:g}_n{n}.json"
                if f.exists():
                    print(f"[skip] {f.name}"); continue
                t1 = time.time(); records = []
                for rep in range(reps):
                    brng = np.random.default_rng([seed, rep, int(rho * 1000), n]); s_b, is_p, pick, txt_idx, units = make_batch(C, roles["U"], caps, n, rho, brng)
                    truth = {"is_p": is_p, "rho": rho}; res = Q.quantify_batch(panel, sp_ref, sq_ref, s_b, truth)
                    rows_r = [{"role": "R", "cluster_id": f"r{int(i)}", "units": u, "score": float(t)} for i, u, t in zip(pick, units, panel.vcs.T(s_b))]
                    ci = Q.confidence_set(rows_p, rows_q, rows_r, fixed_score_outside_eval=True, specified_mixture=True)
                    rows_rb = [{"role": "R", "cluster_id": f"r{int(i)}", "units": u, "score": float(t)} for i, u, t in zip(pick, units, panel.bce.T(s_b))]
                    cib = Q.confidence_set(rows_p_b, rows_q_b, rows_rb, fixed_score_outside_eval=True, specified_mixture=True)
                    rec = {"rep": rep, "rho_design": rho, "rho_realized": res["_truth"]["rho_realized"], "n": n, "methods": {}, "J_batch": res["_J_batch"],
                           "interval_vcs": {k: ci[k] for k in ("lower", "upper", "status", "disjoint_clusters")}, "interval_bce": {k: cib[k] for k in ("lower", "upper", "status", "disjoint_clusters")},
                           "batch_ids_sha256": hashlib.sha256(pick.tobytes() + txt_idx.tobytes()).hexdigest()[:16]}
                    for m in Q.METHODS:
                        o = res[m]; w = o.get("weights") or {}
                        rec["methods"][m] = {**{k: o.get(k) for k in ("rho_raw", "rho_clipped", "status")}, **{k: w.get(k) for k in ("brier", "logloss", "auroc", "ess_fraction", "clip_rate_T")}}
                    records.append(rec)
                summ = summarise(records, rho); res_cell = {"seed": seed, "rho": rho, "n": n, "repeats": reps, "summary": summ, "records": records, "seconds": time.time() - t1, "smoke": a.smoke,
                                                            "note": "repeats share the U pool and the fitted panel: dependent, not 30 new datasets"}
                tmp = f.with_suffix(".partial.json"); tmp.write_text(json.dumps(res_cell, indent=1)); tmp.replace(f)
                s = summ["methods"]; print(f"[seed {seed} rho={rho:g} n={n}] {reps} batches in {time.time() - t1:.0f}s: RMSE raw {s['raw_moment']['rmse']:.3f} vcs {s['vcs_moment']['rmse']:.3f} bce {s['bce_moment']['rmse']:.3f} "
                      f"em_bcts {s['em_bcts']['rmse']:.3f} kde {s['kde_moment']['rmse']:.3f} rulsif {s['rulsif_moment']['rmse']:.3f} nystrom {s['nystrom_moment']['rmse']:.3f}; "
                      f"interval(vcs) cov {summ['interval_vcs']['coverage']:.2f} width {summ['interval_vcs']['mean_width_bounded']}", flush=True)
    return 0


def summarise(records, rho):
    def stats(vals):
        v = np.array([x for x in vals if x is not None], float); return {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None, "sd": float(v.std(ddof=1)) if len(v) > 1 else None}
    summ = {"methods": {}}
    for m in Q.METHODS:
        est = [r["methods"][m]["rho_clipped"] for r in records]; real = [r["rho_realized"] for r in records]; pairs = [(e, t) for e, t in zip(est, real) if e is not None]
        err = np.array([e - t for e, t in pairs]) if pairs else np.array([])
        summ["methods"][m] = {"n_identified": len(pairs), "mae": float(np.mean(np.abs(err))) if len(err) else None, "rmse": float(np.sqrt(np.mean(err ** 2))) if len(err) else None, "bias": float(err.mean()) if len(err) else None,
                              "brier": stats([r["methods"][m].get("brier") for r in records]), "logloss": stats([r["methods"][m].get("logloss") for r in records]), "auroc": stats([r["methods"][m].get("auroc") for r in records]),
                              "ess_fraction": stats([r["methods"][m].get("ess_fraction") for r in records]), "raw_outside": int(sum(1 for r in records if r["methods"][m]["rho_raw"] is not None and not 0 <= r["methods"][m]["rho_raw"] <= 1))}
    for key in ("interval_vcs", "interval_bce"):
        iv = [r[key] for r in records]; n = len(iv); real = [r["rho_realized"] for r in records]
        cov = sum(1 for z, t in zip(iv, real) if z["status"] in ("bounded_interval", "full_parameter_space") and z["lower"] <= t <= z["upper"]); width = [z["upper"] - z["lower"] for z in iv if z["status"] == "bounded_interval"]
        summ[key] = {"n": n, "coverage": cov / n, "mean_width_bounded": float(np.mean(width)) if width else None, "rate_empty": sum(z["status"] == "empty" for z in iv) / n, "rate_full": sum(z["status"] == "full_parameter_space" for z in iv) / n,
                     "rate_bounded": len(width) / n, "disjoint_all": all(z["disjoint_clusters"] for z in iv)}
    summ["rho_realized"] = stats([r["rho_realized"] for r in records]); return summ


if __name__ == "__main__":
    sys.exit(main())
