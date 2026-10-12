"""r7 R7-30 D1-20 training units (torch only; any GPU type): assemble (question, mask) tuples from the D1-10 teacher records and the
feature cache, then train one selector (scripts/r7_d1_selectors.py).
  train : --objective vcs|logistic|direct --lr LR --seed S [--no-question] [--target h|d] --teacher-dir T --manifest-dir D --cache C --out O
          FIT tuples from T/<FIT shards>, CAL tuples from T/<CAL shards>; DEV is never read here
  smoke : --probe-records R --cache C --out O   (the D1-10 smoke's probe records stand in for FIT and CAL; pipeline check only)
Masks: 3 mechanisms × 4 budgets per question (random / grid re-derived from the question id and checked against the recorded digests;
qsim from the recorded ordering).  Targets on the ±1 scale: T = 2 p̃ − 1, h = (C − a)² − (C − b)², d = (b − a)².
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_masks as MK  # noqa: E402
import r7_d1_selectors as SEL  # noqa: E402
import r7_d1_teacher as TE  # noqa: E402
import r7_vl_common as V  # noqa: E402


def T(pair):
    y, n = pair
    return 2.0 * y / max(y + n, 1e-12) - 1.0


def assemble(records, image_index, qid_ids, n_vis=576):
    """records -> arrays (img_of, q_ids padded, q_len, keep [n, 12, N], C, h, d); mask order = mechanism-major × budgets."""
    pos = {i: j for j, i in enumerate(image_index)}; n = len(records); M = len(MK.MECHANISMS) * len(MK.BUDGETS)
    keep = np.zeros((n, M, n_vis), dtype=bool); h = np.zeros((n, M), np.float32); d = np.zeros((n, M), np.float32)
    C = np.array([r["C"] for r in records], np.int64); img_of = np.array([pos[r["image_id"]] for r in records], np.int64)
    ids = [qid_ids[r["qid"]] for r in records]; L = max(len(x) for x in ids); q_ids = np.zeros((n, L), np.int64); q_len = np.array([len(x) for x in ids])
    for i, (r, x) in enumerate(zip(records, ids)):
        q_ids[i, :len(x)] = x; tb = T(r["full"])
        orders = {"random": MK.order_random(r["qid"])[0], "grid": MK.order_grid(r["qid"])[0], "qsim": TE.unb64(r["qsim_order"])}
        for mech in ("random", "grid"):
            if TE.digest(orders[mech]) != r["order_digest"][mech]:
                raise RuntimeError(f"ordering digest mismatch for {r['qid']} / {mech}")
        j = 0
        for mech in MK.MECHANISMS:
            for k in MK.BUDGETS:
                keep[i, j, MK.kept(orders[mech], k)] = True; ta = T(r["masked"][mech][str(k)])
                h[i, j] = (r["C"] - ta) ** 2 - (r["C"] - tb) ** 2; d[i, j] = (tb - ta) ** 2; j += 1
    return img_of, q_ids, q_len, keep, C, h, d


def tuple_data(records, cache: Path, role: str, device):
    import torch
    feats = torch.from_numpy(np.load(cache / f"feats_{role}.npy")); index = json.loads((cache / f"images_{role}.json").read_text())
    qid_ids = {r["qid"]: r["ids"] for r in V.read_jsonl(cache / "question_ids.jsonl")}; emb = torch.load(cache / "embed_table.pt")
    img_of, q_ids, q_len, keep, C, h, d = assemble(records, index, qid_ids, feats.shape[1])
    return SEL.TupleData(feats, img_of, q_ids, q_len, emb, keep, C, h, d, device)


def role_records(teacher_dir: Path, manifest_dir: Path, role: str):
    import r7_d1_analyze as AN
    if role == "DEV":
        raise SystemExit("DEV is sealed")
    rows, cov = AN.load_role(teacher_dir, manifest_dir, role)
    return rows, cov


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--objective", choices=["vcs", "logistic", "direct"], required=True); t.add_argument("--lr", type=float, required=True)
    t.add_argument("--seed", type=int, required=True); t.add_argument("--no-question", action="store_true"); t.add_argument("--target", choices=["h", "d"], default="h")
    for x in ("--teacher-dir", "--manifest-dir", "--cache", "--out"):
        t.add_argument(x, required=True)
    t.add_argument("--steps-a", type=int, default=20000); t.add_argument("--steps-b", type=int, default=5000); t.add_argument("--eval-every", type=int, default=1000)
    s = sub.add_parser("smoke"); s.add_argument("--probe-records", required=True); s.add_argument("--cache", required=True); s.add_argument("--out", required=True)
    s.add_argument("--steps-a", type=int, default=200); s.add_argument("--steps-b", type=int, default=100)
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "smoke":
        if dev == "cuda":                                                              # one sub-folder per GPU type (smokes may run in parallel)
            out = out / torch.cuda.get_device_name(0).replace(" ", "_"); out.mkdir(parents=True, exist_ok=True)
        recs = V.read_jsonl(Path(a.probe_records)); fit = tuple_data(recs, Path(a.cache), "PROBE", dev); res = {}
        for obj in ("vcs", "logistic", "direct"):
            cfg = SEL.Cfg(steps_a=a.steps_a, steps_b=a.steps_b, eval_every=max(a.steps_a // 4, 1), q_per_step=min(64, fit.n), seed=0, warmup=min(20, a.steps_a))
            _, log = SEL.train(cfg, obj, fit, fit, (1.0, 1.0), out / f"smoke_{obj}.json", out / f"smoke_{obj}.pt")
            tm = log["timing"]; full = SEL.Cfg()
            per = {"s_per_phase_a_step": tm["phase_a_train_s"] / max(tm["phase_a_steps"], 1), "s_per_eval_pair": tm["eval_s"] / max(tm["eval_pairs"], 1),
                   "s_per_phase_b_step": tm["phase_b_s"] / max(tm["phase_b_steps"], 1)}
            n_evals = full.steps_a // full.eval_every; cal_pairs = 7989 * 12                # frozen CAL size × 12 masks
            per["unit_hours_estimate"] = (full.steps_a * per["s_per_phase_a_step"] + n_evals * cal_pairs * per["s_per_eval_pair"]
                                          + full.steps_b * per["s_per_phase_b_step"]) / 3600
            res[obj] = {"evals": log["evals"], "selected_step": log["selected_step"], "soft_hard_gap": log["soft_hard_gap"], "seconds": log["seconds"],
                        "timing": tm, "per_unit": per, "q_per_step": cfg.q_per_step, "masks_per_q": cfg.masks_per_q}
            res[obj]["dynamic_tau_50"] = SEL.dynamic_tau(_, fit, cfg, 0.5)
        res["env"] = V.env_info(); (out / "smoke_report.json").write_text(V.jdump(res, indent=1)); print(V.jdump(res, indent=1))
        return 0
    md, td, cache = Path(a.manifest_dir), Path(a.teacher_dir), Path(a.cache)
    meta = json.loads((md / "d1_manifest_meta.json").read_text()); wts = (meta["class_weights_from_FIT"]["w_yes"], meta["class_weights_from_FIT"]["w_no"])
    fit_recs, fit_cov = role_records(td, md, "FIT"); cal_recs, cal_cov = role_records(td, md, "CAL")
    fit, cal = tuple_data(fit_recs, cache, "FIT", dev), tuple_data(cal_recs, cache, "CAL", dev)
    cfg = SEL.Cfg(lr=a.lr, seed=a.seed, steps_a=a.steps_a, steps_b=a.steps_b, eval_every=a.eval_every, no_question=a.no_question, target=a.target)
    tag = f"{a.objective}_lr{a.lr:g}_s{a.seed}" + ("_noq" if a.no_question else "") + ("_dtarget" if a.target == "d" else "")
    if (out / f"{tag}.pt").exists():
        print(f"[r7_d1_train] {tag} already complete"); return 0
    model, log = SEL.train(cfg, a.objective, fit, cal, wts, out / f"{tag}.json", out / f"{tag}.pt", resume_path=out / f"{tag}.resume.pt")
    log["dynamic"] = {str(f): SEL.dynamic_tau(model, cal, cfg, f) for f in (0.25, 0.5, 0.75)}; log["coverage"] = {"FIT": fit_cov, "CAL": cal_cov}
    log["env"] = V.env_info(); (out / f"{tag}.json").write_text(V.jdump(log, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
