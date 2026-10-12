"""r7 R7-40b training units (main env, torch only): assemble (question, mask) tuples from the R7-40a teacher records and the Qwen feature cache,
then train one selector with the frozen D1-20 recipe (scripts/r7_d1_40_selectors.py adapter).
  train : --objective vcs|logistic|direct --lr LR --seed S --teacher-dir T --manifest-dir D --cache C --out O
          FIT tuples from T/<FIT shards>, CAL tuples from T/<CAL shards>; DEV is never read here (refused)
  smoke : --probe-records R --cache C --out O   (the R7-40 probe records stand in for FIT and CAL; pipeline check, per-unit cost at the real
          batch, worst-case memory of a cost-capped micro-batch (synthetic 1280-token images), CUDA fp32 vs CPU fp64 check of the adapter)
Masks: 3 mechanisms × 4 fractions per question (random / grid re-derived from the question id and the image's grid and checked against the
recorded digests; qsim from the recorded ordering).  Targets on the ±1 scale: T = 2 p̃ − 1, h = (C − a)² − (C − b)², d = (b − a)².
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_40_masks as M4  # noqa: E402
import r7_d1_40_selectors as S4  # noqa: E402
import r7_d1_selectors as SEL  # noqa: E402
import r7_d1_teacher as TE  # noqa: E402
import r7_vl_common as V  # noqa: E402

LRS = (1e-4, 3e-4)


def T(pair):
    y, n = pair
    return 2.0 * y / max(y + n, 1e-12) - 1.0


def assemble40(records, index, qid_ids):
    """records -> arrays (img_of, q_ids padded, q_len, rank [n, 3, N_max] int16, kk [n, 4], C, h, d [n, 12]); mask order = mechanism-major ×
    fractions.  The teacher's merged grid must equal the cache's for every image (same processor; checked)."""
    pos = {i: j for j, i in enumerate(index["images"])}; n = len(records); M = len(M4.MECHANISMS) * len(M4.FKEYS)
    nmax = max(index["n"][pos[r["image_id"]]] for r in records)
    rank = np.full((n, len(M4.MECHANISMS), nmax), S4.PAD_RANK, dtype=np.int16); kk = np.zeros((n, len(M4.FKEYS)), np.int32)
    h = np.zeros((n, M), np.float32); d = np.zeros((n, M), np.float32)
    C = np.array([r["C"] for r in records], np.int64); img_of = np.array([pos[r["image_id"]] for r in records], np.int64)
    ids = [qid_ids[r["qid"]] for r in records]; L = max(len(x) for x in ids); q_ids = np.zeros((n, L), np.int64); q_len = np.array([len(x) for x in ids])
    for i, (r, x) in enumerate(zip(records, ids)):
        j_img = pos[r["image_id"]]; nv = index["n"][j_img]; rows, cols = index["grid"][j_img]
        if r["n_vis"] != nv or list(r["grid"]) != [rows, cols] or rows * cols != nv:
            raise RuntimeError(f"teacher / cache grid mismatch for {r['qid']}: {r['n_vis']} {r['grid']} vs {nv} {[rows, cols]}")
        q_ids[i, :len(x)] = x; tb = T(r["full"])
        orders = {"random": M4.order_random(r["qid"], rows, cols)[0], "grid": M4.order_grid(r["qid"], rows, cols, False)[0], "qsim": TE.unb64(r["qsim_order"])}
        for mech in ("random", "grid"):
            if TE.digest(orders[mech]) != r["order_digest"][mech]:
                raise RuntimeError(f"ordering digest mismatch for {r['qid']} / {mech}")
        if len(orders["qsim"]) != nv or sorted(orders["qsim"].tolist()) != list(range(nv)):
            raise RuntimeError(f"qsim ordering of {r['qid']} is not a permutation of its {nv} tokens")
        ks = M4.budgets(nv)
        if [r["budgets"][key] for key in M4.FKEYS] != [ks[key] for key in M4.FKEYS]:
            raise RuntimeError(f"recorded budgets of {r['qid']} differ from k = round(f·N)")
        kk[i] = [ks[key] for key in M4.FKEYS]; j = 0
        for mi, mech in enumerate(M4.MECHANISMS):
            rank[i, mi, :nv] = M4.ranks(orders[mech])
            for key in M4.FKEYS:
                ta = T(r["masked"][mech][key]); h[i, j] = (r["C"] - ta) ** 2 - (r["C"] - tb) ** 2; d[i, j] = (tb - ta) ** 2; j += 1
    return img_of, q_ids, q_len, rank, kk, C, h, d


def tuple_data40(records, cache: Path, role: str, device, cfg):
    import torch
    feats = torch.from_numpy(np.load(cache / f"feats_{role}.npy")); index = json.loads((cache / f"images_{role}.json").read_text())
    qid_ids = {r["qid"]: r["ids"] for r in V.read_jsonl(cache / "question_ids.jsonl")}; emb = torch.load(cache / "embed_table.pt")
    img_of, q_ids, q_len, rank, kk, C, h, d = assemble40(records, index, qid_ids)
    return S4.TupleData40(feats, index["offsets"], index["n"], [g[1] for g in index["grid"]], img_of, q_ids, q_len, emb, rank, kk, C, h, d, device,
                          cfg.max_rows, cfg.max_cols)


def role_records(teacher_dir: Path, manifest_dir: Path, role: str):
    import r7_d1_analyze as AN
    if role == "DEV":
        raise SystemExit("DEV is sealed")
    return AN.load_role(teacher_dir, manifest_dir, role)


def tag(obj, lr, seed):
    return f"{obj}_lr{lr:g}_s{seed}"


# ------------------------------------------------------------------------------------------------------------------------------ smoke checks
def synthetic_data(n_q, n_tok, cols, lq, device, cfg, seed=0):
    """Worst-case-shaped data for the memory probe: every image n_tok tokens, every question lq tokens."""
    import torch
    g = np.random.default_rng(seed); n_img = n_q
    feats = torch.tensor(g.normal(size=(n_img * n_tok, cfg.vis_dim)), dtype=torch.float16); offsets = np.arange(n_img) * n_tok
    rank = np.stack([np.stack([g.permutation(n_tok) for _ in range(3)]) for _ in range(n_q)]).astype(np.int16)
    kk = np.array([[M4.budget(f, n_tok) for f in M4.FRACTIONS]] * n_q, np.int32); C = np.where(g.uniform(size=n_q) < 0.5, 1, -1)
    emb = torch.tensor(g.normal(size=(64, cfg.emb_dim)), dtype=torch.bfloat16)
    return S4.TupleData40(feats, offsets, [n_tok] * n_img, [cols] * n_img, np.arange(n_q), g.integers(1, 64, size=(n_q, lq)), np.full(n_q, lq), emb,
                          rank, kk, C, g.uniform(-1, 1, size=(n_q, 12)), g.uniform(0, 1, size=(n_q, 12)), device, cfg.max_rows, cfg.max_cols)


def memory_probe(device):
    """Peak memory of one phase-A step, one phase-B step and one evaluation pass at the cost cap, every image at 1,280 tokens (32 × 40)."""
    import torch
    cfg = S4.Cfg40(steps_a=1, steps_b=1, eval_every=1, warmup=1); data = synthetic_data(64, 1280, 40, 40, device, cfg)
    res = {"tokens_per_image": 1280, "question_tokens": 40, "groups_phase_a": len(S4.plan_groups(data.lengths, 5, cfg.cost_train)),
           "cost_train": cfg.cost_train, "cost_eval": cfg.cost_eval, "gpu_total_gb": torch.cuda.get_device_properties(0).total_memory / 2 ** 30}
    torch.cuda.reset_peak_memory_stats()
    try:
        S4.train40(cfg, "direct", data, data, (1.0, 1.0)); res["oom"] = False
    except torch.cuda.OutOfMemoryError as e:                                             # recorded, not fatal: the hardware-class check
        res["oom"] = True; res["oom_message"] = str(e).splitlines()[0][:300]
    res["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    del data; torch.cuda.empty_cache()
    return res


def gpu_check(device):
    """CUDA fp32 vs CPU fp64 (same weights) for the adapter with padding: phase-A loss / gradients and a phase-B gate step with a
    deterministic soft mask (relative error <= 1e-3; the frozen r7_d1_attn_check criterion)."""
    import torch
    torch.manual_seed(1); cfg = S4.Cfg40(dropout=0.0); m = S4.Selector40(cfg, "vcs").eval(); ref = copy.deepcopy(m).double()
    data = synthetic_data(4, 300, 15, 9, "cpu", cfg, seed=3); data.ntok[1] = 260                                  # image 1 padded inside the batch
    data.lengths = data.q_len.cpu().numpy() + data.ntok[data.img_of].cpu().numpy()
    b = data.batch(np.arange(4), np.array([[0, 5, 11]] * 4)); res = {}
    for name, mod, dtype, dd in (("cuda", m.to(device), torch.float32, device), ("cpu64", ref, torch.float64, "cpu")):
        bb = {k: (v.to(dd, dtype) if v.is_floating_point() else v.to(dd)) for k, v in b.items()}
        wts = torch.tensor([1.0, 1.0], dtype=dtype, device=dd); Cq = bb["C"].float(); W_b = torch.where(Cq > 0, wts[0], wts[1]).sum().to(dtype)
        mod.zero_grad(set_to_none=True); la = S4.estimator_terms(mod, bb, wts, W_b, 3 * W_b); la.backward()
        ga = {n: p.grad.clone() for n, p in mod.named_parameters() if p.grad is not None}; mod.zero_grad(set_to_none=True)
        tok, pb = mod.encode(bb["q_emb"], bb["q_pad"], bb["v"], S4.full_bias(bb["valid"]).to(dtype), bb["rows"], bb["cols"])
        ms = torch.sigmoid(mod.gates40(tok, bb["v"].shape[1]) / 0.5); bias = torch.where(bb["valid"], torch.log(ms + 1e-6), torch.full_like(ms, S4.NEG))
        lb = mod.readout(pb, mod.encode(bb["q_emb"], bb["q_pad"], bb["v"], bias, bb["rows"], bb["cols"])[1]).mean(); lb.backward()
        gb = {n: p.grad.clone() for n, p in mod.named_parameters() if n.startswith("gate.") and p.grad is not None}
        res[name] = {"la": la.detach(), "lb": lb.detach(), "ga": ga, "gb": gb}
    rel = lambda a, b: float((a.double().cpu() - b.double().cpu()).abs().max() / (b.double().abs().max().cpu() + 1e-12))  # noqa: E731
    a, c = res["cuda"], res["cpu64"]
    r = {"rel_loss_phaseA": rel(a["la"], c["la"]), "rel_loss_phaseB": rel(a["lb"], c["lb"]), "rel_grad_phaseA": max(rel(a["ga"][n], c["ga"][n]) for n in c["ga"]),
         "rel_grad_gate_phaseB": max(rel(a["gb"][n], c["gb"][n]) for n in c["gb"]), "gate_grad_nonzero": bool(any(float(g.abs().max()) > 0 for g in a["gb"].values()))}
    r["match"] = all(v < 1e-3 for k, v in r.items() if k.startswith("rel_")) and r["gate_grad_nonzero"]
    return r


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--objective", choices=["vcs", "logistic", "direct"], required=True); t.add_argument("--lr", type=float, required=True)
    t.add_argument("--seed", type=int, required=True)
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
        res = {"gpu_check": gpu_check(dev) if dev == "cuda" else None}
        recs = V.read_jsonl(Path(a.probe_records)); base = S4.Cfg40(); fit = tuple_data40(recs, Path(a.cache), "PROBE", dev, base)
        for obj in ("vcs", "logistic", "direct"):
            cfg = S4.Cfg40(steps_a=a.steps_a, steps_b=a.steps_b, eval_every=max(a.steps_a // 4, 1), q_per_step=min(64, fit.n), seed=0, warmup=min(20, a.steps_a))
            model, log = S4.train40(cfg, obj, fit, fit, (1.0, 1.0), out / f"smoke_{obj}.json", out / f"smoke_{obj}.pt")
            tm = log["timing"]; full = S4.Cfg40()
            per = {"s_per_phase_a_step": tm["phase_a_train_s"] / max(tm["phase_a_steps"], 1), "s_per_eval_pair": tm["eval_s"] / max(tm["eval_pairs"], 1),
                   "s_per_phase_b_step": tm["phase_b_s"] / max(tm["phase_b_steps"], 1), "micro_batches_per_a_step": tm["micro_batches_a"] / max(tm["phase_a_steps"], 1)}
            n_evals = full.steps_a // full.eval_every; cal_pairs = 7989 * 12                # frozen CAL size × 12 masks
            per["unit_hours_estimate"] = (full.steps_a * per["s_per_phase_a_step"] + n_evals * cal_pairs * per["s_per_eval_pair"]
                                          + full.steps_b * per["s_per_phase_b_step"]) / 3600
            res[obj] = {"evals": log["evals"], "selected_step": log["selected_step"], "soft_hard_gap": log["soft_hard_gap"], "seconds": log["seconds"],
                        "timing": tm, "per_unit": per, "q_per_step": cfg.q_per_step, "masks_per_q": cfg.masks_per_q, "peak_mem_gb": log.get("peak_mem_gb")}
            res[obj]["dynamic_tau_50"] = S4.dynamic_tau40(model, fit, cfg, 0.5)
        if dev == "cuda":
            res["memory_probe"] = memory_probe(dev)
        res["data"] = {"questions": fit.n, "mean_tokens": float(np.mean(fit.lengths)), "max_tokens": int(np.max(fit.lengths))}
        res["env"] = V.env_info(); (out / "smoke_report.json").write_text(V.jdump(res, indent=1)); print(V.jdump(res, indent=1))
        return 0 if (res["gpu_check"] is None or res["gpu_check"]["match"]) else 6
    md, td, cache = Path(a.manifest_dir), Path(a.teacher_dir), Path(a.cache); name = tag(a.objective, a.lr, a.seed)
    pt, lp, done = out / f"{name}.pt", out / f"{name}.json", out / f"{name}.done.json"
    if done.exists():
        print(f"[r7_d1_40_train] {name} already complete"); return 0
    meta = json.loads((md / "d1_manifest_meta.json").read_text()); wts = (meta["class_weights_from_FIT"]["w_yes"], meta["class_weights_from_FIT"]["w_no"])
    fit_recs, fit_cov = role_records(td, md, "FIT"); cal_recs, cal_cov = role_records(td, md, "CAL")
    cfg = S4.Cfg40(lr=a.lr, seed=a.seed, steps_a=a.steps_a, steps_b=a.steps_b, eval_every=a.eval_every)
    cal = tuple_data40(cal_recs, cache, "CAL", dev, cfg)
    if pt.exists():                                                                    # trained, but stopped before the CAL dynamic taus were written
        ck = torch.load(pt, map_location=dev, weights_only=False); model = S4.Selector40(cfg, a.objective).to(dev); model.load_state_dict(ck["state_dict"])
        log = json.loads(lp.read_text())
    else:
        fit = tuple_data40(fit_recs, cache, "FIT", dev, cfg)
        model, log = S4.train40(cfg, a.objective, fit, cal, wts, lp, pt, resume_path=out / f"{name}.resume.pt")
    log["dynamic"] = {str(f): S4.dynamic_tau40(model, cal, cfg, f) for f in (0.25, 0.5, 0.75)}; log["coverage"] = {"FIT": fit_cov, "CAL": cal_cov}
    log["env"] = V.env_info(); lp.write_text(V.jdump(log, indent=1))
    done.write_text(json.dumps({"unit": name, "selected_step": log["selected_step"], "selected_cal_spearman": log["selected_cal_spearman"],
                                "dynamic_taus": {f: v["tau"] for f, v in log["dynamic"].items()}, "gpu": log["env"].get("gpu")}, indent=1))   # marker, written last
    return 0


if __name__ == "__main__":
    sys.exit(main())
