"""REV8 — read-only diagnostics of the frozen D1 selectors on CAL, prompted by an external code review (r8, 2026-10-13; the review package
is not part of this repository).  Critic forwards only: no host call, no DEV / official-test read; the frozen modules are imported, never edited.
Protocol: reports/REV8/REV8_DIAG_PROTOCOL_FROZEN_20261013.yaml.
  unit      --host llava|qwen --scale S --root R --tags T1 T2 ... --out O [--smoke-n N]   one process per data root (CAL loaded once)
  summarize --in O --protocol P --out DIR                                                 CPU: per-unit summaries, image-grouped bootstrap
Per unit, on every CAL question (the 12 D1-10 masks in the frozen tuple order: mechanism-major x budgets):
  A (review item R8-04): eval-mode r = the frozen predict_r / predict_r40, with the teacher's h and d.  Guard first: the frozen pooled CAL
    Spearman(r, h) must reproduce the unit's logged selected_cal_spearman within 1e-4, or the run stops (exit 3) before B / C.
  B (R8-02): the gate's own masks at each budget, critic in eval mode, one Gumbel draw xi per (question, budget) shared by the noisy cells:
    HD = hard top-k(g), HN = hard top-k(g + xi), SD = soft(g), SN = soft(g + xi) = the frozen relaxed top-k; soft cells at tau_end and tau_start.
  C (R8-01): R train-mode (dropout) draws, as phase B queries the critic: the full pass and the 12 masked passes (fixed masks); the gate's top-k
    from train-mode vs eval-mode full-pass tokens; and (C2) the phase-B loss itself, the soft gate mask at tau_end in train mode.
--smoke-n N (code check only): A and its guard on every CAL question, B and C on the first N questions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SEED = 20261013
R_DRAWS = 8
GUARD_TOL = 1e-4
MECHS = ("random", "grid", "qsim")
CELLS = ("HD", "HN", "SD_end", "SN_end", "SD_start", "SN_start")


def gumbel(shape, gen, device):
    """The frozen draw: u ~ U(0, 1) clamped to [1e-9, 1 - 1e-9], xi = -log(-log u) (relaxed_topk adds g - log(-log u))."""
    import torch
    u = torch.rand(shape, generator=gen, device=device).clamp_(1e-9, 1 - 1e-9)
    return -torch.log(-torch.log(u))


def soft_from_logits(z, k, tau, valid=None):
    """The frozen relaxed top-k threshold applied to given logits z (z = g + xi reproduces relaxed_topk / relaxed_topk40 exactly; z = g is the
    noise-free soft mask SD).  valid=None: every token valid (LLaVA, relaxed_topk); else the valid-token rule of relaxed_topk40."""
    import torch
    Nb = z.shape[1]
    if valid is None:
        zs = torch.sort(z.detach(), 1, descending=True).values; nv = torch.full_like(k, Nb)
    else:
        zs = torch.sort(torch.where(valid, z.detach(), torch.full_like(z, -float("inf"))), 1, descending=True).values; nv = valid.sum(1)
    hi = zs.gather(1, (k - 1).clamp(0, Nb - 1)[:, None]); lo = zs.gather(1, k.clamp(max=Nb - 1)[:, None])
    t = torch.where((k >= nv)[:, None], hi - 1.0, 0.5 * (hi + lo))
    return torch.sigmoid((z - t) / tau)


class LlavaHost:
    """Frozen D1-20 path (r7_d1_selectors + r7_d1_train loaders): 576 tokens per image, token budgets."""
    name = "llava"

    def __init__(self, root: Path, dev):
        import torch
        import r7_d1_selectors as SEL
        import r7_d1_train as TR
        self.torch, self.SEL, self.root, self.dev = torch, SEL, root, dev
        recs, self.coverage = TR.role_records(root / "teacher", root / "data", "CAL")
        self.data = TR.tuple_data(recs, root / "cache", "CAL", dev); self.qids = [r["qid"] for r in recs]

    def load(self, tag):
        ck = self.torch.load(self.root / "selectors" / f"{tag}.pt", map_location=self.dev, weights_only=False)
        self.cfg = self.SEL.Cfg(**{k: (tuple(v) if k == "budgets" else v) for k, v in ck["cfg"].items()})
        self.model = self.SEL.Selector(self.cfg, ck["objective"]).to(self.dev); self.model.load_state_dict(ck["state_dict"]); self.model.eval()
        self.budget_keys = [str(k) for k in self.cfg.budgets]; return ck["objective"]

    def predict_r(self, idx):
        return self.SEL.predict_r(self.model, self.data, idx)

    def groups(self, idx, seqs_per_q):
        c = max(1, (64 * 13) // seqs_per_q); return [np.arange(s, min(s + c, len(idx))) for s in range(0, len(idx), c)]

    def batch(self, qi, all_masks=False):
        return self.data.batch(qi, all_masks=True) if all_masks else self.data.batch(qi, np.zeros((len(qi), 1), dtype=int))

    def valid(self, b):
        return self.torch.ones(b["v"].shape[:2], dtype=self.torch.bool, device=self.dev)

    def full(self, b):
        Q, N = b["v"].shape[:2]; return self.model.encode(b["q_emb"], b["q_pad"], b["v"], self.torch.zeros(Q, N, device=self.dev))

    def masked(self, b, bias, expand):
        return self.model.encode(expand(b["q_emb"]), expand(b["q_pad"]), expand(b["v"]), bias)[1]

    def gates(self, tok, b):
        return self.model.gates(tok)

    def k(self, qi, j):
        return self.torch.full((len(qi),), int(self.cfg.budgets[j]), device=self.dev, dtype=self.torch.long)

    def hard(self, g, k, valid):
        return self.SEL.hard_topk(g, k)

    def soft(self, z, k, tau, valid):
        return soft_from_logits(z, k, tau)

    def soft_bias(self, m, valid):
        return self.torch.log(m + 1e-6)


class QwenHost(LlavaHost):
    """Frozen R7-40 path (r7_d1_40_selectors + r7_d1_40_train loaders): variable N per image, budgets k = round(f N)."""
    name = "qwen"

    def __init__(self, root: Path, dev, cfg_dict):
        import torch
        import r7_d1_40_selectors as S4
        import r7_d1_40_train as T4
        import r7_d1_selectors as SEL
        self.torch, self.S4, self.SEL, self.root, self.dev = torch, S4, SEL, root, dev
        cfg = self._cfg(cfg_dict); recs, self.coverage = T4.role_records(root / "teacher", Path("/home/infres/yinwang/CS_QMI/outputs/R7_D1/data"), "CAL")
        self.data = T4.tuple_data40(recs, root / "cache", "CAL", dev, cfg); self.qids = [r["qid"] for r in recs]

    def _cfg(self, d):
        return self.S4.Cfg40(**{k: (tuple(v) if k in ("budgets", "fractions") else v) for k, v in d.items()})

    def load(self, tag):
        ck = self.torch.load(self.root / "selectors" / f"{tag}.pt", map_location=self.dev, weights_only=False)
        self.cfg = self._cfg(ck["cfg"]); self.model = self.S4.Selector40(self.cfg, ck["objective"]).to(self.dev)
        self.model.load_state_dict(ck["state_dict"]); self.model.eval(); self.budget_keys = [f"{f:g}" for f in self.cfg.fractions]
        return ck["objective"]

    def predict_r(self, idx):
        return self.S4.predict_r40(self.model, self.data, idx)

    def groups(self, idx, seqs_per_q):
        return self.S4.plan_groups(self.data.lengths[idx], seqs_per_q, self.cfg.cost_eval)

    def valid(self, b):
        return b["valid"]

    def full(self, b):
        return self.model.encode(b["q_emb"], b["q_pad"], b["v"], self.S4.full_bias(b["valid"]), b["rows"], b["cols"])

    def masked(self, b, bias, expand):
        return self.model.encode(expand(b["q_emb"]), expand(b["q_pad"]), expand(b["v"]), bias, expand(b["rows"]), expand(b["cols"]))[1]

    def gates(self, tok, b):
        return self.model.gates40(tok, b["v"].shape[1])

    def k(self, qi, j):
        return self.S4.k_of(self.data, qi, self.cfg.fractions[j]).long()

    def hard(self, g, k, valid):
        return self.S4.hard_topk40(g, k, valid)

    def soft(self, z, k, tau, valid):
        return soft_from_logits(z, k, tau, valid)

    def soft_bias(self, m, valid):
        return self.torch.where(valid, self.torch.log(m + 1e-6), self.torch.full_like(m, self.S4.NEG))


def jaccard(a, b):
    return (a & b).sum(1).float() / (a | b).sum(1).clamp(min=1).float()


def run_unit(H, tag, log, smoke_n=None, R=R_DRAWS):
    """A, guard, B, C for one loaded unit -> dict of numpy arrays (+ scalars)."""
    torch = H.torch; objective = H.load(tag); data = H.data; n = data.n; idx = np.arange(n); tm = {}
    t0 = time.time(); r = H.predict_r(idx); tm["A_s"] = time.time() - t0
    h, d = data.h.cpu().numpy(), data.d.cpu().numpy()
    rho = float(H.SEL.spearman(r.reshape(-1), h.reshape(-1))); logged = float(log["selected_cal_spearman"])
    if not abs(rho - logged) <= GUARD_TOL:
        print(f"[rev8_diag] GUARD FAILED {tag}: pooled CAL Spearman {rho:.6f} vs logged {logged:.6f}", flush=True); sys.exit(3)
    sub = idx if smoke_n is None else idx[:smoke_n]; ns, nb = len(sub), len(H.budget_keys); taus = (H.cfg.tau_end, H.cfg.tau_start)
    out = {"r": r, "h": h, "d": d, "C": data.C.cpu().numpy(), "img_of": data.img_of.cpu().numpy(), "sub": sub,
           "guard_rho": np.array(rho), "logged_rho": np.array(logged)}
    risk = np.zeros((len(CELLS), ns, nb), np.float32); st = {k: np.zeros((2, 2, ns, nb), np.float32) for k in ("mass_minus_k", "binariness", "over_half_minus_k", "outside_frac")}
    jac_hn = np.zeros((ns, nb), np.float32); gen = torch.Generator(device=H.dev).manual_seed(SEED); t0 = time.time()
    with torch.no_grad():                                                                       # B: critic in eval mode
        H.model.eval()
        for g_ in H.groups(sub, 7):
            qi = sub[g_]; b = H.batch(qi); valid = H.valid(b); tok, pb = H.full(b); g = H.gates(tok, b); Q = len(qi)
            tile = lambda x: torch.cat([x] * len(CELLS), 0)  # noqa: E731   (cell-major)
            for j in range(nb):
                k = H.k(qi, j); xi = gumbel(g.shape, gen, H.dev); keep = {"D": H.hard(g, k, valid), "N": H.hard(g + xi, k, valid)}
                ms = {(ti, nz): H.soft(g + (xi if nz == "N" else 0.0), k, tau, valid) for ti, tau in enumerate(taus) for nz in ("D", "N")}
                bias = torch.cat([H.SEL.hard_bias(keep["D"]), H.SEL.hard_bias(keep["N"]), H.soft_bias(ms[0, "D"], valid), H.soft_bias(ms[0, "N"], valid),
                                  H.soft_bias(ms[1, "D"], valid), H.soft_bias(ms[1, "N"], valid)], 0)
                rr = H.model.readout(tile(pb), H.masked(b, bias, tile)).reshape(len(CELLS), Q)
                risk[:, g_, j] = rr.float().cpu().numpy(); jac_hn[g_, j] = jaccard(keep["D"], keep["N"]).cpu().numpy(); kf = k.float()
                for (ti, nz), m in ms.items():
                    ni = 0 if nz == "D" else 1; vm = m * valid; mass = vm.sum(1)
                    st["mass_minus_k"][ti, ni, g_, j] = (mass - kf).cpu().numpy()
                    st["binariness"][ti, ni, g_, j] = (vm * (1 - m)).sum(1).cpu().numpy()
                    st["over_half_minus_k"][ti, ni, g_, j] = (((m > 0.5) & valid).sum(1).float() - kf).cpu().numpy()
                    st["outside_frac"][ti, ni, g_, j] = ((vm * (~keep[nz]).float()).sum(1) / mass.clamp(min=1e-12)).cpu().numpy()
    tm["B_s"] = time.time() - t0; out.update({"B_risk": risk, "B_jac_hd_hn": jac_hn}); out.update({f"B_{k}": v for k, v in st.items()})
    M = r.shape[1]; Ls, Lq = np.zeros((ns, M)), np.zeros((ns, M)); Lh = np.zeros((2, ns, M)); TAs, TAq = np.zeros((ns, M)), np.zeros((ns, M))
    TBs, TBq = np.zeros(ns), np.zeros(ns); jac = np.zeros((ns, nb)); Lsoft = np.zeros((ns, nb)); t0 = time.time()
    gen2 = torch.Generator(device=H.dev).manual_seed(SEED + 2); torch.manual_seed(SEED + 1)    # dropout draws use the global generator
    with torch.no_grad():                                                                       # C: train mode = dropout on, as in phase B
        for g_ in H.groups(sub, M + 1):
            qi = sub[g_]; b = H.batch(qi, all_masks=True); valid = H.valid(b); Q = len(qi)
            H.model.eval(); tok_e, _ = H.full(b); g_e = H.gates(tok_e, b); ks = [H.k(qi, j) for j in range(nb)]
            keep_e = [H.hard(g_e, ks[j], valid) for j in range(nb)]; bias12 = H.SEL.hard_bias(b["keep"].reshape(Q * M, -1))
            inter = lambda x: x.repeat_interleave(M, 0)  # noqa: E731   (question-major, as the frozen predict_r)
            H.model.train()
            for rep in range(R):
                tok_r, pb_r = H.full(b); g_r = H.gates(tok_r, b)
                for j in range(nb):
                    jac[g_, j] += jaccard(H.hard(g_r, ks[j], valid), keep_e[j]).cpu().numpy() / R
                    m = H.soft(g_r + gumbel(g_r.shape, gen2, H.dev), ks[j], H.cfg.tau_end, valid)
                    Lsoft[g_, j] += H.model.readout(pb_r, H.masked(b, H.soft_bias(m, valid), lambda x: x)).float().cpu().numpy() / R
                pa_r = H.masked(b, bias12, inter); L = H.model.readout(inter(pb_r), pa_r).reshape(Q, M).double().cpu().numpy()
                Ls[g_] += L; Lq[g_] += L * L; Lh[rep * 2 // R, g_] += L / (R // 2)
                if objective != "direct":
                    tb = H.model.T(pb_r)[0].double().cpu().numpy(); ta = H.model.T(pa_r)[0].reshape(Q, M).double().cpu().numpy()
                    TBs[g_] += tb; TBq[g_] += tb * tb; TAs[g_] += ta; TAq[g_] += ta * ta
            H.model.eval()
    var = lambda s, q: (q - s * s / R) / (R - 1)  # noqa: E731   unbiased variance over the R draws
    tm["C_s"] = time.time() - t0
    out.update({"C_Lmean": Ls / R, "C_Lvar": var(Ls, Lq), "C_Lhalf": Lh, "C_jac_gate": jac, "C_Lsoft_train": Lsoft, "C_R": np.array(R)})
    if objective != "direct":
        out.update({"C_TAmean": TAs / R, "C_TAvar": var(TAs, TAq), "C_TBmean": TBs / R, "C_TBvar": var(TBs, TBq)})
    meta = {"tag": tag, "objective": objective, "lr": H.cfg.lr, "seed": H.cfg.seed, "host": H.name, "n_cal": int(n), "n_sub": int(ns),
            "budget_keys": H.budget_keys, "taus": list(taus), "R": R, "guard": {"rho": rho, "logged": logged}, "timing_s": tm,
            "logged_soft_hard_gap": log.get("soft_hard_gap"), "selected_step": log.get("selected_step")}
    return out, meta


def cmd_unit(a):
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"; root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    first = torch.load(root / "selectors" / f"{a.tags[0]}.pt", map_location="cpu", weights_only=False)
    t0 = time.time(); H = LlavaHost(root, dev) if a.host == "llava" else QwenHost(root, dev, first["cfg"]); load_s = time.time() - t0
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    print(f"[rev8_diag] {a.scale}: CAL {H.data.n} questions loaded in {load_s:.0f}s on {gpu}", flush=True)
    for tag in a.tags:
        label = f"{a.scale}__{tag}"; npz = out / f"{label}.npz"
        if npz.exists():
            print(f"[rev8_diag] {label} already done", flush=True); continue
        log = json.loads((root / "selectors" / f"{tag}.json").read_text()); t1 = time.time()
        arrays, meta = run_unit(H, tag, log, a.smoke_n)
        meta.update({"scale": a.scale, "root": str(root), "gpu": gpu, "smoke_n": a.smoke_n, "coverage": H.coverage})
        tmp = out / f"{label}.tmp.npz"; np.savez_compressed(tmp, **arrays); os.replace(tmp, npz)
        (out / f"{label}.json").write_text(json.dumps(meta, indent=1))
        print(f"[rev8_diag] {label}: guard rho {meta['guard']['rho']:.6f} (logged {meta['guard']['logged']:.6f}); "
              f"A {meta['timing_s']['A_s']:.0f}s B {meta['timing_s']['B_s']:.0f}s C {meta['timing_s']['C_s']:.0f}s; total {time.time() - t1:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------------------- summaries
def avg_rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="stable"); xs = x[o]; r = np.empty(len(x)); i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j); i = j + 1
    return r


def spearman_avg(a, b):
    ra, rb = avg_rank(a), avg_rank(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else None


def pairwise(pred, true):
    """Within-group pairwise direction accuracy: pred / true [G, m]; pairs with true ties are excluded (counted); pred ties score 0.5.
    -> per-group (score sum, valid pair count, true-tie pair count)."""
    m = pred.shape[1]; sc = np.zeros(len(pred)); cnt = np.zeros(len(pred)); ties = np.zeros(len(pred))
    for i in range(m):
        for j in range(i + 1, m):
            dt, dp = true[:, i] - true[:, j], pred[:, i] - pred[:, j]; ok = dt != 0
            sc += np.where(ok, np.where(dp == 0, 0.5, (np.sign(dp) == np.sign(dt)).astype(float)), 0.0); cnt += ok; ties += ~ok
    return sc, cnt, ties


def group_boot(img, num, den=None, reps=2000, seed=0):
    """Image-grouped bootstrap of sum(num) / sum(den) (den None: mean) over questions -> (estimate, lo, hi)."""
    img = np.asarray(img); u, inv = np.unique(img, return_inverse=True); den = np.ones_like(num, dtype=float) if den is None else den
    sn, sd = np.bincount(inv, weights=num, minlength=len(u)), np.bincount(inv, weights=den, minlength=len(u))
    rng = np.random.default_rng(seed); w = rng.multinomial(len(u), np.full(len(u), 1.0 / len(u)), size=reps)
    est = sn.sum() / sd.sum(); bs = (w @ sn) / (w @ sd)
    return [float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def summarize_unit(arr, meta):
    nb = len(meta["budget_keys"]); r, h, d = arr["r"], arr["h"], arr["d"]; n = len(r); img = arr["img_of"]
    r3, h3, d3 = (x.reshape(n, len(MECHS), nb) for x in (r, h, d)); S = {"meta": {k: meta[k] for k in ("tag", "objective", "lr", "seed", "scale", "host", "gpu", "n_cal", "n_sub", "R", "guard")}}
    # ---- A: R8-04
    A = {"pooled_spearman_frozen": meta["guard"]["rho"], "pooled_spearman_avg": spearman_avg(r.reshape(-1), h.reshape(-1))}
    A["within_budget_spearman"] = {meta["budget_keys"][j]: spearman_avg(r3[:, :, j].reshape(-1), h3[:, :, j].reshape(-1)) for j in range(nb)}
    cell = [spearman_avg(r3[:, m, j], h3[:, m, j]) for m in range(len(MECHS)) for j in range(nb)]
    A["within_budget_mechanism_spearman"] = {"mean": float(np.mean(cell)), "min": float(np.min(cell)), "max": float(np.max(cell))}
    wq = [spearman_avg(r[i], h[i]) for i in range(n)]; wq = [x for x in wq if x is not None]
    A["within_question_spearman_mean"] = {"mean": float(np.mean(wq)), "n_defined": len(wq), "n_questions": n}
    tri_r, tri_h, tri_d = (x.transpose(0, 2, 1).reshape(n * nb, len(MECHS)) for x in (r3, h3, d3)); tri_img = np.repeat(img, nb); tri_j = np.tile(np.arange(nb), n)
    sc, cnt, ties = pairwise(tri_r, tri_h); hmin = tri_h.min(1); pick = tri_r.argmin(1); best = tri_h.argmin(1)
    reg = {"critic": tri_h[np.arange(len(pick)), pick] - hmin, "random_choice_expected": tri_h.mean(1) - hmin,
           "always_random": tri_h[:, 0] - hmin, "always_grid": tri_h[:, 1] - hmin, "always_qsim": tri_h[:, 2] - hmin,
           "teacher_d_choice": tri_h[np.arange(len(pick)), tri_d.argmin(1)] - hmin}
    A["within_qk"] = {"pairwise_accuracy": group_boot(tri_img, sc, cnt), "pairs": int(cnt.sum()), "h_tie_pairs": int(ties.sum()),
                      "top1_hit_rate": group_boot(tri_img, (pick == best).astype(float)),
                      "regret": {k: group_boot(tri_img, v) for k, v in reg.items()},
                      "regret_critic_minus_random_choice": group_boot(tri_img, reg["critic"] - reg["random_choice_expected"]),
                      "regret_critic_minus_always_grid": group_boot(tri_img, reg["critic"] - reg["always_grid"]),
                      "by_budget": {meta["budget_keys"][j]: {"pairwise_accuracy": float(sc[tri_j == j].sum() / max(cnt[tri_j == j].sum(), 1)),
                                                              "regret_critic": float(reg["critic"][tri_j == j].mean()),
                                                              "regret_random_choice": float(reg["random_choice_expected"][tri_j == j].mean()),
                                                              "regret_always_grid": float(reg["always_grid"][tri_j == j].mean())} for j in range(nb)}}
    bud_r, bud_h = r3.reshape(n * len(MECHS), nb), h3.reshape(n * len(MECHS), nb); sc2, cnt2, ties2 = pairwise(bud_r, bud_h)
    A["within_q_mechanism_across_budgets"] = {"pairwise_accuracy": group_boot(np.repeat(img, len(MECHS)), sc2, cnt2), "h_tie_pairs": int(ties2.sum())}
    S["A_R8_04"] = A
    # ---- B: R8-02 (questions in `sub`)
    sub = arr["sub"]; simg = img[sub]; rk = {c: arr["B_risk"][i] for i, c in enumerate(CELLS)}; B = {"by_budget": {}}; lg = meta.get("logged_soft_hard_gap") or {}
    for j, key in enumerate(meta["budget_keys"]):
        HD, HN, SDe, SNe, SDs, SNs = (rk[c][:, j].astype(float) for c in CELLS); e = {}
        for nm, v in (("relax_det", SDe - HD), ("relax_noise", SNe - HN), ("noise_hard", HN - HD), ("legacy_SN_minus_HD", SNe - HD)):
            e[nm] = group_boot(simg, v)
        e["identity_residual_max"] = float(np.abs((SNe - HD) - ((SNe - HN) + (HN - HD))).max())
        e["means"] = {"HD": float(HD.mean()), "HN": float(HN.mean()), "SD_end": float(SDe.mean()), "SN_end": float(SNe.mean()), "SD_start": float(SDs.mean()), "SN_start": float(SNs.mean())}
        e["tau_start"] = {"relax_det": float((SDs - HD).mean()), "relax_noise": float((SNs - HN).mean()), "legacy": float((SNs - HD).mean())}
        lgk = lg.get(key)
        if lgk is not None:
            v = SNe - HD; se = v.std(ddof=1) / np.sqrt(len(v)); e["logged_gap"] = lgk["gap"]; e["logged_gap_z"] = float((v.mean() - lgk["gap"]) / se) if se > 0 else None
        e["jaccard_HD_HN"] = float(arr["B_jac_hd_hn"][:, j].mean())
        e["mask"] = {f"{st}_{'end' if ti == 0 else 'start'}_{'SD' if ni == 0 else 'SN'}": float(arr[f"B_{st}"][ti, ni, :, j].mean())
                     for st in ("mass_minus_k", "binariness", "over_half_minus_k", "outside_frac") for ti in (0, 1) for ni in (0, 1)}
        B["by_budget"][key] = e
    S["B_R8_02"] = B
    # ---- C: R8-01 (questions in `sub`)
    R = int(arr["C_R"]); rs = r[sub]; Lm, Lv, Lh = arr["C_Lmean"], arr["C_Lvar"], arr["C_Lhalf"]; C = {"R": R, "by_budget": {}}
    for j, key in enumerate(meta["budget_keys"]):
        cols = [m * nb + j for m in range(len(MECHS))]; ev, tr, tv = rs[:, cols].astype(float), Lm[:, cols], Lv[:, cols]
        D = tr - ev; vD = D.var(1, ddof=1) - (tv / R).mean(1); vE = ev.var(1, ddof=1)
        e = {"mean_eval_readout": float(ev.mean()), "mean_train_readout": float(tr.mean()), "mean_shift": group_boot(simg, D.mean(1)),
             "distortion_ratio": float(vD.mean() / vE.mean()) if vE.mean() > 0 else None,
             "distortion_ratio_raw": float(D.var(1, ddof=1).mean() / vE.mean()) if vE.mean() > 0 else None,
             "rank_agreement_R": float(pairwise(tr, ev)[0].sum() / max(pairwise(tr, ev)[1].sum(), 1)),
             "rank_agreement_half": float(np.mean([pairwise(Lh[i][:, cols], ev)[0].sum() / max(pairwise(Lh[i][:, cols], ev)[1].sum(), 1) for i in (0, 1)])),
             "gate_topk_jaccard_train_vs_eval": float(arr["C_jac_gate"][:, j].mean()),
             "phaseB_loss_train_mode": float(arr["C_Lsoft_train"][:, j].mean()), "phaseB_loss_eval_mode_SN": float(arr["B_risk"][CELLS.index("SN_end")][:, j].mean())}
        if "C_TAvar" in arr:
            e["var_TA_mean"] = float(arr["C_TAvar"][:, cols].mean()); e["var_TB_mean"] = float(arr["C_TBvar"].mean()); e["self_gap_2varTB"] = 2 * e["var_TB_mean"]
        C["by_budget"][key] = e
    S["C_R8_01"] = C
    return S


def flags(S, P):
    f = {}; A, B, C = S["A_R8_04"], S["B_R8_02"], S["C_R8_01"]; th = P["flags"]
    pa = A["within_qk"]["pairwise_accuracy"]; f["R8_04_within_qk_not_better_than_chance"] = bool(pa[1] <= 0.5 <= pa[2])
    wb = [v for v in A["within_budget_spearman"].values() if v is not None]
    f["R8_04_pooled_mostly_between_budgets"] = bool(np.mean(wb) < th["between_budget_fraction"] * A["pooled_spearman_avg"])
    rd = np.mean([abs(e["relax_det"][0]) for e in B["by_budget"].values()]); nh = np.mean([abs(e["noise_hard"][0]) for e in B["by_budget"].values()])
    f["R8_02_dominant"] = "relaxation" if rd >= th["dominance_factor"] * nh else ("noise" if nh >= th["dominance_factor"] * rd else "mixed")
    f["R8_02_legacy_reproduced"] = all(e.get("logged_gap_z") is None or abs(e["logged_gap_z"]) <= th["legacy_z"] for e in B["by_budget"].values())
    dr = [e["distortion_ratio"] for e in C["by_budget"].values() if e["distortion_ratio"] is not None]
    f["R8_01_dropout_material_for_ranking"] = bool(max(dr) >= th["distortion_ratio"]) if dr else None
    k25 = list(C["by_budget"])[1]                                                               # the 25 % budget (144 tokens / f 0.25)
    f["R8_01_gate_inputs_perturbed"] = bool(C["by_budget"][k25]["gate_topk_jaccard_train_vs_eval"] < th["gate_jaccard"])
    return f


def cmd_summarize(a):
    import yaml
    P = yaml.safe_load(Path(a.protocol).read_text())["rev8_diag"]; units = {}
    for js in sorted(Path(a.inp).glob("*.json")):
        meta = json.loads(js.read_text()); arr = dict(np.load(js.with_suffix(".npz")))
        S = summarize_unit(arr, meta); S["flags"] = flags(S, P); units[f"{meta['scale']}__{meta['tag']}"] = S
        print(f"[rev8_diag] summarized {meta['scale']}__{meta['tag']}", flush=True)
    res = {"protocol": a.protocol, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "units": units}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / "rev8_diag_results.json").write_text(json.dumps(res, indent=1))
    print(f"[rev8_diag] wrote {out / 'rev8_diag_results.json'} ({len(units)} units)", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("unit"); u.add_argument("--host", choices=["llava", "qwen"], required=True); u.add_argument("--scale", required=True)
    u.add_argument("--root", required=True); u.add_argument("--tags", nargs="+", required=True); u.add_argument("--out", required=True)
    u.add_argument("--smoke-n", type=int, default=None)
    s = sub.add_parser("summarize"); s.add_argument("--in", dest="inp", required=True); s.add_argument("--protocol", required=True); s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return cmd_unit(a) if a.cmd == "unit" else cmd_summarize(a)


if __name__ == "__main__":
    sys.exit(main())
