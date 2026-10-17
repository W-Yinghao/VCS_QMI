"""V10 T03 evaluator (decision 9.5): the ONLY T03 module that reads the population truth; it refuses to run inside the truth barrier.

For the frozen selected checkpoint of a unit (and, as a labelled diagnostic never used for selection, for every checkpoint):
  population  E_M(T - eta)^2 and J_pop(T) on the m x m table (m^2 <= 1024 cells, float64), the identity check
              S - J_pop(T) = E_M(T - eta)^2 (|residual| <= 2e-13, the package checker's tolerance)
  empirical   complete U-J on EVAL (negatives kept; never clipped), with the U/V diagnostic on the SAME frozen critic and data
              (V adds the n diagonal pairs), plus FIT / CAL U-J and V-J; EVAL U-J - S, EVAL U-J - J_pop, J_pop - S
  head gap    MF only: the fixed-feature empirical head problem on the full FIT batch at the checkpoint's features (U mode, package
              QuadraticHead): G(0) = ||Delta||_1, the FW gap of the checkpoint W (training dtype and same-input fp64), and a 1000-iteration
              fp64 polish from W bracketing the suboptimality [J(W_polished) - J(W), FW gap(W)]
  permutation conditional variance on the same base data (FIT and EVAL): R = 512 uniform derangements from the named 'diag_perm'
              stream of (data root seed, condition, role) - identical for every method on the same data - mean / variance of the
              perm1 estimate, the implied perm8 variance (groups of 8), and the z score of the mean against the exact U-J
  saturated   decision 9.5 FIT-count table T* = (p_hat - q_hat) / (p_hat + q_hat) (U and V references), evaluated like a critic; it is
              the empirical FIT U-J optimum of the saturated table class (checked: its FIT U-J dominates every checkpoint of the unit)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from v10 import adapter as A
from v10 import mf_vcs as MF

import t03_common as C
import t03_models as Mo
import t03_objectives as O
from t03_data import GEN, SampleSet, load_role

PERM_DIAG_R, PERM_DIAG_GROUP = 512, C.PERM8_R
HEAD_POLISH_ITERATIONS = 1000


def _outside_barrier(what: str) -> None:
    if C.barrier_active():
        raise C.TruthAccessError(f"{what} is evaluator-only and refuses to run inside the truth barrier")


def load_truth(case_directory: Path) -> dict:
    _outside_barrier("load_truth")
    case_directory = Path(case_directory)
    meta = json.loads((case_directory / C.TRUTH_DIR_NAME / "truth.json").read_text(encoding="utf-8"))
    digest = C.sha256_file(case_directory / C.TRUTH_DIR_NAME / "truth.npz")
    if digest != meta["oracle_file_sha256"]:
        raise RuntimeError(f"{case_directory}: truth.npz sha256 {digest} != truth.json {meta['oracle_file_sha256']}")
    t = GEN.load_evaluation_truth(case_directory)
    p, q, eta = (np.asarray(t[k], dtype=np.float64) for k in ("p", "q", "eta"))
    S = float(t["S"])
    if p.shape != q.shape or p.shape != eta.shape or not math.isfinite(S):
        raise RuntimeError(f"{case_directory}: malformed truth")
    return {"p": p, "q": q, "eta": eta, "S": S, "S_closed_form": float(t["S_closed_form"]), "truth_npz_sha256": digest,
            "truth_json_sha256": C.sha256_file(case_directory / C.TRUTH_DIR_NAME / "truth.json"), "family": meta["family"], "m": meta["m"],
            "rho": meta["rho"]}


def table_metrics(T: np.ndarray, truth: dict, counts: dict[str, O.Counts]) -> dict:
    T = np.asarray(T, dtype=np.float64)
    p, q, eta, S = truth["p"], truth["q"], truth["eta"], truth["S"]
    pop_mse = float(np.sum(0.5 * (p + q) * (T - eta) ** 2))
    j_pop = float(GEN.population_j(p, q, T))
    resid = S - j_pop - pop_mse
    out = {"pop_mse": pop_mse, "J_pop": j_pop, "S": S, "J_pop_minus_S": j_pop - S, "identity_residual": resid,
           "identity_check": "PASS" if abs(resid) <= C.IDENTITY_ATOL else "FAIL", "table_absmax": float(np.abs(T).max())}
    for role, cnt in counts.items():
        R = role.upper()
        uj, vj = O.empirical_j(T, cnt, "u"), O.empirical_j(T, cnt, "v")
        out[f"{R}_UJ"], out[f"{R}_VJ"], out[f"{R}_U_minus_V"] = uj, vj, uj - vj
    if "eval" in counts:
        out["EVAL_UJ_minus_S"] = out["EVAL_UJ"] - S
        out["EVAL_UJ_minus_J_pop"] = out["EVAL_UJ"] - j_pop
        out["EVAL_VJ_minus_S"] = out["EVAL_VJ"] - S
    return out


def head_gap(model, fit: SampleSet, m: int) -> dict:
    """MF fixed-feature empirical head problem on the full FIT batch (U mode) at the checkpoint state."""
    p0 = next(model.parameters())
    with torch.no_grad():
        Af, Bf = model.features(Mo.onehot(fit.x, m, dtype=p0.dtype, device=p0.device), Mo.onehot(fit.y, m, dtype=p0.dtype, device=p0.device))
        scene = O.mf_u_scene(Af, Bf)
        W = model.W.detach()
        head = MF.QuadraticHead([scene], weights=[1.0])
        g_ref = float(head.delta.abs().sum())
        gap_w = float(head.fw_gap(W))
        j_w = float(head.value_and_gradient(W)[0])
        head64 = MF.QuadraticHead([A.same_input_scene(scene, torch.float64)], weights=[1.0])
        W64 = W.to(torch.float64)
        g_ref64 = float(head64.delta.abs().sum())
        gap_w64 = float(head64.fw_gap(W64))
        j_w64 = float(head64.value_and_gradient(W64)[0])
    out = {"scope": "full FIT batch at the checkpoint features (post outer update), U mode, ridge 0", "gap_ref_zero": g_ref,
           "fw_gap_W": gap_w, "gap_relative_to_zero": (gap_w / g_ref) if g_ref > 0 else None, "J_head_W": j_w,
           "fp64": {"gap_ref_zero": g_ref64, "fw_gap_W": gap_w64, "J_head_W": j_w64}}
    if g_ref64 > 0:
        t0 = time.perf_counter()
        try:                                                         # diagnostic only: a solver error is recorded, not raised
            sol = MF.solve_box_head(head64, initial=W64, tolerance=max(1e-12 * g_ref64, 1e-300), max_iterations=HEAD_POLISH_ITERATIONS)
            out["fp64_polish"] = {"iterations": sol.iterations, "status": sol.status, "fw_gap_polished": sol.fw_gap, "J_head_polished": sol.j,
                                  "suboptimality_lower_bound": sol.j - j_w64, "suboptimality_upper_bound_fw_gap": gap_w64,
                                  "seconds": time.perf_counter() - t0}
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            out["fp64_polish"] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"[:500], "seconds": time.perf_counter() - t0}
    else:
        out["fp64_polish"] = {"status": "zero_reference_gap", "note": "Delta = 0: W = 0 is optimal for this fixed-feature head problem"}
    return out


def perm_variance(T: np.ndarray, sample: SampleSet, counts: O.Counts, data_root_seed: int, family: str, m: int, rho: float) -> dict:
    ss = C.stream(data_root_seed, "diag_perm", family, m, rho, C.ROLE_CODE[sample.role])
    stream = O.DerangementStream(C.stream_rng(ss), sample.n)
    sig = stream.draw(PERM_DIAG_R)
    est = O.perm_estimate_from_table(T, sample.x, sample.y, sig)
    exact = O.empirical_j(T, counts, "u")
    var1 = float(np.var(est, ddof=1))
    se = math.sqrt(var1 / len(est)) if var1 > 0 else 0.0
    g = est[: len(est) // PERM_DIAG_GROUP * PERM_DIAG_GROUP].reshape(-1, PERM_DIAG_GROUP).mean(1)
    return {"role": sample.role, "R": int(len(est)), "exact_UJ": exact, "perm1_mean": float(est.mean()), "perm1_var": var1, "perm1_sd": math.sqrt(var1),
            "mean_minus_exact": float(est.mean()) - exact, "z_mean_vs_exact": ((float(est.mean()) - exact) / se) if se > 0 else None,
            "perm8_var_implied": var1 / PERM_DIAG_GROUP, "perm8_var_grouped": float(np.var(g, ddof=1)) if len(g) > 1 else None,
            "perm8_groups": int(len(g)), "stream": C.stream_record(ss), "derangements": stream.record()}


def evaluate_unit(result: dict, unit_dir: Path, suite: Path, record: dict) -> dict:
    """Evaluate a trained unit (result.json content) from its checkpoints; returns the evaluation record (written by the caller)."""
    _outside_barrier("evaluate_unit")
    t0 = time.perf_counter()
    cfg = result["config"]
    cid, m = result["case_id"], int(cfg["m"])
    case_dir = suite / cid
    truth = load_truth(case_dir)
    if truth["m"] != m or truth["family"] != cfg["family"] or float(truth["rho"]) != float(cfg["rho"]):
        raise RuntimeError("truth / unit condition mismatch")
    shas = record["cases"][cid]["roles"]
    samples = {r: load_role(case_dir, r, m, shas[r]["sample_sha256"]) for r in ("fit", "cal", "eval")}
    counts = {r: O.Counts.from_xy(s.x, s.y, m) for r, s in samples.items()}
    blob = torch.load(unit_dir / "checkpoints.pt", map_location="cpu", weights_only=True)
    if blob["unit_id"] != result["unit_id"] or C.sha256_file(unit_dir / "checkpoints.pt") != result["artifacts"]["checkpoints_sha256"]:
        raise RuntimeError("checkpoint file does not belong to this unit result")
    model, _ = Mo.build_critic(cfg["method"], cfg["family"], m, cfg["rho"], cfg["K"], cfg["algo_seed"], dtype=getattr(torch, cfg["dtype"]))
    trace = {int(c["step"]): c for c in result["checkpoints"]}
    traj, tables, checks = [], {}, {"identity": "PASS"}
    readout_diff, readout_exact = 0.0, True
    if sorted(int(s) for s in blob["states"]) != sorted(trace):
        raise RuntimeError("checkpoint steps differ from the result trace")
    for step in sorted(int(s) for s in blob["states"]):
        model.load_state_dict(blob["states"][step])
        T = Mo.critic_table(model, m).double().numpy()
        tables[step] = T
        met = table_metrics(T, truth, counts)
        if met["identity_check"] != "PASS":
            checks["identity"] = "FAIL"
        for key in ("CAL_UJ", "FIT_UJ"):                           # integrity of the stored checkpoints vs the training-time readout
            readout_diff = max(readout_diff, abs(met[key] - trace[step][key]))
            readout_exact &= met[key] == trace[step][key]
        traj.append({"step": step, **{k: met[k] for k in ("pop_mse", "J_pop", "EVAL_UJ", "EVAL_VJ", "FIT_UJ", "CAL_UJ", "identity_residual")}})
    checks["cal_fit_readout_reproduced"] = "PASS" if readout_diff <= C.TOLERANCES["float32"][0] else "FAIL"
    sel = result["selection"]["own_cal"]
    step = int(sel["step"])
    model.load_state_dict(blob["states"][step])
    T = tables[step]
    selected = table_metrics(T, truth, counts)
    selected["perm_conditional_variance"] = {r: perm_variance(T, samples[r], counts[r], cfg["data_root_seed"], cfg["family"], m, cfg["rho"])
                                             for r in ("fit", "eval")}
    if cfg["method"] in C.MF_METHODS:
        selected["head_gap"] = {"status": "MEASURED", **head_gap(model, samples["fit"], m)}
    else:
        selected["head_gap"] = {"status": "NOT_APPLICABLE", "reason": "MLP critic: no fixed-feature relation head"}
    sat = {}
    for mode in ("u", "v"):
        Ts = O.saturated_table(counts["fit"], mode)
        sat[mode.upper()] = table_metrics(Ts, truth, counts)
        if sat[mode.upper()]["identity_check"] != "PASS":
            checks["identity"] = "FAIL"
    dominated = all(sat["U"]["FIT_UJ"] >= trace[s]["FIT_UJ"] - 1e-12 for s in trace)
    checks["saturated_FIT_U_optimum_dominates_checkpoints"] = "PASS" if dominated else "FAIL"
    return {"schema": "v10_t03.evaluation.v1", "unit_id": result["unit_id"], "case_id": cid, "method": cfg["method"],
            "block": cfg["block"], "data_root_seed": cfg["data_root_seed"], "algo_seed": cfg["algo_seed"],
            "truth": {k: truth[k] for k in ("S", "S_closed_form", "truth_npz_sha256", "truth_json_sha256")},
            "sample_sha256": {r: s.sample_sha256 for r, s in samples.items()}, "counts_sha256": {r: c.sha256() for r, c in counts.items()},
            "selected": {"rule": sel["rule"], "step": step, "scope": result["selection"]["scope"]}, "selected_metrics": selected,
            "trajectory_diagnostic": {"note": "every checkpoint, evaluator side; diagnostic only, never used for any selection", "rows": traj},
            "saturated_fit_table": {"note": "decision 9.5: FIT-count saturated table (empirical optimum of the full-table class on FIT); "
                                    "no truth in its construction; not a trained MF feature", **sat},
            "checks": checks, "readout_reproduction": {"exact": readout_exact, "max_abs_diff": readout_diff,
                                                       "tolerance": C.TOLERANCES["float32"][0]}, "status": "PASS" if all(v == "PASS" for v in checks.values()) else "FAIL",
            "seconds": time.perf_counter() - t0}
