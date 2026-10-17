"""V10 T03 training of one unit (decision 9.2): full-batch FIT, 2000 steps, a checkpoint every 250 steps, CAL U-J readout per checkpoint.

Optimiser (decision 9.2 / 8.1): AdamW (torch defaults betas (0.9, 0.999), eps 1e-8), one learning-rate candidate shared by every
optimiser-learned module; weight decay 1e-4 on every non-W parameter, 0 on W; ridge 0; clip_grad_norm 1.0 over every optimiser-trained
parameter (MF perm1 / M0: feature maps + W, then W projected onto [-1, 1]; M1: feature maps only).
  MF_VCS_perm1   one uniform derangement per step (shared 'perm' stream)                 t03_objectives.mf_derangement_terms
  MF_VCS_M0      complete U moments, package objective_terms (q_mode 'u')                t03_objectives.mf_u_moments
  MF_VCS_M1      src/v10/adapter.mf_objective('MF_M1', ...) = the protocol relative zero-head rule (decision 2.1, unchanged code):
                 per step ONE QuadraticHead snapshot, g_ref = ||Delta||_1, tau = 0.01 g_ref, package solve_box_head from the previous
                 step's W (first step W = 0) to tau or the cap (10 / 30 / 100), require_converged False, outer objective_terms on the
                 same tensors; W is never an optimiser parameter, the solved W is the next step's warm start and is saved with the maps.
  MLP_VCS_perm1  T = tanh f, one uniform derangement per step (the same 'perm' stream draw as MF_VCS_perm1)
  MLP_VCS_perm8  eight uniform derangements per step: row 0 = the 'perm' stream draw, rows 1..7 from the 'perm8_extra' stream
Checkpoint state = every critic parameter (MF: maps + W as one indivisible state).  The CAL readout is the complete empirical U-J of
the frozen critic table on the unit's own CAL counts (t03_objectives.empirical_j); FIT U-J the same on FIT.  This module runs only
inside the truth barrier (train_unit refuses otherwise) and never receives a path.
"""
from __future__ import annotations

import math
import resource
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from v10 import adapter as A

import t03_common as C
import t03_models as Mo
import t03_objectives as O
from t03_data import SampleSet

M1_LOG_FIELDS = ("m1_gap_ref_zero", "m1_gap_start", "m1_gap_final", "m1_gap_relative_to_zero", "m1_tolerance_effective",
                 "m1_iterations", "m1_stop_reason", "m1_converged", "m1_hvp_calls", "m1_momentum_restarts", "m1_moments_passes",
                 "m1_fp64_diag_hvp_calls", "m1_fp64_diag_moments_passes", "m1_hvp_calls_total", "m1_moments_passes_total",
                 "m1_W_start_is_zero", "m1_W_update_fro", "m1_W_update_absmax", "m1_W_final_absmax")
# m1_hvp_calls / m1_moments_passes = the working-dtype M1 work of the step; m1_fp64_diag_* = the same-input fp64 diagnostic (0 when it
# did not run); m1_*_total = all M1 work of the step (src/v10/adapter._m1_info at 278abe4, decision 2.3: every added pass is counted)


@dataclass
class UnitConfig:
    unit_id: str
    block: str
    method: str
    family: str
    m: int
    rho: float
    K: int | None
    lr: float
    cap: int | None
    data_root_seed: int
    algo_seed: int
    steps: int = C.STEPS
    ckpt_every: int = C.CKPT_EVERY
    dtype: str = "float32"
    device: str = "cpu"
    perm8_R: int = C.PERM8_R
    weight_decay: float = C.NON_W_WEIGHT_DECAY
    w_weight_decay: float = C.W_WEIGHT_DECAY
    clip_norm: float = C.CLIP_NORM
    selection_rule: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.method not in C.METHODS:
            raise ValueError(f"unknown method {self.method}")
        if (self.method in C.MF_METHODS) != (self.K is not None):
            raise ValueError("K is required for MF methods and must be None for MLP methods")
        if (self.method == "MF_VCS_M1") != (self.cap is not None):
            raise ValueError("cap is required for MF_VCS_M1 and must be None otherwise")
        if self.cap is not None and (isinstance(self.cap, bool) or int(self.cap) != self.cap or self.cap < 0):
            raise ValueError("cap must be a nonnegative integer")
        if self.steps < 1 or self.ckpt_every < 1 or self.steps % self.ckpt_every:
            raise ValueError("steps must be a positive multiple of ckpt_every")
        if not (math.isfinite(self.lr) and self.lr > 0):
            raise ValueError("lr must be finite and positive")

    @property
    def case_id(self) -> str:
        return C.case_id(self.family, self.m, self.rho)

    @property
    def ckpt_steps(self) -> tuple[int, ...]:
        return tuple(range(self.ckpt_every, self.steps + 1, self.ckpt_every))


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _norm(params) -> float:
    gs = [p.grad.detach().double().square().sum() for p in params if p.grad is not None]
    return float(torch.sqrt(sum(gs))) if gs else 0.0


def train_unit(cfg: UnitConfig, fit: SampleSet, cal: SampleSet) -> tuple[dict, dict, dict]:
    """Returns (record, per-step log as columns, checkpoints {step: state_dict}).  Inside the truth barrier only."""
    if not C.barrier_active():
        raise RuntimeError("train_unit runs only inside t03_common.truth_barrier (training never sees evaluator-only material)")
    if fit.role != "fit" or cal.role != "cal" or fit.case_id != cfg.case_id or cal.case_id != cfg.case_id:
        raise ValueError("train_unit needs the FIT and CAL samples of the unit's own case")
    t_start, cpu_start = time.perf_counter(), time.process_time()
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    device, dtype = torch.device(cfg.device), getattr(torch, cfg.dtype)
    kind = Mo.critic_kind(cfg.method)
    model, init = Mo.build_critic(cfg.method, cfg.family, cfg.m, cfg.rho, cfg.K, cfg.algo_seed, dtype=dtype, device=device)
    X = Mo.onehot(fit.x, cfg.m, dtype=dtype, device=device)
    Y = Mo.onehot(fit.y, cfg.m, dtype=dtype, device=device)
    fit_counts, cal_counts = O.Counts.from_xy(fit.x, fit.y, cfg.m), O.Counts.from_xy(cal.x, cal.y, cfg.m)
    # -- optimiser: one lr for every optimiser-learned module; W decay 0; M1's W is solved, never optimised
    if kind == "MF":
        fmap = list(model.left.parameters()) + list(model.right.parameters())
        groups = [{"params": fmap, "lr": cfg.lr, "weight_decay": cfg.weight_decay, "name": "feature_maps"}]
        if cfg.method == "MF_VCS_M1":
            model.W.requires_grad_(False)
        else:
            groups.append({"params": [model.W], "lr": cfg.lr, "weight_decay": cfg.w_weight_decay, "name": "W"})
    else:
        fmap = list(model.parameters())
        groups = [{"params": fmap, "lr": cfg.lr, "weight_decay": cfg.weight_decay, "name": "mlp_critic"}]
    opt = torch.optim.AdamW(groups)
    clip_params = [p for g in groups for p in g["params"]]
    opt_record = {"name": "AdamW", "defaults": {k: v for k, v in opt.defaults.items() if k not in ("lr", "weight_decay")}, "groups": [{"name": g["name"], "lr": g["lr"], "weight_decay": g["weight_decay"],
                  "n_params": sum(p.numel() for p in g["params"])} for g in opt.param_groups], "clip_norm": cfg.clip_norm,
                  "clip_scope": "every optimiser-trained parameter", "W_projection": "clamp to [-1, 1] after every step" if
                  cfg.method in ("MF_VCS_perm1", "MF_VCS_M0") else ("solved head (warm start), not optimised" if kind == "MF" else None)}
    # -- permutation streams (shared across perm methods of the same algo seed and condition)
    perm_ss = C.stream(cfg.algo_seed, "perm", cfg.family, cfg.m, cfg.rho)
    perm_x_ss = C.stream(cfg.algo_seed, "perm8_extra", cfg.family, cfg.m, cfg.rho)
    uses_perm = cfg.method in ("MF_VCS_perm1", "MLP_VCS_perm1", "MLP_VCS_perm8")
    perm = O.DerangementStream(C.stream_rng(perm_ss), fit.n) if uses_perm else None
    perm_x = O.DerangementStream(C.stream_rng(perm_x_ss), fit.n) if cfg.method == "MLP_VCS_perm8" else None
    m1cfg = A.M1Config(max_iterations=int(cfg.cap)) if cfg.method == "MF_VCS_M1" else None
    log: dict[str, list] = {k: [] for k in ("step", "J", "loss", "grad_norm_preclip", "fmap_grad_norm", "W_grad_norm", "w_absmax_preprojection",
                                             "w_absmax", "t_objective", "t_backward", "t_update", "t_step")}
    if m1cfg is not None:
        log.update({k: [] for k in M1_LOG_FIELDS + ("m1_fp64_triggered", "m1_fp64_decisions_agree")})
    ckpts, ckpt_trace = {}, []
    t_ckpt = t_read = 0.0
    t_loop0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        _sync(device)
        t0 = time.perf_counter()
        info, sig = {}, None
        if uses_perm:                                                  # sampling is part of the method's measured step cost
            rows = perm.draw(1)
            if cfg.method == "MLP_VCS_perm8":
                rows = np.concatenate([rows, perm_x.draw(cfg.perm8_R - 1)])
            if step == 1 and not O.is_derangement_matrix(rows, fit.n):
                raise RuntimeError("derangement sampler returned a non-derangement")
            sig = torch.as_tensor(rows, device=device)
        if kind == "MF":
            Afeat, Bfeat = model.features(X, Y)
            if cfg.method == "MF_VCS_perm1":
                terms = O.mf_derangement_terms(Afeat, Bfeat, model.W, sig)
            elif cfg.method == "MF_VCS_M0":
                terms = O.mf_u_moments(Afeat, Bfeat, model.W, check_w=(step == 1))
            else:
                terms, info = A.mf_objective("MF_M1", [O.mf_u_scene(Afeat, Bfeat)], model.W, [1.0], m1=m1cfg)
        else:
            terms = O.mlp_derangement_terms(model, X, Y, sig)
        loss = -terms.f
        _sync(device)
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        _sync(device)
        t2 = time.perf_counter()
        fmap_norm = _norm(fmap)
        w_norm = _norm([model.W]) if kind == "MF" and cfg.method != "MF_VCS_M1" else None
        grad_norm = float(torch.nn.utils.clip_grad_norm_(clip_params, cfg.clip_norm))
        opt.step()
        w_pre = w_abs = None
        with torch.no_grad():
            if cfg.method == "MF_VCS_M1":
                model.W.copy_(info["m1_W"])                            # warm start of the next step; saved with the maps
                w_abs = float(model.W.abs().max())
            elif kind == "MF":
                w_pre = float(model.W.abs().max())
                model.project_head_()
                w_abs = float(model.W.abs().max())
        _sync(device)
        t3 = time.perf_counter()
        J = float(terms.j.detach())
        if not math.isfinite(J) or not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError(f"non-finite objective or parameters at step {step} (J={J})")
        for k, v in (("step", step), ("J", J), ("loss", float(loss.detach())), ("grad_norm_preclip", grad_norm), ("fmap_grad_norm", fmap_norm),
                     ("W_grad_norm", w_norm), ("w_absmax_preprojection", w_pre), ("w_absmax", w_abs), ("t_objective", t1 - t0),
                     ("t_backward", t2 - t1), ("t_update", t3 - t2), ("t_step", t3 - t0)):
            log[k].append(v)
        if m1cfg is not None:
            for k in M1_LOG_FIELDS:
                log[k].append(info[k])
            diag = info["m1_fp64_diagnostic"]
            log["m1_fp64_triggered"].append(bool(diag.get("triggered")))
            log["m1_fp64_decisions_agree"].append(diag.get("decisions_agree"))
        if step % cfg.ckpt_every == 0:
            t4 = time.perf_counter()
            ckpts[step] = Mo.clone_state(model)
            t5 = time.perf_counter()
            T = Mo.critic_table(model, cfg.m).double().cpu().numpy()
            fit_uj, cal_uj = O.empirical_j(T, fit_counts, "u"), O.empirical_j(T, cal_counts, "u")
            t6 = time.perf_counter()
            if not (math.isfinite(fit_uj) and math.isfinite(cal_uj)):
                raise FloatingPointError(f"non-finite readout at checkpoint {step}")
            ckpt_trace.append({"step": step, "FIT_UJ": fit_uj, "CAL_UJ": cal_uj, "train_J_step": J, "state_sha256": Mo.state_hash(model),
                               "table_absmax": float(np.abs(T).max()), "w_absmax": w_abs})
            t_ckpt += t5 - t4
            t_read += t6 - t5
    t_loop = time.perf_counter() - t_loop0
    steps_t = np.asarray(log["t_step"])
    cost = {"wall_seconds_train_unit": time.perf_counter() - t_start, "cpu_seconds_train_unit": time.process_time() - cpu_start,
            "wall_seconds_loop": t_loop, "wall_seconds_checkpoint_copy": t_ckpt, "wall_seconds_cal_fit_readout": t_read,
            "step_seconds": {"sum": float(steps_t.sum()), "median": float(np.median(steps_t)), "p10": float(np.percentile(steps_t, 10)),
                             "p90": float(np.percentile(steps_t, 90)), "max": float(steps_t.max()), "first": float(steps_t[0])},
            "phase_seconds": {k: float(np.sum(log[k])) for k in ("t_objective", "t_backward", "t_update")},
            "peak_rss_kib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "peak_rss_kib_at_start": rss_start,
            "torch_threads": torch.get_num_threads(), "device": str(device), "dtype": cfg.dtype}
    if device.type == "cuda":
        cost["cuda_max_memory_allocated"] = torch.cuda.max_memory_allocated(device)
    record = {"config": asdict(cfg), "case_id": cfg.case_id, "critic": {**init, "architecture": _architecture(cfg)}, "optimizer": opt_record,
              "streams": {"perm": C.stream_record(perm_ss) if uses_perm else None,
                          "perm8_extra": C.stream_record(perm_x_ss) if perm_x is not None else None},
              "derangements": {"perm": perm.record() if perm else None, "perm8_extra": perm_x.record() if perm_x else None,
                               "sampler": "rejection of uniform permutations (numpy Generator.permutation) with any fixed point"},
              "counts_sha256": {"fit": fit_counts.sha256(), "cal": cal_counts.sha256()},
              "checkpoints": ckpt_trace, "final_state_sha256": Mo.state_hash(model), "cost": cost,
              "train_J_last": log["J"][-1], "steps_completed": cfg.steps}
    if m1cfg is not None:
        record["m1"] = _m1_summary(log, m1cfg)
    return record, log, ckpts


def _architecture(cfg: UnitConfig) -> dict:
    if cfg.method in C.MF_METHODS:
        return {"class": "v10.mf_vcs.SimplexBilinearCritic", "x_dim": cfg.m, "y_dim": cfg.m, "r": cfg.K, "s": cfg.K,
                "temperature": C.TEMPERATURE, "features": "Linear + softmax per side", "head": "W in [-1, 1]^{K x K}, zeros at start",
                "input": "one-hot integer observations", "output": "T = a(x)^T W b(y) (no tanh)", "q_mode": "u"}
    return {"class": "t03_models.MLPCritic", "layers": f"Linear({2 * cfg.m},{C.MLP_HIDDEN})-GELU-Linear({C.MLP_HIDDEN},{C.MLP_HIDDEN})-GELU-"
            f"Linear({C.MLP_HIDDEN},1)", "input": "concat(one-hot x, one-hot y)", "output": "T = tanh f",
            "negatives_per_step": cfg.perm8_R if cfg.method == "MLP_VCS_perm8" else 1, "q_mode": "u (uniform derangements)"}


def _m1_summary(log: dict, m1cfg) -> dict:
    it = np.asarray(log["m1_iterations"])
    reasons = {r: int(sum(1 for s in log["m1_stop_reason"] if s == r)) for r in A.M1_STOP_REASONS}
    rel = np.asarray([v for v in log["m1_gap_relative_to_zero"] if v is not None], dtype=np.float64)
    return {"rule": m1cfg.rule, "relative_tolerance": m1cfg.relative_tolerance, "cap": m1cfg.max_iterations, "require_converged": m1cfg.require_converged,
            "ridge": m1cfg.ridge, "gap_scope": "inner_pre_outer_batch (current FIT features before the outer update of the step)",
            "steps": len(it), "stop_reasons": reasons, "converged_fraction": float(np.mean(log["m1_converged"])),
            "iterations": {"sum": int(it.sum()), "median": float(np.median(it)), "max": int(it.max()), "min": int(it.min())},
            "hvp_calls_sum": int(np.sum(log["m1_hvp_calls"])), "moments_passes_sum": int(np.sum(log["m1_moments_passes"])),
            "fp64_diag_hvp_calls_sum": int(np.sum(log["m1_fp64_diag_hvp_calls"])),
            "fp64_diag_moments_passes_sum": int(np.sum(log["m1_fp64_diag_moments_passes"])),
            "hvp_calls_total_sum": int(np.sum(log["m1_hvp_calls_total"])), "moments_passes_total_sum": int(np.sum(log["m1_moments_passes_total"])),
            "momentum_restarts_sum": int(np.sum(log["m1_momentum_restarts"])),
            "gap_relative_to_zero": {"median": float(np.median(rel)) if rel.size else None, "max": float(rel.max()) if rel.size else None,
                                     "last": log["m1_gap_relative_to_zero"][-1]},
            "gap_ref_zero": {"first": log["m1_gap_ref_zero"][0], "last": log["m1_gap_ref_zero"][-1], "min": float(np.min(log["m1_gap_ref_zero"]))},
            "first_step_W_start_is_zero": bool(log["m1_W_start_is_zero"][0]),
            "fp64_diagnostic": {"triggered_steps": int(np.sum(log["m1_fp64_triggered"])),
                                "disagreements": int(sum(1 for t, a in zip(log["m1_fp64_triggered"], log["m1_fp64_decisions_agree"]) if t and a is False))}}
