"""V10 T02 - computation experiment on the actual device (package V10_DESIGN 6.2, configs/V10_PROTOCOL.json "compute").

Parts (python scripts/v10/t02_timing.py run --device cuda --parts head,head_extra,train,register; CPU: --parts head):
  head        the package core/benchmark_torch.py, unchanged, with --full: N = 128 / 512 / 2048 / 8192 positions per scene,
              K = 8 / 16 / 32, methods perm1 (R = 1), perm8 (R = 8), dense, checkpointed (recomputing) blocked dense, moments;
              the same A / B / W in a cell, V target, warmup, CUDA sync around every call, forward + autograd.grad(A, B, W),
              peak allocated / reserved, OOM recorded and continued.  float32 (primary) and float64 (reference); quantiles
              computed here from the per-repeat times the package script records; repeats = the times actually recorded.
  head_extra  on the identical A / B / W draws (the package generator sequence replayed, verified cell by cell against the
              package report's reference J): row-blocked dense WITHOUT checkpointing (all block graphs retained) under the same
              timing protocol, and the conditional variance of the perm1 / perm8 J and W-gradient around exact V at fixed
              A / B / W (256 / 64 independent draws).
  train       encoder-inclusive optimiser steps on real BF-SHG FIT batches (host batch sampler, host encoders; 16 FIT cores
              resident = one host batch, so every step uses 16 crops as the host): MF_perm1 / MF_M0 / MF_M1 (K 8 / 16 / 32);
              M1 rule 'absolute' with a binding iteration cap (tolerance 1e-300: budgets 10 / 30 / 100 always run in full; cap
              references) plus one record row at the absolute target 1e-4 (degenerate from W = 0 at host init, excluded from
              BUDGET); the nine MF_VCS_M1 CAL candidates under the protocol rule (owner decision 2026-10-17 2.1: relative
              zero-head gap 1%, caps 10 / 30 / 100, lr 1e-4 / 3e-4 / 1e-3 tied across encoders + feature maps); MLP_perm1,
              MLP_perm8, MLP_dense (full graph and checkpointed blocks).  MF W-box / permutation checks run on the first step
              only (the validated-loop rule of the package); K16 perm1 / M0 rows with checks on every step quantify that
              overhead.  Synchronised phases: batch, descriptors, scenes (MF feature maps + partition / SceneBatch validation),
              objective, backward, update; CUDA host-device synchronisations per step counted with torch.cuda.set_sync_debug_mode.
              --train-scope m1_relative (with --parts train): ONLY the nine M1 relative-rule candidates, written to
              v10/T02/T02_M1_RELATIVE_TIMING_<dev>.json / T02_M1_RELATIVE_LONG_<dev>.csv; v10/T02/T02_TIMING / T02_LONG (phase-1
              job 1043695) stay untouched and every other row is taken from them (decision 2026-10-17 2.3).
  register    full registrations with the unchanged host searcher Q.optimise_rigid (21 angles x 508 translations, top 3,
              100 Adam steps) on 3 DEV cores (perturbations from the frozen rule Q.perturbations_p2, no truth file read) with
              fixed, UNTRAINED models: timing only - no error, F_true or any other truth-dependent quantity is computed.
              MF moments (K 8 / 16 / 32), MF perm1 (K 16), host MLP perm1 energy.
  budget      (no GPU) v10/T02/BUDGET.json for the decision-8 grid (owner decision 2026-10-17 section 8: 18 CAL candidates, per
              family <= 9 candidates and <= 8 L40S GPU-h, 4 families x 3 formal seeds) from the kept rows (T02_TIMING_cuda.json +
              T02_LONG_cuda.csv) and the M1 relative-rule re-measurement (T02_M1_RELATIVE_*); also the decision's old-timing
              projection recomputed as a formula check, and v10/T02/UNIT_LEDGER.csv (package units) as before.  Status
              MEASURED_PROJECTION / PROVISIONAL (reasons listed) / NOT_RUN (no M1 re-measurement) - see budget().
Outputs: v10/T02/T02_TIMING_<cpu|cuda>.json, v10/T02/T02_LONG_<cpu|cuda>.csv (COSTS_TEMPLATE columns first) in the repository
(scope m1_relative: v10/T02/T02_M1_RELATIVE_TIMING_<dev>.json / T02_M1_RELATIVE_LONG_<dev>.csv); raw files and logs under
<out-root>/T02/runs/<device>_<job>[_m1_relative]/.  Data: 16 FIT cores (train) and 3 DEV cores (register), DataGuard.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from v10 import adapter as A
from v10 import mf_vcs as MF

SEED = 10101                     # V10_PROTOCOL compute.seed
GRID = [(n, k) for n in (128, 512, 2048, 8192) for k in (8, 16, 32)]
HEAD_WARMUP, HEAD_REPEATS, HEAD_BLOCK = 5, 30, 128
TRAIN_WARMUP, TRAIN_TIMED, TRAIN_PHASE = 10, 50, 10
REG_CORES, REG_PERTS = 3, (0, 1)
REPO_T02 = A.REPO / "v10" / "T02"
METHOD_MAP = {"perm1": "permutation1", "perm8": "permutation8", "dense": "dense", "dense_block_checkpoint": "blocked_dense", "moments": "moments"}
TEMPLATE_FIELDS = ["unit_id", "dataset", "method", "phase", "device", "dtype", "n_positions", "simplex_width", "elapsed_seconds",
                   "peak_allocated_bytes", "peak_reserved_bytes", "encoder_included", "solver_iterations", "solver_gap", "status", "notes"]
EXTRA_FIELDS = ["layer", "config", "R_permutations", "warmup", "repeats", "median_ms", "p10_ms", "p25_ms", "p75_ms", "p90_ms", "min_ms",
                "max_ms", "mean_ms", "value_abs_err_vs_V", "grad_abs_err_A", "grad_abs_err_B", "grad_abs_err_W", "block_size",
                "graph_retained", "backward_recomputation", "replay_verified", "n_sc", "n_pos", "checks", "m1_tolerance",
                "m1_max_iterations", "m1_iterations_median", "m1_iterations_max", "m1_gap_median", "m1_gap_max", "m1_converged_fraction",
                "m1_gap_initial_median", "m1_hvp_calls_median", "m1_momentum_restarts_median", "m1_moment_passes_per_batch",
                "m1_w_absmax_last", "degenerate", "J_first", "J_last", "phase_batch_ms", "phase_descriptors_ms", "phase_scenes_ms",
                "phase_objective_ms", "phase_backward_ms", "phase_update_ms", "syncs_per_step", "energy", "core", "perturbation_index",
                "warmup_row", "grid_seconds", "refine_and_final_seconds", "exact_energy_calls", "n_grid", "n_draws", "J_V",
                "cond_mean_minus_V", "cond_sd", "cond_grad_W_rel_err_mean", "source_file",
                # decision 2026-10-17 (M1 relative zero-head rule; appended so that the phase-1 columns keep their positions)
                "lr", "m1_rule", "m1_relative_tolerance", "m1_cap_hit_fraction", "m1_stop_reasons", "m1_gap_ref_zero_first",
                "m1_gap_ref_zero_median", "m1_gap_start_median", "m1_gap_relative_to_zero_median", "m1_gap_relative_to_zero_max",
                "m1_tolerance_effective_median", "m1_W_update_fro_median", "m1_zero_reference_count", "m1_fp64_diag_triggered_fraction",
                "m1_fp64_diag_decisions_agree_fraction", "m1_iterations_first", "m1_w_absmax_first", "encoder_grad_norm_first",
                "m1_iterations_median_all_steps", "m1_cap_hit_fraction_all_steps", "n_steps_recorded"]
CAP = A.M1_CAP_BINDING_TOLERANCE
LR_CANDIDATES = (1e-4, 3e-4, 1e-3)    # decision 2026-10-17 8.1: one learning rate tied across the optimiser-learned modules
TRAIN_CONFIGS = [   # the phase-1 list; its M1 rows are rule 'absolute' (cap-binding cost references and the 1e-4 record row)
    {"method": "MF_perm1", "K": 16}, {"method": "MF_M0", "K": 16},
    {"method": "MF_M1", "K": 16, "m1_rule": "absolute", "m1_max_iterations": 30, "m1_tolerance": CAP},
    {"method": "MF_M1", "K": 16, "m1_rule": "absolute", "m1_max_iterations": 10, "m1_tolerance": CAP},
    {"method": "MF_M1", "K": 16, "m1_rule": "absolute", "m1_max_iterations": 100, "m1_tolerance": CAP},
    {"method": "MF_M1", "K": 16, "m1_rule": "absolute", "m1_max_iterations": 30, "m1_tolerance": A.M1_PROTOCOL_TOLERANCE, "record_only": True},
    {"method": "MF_perm1", "K": 8}, {"method": "MF_M0", "K": 8},
    {"method": "MF_M1", "K": 8, "m1_rule": "absolute", "m1_max_iterations": 30, "m1_tolerance": CAP},
    {"method": "MF_perm1", "K": 32}, {"method": "MF_M0", "K": 32},
    {"method": "MF_M1", "K": 32, "m1_rule": "absolute", "m1_max_iterations": 30, "m1_tolerance": CAP},
    {"method": "MF_perm1", "K": 16, "checks": "every_step"}, {"method": "MF_M0", "K": 16, "checks": "every_step"},
    {"method": "MLP_perm1"}, {"method": "MLP_perm8"}, {"method": "MLP_dense"},
    {"method": "MLP_dense", "dense_block": 32, "dense_checkpoint": True},
]
# decision 2026-10-17 2.1 / 8.1: the nine MF_VCS_M1 CAL candidates (K16, relative zero-head rule) in the frozen order
TRAIN_CONFIGS_M1_RELATIVE = [{"method": "MF_M1", "K": 16, "m1_rule": "relative_zero_head", "m1_max_iterations": cap, "lr": lr}
                             for lr in LR_CANDIDATES for cap in A.M1_CAL_CAPS]
TRAIN_SCOPES = {"all": TRAIN_CONFIGS + TRAIN_CONFIGS_M1_RELATIVE, "m1_relative": TRAIN_CONFIGS_M1_RELATIVE}
SCOPE_PREFIX = {"all": "T02", "m1_relative": "T02_M1_RELATIVE"}
M1_RECORD_NOTE = ("RECORD of the pre-decision setting, not an M1 cost: rule 'absolute', gap target 1e-4 from the package start W = 0; the "
                  "first FW gap equals ||Delta||_1 (m1_gap_initial); when it is below 1e-4 the head stops at iteration 0, W stays 0 and the "
                  "outer objective and its gradient are exactly 0 (degenerate = W stayed 0 on every timed step); excluded from BUDGET")
M1_RELATIVE_NOTE = ("M1 relative zero-head rule (decision 2026-10-17 2.1): per batch stop at G_b(W) <= 0.01 G_b(0) of the batch's own "
                    "snapshot, else at the cap (approximate head, true gap logged); warm start from the previous batch, first W = 0; "
                    "require_converged False; learning rate tied across encoders + feature maps; fp64 same-input diagnostic on batches "
                    "with G_b(0) <= 1e4 x 2 eps (inside the timed step); cost = the measured step under this rule over the T02 window")
REG_ENERGIES = [("MF_moments", 16), ("MF_perm1", 16), ("MF_moments", 8), ("MF_moments", 32), ("MLP_perm1_host", None)]


def quantiles(ms) -> dict:
    a = np.asarray(ms, dtype=np.float64)
    if a.size == 0:
        return {}
    return {"median_ms": float(np.median(a)), "p10_ms": float(np.percentile(a, 10)), "p25_ms": float(np.percentile(a, 25)),
            "p75_ms": float(np.percentile(a, 75)), "p90_ms": float(np.percentile(a, 90)), "min_ms": float(a.min()),
            "max_ms": float(a.max()), "mean_ms": float(a.mean())}


def is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg or "can't allocate memory" in msg


def release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_label(cfgd: dict) -> str:
    lab = cfgd["method"] + (f"_K{cfgd['K']}" if "K" in cfgd else "") + (f"_it{cfgd['m1_max_iterations']}" if "m1_max_iterations" in cfgd else "")
    if cfgd.get("m1_rule") == "relative_zero_head":
        lab += f"_rel{A.M1_RELATIVE_TOLERANCE:.0e}"
    elif "m1_tolerance" in cfgd:
        lab += "_capbound" if cfgd["m1_tolerance"] == CAP else f"_tol{cfgd['m1_tolerance']:.0e}"
    if "lr" in cfgd:
        lab += f"_lr{cfgd['lr']:.0e}"
    if cfgd.get("checks") == "every_step":
        lab += "_checks_every_step"
    if cfgd.get("dense_block"):
        lab += "_block%d_ckpt" % cfgd["dense_block"]
    return lab


def count_syncs(fn, device):
    """Host-device synchronisations during fn() (CUDA only), counted as the warnings of torch.cuda.set_sync_debug_mode('warn')."""
    if torch.device(device).type != "cuda":
        return None
    import warnings
    A.synchronize(device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return sum(1 for w in caught if "synchroniz" in str(w.message).lower())


def ledger_units() -> dict:
    rows = list(csv.DictReader((A.PACKAGE / "configs" / "EXPERIMENT_UNITS.csv").read_text().splitlines()))
    return {(r["method"], int(r["n_positions"]), int(r["simplex_width"])): r["unit_id"] for r in rows if r["stage"] == "T02"}


# ------------------------------------------------------------------------------------------------------------ head (package benchmark)
def part_head(device, run_dir: Path, rows: list, info: dict, gpu: str) -> str:
    bdir = run_dir / "benchmark_torch"
    bdir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((A.PACKAGE / "MANIFEST_SHA256.json").read_text())
    for src, key in ((A.REPO / "src" / "v10" / "mf_vcs.py", "core/mf_vcs.py"), (A.PACKAGE / "core" / "benchmark_torch.py", "core/benchmark_torch.py")):
        sha = A.sha256_file(src)
        if sha != manifest[key]:
            raise RuntimeError(f"{src}: sha256 {sha} != package {manifest[key]}")
        shutil.copy2(src, bdir / Path(key).name)
    units = ledger_units()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    status = "PASS"
    for dt in ("float32", "float64"):
        out = bdir / f"BENCHMARK_TORCH_{torch.device(device).type}_{dt}_full.json"
        cmd = [sys.executable, "benchmark_torch.py", "--device", device, "--dtype", dt, "--full", "--warmup", str(HEAD_WARMUP),
               "--repeats", str(HEAD_REPEATS), "--block-size", str(HEAD_BLOCK), "--seed", str(SEED), "--output", str(out)]
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=bdir, capture_output=True, text=True, env=env, timeout=5400)
        (bdir / f"{dt}_stdout.log").write_text(proc.stdout)
        (bdir / f"{dt}_stderr.log").write_text(proc.stderr)
        data = json.loads(out.read_text()) if out.exists() else None
        info[dt] = {"command": " ".join(cmd), "exit_code": proc.returncode, "seconds": time.time() - t0, "report": str(out),
                    "report_sha256": A.sha256_file(out) if out.exists() else None, "package_status": (data or {}).get("status"),
                    "oom_count": (data or {}).get("oom_count"), "stderr_tail": proc.stderr[-3000:]}
        if data is None or proc.returncode not in (0, 2):
            status = "FAIL"
            continue
        if data.get("status") != "completed":
            status = "FAIL"
        for r in data["rows"]:
            m = METHOD_MAP.get(r["method"], r["method"])
            times = r.get("forward_backward_ms_all", []) if r["status"] == "completed" else []
            q = quantiles(times)
            ge = r.get("gradient_max_absolute_errors_vs_V", {}) or {}
            uid = units.get((m, r["n"], r["rank"]), "")
            rows.append({"layer": "head", "unit_id": uid if dt == "float32" else (uid + ":float64_reference" if uid else ""),
                         "dataset": "fixed_simplex_features", "method": m, "phase": "head_forward_backward", "device": gpu, "dtype": dt,
                         "n_positions": r["n"], "simplex_width": r["rank"], "R_permutations": {"perm1": 1, "perm8": 8}.get(r["method"]),
                         "elapsed_seconds": q.get("median_ms", float("nan")) / 1000 if q else None,
                         "peak_allocated_bytes": r.get("cuda_peak_allocated_bytes"), "peak_reserved_bytes": r.get("cuda_peak_reserved_bytes"),
                         "encoder_included": False, "status": r["status"], "warmup": HEAD_WARMUP if times else 0, "repeats": len(times), **q,
                         "value_abs_err_vs_V": r.get("value_absolute_error_vs_V"), "grad_abs_err_A": ge.get("A"), "grad_abs_err_B": ge.get("B"),
                         "grad_abs_err_W": ge.get("W"), "block_size": r.get("block_size"),
                         "graph_retained": {"dense": True, "dense_block_checkpoint": False}.get(r["method"]),
                         "backward_recomputation": True if r["method"] == "dense_block_checkpoint" else None,
                         "notes": (r.get("objective_comparison") or "") + (f"; {r.get('exception_type')}: {r.get('message', '')[:200]}" if r["status"] in ("oom", "error") else ""),
                         "source_file": out.name})
    return status


# ------------------------------------------------------------------------------------------------------------ head extensions
def part_head_extra(device, run_dir: Path, rows: list, info: dict, gpu: str) -> str:
    dev, dtype = torch.device(device), torch.float32
    generator = torch.Generator(device=dev).manual_seed(SEED)      # replays the package benchmark's draw sequence (float32 run)
    sync = lambda: A.synchronize(dev)                                # noqa: E731
    status = "PASS"
    pkg = run_dir / "benchmark_torch" / f"BENCHMARK_TORCH_{dev.type}_float32_full.json"
    pkg_ref = {}
    if pkg.exists():
        for r in json.loads(pkg.read_text())["rows"]:
            if r.get("reference_J") is not None:
                pkg_ref.setdefault((r["n"], r["rank"]), r["reference_J"])
    info["replay_reference"] = str(pkg) if pkg.exists() else "package float32 report missing: replay not verified"
    mismatches = []
    for n, k in GRID:
        a = torch.randn(n, k, device=dev, dtype=dtype, generator=generator).softmax(-1).detach().requires_grad_()
        b = torch.randn(n, k, device=dev, dtype=dtype, generator=generator).softmax(-1).detach().requires_grad_()
        w = torch.empty(k, k, device=dev, dtype=dtype).uniform_(-.8, .8, generator=generator).requires_grad_()
        scenes = (MF.SceneBatch(a, b, "v"),)
        MF.draw_ordinary_permutations(scenes, 8, generator=generator)   # consumed as in the package (keeps the next cell identical)
        leaves = (a, b, w)
        ref = MF.objective_terms(scenes, w, weights=[1.0])
        ref_j, ref_g = float(ref.j), [g.detach().clone() for g in torch.autograd.grad(ref.j, leaves)]
        replay_ok = None if (n, k) not in pkg_ref else bool(pkg_ref[(n, k)] == ref_j)       # bitwise: same draws, same kernels
        if replay_ok is False:
            mismatches.append({"n": n, "k": k, "package_reference_J": pkg_ref[(n, k)], "replayed_J": ref_j})
            status = "FAIL"
        # (1) row-blocked dense, no checkpointing: every block's score graph is retained until backward
        base = {"layer": "head_extra", "dataset": "fixed_simplex_features", "phase": "head_forward_backward", "device": gpu, "dtype": "float32",
                "n_positions": n, "simplex_width": k, "encoder_included": False, "replay_verified": replay_ok,
                "source_file": "replayed package draws (seed %d)" % SEED}
        row = {**base, "method": "blocked_dense_graph_retained", "block_size": min(n, HEAD_BLOCK), "graph_retained": True,
               "backward_recomputation": False, "warmup": 0, "repeats": 0}
        def fun(scenes=scenes, w=w, n=n):
            return MF.dense_objective_terms(scenes, w, weights=[1.0], chunk_size=min(n, HEAD_BLOCK), checkpoint_blocks=False, check_w=False).j
        try:
            v = fun()
            g = torch.autograd.grad(v, leaves)
            verr = abs(float(v) - ref_j)
            gerr = [float((x - y).abs().max()) for x, y in zip(g, ref_g)]
            del v, g
            row.update({"value_abs_err_vs_V": verr, "grad_abs_err_A": gerr[0], "grad_abs_err_B": gerr[1], "grad_abs_err_W": gerr[2]})
            if verr > 3e-5 or max(gerr) > 3e-5:                          # the package benchmark's float32 correctness limit
                row.update({"status": "correctness_failed"})
                status = "FAIL"
            else:
                for _ in range(HEAD_WARMUP):
                    v = fun()
                    torch.autograd.grad(v, leaves)
                sync()
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                ms = []
                for _ in range(HEAD_REPEATS):
                    sync()
                    t0 = time.perf_counter()
                    v = fun()
                    torch.autograd.grad(v, leaves)
                    sync()
                    ms.append(1000 * (time.perf_counter() - t0))
                    del v
                q = quantiles(ms)
                row.update({"status": "completed", **q, "elapsed_seconds": q["median_ms"] / 1000, "warmup": HEAD_WARMUP, "repeats": len(ms),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(dev) if dev.type == "cuda" else None,
                            "notes": "blocked dense, autograd keeps all block score graphs (no recomputation)"})
        except Exception as e:  # noqa: BLE001
            row.update({"status": "oom" if is_oom(e) else "error", "notes": f"{type(e).__name__}: {str(e)[:300]}"})
            status = status if is_oom(e) else "FAIL"
        rows.append(row)
        release()
        # (2) conditional permutation variance at fixed A / B / W (sampling error of perm1 / perm8 around exact V)
        g2 = torch.Generator(device=dev).manual_seed(A.Q.seed_of(f"V10|T02|permvar|{n}|{k}"))
        for R, draws in ((1, 256), (8, 64)):
            js, gerrs = [], []
            for r in range(draws):
                perms = torch.stack([torch.randperm(n, device=dev, generator=g2) for _ in range(R)])
                with torch.set_grad_enabled(r < 32):
                    t = MF.sampled_permutation_terms(scenes, w, weights=[1.0], permutations=(perms,), check_w=False, check_permutations=False)
                    js.append(float(t.j))
                    if r < 32:
                        gw = torch.autograd.grad(t.j, w)[0]
                        gerrs.append(float((gw - ref_g[2]).norm() / ref_g[2].norm()))
            js = np.asarray(js)
            rows.append({**base, "layer": "perm_conditional_variance", "method": f"permutation{R}", "R_permutations": R, "status": "completed",
                         "n_draws": draws, "J_V": ref_j, "cond_mean_minus_V": float(js.mean() - ref_j), "cond_sd": float(js.std(ddof=1)),
                         "cond_grad_W_rel_err_mean": float(np.mean(gerrs)), "notes": "fixed A/B/W; independent ordinary permutations; "
                         "sampling deviation from exact V, not an estimator of independent-sample variance"})
        fun = a = b = w = scenes = leaves = ref = ref_g = None                # free the cell before the next one
        release()
        print(json.dumps({"head_extra": [n, k], "status": rows[-3]["status"]}), flush=True)
    info["generator_replay"] = "torch generator seeded %d, draw order a, b, w, 8 permutations per cell (package benchmark order)" % SEED
    info["replay_mismatches"] = mismatches
    info["replay_verified_cells"] = sum(1 for r in rows if r.get("layer") == "head_extra" and r.get("replay_verified") is True)
    return status


# ------------------------------------------------------------------------------------------------------------ train steps
def m1_config(cfgd: dict) -> A.M1Config:
    """The M1Config of a train config: rule 'absolute' rows keep their absolute tolerance; every other row the protocol rule."""
    if cfgd.get("m1_rule") == "absolute":
        return A.M1Config(rule="absolute", max_iterations=cfgd.get("m1_max_iterations", 30), tolerance=cfgd["m1_tolerance"])
    return A.M1Config(max_iterations=cfgd.get("m1_max_iterations", 30))


def _median(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def m1_summary(timed: list, recs: list) -> dict:
    """Per-config M1 statistics: the timed window (the step-time median) plus the first (host-init) step and all recorded steps."""
    stop = {}
    for r in timed:
        stop[r["m1_stop_reason"]] = stop.get(r["m1_stop_reason"], 0) + 1
    diag = [r["m1_fp64_diagnostic"] for r in timed]
    trig = [d for d in diag if d.get("triggered")]
    rel = [r["m1_gap_relative_to_zero"] for r in timed if r["m1_gap_relative_to_zero"] is not None]
    first = recs[0]
    return {"m1_rule": first["m1_rule"], "m1_relative_tolerance": first["m1_relative_tolerance"],
            "m1_cap_hit_fraction": float(np.mean([r["m1_stop_reason"] == "iteration_budget_exhausted" for r in timed])), "m1_stop_reasons": json.dumps(stop),
            "m1_gap_ref_zero_first": first["m1_gap_ref_zero"], "m1_gap_ref_zero_median": _median([r["m1_gap_ref_zero"] for r in timed]),
            "m1_gap_start_median": _median([r["m1_gap_start"] for r in timed]), "m1_gap_relative_to_zero_median": _median(rel),
            "m1_gap_relative_to_zero_max": float(max(rel)) if rel else None,
            "m1_tolerance_effective_median": _median([r["m1_tolerance_effective"] for r in timed]),
            "m1_W_update_fro_median": _median([r["m1_W_update_fro"] for r in timed]),
            "m1_zero_reference_count": stop.get("zero_reference_gap", 0),
            "m1_fp64_diag_triggered_fraction": float(np.mean([bool(d.get("triggered")) for d in diag])),
            "m1_fp64_diag_decisions_agree_fraction": float(np.mean([bool(d.get("decisions_agree")) for d in trig])) if trig else None,
            "m1_iterations_first": first["m1_iterations"], "m1_w_absmax_first": first["w_absmax"],
            "encoder_grad_norm_first": (first.get("encoder_gradient_diagnostic") or {}).get("encoders_grad_norm"),
            "m1_iterations_median_all_steps": float(np.median([r["m1_iterations"] for r in recs])),
            "m1_cap_hit_fraction_all_steps": float(np.mean([r["m1_stop_reason"] == "iteration_budget_exhausted" for r in recs])),
            "n_steps_recorded": len(recs)}


def part_train(device, run_dir: Path, rows: list, info: dict, gpu: str, guard: A.DataGuard, configs=None) -> str:
    dev = torch.device(device)
    configs = TRAIN_CONFIGS if configs is None else configs
    fit_all = A.fit_cores(guard, 1.0, SEED)
    fit = sorted(set(A.HostBatchSampler(fit_all, SEED).draw().cores))     # one host batch: 16 distinct FIT cores resident
    imgs = A.load_learned_inputs(guard, fit, {"FIT"}, "T02 train-step timing (16 FIT cores resident = one host batch)", dev)
    info["resident_fit_cores"] = fit
    info["configs_run"] = [train_label(c) for c in configs]
    status = "PASS"
    per_step = {}
    for cfgd in configs:
        label = train_label(cfgd)
        is_m1 = cfgd["method"] == "MF_M1"
        lr = cfgd.get("lr")
        row = {"layer": "train_step", "config": label, "dataset": "BF_SHG", "method": cfgd["method"], "phase": "train_step", "device": gpu,
               "dtype": "float32", "n_positions": 128, "simplex_width": cfgd.get("K"), "encoder_included": True, "warmup": 0,
               "repeats": 0, "n_sc": 16, "n_pos": 128, "checks": cfgd.get("checks", "first_step") if cfgd["method"].startswith("MF_") else None,
               "m1_tolerance": cfgd.get("m1_tolerance"), "m1_max_iterations": cfgd.get("m1_max_iterations"), "lr": lr,
               "m1_rule": (cfgd.get("m1_rule") or "relative_zero_head") if is_m1 else None, "block_size": cfgd.get("dense_block"),
               "graph_retained": ({True: False, False: True}[bool(cfgd.get("dense_checkpoint"))] if cfgd["method"] == "MLP_dense" else None),
               "backward_recomputation": (bool(cfgd.get("dense_checkpoint")) if cfgd["method"] == "MLP_dense" else None),
               "R_permutations": {"MF_perm1": 1, "MLP_perm1": 1, "MLP_perm8": 8}.get(cfgd["method"])}
        t = None
        try:
            cfg = A.TrainConfig(method=cfgd["method"], seed=SEED, K=cfgd.get("K", 16), checks=cfgd.get("checks", "first_step"), m1=m1_config(cfgd),
                                dense_block=cfgd.get("dense_block"), dense_checkpoint=cfgd.get("dense_checkpoint", False),
                                **({"encoder_lr": lr, "head_lr": lr} if lr is not None else {}))
            t = A.Trainer(cfg, imgs, fit, dev)
            recs = [t.step() for _ in range(TRAIN_WARMUP)]
            A.synchronize(dev)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            ms = []
            for _ in range(TRAIN_TIMED):
                A.synchronize(dev)
                t0 = time.perf_counter()
                recs.append(t.step())
                A.synchronize(dev)
                ms.append(1000 * (time.perf_counter() - t0))
            peak_a = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None
            peak_r = torch.cuda.max_memory_reserved(dev) if dev.type == "cuda" else None
            phases = {}
            for _ in range(TRAIN_PHASE):
                marks = []

                def mark(name, marks=marks):
                    A.synchronize(dev)
                    marks.append((name, time.perf_counter()))
                A.synchronize(dev)
                t0 = time.perf_counter()
                recs.append(t.step(mark=mark))
                prev = t0
                for name, tt in marks:
                    phases.setdefault(name, []).append(1000 * (tt - prev))
                    prev = tt
            box = {}
            syncs = count_syncs(lambda tr=t: box.setdefault("rec", tr.step()), dev)
            if "rec" in box:
                recs.append(box["rec"])
            timed = recs[TRAIN_WARMUP:TRAIN_WARMUP + TRAIN_TIMED]
            q = quantiles(ms)
            note = ("host batch (16 FIT crops x 128 positions), host encoders, AdamW + clip 1.0; per-step scalar logging included "
                    "(syncs_per_step = CUDA synchronisations of one extra step); phases from %d extra steps with synchronised marks"
                    % TRAIN_PHASE)
            if cfgd["method"].startswith("MF_"):
                note += ("; MF W-box / permutation checks on %s; SceneBatch validation (finite / simplex, several syncs per scene) "
                         "always runs and is inside phase 'scenes'" % ("every step" if cfgd.get("checks") == "every_step" else "the first step only"))
            row.update({"status": "completed", **q, "elapsed_seconds": q["median_ms"] / 1000, "warmup": TRAIN_WARMUP, "repeats": len(ms),
                        "peak_allocated_bytes": peak_a, "peak_reserved_bytes": peak_r, "J_first": recs[0]["J"], "J_last": recs[-1]["J"],
                        **{f"phase_{k}_ms": float(np.median(v)) for k, v in phases.items() if v}, "syncs_per_step": syncs, "notes": note})
            if is_m1:
                its = [r["m1_iterations"] for r in timed]
                gaps = [r["m1_gap_final"] for r in timed]
                row.update({"m1_iterations_median": float(np.median(its)), "m1_iterations_max": int(max(its)),
                            "m1_gap_median": float(np.median(gaps)), "m1_gap_max": float(max(gaps)),
                            "m1_converged_fraction": float(np.mean([r["m1_converged"] for r in timed])),
                            "m1_gap_initial_median": float(np.median([r["m1_gap_start"] for r in timed])),
                            "m1_hvp_calls_median": float(np.median([r["m1_hvp_calls"] for r in timed])),
                            "m1_momentum_restarts_median": float(np.median([r["m1_momentum_restarts"] for r in timed])),
                            "m1_moment_passes_per_batch": float(np.median([r["m1_moments_passes"] for r in timed])),
                            "m1_w_absmax_last": recs[-1]["w_absmax"], "degenerate": all(r["w_absmax"] == 0.0 for r in timed),
                            "solver_iterations": float(np.median(its)), "solver_gap": float(np.median(gaps)), **m1_summary(timed, recs)})
                if cfgd.get("record_only"):
                    row["notes"] = M1_RECORD_NOTE + "; " + row["notes"]
                    row["status"] = "completed_record_only"
                elif cfgd.get("m1_rule") == "absolute":
                    row["notes"] = ("M1 rule 'absolute' with a binding iteration cap (tolerance 1e-300; require_converged False): the cost "
                                    "of the whole 10 / 30 / 100-iteration budget every batch (cap reference); " + row["notes"])
                else:
                    row["notes"] = M1_RELATIVE_NOTE + "; " + row["notes"]
            per_step[label] = recs
        except Exception as e:  # noqa: BLE001
            row.update({"status": "oom" if is_oom(e) else "error", "notes": f"{type(e).__name__}: {str(e)[:400]}",
                        "traceback": traceback.format_exc()[-3000:]})
            status = status if is_oom(e) else "FAIL"
        rows.append({k: v for k, v in row.items() if k != "traceback"})
        info.setdefault("configs", {})[label] = {k: row.get(k) for k in ("status", "median_ms", "p90_ms", "repeats", "peak_allocated_bytes",
                                                                         "syncs_per_step", "m1_iterations_median", "m1_gap_median",
                                                                         "m1_converged_fraction", "m1_gap_initial_median", "degenerate",
                                                                         "m1_rule", "lr", "m1_cap_hit_fraction", "m1_stop_reasons",
                                                                         "m1_gap_ref_zero_first", "m1_iterations_first", "m1_w_absmax_first",
                                                                         "encoder_grad_norm_first", "m1_gap_relative_to_zero_median",
                                                                         "m1_fp64_diag_triggered_fraction", "notes", "traceback")}
        print(json.dumps({"train": label, "status": row["status"], "median_ms": row.get("median_ms"), "syncs": row.get("syncs_per_step")}), flush=True)
        del t
        release()
    A.write_json_atomic(run_dir / "train_steps_per_step.json", per_step)
    del imgs
    release()
    return status


# ------------------------------------------------------------------------------------------------------------ registrations
def part_register(device, run_dir: Path, rows: list, info: dict, gpu: str, guard: A.DataGuard) -> str:
    dev = torch.device(device)
    cores = guard.included("DEV")[:REG_CORES]
    pert = {}
    for c in cores:                       # the frozen REG-P2 rule (Q.cmd_perturb wrote perturbations.json from it): no truth file read
        guard.log_rule(c, {"DEV"}, "T02 registration timing: perturbations from the rule Q.perturbations_p2 (timing only)")
        pert[c] = A.Q.perturbations_p2(c)
    inputs = A.GuardedInputs(guard, {"DEV"}, "T02 registration timing (untrained models; timing only)", pert, dev)
    torch.manual_seed(SEED)
    eU, eV, cr = A.P.build_models(nn)
    eU, eV, cr = (m.to(dev) for m in (eU, eV, cr))
    heads = {}
    for K in sorted({k for _, k in REG_ENERGIES if k}):
        h = A.MFHead(A.DESC, A.DESC, K, K, temperature=1.0).to(dev)
        gen = torch.Generator(device="cpu").manual_seed(A.Q.seed_of(f"V10|T02|W|{SEED}|{K}"))
        with torch.no_grad():
            h.W.copy_(torch.empty(K, K, dtype=torch.float64).uniform_(-0.9, 0.9, generator=gen))
        heads[K] = h
    A.freeze_for_test(eU, eV, cr, *heads.values())
    DU, enc_ms = {}, []
    for c in cores:
        A.synchronize(dev)
        t0 = time.perf_counter()
        DU[c] = A.reference_descriptors(eU, inputs, c)
        A.synchronize(dev)
        enc_ms.append(1000 * (time.perf_counter() - t0))
    q = quantiles(enc_ms)
    rows.append({"layer": "registration", "config": "reference_encoding", "dataset": "BF_SHG", "method": "host_encoder_eU", "phase": "reference_encoding",
                 "device": gpu, "dtype": "float32", "encoder_included": True, "status": "completed", **q, "elapsed_seconds": q["median_ms"] / 1000,
                 "warmup": 0, "repeats": len(enc_ms), "notes": "first call per core includes loading the reference image (Q.Inputs cache)"})
    for c in cores:
        for i in REG_PERTS:
            inputs.moving(c, i)                                          # perturbed moving images prepared outside the timed searches
    status = "PASS"
    for energy, K in REG_ENERGIES:
        label = energy + (f"_K{K}" if K else "")

        def make(c, i):
            mov = inputs.moving(c, i)[1]
            rng = np.random.default_rng(A.Q.seed_of(f"R6-REG-P2|pos|{c}|{i}"))
            if energy == "MLP_perm1_host":
                return A.Q.RigidEnergy(torch, F, "critic", DU[c], mov, rng, eV=eV, critic=cr)
            return A.MFRigidEnergy(DU[c], mov, rng, eV=eV, head=heads[K], estimator="moments" if energy == "MF_moments" else "perm1")
        try:
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            w = A.register_timed(make(cores[0], REG_PERTS[0]), dev)
            rows.append(reg_row(gpu, label, energy, K, cores[0], REG_PERTS[0], w, warmup=True))
            secs = []
            for c in cores:
                for i in REG_PERTS:
                    r = A.register_timed(make(c, i), dev)
                    secs.append(r["seconds"])
                    rows.append(reg_row(gpu, label, energy, K, c, i, r, warmup=False))
            peak_a = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None
            peak_r = torch.cuda.max_memory_reserved(dev) if dev.type == "cuda" else None
            for r in rows[-len(secs):]:
                r.update({"peak_allocated_bytes": peak_a, "peak_reserved_bytes": peak_r})
            info.setdefault("energies", {})[label] = {"status": "completed", "seconds_median": float(np.median(secs)), "n": len(secs),
                                                      "peak_allocated_bytes": peak_a}
        except Exception as e:  # noqa: BLE001
            rows.append({"layer": "registration", "config": label, "energy": label, "status": "oom" if is_oom(e) else "error", "device": gpu,
                         "warmup": 0, "repeats": 0, "notes": f"{type(e).__name__}: {str(e)[:400]}"})
            info.setdefault("energies", {})[label] = {"status": "error", "error": repr(e)[:400], "traceback": traceback.format_exc()[-3000:]}
            status = status if is_oom(e) else "FAIL"
        print(json.dumps({"register": label, "status": info["energies"][label]["status"]}), flush=True)
        release()
    info["cores"] = cores
    info["perturbation_indices"] = list(REG_PERTS)
    return status


def reg_row(gpu, label, energy, K, core, i, r, warmup):
    return {"layer": "registration", "config": label, "dataset": "BF_SHG", "method": energy, "energy": label, "phase": "registration", "device": gpu,
            "dtype": "float32", "n_positions": 4096, "simplex_width": K, "encoder_included": True, "status": "completed",
            "elapsed_seconds": r["seconds"], "median_ms": 1000 * r["seconds"], "warmup": 0, "repeats": 1, "core": core,
            "perturbation_index": i, "warmup_row": warmup,
            "grid_seconds": r["grid_seconds"], "refine_and_final_seconds": r["refine_and_final_seconds"], "exact_energy_calls": r["exact_energy_calls"],
            "n_grid": r["n_grid"], "notes": "unchanged host searcher; fixed untrained models; timing only (no error / truth quantity)"}


# ------------------------------------------------------------------------------------------------------------ run
def t01_status() -> dict:
    out = {}
    for name in ("TORCH_VALIDATION.json", "GEOMETRY_GRADIENT_CHECK.json"):
        f = A.REPO / "v10" / "T01" / name
        if not f.exists():
            out[name] = {"overall_status": "NOT_RUN", "note": "file missing"}
            continue
        j = json.loads(f.read_text())
        out[name] = {k: j.get(k) for k in ("overall_status", "host_adapter_verified", "pending_parts", "failed_parts", "blocked_parts",
                                           "provenance_mismatches", "generated_utc")}
        out[name]["code_hash_at_merge"] = ((j.get("provenance") or {}).get("current") or {}).get("code_hash_src_scripts_v10")
    return out


def kept_rows_source(repo_out: Path, devtype: str) -> dict:
    """Provenance of the timing file whose rows an M1-only re-measurement keeps (phase 1: job 1043695)."""
    f = repo_out / f"T02_TIMING_{devtype}.json"
    if not f.exists():
        return {"status": "missing", "path": str(f)}
    j = json.loads(f.read_text())
    e = j.get("environment", {})
    return {"path": f"v10/T02/{f.name}", "sha256": A.sha256_file(f), "long_csv": j.get("long_csv"), "job": e.get("slurm_job_id"),
            "git_commit": e.get("git_commit"), "code_hash": e.get("code_hash_src_scripts_v10"), "gpu_name": e.get("gpu_name"),
            "parts": {k: v.get("status") for k, v in j.get("parts", {}).items()},
            "kept": "every row of that file (head, head_extra, permutation variance, the non-M1 and the rule-'absolute' M1 train steps, "
                    "registrations); only the M1 relative-rule configs are measured here"}


def run(args) -> int:
    pin = A.check_job_pin()
    device = args.device
    scope = args.train_scope
    parts_req = [p for p in args.parts.split(",") if p]
    if scope == "m1_relative" and parts_req != ["train"]:
        print(f"--train-scope m1_relative re-measures only the M1 relative-rule train configs (decision 2026-10-17 2.3): use --parts "
              f"train (got {args.parts})", flush=True)
        return 2
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable: nothing run", flush=True)
        return 3
    devtype = torch.device(device).type
    label = os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = args.out_root / "T02" / "runs" / (f"{devtype}_{label}" + ("" if scope == "all" else f"_{scope}"))
    run_dir.mkdir(parents=True, exist_ok=True)
    env = A.environment(device)
    print(json.dumps({"t02_run": str(run_dir), "scope": scope, "parts": parts_req, "precision_record": env["precision"]}, default=str), flush=True)
    gpu = env.get("gpu_name") or "CPU"
    guard = A.DataGuard()
    rows, parts = [], {}
    for part in parts_req:
        info: dict = {}
        t0 = time.time()
        try:
            if part == "head":
                st = part_head(device, run_dir, rows, info, gpu)
            elif part == "head_extra":
                st = part_head_extra(device, run_dir, rows, info, gpu)
            elif part == "train":
                st = part_train(device, run_dir, rows, info, gpu, guard, TRAIN_SCOPES[scope])
            elif part == "register":
                st = part_register(device, run_dir, rows, info, gpu, guard)
            else:
                raise ValueError(f"unknown part {part}")
        except Exception as e:  # noqa: BLE001
            st = "FAIL"
            info.update({"error": repr(e)[:2000], "traceback": traceback.format_exc()[-8000:]})
        parts[part] = {"status": st, "seconds": time.time() - t0, **info}
        print(f"[t02] part {part}: {st} ({time.time() - t0:.0f} s)", flush=True)
        A.write_json_atomic(run_dir / "T02_TIMING.partial.json", {"parts": parts, "n_rows": len(rows)})
    kept = kept_rows_source(args.repo_out, devtype) if scope == "m1_relative" else None
    for name in ("head", "head_extra", "train", "register"):
        parts.setdefault(name, {"status": "NOT_RUN"} if kept is None else
                         {"status": "NOT_RUN", "reason": "not affected by the M1 rule change: the measurement of %s (job %s) is kept"
                                                         % (kept.get("path"), kept.get("job"))})
    prefix = SCOPE_PREFIX[scope]
    long_name, timing_name = f"{prefix}_LONG_{devtype}.csv", f"{prefix}_TIMING_{devtype}.json"
    prec_end = A.precision_record(device)
    out = {"schema": "V10-T02-TIMING-v3", "generated_utc": A.utc_now(), "environment": env, "job_pin": pin, "run_dir": str(run_dir),
           "device_type": devtype, "t01_status_at_run": t01_status(), "train_scope": scope,
           "rows_kept_from": kept, "precision_at_end": prec_end,
           "precision_settings_unchanged": A.precision_settings(prec_end) == A.precision_settings(env["precision"]),
           "validity_note": "timings are valid only for implementations that pass T01; the package benchmark marks a cell "
                            "correctness_failed when dense / moments disagree and does not time it",
           "status_vocabulary": "PASS (ran; OOM rows recorded, not dropped) / FAIL (error, correctness failure or exception) / NOT_RUN",
           "parts": parts, "data_reads": guard.reads, "seed": SEED,
           "protocol": {"grid": GRID, "head_warmup": HEAD_WARMUP, "head_repeats": HEAD_REPEATS, "head_block": HEAD_BLOCK,
                        "train_warmup": TRAIN_WARMUP, "train_timed": TRAIN_TIMED, "train_phase_steps": TRAIN_PHASE,
                        "train_configs": TRAIN_SCOPES[scope], "registration_energies": REG_ENERGIES, "registration_cores": REG_CORES,
                        "registration_perturbations": REG_PERTS, "m1_cap_binding_tolerance": CAP,
                        "m1_absolute_tolerance_record_row": A.M1_PROTOCOL_TOLERANCE, "m1_relative_tolerance": A.M1_RELATIVE_TOLERANCE,
                        "m1_cal_caps": A.M1_CAL_CAPS, "lr_candidates": LR_CANDIDATES},
           "long_csv": f"v10/T02/{long_name}", "n_rows": len(rows)}
    args.repo_out.mkdir(parents=True, exist_ok=True)
    fields = TEMPLATE_FIELDS + EXTRA_FIELDS
    for path in (run_dir / long_name, args.repo_out / long_name):
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
    A.write_json_atomic(run_dir / timing_name, out)
    A.write_json_atomic(args.repo_out / timing_name, out)
    print(json.dumps({"T02": {k: v["status"] for k, v in parts.items()}, "rows": len(rows)}), flush=True)
    return 0 if all(v["status"] in ("PASS", "NOT_RUN") for v in parts.values()) else 1


# ------------------------------------------------------------------------------------------------------------ budget (decision 8) + ledger
DECISION_DOC = "v10/decisions/V10_PHASE1_DECISIONS_20261017_CN.md section 8"
DECISION_CONTRACT = A.REPO / "v10" / "decisions" / "V10_PROTOCOL_DECISIONS_20261017.json"
BUDGET_COUNTS = {"train_steps_per_fit": 3000, "checkpoint_interval": 250, "checkpoints_per_fit": 12, "cal_cores": 8, "dev_cores": 32,
                 "perturbations_per_core": 10, "tuning_seed": 1017, "final_seeds": [1001, 1002, 1003], "lr_candidates": list(LR_CANDIDATES),
                 "m1_caps": list(A.M1_CAL_CAPS), "m1_relative_tolerance": A.M1_RELATIVE_TOLERANCE, "K": 16, "family_cap_candidates": 9,
                 "family_cap_L40S_gpu_h": 8.0}
FAMILIES = ("MF_VCS_perm1", "MF_VCS_M0", "MF_VCS_M1", "MLP_VCS_perm1")
FAMILY_BASE_TRAIN = {"MF_VCS_perm1": "MF_perm1_K16", "MF_VCS_M0": "MF_M0_K16", "MLP_VCS_perm1": "MLP_perm1"}   # step cost lr-independent
FAMILY_REG = {"MF_VCS_perm1": "MF_perm1_K16", "MF_VCS_M0": "MF_moments_K16", "MF_VCS_M1": "MF_moments_K16", "MLP_VCS_perm1": "MLP_perm1_host"}
M1_CAPBOUND = {10: "MF_M1_K16_it10_capbound", 30: "MF_M1_K16_it30_capbound", 100: "MF_M1_K16_it100_capbound"}
NOT_MEASURED = {"InfoNCE_host[J_perm1]": "decision 2026-10-17 section 4; V10 trainer not connected yet",
                "CoMIR_native": "external system: own training / latent / registration costs",
                "RuLSIF_kernel exact moments": "not implemented in V10",
                "MI_native / NMI_native / MIND_2D_adapted": "non-learned; registration cost only; not connected to V10",
                "MLP_VCS_perm8 registration energy": "not implemented"}


def m1_relative_label(lr: float, cap: int) -> str:
    return train_label({"method": "MF_M1", "K": 16, "m1_rule": "relative_zero_head", "m1_max_iterations": cap, "lr": lr})


def decision8_candidates() -> list:
    """The 18 CAL tuning candidates of decision 2026-10-17 8.1, per family in the frozen order (learning rate ascending; MF_VCS_M1: cap
    10 / 30 / 100 within each learning rate)."""
    out = []
    for fam in FAMILIES:
        k = 0
        for lr in LR_CANDIDATES:
            for cap in (A.M1_CAL_CAPS if fam == "MF_VCS_M1" else (None,)):
                k += 1
                out.append({"family": fam, "ordinal_in_family": k, "lr": lr, "m1_cap": cap})
    return out


def contract_check(path: Path = DECISION_CONTRACT) -> dict:
    """The owner's decision contract (read-only) against the counts this budget uses; returns its old-timing reference numbers."""
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    d = json.loads(path.read_text())
    nr, m1 = d["natural_registration"], d["m1"]
    c = BUDGET_COUNTS
    pairs = {"MF_K": (nr.get("MF_K"), c["K"]), "training_steps": (nr.get("training_steps"), c["train_steps_per_fit"]),
             "checkpoint_interval": (nr.get("checkpoint_interval"), c["checkpoint_interval"]),
             "checkpoints_per_fit": (nr.get("checkpoints_per_fit"), c["checkpoints_per_fit"]),
             "tuning_algorithm_seed": (nr.get("tuning_algorithm_seed"), c["tuning_seed"]),
             "final_algorithm_seeds": (nr.get("final_algorithm_seeds"), c["final_seeds"]),
             "learning_rate_candidates": (nr.get("learning_rate_candidates"), c["lr_candidates"]),
             "CAL.cores": ((nr.get("CAL") or {}).get("cores"), c["cal_cores"]),
             "CAL.perturbations_per_core": ((nr.get("CAL") or {}).get("perturbations_per_core"), c["perturbations_per_core"]),
             "DEV.cores": ((nr.get("DEV") or {}).get("cores"), c["dev_cores"]),
             "candidate_counts": (nr.get("candidate_counts"), {"MF_VCS_perm1": 3, "MF_VCS_M0": 3, "MF_VCS_M1": 9, "MLP_VCS_perm1": 3, "total": 18}),
             "tuning_budget_per_family.max_candidates": ((nr.get("tuning_budget_per_family") or {}).get("max_candidates"), c["family_cap_candidates"]),
             "tuning_budget_per_family.max_L40S_allocated_GPU_hours": ((nr.get("tuning_budget_per_family") or {}).get("max_L40S_allocated_GPU_hours"),
                                                                       c["family_cap_L40S_gpu_h"]),
             "m1.max_iterations_CAL": (m1.get("max_iterations_CAL"), c["m1_caps"]),
             "m1.relative_tolerance": (m1.get("relative_tolerance"), c["m1_relative_tolerance"]),
             "m1.require_converged": (m1.get("require_converged"), False), "m1.absolute_floor": (m1.get("absolute_floor"), False)}
    counts = decision8_candidates()
    per_fam = {f: sum(x["family"] == f for x in counts) for f in FAMILIES}
    pairs["candidate_counts_of_this_grid"] = (nr.get("candidate_counts"), {**per_fam, "total": len(counts)})
    mismatches = {k: {"contract": a, "used": b} for k, (a, b) in pairs.items() if a != b}
    return {"status": "consistent" if not mismatches else "MISMATCH", "path": "v10/decisions/" + path.name, "sha256": A.sha256_file(path),
            "decision_id": d.get("decision_id"), "checked": sorted(pairs), "mismatches": mismatches,
            "old_projection_reference": nr.get("old_measured_cost_projection_only")}


def _fnum(x):
    return float(x) if x not in (None, "", "None") else None


def _train_medians(rows) -> dict:
    return {r["config"]: _fnum(r["median_ms"]) for r in rows if r.get("layer") == "train_step" and r.get("status") == "completed"}


def _reg_medians(rows) -> dict:
    reg = {}
    for r in rows:
        if r.get("layer") == "registration" and r.get("phase") == "registration" and r.get("status") == "completed" and r.get("warmup_row") == "False":
            reg.setdefault(r["config"], []).append(_fnum(r["elapsed_seconds"]))
    return {k: float(np.median(v)) for k, v in reg.items()}


def _gpu_h(step_ms, s_reg, n_reg):
    if step_ms is None or s_reg is None:
        return None
    return (BUDGET_COUNTS["train_steps_per_fit"] * step_ms / 1000 + n_reg * s_reg) / 3600


def project(step_of, reg_med: dict) -> dict:
    """Decision-8 projection for step_of(candidate) -> (step ms, train config): per candidate the CAL tuning cost (3000 steps + 12
    checkpoints x 8 CAL cores x 10 perturbations), per family the total against the caps (<= 9 candidates, <= 8 L40S GPU-h) and how
    many candidates fit in the frozen order, the final fits (4 families x 3 seeds; 3000 steps + 960 CAL + 320 DEV registrations each)
    by the CAL-selected M1 candidate, and the grand totals."""
    c = BUDGET_COUNTS
    n_cal = c["checkpoints_per_fit"] * c["cal_cores"] * c["perturbations_per_core"]
    n_final = n_cal + c["dev_cores"] * c["perturbations_per_core"]
    cands = []
    for cd in decision8_candidates():
        step, cfg = step_of(cd)
        s_reg = reg_med.get(FAMILY_REG[cd["family"]])
        cands.append({**cd, "train_config": cfg, "train_step_ms_median": step, "registration_energy": FAMILY_REG[cd["family"]],
                      "s_per_registration_median": s_reg, "cal_registrations": n_cal, "tuning_gpu_h": _gpu_h(step, s_reg, n_cal),
                      "final_registrations_per_seed": n_final, "final_unit_gpu_h": _gpu_h(step, s_reg, n_final)})
    fams = {}
    for fam in FAMILIES:
        fc = [x for x in cands if x["family"] == fam]
        done = all(x["tuning_gpu_h"] is not None for x in fc)
        total = float(sum(x["tuning_gpu_h"] for x in fc)) if done else None
        cum, n_in = 0.0, 0
        for x in fc:                                                      # the frozen order; stop at the first limit reached
            if x["tuning_gpu_h"] is None or n_in + 1 > c["family_cap_candidates"] or cum + x["tuning_gpu_h"] > c["family_cap_L40S_gpu_h"]:
                break
            cum += x["tuning_gpu_h"]
            n_in += 1
        fams[fam] = {"n_candidates": len(fc), "tuning_gpu_h": total, "cap_candidates": c["family_cap_candidates"],
                     "cap_L40S_gpu_h": c["family_cap_L40S_gpu_h"],
                     "within_family_caps": bool(done and len(fc) <= c["family_cap_candidates"] and total <= c["family_cap_L40S_gpu_h"]),
                     "candidates_within_caps_in_frozen_order": n_in, "margin_gpu_h": (c["family_cap_L40S_gpu_h"] - total) if done else None}
    seeds = len(c["final_seeds"])
    tuning_total = float(sum(fams[f]["tuning_gpu_h"] for f in FAMILIES)) if all(fams[f]["tuning_gpu_h"] is not None for f in FAMILIES) else None
    final_fixed = {}
    for fam in FAMILIES:
        if fam != "MF_VCS_M1":
            vals = [x["final_unit_gpu_h"] for x in cands if x["family"] == fam]
            final_fixed[fam] = seeds * max(vals) if all(v is not None for v in vals) else None    # lr-independent here (max = any)
    m1_final = {f"lr{x['lr']:.0e}_cap{x['m1_cap']}": (seeds * x["final_unit_gpu_h"] if x["final_unit_gpu_h"] is not None else None)
                for x in cands if x["family"] == "MF_VCS_M1"}
    fixed_ok = all(v is not None for v in final_fixed.values()) and tuning_total is not None
    totals = {k: (tuning_total + sum(final_fixed.values()) + v if (fixed_ok and v is not None) else None) for k, v in m1_final.items()}
    by_cap = {}
    for cap in A.M1_CAL_CAPS:
        vals = [totals[f"lr{lr:.0e}_cap{cap}"] for lr in LR_CANDIDATES]
        by_cap[str(cap)] = max(vals) if all(v is not None for v in vals) else None
    return {"candidates": cands, "families": fams, "tuning_total_gpu_h": tuning_total, "final_seeds_per_family": seeds,
            "final_fixed_families_gpu_h": final_fixed, "final_MF_VCS_M1_gpu_h_by_selected_candidate": m1_final,
            "total_gpu_h_by_selected_M1_candidate": totals, "total_gpu_h_by_selected_M1_cap_max_over_lr": by_cap}


def reproduce_old(ref: dict, old: dict | None) -> dict:
    """The decision's old-timing projection (8.2, from job 1043695 with M1 at the binding caps) recomputed with this formula."""
    if not old:
        return {"status": "NOT_RUN", "reason": "contract reference missing"}
    rows = {f: (ref["families"][f]["tuning_gpu_h"], old["tuning_GPU_hours"].get(f)) for f in FAMILIES}
    rows["tuning_total"] = (ref["tuning_total_gpu_h"], old["tuning_GPU_hours"].get("total"))
    for cap, v in old["tuning_plus_3_final_seeds_per_family_GPU_hours_by_selected_M1_cap"].items():
        rows[f"total_selected_M1_cap{cap}"] = (ref["total_gpu_h_by_selected_M1_cap_max_over_lr"].get(str(cap)), v)
    out = {k: {"recomputed": a, "decision_value": b, "equal_at_4_decimals": a is not None and b is not None and round(a, 4) == round(b, 4)}
           for k, (a, b) in rows.items()}
    return {"status": "PASS" if all(v["equal_at_4_decimals"] for v in out.values()) else "FAIL", "rows": out}


def _provenance(j: dict, path: Path) -> dict:
    e = j.get("environment", {})
    return {"path": f"v10/T02/{path.name}", "sha256": A.sha256_file(path), "job": e.get("slurm_job_id"), "git_commit": e.get("git_commit"),
            "code_hash": e.get("code_hash_src_scripts_v10"), "gpu_name": e.get("gpu_name"), "node": e.get("slurm_nodelist") or e.get("host"),
            "torch": e.get("torch"), "precision": e.get("precision")}


def squeue_snapshot() -> dict:
    try:
        p = subprocess.run(["squeue", "-u", os.environ.get("USER", "yinwang"), "-h", "-o", "%i %q %T %P %j"], capture_output=True, text=True, timeout=60)
        lines = [ln.split() for ln in p.stdout.splitlines() if ln.strip()]
        return {"utc": A.utc_now(), "normal_qos_count": sum(1 for ln in lines if len(ln) > 1 and ln[1] == "normal"),
                "running_gpu_jobs": sum(1 for ln in lines if len(ln) > 3 and ln[2] == "RUNNING" and any(x in ln[3] for x in ("RTX6000PRO", "H100", "L40S", "A100"))),
                "pending": sum(1 for ln in lines if len(ln) > 2 and ln[2] == "PENDING")}
    except Exception as e:  # noqa: BLE001
        return {"utc": A.utc_now(), "error": repr(e)[:200]}


def budget(args) -> int:
    """BUDGET.json for the decision-8 grid (18 CAL candidates, per-family caps 9 candidates / 8 L40S GPU-h, 4 families x 3 formal
    seeds).  Kept rows: v10/T02/T02_TIMING_cuda.json + T02_LONG_cuda.csv (phase-1 job 1043695: MF perm1 / M0 K16, MLP perm1 step
    times, registration energies, the rule-'absolute' M1 cap references); re-measured rows: v10/T02/T02_M1_RELATIVE_* (the nine
    MF_VCS_M1 relative-rule candidates).  UNIT_LEDGER.csv (package units) is rewritten from the kept rows as before."""
    A.check_job_pin()
    base_json, base_csv = args.repo_out / "T02_TIMING_cuda.json", args.repo_out / "T02_LONG_cuda.csv"
    m1_json, m1_csv = args.repo_out / "T02_M1_RELATIVE_TIMING_cuda.json", args.repo_out / "T02_M1_RELATIVE_LONG_cuda.csv"
    units = list(csv.DictReader((A.PACKAGE / "configs" / "EXPERIMENT_UNITS.csv").read_text().splitlines()))
    if not (base_json.exists() and base_csv.exists()):
        A.write_json_atomic(args.repo_out / "BUDGET.json", {"schema": "V10-T02-BUDGET-v3", "status": "NOT_RUN", "generated_utc": A.utc_now(),
                                                            "reason": f"{base_json.name} / {base_csv.name} missing"})
        return 1
    base = json.loads(base_json.read_text())
    base_rows = list(csv.DictReader(base_csv.read_text().splitlines()))
    train_base, reg_med = _train_medians(base_rows), _reg_medians(base_rows)
    have_m1 = m1_json.exists() and m1_csv.exists()
    m1 = json.loads(m1_json.read_text()) if have_m1 else None
    m1_rows = list(csv.DictReader(m1_csv.read_text().splitlines())) if have_m1 else []
    train_m1 = _train_medians(m1_rows)

    def step_relative(cd):
        lab = m1_relative_label(cd["lr"], cd["m1_cap"]) if cd["family"] == "MF_VCS_M1" else FAMILY_BASE_TRAIN[cd["family"]]
        return (train_m1 if cd["family"] == "MF_VCS_M1" else train_base).get(lab), lab

    def step_capbound(cd):
        lab = M1_CAPBOUND[cd["m1_cap"]] if cd["family"] == "MF_VCS_M1" else FAMILY_BASE_TRAIN[cd["family"]]
        return train_base.get(lab), lab
    proj, capref = project(step_relative, reg_med), project(step_capbound, reg_med)
    contract = contract_check()
    repro = reproduce_old(capref, contract.get("old_projection_reference"))
    current = A.code_hash()
    t01 = t01_status()
    tv = t01["TORCH_VALIDATION.json"]
    base_parts = {k: v.get("status") for k, v in base.get("parts", {}).items()}
    m1_parts = {k: v.get("status") for k, v in (m1 or {}).get("parts", {}).items()}
    reasons = []
    if tv.get("host_adapter_verified") is not True:
        reasons.append("T01 host_adapter_verified is not true")
    if tv.get("code_hash_at_merge") != current:
        reasons.append(f"T01 merge code hash {tv.get('code_hash_at_merge')} != current {current}")
    if not all(base_parts.get(p) == "PASS" for p in ("head", "head_extra", "train", "register")):
        reasons.append(f"kept T02 parts not all PASS: {base_parts}")
    if m1 is not None:
        e = m1.get("environment", {})
        if m1_parts.get("train") != "PASS":
            reasons.append(f"M1 relative train part {m1_parts.get('train')}")
        if m1.get("train_scope") != "m1_relative":
            reasons.append(f"M1 timing scope {m1.get('train_scope')}")
        if e.get("code_hash_src_scripts_v10") != current:
            reasons.append(f"M1 relative timing code hash {e.get('code_hash_src_scripts_v10')} != current {current}")
        if e.get("gpu_name") != base.get("environment", {}).get("gpu_name"):
            reasons.append(f"GPU of the re-measurement {e.get('gpu_name')} != GPU of the kept rows {base.get('environment', {}).get('gpu_name')}")
        missing = [x["train_config"] for x in proj["candidates"] if x["tuning_gpu_h"] is None]
        if missing:
            reasons.append(f"unmeasured candidates: {missing}")
    if contract.get("status") != "consistent":
        reasons.append(f"decision contract check: {contract.get('status')}")
    if repro.get("status") != "PASS":
        reasons.append("the decision's old-timing projection is not reproduced by this formula")
    status = "NOT_RUN" if m1 is None else ("MEASURED_PROJECTION" if not reasons else "PROVISIONAL")
    m1_meas = {r["config"]: {k: r.get(k) for k in ("status", "median_ms", "p90_ms", "repeats", "m1_iterations_median", "m1_iterations_max",
                                                   "m1_cap_hit_fraction", "m1_stop_reasons", "m1_gap_ref_zero_first", "m1_gap_ref_zero_median",
                                                   "m1_gap_relative_to_zero_median", "m1_gap_relative_to_zero_max", "m1_iterations_first",
                                                   "m1_w_absmax_first", "encoder_grad_norm_first", "m1_fp64_diag_triggered_fraction",
                                                   "m1_fp64_diag_decisions_agree_fraction", "m1_zero_reference_count",
                                                   "m1_iterations_median_all_steps", "m1_cap_hit_fraction_all_steps", "n_steps_recorded")}
               for r in m1_rows if r.get("layer") == "train_step"}
    for lab, v in m1_meas.items():
        cap = next((c_ for c_ in A.M1_CAL_CAPS if lab.startswith(f"MF_M1_K16_it{c_}_")), None)
        ref_ms = train_base.get(M1_CAPBOUND.get(cap)) if cap else None
        v["step_ratio_vs_cap_reference"] = (_fnum(v["median_ms"]) / ref_ms) if (ref_ms and _fnum(v.get("median_ms"))) else None
    out = {"schema": "V10-T02-BUDGET-v3", "status": status, "generated_utc": A.utc_now(), "decision": DECISION_DOC,
           "status_rule": "MEASURED_PROJECTION only when T01 host_adapter_verified is true at the current code hash, the kept T02 parts "
                          "are PASS, the M1 relative-rule train part is PASS at the current code hash on the GPU type of the kept rows, "
                          "every candidate is measured, the decision contract agrees with the counts used and the decision's old-timing "
                          "projection is reproduced; otherwise PROVISIONAL (reasons listed); NOT_RUN without the M1 re-measurement",
           "provisional_reasons": reasons,
           "status_inputs": {"t01": t01, "kept_t02_parts": base_parts, "m1_relative_t02_parts": m1_parts, "current_code_hash": current},
           "sources": {"rows_kept": {**_provenance(base, base_json), "long_csv": f"v10/T02/{base_csv.name}",
                                     "train_configs_used": sorted(set(FAMILY_BASE_TRAIN.values()) | set(M1_CAPBOUND.values())),
                                     "registration_energies_used": sorted(set(FAMILY_REG.values())),
                                     "note": "not re-measured: unaffected by the M1 rule change (decision 2026-10-17 2.3)"},
                       "rows_remeasured": ({**_provenance(m1, m1_json), "long_csv": f"v10/T02/{m1_csv.name}",
                                            "train_configs_used": [m1_relative_label(lr, cap) for lr in LR_CANDIDATES for cap in A.M1_CAL_CAPS]}
                                           if m1 is not None else {"status": "missing", "path": f"v10/T02/{m1_json.name}"})},
           "protocol_counts": BUDGET_COUNTS, "contract_check": {k: v for k, v in contract.items() if k != "old_projection_reference"},
           "projection": proj,
           "m1_cap_reference_projection": {"note": "MF_VCS_M1 at the binding caps (rule 'absolute', tolerance 1e-300, job 1043695, lr 1e-3): "
                                                   "every batch runs the whole cap, so the solver work of the relative rule cannot exceed it "
                                                   "(its bookkeeping and the fp64 diagnostic add a small measured overhead)",
                                           "families": capref["families"], "tuning_total_gpu_h": capref["tuning_total_gpu_h"],
                                           "total_gpu_h_by_selected_M1_cap_max_over_lr": capref["total_gpu_h_by_selected_M1_cap_max_over_lr"]},
           "old_timing_projection_reproduced": repro,
           "m1_relative_measurement": m1_meas,
           "not_measured_in_T02": NOT_MEASURED,
           "excludes": ["checkpoint writing and other in-job overhead (inside the 8 GPU-h family cap; not measured)", "data I/O",
                        "queue wait (reported separately)", "failed or re-run jobs", "new validation jobs", "missing baselines", "T03",
                        "TEST_REPEAT", "EPFL"],
           "assumptions": ["one L40S per fit; step and registration times are the medians of the timing jobs (kept rows: job 1043695)",
                           "per-step cost from the T02 window from the host init (10 warm-up + 50 timed steps, seed 10101); for M1 the "
                           "relative rule's iteration count later in a 3000-step run is not measured: the cap reference is the per-cap bound",
                           "MF perm1 / M0 and MLP perm1 step costs are learning-rate independent (fixed computation, measured at lr 1e-3)",
                           "registration cost per call from untrained models (fixed grid + 3 x 100 Adam steps: weight-independent)",
                           "CAL tuning per candidate: 3000 steps + 12 checkpoints x 8 CAL cores x 10 perturbations; final per seed: 3000 "
                           "steps + 960 CAL + 320 DEV registrations"],
           "superseded": {"phase1_budget": "commit 11746dd v10/T02/BUDGET.json (29.78 GPU-h incl. the EPFL proxy, 21.089 for the BF batch): "
                                           "not the budget of this grid (decision 8.2)"},
           "resources": {"queue_snapshot": squeue_snapshot(), "v10_queue_rule": "at most 2 queued V10 jobs; normal-QOS count <= 29"}}
    A.write_json_atomic(args.repo_out / "BUDGET.json", out)
    status_by = {}
    for r in base_rows:
        if r["layer"] == "head" and r["unit_id"]:
            status_by[r["unit_id"]] = r
    with open(args.repo_out / "UNIT_LEDGER.csv", "w", newline="") as f:
        fields = list(units[0].keys()) + ["run_status", "run_dtype", "run_median_ms", "run_device", "evidence"]
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for u in units:
            r32, r64 = status_by.get(u["unit_id"]), status_by.get(u["unit_id"] + ":float64_reference")
            if u["stage"] == "T02":
                st = (r32 or {}).get("status", "NOT_RUN")
                wr.writerow({**u, "run_status": st + (f"; float64 reference {r64['status']}" if r64 else "; float64 reference NOT_RUN"),
                             "run_dtype": "float32 (+float64 reference)", "run_median_ms": (r32 or {}).get("median_ms", ""),
                             "run_device": (r32 or {}).get("device", ""), "evidence": "v10/T02/T02_LONG_cuda.csv; v10/T02/T02_TIMING_cuda.json"})
            else:
                wr.writerow({**u, "run_status": "NOT_RUN (phase 1 does not launch T03-T05)", "run_dtype": "", "run_median_ms": "",
                             "run_device": "", "evidence": ""})
    print(json.dumps({"BUDGET": status, "reasons": reasons, "tuning_total_gpu_h": proj["tuning_total_gpu_h"],
                      "total_by_M1_cap": proj["total_gpu_h_by_selected_M1_cap_max_over_lr"],
                      "families": {f: proj["families"][f]["tuning_gpu_h"] for f in FAMILIES}, "old_reproduced": repro.get("status")}), flush=True)
    return 0 if status != "NOT_RUN" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", choices=("cpu", "cuda"), required=True)
    r.add_argument("--parts", default="head,head_extra,train,register")
    r.add_argument("--train-scope", choices=tuple(TRAIN_SCOPES), default="all",
                   help="all: the phase-1 train configs + the M1 relative-rule candidates -> T02_TIMING / T02_LONG; m1_relative (with "
                        "--parts train): only the nine MF_VCS_M1 relative-rule candidates -> T02_M1_RELATIVE_TIMING / _LONG, every other "
                        "row stays the earlier measurement (decision 2026-10-17 2.3)")
    r.add_argument("--out-root", type=Path, default=A.OUT_ROOT)
    r.add_argument("--repo-out", type=Path, default=REPO_T02)
    b = sub.add_parser("budget")
    b.add_argument("--repo-out", type=Path, default=REPO_T02)
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else budget(args)  # both refuse when the job pin (V10_EXPECT_CODE) does not match


if __name__ == "__main__":
    sys.exit(main())
