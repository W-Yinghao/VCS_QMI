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
              M1 with a binding iteration cap (tolerance 1e-300: budgets 10 / 30 / 100 always run in full) plus one record row
              at the protocol tolerance 1e-4 (degenerate from W = 0 at host init, excluded from BUDGET); MLP_perm1, MLP_perm8,
              MLP_dense (full graph and checkpointed blocks).  MF W-box / permutation checks run on the first step only (the
              validated-loop rule of the package); K16 perm1 / M0 rows with checks on every step quantify that overhead.
              Synchronised phases: batch, descriptors, scenes (MF feature maps + partition / SceneBatch validation), objective,
              backward, update; CUDA host-device synchronisations per step counted with torch.cuda.set_sync_debug_mode.
  register    full registrations with the unchanged host searcher Q.optimise_rigid (21 angles x 508 translations, top 3,
              100 Adam steps) on 3 DEV cores (perturbations from the frozen rule Q.perturbations_p2, no truth file read) with
              fixed, UNTRAINED models: timing only - no error, F_true or any other truth-dependent quantity is computed.
              MF moments (K 8 / 16 / 32), MF perm1 (K 16), host MLP perm1 energy.
  budget      (no GPU) v10/T02/BUDGET.json and v10/T02/UNIT_LEDGER.csv from v10/T02/T02_TIMING_cuda.json + T02_LONG_cuda.csv.
              Status MEASURED_PROJECTION only when T01 host_adapter_verified is true and every T02 part is PASS; else PROVISIONAL.
Outputs: v10/T02/T02_TIMING_<cpu|cuda>.json, v10/T02/T02_LONG_<cpu|cuda>.csv (COSTS_TEMPLATE columns first) in the repository;
raw files and logs under <out-root>/T02/runs/<device>_<job>/.  Data: 16 FIT cores (train) and 3 DEV cores (register), DataGuard.
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
                "cond_mean_minus_V", "cond_sd", "cond_grad_W_rel_err_mean", "source_file"]
CAP = A.M1_CAP_BINDING_TOLERANCE
TRAIN_CONFIGS = [
    {"method": "MF_perm1", "K": 16}, {"method": "MF_M0", "K": 16},
    {"method": "MF_M1", "K": 16, "m1_max_iterations": 30, "m1_tolerance": CAP},
    {"method": "MF_M1", "K": 16, "m1_max_iterations": 10, "m1_tolerance": CAP},
    {"method": "MF_M1", "K": 16, "m1_max_iterations": 100, "m1_tolerance": CAP},
    {"method": "MF_M1", "K": 16, "m1_max_iterations": 30, "m1_tolerance": A.M1_PROTOCOL_TOLERANCE, "record_only": True},
    {"method": "MF_perm1", "K": 8}, {"method": "MF_M0", "K": 8}, {"method": "MF_M1", "K": 8, "m1_max_iterations": 30, "m1_tolerance": CAP},
    {"method": "MF_perm1", "K": 32}, {"method": "MF_M0", "K": 32}, {"method": "MF_M1", "K": 32, "m1_max_iterations": 30, "m1_tolerance": CAP},
    {"method": "MF_perm1", "K": 16, "checks": "every_step"}, {"method": "MF_M0", "K": 16, "checks": "every_step"},
    {"method": "MLP_perm1"}, {"method": "MLP_perm8"}, {"method": "MLP_dense"},
    {"method": "MLP_dense", "dense_block": 32, "dense_checkpoint": True},
]
M1_RECORD_NOTE = ("RECORD of the protocol setting, not an M1 cost: absolute gap target 1e-4 from the package start W = 0; the first FW "
                  "gap equals ||Delta||_1 (m1_gap_initial); when it is below 1e-4 the head stops at iteration 0, W stays 0 and the "
                  "outer objective and its gradient are exactly 0 (degenerate = W stayed 0 on every timed step); excluded from BUDGET")
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
    if "m1_tolerance" in cfgd:
        lab += "_capbound" if cfgd["m1_tolerance"] == CAP else f"_tol{cfgd['m1_tolerance']:.0e}"
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
def part_train(device, run_dir: Path, rows: list, info: dict, gpu: str, guard: A.DataGuard) -> str:
    dev = torch.device(device)
    fit_all = A.fit_cores(guard, 1.0, SEED)
    fit = sorted(set(A.HostBatchSampler(fit_all, SEED).draw().cores))     # one host batch: 16 distinct FIT cores resident
    imgs = A.load_learned_inputs(guard, fit, {"FIT"}, "T02 train-step timing (16 FIT cores resident = one host batch)", dev)
    info["resident_fit_cores"] = fit
    status = "PASS"
    per_step = {}
    for cfgd in TRAIN_CONFIGS:
        label = train_label(cfgd)
        is_m1 = cfgd["method"] == "MF_M1"
        row = {"layer": "train_step", "config": label, "dataset": "BF_SHG", "method": cfgd["method"], "phase": "train_step", "device": gpu,
               "dtype": "float32", "n_positions": 128, "simplex_width": cfgd.get("K"), "encoder_included": True, "warmup": 0,
               "repeats": 0, "n_sc": 16, "n_pos": 128, "checks": cfgd.get("checks", "first_step") if cfgd["method"].startswith("MF_") else None,
               "m1_tolerance": cfgd.get("m1_tolerance"), "m1_max_iterations": cfgd.get("m1_max_iterations"),
               "m1_moment_passes_per_batch": 2 if is_m1 else None, "block_size": cfgd.get("dense_block"),
               "graph_retained": ({True: False, False: True}[bool(cfgd.get("dense_checkpoint"))] if cfgd["method"] == "MLP_dense" else None),
               "backward_recomputation": (bool(cfgd.get("dense_checkpoint")) if cfgd["method"] == "MLP_dense" else None),
               "R_permutations": {"MF_perm1": 1, "MLP_perm1": 1, "MLP_perm8": 8}.get(cfgd["method"])}
        t = None
        try:
            cfg = A.TrainConfig(method=cfgd["method"], seed=SEED, K=cfgd.get("K", 16), checks=cfgd.get("checks", "first_step"),
                                m1=A.M1Config(max_iterations=cfgd.get("m1_max_iterations", 30), tolerance=cfgd.get("m1_tolerance", A.M1_PROTOCOL_TOLERANCE)),
                                dense_block=cfgd.get("dense_block"), dense_checkpoint=cfgd.get("dense_checkpoint", False))
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
                gaps = [r["m1_gap"] for r in timed]
                row.update({"m1_iterations_median": float(np.median(its)), "m1_iterations_max": int(max(its)),
                            "m1_gap_median": float(np.median(gaps)), "m1_gap_max": float(max(gaps)),
                            "m1_converged_fraction": float(np.mean([r["m1_converged"] for r in timed])),
                            "m1_gap_initial_median": float(np.median([r["m1_gap_initial"] for r in timed])),
                            "m1_hvp_calls_median": float(np.median([r["m1_hvp_calls"] for r in timed])),
                            "m1_momentum_restarts_median": float(np.median([r["m1_momentum_restarts"] for r in timed])),
                            "m1_w_absmax_last": recs[-1]["w_absmax"], "degenerate": all(r["w_absmax"] == 0.0 for r in timed),
                            "solver_iterations": float(np.median(its)), "solver_gap": float(np.median(gaps))})
                if cfgd.get("record_only"):
                    row["notes"] = M1_RECORD_NOTE + "; " + row["notes"]
                    row["status"] = "completed_record_only"
                else:
                    row["notes"] = ("M1 with a binding iteration cap (tolerance 1e-300; require_converged False): the cost of the "
                                    "whole 10 / 30 / 100-iteration budget; gap and convergence logged per batch; " + row["notes"])
            per_step[label] = recs
        except Exception as e:  # noqa: BLE001
            row.update({"status": "oom" if is_oom(e) else "error", "notes": f"{type(e).__name__}: {str(e)[:400]}",
                        "traceback": traceback.format_exc()[-3000:]})
            status = status if is_oom(e) else "FAIL"
        rows.append({k: v for k, v in row.items() if k != "traceback"})
        info.setdefault("configs", {})[label] = {k: row.get(k) for k in ("status", "median_ms", "p90_ms", "repeats", "peak_allocated_bytes",
                                                                         "syncs_per_step", "m1_iterations_median", "m1_gap_median",
                                                                         "m1_converged_fraction", "m1_gap_initial_median", "degenerate",
                                                                         "notes", "traceback")}
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
        out[name] = ({k: json.loads(f.read_text()).get(k) for k in ("overall_status", "host_adapter_verified", "pending_parts", "failed_parts",
                                                                    "blocked_parts", "provenance_mismatches", "generated_utc")}
                     if f.exists() else {"overall_status": "NOT_RUN", "note": "file missing"})
    return out


def run(args) -> int:
    pin = A.check_job_pin()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable: nothing run", flush=True)
        return 3
    devtype = torch.device(device).type
    label = os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = args.out_root / "T02" / "runs" / f"{devtype}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = A.environment(device)
    gpu = env.get("gpu_name") or "CPU"
    guard = A.DataGuard()
    rows, parts = [], {}
    for part in [p for p in args.parts.split(",") if p]:
        info: dict = {}
        t0 = time.time()
        try:
            if part == "head":
                st = part_head(device, run_dir, rows, info, gpu)
            elif part == "head_extra":
                st = part_head_extra(device, run_dir, rows, info, gpu)
            elif part == "train":
                st = part_train(device, run_dir, rows, info, gpu, guard)
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
    for name in ("head", "head_extra", "train", "register"):
        parts.setdefault(name, {"status": "NOT_RUN"})
    long_name, timing_name = f"T02_LONG_{devtype}.csv", f"T02_TIMING_{devtype}.json"
    out = {"schema": "V10-T02-TIMING-v2", "generated_utc": A.utc_now(), "environment": env, "job_pin": pin, "run_dir": str(run_dir),
           "device_type": devtype, "t01_status_at_run": t01_status(),
           "validity_note": "timings are valid only for implementations that pass T01; the package benchmark marks a cell "
                            "correctness_failed when dense / moments disagree and does not time it",
           "status_vocabulary": "PASS (ran; OOM rows recorded, not dropped) / FAIL (error, correctness failure or exception) / NOT_RUN",
           "parts": parts, "data_reads": guard.reads, "seed": SEED,
           "protocol": {"grid": GRID, "head_warmup": HEAD_WARMUP, "head_repeats": HEAD_REPEATS, "head_block": HEAD_BLOCK,
                        "train_warmup": TRAIN_WARMUP, "train_timed": TRAIN_TIMED, "train_phase_steps": TRAIN_PHASE,
                        "train_configs": TRAIN_CONFIGS, "registration_energies": REG_ENERGIES, "registration_cores": REG_CORES,
                        "registration_perturbations": REG_PERTS, "m1_cap_binding_tolerance": CAP,
                        "m1_protocol_tolerance": A.M1_PROTOCOL_TOLERANCE}, "long_csv": f"v10/T02/{long_name}", "n_rows": len(rows)}
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


# ------------------------------------------------------------------------------------------------------------ budget + ledger
PROTOCOL_COUNTS = {"train_steps_per_unit": 3000, "checkpoints_per_unit": 12, "bf_cal_cores": 8, "bf_dev_cores": 32, "perturbations_per_image": 10,
                   "epfl_cal_windows": 10, "epfl_dev_windows": 74, "training_seeds": [1001, 1002, 1003], "tuning_seed": 1017,
                   "cal_candidates_per_family": 9}
UNIT_METHOD_MAP = {"MF_VCS_perm1": ("MF_perm1_K16", "MF_perm1_K16"), "MF_VCS_M0": ("MF_M0_K16", "MF_moments_K16"),
                   "MF_VCS_M1": ("MF_M1_K16_it30_capbound", "MF_moments_K16"), "MLP_VCS_perm1": ("MLP_perm1", "MLP_perm1_host"),
                   "MLP_VCS_perm8": ("MLP_perm8", None)}
V9_OUT = Path("/projects/EEG-foundation-model/yinghao/outputs_v9")
V9_COST_SOURCES = {   # read-only at budget time; device / job labels from v10/T00/HOST_ADAPTER_AUDIT.md section 7 (not in the JSONs)
    "V9-G1-vcs-frac1-eps0-s901": {"train": V9_OUT / "reg/units/V9-G1-vcs-frac1-eps0-s901/run/train.json",
                                  "results": V9_OUT / "reg/units/V9-G1-vcs-frac1-eps0-s901/p2/results/vcs_s901.json",
                                  "device_job_per_T00_audit": "RTX PRO 6000 Blackwell, job 1041939", "registrations": 12 * 8 * 10 + 32 * 10},
    "V9-G2-vcs-frac1-eps0-s901": {"results": V9_OUT / "reg_epfl/units/V9-G2-vcs-frac1-eps0-s901/p3/results/vcs_s901.json",
                                  "device_job_per_T00_audit": "L40S, job 1042139", "registrations": 12 * 10 * 10 + 74 * 10}}


def v9_cost_facts() -> dict:
    """V9 cost fields read (read-only) from the V9 train / result JSONs, with path and sha256; nothing hard-coded."""
    out = {"rule": "read-only from the V9 JSONs at budget time; registrations = 12 checkpoints x CAL x 10 + DEV x 10 (host protocol)"}
    for unit, src in V9_COST_SOURCES.items():
        e = {"device_job_per_T00_audit": src["device_job_per_T00_audit"], "registrations": src["registrations"]}
        try:
            if "train" in src:
                j = json.loads(src["train"].read_text())
                e.update({"train_steps": j.get("steps"), "train_seconds": j.get("seconds"), "train_peak_gib": j.get("peak_gib"),
                          "train_json": str(src["train"]), "train_json_sha256": A.sha256_file(src["train"])})
            j = json.loads(src["results"].read_text())
            e.update({"s_per_registration_cal": j.get("cal_sec_per_registration"), "s_per_registration_dev": j.get("dev_sec_per_registration"),
                      "registration_phase_seconds": j.get("seconds_total"), "device_in_json": j.get("device"),
                      "results_json": str(src["results"]), "results_json_sha256": A.sha256_file(src["results"])})
        except Exception as exc:  # noqa: BLE001 - a missing source is recorded, never filled in
            e["status"] = f"source unreadable: {type(exc).__name__}: {str(exc)[:200]}"
        out[unit] = e
    return out


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
    A.check_job_pin()
    src = args.repo_out / "T02_TIMING_cuda.json"
    units = list(csv.DictReader((A.PACKAGE / "configs" / "EXPERIMENT_UNITS.csv").read_text().splitlines()))
    if not src.exists():
        A.write_json_atomic(args.repo_out / "BUDGET.json", {"schema": "V10-T02-BUDGET-v2", "status": "NOT_RUN", "reason": f"{src} missing"})
        return 1
    t02 = json.loads(src.read_text())
    rows = list(csv.DictReader((args.repo_out / "T02_LONG_cuda.csv").read_text().splitlines()))
    fnum = lambda x: float(x) if x not in (None, "", "None") else None   # noqa: E731
    train = {r["config"]: fnum(r["median_ms"]) for r in rows if r["layer"] == "train_step" and r["status"] == "completed"}
    reg = {}
    for r in rows:
        if r["layer"] == "registration" and r["phase"] == "registration" and r["status"] == "completed" and r["warmup_row"] == "False":
            reg.setdefault(r["config"], []).append(fnum(r["elapsed_seconds"]))
    reg_med = {k: float(np.median(v)) for k, v in reg.items()}
    c = PROTOCOL_COUNTS
    n_reg_bf = c["checkpoints_per_unit"] * c["bf_cal_cores"] * c["perturbations_per_image"] + c["bf_dev_cores"] * c["perturbations_per_image"]
    n_reg_ep = c["checkpoints_per_unit"] * c["epfl_cal_windows"] * c["perturbations_per_image"] + c["epfl_dev_windows"] * c["perturbations_per_image"]
    per_unit = {}
    for method, (tcfg, rcfg) in UNIT_METHOD_MAP.items():
        step_ms, reg_s = train.get(tcfg), reg_med.get(rcfg) if rcfg else None
        e = {"train_config": tcfg, "registration_energy": rcfg, "train_step_ms_median": step_ms, "s_per_registration_median": reg_s}
        if step_ms is not None:
            e["train_s"] = c["train_steps_per_unit"] * step_ms / 1000
        if reg_s is not None:
            e["registration_s_bf"] = n_reg_bf * reg_s
            e["registration_s_epfl_proxy"] = n_reg_ep * reg_s
        if step_ms is not None and reg_s is not None:
            e["gpu_h_bf_unit"] = (e["train_s"] + e["registration_s_bf"]) / 3600
            e["gpu_h_epfl_unit_proxy"] = (e["train_s"] + e["registration_s_epfl_proxy"]) / 3600
        e["status"] = "measured" if (step_ms is not None and reg_s is not None) else ("partially_measured" if (step_ms or reg_s) else "not_measured")
        per_unit[method] = e
    measured = [m for m, e in per_unit.items() if e["status"] == "measured"]
    seeds = len(c["training_seeds"])
    bf_h = sum(per_unit[m]["gpu_h_bf_unit"] for m in measured) * seeds
    ep_h = sum(per_unit[m]["gpu_h_epfl_unit_proxy"] for m in measured) * seeds
    tune_h = sum(c["cal_candidates_per_family"] * (per_unit[m]["train_s"] + c["checkpoints_per_unit"] * c["bf_cal_cores"] * c["perturbations_per_image"]
                                                   * per_unit[m]["s_per_registration_median"]) / 3600 for m in measured)
    total = bf_h + ep_h + tune_h
    t01 = t01_status()
    verified = t01["TORCH_VALIDATION.json"].get("host_adapter_verified") is True
    t02_parts = {k: v.get("status") for k, v in t02["parts"].items()}
    parts_ok = all(t02_parts.get(p) == "PASS" for p in ("head", "head_extra", "train", "register"))
    status = ("MEASURED_PROJECTION" if (measured and verified and parts_ok) else "PROVISIONAL" if measured else "NOT_RUN")
    out = {"schema": "V10-T02-BUDGET-v2", "status": status, "generated_utc": A.utc_now(),
           "status_rule": "MEASURED_PROJECTION only when T01 host_adapter_verified is true and every T02 part is PASS; otherwise "
                          "PROVISIONAL (numbers measured, validity not established) or NOT_RUN (nothing measured)",
           "status_inputs": {"t01": t01, "t02_parts": t02_parts},
           "source": {"t02_timing": str(src), "sha256": A.sha256_file(src), "job": t02["environment"].get("slurm_job_id"),
                      "device": t02["environment"].get("gpu_name"), "gpu_total_bytes": t02["environment"].get("gpu_total_bytes"),
                      "git_commit": t02["environment"].get("git_commit"), "code_hash": t02["environment"].get("code_hash_src_scripts_v10")},
           "measured": {"train_step_ms_median": train, "s_per_registration_median": reg_med,
                        "head_only": "v10/T02/T02_LONG_cuda.csv layer=head / head_extra / perm_conditional_variance"},
           "protocol_counts": c, "registrations_per_unit": {"BF_SHG": n_reg_bf, "EPFL_RGB_NIR": n_reg_ep},
           "per_unit_projection": per_unit,
           "grid_projection_measured_methods": {"methods": measured, "seeds": seeds, "BF_SHG_gpu_h": bf_h, "EPFL_gpu_h_proxy": ep_h,
                                                "CAL_tuning_gpu_h_BF": tune_h, "total_gpu_h": total,
                                                "wall_h_at_concurrency": {str(k): total / k for k in (1, 2, 4, 8)}},
           "not_measured_in_T02": {"InfoNCE_host": "native InfoNCE energy undecided (T00 D3); host trains InfoNCE critics then scores J(tanh f)",
                                   "CoMIR_native": "external system: own training / latent / registration costs (see V9 G1 / G2 results)",
                                   "RuLSIF_kernel": "kernel fit + registration not implemented in V10 phase 1",
                                   "MI_native / NMI_native / MIND_2D_adapted": "non-learned; registration cost only (V9 measured MI / NMI)",
                                   "MLP_VCS_perm8 registration energy": "a perm8 registration energy is not implemented in phase 1"},
           "v9_reference_costs": v9_cost_facts(),
           "resources": {"per_user_gpu_cap": 8, "normal_qos_submit_cap": 30,
                         "v10_queue_rule_2026_10_16": "V10 holds at most 2 normal-QOS jobs while V9-N2 keeps the count at 27; never above 29",
                         "queue_snapshot": squeue_snapshot()},
           "assumptions": ["one GPU per unit; the measured GPU type only (other types scale differently: V9 RTX PRO 6000 ~ 2x L40S)",
                           "training: 3000 steps per unit (host), checkpoint every 250 steps, step time = encoder-inclusive median of T02",
                           "registration: CAL selection over 12 checkpoints x CAL images x 10 perturbations + DEV x 10 (host protocol); "
                           "untrained-model registration timing used (search cost is weight-independent: fixed grid + 3 x 100 Adam steps)",
                           "EPFL: same step / registration cost per window as BF-SHG (512 px windows, same searcher) - proxy, not measured",
                           "CAL tuning: 9 candidates per measured family on BF-SHG, each trained once and CAL-registered",
                           "MF_VCS_M1 at max_iterations 30 with a binding cap (the whole budget every batch; candidates 10 / 30 / "
                           "100 in T02_LONG_cuda.csv); the protocol gap target 1e-4 is degenerate from W = 0 at host init (open "
                           "owner decision), so the capped cost is an upper bound per batch, not the protocol run",
                           "MF step times: W-box / permutation checks on the first step only; SceneBatch validation included",
                           "no I/O, queue wait or failed-job reruns included"]}
    A.write_json_atomic(args.repo_out / "BUDGET.json", out)
    status_by = {}
    for r in rows:
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
    print(json.dumps({"BUDGET": out["status"], "measured_methods": measured, "total_gpu_h": total}), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", choices=("cpu", "cuda"), required=True)
    r.add_argument("--parts", default="head,head_extra,train,register")
    r.add_argument("--out-root", type=Path, default=A.OUT_ROOT)
    r.add_argument("--repo-out", type=Path, default=REPO_T02)
    b = sub.add_parser("budget")
    b.add_argument("--repo-out", type=Path, default=REPO_T02)
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else budget(args)  # both refuse when the job pin (V10_EXPECT_CODE) does not match


if __name__ == "__main__":
    sys.exit(main())
