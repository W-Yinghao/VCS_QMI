"""R22-A: the SGR student of N2F/N2M/N2P-C1 trained with the VCS objective as its main loss (R22_ALTERNATIVES_CN.md section 2.A).

Same host, data, recipe and seeds as the hinge students (HOST_LOCK, Flickr30k observed pool of 145000 pairs, A = 14500, rho = 0.4,
40 epochs, batch 128, Adam 2e-4, lr x0.1 at epoch 30, clip 2.0 on the host parameters, loss_scale "sum", the V9 precision settings,
seeds 901/902/903); only the loss changes (losses.py).  Reads train + full DEV; there is no TEST loader.  Inputs and helpers come from
the N2P-C1 adapter (v9_n2p/c_server/c1_student.py) unchanged.

  python -I -B r22_student.py smoke     --root ROOT --seed 901 --arms VCS,VCSW,BCE,NCE,HINGE --steps 200
  python -I -B r22_student.py run       --root ROOT --seed SEED --arm ARM          40 epochs; resubmit the same line after a time limit
  python -I -B r22_student.py identity  --root ROOT                                hinge-baseline identity check (no torch)
  python -I -B r22_student.py summarize --root ROOT                                per-seed table and paired differences
R22-A2 (section A2, 2026-10-22): arms VCS_SHARP, VCSW_SHARP, BCE_SHARP (losses_sharp.py: f = 20 s + b, b0 -10, b in its own Adam group,
base lr 1e-3); same commands, same output root (runs/s<seed>__<ARM>); summarize adds the A2 rows, A2-vs-hinge / A2-vs-A differences and
the pre-registered seed-expansion rule.
R22-A3 (section A3, 2026-10-23): arms NCE_VCSW, NCE_BCEW (losses_nce_w.py: A's symmetric InfoNCE, tau 0.05, the positive pair's row and
column CE terms x v_i / mean_v_full with the N2F VCS_Weight / BCE_Weight files; negatives unweighted); unweighted NCE reused from A
(a3-identity writes A3_IDENTITY.json: NCE reuse, weight-file identity, the N2F hinge+BCE baseline); summarize adds the A3 rows, paired
differences and the pre-registered expansion rule.
  python -I -B r22_student.py a3-identity --root ROOT                              (no GPU)
Exit status: 0 done, 4 check failed, 9 forbidden file access, 143 terminated (last complete epoch kept).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "v9_n2p" / "c_server"))
sys.path.insert(0, str(HERE))
import c1_student as c1  # noqa: E402  (paths, precision, loaders, records; also puts REPO/src on sys.path)
import trace_open  # noqa: E402

SEEDS = (901, 902, 903)
MAIN_ARMS = ("VCS", "VCSW", "BCE", "NCE")
A2_ARMS = ("VCS_SHARP", "VCSW_SHARP", "BCE_SHARP")                  # R22-A2 (losses_sharp.py)
A3_ARMS = ("NCE_VCSW", "NCE_BCEW")                                   # R22-A3 (losses_nce_w.py)
WEIGHT_FILE = {"VCS": "Unweighted_A", "VCSW": "VCS_Weight", "BCE": "Unweighted_A", "NCE": "Unweighted_A", "HINGE": "Unweighted_A",
               "VCS_SHARP": "Unweighted_A", "VCSW_SHARP": "VCS_Weight", "BCE_SHARP": "Unweighted_A",
               "NCE_VCSW": "VCS_Weight", "NCE_BCEW": "BCE_Weight"}                                       # N2F files
N2F = Path("/home/infres/yinwang/CS_QMI/handoff/v9_full_20261017")
N2M = Path("/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1")
RECALLS = c1.RECALLS
Refused = c1.Refused


def weight_inputs(seed, wname):
    from v9.n2.weights import read_weight_lock
    base = c1.OLD / f"s{seed}"
    rows_p, w_p = base / "pairs.observed_pairs.json", base / "weights" / f"{wname}.csv"
    frozen = json.loads((N2F / "FROZEN.json").read_text())["input_sha256"]
    for p in (rows_p, w_p):
        if c1.sha256(p) != frozen[str(p)]:
            raise Refused(4, f"{p} differs from the N2F frozen input")
    lock = json.loads(Path(str(w_p) + ".lock.json").read_text())
    return {"rows": json.loads(rows_p.read_text()), "weights": read_weight_lock(w_p), "lock": lock,
            "paths": {"observed": str(rows_p), "weights": str(w_p)}, "sha256": {"observed": c1.sha256(rows_p), "weights": lock["sha256"]}}


def make_student(ctx, inputs, seed, arm, out, stop_epoch_steps=None, collate=None, trace_hits=None, validate_hook=None):
    from losses import OBJECTIVES
    from v9.n2.train import WeightedStudent
    if arm in A2_ARMS:                                                   # imported only for the A2 arms: the A arms run as before
        from losses_sharp import OBJECTIVES_SHARP
        OBJECTIVES = {**OBJECTIVES, **OBJECTIVES_SHARP}
    if arm in A3_ARMS:                                                   # likewise for the A3 arms
        from losses_nce_w import OBJECTIVES_A3
        OBJECTIVES = {**OBJECTIVES, **OBJECTIVES_A3}
        if inputs["lock"].get("record", {}).get("method") != WEIGHT_FILE[arm]:
            raise Refused(4, f"{inputs['paths']['weights']}: lock method {inputs['lock'].get('record', {}).get('method')!r} is not {WEIGHT_FILE[arm]}")

    class Student(WeightedStudent):
        def validate(self, host):
            t0 = time.time()
            result = super().validate(host)
            c1.require_clean_trace(trace_hits)
            if validate_hook is not None:
                validate_hook(time.time() - t0)
            return result

    spec = c1.protocol()
    return Student(ctx["cfg"], inputs["rows"], inputs["weights"], collate or ctx["collate"], ctx["dev_images"], ctx["dev_captions"], out, seed,
                   spec["lr_mult"], device="cuda", weight_lock=inputs["lock"], formal=True, loss_scale=spec["loss_scale"],
                   stop_epoch_steps=stop_epoch_steps, objective=OBJECTIVES[arm]())


def gpu_setup():
    precision = c1.configure_precision(c1.protocol()["precision"])
    import torch
    if not torch.cuda.is_available():
        raise Refused(4, "an allocated CUDA device is required")
    return precision, c1.environment()


# ------------------------------------------------------------------------------------------------ smoke
def smoke(root, seed, arms, steps, trace_hits=None):
    precision, env = gpu_setup()
    ctx = c1.load_context()
    tag = f"{os.environ.get('SLURM_JOB_ID', 'nojob')}_{int(time.time())}"
    report = {"schema": "r22-vl-smoke-v1", "utc": c1.utc(), "seed": seed, "steps": steps, "environment": env, "precision_effective": precision, "arms": {}}
    for arm in arms:
        out = root / "smoke" / tag / f"s{seed}__{arm}"
        inputs = weight_inputs(seed, WEIGHT_FILE[arm])
        stamps, val_s = [], []

        def timed_collate(rows, real=ctx["collate"]):
            stamps.append(time.time())
            return real(rows)

        student = make_student(ctx, inputs, seed, arm, out, stop_epoch_steps=steps, collate=timed_collate, trace_hits=trace_hits,
                               validate_hook=val_s.append)
        t0 = time.time()
        student.fit(1, log_every=1)
        log = json.loads((out / "train_log.json").read_text())
        losses = [r["loss"] for r in log["steps"]]
        gaps = [b - a for a, b in zip(stamps[10:-1], stamps[11:])]       # step k -> k+1 after 10 warm-up steps (loss.item() syncs each step)
        k = min(20, len(losses) // 2)
        gn = sorted(r["grad_norm_pre_clip"] for r in log["steps"] if r.get("grad_norm_pre_clip") is not None)
        rec = {"steps_run": log["epochs"][0]["steps_in_epoch"], "loss_first": losses[0], "loss_last": losses[-1],
               "loss_mean_first": sum(losses[:k]) / k, "loss_mean_last": sum(losses[-k:]) / k,
               "decreasing": sum(losses[-k:]) / k < sum(losses[:k]) / k, "all_finite": all(math.isfinite(x) for x in losses),
               "step_seconds_median": sorted(gaps)[len(gaps) // 2] if gaps else None, "dev_eval_seconds": val_s[0] if val_s else None,
               "dev": log["epochs"][0]["dev"], "last_step_stats": {k2: v for k2, v in log["steps"][-1].items() if k2 not in ("step", "epoch", "lr")},
               "grad_norm_pre_clip_median": gn[len(gn) // 2] if gn else None,
               "clipped_fraction": (sum(g > 2.0 for g in gn) / len(gn)) if gn else None,
               "checkpoints_written": (out / "best.pt").is_file() and (out / "last.pt").is_file(), "seconds": round(time.time() - t0, 1),
               "objective": json.loads((out / "run_manifest.json").read_text()).get("objective")}
        rec["ok"] = rec["all_finite"] and rec["steps_run"] == steps and math.isfinite(rec["dev"]["rsum"]) and rec["checkpoints_written"] \
            and (rec["decreasing"] or arm == "HINGE")
        if arm in A2_ARMS:                                               # b moves, the epoch record (hard-negative fraction, T quantiles) exists
            oe = log["epochs"][0].get("objective_epoch") or {}
            b0 = json.loads((out / "run_manifest.json").read_text())["objective"]["b0"]
            rec.update(b0=b0, b_first_step=log["steps"][0]["b"], b_last=oe.get("b"), objective_epoch=oe,
                       hard_neg_frac_first_step=log["steps"][0].get("hard_neg_frac_Q"), hard_neg_frac_last_step=log["steps"][-1].get("hard_neg_frac_Q"))
            rec["b_moved"] = oe.get("b") is not None and abs(oe["b"] - b0) > 1e-3
            rec["ok"] = rec["ok"] and rec["b_moved"] and all(math.isfinite(oe.get(k, float("nan"))) for k in ("hard_neg_frac_Q", "J_hat", "EQ_T_mean"))
        report["arms"][arm] = rec
        print(json.dumps({"arm": arm, **{x: rec[x] for x in ("ok", "loss_mean_first", "loss_mean_last", "step_seconds_median", "dev_eval_seconds")},
                          "dev_rsum": rec["dev"]["rsum"], "gpu": env["gpu"]}), flush=True)
    report["checks_pass"] = all(r["ok"] for r in report["arms"].values())
    return report, root / "smoke" / tag


# ------------------------------------------------------------------------------------------------ full run
def run(root, seed, arm, trace_hits=None):
    out = root / "runs" / f"s{seed}__{arm}"
    if (out / "unit_result.json").is_file():
        return json.loads((out / "unit_result.json").read_text())["status"]
    precision, env = gpu_setup()
    ctx = c1.load_context()
    spec, cfg = c1.protocol(), ctx["cfg"]
    inputs = weight_inputs(seed, WEIGHT_FILE[arm])
    t0 = time.time()
    c1.attempt_log(out, {"event": "start", "arm": arm, "environment": env, "precision_effective": precision,
                         "resumes_from_last_pt": (out / "last.pt").is_file()})
    try:
        student = make_student(ctx, inputs, seed, arm, out, trace_hits=trace_hits)
        checks = c1.pool_checks(student, inputs["rows"], inputs["weights"], seed, cfg, {**c1.PROTOCOL_POOL, "updates": spec["updates"]})
        if not checks["ok"]:
            raise Refused(4, f"pool checks failed: {checks}")
        student.fit(cfg.num_epochs, log_every=20)
        post, done = c1.log_checks(out, cfg.num_epochs, c1.PROTOCOL_POOL["steps_per_epoch"])
        if not post["ok"]:
            raise Refused(4, f"completed log fails the checks {post}")
        c1.require_clean_trace(trace_hits)
    except BaseException as exc:
        status = "INTERRUPTED" if isinstance(exc, (SystemExit, KeyboardInterrupt)) else "FAILED"
        c1.attempt_log(out, {"event": "end", "status": status, "elapsed_s": round(time.time() - t0, 1), "error": repr(exc)})
        raise
    c1.attempt_log(out, {"event": "end", "status": "COMPLETE", "elapsed_s": round(time.time() - t0, 1)})
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    c1.write_json(out / "unit_result.json", {
        "schema": "r22-vl-unit-v1", "unit_id": f"R22A{'2' if arm in A2_ARMS else '3' if arm in A3_ARMS else ''}_f30k_r04_s{seed}_{arm}", "seed": seed, "arm": arm, "status": "COMPLETE_VERIFIED",
        "objective": student.manifest.get("objective"), "weights": WEIGHT_FILE[arm], "inputs": inputs["paths"], "inputs_sha256": inputs["sha256"],
        "initial_model_sha256": c1.initial_model_sha256(cfg, seed), "data_order_sha256": checks["data_order_sha256"],
        "pool_checks": checks, "log_checks": post, "done": done, "environment": env, "precision_effective": precision,
        "artifacts_sha256": {n: c1.sha256(out / n) for n in ("best.pt", "last.pt", "train_log.json", "run_manifest.json", "done.json")},
        "source": {"repo_head": head, "losses_sha256": c1.sha256(HERE / "losses.py"), "adapter_sha256": c1.sha256(__file__),
                   "train_py_sha256": c1.sha256(REPO / "src/v9/n2/train.py")},
        "dev_only": True, "test_read": False, "label": "PILOT/DEVELOPMENT",
        **({"losses_sharp_sha256": c1.sha256(HERE / "losses_sharp.py")} if arm in A2_ARMS else {}),
        **({"losses_nce_w_sha256": c1.sha256(HERE / "losses_nce_w.py")} if arm in A3_ARMS else {})})
    return "COMPLETE_VERIFIED"


# ------------------------------------------------------------------------------------------------ hinge baselines
def endpoints(log):
    eps = log["epochs"]
    rs = [e["dev"]["rsum"] for e in eps]
    b = max(range(len(rs)), key=lambda i: (rs[i], -i))                  # earliest exact tie
    return {"best_rsum": rs[b], "best_epoch": eps[b]["epoch"], **{k: eps[b]["dev"][k] for k in RECALLS}, "last_rsum": rs[-1],
            "mean_last5_rsum": sum(rs[-5:]) / 5, "epochs": len(eps)}


def baseline_dirs(seed):
    groups = json.loads((N2F / "GROUPS.json").read_text())

    def unit(method):
        gid = next(g["id"] for g in groups if g["seed"] == seed and {"method": method, "scale": "sum"} in g["units"])
        return N2F / "results" / gid / f"{method}_sum"

    return {"HINGE_ConstantU_N2M": N2M / "runs" / f"s{seed}__ConstantU", "HINGE_VCSW_N2M": N2M / "runs" / f"s{seed}__VCS",
            "HINGE_Unweighted_N2F": unit("Unweighted_A"), "HINGE_VCSW_N2F": unit("VCS_Weight")}


def identity(root):
    """Same code (all host modules except the train.py objective branch), HOST_LOCK, inputs, recipe, seeds, environment and precision?"""
    frozen = json.loads((N2F / "FROZEN.json").read_text())
    runtime = json.loads((N2M / "runtime" / "RUNTIME.json").read_text())["inherited_sha256"]
    rep = {"schema": "r22-vl-hinge-identity-v1", "utc": c1.utc(), "checks": {}, "baselines": {}}
    code = {}
    for rel in ("src/v9/n2/data.py", "src/v9/n2/eval.py", "src/v9/n2/fixtures.py", "src/v9/n2/noise.py", "src/v9/n2/sgr_host.py",
                "src/v9/n2/teacher.py", "src/v9/n2/weights.py", "src/v9/host_config.py", "src/v9/reference_core.py"):
        mine = c1.sha256(REPO / rel)
        code[rel] = {"r22": mine, "n2f": frozen["code_sha256"].get("source/" + rel), "n2m": runtime.get(rel)}
        code[rel]["same"] = mine == code[rel]["n2f"] == code[rel]["n2m"]
    base_train = subprocess.run(["git", "-C", str(REPO), "show", "01dab37d:src/v9/n2/train.py"], capture_output=True).stdout
    code["src/v9/n2/train.py (branch base 01dab37d)"] = {"r22_base": hashlib.sha256(base_train).hexdigest(),
                                                       "n2f": frozen["code_sha256"]["source/src/v9/n2/train.py"],
                                                       "n2m_inherited": runtime["src/v9/n2/train.py"],
                                                       "note": "R22 adds an objective branch; objective=None runs the original hinge "
                                                               "(test_r22.py: bitwise equal to the frozen N2F train.py on a CPU fixture). "
                                                               "N2M ran this file plus its telemetry patch (objective, optimiser, batches, RNG unchanged)."}
    code["src/v9/n2/train.py (branch base 01dab37d)"]["same"] = len({code["src/v9/n2/train.py (branch base 01dab37d)"][k]
                                                                    for k in ("r22_base", "n2f", "n2m_inherited")}) == 1
    rep["checks"]["code"] = code
    hl = c1.sha256(c1.HOST_LOCK)
    rep["checks"]["host_lock"] = {"r22": hl, "n2f": frozen["code_sha256"]["source/HOST_LOCK.yaml"], "n2m": runtime["HOST_LOCK.yaml"],
                                  "same": hl == frozen["code_sha256"]["source/HOST_LOCK.yaml"] == runtime["HOST_LOCK.yaml"]}
    rep["checks"]["token_cache"] = {"r22": str(c1.TOKENS), "n2f": str(N2F / "source" / "token_cache"), "same": c1.TOKENS == N2F / "source" / "token_cache"}
    data_ok = all(Path(p).stat().st_size == i["size"] and Path(p).stat().st_mtime_ns == i["mtime_ns"] for p, i in frozen["data_files"].items())
    rep["checks"]["data_files_unchanged_since_n2f"] = data_ok and str(c1.DATA) == frozen["data"]
    seeds = {}
    for seed in SEEDS:
        obs = c1.OLD / f"s{seed}" / "pairs.observed_pairs.json"
        s_obs = c1.sha256(obs)
        d = baseline_dirs(seed)
        idm = {k: json.loads((d[k] / "identity.json").read_text()) for k in ("HINGE_ConstantU_N2M", "HINGE_VCSW_N2M")}
        idf = {k: json.loads((d[k] / "full_result.json").read_text())["identity"] for k in ("HINGE_Unweighted_N2F", "HINGE_VCSW_N2F")}
        wv, wu = (c1.OLD / f"s{seed}" / "weights" / f"{n}.csv" for n in ("VCS_Weight", "Unweighted_A"))
        seeds[seed] = {
            "observed_same": s_obs == frozen["input_sha256"][str(obs)] == idm["HINGE_ConstantU_N2M"]["observed_sha256"] == idm["HINGE_VCSW_N2M"]["observed_sha256"],
            "vcs_weights_same_as_n2m_hinge_vcs": c1.sha256(wv) == frozen["input_sha256"][str(wv)] == idm["HINGE_VCSW_N2M"]["weights_sha256"],
            "unweighted_file_frozen": c1.sha256(wu) == frozen["input_sha256"][str(wu)],
            "n2m_recipe": all(i["scale"] == "sum" and i["epochs"] == 40 and i["learning_rate_multiplier"] == 1.0 and i["seed"] == seed and not i["oracle"]
                              and i["test_read"] is False for i in idm.values()),
            "n2f_recipe": all(i["scale"] == "sum" and i["epochs"] == 40 and i["learning_rate_multiplier"] == 1.0 and i["seed"] == seed and not i["oracle"]
                              for i in idf.values())}
        for k, p in d.items():
            log = json.loads((p / "train_log.json").read_text())
            hw = (json.loads((p / "full_result.json").read_text()).get("hardware") or {})
            rep["baselines"][f"s{seed}__{k}"] = {"seed": seed, "arm": k, "dir": str(p), **endpoints(log), "gpu": hw.get("gpu"),
                                                 "python": hw.get("python"), "torch": hw.get("torch"), "numpy": hw.get("numpy"),
                                                 "deterministic_algorithms": hw.get("deterministic_algorithms"), "tf32": hw.get("tf32")}
    rep["checks"]["per_seed"] = seeds
    envs = {(b["python"], b["torch"], b["deterministic_algorithms"], b["tf32"]) for b in rep["baselines"].values()}
    rep["checks"]["environment"] = {"baselines": sorted(map(str, envs)), "r22_expected": str(("3.12.12", "2.9.0+cu128", True, False)),
                                    "same": envs == {("3.12.12", "2.9.0+cu128", True, False)}}
    rep["checks"]["hardware_note"] = ("N2M hinge arms ran on RTX PRO 6000 only and N2F hinge VCS_Weight_sum on L40S; R22 arms use "
                                      "RTX6000PRO/L40S/H100 as scheduled, so a cross-arm difference includes a hardware component; "
                                      "N2F (L40S) vs N2M (RTX) for the identical hinge+VCS recipe differ by +1.32/+0.36/+2.96 rSum")
    ok = all(v["same"] for v in code.values()) and rep["checks"]["host_lock"]["same"] and rep["checks"]["token_cache"]["same"] \
        and rep["checks"]["data_files_unchanged_since_n2f"] and all(all(v.values()) for v in seeds.values()) and rep["checks"]["environment"]["same"]
    rep["status"] = "REUSE_IDENTICAL_EXCEPT_HARDWARE" if ok else "NOT_IDENTICAL_RERUN_NEEDED"
    c1.write_json(root / "HINGE_IDENTITY.json", rep)
    return rep


# ------------------------------------------------------------------------------------------------ R22-A3 identity
A_COMMIT, A2_COMMIT = "df8a0707", "ccb5f331"        # R22-A launch (its code ran A's NCE arm); R22-A2 launch (train.py with the A2 hooks)
A3_HINGE = {"NCE_VCSW": "VCS_Weight", "NCE_BCEW": "BCE_Weight"}     # the N2F hinge arm (method, scale sum) that used the same weight file
ENV_KEYS = ("python", "torch", "cuda", "cudnn", "numpy")


def _git_blob_sha(commit, rel):
    out = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{rel}"], capture_output=True, check=True).stdout
    return hashlib.sha256(out).hexdigest()


def _n2f_unit(seed, method):
    groups = json.loads((N2F / "GROUPS.json").read_text())
    gid = next(g["id"] for g in groups if g["seed"] == seed and {"method": method, "scale": "sum"} in g["units"])
    return N2F / "results" / gid / f"{method}_sum"


def a3_identity(root):
    """A3 reuses A's unweighted NCE: same code path (losses.py unchanged, every host/adapter module unchanged since the A launch except
    train.py's A2 hooks, which an objective without param_groups/epoch_stats never reaches), HOST_LOCK, inputs, recipe, seed, environment;
    after an A3 run also the same initial model, batch order and run manifest.  The A3 weight files are the N2F hinge arms' files."""
    changed = subprocess.run(["git", "-C", str(REPO), "diff", "--name-only", A_COMMIT], capture_output=True, text=True, check=True).stdout.split()
    code = {"changed_since_a_launch": changed,
            "only_r22_dir_and_train_py": all(x.startswith("v9_r22/vcsloss/") or x == "src/v9/n2/train.py" for x in changed),
            "losses_py_unchanged": "v9_r22/vcsloss/losses.py" not in changed,
            "train_py_is_a2_launch": c1.sha256(REPO / "src/v9/n2/train.py") == _git_blob_sha(A2_COMMIT, "src/v9/n2/train.py"),
            "note": "train.py differs from the A launch by the A2 hooks only (test_r22_a2.py: A arms bitwise against the df8a0707 train.py; "
                    "test_r22_a3.py: NCE_VCSW with v = 1 bitwise equal to A's NCE through the trainer)"}
    code["same"] = code["only_r22_dir_and_train_py"] and code["losses_py_unchanged"] and code["train_py_is_a2_launch"]
    rep = {"schema": "r22-vl-a3-identity-v1", "utc": c1.utc(), "a_commit": A_COMMIT, "a2_commit": A2_COMMIT, "code": code,
           "weights": {}, "nce_reuse": {}, "baselines": {}}
    a_train_sha = _git_blob_sha(A_COMMIT, "src/v9/n2/train.py")
    for seed in SEEDS:
        for arm, method in A3_HINGE.items():                             # original reader + N2F frozen sha (weight_inputs), N2F hinge manifest
            inp = weight_inputs(seed, WEIGHT_FILE[arm])
            w = list(inp["weights"].values()); sum_w = math.fsum(w)
            man = json.loads((_n2f_unit(seed, method) / "run_manifest.json").read_text())
            chk = {"file": inp["paths"]["weights"], "sha256": inp["sha256"]["weights"], "n2f_frozen_sha_and_lock_reader": True,
                   "lock_method": inp["lock"].get("record", {}).get("method"), "n": len(w), "sum_w_U": sum_w,
                   "in_unit_interval": all(0.0 <= x <= 1.0 for x in w), "n2f_hinge_unit": str(_n2f_unit(seed, method))}
            chk[f"same_as_n2f_hinge_{method}_sum"] = man["weight_lock_sha256"] == inp["sha256"]["weights"] and math.isclose(man["sum_w_U"], sum_w, rel_tol=1e-12)
            if arm == "NCE_VCSW" and (root / "runs" / f"s{seed}__VCSW" / "unit_result.json").is_file():
                chk["same_as_a_vcsw"] = json.loads((root / "runs" / f"s{seed}__VCSW" / "unit_result.json").read_text())["inputs_sha256"]["weights"] == inp["sha256"]["weights"]
            chk["ok"] = chk["lock_method"] == method and chk["n"] == c1.PROTOCOL_POOL["n_U"] and chk["in_unit_interval"] \
                and chk[f"same_as_n2f_hinge_{method}_sum"] and chk.get("same_as_a_vcsw", True)
            rep["weights"][f"s{seed}__{arm}"] = chk
        b = _n2f_unit(seed, "BCE_Weight")
        hw = json.loads((b / "full_result.json").read_text()).get("hardware") or {}
        rep["baselines"][f"s{seed}__HINGE_BCEW_N2F"] = {"seed": seed, "arm": "HINGE_BCEW_N2F", "dir": str(b),
                                                       **endpoints(json.loads((b / "train_log.json").read_text())), "gpu": hw.get("gpu")}
        nce = root / "runs" / f"s{seed}__NCE"
        if not (nce / "unit_result.json").is_file():
            rep["nce_reuse"][seed] = {"a_nce_complete": False}
            continue
        res, nman = json.loads((nce / "unit_result.json").read_text()), json.loads((nce / "run_manifest.json").read_text())
        obs, wu = c1.OLD / f"s{seed}" / "pairs.observed_pairs.json", c1.OLD / f"s{seed}" / "weights" / "Unweighted_A.csv"
        pre = {"a_nce_complete": res["status"] == "COMPLETE_VERIFIED", "a_nce_gpu": res["environment"]["gpu"],
               "losses_py_same": res["source"]["losses_sha256"] == c1.sha256(HERE / "losses.py"),
               "a_nce_ran_a_launch_train_py": res["source"]["train_py_sha256"] == a_train_sha,
               "observed_same": res["inputs_sha256"]["observed"] == c1.sha256(obs),
               "unweighted_file_same": res["inputs_sha256"]["weights"] == c1.sha256(wu),
               "objective_config_same": res["objective"] == {"name": "infonce", "tau": 0.05, "loss": "symmetric InfoNCE over rows and columns of s / tau", "weights": "none"},
               "environment": {k: res["environment"].get(k) for k in ENV_KEYS}}
        pre["environment_expected"] = (pre["environment"]["python"], pre["environment"]["torch"], pre["environment"]["numpy"]) == ("3.12.12", "2.9.0+cu128", "2.5.2")
        post = {}
        for arm in A3_ARMS:
            out = root / "runs" / f"s{seed}__{arm}"
            if not (out / "unit_result.json").is_file():
                post[arm] = {"status": "NOT_COMPLETE" if out.is_dir() else "NOT_LAUNCHED"}
                continue
            r3, m3 = json.loads((out / "unit_result.json").read_text()), json.loads((out / "run_manifest.json").read_text())
            post[arm] = {"status": r3["status"], "gpu": r3["environment"]["gpu"],
                         "initial_model_same": r3["initial_model_sha256"] == res["initial_model_sha256"],
                         "data_order_same": r3["data_order_sha256"] == res["data_order_sha256"],
                         "recipe_same": all(m3[k] == nman[k] for k in ("cfg", "seed", "lr_mult", "n_pool", "n_A", "n_U", "formal")),
                         "precision_same": r3["precision_effective"] == res["precision_effective"],
                         "environment_same": {k: r3["environment"].get(k) for k in ENV_KEYS} == pre["environment"]}
            post[arm]["ok"] = all(post[arm][k] for k in ("initial_model_same", "data_order_same", "recipe_same", "precision_same", "environment_same"))
        rep["nce_reuse"][seed] = {**pre, "pre_ok": all(pre[k] for k in ("a_nce_complete", "losses_py_same", "a_nce_ran_a_launch_train_py", "observed_same",
                                                                       "unweighted_file_same", "objective_config_same", "environment_expected")),
                                  "post_run": post}
    launched = [s for s in SEEDS if any((root / "runs" / f"s{s}__{a}").is_dir() for a in A3_ARMS)] or [901]
    nr = [rep["nce_reuse"][s] for s in launched]
    posts = [p for r in nr for p in r.get("post_run", {}).values() if "ok" in p]
    ok = code["same"] and all(c["ok"] for c in rep["weights"].values()) and all(r.get("pre_ok") for r in nr) and all(p["ok"] for p in posts)
    complete = all("ok" in p for r in nr for p in r.get("post_run", {}).values()) and bool(posts)    # every A3 unit of the launched seeds done
    rep["seeds_checked_for_reuse"] = launched
    rep["status"] = ("NOT_IDENTICAL_RERUN_NEEDED" if not ok else
                     "REUSE_IDENTICAL_EXCEPT_HARDWARE" if posts and complete else "PRE_RUN_IDENTICAL_POST_RUN_PENDING")
    rep["hardware_note"] = ("A3 arms use RTX6000PRO/L40S/H100 as scheduled; A's NCE s901/s902 ran where recorded (a_nce_gpu), the N2M hinge arms "
                            "on RTX PRO 6000, N2F hinge VCS_Weight_sum / BCE_Weight_sum where their full_result.json says: cross-arm differences "
                            "include a hardware component")
    c1.write_json(root / "A3_IDENTITY.json", rep)
    return rep


# ------------------------------------------------------------------------------------------------ summary
PAIRS = (("VCS", "HINGE_ConstantU_N2M"), ("VCS", "HINGE_Unweighted_N2F"), ("VCSW", "HINGE_VCSW_N2M"), ("VCSW", "HINGE_VCSW_N2F"),
         ("VCS", "BCE"), ("VCS", "NCE"), ("VCSW", "VCS"), ("BCE", "HINGE_Unweighted_N2F"), ("NCE", "HINGE_Unweighted_N2F"))
PAIRS_A2 = (("VCS_SHARP", "HINGE_ConstantU_N2M"), ("VCSW_SHARP", "HINGE_VCSW_N2M"), ("VCS_SHARP", "HINGE_Unweighted_N2F"),
            ("VCSW_SHARP", "HINGE_VCSW_N2F"), ("VCS_SHARP", "VCS"), ("VCSW_SHARP", "VCSW"), ("BCE_SHARP", "BCE"), ("VCS_SHARP", "BCE_SHARP"),
            ("VCS_SHARP", "NCE"), ("VCSW_SHARP", "VCS_SHARP"))
# R22_ALTERNATIVES_CN.md A2, written before any A2 run: seeds 902/903 are added when VCS_SHARP or VCSW_SHARP (seed 901) is within 10 best-rSum of
# its hinge counterpart (Constant-U 443.6 / hinge+VCS weights 467.0, N2M); "within" = shortfall < 10 (an A2 arm above its hinge also qualifies).
A2_RULE = {"seed": 901, "margin": 10.0, "pairs": (("VCS_SHARP", "HINGE_ConstantU_N2M"), ("VCSW_SHARP", "HINGE_VCSW_N2M"))}
PAIRS_A3 = (("NCE_VCSW", "HINGE_VCSW_N2M"), ("NCE_VCSW", "HINGE_VCSW_N2F"), ("NCE_VCSW", "NCE_BCEW"), ("NCE_VCSW", "NCE"),
            ("NCE_BCEW", "HINGE_BCEW_N2F"), ("NCE_BCEW", "NCE"))
# R22_ALTERNATIVES_CN.md A3, written before any A3 run: seeds 902/903 are added when NCE_VCSW (seed 901) reaches best rSum >= 464.0 (hinge+VCS
# weights 467.0, N2M seed 901, minus 3).  After three seeds each final pair is judged by mean difference >= +3 and >= 2/3 positive seeds;
# "hinge+VCS weights" is the N2M arm the 467.0 comes from, the N2F arm (same recipe, other hardware) is judged alongside.
A3_RULE = {"seed": 901, "arm": "NCE_VCSW", "threshold_best_rsum": 464.0, "final_mean_min": 3.0, "final_positive_min": 2,
           "final_pairs": (("NCE_VCSW", "HINGE_VCSW_N2M"), ("NCE_VCSW", "NCE_BCEW")), "also_reported": (("NCE_VCSW", "HINGE_VCSW_N2F"),)}


def _pairs(units, pairs):
    out = []
    for a, b in pairs:
        diffs = {}
        for seed in SEEDS:
            ua, ub = units.get(f"s{seed}__{a}", {}), units.get(f"s{seed}__{b}", {})
            if "best_rsum" in ua and "best_rsum" in ub:
                diffs[seed] = {m: ua[m] - ub[m] for m in ("best_rsum", *RECALLS, "last_rsum", "mean_last5_rsum")}
        d = [x["best_rsum"] for x in diffs.values()]
        out.append({"a": a, "b": b, "per_seed": diffs, "n_seeds": len(d), "mean_best_rsum_diff": sum(d) / len(d) if d else None,
                    "positive_seeds": sum(x > 0 for x in d)})
    return out


def a2_rule(units):
    rows = {}
    for a, b in A2_RULE["pairs"]:
        ua, ub = units.get(f"s{A2_RULE['seed']}__{a}", {}), units.get(f"s{A2_RULE['seed']}__{b}", {})
        if "best_rsum" in ua and "best_rsum" in ub:
            d = ua["best_rsum"] - ub["best_rsum"]
            rows[a] = {"vs": b, "a_best_rsum": ua["best_rsum"], "hinge_best_rsum": ub["best_rsum"], "diff": d, "within": d > -A2_RULE["margin"]}
    decision = "ADD_SEEDS_902_903" if any(r["within"] for r in rows.values()) else "STOP" if len(rows) == len(A2_RULE["pairs"]) else "PENDING"
    return {"rule": {**A2_RULE, "pairs": [list(x) for x in A2_RULE["pairs"]]}, "rows": rows, "decision": decision}


def a3_rule(units, pairs_a3):
    u = units.get(f"s{A3_RULE['seed']}__{A3_RULE['arm']}", {})
    decision = "PENDING" if "best_rsum" not in u else "ADD_SEEDS_902_903" if u["best_rsum"] >= A3_RULE["threshold_best_rsum"] else "STOP"
    final = []
    for a, b in A3_RULE["final_pairs"] + A3_RULE["also_reported"]:
        p = next(x for x in pairs_a3 if (x["a"], x["b"]) == (a, b))
        met = p["n_seeds"] == len(SEEDS) and p["mean_best_rsum_diff"] >= A3_RULE["final_mean_min"] and p["positive_seeds"] >= A3_RULE["final_positive_min"]
        final.append({"a": a, "b": b, "pre_registered": (a, b) in A3_RULE["final_pairs"], "n_seeds": p["n_seeds"],
                      "mean_best_rsum_diff": p["mean_best_rsum_diff"], "positive_seeds": p["positive_seeds"],
                      "verdict": "PENDING_3_SEEDS" if p["n_seeds"] < len(SEEDS) else "MET" if met else "NOT_MET"})
    rule = {k: ([list(x) for x in v] if isinstance(v, tuple) else v) for k, v in A3_RULE.items()}
    return {"rule": rule, "seed901_best_rsum": u.get("best_rsum"), "decision": decision, "final": final}


def _critic_track(log, best_epoch):
    eps = {e["epoch"]: e.get("objective_epoch") or {} for e in log["epochs"]}
    pick = lambda e: {k: eps.get(e, {}).get(k) for k in ("b", "hard_neg_frac_Q", "EP_T_mean", "EQ_T_mean", "J_hat")}
    return {"critic_epoch0": pick(0), "critic_best_epoch": pick(best_epoch), "critic_last_epoch": pick(max(eps))}


def summarize(root):
    ident, a3 = identity(root), a3_identity(root)
    units, rows = {}, []
    for seed in SEEDS:
        for arm in MAIN_ARMS + A2_ARMS + A3_ARMS:
            out = root / "runs" / f"s{seed}__{arm}"
            key = f"s{seed}__{arm}"
            if arm in A2_ARMS + A3_ARMS and not out.is_dir():
                continue                                                  # A2 / A3 seed not launched
            if not (out / "unit_result.json").is_file():
                units[key] = {"seed": seed, "arm": arm, "status": "NOT_COMPLETE",
                              "epochs_done": len(json.loads((out / "train_log.json").read_text())["epochs"]) if (out / "train_log.json").is_file() else None}
                continue
            res = json.loads((out / "unit_result.json").read_text())
            att = [json.loads(x) for x in (out / "attempts.jsonl").read_text().splitlines()]
            starts = [a for a in att if a["event"] == "start"]
            units[key] = {"seed": seed, "arm": arm, "status": res["status"], **endpoints(json.loads((out / "train_log.json").read_text())),
                          "gpu": sorted({a["environment"]["gpu"] for a in starts}), "attempts": len(starts),
                          "attempt_seconds_known": sum(a["elapsed_s"] for a in att if a["event"] == "end"),
                          "attempts_without_end_record": len(starts) - sum(a["event"] == "end" for a in att),
                          "initial_model_sha256": res["initial_model_sha256"], "data_order_sha256": res["data_order_sha256"]}
            if arm in A2_ARMS:
                units[key].update(_critic_track(json.loads((out / "train_log.json").read_text()), units[key]["best_epoch"]))
        for k, b in list(ident["baselines"].items()) + list(a3["baselines"].items()):
            if b["seed"] == seed:
                units[k] = {**b, "status": "REUSED"}
    pairs, pairs_a2, rule = _pairs(units, PAIRS), _pairs(units, PAIRS_A2), a2_rule(units)
    pairs_a3 = _pairs(units, PAIRS_A3)
    rule3 = a3_rule(units, pairs_a3)
    inits = {seed: {units[f"s{seed}__{a}"]["initial_model_sha256"] for a in MAIN_ARMS + A2_ARMS + A3_ARMS
                    if "initial_model_sha256" in units.get(f"s{seed}__{a}", {})} for seed in SEEDS}
    summary = {"schema": "r22-vl-summary-v1", "utc": c1.utc(), "primary": "best full DEV rSum; earliest exact tie", "label": "PILOT/DEVELOPMENT",
               "dev_only": True, "test_read": False, "hinge_identity": ident["status"], "units": units, "pairs": pairs,
               "pairs_a2": pairs_a2, "a2_seed_expansion": rule, "a3_identity": a3["status"], "pairs_a3": pairs_a3, "a3_rule": rule3,
               "same_initial_model_within_seed": {s: len(v) <= 1 for s, v in inits.items()},
               "unit_of_analysis": "seed (three seeds; descriptive, not a significance test)"}
    c1.write_json(root / "SUMMARY.json", summary)
    cols = ["seed", "arm", "status", "best_rsum", "best_epoch", *RECALLS, "last_rsum", "mean_last5_rsum", "gpu"]
    with open(root / "SUMMARY_UNITS.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for u in units.values():
            w.writerow([u.get(c) if not isinstance(u.get(c), list) else "/".join(u[c]) for c in cols])
    print(f"{'unit':<28}{'best':>8}{'ep':>4}{'last':>8}{'last5':>8}  gpu")
    for k, u in units.items():
        if "best_rsum" in u:
            g = u["gpu"] if isinstance(u["gpu"], str) else "/".join(u["gpu"] or [])
            print(f"{k:<28}{u['best_rsum']:8.1f}{u['best_epoch']:4d}{u['last_rsum']:8.1f}{u['mean_last5_rsum']:8.1f}  {g}")
        else:
            print(f"{k:<28}  {u['status']} (epochs done {u.get('epochs_done')})")
    for title, prs in (("A", pairs), ("A2", pairs_a2), ("A3", pairs_a3)):
        print(f"-- {title} paired best-rSum differences")
        for p in prs:
            per = " ".join(f"{s}:{x['best_rsum']:+.1f}" for s, x in p["per_seed"].items())
            m = f"{p['mean_best_rsum_diff']:+.2f}" if p["mean_best_rsum_diff"] is not None else "n/a"
            print(f"{p['a']} - {p['b']}: mean {m} ({p['positive_seeds']}/{p['n_seeds']} positive) {per}")
    for k, u in units.items():
        if "critic_last_epoch" in u:
            c0, cb, cl = u["critic_epoch0"], u["critic_best_epoch"], u["critic_last_epoch"]
            print(f"{k}: b {c0['b']:.3f} (ep0) / {cb['b']:.3f} (best) / {cl['b']:.3f} (last); hard-neg frac {c0['hard_neg_frac_Q']:.4f} / "
                  f"{cb['hard_neg_frac_Q']:.4f} / {cl['hard_neg_frac_Q']:.4f}")
    print("A2 seed expansion:", rule["decision"], {a: round(r["diff"], 2) for a, r in rule["rows"].items()})
    print(f"A3 identity: {a3['status']}; A3 seed expansion (NCE_VCSW s901 best >= {A3_RULE['threshold_best_rsum']}): {rule3['decision']} "
          f"(s901 NCE_VCSW best {rule3['seed901_best_rsum']})")
    for f in rule3["final"]:
        m = f"{f['mean_best_rsum_diff']:+.2f}" if f["mean_best_rsum_diff"] is not None else "n/a"
        print(f"A3 final {'(pre-registered)' if f['pre_registered'] else '(also reported)'} {f['a']} - {f['b']}: mean {m}, "
              f"{f['positive_seeds']}/{f['n_seeds']} positive -> {f['verdict']}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("smoke", "run", "identity", "a3-identity", "summarize"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed", type=int, choices=SEEDS, default=901)
    ap.add_argument("--arm", choices=MAIN_ARMS + A2_ARMS + A3_ARMS)
    ap.add_argument("--arms", default="VCS,VCSW,BCE,NCE,HINGE")
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if a.command == "identity":
        print(json.dumps({"hinge_identity": identity(root)["status"]}))
        return 0
    if a.command == "a3-identity":
        print(json.dumps({"a3_identity": a3_identity(root)["status"]}))
        return 0
    if a.command == "summarize":
        summarize(root)
        return 0
    if a.command == "run" and not a.arm:
        ap.error("run needs --arm")
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))     # time limit: the last complete epoch is kept
    finish = trace_open.install("c1")                                                # forbids TEST, eval_only, noise files; DEV allowed
    attempt = f"{os.environ.get('SLURM_JOB_ID', 'nojob')}_{int(time.time())}"
    trace_path = root / "traces" / f"{a.command}_s{a.seed}_{a.arm or 'smoke'}_{attempt}.json"
    code, result, report = 0, None, None
    try:
        if a.command == "smoke":
            report, sdir = smoke(root, a.seed, [x for x in a.arms.split(",") if x], a.steps, finish.forbidden_hits)
        else:
            result = run(root, a.seed, a.arm, finish.forbidden_hits)
    except Refused as exc:
        code, result = exc.code, str(exc)
        print(f"REFUSED({exc.code}): {exc}", file=sys.stderr)
    finally:
        trace = finish(trace_path, {"command": a.command, "seed": a.seed, "arm": a.arm, "result": result})
    if trace["status"] != "PASS":
        print(f"FORBIDDEN_ACCESS: {trace['forbidden_hits']}", file=sys.stderr)
        code = 9
    if report is not None:
        report["read_evidence"] = {"trace": str(trace_path), "status": trace["status"]}
        report["status"] = "PASS" if report["checks_pass"] and trace["status"] == "PASS" else "FAIL"
        c1.write_json(sdir / "SMOKE.json", report)
        print(json.dumps({"smoke": report["status"], "report": str(sdir / "SMOKE.json")}))
        code = code or (0 if report["status"] == "PASS" else 4)
    elif code == 0:
        print(json.dumps({"seed": a.seed, "arm": a.arm, "unit": result}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
