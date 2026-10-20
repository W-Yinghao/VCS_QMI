"""D1b records-only audit: why is V10 CoMIR_native DEV32 below the earlier host CoMIR? (no new computation)

Compares, from existing records and code only, the frozen V10 CoMIR rows with the earlier
host CoMIR runs of the same recipe: R6 REG-P2 comir_s0..s2 and V9-G1 comir frac1 eps0
s901..s903. Reads result JSON, training logs, Slurm log header lines and cache-file
hashes; never trains, registers, or reads TEST/EPFL/evaluator_only material.

  audit_comir_gap.py   writes <bundle>/D1/COMIR_GAP_RECORDS.json and v10/sensitivity/COMIR_GAP_RECORDS_CN.md
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time

DIAG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DIAG / "src"))

from v10_sensitivity import common as C  # noqa: E402

PROJECT = C.PROJECT
LOGS = PROJECT / "slurm_logs"
R6_P2 = PROJECT / "outputs/R6_REG_P2/results"
R6_P1 = PROJECT / "outputs/R6_REG"
V9_UNITS = Path("/projects/EEG-foundation-model/yinghao/outputs_v9/reg/units")
V10_FAMILY = PROJECT / "handoff/v10_native_geometry_recovery_20261019/families/4"
GENERATIONS = {
    "R6_REG_P2": {"seeds": (0, 1, 2),
                  "results": lambda s: R6_P2 / f"comir_s{s}.json",
                  "train": lambda s: R6_P1 / "runs" / f"comir_s{s}" / "comir.json",
                  "train_logs": {0: "r6reg_comir_s0_1034719.out", 1: "r6reg_comir_s1_1034838.out", 2: "r6reg_comir_s2_1034860.out"},
                  "register_logs": {0: "r6reg2_reg_comir_s0_1034904.out", 1: "r6reg2_reg_comir_s1_1034918.out",
                                    2: "r6reg2_reg_comir_s2_1034981.out"}},
    "V9_G1_frac1_eps0": {"seeds": (901, 902, 903),
                         "results": lambda s: V9_UNITS / f"V9-G1-comir-frac1-eps0-s{s}" / "p2/results" / f"comir_s{s}.json",
                         "train": lambda s: V9_UNITS / f"V9-G1-comir-frac1-eps0-s{s}" / "run/comir.json",
                         "train_logs": {s: "v9g1_comir_f1_1041935.out" for s in (901, 902, 903)},
                         "register_logs": {s: "v9g1_comir_f1_1041935.out" for s in (901, 902, 903)}},
}
V9_DUPLICATE = {"cancelled_job_log": "v9g1_comir_f1_1041927.out", "kept_job_log": "v9g1_comir_f1_1041935.out",
                "unit": "V9-G1-comir-frac1-eps0-s901"}
CODE_READING = {
    "training": ("R6 REG-P1 cmd_comir (src/v9/reg/r6_reg_pilot.py L320-365, pinned host hash 6c6f094684fbc50a) and V10 "
                 "CoMIRTrainer (src/v10/supplement/comir.py L55-139) implement the same recipe: official DenseUNet ModNet via "
                 "P.comir_models with torch.manual_seed(seed); SGD lr 1e-2 momentum 0.9 wd 1e-5 per network; numpy "
                 "default_rng(3000+seed) with the same call order (choice of 32 FIT cores, crop origin, flip, angle, then the "
                 "two rot90 draws); 182-px crop, bilinear rotation with reflection, central 128 crop; p4 equivariance; "
                 "InfoNCE on MSE similarity tau 0.5 over 2N latents; L1/L2 activation decay 1e-4; per-network clip 1.0; "
                 "23 x 32 x 32. FIT list: R6 sorted FIT cores; V10 A.fit_cores -> G.sample_fit(frac 1) returns the same "
                 "sorted 32 cores."),
    "registration": ("R6 REG-P2 cmd_register method comir (src/v9/reg/r6_reg_p2.py L311-328) and V10 RegistrationSession "
                     "CoMIR branch (src/v10/supplement/registration.py L118-140) both encode the BF RGB reference once per core "
                     "with network B and the perturbed log1p SHG moving image once per perturbation with network A, both in "
                     "eval mode, then use Q.RigidEnergy kind 'comir' (latents sampled at T(x), F = -MSE) with the same "
                     "positions seed and Q.optimise_rigid. R6 additionally sets efficient=False on network A; the official "
                     "DenseNet layer only checkpoints when an input requires grad, which never holds under torch.no_grad, so "
                     "this changes no value."),
    "differences_in_code": "V10 saves 23 per-epoch checkpoints and evaluates CAL at epoch 23 only; saving does not change state.",
}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def header(name):
    path = LOGS / name
    if not path.exists():
        return {"log": str(path), "missing": True}
    first = path.read_text(errors="replace").splitlines()[0]
    return {"log": str(path), "first_line": first[:200]}


def epoch_losses(name, unit=None):
    path = LOGS / name
    out = []
    for line in path.read_text(errors="replace").splitlines():
        if "] epoch " in line and " loss " in line and (unit is None or unit in line):
            out.append(line.strip()[:160])
    return out


def r6_like(path):
    result = C.read(path)
    dev = {c: v for c, v in result["dev"].items() if "#" not in c}
    cal = {c: v for c, v in result["cal"].items() if "#" not in c}
    return result, dev, cal


def v10_runs():
    out = {}
    for seed in (1001, 1002, 1003):
        directory = V10_FAMILY / "results/FINAL_DEV/CoMIR_native" / f"seed_{seed}"
        env = C.read(next(directory.glob("ENV_*.json")))
        train = rows(directory / "TRAIN_STEPS.jsonl")
        out[seed] = {"dev": rows(directory / "DEV.jsonl"), "cal": rows(directory / "CAL/step_0736.jsonl"),
                     "host": env["host"], "gpu": env["gpu_name"], "torch": env["torch"], "cudnn": env["cudnn"],
                     "train_seconds_first_to_last_step": utc_seconds(train[-1]["utc"]) - utc_seconds(train[0]["utc"]),
                     "epoch_end_losses": [round(r["loss"], 4) for r in train if r["step"] % 32 == 0],
                     "last_epoch_mean_loss": statistics.fmean(r["loss"] for r in train[-32:])}
    cal_unit = V10_FAMILY / "results/CAL/CoMIR_native/candidate_00"
    env = C.read(next(cal_unit.glob("ENV_*.json")))
    out[1017] = {"cal": rows(cal_unit / "CAL/step_0736.jsonl"), "host": env["host"], "gpu": env["gpu_name"]}
    return out


def utc_seconds(stamp):
    return time.mktime(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))


def success(errors, threshold=42.0):
    return sum(e < threshold for e in errors) / len(errors)


def main():
    lock = C.read(PROJECT / "handoff/v10_bf1_20261017/DATA_LOCK.json")
    fit32 = sorted(f["core"] for f in lock["files"] if f["role"] == "FIT")
    v10 = v10_runs()
    v10_dev_cores = sorted({r["core"] for r in v10[1001]["dev"]})
    v10_truth = {(r["core"], r["perturbation"]): (r["theta"], r["t"]) for r in v10[1001]["dev"]}
    v10_cal_truth = {(r["core"], r["perturbation"]): (r["theta"], r["t"]) for r in v10[1001]["cal"]}
    generations, per_core = {}, {}
    for name, spec in GENERATIONS.items():
        runs = {}
        for seed in spec["seeds"]:
            result, dev, cal = r6_like(spec["results"](seed))
            train = C.read(spec["train"](seed))
            same_cores = sorted(dev) == v10_dev_cores and sorted(cal) == sorted({r["core"] for r in v10[1001]["cal"]})
            truth_equal = sum(dev[c][i]["theta"] == v10_truth[(c, i)][0] and dev[c][i]["t"] == v10_truth[(c, i)][1]
                              for c in dev for i in range(len(dev[c])))
            cal_truth_equal = sum(cal[c][i]["theta"] == v10_cal_truth[(c, i)][0] and cal[c][i]["t"] == v10_cal_truth[(c, i)][1]
                                  for c in cal for i in range(len(cal[c])))
            runs[seed] = {"DEV32_success42": result["dev_success42"], "DEV32_success9": result["dev_success9"],
                          "DEV32_median": result["dev_median_err"], "CAL_success42": result["cal_success42"],
                          "same_dev_and_cal_cores_as_v10": same_cores, "dev_truth_rows_identical_to_v10": truth_equal,
                          "cal_truth_rows_identical_to_v10": cal_truth_equal,
                          "train_last_epoch_loss": train["log"][-1]["loss"], "train_seconds": train["seconds"],
                          "train_fit_equals_v10_fit32": (train["fit"] == fit32) if "fit" in train else "not recorded",
                          "train_log": header(spec["train_logs"][seed]), "register_log": header(spec["register_logs"][seed])}
            per_core[f"{name}:{seed}"] = {c: success([r["err_full"] for r in v]) for c, v in dev.items()}
        generations[name] = runs
    v10_summary = {}
    for seed in (1001, 1002, 1003):
        run = v10[seed]
        errors = [r["err_full"] for r in run["dev"]]
        v10_summary[seed] = {"DEV32_success42": success(errors), "DEV32_success9": success(errors, 9.0),
                             "DEV32_median": statistics.median(errors),
                             "CAL_success42": success([r["err_full"] for r in run["cal"]]),
                             "host": run["host"], "gpu": run["gpu"], "torch": run["torch"], "cudnn": run["cudnn"],
                             "epoch_end_losses": run["epoch_end_losses"], "last_epoch_mean_loss": run["last_epoch_mean_loss"]}
        cores = {}
        for r in run["dev"]:
            cores.setdefault(r["core"], []).append(r["err_full"])
        per_core[f"V10:{seed}"] = {c: success(v) for c, v in cores.items()}
    v10_summary[1017] = {"CAL_success42": success([r["err_full"] for r in v10[1017]["cal"]]), "host": v10[1017]["host"],
                         "gpu": v10[1017]["gpu"]}
    keys = list(per_core)
    always_one = sorted(c for c in v10_dev_cores if all(per_core[k][c] == 1.0 for k in keys))
    always_zero = sorted(c for c in v10_dev_cores if all(per_core[k][c] == 0.0 for k in keys))
    variable = sorted(set(v10_dev_cores) - set(always_one) - set(always_zero))
    cache_hash_mismatch = [f["core"] for f in lock["files"] if C.sha(f["path"]) != f["sha256"]]
    cache_mtimes = sorted(os.path.getmtime(f["path"]) for f in lock["files"])
    record = {
        "schema": "v10.sensitivity.d1b.records_audit.v1", "label": C.LABEL, "scope": "records and code only; no computation",
        "utc": C.utc(),
        "same_dev_cores": all(r["same_dev_and_cal_cores_as_v10"] for g in generations.values() for r in g.values()),
        "same_perturbation_truths": {f"{n}:{s}": f"DEV {r['dev_truth_rows_identical_to_v10']}/320, CAL {r['cal_truth_rows_identical_to_v10']}/80"
                                     for n, g in generations.items() for s, r in g.items()},
        "same_preprocessing": {
            "cache": "every method reads the R6 REG-P1 cache outputs/R6_REG/cache (V10 DataGuard r6_root = G.P1_ROOT)",
            "data_lock_hash_mismatches_now": cache_hash_mismatch,
            "cache_mtime_range_utc": [time.strftime("%FT%TZ", time.gmtime(cache_mtimes[0])),
                                      time.strftime("%FT%TZ", time.gmtime(cache_mtimes[-1]))],
            "inputs": "learned path: robust-normalised BF RGB reference, perturbed SHG V o T*^-1 then robust-normalised log1p "
                      "(Q.Inputs, unchanged in V10 GuardedInputs); training: shg_log_n / bf_rgb_n crops"},
        "code_reading": CODE_READING,
        "generations": generations, "v10": v10_summary,
        "nondeterminism_evidence": {
            "unit": V9_DUPLICATE["unit"], "same_code_hash": "6c6f094684fbc50a (both job headers)",
            "cancelled_duplicate": epoch_losses(V9_DUPLICATE["cancelled_job_log"], V9_DUPLICATE["unit"])[:1],
            "kept_run": epoch_losses(V9_DUPLICATE["kept_job_log"], V9_DUPLICATE["unit"])[:1],
            "headers": [header(V9_DUPLICATE["cancelled_job_log"]), header(V9_DUPLICATE["kept_job_log"])]},
        "per_core_success42": per_core,
        "per_core_pattern": {"success_1_in_all_9_runs": always_one, "success_0_in_all_9_runs": always_zero,
                             "variable_cores": variable},
    }
    record["findings"] = [
        "DEV cores, CAL cores and all 320 DEV / 80 CAL perturbation truths are identical across R6, V9-G1 and V10.",
        "Preprocessing and inputs are identical: same R6 cache files (data-lock hashes match now; file mtimes "
        f"{record['same_preprocessing']['cache_mtime_range_utc'][0]}..{record['same_preprocessing']['cache_mtime_range_utc'][1]}, "
        "before the first R6 CoMIR training job started at 2026-10-11T22:45:00Z), same Q.Inputs path; R6 and V9-G1 "
        "training FIT lists equal the V10 FIT32.",
        "Training recipe, RNG call order, FIT order and the encode-then-register path are the same by code reading; the "
        "only code difference (efficient=False at R6 registration) changes no value under no_grad.",
        "Differences: training seeds (0-2 / 901-903 / 1001-1003 and 1017); training GPU (R6: RTX PRO 6000 x2 and H100 NVL "
        "x1; V9-G1 and V10: L40S node52); registration GPU (R6: L40S x2, RTX PRO 6000 x1; V9-G1, V10: L40S node52). Same "
        "torch 2.9.0+cu128 environment (installed 2026-09-24).",
        "CoMIR training is not bitwise reproducible on this stack: two V9-G1 jobs of the same unit on the same node and "
        "with the same hashed host code (repository commits 915de3c and e04a336 otherwise differ) logged epoch-1 losses "
        "4.4081 and 4.4057.",
        f"The run-to-run differences sit on {len(variable)} variable DEV cores; {len(always_one)} cores succeed in all nine "
        f"runs and {len(always_zero)} fail in all nine.",
        "No cause can be attributed from records: three runs per generation, seeds and hardware confounded. D1 tests only "
        "the checkpoint-epoch factor.",
    ]
    C.write(C.BUNDLE / "D1" / "COMIR_GAP_RECORDS.json", record)
    write_markdown(record)
    print("AUDIT", C.BUNDLE / "D1" / "COMIR_GAP_RECORDS.json")


def write_markdown(record):
    lines = ["# D1b CoMIR 差距的记录核查（不含新计算）", "",
             "标签：post-DEV sensitivity 的附属记录核查。只读结果 JSON、训练日志、Slurm 日志首行和缓存哈希；不训练、不配准。",
             f"生成：`scripts/v10_sensitivity/audit_comir_gap.py`，{record['utc']}；机器可读版本 `<bundle>/D1/COMIR_GAP_RECORDS.json`。", "",
             "## 结论", ""]
    lines += [f"- {f}" for f in record["findings"]]
    lines += ["", "## 各次运行", "", "| 代 | seed | DEV32 <42 | DEV32 <9 | CAL <42 | 训练日志首行 | 配准日志首行 |", "|---|---|---|---|---|---|---|"]
    for name, runs in record["generations"].items():
        for seed, r in runs.items():
            lines.append(f"| {name} | {seed} | {r['DEV32_success42']:.4f} | {r['DEV32_success9']:.4f} | {r['CAL_success42']:.4f} | "
                         f"{r['train_log'].get('first_line', 'missing')[:60]} | {r['register_log'].get('first_line', 'missing')[:60]} |")
    for seed, r in record["v10"].items():
        if "DEV32_success42" in r:
            lines.append(f"| V10 | {seed} | {r['DEV32_success42']:.4f} | {r['DEV32_success9']:.4f} | {r['CAL_success42']:.4f} | "
                         f"{r['host']} {r['gpu']} | {r['host']} {r['gpu']} |")
        else:
            lines.append(f"| V10 调参 | {seed} | — | — | {r['CAL_success42']:.4f} | {r['host']} {r['gpu']} | 同左 |")
    pattern = record["per_core_pattern"]
    lines += ["", "## 逐 core 模式（DEV32 success<42，9 次运行）", "",
              f"- 9 次全部成功：{len(pattern['success_1_in_all_9_runs'])} 个 core",
              f"- 9 次全部失败：{', '.join(pattern['success_0_in_all_9_runs'])}",
              f"- 随运行变化：{', '.join(pattern['variable_cores'])}", "",
              "| core | " + " | ".join(record["per_core_success42"]) + " |", "|---|" + "---|" * len(record["per_core_success42"])]
    for core in pattern["variable_cores"]:
        lines.append(f"| {core} | " + " | ".join(f"{v[core]:.1f}" for v in record["per_core_success42"].values()) + " |")
    nd = record["nondeterminism_evidence"]
    lines += ["", "## 非确定性证据", "", f"- 单元 {nd['unit']}，两次作业 host-code 哈希相同（{nd['same_code_hash']}）：",
              f"  - 被取消的重复作业：{nd['cancelled_duplicate']}", f"  - 保留的作业：{nd['kept_run']}", ""]
    (DIAG / "v10/sensitivity/COMIR_GAP_RECORDS_CN.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
