"""Neutral read-out of the pre-registered D1-D3 diagnostics (CPU, records only; run after the jobs).

  summarize.py            writes <bundle>/RESULTS.json and <bundle>/RESULTS_CN.md
  summarize.py --check    computes and prints, writes nothing

Applies the pre-registered interpretation grid mechanically and labels everything as a
post-DEV sensitivity diagnostic. Incomplete diagnostics are reported as such, never
classified. Framing beyond the grid waits for the owner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

DIAG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DIAG / "src"))

from v10_sensitivity import analysis as S  # noqa: E402
from v10_sensitivity import common as C  # noqa: E402

FINALS = ("FINAL_1001", "FINAL_1002", "FINAL_1003")


def rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def verified(directory):
    path = Path(directory) / "COMPLETE.json"
    if not path.exists():
        return None
    result = C.read(path)
    C.require(result.get("status") == "COMPLETE", f"{path} is not COMPLETE")
    for name, expected in result["artifacts"].items():
        C.require(C.sha(Path(directory) / name) == expected, f"Artifact changed: {directory}/{name}")
    return result


def frozen_dir(ref, key=None):
    base = Path(ref["path"]).parent
    return base / (ref["units"][key]["dir"] if key else ref["unit"]["dir"])


def d1(pre, out_root):
    ref = pre["frozen_inputs"]["comir_release"]
    units, missing = {}, []
    for key in ("CAL_1017",) + FINALS:
        result = verified(out_root / "D1" / key)
        if result is None:
            missing.append(key)
            continue
        units[key] = {"selected_epoch": result["selected"]["epoch"], "selected_step": result["selected"]["step"],
                      "cal": [{"epoch": r["epoch"], **r["metrics"]} for r in result["cal_rows"]],
                      "qc_replay_cal_0736": result["qc_replay_cal_0736"]}
        if key in FINALS:
            new = rows(out_root / "D1" / key / "DEV.jsonl")
            old = rows(frozen_dir(ref, key) / "DEV.jsonl")
            units[key].update(
                DEV32_selected=S.metrics(new), DEV32_frozen_epoch23=S.metrics(old),
                DEV24_selected=result["DEV24"], DEV24_frozen_epoch23=C.read(frozen_dir(ref, key) / "COMPLETE.json")["DEV24"],
                strata_selected=S.rotation_strata(new), strata_frozen=S.rotation_strata(old),
                core_diff_success42=S.paired_core_differences(new, old, 42.0),
                core_diff_success9=S.paired_core_differences(new, old, 9.0),
                qc_replay_dev=result.get("QC_REPLAY_DEV", {}).get("status"))
    report = {"status": "INCOMPLETE" if missing else "COMPLETE", "missing": missing, "units": units}
    if missing:
        return report
    diffs42 = S.averaged_core_differences([units[k]["core_diff_success42"] for k in FINALS])
    diffs9 = S.averaged_core_differences([units[k]["core_diff_success9"] for k in FINALS])
    paired42, paired9 = S.bootstrap_mean(diffs42.values()), S.bootstrap_mean(diffs9.values())
    per_seed = {k: S.bootstrap_mean(units[k]["core_diff_success42"].values()) for k in FINALS}
    earlier = [k for k in FINALS if units[k]["selected_step"] != 736]
    if not earlier:
        cell = "G1_all_select_23"
    elif paired42["mean"] >= 0.03 and paired42["ci_low"] > 0:
        cell = "G2_material_gain"
    elif paired42["mean"] <= -0.03 and paired42["ci_high"] < 0:
        cell = "G4_material_loss"
    else:
        cell = "G3_no_material_change"
    gap = all(units[k]["DEV32_selected"]["success42"] >= 0.89 for k in FINALS)
    report.update(paired_success42_mean_over_seeds=paired42, paired_success9_mean_over_seeds=paired9,
                  paired_success42_per_seed=per_seed, finals_selecting_earlier=earlier,
                  grid_cell=cell, grid_text=pre["D1"]["interpretation_grid"][cell],
                  gap_clause="checkpoint timing accounts for the gap" if gap else
                  "gap not explained by checkpoint timing (see the records audit; no causal claim)")
    return report


def d2(pre, out_root):
    ref = pre["frozen_inputs"]["mind_release"]
    directory = out_root / "D2" / pre["D2"]["variant"]
    result = verified(directory)
    if result is None:
        return {"status": "INCOMPLETE"}
    frozen = frozen_dir(ref)
    new_dev, old_dev = rows(directory / "VARIANT/DEV.jsonl"), rows(frozen / "DEV.jsonl")
    new_cal, old_cal = rows(directory / "VARIANT/CAL_step_0000.jsonl"), rows(frozen / "CAL/step_0000.jsonl")
    frozen_complete = C.read(frozen / "COMPLETE.json")
    strata_new, strata_old = S.rotation_strata(new_dev), S.rotation_strata(old_dev)
    paired = S.bootstrap_mean(S.paired_core_differences(new_dev, old_dev, 42.0).values())
    paired_overall_drop = paired["mean"] <= -0.03 and paired["ci_high"] < 0
    gain_high = strata_new["20-30"]["success42"] - strata_old["20-30"]["success42"]
    if gain_high >= 0.10 and strata_new["spread_success42"] <= 0.5 * strata_old["spread_success42"]:
        cell = "G1_orientation_explains"
    elif gain_high >= 0.03:
        cell = "G2_partial"
    else:
        cell = "G3_not_orientation"
    return {"status": "COMPLETE", "qc_replay_frozen_cal": result["replay_frozen_cal"]["status"],
            "variant": {"CAL": S.metrics(new_cal), "DEV32": S.metrics(new_dev), "DEV24": result["DEV24"],
                        "DEV32_strata": strata_new, "CAL_strata": S.rotation_strata(new_cal),
                        "failures": S.failure_split(new_dev)},
            "frozen": {"CAL": S.metrics(old_cal), "DEV32": S.metrics(old_dev), "DEV24": frozen_complete["DEV24"],
                       "DEV32_strata": strata_old, "CAL_strata": S.rotation_strata(old_cal),
                       "failures": S.failure_split(old_dev)},
            "paired_success42_variant_minus_frozen": paired, "stratum_20_30_gain": gain_high,
            "grid_cell": cell, "grid_text": pre["D2"]["interpretation_grid"][cell],
            "G4_overall_worse": paired_overall_drop,
            "always": pre["D2"]["interpretation_grid"]["always"]}


def d3(bundle):
    summaries = [C.read(p) for p in sorted((bundle / "D3/summaries").glob("*.json"))]
    table = []
    for s in summaries:
        row = {k: s.get(k) for k in ("job", "family", "partition", "status", "gpu_name", "gpu_capability", "host",
                                     "cuda_numerics", "accepted_l40s_job")}
        if s.get("comparison"):
            row["deltas"] = {k: v["difference"] for k, v in s["comparison"].items() if isinstance(v, dict)}
            row["resume_bitwise_equal_this_gpu"] = s["comparison"]["resume_bitwise_equal_this_gpu"]
        table.append(row)
    expected = {(j["family"], j["partition"]) for j in pre_d3_jobs()}
    passed = {(r["family"], r["partition"]) for r in table if r["status"] == "PASS"}
    return {"status": "COMPLETE" if expected <= passed else "INCOMPLETE", "jobs": table,
            "missing_pass": sorted(f"{f}@{p}" for f, p in expected - passed),
            "interpretation": C.prereg()["D3"]["interpretation"]}


def pre_d3_jobs():
    return C.prereg()["D3"]["jobs"]


def costs(bundle):
    receipts = [C.read(p) for p in sorted((bundle / "exits").glob("*.json"))]
    total = {}
    for r in receipts:
        key = f"{r['diagnostic']}|{r['partition']}"
        total[key] = total.get(key, 0.0) + r["allocated_seconds"] / 3600.0
    return {"receipts": receipts, "allocated_gpu_hours": total,
            "note": "allocated from Slurm StartTime to batch end; queue time excluded; per GPU type"}


def markdown(results):
    lines = ["# V10 看过 DEV 之后的敏感性诊断 D1–D3：结果（中性读数）", "",
             "标签：post-DEV sensitivity。这些结果不替换、不重排任何冻结主表行；解读格为预登记格的机械套用，进一步表述等负责人决定。", ""]
    one = results["D1"]
    lines += ["## D1 CoMIR_native 在 CAL 上选 epoch", "", f"状态：{one['status']}"]
    if one["status"] == "COMPLETE":
        lines += ["", "| 单元 | 选中 epoch | CAL 重放(736) | DEV32 选中 <42/<9/中位 | DEV32 冻结 e23 | DEV24 选中 | DEV24 冻结 |", "|---|---|---|---|---|---|---|"]
        for key, u in one["units"].items():
            fmt = lambda m: f"{m['success42']:.4f}/{m['success9']:.4f}/{m['median_err']:.2f}" if m else "—"
            lines.append(f"| {key} | {u['selected_epoch']} | {u['qc_replay_cal_0736']} | {fmt(u.get('DEV32_selected'))} | "
                         f"{fmt(u.get('DEV32_frozen_epoch23'))} | {fmt(u.get('DEV24_selected'))} | {fmt(u.get('DEV24_frozen_epoch23'))} |")
        p = one["paired_success42_mean_over_seeds"]
        lines += ["", f"三 seed 平均逐 core 配对差（success<42，选中减冻结）：{p['mean']:+.4f}，95% CI [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]（32 core）。",
                  f"预登记解读格：{one['grid_cell']}——{one['grid_text']}", f"差距条款：{one['gap_clause']}", ""]
    two = results["D2"]
    lines += ["## D2 MIND_2D_warp_then_describe", "", f"状态：{two['status']}"]
    if two["status"] == "COMPLETE":
        lines += ["", "| 读数 | 变体 <42/<9/中位 | 冻结 <42/<9/中位 |", "|---|---|---|"]
        for part in ("CAL", "DEV32", "DEV24"):
            a, b = two["variant"][part], two["frozen"][part]
            lines.append(f"| {part} | {a['success42']:.4f}/{a['success9']:.4f}/{a['median_err']:.2f} | "
                         f"{b['success42']:.4f}/{b['success9']:.4f}/{b['median_err']:.2f} |")
        lines += ["", "| DEV32 \\|θ*\\| 层 | 变体 n / <42 | 冻结 n / <42 |", "|---|---|---|"]
        for name in ("0-10", "10-20", "20-30"):
            a, b = two["variant"]["DEV32_strata"][name], two["frozen"]["DEV32_strata"][name]
            lines.append(f"| {name} | {a['n']} / {a['success42']:.3f} | {b['n']} / {b['success42']:.3f} |")
        p = two["paired_success42_variant_minus_frozen"]
        lines += ["", f"逐 core 配对差（变体减冻结）：{p['mean']:+.4f}，95% CI [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]。",
                  f"预登记解读格：{two['grid_cell']}——{two['grid_text']}；G4（整体更差）：{two['G4_overall_worse']}。",
                  f"冻结 CAL 重放：{two['qc_replay_frozen_cal']}。{two['always']}", ""]
    three = results["D3"]
    lines += ["## D3 补充方法 GPU gate 跨硬件", "", f"状态：{three['status']}", "", "| job | 方法 | 分区 | GPU | 结果 | CUDA 数值 |", "|---|---|---|---|---|---|"]
    for r in three["jobs"]:
        lines.append(f"| {r['job']} | {r['family']} | {r['partition']} | {r.get('gpu_name')} | {r['status']} | {r.get('cuda_numerics')} |")
    lines += ["", "PASS 只说明实现层面的数值恒等式、续跑和搜索机制在该 GPU 型号上成立，不说明与 L40S 等价。", "",
              "## 成本（已分配 GPU 小时，按诊断与分区）", ""]
    for key, value in results["costs"]["allocated_gpu_hours"].items():
        lines.append(f"- {key}: {value:.3f}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    pre = C.prereg()
    results = {"schema": "v10.sensitivity.results.v1", "label": C.LABEL, "utc": C.utc(),
               "preregistration_sha256": C.sha(C.PREREGISTRATION),
               "D1": d1(pre, C.OUT_ROOT), "D2": d2(pre, C.OUT_ROOT), "D3": d3(C.BUNDLE), "costs": costs(C.BUNDLE)}
    if args.check:
        print(json.dumps({k: results[k].get("status") for k in ("D1", "D2", "D3")}, indent=1))
        return
    C.write(C.BUNDLE / "RESULTS.json", results)
    (C.BUNDLE / "RESULTS_CN.md").write_text(markdown(results))
    print("RESULTS", C.BUNDLE / "RESULTS.json")


if __name__ == "__main__":
    main()
