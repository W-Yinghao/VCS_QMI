"""V10 TEST_REPEAT134 analysis, v4 same-job pairing (CPU, re-runnable, deterministic; run only on recorded outputs).

    analyze.py           write <bundle>/analysis/TEST_REPEAT134_RESULTS.json and v10/test_repeat134/V10_TEST_REPEAT134_REPORT_CN.md
    analyze.py --check   render in memory and compare with both files

Reads the frozen records (bundle == committed), each unit's recorded DEV rows, the unit jobs' outputs (same-job DEV32
re-evaluation, in-job check, TEST rows) when present, INPUT_LOCK.json, the exit receipts and every DEV re-evaluation pair
(gate records of v1 / v2 and the in-job re-evaluations) for the evaluation-noise section.  Every TEST unit is verified
before its numbers are used; anything absent is "pending".  Paired differences use phase2_report.paired unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as C  # noqa: E402

REPORT = C.REPO / C.RECORDS / "V10_TEST_REPEAT134_REPORT_CN.md"
RESULTS = C.BUNDLE / "analysis" / "TEST_REPEAT134_RESULTS.json"
MK = ("success42", "success9", "median_err")
PR = None   # phase2_report (half-up formatting and the paired bootstrap)


def load_phase2():
    spec = importlib.util.spec_from_file_location("phase2_report", C.REPO / "v10/tools/phase2_report.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["phase2_report"] = mod
    spec.loader.exec_module(mod)
    return mod


def f4(x):
    return "" if x is None else PR.f4(x)


def f3(x):
    return "" if x is None else PR.f3(x)


def s4(x):
    return PR.s4(x)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def pair_summary(diag: dict) -> dict:
    cr, cn = diag["counts"]["recorded"], diag["counts"]["new"]
    return {"d_lt42": cn["lt42"] - cr["lt42"], "d_lt9": cn["lt9"] - cr["lt9"], "flips": diag["flips"],
            "d_median": diag["median"]["new"] - diag["median"]["recorded"], "abs_delta_err_full": diag["abs_delta_err_full"],
            "bitwise_rows": diag["bitwise_rows"], "F_true_bitwise": diag["bitwise_fields"].get("F_true"),
            "identity_and_truths_bitwise": diag["identity_equal"] and diag["perturbation_fields_equal"]}


# ------------------------------------------------------------------------------------------------------------ collect
def collect() -> dict:
    rec = C.records()
    man, prot = rec["manifest"], rec["protocol"]
    act_sha = rec["activation_sha256"]
    test_cores, kept = man["test"]["cores"], set(man["test"]["sensitivity112"]["cores"])
    by_key = {u["key"]: u for u in man["units"]}
    lock_path = C.BUNDLE / "INPUT_LOCK.json"
    lock = C.read(lock_path) if lock_path.exists() else None
    receipts = {p.stem: C.read(p) for p in sorted((C.BUNDLE / "exits").glob("*.json"))}
    units, truths, noise = [], {}, []
    for u in man["units"]:
        d = Path(u["unit_dir"])
        C.require(C.sha(d / "DEV.jsonl") == u["dev"]["rows_sha256"], f"{u['key']}: recorded DEV rows changed")
        dev = C.lines(d / "DEV.jsonl")
        C.require(C.metrics(dev) == u["dev"]["DEV32"], f"{u['key']}: DEV32 recomputation")
        plan = C.test_plan(rec, u["key"])
        entry = {"u": u, "plan": plan, "dev_rows": dev, "test": None, "blocked": None, "reeval_rows": None}
        out = C.BUNDLE / "results" / "TEST_REPEAT" / u["key"]
        if (out / "COMPLETE.json").exists():
            comp = C.read(out / "COMPLETE.json")
            C.require(comp["status"] == "COMPLETE" and comp["identity"] == u["identity"] and comp["checkpoint"] == u["checkpoint"]
                      and comp["activation_sha256"] == act_sha, f"{u['key']}: TEST record identity")
            for name, value in comp["artifacts"].items():
                C.require(C.sha(out / name) == value, f"{u['key']}/{name}: artifact hash")
            rows = C.lines(out / "TEST_REPEAT.jsonl")
            C.require(len(rows) == 1340 and {(r["core"], r["perturbation"]) for r in rows}
                      == {(c, i) for c in test_cores for i in range(10)}, f"{u['key']}: TEST coverage")
            C.require(all(r["role"] == C.ROLE and r["checkpoint_sha256"] == u["checkpoint"]["sha256"] for r in rows),
                      f"{u['key']}: TEST role / checkpoint")
            C.require(C.metrics(rows) == comp["TEST134"] and C.metrics([r for r in rows if r["core"] in kept]) == comp["TEST112"],
                      f"{u['key']}: TEST metrics recomputation")
            C.require(comp["hardware"]["gpu_type"] == plan["gpu_type"] == u["hardware"]["gpu_type"], f"{u['key']}: hardware")
            ref = comp["dev_reevaluation"]
            reeval = C.lines(out / "DEV_REEVAL.jsonl")
            C.require(ref["check"]["status"] == "PASS" and C.sha(out / "DEV_REEVAL.jsonl") == ref["rows_sha256"]
                      and C.metrics(reeval) == ref["DEV32"], f"{u['key']}: same-job DEV re-evaluation")
            C.require(comp["job"] in comp["jobs"] and all(j in receipts for j in comp["jobs"])
                      and receipts[comp["job"]]["exit_code"] == 0, f"{u['key']}: exit receipts of the unit's jobs")
            for r in rows:
                k = (r["core"], r["perturbation"])
                C.require(truths.setdefault(k, (r["theta"], tuple(r["t"]))) == (r["theta"], tuple(r["t"])),
                          f"perturbation truth differs at {k}")
            entry["test"] = {"comp": comp, "rows": rows, "alloc_s": sum(receipts[j]["allocated_seconds"] for j in comp["jobs"])}
            entry["reeval_rows"] = reeval
        elif (out / "BLOCKED.json").exists():
            entry["blocked"] = C.read(out / "BLOCKED.json")
            if (out / "DEV_REEVAL.jsonl").exists():
                entry["reeval_rows"] = C.lines(out / "DEV_REEVAL.jsonl")
        if entry["reeval_rows"] is not None and len(entry["reeval_rows"]) == 320:
            diag = C.gate_diagnostics(dev, entry["reeval_rows"], C.read(d / "DEV.identity.json"),
                                      C.read(out / "DEV_REEVAL.identity.json"))
            noise.append({"source": "in-job (v4)", "unit": u["key"], "family": u["family"], "gpu_type": plan["gpu_type"],
                          "job": (entry["test"]["comp"]["job"] if entry["test"] else entry["blocked"]["job"]),
                          **pair_summary(diag)})
        units.append(entry)
    for gp in sorted(C.BUNDLE.glob("gate/*/*/GATE.json")):          # v1 / v2 gate re-evaluations, any verdict
        g = C.read(gp)
        if not receipts.get(g["job"]) or not (gp.parent / "DEV.jsonl").exists():
            continue
        u = by_key[g["unit"]]
        d = Path(u["unit_dir"])
        new = C.lines(gp.parent / "DEV.jsonl")
        if len(new) != 320:
            continue
        diag = C.gate_diagnostics(C.lines(d / "DEV.jsonl"), new, C.read(d / "DEV.identity.json"),
                                  C.read(gp.parent / "DEV.identity.json"))
        noise.append({"source": f"gate ({g.get('status')})", "unit": u["key"], "family": u["family"],
                      "gpu_type": g["gpu_type"], "job": g["job"], **pair_summary(diag)})
    noise.sort(key=lambda x: (x["family"], x["gpu_type"], x["source"], x["job"]))
    return {"rec": rec, "lock": lock, "receipts": receipts, "units": units, "truth_keys": len(truths), "noise": noise}


# ------------------------------------------------------------------------------------------------------------ analyse
def slim(rows):
    return [{"core": r["core"], "err": r["err_full"]} for r in rows]


def groups_of(cores):
    g = {}
    for c in cores:
        g.setdefault(c.split("_", 1)[1], []).append(c)
    return g


def analyse(D: dict, pr) -> dict:
    man = D["rec"]["manifest"]
    tc = man["test"]["cores"]
    sets = {"TEST134": (tc, man["test"]["position_groups"]),
            "TEST112": (man["test"]["sensitivity112"]["cores"], man["test"]["sensitivity112"]["position_groups"])}
    fam = defaultdict(list)
    for e in D["units"]:
        fam[e["u"]["family"]].append(e)
    means = {}
    for f, es in fam.items():
        prim = [e for e in es if e["u"]["tier"] == "primary"]
        done = all(e["test"] is not None for e in prim)
        row = {"n": len(prim), "gpus": sorted({e["plan"]["gpu_type"] for e in prim}),
               "DEV32_recorded": {k: statistics.fmean(e["u"]["dev"]["DEV32"][k] for e in prim) for k in MK}}
        for s in ("TEST134", "TEST112"):
            row[s] = ({k: statistics.fmean(e["test"]["comp"][s][k] for e in prim) for k in MK} if done else None)
        row["DEV32_same_job"] = ({k: statistics.fmean(e["test"]["comp"]["dev_reevaluation"]["DEV32"][k] for e in prim)
                                  for k in MK} if done else None)
        means[f] = row
    contrasts = []
    for a, b, grp in pr.CONTRASTS:
        A = [e for e in fam[a] if e["u"]["tier"] == "primary"]
        Bu = [e for e in fam[b] if e["u"]["tier"] == "primary"]
        dev_cores = A[0]["u"]["dev"]["cores"]
        entry = {"A": a, "B": b, "group": grp,
                 "DEV32_recorded": pr.paired([{"dev_rows": slim(e["dev_rows"])} for e in A],
                                             [{"dev_rows": slim(e["dev_rows"])} for e in Bu], dev_cores, groups_of(dev_cores),
                                             f"{a}|{b}|DEV32")}
        ready = all(e["test"] is not None for e in A + Bu)
        entry["DEV32_same_job"] = (pr.paired([{"dev_rows": slim(e["reeval_rows"])} for e in A],
                                             [{"dev_rows": slim(e["reeval_rows"])} for e in Bu], dev_cores,
                                             groups_of(dev_cores), f"TEST_REPEAT134|{a}|{b}|DEV32_same_job") if ready else None)
        for name, (cores, groups) in sets.items():
            entry[name] = (pr.paired([{"dev_rows": slim(e["test"]["rows"])} for e in A],
                                     [{"dev_rows": slim(e["test"]["rows"])} for e in Bu], cores, groups,
                                     f"TEST_REPEAT134|{a}|{b}|{name}") if ready else None)
        contrasts.append(entry)
    groups = defaultdict(list)
    for n in D["noise"]:
        groups[(n["family"], n["gpu_type"])].append(n)
    noise_summary = []
    for (f, g), xs in sorted(groups.items()):
        noise_summary.append({"family": f, "gpu_type": g, "pairs": len(xs),
                              "d_lt42": [x["d_lt42"] for x in xs], "d_lt9": [x["d_lt9"] for x in xs],
                              "flips_lt9_max": max(x["flips"]["success9"] for x in xs),
                              "abs_d_median_max": max(abs(x["d_median"]) for x in xs),
                              "q50_median": statistics.median(x["abs_delta_err_full"]["q50"] for x in xs),
                              "q90_max": max(x["abs_delta_err_full"]["q90"] for x in xs),
                              "max_abs": max(x["abs_delta_err_full"]["max"] for x in xs),
                              "bitwise_rows": [x["bitwise_rows"] for x in xs],
                              "F_true_bitwise_all": all(x["F_true_bitwise"] for x in xs)})
    cost = Counter()
    for e in D["units"]:
        if e["test"]:
            cost[e["test"]["comp"]["hardware"]["gpu_type"]] += e["test"]["alloc_s"] / 3600
    return {"means": means, "contrasts": contrasts, "noise_summary": noise_summary, "cost": dict(cost),
            "done": sum(e["test"] is not None for e in D["units"]), "blocked": sum(e["blocked"] is not None for e in D["units"])}


# ------------------------------------------------------------------------------------------------------------- render
def render(D: dict, A: dict) -> str:
    rec = D["rec"]
    man, prot, dec = rec["manifest"], rec["protocol"], rec["decision"]
    dec4 = rec["decision_v4"]
    L = []
    w = L.append
    cell = lambda m: f"{f4(m['success42'])} / {f4(m['success9'])} / {f3(m['median_err'])}" if m else "pending"
    w("# V10 TEST_REPEAT134：BF 重复基准（单独报告，不是独立确认；v4 同作业配对）")
    w("")
    w(f"**状态：最终单元 {A['done']}/42 完成，{A['blocked']} 个 BLOCKED，其余 pending。** 本文由 "
      "`scripts/v10_test_repeat/analyze.py` 从冻结记录与已完成的作业输出生成；同一输入两次运行逐字节相同。")
    w("")
    w("> **读法。** " + prot["reporting"]["label"] + "。规则原文：" + "；".join(prot["reporting"]["rules"]) + "。")
    w("")
    w("## 1 负责人决定与冻结记录")
    w("")
    w(f"- 启动（{dec['date']}，UTC {dec['time_utc']}）问题原文：「{dec['question_verbatim']}」选项：「{dec['chosen_option_verbatim']}」"
      f"——「{dec['chosen_option_text_verbatim']}」")
    w(f"- v4（{dec4['date']}）问题原文：「{dec4['question_verbatim']}」选项：「{dec4['chosen_option_verbatim']}」——"
      f"「{dec4['chosen_option_text_verbatim']}」未选方案：" + "；".join(x["option"] for x in dec4["options_not_chosen"]) + "。")
    for name, value in rec["shas"].items():
        w(f"- `{C.RECORDS}/{name}` sha256 `{value}`")
    t = man["test"]
    w(f"- TEST134 core 列表 sha256 `{t['cores_sha256']}`；TEST112 敏感性集 sha256 `{t['sensitivity112']['cores_sha256']}`。"
      f"先前暴露：{t['prior_exposure']}。扰动：{prot['perturbations']['rule']}。{prot['perturbations']['difference_from_V9_G1']}。")
    ij = prot["in_job_check"]
    w(f"- 作业内检查：(a) {ij['a']}；(e) {ij['e']}；(f) {ij['f']}。{ij['band_note']}")
    w("")
    lk = D["lock"]
    w("## 2 TEST 输入锁")
    w("")
    w("pending（INPUT_LOCK.json 尚不存在）。" if lk is None else
      f"状态 {lk['status']}（作业 {lk['job']}；{lk['raw_members_verified']} 个原始成员 sha256 核对；与 V9-G1 预处理缓存"
      f"{'逐位相同' if lk['v9_cache_comparison']['identical'] else '不同'}）。")
    w("")
    w("## 3 逐单元：TEST 与同作业 DEV32 并列，并附记录的 DEV32（不合并）")
    w("")
    rows = []
    for e in D["units"]:
        u, tr = e["u"], e["test"]
        c = tr["comp"] if tr else None
        st = "BLOCKED" if e["blocked"] else "pending"
        rows.append([u["family"], u["seed"], u["tier"].replace("secondary_replication", "secondary"), u["row_label"],
                     e["plan"]["gpu_type"], cell(c["TEST134"]) if c else st, cell(c["TEST112"]) if c else st,
                     cell(c["dev_reevaluation"]["DEV32"]) if c else st, cell(u["dev"]["DEV32"]),
                     c["job"] if c else (e["blocked"]["job"] if e["blocked"] else "")])
    for line in table(["族", "seed", "层", "标签", "GPU", "TEST134 s<42 / s<9 / 中位", "TEST112", "DEV32（同作业重评）",
                       "DEV32（记录）", "作业"], rows):
        w(line)
    w("")
    w("## 4 族的 seed 均值（主体 seeds 1001–1003）")
    w("")
    rows = [[f, m["n"], "，".join(m["gpus"]), cell(m["TEST134"]), cell(m["TEST112"]), cell(m["DEV32_same_job"]),
             cell(m["DEV32_recorded"])] for f, m in A["means"].items()]
    for line in table(["族", "seed 数", "GPU", "TEST134", "TEST112", "DEV32（同作业）", "DEV32（记录）"], rows):
        w(line)
    w("")
    w("## 5 以 core 为单位的配对差（success<42，A − B；描述性）")
    w("")

    def pc(p):
        if p is None:
            return "pending"
        return (f"{s4(p['point']['s42'])} [{s4(p['core_ci']['s42'][0])}, {s4(p['core_ci']['s42'][1])}] · 位置组 "
                f"[{s4(p['group_ci']['s42'][0])}, {s4(p['group_ci']['s42'][1])}]")
    rows = [[f"{c['A']} − {c['B']}" + ("（机制）" if c["group"] == "mech" else ""), pc(c["TEST134"]), pc(c["TEST112"]),
             pc(c["DEV32_same_job"]), pc(c["DEV32_recorded"])] for c in A["contrasts"]]
    for line in table(["对照", "TEST134 [core 95%] · 位置组", "TEST112", "DEV32（同作业）", "DEV32（记录，与第二阶段报告相同的键）"], rows):
        w(line)
    w("")
    w("这些区间不是检验，不能读作优劣；RuLSIF_exact（v1）与 RuLSIF_exact_v2（A1）不进入配对。")
    w("")
    w("## 6 评估噪声（同一冻结 checkpoint 的 DEV32 重评，对照记录的 DEV32）")
    w("")
    w(prot["evaluation_noise"]["statement"])
    w("")
    rows = [[n["family"], n["gpu_type"], n["source"], n["job"], f"{n['d_lt42']:+d}", f"{n['d_lt9']:+d}",
             f"{n['flips']['success42']} / {n['flips']['success9']}", f"{n['d_median']:+.4f}",
             f3(n["abs_delta_err_full"]["q50"]), f3(n["abs_delta_err_full"]["q90"]), f3(n["abs_delta_err_full"]["max"]),
             n["bitwise_rows"], "是" if n["F_true_bitwise"] else "否"] for n in D["noise"]]
    for line in table(["族", "GPU", "来源", "作业", "Δ<42", "Δ<9", "翻转 <42 / <9", "Δ中位", "|Δerr| q50", "q90", "max",
                       "逐位相同行", "F_true 逐位"], rows or [["—"] * 13]):
        w(line)
    w("")
    rows = [[s["family"], s["gpu_type"], s["pairs"], f"{min(s['d_lt42']):+d}…{max(s['d_lt42']):+d}",
             f"{min(s['d_lt9']):+d}…{max(s['d_lt9']):+d}", s["flips_lt9_max"], f"{s['abs_d_median_max']:.4f}",
             f3(s["q50_median"]), f3(s["q90_max"]), f3(s["max_abs"]), f"{min(s['bitwise_rows'])}…{max(s['bitwise_rows'])}"]
            for s in A["noise_summary"]]
    for line in table(["族", "GPU", "对数", "Δ<42 范围", "Δ<9 范围", "<9 翻转最大", "|Δ中位| 最大", "q50 中位", "q90 最大",
                       "max", "逐位相同行范围"], rows or [["—"] * 11]):
        w(line)
    w("")
    w("## 7 成本")
    w("")
    est = prot["cost_estimate"]
    w(f"- 预估（{est['basis']}）：" + "，".join(f"{k} {v:.2f}" for k, v in est["unit_jobs_gpu_h"].items())
      + f" GPU-h；输入准备 {est['prepare_gpu_h']:.2f}；合计 {est['total_gpu_h']:.2f} GPU-h。")
    w("- 实际（exit receipt 的已分配时间）：" + ("，".join(f"{k} {v:.3f}" for k, v in sorted(A["cost"].items())) or "pending") + "。")
    w("")
    w("## 8 局限")
    w("")
    w("- TEST134 已在 V9-G1 读过一次；这是重复基准，不是独立确认；不与 DEV 合并；不作方法优劣、强基线或新颖性结论（决定 §5）。")
    w("- 评估本身有噪声（第 6 节）：小于该噪声的方法差，在 TEST 和 DEV（包括 V10 主 DEV 表）上都不可解释。")
    w("- 每个单元在产生其 DEV32 的 GPU 型号上评估；不声称跨硬件等价。")
    w("")
    return "\n".join(L) + "\n"


def sidecar(D: dict, A: dict) -> dict:
    rec = D["rec"]
    return {"schema": "v10.test_repeat134.results.v4", "scope": "evaluator_only", "records": rec["shas"],
            "done": A["done"], "blocked": A["blocked"],
            "units": [{"key": e["u"]["key"], "family": e["u"]["family"], "seed": e["u"]["seed"], "tier": e["u"]["tier"],
                       "row_label": e["u"]["row_label"], "gpu_type": e["plan"]["gpu_type"], "DEV32_recorded": e["u"]["dev"]["DEV32"],
                       "DEV32_same_job": e["test"]["comp"]["dev_reevaluation"]["DEV32"] if e["test"] else None,
                       "TEST134": e["test"]["comp"]["TEST134"] if e["test"] else None,
                       "TEST112": e["test"]["comp"]["TEST112"] if e["test"] else None,
                       "blocked": e["blocked"]["check"] if e["blocked"] else None,
                       "job": e["test"]["comp"]["job"] if e["test"] else None} for e in D["units"]],
            "family_means": A["means"], "contrasts": A["contrasts"], "noise": D["noise"], "noise_summary": A["noise_summary"],
            "cost": A["cost"], "input_lock": None if D["lock"] is None else D["lock"]["status"], "truth_keys": D["truth_keys"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    global PR
    pr = PR = load_phase2()
    D = collect()
    A = analyse(D, pr)
    outs = ((REPORT, render(D, A)), (RESULTS, C.dumps(sidecar(D, A))))
    ok = True
    for path, text in outs:
        if a.check:
            same = path.exists() and path.read_text() == text
            ok &= same
            print(f"{path}: {'identical' if same else 'DIFFERS'}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print("wrote", path)
    print(f"TEST_REPEAT134 v4: {A['done']}/42 complete, {A['blocked']} blocked, {len(D['noise'])} DEV re-evaluation pairs")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
