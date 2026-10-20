#!/usr/bin/env python3
"""Independent cross-check of v10/V10_PHASE2_RESULTS_REPORT_20261019_CN.md and its sidecar JSON.

Written separately from v10/tools/phase2_report.py and sharing no code with it: standard library only (no numpy), raw records
read directly by path globbing, the DEV24 exclusion list typed from the owner decision, its own metric, pairing and bootstrap
code.  It recomputes a sample of the report's numbers and checks (a) the unrounded sidecar values and (b) the rendered strings
in the report.  EPFL is never recomputed: only the official analysis EPFL_RESULTS.json is read, and the report's EPFL rows
must equal it.  Exit 1 on any mismatch.

    /home/infres/yinwang/CS_QMI/env/bin/python v10/tools/phase2_report_crosscheck.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "v10" / "V10_PHASE2_RESULTS_REPORT_20261019_CN.md"
SIDECAR = REPO / "v10" / "V10_PHASE2_RESULTS_20261019.json"
H = "/home/infres/yinwang/CS_QMI/handoff"
DEV24_EXCLUDED = {"1B_F4", "1B_H4", "1B_H9", "2B_D5", "2B_G3", "2B_G5", "2B_H2", "2B_I5"}   # decision section 6.1
FAILS = []
NCHECK = 0


def ok(cond, what):
    global NCHECK
    NCHECK += 1
    if not cond:
        FAILS.append(what)
        print("FAIL", what)


def hu(x, nd, sign=False):
    with localcontext() as c:
        c.prec = 60
        q = Decimal(repr(float(x))).quantize(Decimal("1e-12"), rounding=ROUND_HALF_EVEN)
        q = q.quantize(Decimal("1e-%d" % nd), rounding=ROUND_HALF_UP)
    if q == 0:
        q = abs(q)
    t = format(q, "f")
    return ("+" if q >= 0 else "") + t if sign else t


def rows(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def rate(errs, thr):
    return sum(1 for e in errs if e < thr) / len(errs)


def m3(errs):
    return (rate(errs, 42), rate(errs, 9), statistics.median(errs))


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


report = REPORT.read_text()
side = json.loads(SIDECAR.read_text())

# ---------------------------------------------------------------- 1. per-unit DEV32 / DEV24 straight from DEV.jsonl
finals = {}
patterns = [H + "/v10_bf1_20261017/results/FINAL_DEV/*/seed_*/DEV.jsonl",
            H + "/v10_method_pipeline_20261019/families/*/results/FINAL_DEV/*/seed_*/DEV.jsonl",
            H + "/v10_native_geometry_recovery_20261019/families/*/results/FINAL_DEV/*/seed_*/DEV.jsonl"]
for pat in patterns:
    for p in sorted(glob.glob(pat)):
        parts = Path(p).parts
        fam, seed = parts[-3], int(parts[-2].split("_")[1])
        finals[(fam, seed)] = rows(p)
ok(len(finals) == 27, f"27 primary final units found ({len(finals)})")
side_units = {(u["method"], u["seed"]): u for u in side["units"] if u["phase"] == "FINAL_DEV"}
for (fam, seed), rs in sorted(finals.items()):
    errs = [r["err_full"] for r in rs]
    e24 = [r["err_full"] for r in rs if r["core"] not in DEV24_EXCLUDED]
    ok(len(errs) == 320 and len(e24) == 240, f"{fam} {seed}: 320/240 rows")
    a, b = m3(errs), m3(e24)
    su = side_units[(fam, seed)]
    ok((su["dev32"]["success42"], su["dev32"]["success9"], su["dev32"]["median_err"]) == a, f"{fam} {seed}: DEV32 sidecar")
    ok((su["dev24"]["success42"], su["dev24"]["success9"], su["dev24"]["median_err"]) == b, f"{fam} {seed}: DEV24 sidecar")
    label = fam + ("（v1，失败的适配）" if fam == "RuLSIF_exact" else "")
    line = (f"| {label} | {seed} | {su['job']} | {su['host']} | {su['selected_step']} | {hu(a[0], 4)} | {hu(a[1], 4)} | {hu(a[2], 3)} | "
            f"{hu(b[0], 4)} | {hu(b[1], 4)} | {hu(b[2], 3)} |")
    ok(line in report, f"{fam} {seed}: report row 3.1")

# ---------------------------------------------------------------- 2. seed means (table 3.2)
bym = defaultdict(list)
for (fam, seed), rs in finals.items():
    bym[fam].append(m3([r["err_full"] for r in rs]))
for fam, vals in bym.items():
    mean = tuple(statistics.fmean(v[i] for v in vals) for i in range(3))
    sm = side["main"][fam]["mean32"]
    ok(abs(sm["success42"] - mean[0]) < 1e-12 and abs(sm["success9"] - mean[1]) < 1e-12 and abs(sm["median_err"] - mean[2]) < 1e-9,
       f"{fam}: seed mean sidecar")
    ok(f"| {fam} | {len(vals)} | {hu(mean[0], 4)} | {hu(mean[1], 4)} | {hu(mean[2], 3)} |" in report, f"{fam}: report row 3.2")


# ---------------------------------------------------------------- 3. paired contrasts (point estimates) and a bootstrap
def core_scores(fam, keep):
    units = [rs for (f, s), rs in finals.items() if f == fam]
    out = {}
    for c in keep:
        p42 = statistics.fmean(sum(r["err_full"] < 42 for r in rs if r["core"] == c) / 10 for rs in units)
        p9 = statistics.fmean(sum(r["err_full"] < 9 for r in rs if r["core"] == c) / 10 for rs in units)
        med = statistics.fmean(statistics.median([r["err_full"] for r in rs if r["core"] == c]) for rs in units)
        out[c] = (p42, p9, med)
    return out


dev_cores = sorted({r["core"] for r in finals[("MF_VCS_M1", 1001)]})
dev24_cores = [c for c in dev_cores if c not in DEV24_EXCLUDED]
side_con = {(c["A"], c["B"]): c for c in side["contrasts"]}
for a, b in (("MF_VCS_M1", "MF_VCS_M0"), ("MF_VCS_M1", "MI_native"), ("MLP_VCS_perm8", "MLP_VCS_perm1"),
             ("MF_VCS_M1", "InfoNCE_host[J_perm1]")):
    for name, keep in (("DEV32", dev_cores), ("DEV24", dev24_cores)):
        sa, sb = core_scores(a, keep), core_scores(b, keep)
        d42 = statistics.fmean(sa[c][0] - sb[c][0] for c in keep)
        d9 = statistics.fmean(sa[c][1] - sb[c][1] for c in keep)
        dm = statistics.median(sa[c][2] - sb[c][2] for c in keep)
        pt = side_con[(a, b)][name]["point"]
        ok(abs(pt["s42"] - d42) < 1e-12 and abs(pt["s9"] - d9) < 1e-12 and abs(pt["med"] - dm) < 1e-9,
           f"{a}-{b} {name}: point estimates")
        if name == "DEV32" and (a, b) == ("MF_VCS_M1", "MF_VCS_M0"):
            rnd = random.Random(7)
            diffs = [sa[c][1] - sb[c][1] for c in keep]
            boots = sorted(statistics.fmean(rnd.choice(diffs) for _ in keep) for _ in range(4000))
            lo, hi = boots[int(0.025 * 4000)], boots[int(0.975 * 4000) - 1]
            glo, ghi = side_con[(a, b)][name]["core_ci"]["s9"]
            ok(abs(lo - glo) < 0.01 and abs(hi - ghi) < 0.01, f"{a}-{b} DEV32 s9 bootstrap CI ({hu(lo, 4)},{hu(hi, 4)}) vs ({hu(glo, 4)},{hu(ghi, 4)})")
            ok(side_con[(a, b)][name]["group_ci"] == side_con[(a, b)][name]["core_ci"], "singleton position groups")
    p = side_con[(a, b)]["DEV32"]
    lab = f"{a} − {b}" + ("（机制）" if side_con[(a, b)]["group"] == "mech" else "")
    ok(f"| {lab} | {hu(p['point']['s9'], 4, True)} | [{hu(p['core_ci']['s9'][0], 4, True)}, {hu(p['core_ci']['s9'][1], 4, True)}]" in report,
       f"{a}-{b}: report row 4.2")

# ---------------------------------------------------------------- 4. costs from the exit receipts
def receipts(globpat, key):
    out = {}
    for p in glob.glob(globpat):
        d = json.load(open(p))
        out[str(d[key])] = d
    return out


bf1_exits = receipts(H + "/v10_bf1_20261017/exits/*.json", "job_id")
fam_jobs = defaultdict(lambda: {"CAL": set(), "FINAL": set()})
for p in glob.glob(H + "/v10_bf1_20261017/results/*/*/*/COMPLETE.json"):
    d = json.load(open(p))
    fam_jobs[d["identity"]["family"]]["CAL" if d["identity"]["phase"] == "CAL" else "FINAL"].add(d["job_id"])
supp_exits = {}
for root in glob.glob(H + "/v10_method_pipeline_20261019/families/*") + glob.glob(H + "/v10_native_geometry_recovery_20261019/families/*"):
    supp_exits.update(receipts(root + "/exits/*.json", "job"))
    for p in glob.glob(root + "/results/*/*/*/COMPLETE.json"):
        d = json.load(open(p))
        fam_jobs[d["identity"]["family"]]["CAL" if d["identity"]["phase"] == "CAL" else "FINAL"].add(d["job"])
allx = {**bf1_exits, **supp_exits}
tot_cal = tot_fin = 0.0
for fam, js in sorted(fam_jobs.items()):
    cal = sum(allx[j]["end_unix"] - allx[j]["start_unix"] for j in js["CAL"])
    fin = sum(allx[j]["end_unix"] - allx[j]["start_unix"] for j in js["FINAL"])
    ok(all(allx[j]["exit_code"] == 0 for j in js["CAL"] | js["FINAL"]), f"{fam}: exit codes")
    sc = side["cost"][fam]
    ok(abs(sc["cal_s"] - cal) < 1e-6 and abs(sc["fin_s"] - fin) < 1e-6, f"{fam}: CAL / final GPU seconds")
    ok(f"| {fam} | " in report and f"| {hu(cal / 3600, 3)} |" in report and f"| {hu(fin / 3600, 3)} |" in report, f"{fam}: report 5.1")
    tot_cal, tot_fin = tot_cal + cal, tot_fin + fin
ok(f"| 合计 |  |  | {hu(tot_cal / 3600, 3)} |  | {hu(tot_fin / 3600, 3)} | {hu((tot_cal + tot_fin) / 3600, 3)} |" in report, "report 5.1 totals")

# queue wait of the BF1 MLP CAL job (parallel8 ledger submit -> receipt start)
par8 = json.load(open(H + "/v10_parallel8_20261018/LEDGER.json"))
u = next(x for x in par8["units"] if x["key"] == "cal_3")
w = (bf1_exits[u["job"]]["start_unix"] - ts(u["submitted_utc"])) / 3600
ok(f"| MLP_VCS_perm1 | {hu(w, 2)}（提交时带 `afterok:1045975`）" in report, f"MLP CAL queue wait {hu(w, 2)} h in report")

# ---------------------------------------------------------------- 5. diagnostics
sel_full = tot = 0
for p in glob.glob(H + "/v10_bf1_20261017/results/*/*/*/COMPLETE.json"):
    m = json.load(open(p))["selected"]["metrics"]
    tot += 1
    sel_full += (m["success42"] == 1.0 and m["success9"] == 1.0)
ok(f"BF1 四族共 {tot} 个选中 checkpoint，其中 {sel_full} 个在 CAL 上两阈值都是 80/80" in report, f"CAL saturation {sel_full}/{tot}")

ts_rows = rows(H + "/v10_bf1_20261017/results/CAL/MF_VCS_M1/candidate_00/TRAIN_STEPS.jsonl")
frac = sum(r["m1_stop_reason"] == "iteration_budget_exhausted" for r in ts_rows) / len(ts_rows)
ok(f"| 调参 c0 | 1e-4 | 10 | {hu(frac, 4)} |" in report, f"M1 c0 cap fraction {frac}")
ts_rows = rows(H + "/v10_bf1_20261017/results/FINAL_DEV/MF_VCS_M1/seed_1001/TRAIN_STEPS.jsonl")
frac = sum(r["m1_stop_reason"] == "iteration_budget_exhausted" for r in ts_rows) / len(ts_rows)
ok(f"| 最终 seed 1001 | 3e-4 | 100 | {hu(frac, 4)} |" in report, f"M1 seed 1001 cap fraction {frac}")
clip = []
for s in (1001, 1002, 1003):
    g = [r["grad_norm_preclip"] for r in rows(H + f"/v10_bf1_20261017/results/FINAL_DEV/MLP_VCS_perm1/seed_{s}/TRAIN_STEPS.jsonl")]
    clip.append(f"{s} {hu(sum(x > 1.0 for x in g) / len(g), 4)}")
ok(" / ".join(clip) in report, "MLP perm1 clip fractions")
g = [r["grad_norm_preclip"] for r in rows(H + "/v10_bf1_20261017/results/CAL/MF_VCS_perm1/candidate_02/TRAIN_STEPS.jsonl")]
late = statistics.median(g[1000:])
ok(f"{late:.3g}" in report, f"MF perm1 lr 1e-3 late gradient median {late:.3g}")

neg = n = 0
for p in glob.glob(H + "/v10_method_pipeline_20261019/families/5/results/*/*/*/CAL/*.jsonl") + \
        glob.glob(H + "/v10_method_pipeline_20261019/families/5/results/FINAL_DEV/*/*/DEV.jsonl"):
    for r in rows(p):
        n += 1
        neg += r["F_true"] < 0
ok(neg == n == 1920 and f"全部 {n} 行中真值处 F_true < 0" in report, f"RuLSIF F_true<0 {neg}/{n}")

mind = rows(H + "/v10_method_pipeline_20261019/families/6/results/FINAL_DEV/MIND_2D_adapted/seed_1001/DEV.jsonl")
cells = []
for lo, hi in ((0, 10), (10, 20), (20, 30)):
    sub = [r for r in mind if lo <= abs(r["theta"]) < hi or (hi == 30 and abs(r["theta"]) == 30)]
    cells.append(f"{len(sub)} / {hu(rate([r['err_full'] for r in sub], 42), 4)} / {hu(rate([r['err_full'] for r in sub], 9), 4)}")
ok("| MIND_2D_adapted | " + " | ".join(cells) + " |" in report, "MIND rotation strata")

hard = defaultdict(int)
nunits = 0
for (fam, seed), rs in finals.items():
    if fam == "RuLSIF_exact":
        continue
    nunits += 1
    for c in dev_cores:
        if rate([r["err_full"] for r in rs if r["core"] == c], 9) < 0.5:
            hard[c] += 1
listed = sorted(c for c, k in hard.items() if 2 * k > nunits)
ok(all(f"| {c} | {c[:2]} |" in report for c in listed) and len(listed) == len(side["diagnostics"]["hard"]["cores"]), "hard cores")

# ---------------------------------------------------------------- 6. replication
rep = {}
for b in ("v10_seed_replication_pool_20261019", "v10_seed_replication_hw_20261019"):
    for p in glob.glob(f"{H}/{b}/results/*/DEV.jsonl"):
        key = Path(p).parent.name
        env = json.load(open(glob.glob(str(Path(p).parent / "ENV_*.json"))[0]))
        rep[key] = (m3([r["err_full"] for r in rows(p)]), env["gpu_name"], Path(p).parent / "COMPLETE.json")
ok(len(rep) == 12, "12 replication units")
for key, (m, gpu, cp) in rep.items():
    ok(f"| {key} | " in report and f"| {hu(m[0], 4)} | {hu(m[1], 4)} | {hu(m[2], 3)} |" in report, f"replication {key} row")
ngpu = defaultdict(int)
for key, (m, gpu, cp) in rep.items():
    ngpu[gpu] += 1
ok(ngpu["NVIDIA RTX PRO 6000 Blackwell Server Edition"] == 10 and ngpu["NVIDIA H100 NVL"] == 2, f"replication GPU split {dict(ngpu)}")

# ---------------------------------------------------------------- 7. same-total-cost: when the M1 selection became available
led = rows(H + "/v10_bf1_20261017/COST_LEDGER.jsonl")
sel = json.load(open(H + "/v10_bf1_20261017/results/CAL/MF_VCS_M1/SELECTION.json"))["selected"]
ev = [r for r in led if r["event"] == "CHECKPOINT" and r.get("unit") == f"candidate_{sel['candidate']['order']:02d}"
      and r.get("step") == sel["choice"]["step"]]
ok(len(ev) == 1 and abs(side["same_total_cost"]["MF_VCS_M1"]["sel_t"] - ev[0]["allocated_seconds"]) < 1e-9,
   "M1 selection availability time")
ok(f"| {hu(ev[0]['allocated_seconds'] / 3600, 3)} |" in report, "M1 availability in report")

# node57 clock lag lower bounds from the START events
lags = []
for r in led:
    if r["event"] == "START" and r["job_id"] in bf1_exits:
        d = bf1_exits[r["job_id"]]["start_unix"] - ts(r["utc"])
        if d > 0:
            lags.append(d)
ok(len(lags) == 4 and all(85 <= d <= 95 for d in lags), f"BF1 node clock lags {lags}")

# ---------------------------------------------------------------- 8. BF RuLSIF_exact_v2 (A1)
v2 = {}
for p in sorted(glob.glob(H + "/v10_rulsif_v2_20261019/results/FINAL_DEV/RuLSIF_exact_v2/seed_*/DEV.jsonl")):
    seed = int(Path(p).parent.name.split("_")[1])
    env = json.load(open(glob.glob(str(Path(p).parent / "ENV_*.json"))[0]))
    v2[seed] = (m3([r["err_full"] for r in rows(p)]), env["gpu_name"], env["host"])
ok(sorted(v2) == [1001, 1002, 1003], "RuLSIF v2 finals present")
short = {"NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTX PRO 6000", "NVIDIA L40S": "L40S", "NVIDIA H100 NVL": "H100 NVL"}
for seed, (m, gpu, host) in v2.items():
    ok(f"| {seed} | " in report and f"{host}（{short[gpu]}）" in report and
       f"| {hu(m[0], 4)} | {hu(m[1], 4)} | {hu(m[2], 3)} |" in report, f"RuLSIF v2 seed {seed} row ({short[gpu]})")
pool = [v2[s][0] for s in v2]
mean = tuple(statistics.fmean(x[i] for x in pool) for i in range(3))
ok(f"| RuLSIF_exact_v2（A1）合并 | 3 | {hu(mean[0], 4)} | {hu(mean[1], 4)} | {hu(mean[2], 3)} |" in report, "RuLSIF v2 pooled mean row")
l40 = [v2[s][0] for s in v2 if v2[s][1] == "NVIDIA L40S"]
ok(len(l40) == 1 and f"| RuLSIF_exact_v2（A1）L40S 层 | 1 | {hu(l40[0][0], 4)} | {hu(l40[0][1], 4)} | {hu(l40[0][2], 3)} |" in report,
   "RuLSIF v2 L40S stratum row")

# ---------------------------------------------------------------- 9. K8/K32 capacity extension: deltas vs the K16 finals
OUTC = "/projects/EEG-foundation-model/yinghao/outputs_v10"
tab = json.load(open(H + "/v10_capacity_ext_20261019/hw_amendment_20261019/CAPACITY_TABLE_HW.json"))
nunit = 0
for cell in tab["cells"]:
    for e in cell["units"]:
        root = OUTC + ("/capacity_ext_hw_20261019/" if e["source"] == "hw_amendment" else "/capacity_ext_20261019/")
        mk = m3([r["err_full"] for r in rows(root + e["unit"] + "/DEV.jsonl")])
        m16 = m3([r["err_full"] for r in finals[(cell["family"], e["seed"])]])
        d42 = mk[0] - m16[0]
        ok(abs(e["DEV32_delta"]["success42"] - d42) < 1e-12 and f"| {hu(d42, 4, True)} |" in report,
           f"capacity {e['unit']} delta s42")
        nunit += 1
ok(nunit == 18, "18 capacity units")
grids = {v[b][m]["grid"] for c in tab["cells"] for v in [c["pooled"]] + [x for g, x in c["strata"].items() if g == "NVIDIA L40S"]
         for b in ("DEV32", "DEV24") for m in ("success42", "success9")}
ok(grids == {"B_no_detectable_response"} and "即所有 K 效应都是 B 格" in report, "all pooled / L40S capacity cells are grid B")

# ---------------------------------------------------------------- 10. post-DEV sensitivity: frozen rows equal the frozen units
res = json.load(open(H + "/v10_sensitivity_20261019/RESULTS.json"))
for s in (1001, 1002, 1003):
    m = m3([r["err_full"] for r in finals[("CoMIR_native", s)]])
    fz = res["D1"]["units"][f"FINAL_{s}"]["DEV32_frozen_epoch23"]
    ok((fz["success42"], fz["success9"], fz["median_err"]) == m, f"D1 frozen epoch 23 seed {s} equals the CoMIR final")
ok(f"{hu(res['D1']['paired_success42_mean_over_seeds']['mean'], 4, True)}" in report and res["D1"]["grid_cell"] in report
   and res["D2"]["grid_cell"] in report, "D1/D2 paired means and grid cells in the report")

# ---------------------------------------------------------------- 11. EPFL: only the official analysis, never the raw results
EP = H + "/v10_epfl_production_20261019/analysis/EPFL_RESULTS.json"
ok("单元宏 s<42" not in report.split("## 10 EPFL")[0], "no EPFL table before section 10")
if not Path(EP).exists():
    ok("- 状态：pending（`EPFL_RESULTS.json` 尚不存在）。" in report, "EPFL pending without the official analysis")
else:
    eb = open(EP, "rb").read()
    ok(hashlib.sha256(eb).hexdigest() in report, "EPFL_RESULTS.json sha256 printed in the report")
    ed = json.loads(eb)
    if ed.get("schema") == "v10.epfl.results.v1":
        sec = report.split("## 10 EPFL")[1].split("## 11 ")[0]
        for fam, e in ed["families"].items():
            for sd, r in e["finals"].items():
                if r is None:
                    ok(f"| {fam} | {sd} | pending |" in sec, f"EPFL {fam} {sd}: pending row")
                    continue
                um = r["stats"]["DEV72_primary"]["unit_macro"]
                rg = r["stats"]["DEV72_primary"]["registration_level"]
                ok(f"| {fam} | {sd} | {r['release']} | " in sec and f"| {r['job']} | {r['selected']['step']} | {hu(um['success42'], 4)} | "
                   f"{hu(um['success9'], 4)} | {hu(um['median_native_px'], 3)} | {hu(rg['success42'], 4)} | {hu(rg['success9'], 4)} |" in sec,
                   f"EPFL {fam} {sd}: final row equals the official analysis")
            sm = e["summary"]
            if sm and sm.get("seed_mean"):
                m = sm["seed_mean"]["DEV72_primary"]["unit_macro"]
                b = sm["bootstrap95"]["DEV72_primary"]["success42"]
                ok(f"| {fam} | {len(sm['seeds_complete'])} | " in sec and f"{hu(m['success42'], 4)} [{hu(b[0], 4)}, {hu(b[1], 4)}]" in sec,
                   f"EPFL {fam}: seed mean and interval equal the official analysis")
        for t, x in ed["rulsif_v2"].items():
            ok(f"- T{t}（role `{x['role']}`" in sec and x["status"] in sec and "容量与成本共同受约束的核基线" in sec,
               f"EPFL RuLSIF v2 T{t}: status and label")

print(f"{NCHECK} checks, {len(FAILS)} failures")
sys.exit(1 if FAILS else 0)
