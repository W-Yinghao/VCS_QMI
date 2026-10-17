#!/usr/bin/env python3
"""V10 phase-1 (T00-T02) stage report generator -> v10/V10_PHASE1_STAGE_REPORT_20261017.md (Chinese).

Layout follows the package's 「首批回传格式」 (v10/package_v10_server_v1_20261016/01_SERVER_AGENT_PROMPT_CN.md): the report opens
with 当前真实提交 / 已完成代码与验证 / 仍缺的输入 / 实际测量表 / 下一批单元与资源, then the open owner decisions, the failed
attempts, the main-session decisions, queue handling, limitations and the file list.

Every number, table and git fact is read at generation time from
  v10/T00   PREFLIGHT.json, PREFLIGHT_RUN.log, MANIFEST_BUILD_SUMMARY.json, MANIFEST_SHA256.txt, host_map.json, EXPOSURE_NOTES.md
  v10/T01   TORCH_VALIDATION.json, GEOMETRY_GRADIENT_CHECK.json
  v10/T02   T02_TIMING_{cpu,cuda}.json, T02_LONG_{cpu,cuda}.csv, BUDGET.json, UNIT_LEDGER.csv
  package   configs/V10_PROTOCOL.json, MANIFEST_SHA256.json, PACKAGE_STATUS.json, core/benchmark_torch.py (its tolerance),
            synthetic/src/v10_synthetic.py (CLI defaults, by ast) and synthetic/validation/example_data (truth.json S values)
  git       branch, HEAD, log, status; code hashes recomputed from git objects at every commit
  code      slurm/v10 (queue rules, sbatch resources), src/v10/adapter.py (method tuples, by ast), src/v9/reg (MIND search)
  runs      /home/infres/yinwang/CS_QMI/slurm_logs/v10_*.out and the raw files under outputs_v10 named in T02_TIMING_cuda.json
            (benchmark JSONs, checked against the recorded sha256; train_steps_per_step.json, sha256 printed)
  queue     one read-only `squeue -u $USER` call (skip with --no-squeue)
Hand-written: prose, the options / recommendations of the open decisions, PROPOSAL (the one proposal parameter) and
MAIN_SESSION (main-session statements without a file record, rendered with that attribution).
Consistency checks (BUDGET recomputed, CSV vs JSON, hashes, check counts) abort the run on any mismatch.  Writes only the report.

    /home/infres/yinwang/CS_QMI/env/bin/python scripts/v10/phase1_report.py [--output PATH] [--no-squeue]
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import os
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SELF_REL = Path(__file__).resolve().relative_to(REPO).as_posix()
V10 = REPO / "v10"
T00, T01, T02 = V10 / "T00", V10 / "T01", V10 / "T02"
PKG = V10 / "package_v10_server_v1_20261016"
PKG_REL = PKG.relative_to(REPO).as_posix()
LOGS = Path("/home/infres/yinwang/CS_QMI/slurm_logs")
REPORT_DEFAULT = V10 / "V10_PHASE1_STAGE_REPORT_20261017.md"
CODE_PATHS = ("src/v10", "scripts/v10")               # = src/v10/adapter.py code_hash() = job header "code="
PIN_PATHS = ("src/v10", "scripts/v10", "slurm/v10")    # = slurm/v10/submit.sh job pin = job header "pin="
RESULT_PATHS = ("v10/T00", "v10/T01", "v10/T02", "src/v10", "scripts/v10", "slurm/v10", "tests/v10")
STATUS_PATHS = RESULT_PATHS + ("src/v9", "scripts/v9", "slurm/v9")
# 「首批回传格式」 attachments: PREFLIGHT.json、HOST_ADAPTER_AUDIT.md、manifest及hash、TORCH_VALIDATION.json、GEOMETRY_GRADIENT_CHECK.json、
# T02长表、BUDGET.json、单元ledger
PACKAGE_REQUIRED = ("v10/T00/PREFLIGHT.json", "v10/T00/HOST_ADAPTER_AUDIT.md", "v10/T00/V10_DATA_MANIFEST.csv",
                    "v10/T00/MANIFEST_SHA256.txt", "v10/T01/TORCH_VALIDATION.json", "v10/T01/GEOMETRY_GRADIENT_CHECK.json",
                    "v10/T02/T02_LONG_cpu.csv", "v10/T02/T02_LONG_cuda.csv", "v10/T02/BUDGET.json", "v10/T02/UNIT_LEDGER.csv")

# Main-session statements (task brief of 2026-10-17) without any file record; rendered only with this attribution.
MAIN_SESSION = {"attribution": "主会话陈述（2026-10-17 任务说明；仓库、日志与结果文件中无记录）",
                "nice_raise_max": 1300, "nice_restored_utc": "2026-10-17T01:36:48Z"}
# The only proposal parameter (recommended default of open decision 1; a design value, not a measurement).
PROPOSAL = {"m1_relative_gap_target": 1e-2}

OUTLIER_FACTOR = 10   # CPU timing anomaly rule of section 4.4 (a reporting rule, not a measurement)
HEAD_METHODS = (("permutation1", "perm1"), ("permutation8", "perm8"), ("dense", "dense"),
                ("blocked_dense", "分块 dense（ckpt 重算）"), ("blocked_dense_graph_retained", "分块 dense（保留图）"),
                ("moments", "moments"))


# ------------------------------------------------------------------------------------------------ small helpers
def fail(msg: str):
    raise SystemExit(f"[phase1_report] consistency check failed: {msg}")


def check(cond, msg: str):
    if not cond:
        fail(msg)


def close(a, b, rel=1e-9, abs_=1e-12) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= max(abs_, rel * max(abs(float(a)), abs(float(b))))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    return sha256_bytes(Path(p).read_bytes())


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_csv(p) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x):
    if x is None or isinstance(x, bool):
        return None if x is None else float(x)
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).strip()
    return None if x in ("", "None", "nan", "NaN") else float(x)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_utc(iso: str) -> str:
    return dt.datetime.fromisoformat(iso).astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fe(x, d=2) -> str:
    """compact scientific: 2.94e-5"""
    if x is None:
        return "—"
    x = float(x)
    if x == 0:
        return "0"
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}e{int(e)}"


def fg(x, sig=3) -> str:
    if x is None:
        return "—"
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-3:
        return fe(x, sig - 1)
    return f"{x:#.{sig}g}".rstrip(".")


def fi(x) -> str:
    return "—" if x is None else str(int(round(float(x))))


def fv(x) -> str:
    """plain value as given (design grids: rho, lr)"""
    return f"{float(x):g}"


def fms(x) -> str:
    if x is None:
        return "—"
    x = float(x)
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def fmib(b) -> str:
    b = num(b)
    if b is None:
        return "—"
    m = b / 2 ** 20
    return f"{m:.0f}" if m >= 100 else f"{m:.1f}"


def fh(x, d=2) -> str:
    return "—" if x is None else f"{float(x):.{d}f}"


def fs(x, d=3) -> str:
    return "—" if x is None else f"{float(x):.{d}f}"


def c(s) -> str:
    return f"`{s}`"


def md_table(header, rows, align) -> list[str]:
    check(len(header) == len(align), f"table header / align length {header}")
    cell = lambda v: str(v).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    out = ["| " + " | ".join(cell(h) for h in header) + " |", "|" + "|".join({"l": ":---", "r": "---:", "c": ":---:"}[a] for a in align) + "|"]
    for r in rows:
        check(len(r) == len(header), f"table row length {r}")
        out.append("| " + " | ".join(cell(v) for v in r) + " |")
    return out


# ------------------------------------------------------------------------------------------------ git
def git(*args, check_rc=True) -> str:
    p = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    if check_rc and p.returncode != 0:
        fail(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout


def git_blob(commit: str, path: str) -> bytes:
    p = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"{commit}:{path}"], capture_output=True)
    if p.returncode != 0:
        fail(f"git cat-file {commit}:{path}")
    return p.stdout


def hash_entries(entries: dict) -> str:
    """adapter.code_hash(): sha256 over '<sha256>  <path>\\n' lines, paths sorted, first 16 hex."""
    lines = "".join(f"{sha256_bytes(entries[p])}  {p}\n" for p in sorted(entries))
    return sha256_bytes(lines.encode())[:16]


def code_hash_at(commit: str, paths, exclude=()) -> tuple[str, int]:
    files = [f for f in git("ls-tree", "-r", "--name-only", commit, "--", *paths).splitlines() if f and f not in exclude]
    return hash_entries({f: git_blob(commit, f) for f in files}), len(files)


def code_hash_tree(paths, include=(), exclude=()) -> tuple[str, int]:
    files = ({f for f in git("ls-files", "--", *paths).splitlines() if f} | set(include)) - set(exclude)
    return hash_entries({f: (REPO / f).read_bytes() for f in files}), len(files)


# ------------------------------------------------------------------------------------------------ logs
def parse_header(line: str) -> dict:
    return {m.group(1): m.group(2) for m in re.finditer(r"(\w+)=(.*?)(?=\s\w+=|$)", line.strip())}


def parse_job_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    hdr = parse_header(lines[0]) if lines else {}
    parts = []
    for m in re.finditer(r"^\[t01\] part (\w+) on (\w+): (\w+) \((\d+) pass / (\d+) fail / (\d+) blocked, (\d+) s\) "
                         r"required failures: (\[.*?\]) blocked: (\[.*?\])$", text, re.M):
        parts.append({"part": m.group(1), "dev": m.group(2), "status": m.group(3), "n_pass": int(m.group(4)),
                      "n_fail": int(m.group(5)), "n_blocked": int(m.group(6)), "seconds": int(m.group(7)),
                      "required_failures": ast.literal_eval(m.group(8))})
    req_rows = {}
    for m in re.finditer(r"^\[(\w+)/(\w+)\] FAIL\s+(\S+) (\{.*\})$", text, re.M):
        req_rows[m.group(3)] = json.loads(m.group(4))
    t02 = [{"part": a, "status": b, "seconds": int(s)} for a, b, s in re.findall(r"^\[t02\] part (\w+): (\w+) \((\d+) s\)$", text, re.M)]
    end = re.search(r"^\[v10\] (.*)end=(\S+)$", text, re.M)
    tests = re.search(r"^(\d+) passed", text, re.M)
    merges = [json.loads(ln) for ln in lines if ln.startswith('{"TORCH_VALIDATION"')]
    return {"path": str(path), "sha256": sha256_bytes(path.read_bytes()), "bytes": path.stat().st_size, "header": hdr,
            "parts": parts, "required_rows": req_rows, "t02": t02,
            "end_utc": end.group(2) if end else None, "rcs": dict(re.findall(r"(\w+)=(\d+)", end.group(1))) if end else {},
            "tests_passed": int(tests.group(1)) if tests else None, "merge": merges[-1] if merges else None}


# ------------------------------------------------------------------------------------------------ collection
def collect(args) -> dict:
    D: dict = {"gen_utc": utc_now()}
    # --- T00
    D["preflight"] = load_json(T00 / "PREFLIGHT.json")
    D["preflight_log"] = (T00 / "PREFLIGHT_RUN.log").read_text()
    D["mbs"] = load_json(T00 / "MANIFEST_BUILD_SUMMARY.json")
    D["host_map"] = load_json(T00 / "host_map.json")
    D["exposure_notes"] = (T00 / "EXPOSURE_NOTES.md").read_text()
    sha_txt = {}
    for ln in (T00 / "MANIFEST_SHA256.txt").read_text().splitlines():
        if ln.strip():
            h, name = ln.split(None, 1)
            sha_txt[name.strip()] = h
    for name, h in sha_txt.items():
        check(sha256_file(T00 / name) == h, f"v10/T00/MANIFEST_SHA256.txt: {name}")
    D["manifest_sha256"] = sha_txt["V10_DATA_MANIFEST.csv"]
    check(D["preflight"]["data_inventory"]["sha256"] == D["manifest_sha256"], "PREFLIGHT.json manifest sha256")
    check(D["mbs"]["checks_passed"] == len(D["mbs"]["checks"]) and not D["mbs"]["checks_failed"], "manifest build checks")
    # --- package
    D["protocol"] = load_json(PKG / "configs" / "V10_PROTOCOL.json")
    D["design"] = (PKG / "V10_DESIGN_CN.md").read_text(encoding="utf-8")
    D["pkg_status"] = load_json(PKG / "PACKAGE_STATUS.json")
    pm = load_json(PKG / "MANIFEST_SHA256.json")
    D["pkg_manifest_n"] = len(pm)
    D["pkg_manifest_bad"] = [k for k, v in pm.items() if not (PKG / k).exists() or sha256_file(PKG / k) != v]
    D["pkg_manifest"] = pm
    bt = (PKG / "core" / "benchmark_torch.py").read_text()
    m = re.search(r"atol=([0-9.e+-]+) if dtype==torch\.float32 else ([0-9.e+-]+)", bt)
    check(m is not None, "package benchmark tolerance")
    D["bench_atol"] = {"float32": float(m.group(1)), "float64": float(m.group(2))}
    D["synthetic"] = synthetic_facts()
    m = re.search(r"preflight\.py sha256=([0-9a-f]{64})", D["preflight_log"])
    check(m is not None and m.group(1) == pm["scripts/preflight.py"] == sha256_file(PKG / "scripts" / "preflight.py"),
          "preflight.py sha256 in PREFLIGHT_RUN.log vs package manifest")
    check(sha256_file(REPO / "src" / "v10" / "mf_vcs.py") == pm["core/mf_vcs.py"], "src/v10/mf_vcs.py vs package core/mf_vcs.py")
    # --- T01
    tv, gg = load_json(T01 / "TORCH_VALIDATION.json"), load_json(T01 / "GEOMETRY_GRADIENT_CHECK.json")
    D["tv"], D["gg"] = tv, gg
    order = list(tv["provenance"]["parts"].keys())
    parts = {**tv["parts"], **gg["parts"]}
    check(set(order) == set(parts), "T01 part list vs provenance")
    D["t01_order"], D["t01_parts"] = order, parts
    cur = tv["provenance"]["current"]
    check(cur == gg["provenance"]["current"], "T01 provenance current")
    for k, p in parts.items():
        n = p["n_pass"] + p["n_fail"] + p["n_blocked"] + p["n_not_run"]
        check(n == p["n_checks"] == len(p["checks"]), f"{k}: check counts")
        check(p["status"] == "PASS" and not p["required_failures"] and not p["required_blocked"] and not p["required_not_run"], f"{k} status")
        check(sorted(p["diagnostic_failures"]) == sorted(x["name"] for x in p["checks"] if x["status"] != "PASS"), f"{k} diag list")
        check(all(x["status"] == "PASS" for x in p["checks"] if x.get("required")), f"{k}: required rows")
        check(sum(1 for x in p["checks"] if x.get("required")) == p["n_required"], f"{k}: n_required")
        for key in ("code_hash_src_scripts_v10", "host_code_hash", "mf_vcs_sha256"):
            check(p["provenance"][key] == cur[key], f"{k}: provenance {key}")
    for doc in (tv, gg):
        check(doc["overall_status"] == "PASS" and doc["host_adapter_verified"] is True and not doc["provenance_mismatches"]
              and not doc["pending_parts"] and not doc["failed_parts"] and not doc["blocked_parts"], "T01 merged status")
    D["t01_current"] = cur
    # --- T02
    D["t2"] = {d: load_json(T02 / f"T02_TIMING_{d}.json") for d in ("cpu", "cuda")}
    D["long"] = {d: load_csv(T02 / f"T02_LONG_{d}.csv") for d in ("cpu", "cuda")}
    check(D["t2"]["cuda"]["n_rows"] == len(D["long"]["cuda"]), "T02 cuda rows")
    for d in ("cpu", "cuda"):
        check(D["t2"][d]["environment"]["code_hash_src_scripts_v10"] == cur["code_hash_src_scripts_v10"], f"T02 {d} code hash")
    D["budget"] = load_json(T02 / "BUDGET.json")
    D["ledger"] = load_csv(T02 / "UNIT_LEDGER.csv")
    check_budget(D)
    check_ledger(D)
    # --- raw run files named by T02_TIMING_cuda.json
    run_dir = Path(D["t2"]["cuda"]["run_dir"])
    D["bench_raw"] = {}
    for dtype in ("float32", "float64"):
        h = D["t2"]["cuda"]["parts"]["head"][dtype]
        p = Path(h["report"])
        if p.exists() and sha256_file(p) == h["report_sha256"]:
            D["bench_raw"][dtype] = load_json(p)
    ps = run_dir / "train_steps_per_step.json"
    D["per_step"] = {"path": str(ps), "sha256": sha256_file(ps), "data": load_json(ps)} if ps.exists() else None
    # --- logs
    D["logs"] = {}
    for p in sorted(LOGS.glob("v10_*.out")):
        L = parse_job_log(p)
        D["logs"][L["header"].get("job")] = L
    m = re.search(r"SLURM_JOB_ID=(\d+)", D["preflight_log"])
    D["preflight_job"] = {"job": m.group(1) if m else None, "header": parse_header(D["preflight_log"].splitlines()[0]),
                          "end": (re.search(r"end=(\S+)", D["preflight_log"]) or [None, None])[1],
                          "exit": (re.search(r"exit=(\d+)", D["preflight_log"]) or [None, None])[1]}
    collect_git(D)
    # job header code / pin == git-object hashes at the job commit
    for job, L in D["logs"].items():
        h = L["header"]
        check(code_hash_at(h["commit"], CODE_PATHS)[0] == h["code"], f"job {job}: code hash at {h['commit'][:7]}")
        check(code_hash_at(h["commit"], PIN_PATHS)[0] == h["pin"] == h["expect"], f"job {job}: pin at {h['commit'][:7]}")
    D["queue"] = None if args.no_squeue else squeue()
    D["slurm"] = slurm_facts()
    D["adapter_methods"] = adapter_methods()
    D["mind_files"] = sorted(str(p.relative_to(REPO)) for p in (REPO / "src" / "v9" / "reg").glob("*.py")
                             if re.search(r"\bmind\b", p.read_text(errors="replace"), re.I))
    D["reads"] = phase1_reads(D)
    return D


def synthetic_facts() -> dict:
    src = (PKG / "synthetic" / "src" / "v10_synthetic.py").read_text()
    tree = ast.parse(src)
    fam = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "FAMILIES" for t in node.targets):
            fam = list(ast.literal_eval(node.value).keys())
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            kw = {k.arg: k.value for k in node.keywords}
            if "default" not in kw:
                continue
            name = ast.literal_eval(kw["dest"]) if "dest" in kw else node.args[0].value.lstrip("-").replace("-", "_")
            try:
                defaults[name] = ast.literal_eval(kw["default"])
            except ValueError:
                defaults[name] = ast.unparse(kw["default"])
    if defaults.get("families") == "list(FAMILIES)":
        defaults["families"] = fam
    ex = PKG / "synthetic" / "validation" / "example_data"
    cases = []
    for cse in load_json(ex / "manifest.json")["cases"]:
        t = load_json(ex / cse["truth_directory"] / "truth.json")
        cases.append({"case": cse["case_id"], "family": t["family"], "m": t["m"], "rho": t["rho"], "S": t["S"],
                      "S_closed_form": t["S_closed_form"]})
    grid = {(f, m, float(r)) for f in defaults["families"] for m in defaults["m"] for r in defaults["rho"]}
    check(grid == {(x["family"], x["m"], float(x["rho"])) for x in cases}, "synthetic default grid vs example cases")
    chk = load_json(PKG / "synthetic" / "validation" / "CHECK_RESULTS.json")
    return {"defaults": defaults, "cases": cases, "check_status": chk["status"], "check_n": chk["n_checks"],
            "generator_sha256": sha256_file(PKG / "synthetic" / "src" / "v10_synthetic.py")}


def check_budget(D):
    B, rows = D["budget"], D["long"]["cuda"]
    check(B["status"] == "MEASURED_PROJECTION", "BUDGET status")
    check(B["source"]["sha256"] == sha256_file(T02 / "T02_TIMING_cuda.json"), "BUDGET source sha256")
    c = B["protocol_counts"]
    n_bf = c["checkpoints_per_unit"] * c["bf_cal_cores"] * c["perturbations_per_image"] + c["bf_dev_cores"] * c["perturbations_per_image"]
    n_ep = c["checkpoints_per_unit"] * c["epfl_cal_windows"] * c["perturbations_per_image"] + c["epfl_dev_windows"] * c["perturbations_per_image"]
    check(n_bf == B["registrations_per_unit"]["BF_SHG"] and n_ep == B["registrations_per_unit"]["EPFL_RGB_NIR"], "registrations per unit")
    train = {r["config"]: num(r["median_ms"]) for r in rows if r["layer"] == "train_step"}
    regs = defaultdict(list)
    for r in rows:
        if r["layer"] == "registration" and r["phase"] == "registration" and r["status"] == "completed" and r["warmup_row"] == "False":
            regs[r["config"]].append(num(r["elapsed_seconds"]))
    for k, v in B["measured"]["train_step_ms_median"].items():
        check(close(v, train.get(k)), f"BUDGET train step {k}")
    for k, v in B["measured"]["s_per_registration_median"].items():
        check(close(v, statistics.median(regs[k])), f"BUDGET registration {k}")
    bf = ep = tune = 0.0
    measured = []
    for m, e in B["per_unit_projection"].items():
        if e["status"] != "measured":
            continue
        measured.append(m)
        tr = c["train_steps_per_unit"] * e["train_step_ms_median"] / 1000
        g_bf = (tr + n_bf * e["s_per_registration_median"]) / 3600
        g_ep = (tr + n_ep * e["s_per_registration_median"]) / 3600
        check(close(g_bf, e["gpu_h_bf_unit"]) and close(g_ep, e["gpu_h_epfl_unit_proxy"]), f"BUDGET per-unit {m}")
        bf += g_bf
        ep += g_ep
        tune += c["cal_candidates_per_family"] * (tr + c["checkpoints_per_unit"] * c["bf_cal_cores"] * c["perturbations_per_image"]
                                                  * e["s_per_registration_median"]) / 3600
    s = len(c["training_seeds"])
    g = B["grid_projection_measured_methods"]
    check(measured == g["methods"] and s == g["seeds"], "BUDGET measured methods")
    check(close(bf * s, g["BF_SHG_gpu_h"]) and close(ep * s, g["EPFL_gpu_h_proxy"]) and close(tune, g["CAL_tuning_gpu_h_BF"])
          and close(bf * s + ep * s + tune, g["total_gpu_h"]), "BUDGET grid totals")
    D["budget_counts"] = {"n_reg_bf": n_bf, "n_reg_ep": n_ep, "n_reg_cal_bf": c["checkpoints_per_unit"] * c["bf_cal_cores"] * c["perturbations_per_image"]}


def check_ledger(D):
    L = D["ledger"]
    units = load_csv(PKG / "configs" / "EXPERIMENT_UNITS.csv")
    check([u["unit_id"] for u in units] == [r["unit_id"] for r in L], "UNIT_LEDGER unit ids vs package EXPERIMENT_UNITS.csv")
    for r in L:
        if r["stage"] == "T02":
            check(r["run_status"].startswith("completed"), f"ledger {r['unit_id']}")
        else:
            check(r["run_status"].startswith("NOT_RUN"), f"ledger {r['unit_id']}")


def collect_git(D):
    g = {"branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(), "head": git("rev-parse", "HEAD").strip()}
    g["head_utc"] = to_utc(git("log", "-1", "--format=%cI", "HEAD").strip())
    excl = [f":(exclude){SELF_REL}", f":(exclude){REPORT_DEFAULT.relative_to(REPO).as_posix()}"]
    g["results_commit"] = git("log", "-1", "--format=%H", "--", *RESULT_PATHS, *excl).strip()
    g["package_commit"] = git("log", "--format=%H", "--diff-filter=A", "--", f"{PKG_REL}/V10_DESIGN_CN.md").strip().splitlines()[-1]
    g["v9_diff_since_package"] = git("diff", "--stat", g["package_commit"], "HEAD", "--", "src/v9", "scripts/v9", "slurm/v9").strip()
    st = [ln for ln in git("status", "--porcelain", "--untracked-files=all", "--", *STATUS_PATHS).splitlines() if ln.strip()]
    own = {SELF_REL, REPORT_DEFAULT.relative_to(REPO).as_posix()}
    g["status_own"] = [ln for ln in st if ln[3:] in own]
    g["status_other"] = [ln for ln in st if ln[3:] not in own and "__pycache__" not in ln]
    commits = []
    for ln in git("log", "--reverse", "--format=%H%x1f%cI%x1f%s", f"{g['package_commit']}^..HEAD").splitlines():
        h, iso, subj = ln.split("\x1f")
        files = [f for f in git("show", "--name-only", "--format=", h).splitlines() if f]
        tops = sorted({"/".join(f.split("/")[:2]) for f in files})
        ch, nf = code_hash_at(h, CODE_PATHS)
        commits.append({"sha": h, "utc": to_utc(iso), "subject": subj, "code_hash": ch if nf else None, "n_code_files": nf,
                        "pin": code_hash_at(h, PIN_PATHS)[0], "tops": tops, "n_files": len(files)})
    g["commits"] = commits
    g["msgs"] = {h["sha"]: git("log", "-1", "--format=%B", h["sha"]) for h in commits}
    g["tree_hash_excl"] = code_hash_tree(CODE_PATHS, exclude=[SELF_REL])[0]
    g["tree_hash_incl"] = code_hash_tree(CODE_PATHS, include=[SELF_REL])[0]
    g["tree_pin_incl"] = code_hash_tree(PIN_PATHS, include=[SELF_REL])[0]
    g["self_sha256"] = sha256_file(REPO / SELF_REL)
    g["self_lines"] = (REPO / SELF_REL).read_text().count("\n")
    g["self_tracked"] = bool(git("ls-files", "--", SELF_REL).strip())
    D["git"] = g


def squeue() -> dict:
    try:
        p = subprocess.run(["squeue", "-u", os.environ.get("USER", "yinwang"), "-h", "-o", "%i|%q|%T|%j"],
                           capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            return {"utc": utc_now(), "error": p.stderr.strip()[:200]}
        rows = [ln.split("|") for ln in p.stdout.splitlines() if ln.strip()]
        v10 = [r for r in rows if r[3].startswith("v10_")]
        n2 = [r for r in rows if r[3].startswith("v9n2_")]
        return {"utc": utc_now(), "v10": len(v10), "normal": sum(1 for r in rows if r[1] == "normal"),
                "v9n2_running": sum(1 for r in n2 if r[2] == "RUNNING"), "v9n2_pending": sum(1 for r in n2 if r[2] == "PENDING")}
    except Exception as e:  # noqa: BLE001 - the queue line is optional
        return {"utc": utc_now(), "error": repr(e)[:200]}


def slurm_facts() -> dict:
    out = {}
    sub = (REPO / "slurm" / "v10" / "submit.sh").read_text()
    out["v10_max"] = int(re.search(r'n_v10" -ge (\d+)', sub).group(1))
    out["normal_max"] = int(re.search(r'n_normal" -ge (\d+)', sub).group(1))
    for name in ("t01_cpu.sbatch", "t01_t02_gpu.sbatch"):
        txt = (REPO / "slurm" / "v10" / name).read_text()
        out[name] = dict(re.findall(r"^#SBATCH --([\w-]+)=(\S+)", txt, re.M))
    return out


def adapter_methods() -> dict:
    src = (REPO / "src" / "v10" / "adapter.py").read_text()
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", "") in ("MF_METHODS", "MLP_METHODS"):
                    out[t.id] = list(ast.literal_eval(node.value))
    m = re.search(r"if cfg\.method not in (\([^)]*\)) \+ MLP_METHODS", src)
    check(m is not None, "adapter Trainer method check not found")
    out["TRAINER"] = list(ast.literal_eval(m.group(1))) + out.get("MLP_METHODS", [])
    out["bf_only_gate"] = "Only BF_SHG rows of the V10 manifest" in src
    return out


def phase1_reads(D) -> dict:
    by_role = defaultdict(set)
    for p in D["t01_parts"].values():
        for r in p.get("data_reads", []):
            by_role[r["role"]].add(r.get("core"))
    for d in ("cpu", "cuda"):
        for r in D["t2"][d].get("data_reads", []):
            by_role[r["role"]].add(r.get("core"))
    return {k: len(v) for k, v in by_role.items()}


# ------------------------------------------------------------------------------------------------ lookups
def long_rows(D, dev, **kv):
    return [r for r in D["long"][dev] if all(r.get(k) == v for k, v in kv.items())]


def head_cell(D, dev, dtype, n, k, method):
    layer = "head_extra" if method == "blocked_dense_graph_retained" else "head"
    rr = long_rows(D, dev, layer=layer, dtype=dtype, n_positions=str(n), simplex_width=str(k), method=method)
    check(len(rr) <= 1, f"duplicate head cell {dev} {dtype} {n} {k} {method}")
    return rr[0] if rr else None


def check_row(D, part, name):
    for x in D["t01_parts"][part]["checks"]:
        if x["name"] == name:
            return x
    return None


def cpu_outliers(D) -> set:
    """CPU head rows whose median exceeds 10x the median of the same method / dtype at a cell with N and K both >= (not equal)."""
    rows = [r for r in D["long"]["cpu"] if r["layer"] == "head"]
    flagged = set()
    for r in rows:
        n, k, m = int(r["n_positions"]), int(r["simplex_width"]), num(r["median_ms"])
        for r2 in rows:
            if r2 is r or r2["dtype"] != r["dtype"] or r2["method"] != r["method"]:
                continue
            n2, k2 = int(r2["n_positions"]), int(r2["simplex_width"])
            if n2 >= n and k2 >= k and (n2, k2) != (n, k) and m > OUTLIER_FACTOR * num(r2["median_ms"]):
                flagged.add((r["dtype"], n, k, r["method"]))
                break
    return flagged


def cuda_outliers(D) -> int:
    rows = [r for r in D["long"]["cuda"] if r["layer"] in ("head", "head_extra")]
    n_flag = 0
    for r in rows:
        n, k, m = int(r["n_positions"]), int(r["simplex_width"]), num(r["median_ms"])
        if any(r2 is not r and r2["dtype"] == r["dtype"] and r2["method"] == r["method"] and int(r2["n_positions"]) >= n
               and int(r2["simplex_width"]) >= k and (int(r2["n_positions"]), int(r2["simplex_width"])) != (n, k)
               and m > OUTLIER_FACTOR * num(r2["median_ms"]) for r2 in rows):
            n_flag += 1
    return n_flag


def per_step_stats(D) -> dict:
    """M1 per-step records of the T02 GPU train part (raw file): degenerate start of the tol-1e-4 record row, cap-bound gaps."""
    if D["per_step"] is None:
        return {}
    data, proto = D["per_step"]["data"], D["t2"]["cuda"]["protocol"]
    w0, nt = proto["train_warmup"], proto["train_timed"]
    out = {}
    for cfg, recs in data.items():
        if not cfg.startswith("MF_M1"):
            continue
        timed = recs[w0:w0 + nt]
        lead = 0
        for r in recs:
            if r["m1_iterations"] == 0 and r["w_absmax"] == 0.0:
                lead += 1
            else:
                break
        out[cfg] = {"n_recs": len(recs), "lead_degenerate": lead,
                    "lead_delta": (min(r["m1_gap_initial"] for r in recs[:lead]), max(r["m1_gap_initial"] for r in recs[:lead])) if lead else None,
                    "first_active": (lead, recs[lead]["m1_gap_initial"]) if lead < len(recs) else None,
                    "first_delta": recs[0]["m1_gap_initial"], "first_w": recs[0]["w_absmax"],
                    "timed_w0": sum(1 for r in timed if r["w_absmax"] == 0.0), "timed_conv": sum(1 for r in timed if r["m1_converged"]),
                    "timed_min_gap": min(r["m1_gap"] for r in timed), "tol": recs[0]["m1_tolerance"],
                    "max_it": recs[0]["m1_max_iterations"], "n_timed": len(timed)}
    return out


# ------------------------------------------------------------------------------------------------ rendering
def short(s: str, n: int = 78) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def part_label(k: str) -> tuple[str, str]:
    base, dev = k.rsplit("_", 1)
    return base, dev


def commit_with(D, phrase: str) -> str:
    for x in D["git"]["commits"]:
        if phrase in " ".join(D["git"]["msgs"][x["sha"]].split()):
            return x["sha"][:7]
    fail(f"no V10 commit message contains {phrase!r}")


def train_ks(D) -> tuple[str, list[str]]:
    """main K of the BUDGET unit mapping and the other K values timed in T02 train steps"""
    main = re.search(r"_K(\d+)", D["budget"]["per_unit_projection"]["MF_VCS_M0"]["train_config"]).group(1)
    ks = sorted({r["simplex_width"] for r in D["long"]["cuda"] if r["layer"] == "train_step" and r["simplex_width"]}, key=int)
    return main, [k for k in ks if k != main]


def quote(D, fragment: str) -> str:
    """a verbatim fragment of the package design (V10_DESIGN_CN.md), checked to exist"""
    check(fragment in D["design"], f"design quote not found: {fragment!r}")
    return f"“{fragment}”"


def result_jobs(D) -> list[str]:
    res = D["t01_current"]["code_hash_src_scripts_v10"]
    return [j for j in sorted(D["logs"]) if D["logs"][j]["header"]["code"] == res]


def jobs_without_log(D) -> list[dict]:
    known = set(D["logs"]) | {D["preflight_job"]["job"]}
    out, seen = [], set()
    for x in D["git"]["commits"]:
        flat = " ".join(D["git"]["msgs"][x["sha"]].split())
        for m in re.finditer(r"[^.;]*\bjob (\d{7})\b[^.;]*[.;]", flat):
            j = m.group(1)
            if j not in known and j not in seen:
                seen.add(j)
                out.append({"job": j, "commit": x["sha"], "sentence": m.group(0).strip()})
    return out


def summarize_log(L) -> str:
    bad = [f"{p['part']}_{p['dev']}" for p in L["parts"] if p["status"] != "PASS"]
    s = f"T01 {len(L['parts'])} 个 part：" + (f"{'、'.join(bad)} FAIL，其余 PASS" if bad else "全部 PASS")
    if L["t02"]:
        s += "；T02 " + "、".join(f"{t['part']} {t['status']}" for t in L["t02"])
    if L["merge"]:
        s += (f"；作业内合并 TORCH_VALIDATION {L['merge']['TORCH_VALIDATION']}、GEOMETRY {L['merge']['GEOMETRY_GRADIENT_CHECK']}、"
              f"host_adapter_verified {str(L['merge']['host_adapter_verified']).lower()}"
              + (f"（{len(L['merge']['pending'])} 个 CUDA part 当时待跑）" if L["merge"].get("pending") else ""))
    return s


def sec_header(D) -> list[str]:
    g = D["git"]
    return [
        "# V10 MF-VCS 第一阶段（T00–T02）阶段报告",
        "",
        f"生成：`{SELF_REL}`，{D['gen_utc']}；分支 `{g['branch']}`，生成时 HEAD `{g['head'][:7]}`，结果提交 `{g['results_commit'][:7]}`。"
        "全部数值、表格与 git 事实在生成时读自 §11 所列文件、包文件、git 对象与作业日志（另有一次只读 squeue）；手写的只有文字、决定选项与建议，"
        "以及明确标为“主会话陈述”的条目。",
        "",
        "**范围**：第一阶段只做了预检、数学/宿主接入验证与计时。**没有测量任何配准精度、S 估计质量或训练后任务表现**；"
        "本报告不含性能、加速或新颖性结论，计时表只是成本输入。",
        "",
        "**待 owner 决定 5 项（§6；每项列选项、证据与标为“建议”的默认值，本报告不作决定）**：① MF-M1 头停止规则；② TF32 策略；"
        "③ InfoNCE 配准能量；④ T04/T05 前尚缺的基线；⑤ 数据角色（TEST、CONFIRM、EPFL 激活）。决定 1–4 之前不提交 T04/T05 作业。",
        "",
    ]


def sec1(D) -> list[str]:
    g, cur = D["git"], D["t01_current"]
    res = cur["code_hash_src_scripts_v10"]
    same = [x["sha"][:7] for x in g["commits"] if x["code_hash"] == res]
    rj = result_jobs(D)
    pins = sorted({D["logs"][j]["header"]["pin"] for j in rj})
    pm_bad = D["pkg_manifest_bad"]
    tree = ("测量代码、作业脚本、测试、v10/T00–T02 与 V9 路径无未提交改动" if not g["status_other"]
            else "未提交改动：" + "；".join(g["status_other"]))
    own = ("、".join(c(ln[3:]) + ("（未跟踪）" if ln.startswith("??") else "（已改动）") for ln in g["status_own"])
           + "，二者在此 HEAD 之后的同一提交中加入") if g["status_own"] else "均已提交"
    q = D["queue"]
    qtxt = ("未查询（--no-squeue）" if q is None else f"squeue 失败：{q['error']}" if "error" in q
            else f"生成时（{q['utc']}）队列中 V10 作业 {q['v10']} 个")
    backend = check_row(D, "verify_torch_cpu", "backend_copy_sha256_equals_package")["value"]
    rows = [
        ("分支 / 生成时 HEAD", f"{c(g['branch'])} / {c(g['head'])}（{g['head_utc']}）；{tree}；本生成器与本报告：{own}"),
        ("结果提交", f"{c(g['results_commit'][:7])}：最后一个改动 v10/T00–T02、src/v10、scripts/v10、slurm/v10 或 tests/v10 的提交（不计本生成器与本报告）"),
        ("测量代码身份", f"code hash {c(res)}（{' + '.join(CODE_PATHS)} 已跟踪文件，定义同 src/v10/adapter.py code_hash）："
                    f"由 git 对象在 {'、'.join(same)} 重算均得此值，等于结果作业 {'、'.join(rj)} 作业头的 code= 与 T01/T02 JSON 的 provenance；"
                    f"作业 pin（另含 slurm/v10）{'、'.join(c(p) for p in pins)}"),
        ("宿主与后端", f"宿主代码 hash {c(cur['host_code_hash'])}（V9-G1 g1.py + r6_reg_pilot.py + r6_reg_p2.py，只读导入）；"
                  f"自包提交 {c(g['package_commit'][:7])} 起 src/v9、scripts/v9、slurm/v9 的 git diff "
                  f"{'为空' if not g['v9_diff_since_package'] else '非空：' + g['v9_diff_since_package']}；"
                  f"后端 src/v10/mf_vcs.py sha256 {c(backend[:16])}… = 包 core/mf_vcs.py"),
        ("包", f"{c(PKG_REL)}（{c(g['package_commit'][:7])} 原样提交）；包 MANIFEST_SHA256.json 所列 {D['pkg_manifest_n']} 个文件"
              + ("全部一致" if not pm_bad else f"中 {len(pm_bad)} 个不一致：{', '.join(pm_bad)}")),
        ("本报告提交的影响", f"生成器位于 scripts/v10，code hash 的定义覆盖该目录：提交后树的 code hash 变为 {c(g['tree_hash_incl'])}、"
                     f"pin 变为 {c(g['tree_pin_incl'])}；其余测量代码不变（去掉生成器重算 = {c(g['tree_hash_excl'])}"
                     f"{'，等于结果' if g['tree_hash_excl'] == res else '，≠ 结果：测量代码已改'}）。{qtxt}，没有按旧 pin 排队的作业。"
                     "提交后若在新 HEAD 重跑 `t01_validate.py merge`，会把这些 part 记为 provenance 不一致；核对时应以去掉生成器的 hash 为准"),
    ]
    L = ["## 1 当前真实提交", ""] + md_table(["项", "值"], rows, "ll") + [""]
    crow = []
    for x in g["commits"]:
        crow.append((c(x["sha"][:7]), x["utc"], c(x["code_hash"]) if x["code_hash"] else "—（无代码文件）", ", ".join(x["tops"]),
                     short(x["subject"])))
    L += ["V10 提交（包提交起，git log；code hash 由 git 对象重算）：", ""]
    L += md_table(["提交", "UTC", "code hash", "改动路径", "说明（提交标题）"], crow, "lllll") + [""]
    # jobs
    jrows = []
    pj = D["preflight_job"]
    h = pj["header"]
    others = D["host_map"].get("v9_jobs_snapshot", {}).get("other_normal_qos_jobs_not_v9", {})
    label = (f"T00 预检（在已存在的 CPU 作业内运行，T00 审计记为 {others[pj['job']]}；非 V10 提交）" if pj["job"] in others
             else "T00 预检（非 V10 提交）")
    jrows.append((pj["job"], label, c(h.get("commit", "—")), h.get("host", "—"), f"{h.get('start')} – {pj['end']}",
                  f"exit {pj['exit']}；{D['preflight']['status']}", "v10/T00/PREFLIGHT_RUN.log"))
    res_set = set(rj)
    nolog = {x["job"]: x for x in jobs_without_log(D)}
    for job in sorted(set(D["logs"]) | set(nolog)):
        if job in nolog:
            x = nolog[job]
            jrows.append((job, "T01 CUDA + T02（GPU）", "pin 为该提交之前的代码", "—", "排队中取消，未启动",
                          f"无日志；见 {c(x['commit'][:7])} 提交说明：“{x['sentence']}”", "—"))
            continue
        Lg = D["logs"][job]
        hh = Lg["header"]
        kind = {"slurm/v10/t01_cpu.sbatch": "T01 CPU + T02 CPU 数值头", "slurm/v10/t01_t02_gpu.sbatch": "T01 CUDA + T02 全部 + BUDGET"}.get(hh["script"], hh["script"])
        node = hh["node"] + "，" + hh["partition"] + ("，" + hh["gpu"] if hh["gpu"].lower() not in ("none", "") else "")
        tag = "；**当前结果**" if job in res_set else "；已作废（§7）"
        jrows.append((job, kind, f"{c(hh['commit'][:7])} / {c(hh['code'])}", node, f"{hh['start']} – {Lg['end_utc']}",
                      summarize_log(Lg) + tag, Path(Lg["path"]).name))
    L += ["作业（作业头、结束行与 part 行解析自 slurm 日志；T00 行解析自 PREFLIGHT_RUN.log）：", ""]
    L += md_table(["作业", "内容", "提交 / code hash", "节点", "开始 – 结束（UTC）", "结果", "日志"], jrows, "lllllll") + [""]
    return L


COVERAGE = (
    ("verify_torch", None, "包 core/verify_torch.py（未改动）；后端与脚本 sha256 = 包清单"),
    ("core_tol", None, "设计容差核心数学：V/U/独立池矩 vs dense 的值与梯度、头部值/梯度/HVP、盒 QP 对独立 oracle、NaN/inf/不可行 W/非 simplex 拒绝"),
    ("adapter", r"^(host_code_hash_pinned|v10_manifest_sha256_pinned|fit_cores_|v9_writer_tripwires|held_records|guard_|host_batch_rng)",
     "数据门、宿主/manifest 固定、V9 写入 tripwire、宿主 batch RNG"),
    ("adapter", r"^(real_descriptors|real_features|scene_s_holds|real_moments_vs_dense)",
     "真实 FIT batch：矩 vs dense 的值与 A/B/W、encoder + feature map 梯度（含分块 checkpoint）"),
    ("adapter", r"^real_all_\d+_permutations", "全部排列的均值 = V（值与梯度）"),
    ("adapter", r"^(unequal_n|V_scene_n1|partition_scenes)", "场景权重、不等 n、n = 1 的 V"),
    ("adapter", r"^(U_mode|U_differs|independent_pool)", "Q 模式：U 与独立池"),
    ("adapter", r"^(M1_|MF_M1_)", "M1 头：严格解、盒约束与 detach、外层梯度 = 固定 W 偏导、训练模式记录 gap、上限强制步"),
    ("adapter", r"^MLP_", "宿主 MLP critic：perm1 = 宿主公式、dense = 排列均值、分块 checkpoint、perm8"),
    ("adapter", r"^(registration_|host_chain_dv)", "配准侧 warp 链：宿主位置与排列、批量能量 = 包矩 / perm1、几何梯度 moments vs dense"),
    ("adapter", r"^(fixed_W_|test_time_)", "测试时固定 W：状态 hash 不变、eval 与冻结、几何梯度有限非零、拒绝可训练模型"),
    ("adapter", r"^checkpoint_roundtrip", "checkpoint 往返：(encoder, feature maps, W) 状态 hash、能量与几何梯度"),
    ("adapter", r"^(MF_perm1_|MF_M0_|same_first_batch)", "一步训练：MF_perm1 与 M0 同批同场景、W 投影可行、encoder 与 feature map 实际更新"),
    ("adapter", r"^weight_decay", "weight decay 分组（W 为 0）"),
    ("host_replay", None, "宿主 replay：未改动的 G.cmd_train vs V10 MLP_perm1 trainer"),
    ("geometry", r"^(kink_margin|zero_kink)", "几何：采样点与插值折点的距离、零跨越"),
    ("geometry", r"^(dense_vs_moments|geometry_grad_nonzero)", "几何：dense vs moments 值与几何梯度、P/Q 各项梯度非零"),
    ("geometry", r"^fd_vs_autograd", "几何：fp64 中心差分 vs autograd（P/Q 各项；moments / dense / perm1；步长 {fd_steps}）"),
    ("geometry", r"^fp32_vs_fp64", "几何：fp32 vs fp64（默认 TF32 卷积与 IEEE 卷积，诊断）"),
)


def coverage_rows(D) -> list:
    cnt = defaultdict(lambda: Counter())
    unclassified = []
    for k, p in D["t01_parts"].items():
        base, dev = part_label(k)
        for x in p["checks"]:
            for i, (b, rx, _) in enumerate(COVERAGE):
                if b == base and (rx is None or re.match(rx, x["name"])):
                    kind = "r" if x.get("required") else "d"
                    cnt[i][(dev, kind, "n")] += 1
                    cnt[i][(dev, kind, "p")] += x["status"] == "PASS"
                    break
            else:
                unclassified.append(x["name"])
    check(not unclassified, f"T01 checks without coverage group: {unclassified[:5]}")
    rows = []
    for i, (b, rx, label) in enumerate(COVERAGE):
        C = cnt[i]
        cellf = lambda dev, kind: (f"{C[(dev, kind, 'p')]}/{C[(dev, kind, 'n')]}" if C[(dev, kind, "n")] else "—")  # noqa: E731
        lab = label.format(fd_steps="、".join(fe(h, 1) for h in D["gg"]["method"]["fd_steps"])) if "{" in label else label
        rows.append((b, lab, cellf("cpu", "r"), cellf("cpu", "d"), cellf("cuda", "r"), cellf("cuda", "d")))
    return rows


def sec2(D) -> list[str]:
    P, order = D["t01_parts"], D["t01_order"]
    pf, mbs, hm = D["preflight"], D["mbs"], D["host_map"]
    rows = [("T00 预检（包 scripts/preflight.py，未改动）", "PASS" if not pf["errors"] else "FAIL",
             f"状态 {pf['status']}（只表示清单已写出）；errors {len(pf['errors'])}，warnings {len(pf['data_inventory']['warnings'])}",
             "v10/T00/PREFLIGHT.json、PREFLIGHT_RUN.log"),
            ("T00 数据 manifest", "PASS" if not mbs["checks_failed"] else "FAIL",
             f"构建检查 {mbs['checks_passed']}/{len(mbs['checks'])}；{mbs['manifest']['rows']} 行，sha256 {D['manifest_sha256'][:16]}… "
             "= MANIFEST_SHA256.txt = 预检记录", "v10/T00/V10_DATA_MANIFEST.csv、MANIFEST_BUILD_SUMMARY.json、EXPOSURE_NOTES.md"),
            ("T00 宿主源码映射", "PASS（已写出）",
             f"映射状态 {hm['status']['T00_host_source_map']}；与包假设的差异 {len(hm['target_differences_vs_package'])} 项"
             f"（{hm['target_differences_vs_package'][0]['id']}–{hm['target_differences_vs_package'][-1]['id']}）登记未改；数值接入见 T01",
             "v10/T00/HOST_ADAPTER_AUDIT.md、host_map.json")]
    for k in order:
        p = P[k]
        base, dev = part_label(k)
        diag = p["n_checks"] - p["n_required"]
        rows.append((f"T01 {base}（{dev}）", p["status"],
                     f"{p['n_checks']} 项：必需 {p['n_required']} 全 PASS；诊断 {diag}" + (f"，其中 FAIL {p['n_fail']}" if p["n_fail"] else "")
                     + (f"；BLOCKED {p['n_blocked']}" if p["n_blocked"] else ""),
                     f"作业 {p['provenance']['slurm_job_id']}，{p['seconds']:.0f} s"))
    tv = D["tv"]
    rows.append(("T01 合并", tv["overall_status"],
                 f"TORCH_VALIDATION {tv['overall_status']}、GEOMETRY_GRADIENT_CHECK {D['gg']['overall_status']}、host_adapter_verified "
                 f"{str(tv['host_adapter_verified']).lower()}；provenance 不一致 {len(tv['provenance_mismatches'])}、pending {len(tv['pending_parts'])}",
                 "v10/T01/TORCH_VALIDATION.json、GEOMETRY_GRADIENT_CHECK.json"))
    for d in ("cpu", "cuda"):
        t = D["t2"][d]
        for part, v in t["parts"].items():
            if v["status"] == "NOT_RUN":
                det = "按设计只在 GPU 作业中运行"
            elif part == "head":
                nmeth = len({r["method"] for r in D["long"][d] if r["layer"] == "head"})
                det = f"fp32 + fp64，各 {len(t['protocol']['grid'])} 格 × {nmeth} 法；OOM {v['float32']['oom_count'] + v['float64']['oom_count']}"
            elif part == "head_extra":
                det = f"分块保留图 + 排列条件方差；包抽样重放核对 {v['replay_verified_cells']}/{len(t['protocol']['grid'])} 格，不一致 {len(v['replay_mismatches'])}"
            elif part == "train":
                det = f"{len(v['configs'])} 个配置，encoder 在内；完成 {sum(1 for x in v['configs'].values() if x['status'].startswith('completed'))}"
            else:
                det = f"{len(v['energies'])} 个能量，完成 {sum(1 for x in v['energies'].values() if x['status'] == 'completed')}"
            rows.append((f"T02 {part}（{d}）", v["status"], det, f"v10/T02/T02_TIMING_{d}.json"))
    B = D["budget"]
    part_meas = [m for m, e in B["per_unit_projection"].items() if e["status"] == "partially_measured"]
    rows.append(("BUDGET", B["status"], f"已测 {len(B['grid_projection_measured_methods']['methods'])} 族："
                 f"{'、'.join(B['grid_projection_measured_methods']['methods'])}；部分测得：{'、'.join(part_meas) or '无'}（仅训练步）",
                 "v10/T02/BUDGET.json"))
    led = D["ledger"]
    n_t02 = sum(1 for r in led if r["stage"] == "T02")
    rows.append(("UNIT_LEDGER（= 包 EXPERIMENT_UNITS.csv 的单元）", "PASS（T02 格）/ NOT_RUN（其余）",
                 f"{len(led)} 个单元：T02 计算格 {n_t02} 个 completed（fp32 + fp64 参考）；T04/T05 {len(led) - n_t02} 个 NOT_RUN",
                 "v10/T02/UNIT_LEDGER.csv"))
    tests = "；".join(f"{j} {L['tests_passed']} passed" for j, L in sorted(D["logs"].items()) if L["tests_passed"] is not None)
    rows.append(("单元测试 tests/v10（pytest，作业内记录、不门控）", "PASS", tests, "作业日志"))
    lv = D["pkg_status"]["local_validation"]
    rows.append(("包作者本地检查", "NOT_RUN（服务器未重跑）",
                 f"包记录：NumPy 数学 {lv['independent_numpy_math']}、synthetic {lv['synthetic_checks']}、语法与 manifest "
                 f"{lv['package_syntax_and_manifest_checks']} 项通过", f"{PKG_REL}/PACKAGE_STATUS.json"))
    rows.append(("T03 受控真值 / T04 完整训练 / T05 完整配准", "NOT_RUN", "第一阶段不启动（§5 为提议）", "v10/T02/UNIT_LEDGER.csv"))
    L = ["## 2 已完成代码/验证", ""] + md_table(["项", "状态", "计数 / 要点", "证据"], rows, "llll") + [""]
    req = sum(p["n_required"] for p in P.values())
    tot = sum(p["n_checks"] for p in P.values())
    dfail = sum(p["n_fail"] for p in P.values())
    blk = sum(p["n_blocked"] for p in P.values())
    nr = sum(p["n_not_run"] for p in P.values())
    L += [f"T01 合计 {tot} 项（{len(P)} 个 part）：**必需 {req} 项全部 PASS**（BLOCKED {blk}、NOT_RUN {nr}）；诊断 {tot - req} 项（不门控），"
          f"其中 {dfail} 项 FAIL，按类别如下。", ""]
    L += md_table(["类别", "part", "FAIL 行", "关键数值", "说明"], diag_rows(D), "lllll") + [""]
    L += ["T01 覆盖（必需 / 诊断，PASS/总数；按检查名分组，每项检查只计一次）：", ""]
    L += md_table(["part", "内容", "CPU 必需", "CPU 诊断", "CUDA 必需", "CUDA 诊断"], coverage_rows(D), "llrrrr") + [""]
    roles = {"src/v10/mf_vcs.py": "包后端原样（sha256 = 包）", "src/v10/adapter.py": "真实宿主 adapter（V9-G1 只读导入）",
             "src/v10/__init__.py": "包初始化", "scripts/v10/t00_build_manifest.py": "T00 manifest 构建（只读元数据）",
             "scripts/v10/t01_validate.py": "T01 验证与合并", "scripts/v10/t01_host_replay.py": "T01 宿主 replay",
             "scripts/v10/t02_timing.py": "T02 计时、BUDGET、UNIT_LEDGER", SELF_REL: "本报告生成器（不参与测量）"}
    files = sorted({f for f in git("ls-files", "--", "src/v10", "scripts/v10", "slurm/v10", "tests/v10").splitlines() if f} | {SELF_REL})
    crow = []
    for f in files:
        b = (REPO / f).read_bytes()
        crow.append((c(f), b.count(b"\n"), c(sha256_bytes(b)[:16]), roles.get(f, "作业脚本" if f.startswith("slurm/") else
                                                                         "单元测试" if f.startswith("tests/") else "")))
    L += ["代码清单（当前树）：", ""] + md_table(["文件", "行", "sha256[:16]", "作用"], crow, "lrll") + [""]
    return L


def diag_rows(D) -> list:
    P = D["t01_parts"]
    classes = defaultdict(list)
    for k, p in P.items():
        for x in p["checks"]:
            if x["status"] == "PASS":
                continue
            n = x["name"]
            cls = ("pkg_atol" if n == "package_errors_within_design_atol" else "detect" if "_detectable_" in n
                   else "mlp" if n.startswith("MLP_dense") else "m1" if n.startswith("MF_M1_protocol_tolerance") else "other")
            classes[cls].append((k, x))
    check(not classes.get("other"), f"unclassified diagnostic failures: {[x['name'] for _, x in classes.get('other', [])]}")
    out = []
    if classes.get("pkg_atol"):
        it = classes["pkg_atol"]
        x0 = it[0][1]
        above = [r for _, x in it for r in x["rows_above"]]
        check(all(r["name"].startswith("solver_gap") for r in above), "package rows above design atol are not all solver-gap rows")
        out.append(("包脚本误差 vs 设计 atol", "、".join(k for k, _ in it), len(it),
                    f"每次 {x0['n_above_design_atol']}/{x0['n_rows']} 行高于设计 atol {fe(above[0]['design_atol'], 0)}；"
                    f"最大 {fe(max(r['error'] for r in above))}，包自身限 {fe(above[0]['package_limit'], 0)}",
                    f"包脚本不记录参考量级；超出的行都是 solver gap 行（在包自身 {fe(above[0]['package_limit'], 0)} 限内）"
                    + ("；core_tol 以参考量级重跑这些恒等式，全部 PASS" if "core_tol re-runs" in x0.get("note", "")
                       and all(P[k]["status"] == "PASS" for k in P if k.startswith("core_tol")) else "")))
    if classes.get("detect"):
        it = classes["detect"]
        eff = [x["effect_over_tol"] for _, x in it]
        mult = re.search(r"exceed (\d+) x", it[0][1]["note"])
        jn = re.search(r"\(J ~ ([0-9e.+-]+)\)", it[0][1]["note"])
        bn = re.search(r"\(bound >= ([0-9e.+-]+)\)", it[0][1]["note"])
        gate = [x for k in P for x in P[k]["checks"] if "_detectable_" in x["name"] and "sharpened_x10_float64" in x["name"]]
        check(gate and all(x.get("required") and x["status"] == "PASS" for x in gate), "detectability gate rows")
        per = Counter(k for k, _ in it)
        out.append(("可检测性（错误规则的效应是否超过等式界）", "、".join(f"{k} {v}" for k, v in per.items()), len(it),
                    f"effect/界 = {fg(min(eff))}–{fg(max(eff))}（门槛 {mult.group(1) if mult else '?'}）",
                    f"host_init（J 约 {jn.group(1) if jn else '?'}）与 fp32（界 ≥ {bn.group(1) if bn else '?'}）下效应按构造不足；"
                    f"这类检查只在 fp64 sharpened_x10 状态门控，那 {len(gate)} 行均为必需且 PASS"))
    if classes.get("mlp"):
        it = classes["mlp"]
        parts = sorted({k for k, _ in it})
        vals = "；".join(f"{x['name'].replace('MLP_dense_', '')} {fe(x['max_abs_err'])}" for _, x in it)
        other_dev = [k for k in P if k.startswith("adapter_") and k not in parts]
        cuda_vals = []
        for k in other_dev:
            rr = [check_row(D, k, x["name"]) for _, x in it]
            rr = [r for r in rr if r]
            if rr:
                cuda_vals.append(f"{k} 同名 {len(rr)} 行 " + "、".join(r["status"] for r in rr) + "（" + "、".join(fe(r["max_abs_err"]) for r in rr) + "）")
        gate = "MLP_dense_blocked_checkpointed_float32_grads_vs_float64_value_same_inputs"
        gv = "；".join(f"{k} {check_row(D, k, gate)['status']} {fe(check_row(D, k, gate)['max_abs_err'])}"
                       for k in P if k.startswith("adapter_") and check_row(D, k, gate))
        npairs = next((x.get("n_pairs") for _, x in it if x.get("n_pairs")), None)
        out.append(("fp32 全图 dense-MLP 累加误差", "、".join(parts), len(it), vals,
                    f"两侧都是 fp32，全图对 {npairs} 对求和；{'；'.join(cuda_vals)}；门控行（分块 fp32 vs fp64）{gv}（§8 主会话决定）"))
    if classes.get("m1"):
        it = classes["m1"]
        out.append(("M1 协议容差从 W = 0 起步", "、".join(k for k, _ in it), len(it),
                    "‖Δ‖₁ = " + " / ".join(fe(x["m1_gap_initial_equals_Delta_l1"], 4) for _, x in it)
                    + f" < {fe(it[0][1]['m1_tolerance'], 0)}；迭代 {it[0][1]['m1_iterations']}，|W|max {fg(it[0][1]['w_absmax'])}，"
                    f"encoder 梯度 {fg(it[0][1]['encoder_grad_norm'])}", "协议设置的诊断，见 §6 决定 1"))
    return out


def sec3(D) -> list[str]:
    B, proto = D["budget"], D["protocol"]
    nm = B["not_measured_in_T02"]
    am = D["adapter_methods"]
    L = ["## 3 仍缺的输入", "",
         "1. **owner 决定 1–5**（§6）。决定 1、2 影响所有已测族的单元；决定 3、4 决定哪些基线能进入同一比较；决定 5 决定 TEST 与 EPFL 的使用。",
         f"2. **尚未实现或未接入 V10 的方法**（BUDGET `not_measured_in_T02`）：{'、'.join(nm)}（逐项现状见 §6.4）。"
         f"V10 trainer 只接受 {', '.join(am.get('TRAINER', []))}（src/v10/adapter.py），InfoNCE、CoMIR、核方法都未接入。",
         f"3. **EPFL**：协议激活规则为 {c(proto['registration']['dataset_activation'])}；V10 adapter 的数据门只开放 BF_SHG 行，"
         "EPFL 宿主路径（V9-G2 / REG-P3）尚未接入或审计；BUDGET 中的 EPFL 成本是代理值。",
         "4. **T03 代码**：包只提供离散真值生成器与检查脚本（synthetic/README.md：不含神经训练器）；V10 的 T03 训练/CAL/评估脚本尚未编写。",
         f"5. **冻结的 V10 T04/T05 协议**：包协议状态为 {c(proto['status'])}，`execution.full_training_budget_seconds` = "
         f"{json.dumps(proto['execution']['full_training_budget_seconds'])}，`independent_confirmation_source` = "
         f"{json.dumps(proto['execution']['independent_confirmation_source'])}；"
         "BUDGET 已给出实测投影，但单元的冻结协议（种子、CAL 规则、停止规则、TF32、能量定义）要在决定 1–5 后写定。",
         f"6. **CONFIRM 数据**：T00 结论为 {c(D['mbs']['confirm_assessment']['result'])}（可登记组 {D['mbs']['confirm_assessment']['eligible_groups']}）。",
         ""]
    return L


def sec4(D) -> list[str]:
    t2c, t2p = D["t2"]["cuda"], D["t2"]["cpu"]
    env, envc = t2c["environment"], t2p["environment"]
    proto = t2c["protocol"]
    grid = proto["grid"]
    L = ["## 4 实际测量表", ""]
    # 4.1 time
    hx = t2c["parts"]["head_extra"]
    L += ["### 4.1 数值头时间（L40S，fp32，中位 ms）", "",
          f"设备 {env['gpu_name']}（{env['host']}，作业 {env['slurm_job_id']}），torch {env['torch']}；包 `core/benchmark_torch.py --full`"
          f"（seed {t2c['seed']}）：每格同一组随机 simplex 特征 A/B 与 W、单场景 V 目标；warmup {proto['head_warmup']}、重复 {proto['head_repeats']}，"
          "计时 = 前向 + autograd.grad(A, B, W)，每次调用前后 CUDA 同步；**不含 encoder**。"
          f"分块 dense 每块 {proto['head_block']} 行：“ckpt 重算”= 非重入 checkpoint、反向重算；“保留图”= 同样分块但保留各块反向图"
          f"（head_extra，按包的抽样顺序重放核对 {hx['replay_verified_cells']}/{len(grid)} 格）。", ""]
    hdr = ["N", "K"] + [lab for _, lab in HEAD_METHODS]
    rows = []
    for n, k in grid:
        rows.append([n, k] + [fms(num((head_cell(D, "cuda", "float32", n, k, m) or {}).get("median_ms"))) for m, _ in HEAD_METHODS])
    L += md_table(hdr, rows, "rr" + "r" * len(HEAD_METHODS)) + [""]
    # 4.2 memory
    braw = D["bench_raw"].get("float32")
    base_txt = ""
    if braw:
        bases = [r["cuda_baseline_allocated_bytes"] for r in braw["rows"] if r.get("cuda_baseline_allocated_bytes") is not None]
        base_txt = (f"本 run 各格调用前的基线 allocated 为 {fmib(min(bases))}–{fmib(max(bases))} MiB（原始 benchmark JSON，"
                    "sha256 与 T02_TIMING_cuda.json 的记录一致；保留图一列由 T02 脚本单独测，峰值同样含基线）")
    head_rows = [r for r in D["long"]["cuda"] if r["layer"] in ("head", "head_extra")]
    bad = sum(1 for r in head_rows if r["status"] != "completed")
    n_oom = sum(1 for r in head_rows if r["status"] == "oom")
    L += ["### 4.2 数值头峰值显存（L40S，fp32，MiB）", "",
          f"CUDA max_memory_allocated，**含进程基线**：{base_txt}。数值头与 head_extra 共 {len(head_rows)} 行，非 completed {bad} 行，OOM {n_oom} 行。", ""]
    rows = []
    for n, k in grid:
        rows.append([n, k] + [fmib((head_cell(D, "cuda", "float32", n, k, m) or {}).get("peak_allocated_bytes")) for m, _ in HEAD_METHODS])
    L += md_table(hdr, rows, "rr" + "r" * len(HEAD_METHODS)) + [""]
    # 4.3 fp64 vs fp32
    L += ["### 4.3 fp32 与 fp64（L40S，数值头，中位 ms）", ""]
    ms3 = (("moments", "moments"), ("dense", "dense"), ("blocked_dense", "分块(ckpt)"))
    hdr3 = ["N", "K"] + [f"{lab} {dt_}" for _, lab in ms3 for dt_ in ("fp32", "fp64")]
    rows = []
    for n, k in grid:
        rows.append([n, k] + [fms(num((head_cell(D, "cuda", dt_, n, k, m) or {}).get("median_ms"))) for m, _ in ms3 for dt_ in ("float32", "float64")])
    L += md_table(hdr3, rows, "rr" + "r" * 6) + [""]
    err = {}
    for dt_ in ("float32", "float64"):
        rr = [r for r in D["long"]["cuda"] if r["layer"] in ("head", "head_extra") and r["dtype"] == dt_
              and r["method"] in ("dense", "blocked_dense", "blocked_dense_graph_retained")]
        err[dt_] = (max(num(r["value_abs_err_vs_V"]) for r in rr),
                    max(max(num(r["grad_abs_err_A"]), num(r["grad_abs_err_B"]), num(r["grad_abs_err_W"])) for r in rr))
    at = D["bench_atol"]
    L += [f"精确 V 的其他实现（dense、分块 ckpt；fp32 另有保留图）相对 moments 的最大 |ΔJ|：fp32 {fe(err['float32'][0])}、fp64 {fe(err['float64'][0])}；"
          f"A/B/W 梯度最大绝对差：fp32 {fe(err['float32'][1])}、fp64 {fe(err['float64'][1])}。包 benchmark 的正确性门限 fp32 {fe(at['float32'], 0)}、"
          f"fp64 {fe(at['float64'], 0)}，超限格记 correctness_failed 且不计时（本表 {sum(1 for r in head_rows if r['status'] == 'correctness_failed')} 格）。"
          "perm1/perm8 与 V 的差是抽样偏差（4.5）。", ""]
    # 4.4 CPU
    flagged = cpu_outliers(D)
    cpu_rows = [r for r in D["long"]["cpu"] if r["layer"] == "head"]
    ms4 = (("permutation1", "perm1"), ("permutation8", "perm8"), ("dense", "dense"), ("blocked_dense", "分块(ckpt)"), ("moments", "moments"))
    hdr4 = ["N", "K"] + [f"{lab} fp32" for _, lab in ms4] + ["moments fp64", "dense fp64"]

    def cpu_cell(dt_, n, k, m):
        r = head_cell(D, "cpu", dt_, n, k, m)
        v = fms(num((r or {}).get("median_ms")))
        return v + ("†" if (dt_, n, k, m) in flagged else "")
    rows = []
    for n, k in grid:
        rows.append([n, k] + [cpu_cell("float32", n, k, m) for m, _ in ms4] + [cpu_cell("float64", n, k, "moments"), cpu_cell("float64", n, k, "dense")])
    L += ["### 4.4 数值头时间（CPU，中位 ms）", "",
          f"{envc['host']}（{envc['slurm_partition']}，作业 {envc['slurm_job_id']}，torch 线程 {envc['torch_threads']}），同一包 benchmark；CPU 不测显存。"
          f"† = 该格中位数比同方法、同 dtype、N 与 K 都不更小的另一格高 {OUTLIER_FACTOR} 倍以上（违反成本随规模单调的预期）：CPU 全部 {len(cpu_rows)} 行中 {len(flagged)} 行"
          f"（本表只列 fp32 {len(ms4)} 法与 fp64 2 法），L40S {cuda_outliers(D)}/{len(head_rows)} 行；原因未诊断。CPU 数字不进入 BUDGET（BUDGET 的来源是 T02_TIMING_cuda.json）。", ""]
    L += md_table(hdr4, rows, "rr" + "r" * 7) + [""]
    # 4.5 permutation variance
    pv = [r for r in D["long"]["cuda"] if r["layer"] == "perm_conditional_variance"]
    draws = {r["method"]: r["n_draws"] for r in pv}
    src = (REPO / "scripts" / "v10" / "t02_timing.py").read_text()
    mg = re.search(r"set_grad_enabled\(r < (\d+)\)", src)
    rows = []
    for n, k in grid:
        p1 = [r for r in pv if r["n_positions"] == str(n) and r["simplex_width"] == str(k) and r["method"] == "permutation1"][0]
        p8 = [r for r in pv if r["n_positions"] == str(n) and r["simplex_width"] == str(k) and r["method"] == "permutation8"][0]
        rows.append([n, k, fe(num(p1["J_V"]), 3), fe(num(p1["cond_sd"])), fe(num(p8["cond_sd"])), fg(num(p1["cond_grad_W_rel_err_mean"])),
                     fg(num(p8["cond_grad_W_rel_err_mean"]))])
    L += ["### 4.5 排列的条件抽样偏差（L40S，fp32）", "",
          f"固定 4.1 的 A/B/W，独立重抽普通排列（含固定点）：perm1 {draws.get('permutation1')} 次、perm8 {draws.get('permutation8')} 次；"
          f"sd 为 J 的条件标准差（同一组特征上的抽样偏差，不是独立样本方差）；W 梯度相对误差 = 前 {mg.group(1) if mg else '?'} 次抽样的 "
          "‖∇W J_perm − ∇W J_V‖ / ‖∇W J_V‖ 均值。", ""]
    L += md_table(["N", "K", "J_V", "sd perm1", "sd perm8", "W 梯度相对误差 perm1", "perm8"], rows, "rrrrrrr") + [""]
    # 4.6 train steps
    tr = [r for r in D["long"]["cuda"] if r["layer"] == "train_step"]
    cap = proto["m1_cap_binding_tolerance"]
    rows = []
    for r in tr:
        m1 = r["method"] == "MF_M1"
        rows.append([c(r["config"]) + ("（记录行）" if r["status"] == "completed_record_only" else ""), fms(num(r["median_ms"])), fms(num(r["p90_ms"])),
                     fmib(r["peak_allocated_bytes"]), fms(num(r["phase_objective_ms"])), fms(num(r["phase_backward_ms"])),
                     fi(num(r["m1_iterations_median"])) if m1 else "—", fe(num(r["m1_gap_median"])) if m1 else "—",
                     fe(num(r["m1_gap_initial_median"])) if m1 else "—", fi(num(r["m1_hvp_calls_median"])) if m1 else "—",
                     fg(num(r["m1_converged_fraction"])) if m1 else "—"])
    n_sc = tr[0]["n_sc"]
    n_pos = tr[0]["n_pos"]
    n_every = sum(1 for r in tr if r["checks"] == "every_step")
    clip_m = re.search(r"AdamW \+ clip ([\d.]+)", tr[0]["notes"])
    rec_lab = next((r["config"] for r in tr if r["status"] == "completed_record_only"), "—")
    L += ["### 4.6 含 encoder 的训练步（L40S，fp32，ms）", "",
          f"宿主 batch：{n_sc} 个 FIT crop × {n_pos} 位置，宿主 encoder，AdamW + clip {clip_m.group(1) if clip_m else '?'}（CSV notes）；warmup {proto['train_warmup']}、计时 {proto['train_timed']} 步"
          f"（中位、p90）；阶段时间（目标、反向）来自另外 {proto['train_phase_steps']} 步的同步打点。MF 的 W-box/排列检查只在首步（另有 {n_every} 行每步检查）。"
          f"M1 的 capbound 行 = tolerance {fe(cap, 0)}、require_converged False，迭代预算每批用满；`{rec_lab}` 行只记录协议设置，不进 BUDGET（§6 决定 1）。", ""]
    L += md_table(["配置", "中位", "p90", "峰值 MiB", "目标", "反向", "M1 迭代", "M1 gap 中位", "M1 起始 gap 中位", "HVP 次数", "收敛比例"],
                  rows, "lrrrrrrrrrr") + [""]
    # 4.7 registration
    rg = [r for r in D["long"]["cuda"] if r["layer"] == "registration" and r["phase"] == "registration"]
    enc = [r for r in D["long"]["cuda"] if r["layer"] == "registration" and r["phase"] == "reference_encoding"]
    cfgs = list(dict.fromkeys(r["config"] for r in rg))
    reg = t2c["parts"]["register"]
    m2 = re.search(r"best N_STARTS = (\d+) refined by N_ADAM = (\d+) Adam steps", D["host_map"]["contract"]["geometry"]["search"])
    check(m2 is not None, "host_map search description")
    search = f"网格候选 + 网格最优 {m2.group(1)} 个起点各 {m2.group(2)} 步 Adam 精修，Q.optimise_rigid；host_map.json"
    rows = []
    n_warm = sorted({sum(1 for r in rg if r["config"] == cfg and r["warmup_row"] == "True") for cfg in cfgs})
    for cfg in cfgs:
        timed = [r for r in rg if r["config"] == cfg and r["warmup_row"] == "False" and r["status"] == "completed"]
        sec = [num(r["elapsed_seconds"]) for r in timed]
        rows.append([c(cfg), len(timed), fs(statistics.median(sec)), f"{fs(min(sec))}–{fs(max(sec))}",
                     fs(statistics.median(num(r["grid_seconds"]) for r in timed)),
                     fs(statistics.median(num(r["refine_and_final_seconds"]) for r in timed)),
                     "、".join(sorted({r["exact_energy_calls"] for r in timed})), "、".join(sorted({r["n_grid"] for r in timed})),
                     "、".join(sorted({r["n_positions"] for r in timed})), fmib(max(num(r["peak_allocated_bytes"]) for r in timed))])
    L += ["### 4.7 完整配准（L40S，fp32，每次配准秒数）", "",
          f"宿主共享搜索器不变（{search}），**未训练的固定模型，只计时**（不计算误差或真值量）；"
          f"{len(reg['cores'])} 个 DEV core × {len(reg['perturbation_indices'])} 个扰动 = 每能量计时 {len(reg['cores']) * len(reg['perturbation_indices'])} 次"
          f"（另 {'/'.join(map(str, n_warm))} 次 warmup 不计）。参考侧 encoder 一次：{fs(num(enc[0]['elapsed_seconds'])) if enc else '—'} s（含读取参考图）。", ""]
    L += md_table(["能量", "n", "中位", "范围", "网格阶段", "精修 + 终值", "精确能量调用", "网格候选", "位置 N", "峰值 MiB"],
                  rows, "lrrrrrrrrr") + [""]
    # 4.8 V9 reference
    v9 = D["budget"]["v9_reference_costs"]
    rows = []
    for unit, e in v9.items():
        if not isinstance(e, dict):
            continue
        rows.append([c(unit), e.get("device_job_per_T00_audit"), e.get("registrations"), fs(e.get("s_per_registration_cal")),
                     fs(e.get("s_per_registration_dev")), fs(e.get("train_seconds"), 1) if e.get("train_seconds") else "—",
                     fg(e.get("train_peak_gib")) if e.get("train_peak_gib") else "—"])
    L += ["### 4.8 V9 参考成本（只读，BUDGET.json `v9_reference_costs`）", ""]
    L += md_table(["V9 单元", "设备 / 作业", "配准次数", "s/次 CAL", "s/次 DEV", "训练 s", "训练峰值 GiB"], rows, "llrrrrr") + [""]
    return L


DEPENDS = {"MLP_VCS_perm1": "2", "MLP_VCS_perm8": "2、4", "MF_VCS_perm1": "2", "MF_VCS_M0": "2", "MF_VCS_M1": "1、2",
           "InfoNCE_host": "2、3、4", "CoMIR_native": "2、4", "RuLSIF_kernel": "4", "MI_native": "4", "NMI_native": "4",
           "MIND_2D_adapted": "4"}


def unit_plan(D) -> dict:
    """T04/T05 rows of UNIT_LEDGER.csv joined with the BUDGET per-unit projection."""
    B = D["budget"]
    pu = B["per_unit_projection"]
    groups = defaultdict(list)
    for r in D["ledger"]:
        if r["stage"] != "T02":
            groups[(r["dataset"], r["method"], r["stage"])].append(r)
    rows, tot = [], defaultdict(float)
    for (ds, m, st), rr in groups.items():
        e = pu.get(m, {})
        key = "gpu_h_bf_unit" if ds == "BF_SHG" else "gpu_h_epfl_unit_proxy"
        h = e.get(key) if e.get("status") == "measured" else None
        sub = h * len(rr) if h is not None else None
        if sub is not None:
            tot[ds] += sub
        seeds = sorted({x["seed"] for x in rr})
        rows.append({"dataset": ds, "method": m, "stage": st, "n": len(rr), "seeds": seeds, "per_unit": h, "sub": sub,
                     "status": e.get("status", "not_in_budget"), "depends": DEPENDS.get(m, "?") + ("、5" if ds != "BF_SHG" else ""),
                     "run_status": sorted({x["run_status"] for x in rr})})
    g = B["grid_projection_measured_methods"]
    check(close(tot["BF_SHG"], g["BF_SHG_gpu_h"]) and close(tot["EPFL_RGB_NIR"], g["EPFL_gpu_h_proxy"]), "unit plan vs BUDGET grid")
    return {"rows": rows, "tot": tot}


def sec5(D) -> list[str]:
    syn, proto = D["synthetic"], D["protocol"]
    d, head, reg = syn["defaults"], proto["head"], proto["registration"]
    n_cond = len(syn["cases"])
    per = d["n_fit"] + d["n_cal"] + d["n_eval"]
    mf = [m for m in reg["learned_methods"] if m.startswith("MF_")]
    mlp = [m for m in reg["learned_methods"] if m.startswith("MLP_")]
    K, lrs, seeds, ts = head["width_candidates"], head["feature_head_learning_rate_candidates"], reg["training_seeds"], reg["tuning_seed"]
    n_cfg = len(mf) * len(K) + len(mlp)
    pkg_ok = D["pkg_manifest"].get("synthetic/src/v10_synthetic.py") == syn["generator_sha256"]
    L = ["## 5 下一批具体单元与预计资源（提议，未提交）", "",
         "### 5.1 T03 受控真值（CPU / 小 GPU；T03 代码尚未编写）", "",
         f"- 数据：包生成器 `synthetic/src/v10_synthetic.py`（sha256 {syn['generator_sha256'][:16]}…，{'= 包清单' if pkg_ok else '≠ 包清单'}）的默认参数："
         f"seed {d['seed']}，{'、'.join(d['families'])}，m ∈ {{{', '.join(map(str, d['m']))}}}，ρ ∈ {{{', '.join(fv(x) for x in d['rho'])}}}；"
         f"每条件 FIT/CAL/EVAL = {d['n_fit']}/{d['n_cal']}/{d['n_eval']} 个独立基本观测，{n_cond} 个条件共 {n_cond * per} 个。"
         f"输出写 `outputs_v10/T03/`，再跑包的 `synthetic/tests/check_synthetic.py`（包示例套件上同一脚本 {syn['check_n']} 项 {syn['check_status']}）。"
         "truth 只交给评估器。",
         f"- 方法（建议网格）：{'、'.join(mf)} × K ∈ {{{', '.join(map(str, K))}}}，加 {'、'.join(mlp)}（整数观测 one-hot 输入）；"
         f"CAL 在调参 seed {ts} 上按 CAL 经验 J 从 lr ∈ {{{', '.join(fe(x, 0) for x in lrs)}}} 中选，最终 seed {'/'.join(map(str, seeds))}："
         f"{n_cond} 条件 × {n_cfg} 配置 → 最终训练 {n_cond * n_cfg * len(seeds)} 次、CAL 候选 {n_cond * n_cfg * len(lrs)} 次。M1 行等决定 1。",
         f"- 目标模式：包 §2.1 把 U 留作{quote(D, '受控 iid 估计实验')}；建议 V（与配准一致）为主，U 作命名的敏感性行。"
         f"其他估计器按包 §6.3 {quote(D, 'MINE/DV/NWJ/SMILE/InfoNCE以各自原生量和共同下游任务比较，不用S真值给非S输出算RMSE')}，不在本提议内。",
         "- 指标（包 §6.3）：E_M(T−η)²（evaluator-only truth）、EVAL 经验 J 对 S 的 bias / variance / RMSE（跨 seed）、同一基础数据上的排列方差、M1 gap、总时间。",
         "- 需先写：`scripts/v10/t03_synthetic.py`（训练只经 `load_samples` 读 FIT/CAL，评估器单独 `load_evaluation_truth`）、单元测试与 smoke。",
         "- 资源：CPU（cpu-high）为主，**未实测**；第一个作业 = 生成 + 检查 + 1 个条件的 smoke 计时，再写预算。",
         "", "条件（精确总体 S 读自包示例套件的 truth.json；S 只取决于 family、m、ρ）：", ""]
    L += md_table(["条件", "family", "m", "ρ", "S"], [(c(x["case"]), x["family"], x["m"], fv(x["rho"]), fg(x["S"], 4)) for x in syn["cases"]], "llrrr")
    L += [""]
    # 5.2 T04/T05
    plan = unit_plan(D)
    B = D["budget"]
    c_ = B["protocol_counts"]
    g = B["grid_projection_measured_methods"]
    rows = []
    for r in plan["rows"]:
        seeds_txt = f"{r['n']}（seed {'/'.join(r['seeds'])}）" if r["seeds"] != ["not_applicable"] else f"{r['n']}（非学习，无 seed）"
        rows.append((c(r["method"]), r["dataset"], r["stage"], seeds_txt,
                     fh(r["per_unit"]) if r["per_unit"] is not None else ("未测（仅训练步）" if r["status"] == "partially_measured" else "未测"),
                     fh(r["sub"]) if r["sub"] is not None else "—", r["depends"], "、".join(x.split(" (")[0] for x in r["run_status"])))
    nf = len(g["methods"])
    cand = B["protocol_counts"]["cal_candidates_per_family"]
    rows.append(("CAL 调参（已测族）", "BF_SHG", "T04_T05", f"{nf} 族 × {cand} 候选（调参 seed {B['protocol_counts']['tuning_seed']}）",
                 "—", fh(g["CAL_tuning_gpu_h_BF"]), "1、2", "NOT_RUN"))
    L += ["### 5.2 T04/T05 单元（UNIT_LEDGER.csv；GPU-h 取 BUDGET.json，L40S；全部 NOT_RUN）", "",
          f"每单元 = {B['protocol_counts']['train_steps_per_unit']} 训练步 + 配准 {D['budget_counts']['n_reg_bf']} 次（BF-SHG：{c_['checkpoints_per_unit']} checkpoint × "
          f"{c_['bf_cal_cores']} CAL × {c_['perturbations_per_image']} + {c_['bf_dev_cores']} DEV × {c_['perturbations_per_image']}）/ {D['budget_counts']['n_reg_ep']} 次"
          f"（EPFL：{c_['checkpoints_per_unit']} × {c_['epfl_cal_windows']} CAL × {c_['perturbations_per_image']} + {c_['epfl_dev_windows']} DEV × {c_['perturbations_per_image']}，代理）；"
          f"M1 按 {c(B['per_unit_projection']['MF_VCS_M1']['train_config'])}（迭代上限用满，上界）计。已测合计 **{fh(g['total_gpu_h'])} GPU-h**（BF-SHG {fh(g['BF_SHG_gpu_h'])} + EPFL 代理 "
          f"{fh(g['EPFL_gpu_h_proxy'])} + BF CAL 调参 {fh(g['CAL_tuning_gpu_h_BF'])}）；“依赖决定”列为 §6 的编号。", ""]
    L += md_table(["方法", "数据", "阶段", "单元", "每单元 GPU-h", "小计", "依赖决定", "状态"], rows, "llllrrll") + [""]
    # 5.3 batches
    sl = D["slurm"]
    bf_units = sum(r["n"] for r in plan["rows"] if r["dataset"] == "BF_SHG" and r["per_unit"] is not None)
    ep_units = sum(r["n"] for r in plan["rows"] if r["dataset"] != "BF_SHG" and r["per_unit"] is not None)
    other_units = sum(r["n"] for r in plan["rows"] if r["per_unit"] is None)
    b1 = g["CAL_tuning_gpu_h_BF"] + g["BF_SHG_gpu_h"]
    pu = B["per_unit_projection"]
    m1 = pu["MF_VCS_M1"]
    alt = []
    m1cfg = m1["train_config"]
    kpref = m1cfg.split("_it")[0]
    alts = sorted((k for k in B["measured"]["train_step_ms_median"] if k.startswith(kpref + "_it") and k.endswith("_capbound")),
                  key=lambda k: int(re.search(r"_it(\d+)_", k).group(1)))
    for cfg in alts:
        ms = B["measured"]["train_step_ms_median"][cfg]
        alt.append((cfg, (c_["train_steps_per_unit"] * ms / 1000 + D["budget_counts"]["n_reg_bf"] * m1["s_per_registration_median"]) / 3600))
    L += ["### 5.3 建议的批次（均为提议；每批提交前再按队列规则核对）", "",
          "- **批 0（不需 owner 决定）**：T03 数据生成 + 检查 + T03 代码与 smoke（CPU）；成本未测。",
          f"- **批 1（决定 1、2 之后）**：BF-SHG CAL 调参 {fh(g['CAL_tuning_gpu_h_BF'])} GPU-h，再跑已测 {nf} 族 × {len(c_['training_seeds'])} seed"
          f"（{bf_units} 单元，{fh(g['BF_SHG_gpu_h'])} GPU-h）；合计 {fh(b1)} GPU-h，V10 同时最多 {sl['v10_max']} 个作业时墙钟约 {fh(b1 / sl['v10_max'], 1)} h（不含排队）。",
          f"- **批 2（决定 5 / EPFL 审计之后）**：EPFL 已测 {nf} 族（{ep_units} 单元，{fh(g['EPFL_gpu_h_proxy'])} GPU-h，代理值）；EPFL 的 CAL 调参未计入 BUDGET。",
          f"- **批 3（决定 3、4 与实现之后）**：其余 {other_units} 个单元（{'、'.join(dict.fromkeys(r['method'] for r in plan['rows'] if r['per_unit'] is None))}）；"
          "先做 T01 式核对与 T02 式计时再定 GPU-h。",
          f"- M1 迭代上限的成本（BF-SHG 每单元 GPU-h，按 BUDGET 同一公式由 T02 步时推算；BUDGET 只含 {c(m1cfg)}）："
          + "；".join(f"{c(k)} {fh(v)}" for k, v in alt)
          + f"。包把 max_iter ∈ {{{', '.join(map(str, D['protocol']['head']['M1_max_iterations_candidates']))}}} 也算作调参，BUDGET 的 CAL 调参未含此维度。", ""]
    return L


def sec6(D) -> list[str]:
    proto, head = D["protocol"], D["protocol"]["head"]
    t2c = D["t2"]["cuda"]
    env = t2c["environment"]
    ps = per_step_stats(D)
    L = ["## 6 待 owner 决定（OPEN DECISIONS；本报告不作决定，“建议”仅为默认提议）", ""]
    # ---- 1
    a = {dev: check_row(D, f"adapter_{dev}", "MF_M1_protocol_tolerance_from_W0_moves_W_and_trains_encoders") for dev in ("cpu", "cuda")}
    cb = {dev: check_row(D, f"adapter_{dev}", "MF_M1_cap_binding_step_logs_gap_iterations_hvp") for dev in ("cpu", "cuda")}
    cw = {dev: check_row(D, f"adapter_{dev}", "MF_M1_cap_binding_W_is_inner_solution_nonzero_in_box") for dev in ("cpu", "cuda")}
    tr = {r["config"]: r for r in D["long"]["cuda"] if r["layer"] == "train_step"}
    rec = next((k for k, r in tr.items() if r["status"] == "completed_record_only"), None)
    cap = t2c["protocol"]["m1_cap_binding_tolerance"]
    capcfg = [k for k, r in tr.items() if r["method"] == "MF_M1" and num(r["m1_tolerance"]) == cap]
    tol = head["M1_gap_absolute_target"]
    L += ["### 6.1 决定 1：MF-M1 头的停止规则（影响全部 MF_VCS_M1 单元与 T03 的 M1 行）", "",
          f"- **问题**：协议 `head.M1_gap_absolute_target` = {fe(tol, 0)}（绝对 FW gap；包 §7：{quote(D, 'gap目标1e-4是起始工程容差，达到iteration cap时报告不足，不硬称精确包络梯度')}），包的起点 W = 0，W = 0 处的 FW gap 恰为 ‖Δ‖₁"
          "（Δ = F 在 W = 0 处对 W 的梯度，即 P 与 Q 的一阶矩差 E_P[abᵀ] − E_Q[abᵀ]）。‖Δ‖₁ 低于目标时求解器第 0 次迭代即停：W 保持 0，"
          "外层目标恒为 0，feature map 与 encoder 的梯度恰为 0（只剩 AdamW 衰减）。",
          f"- **证据 T01**（真实宿主初始化，K16）：‖Δ‖₁ = {fe(a['cpu']['m1_gap_initial_equals_Delta_l1'], 4)}（CPU）/ "
          f"{fe(a['cuda']['m1_gap_initial_equals_Delta_l1'], 4)}（CUDA）→ 迭代 {a['cpu']['m1_iterations']}、|W|max {fg(a['cpu']['w_absmax'])}、"
          f"encoder 梯度范数 {fg(a['cpu']['encoder_grad_norm'])}、J {fg(a['cpu']['J'])}（诊断行，记为 FAIL、不门控）。同一 batch（CPU）强制用满 "
          f"{cb['cpu']['m1_iterations']} 次迭代：gap {fe(cb['cpu']['m1_gap_initial'], 3)} → {fe(cb['cpu']['m1_gap'], 3)}，|W|max {fg(cw['cpu']['w_absmax'])}，"
          f"converged {str(cb['cpu']['m1_converged']).lower()}（必需行 PASS）。"]
    if rec and rec in ps:
        s = ps[rec]
        L += [f"- **证据 T02 协议记录行** {c(rec)}（seed {t2c['seed']}，逐步文件 `train_steps_per_step.json` sha256 {D['per_step']['sha256'][:16]}…）："
              f"前 {s['lead_degenerate']} 步" + (f"（恰为全部 {t2c['protocol']['train_warmup']} 个 warmup 步，不在计时窗口内）" if s["lead_degenerate"] == t2c["protocol"]["train_warmup"]
                                                 else f"（都在 {t2c['protocol']['train_warmup']} 个 warmup 步内）" if s["lead_degenerate"] < t2c["protocol"]["train_warmup"] else "")
              + f"全部 0 次迭代、W = 0、J = 0，各步 ‖Δ‖₁ ∈ [{fe(s['lead_delta'][0])}, {fe(s['lead_delta'][1])}]；"
              f"第 {s['first_active'][0] + 1} 步（index {s['first_active'][0]}）‖Δ‖₁ = {fe(s['first_active'][1])} 超过目标后才开始迭代；"
              f"之后计时 {s['n_timed']} 步中 W = 0 的 {s['timed_w0']} 步、收敛 {s['timed_conv']} 步。即起步时刻由 batch 间 ‖Δ‖₁ 的波动决定。"]
    if capcfg and ps:
        fd = defaultdict(list)
        for k in capcfg:
            if k in ps:
                fd[tr[k]["simplex_width"]].append(ps[k]["first_delta"])
        L += ["- **证据 T02 首个 batch**（W = 0 起步）的 ‖Δ‖₁：" + "、".join(f"K{kk} {fe(v[0])}" for kk, v in sorted(fd.items(), key=lambda t: int(t[0])))
              + f"，均 < {fe(tol, 0)}。",
              f"- **证据 T02 上限强制行**（tolerance {fe(cap, 0)}）：" + "；".join(
                  f"{c(k)} 计时步 gap 中位 {fe(num(tr[k]['m1_gap_median']), 3)}、最小 {fe(ps[k]['timed_min_gap'], 3)}" for k in capcfg if k in ps)
              + (f"。计时步中没有一步的 gap 到达 {fe(tol, 0)}：一旦起步，迭代上限总是先到。" if all(ps[k]["timed_min_gap"] > tol for k in capcfg if k in ps)
                 else f"。部分计时步的 gap 低于 {fe(tol, 0)}。")]
    L += ["- **选项**：",
          f"  - A 尺度相对目标：G(W) ≤ ε_rel·‖Δ‖₁（同一 batch、同一目标；W = 0 时相对 gap 为 1，至少迭代一次）。以 |F| 作分母不合适（W = 0 时 F = 0）。",
          "  - B 非零起点 W：例如现有的 seeded uniform 初始化（目前只用于检查与计时）或取 M0 的 W。现实现已在 batch 间热启动（沿用上一 batch 的 W），"
          "问题只出在训练起点 W = 0；非零起点使 M1 与 M0 / perm1（均从 W = 0 起）的初始化不对称，须登记。",
          "  - C 更小的绝对目标：仍随 K、温度与特征尺度变化；按上面的 gap，一旦起步，上限总是先到。",
          f"  - D 维持 {fe(tol, 0)}：起步延迟取决于 batch 噪声（T02 记录行中为 {ps[rec]['lead_degenerate'] if rec in ps else '?'} 步）。",
          f"- **建议默认（建议，未执行）**：A，ε_rel = {fe(PROPOSAL['m1_relative_gap_target'], 0)}；迭代上限沿用包的 "
          f"{{{', '.join(map(str, head['M1_max_iterations_candidates']))}}} 作为 CAL 维度；require_converged = "
          f"{str(head['M1_require_converged_training'])}，逐批记录 gap 与 ‖Δ‖₁；W 起点保持 0。需改 `M1Config`（src/v10/adapter.py）"
          "并重跑 T01 的 M1 行（小）；BUDGET 的 M1 成本（迭代上限用满）在 A 下仍是上界。", ""]
    # ---- 2
    geo = {}
    for k in ("geometry_cpu", "geometry_cuda"):
        for x in D["t01_parts"][k]["checks"]:
            m = re.match(r"fp32_vs_fp64_geometry_grad_(default_precision|cudnn_conv_ieee)_(?:(sharpened_x10)_)?(?:host_init_)?(T\d)_", x["name"])
            if m:
                state = m.group(2) or "host_init"
                geo[(k.split("_")[1], m.group(1), state, m.group(3))] = x
    states = sorted({(k[2], k[3]) for k in geo})
    tl = {}
    for t in D["gg"]["method"]["transforms"]:
        mt = re.match(r"(T\d)_rot([+-][\d.]+)deg", t)
        if mt:
            tl[mt.group(1)] = f"{mt.group(1)}（{mt.group(2)}°）"
    tol32 = D["tv"]["tolerances_design"]["float32"]
    check(all(x["allclose"] for x in geo.values()), "geometry fp32 vs fp64 rows not all within the fp32 bound")
    grow = []
    for st, tt in states:
        ref = geo.get(("cpu", "default_precision", st, tt))
        cell = lambda dev, mode: fe(geo[(dev, mode, st, tt)]["max_rel_err"]) if (dev, mode, st, tt) in geo else "—"  # noqa: E731
        grow.append((f"{st} {tl.get(tt, tt)}", fe(ref["ref_max_abs"]) if ref else "—", fg(ref["ref_over_atol"]) if ref else "—",
                     cell("cpu", "default_precision"), cell("cuda", "default_precision"), cell("cuda", "cudnn_conv_ieee")))
    rep = re.search(r"([\d.]+) px on RTX PRO 6000, ([\d.]+) px on L40S", D["host_map"]["environment_numerics"]["repeatability_v9_finding"])
    L += ["### 6.2 决定 2：TF32 策略（全部 GPU 方法）", "",
          f"- **问题**：torch {env['torch']} 默认 cuDNN fp32 卷积用 TF32（作业环境 `cudnn_conv_fp32_precision = {env['cudnn_conv_fp32_precision']}`），"
          f"fp32 matmul = `{env['float32_matmul_precision']}`；宿主文件中没有任何精度开关（T00）。第一阶段全部 CUDA 计时与 T01 的 CUDA fp32 行都在此默认下运行。",
          f"- **证据**：几何梯度 fp32 对 fp64 的最大相对误差（T01 geometry 诊断行，同一 fp32 舍入权重；全部在 fp32 门限 atol {fe(tol32['atol'], 0)} + rtol "
          f"{fe(tol32['rtol'], 0)} 内，但 host_init 行的参考梯度低于 atol，该门限在此不敏感）：", ""]
    L += md_table(["状态 / 变换", "参考 max|g|", "参考 / atol", "CPU", "L40S 默认（TF32 卷积）", "L40S IEEE 卷积"], grow, "lrrrrr") + [""]
    L += [f"  V9 记录：学习型 critic 的配准不逐位可重复（同设备最大 {rep.group(1) if rep else '?'} px RTX PRO 6000、{rep.group(2) if rep else '?'} px L40S），"
          "原因未定位（host_map.json）。BUDGET 的全部 GPU 成本在 TF32 卷积下测得，IEEE 卷积下的成本未测。"
          f"包 §6.1：{quote(D, '若大规模fp32消减错误超限，使用显式更高精度积累并把成本计入')}；T01 的 fp32 门控行没有超限。",
          "- **选项**：A 全部方法保持 torch 默认并写进协议（作业环境已逐次记录这些开关）；B 全部方法关闭 TF32（cuDNN conv fp32_precision = ieee）："
          "几何梯度误差降到上表 IEEE 列的量级，需先重测 T02 train / register；C 训练用默认、测试与配准能量用 IEEE：训练与测试数值设置不同，须登记。",
          "- **建议默认（建议，未执行）**：A——所有方法同一设置，与 V9 宿主和已测 BUDGET 一致；若 owner 更看重跨 GPU 型号的数值可重复性，选 B 并先补测成本。", ""]
    # ---- 3
    d3 = next(x for x in D["host_map"]["target_differences_vs_package"] if x["id"] == "D3")
    rg = [r for r in D["long"]["cuda"] if r["layer"] == "registration" and r["phase"] == "registration"]
    am = D["adapter_methods"]
    L += ["### 6.3 决定 3：InfoNCE 的配准能量", "",
          "- **问题**（T00 差异 D3）：V9 宿主用与 vcs 相同的 J(tanh f) perm1 能量为 logistic 与 InfoNCE 训练的 critic 配准"
          f"（host_map.json D3：{d3['host']}）；包的 InfoNCE_host 行把“原生目标 / 候选归一化”列为改变的因素。",
          f"- **证据**：V10 trainer 只接受 {', '.join(am.get('TRAINER', []))}，InfoNCE 训练未接入；"
          f"宿主每次配准 N = {'、'.join(sorted({r['n_positions'] for r in rg}))} 个位置、网格 {'、'.join(sorted({r['n_grid'] for r in rg}))} 个候选"
          "（T02_LONG_cuda.csv）；T02 未测任何 InfoNCE 能量。",
          "- **选项**：A 宿主能量：InfoNCE 训练 + J(tanh f) perm1 能量（与 V9 定义相同，表中命名 InfoNCE_host[J]）；B 原生 InfoNCE 能量："
          "每个候选需要 N×N 打分与候选归一化，是新定义，需实现、T01 式核对与 T02 式计时；C A 进主表，B 作单列命名变体。",
          "- **建议默认（建议，未执行）**：A（命名明确，表注披露 D3）；B 只在实测成本后作为单列变体（即 C）。", ""]
    # ---- 4
    B = D["budget"]
    nm = B["not_measured_in_T02"]
    p8 = B["per_unit_projection"]["MLP_VCS_perm8"]
    st = [("RuLSIF / 核（精确矩）", "未实现。V9 宿主的核行是 RFF 岭回归闭式解、每个 core 一个排列的负对，不是包 §5 的可分核精确矩", nm.get("RuLSIF_kernel", "")),
          ("MIND-2D 适配", f"未实现；src/v9/reg 中含 MIND 的文件 {len(D['mind_files'])} 个", nm.get("MI_native / NMI_native / MIND_2D_adapted", "")),
          ("MLP perm8 配准能量", f"未实现；训练侧已有（T02 步时 {fms(p8['train_step_ms_median'])} ms）", nm.get("MLP_VCS_perm8 registration energy", "")),
          ("CoMIR 原生", "V9 有完整流程，未接入 V10（seed、输出路径、数据门）", nm.get("CoMIR_native", "")),
          ("MI / NMI", "V9 有，未接入 V10", nm.get("MI_native / NMI_native / MIND_2D_adapted", "")),
          ("InfoNCE", "待决定 3；V10 trainer 未接入", nm.get("InfoNCE_host", ""))]
    L += ["### 6.4 决定 4：T04/T05 之前尚缺的基线", ""]
    L += md_table(["方法", "V10 现状", "BUDGET 记录"], st, "lll") + [""]
    meas = B["grid_projection_measured_methods"]["methods"]
    L += [f"- **选项**：A 全部实现并核对后再启动任何 T04 单元；B 分批：已测 {len(meas)} 族（{'、'.join(meas)}）在决定 1–2 后先跑，"
          "各基线并行实现 → T01 式核对 → T02 式计时 → 在同一冻结协议下补跑；全部基线完成前不写任何比较结论；"
          "C 删减基线（包要求强基线与 same-data / same-total-cost 比较，不建议）。",
          "- **建议默认（建议，未执行）**：B。", ""]
    # ---- 5
    mb = D["mbs"]
    bf, ep = mb["bf_shg"], mb["epfl"]
    rset = mb["manifest"]["rows_by_dataset_inclusion_role_split"]
    hold = rset.get("BF_SHG|exclude|DEV|test")
    resid = re.search(r"residual of about (\d+) native px", D["exposure_notes"])
    thr = proto["registration"]["threshold_native_px"]
    rd = D["reads"]
    L += ["### 6.5 决定 5：数据角色（TEST、CONFIRM、EPFL）", "",
          f"- **证据（T00）**：BF-SHG {mb['counts']['bf_total_records']} 个 core 全部已被读过（{bf['trainval_cores']} 个 train/validation 由 R6/V9 使用；"
          f"{bf['test_cores']} 个官方 test 由 V9-G1 做过一次确认读取，读取开始 {bf['test_read_utc']}）；EPFL {ep['pairs']} 对全部被 REG-P3 审计读过"
          f"（其中 {len(ep['p60_ids'])} 对另被 P60 读过）；可登记为 CONFIRM 的组 {mb['confirm_assessment']['eligible_groups']} 个。"
          f"TEST 在 V10 manifest 中为 HOLD（role DEV、inclusion exclude，{hold} 行）。{bf['slide_position_overlap']['shared_positions']} 个 TMA 位置同时出现在 "
          f"1B 与 2B 两张切片，是否同一组织块未记录。EPFL 保留 {ep['kept']} 个窗口（对齐排除 {ep['excluded_alignment']}、尺寸排除 {ep['excluded_size']}），"
          f"发布对齐的残差约 {resid.group(1) if resid else '?'} native px，严格阈值 {fg(thr['EPFL_RGB_NIR'][1])} native px 接近该限。"
          + ("V10 adapter 的数据门只开放 BF_SHG 行（src/v10/adapter.py DataGuard）。" if D["adapter_methods"].get("bf_only_gate") else "")
          + f"第一阶段实际读取：FIT {rd.get('FIT', 0)} 个 core、DEV {rd.get('DEV', 0)} 个 core（仅 T02 计时，未训练模型，"
          f"不计算误差）、CAL {rd.get('CAL', 0)}、TEST {rd.get('TEST', 0)}。",
          f"  包 §4.3：{quote(D, 'V9已读或参与方法设计的集合，在V10仍是开发/重复benchmark。')}",
          f"- **选项**：A BF-SHG 只用 DEV（{bf['roles']['DEV']} 个 core）作开发主表；B 另把 {bf['test_cores']} 个 TEST core 作为单独的重复基准块"
          "（V10 协议冻结后、新 manifest 版本；与 DEV 分表、不合并）；C 寻找新的 CONFIRM 数据（超出现有授权，需单独提案；T00 未评估服务器上的其他数据）。"
          "EPFL：在 V10 的 EPFL 宿主路径（V9-G2 / REG-P3）adapter 与对齐审计完成后激活，单独分表。",
          "- **建议默认（建议，未执行）**：A 立即；B 在 V10 协议冻结后；EPFL 审计后激活；全部结论标为开发 / 重复基准结果；C 只作独立提案。", ""]
    return L


def sec7(D) -> list[str]:
    rj = set(result_jobs(D))
    rows = []
    for job, Lg in sorted(D["logs"].items()):
        if job in rj:
            continue
        h = Lg["header"]
        req = [f for p in Lg["parts"] if p["status"] != "PASS" for f in p["required_failures"]]
        exc = [f for f in req if f.endswith(":exception")]
        items = []
        if exc:
            errs = sorted({Lg["required_rows"].get(f, {}).get("error", "?") for f in exc})
            pat = "real_m1_*:exception" if all(f.startswith("real_m1_") for f in exc) else ", ".join(exc)
            items.append(f"{len(exc)} 行 {c(pat)}：{'；'.join(errs)}")
        for f in req:
            if f in exc:
                continue
            r = Lg["required_rows"].get(f, {})
            items.append(f"{c(f)}：max_abs_err {fe(r.get('max_abs_err'))} = {fg(r.get('worst_err_over_bound'))} × fp32 界")
        bad = [f"{p['part']}_{p['dev']}" for p in Lg["parts"] if p["status"] != "PASS"]
        ok = [f"{p['part']}_{p['dev']}" for p in Lg["parts"] if p["status"] == "PASS"]
        mentions = [x for x in D["git"]["commits"] if re.search(rf"\b{job}\b", D["git"]["msgs"][x["sha"]]) and x["sha"] != h["commit"]]
        dirs = [str(p) for p in (Path("/projects/EEG-foundation-model/yinghao/outputs_v10/T01/runs") / f"cpu_{job}",
                                 Path("/projects/EEG-foundation-model/yinghao/outputs_v10/T02/runs") / f"cpu_{job}") if p.exists()]
        reasons = []
        if exc:
            reasons.append("exception 行：检查代码 bug（关键字参数冲突），不是数值失败")
        if any(not f.endswith(":exception") and f.startswith("MLP_dense") for f in req):
            fr = check_row(D, "adapter_cpu", "MLP_dense_full_float32_grads_vs_float64_value_same_inputs")
            br = check_row(D, "adapter_cpu", "MLP_dense_blocked_checkpointed_float32_grads_vs_float64_value_same_inputs")
            reasons.append(f"fp32 行：两侧都是 fp32 实现，误差来自 fp32 全图实现（同一输入对 fp64：全图 {fe(fr['max_abs_err'])}、分块 {fe(br['max_abs_err'])}，当前 T01 CPU）")
        rows.append((job, f"{c(h['commit'][:7])} / {c(h['code'])}", f"{'、'.join(bad)} FAIL（必需失败 {len(req)} 行）：" + "；".join(items)
                     + f"。其余 {len(ok)} 个 T01 part PASS", "；".join(reasons),
                     "；".join(f"{c(x['sha'][:7])} {short(x['subject'], 70)}" for x in mentions),
                     f"{Path(Lg['path']).name}（sha256 {Lg['sha256'][:12]}…）；" + "、".join(dirs)))
    for x in jobs_without_log(D):
        rows.append((x["job"], "—", "未运行（排队中取消）", "pin 指向门控修改之前的代码", f"{c(x['commit'][:7])}：“{x['sentence']}”", "无日志"))
    L = ["## 7 失败尝试（保留原因与原日志）", ""]
    L += md_table(["作业", "提交 / code hash", "失败内容", "原因", "相关提交（git）", "日志与原始文件"], rows, "llllll") + [""]
    L += ["这些作业的结果文件已被结果作业取代（结果提交链见 §1），日志与 part 文件原样保留；没有任何失败行被删除或改判，"
          "fp32 门控的修改见 §8。", ""]
    return L


def sec8(D) -> list[str]:
    ms = [x for x in D["git"]["commits"] if "main-session decision" in x["subject"]]
    tvd = D["tv"]["tolerances_design"]
    L = ["## 8 已执行的主会话决定与已登记的实现选择", ""]
    for x in ms:
        L += [f"- **主会话决定** {c(x['sha'][:7])}（{x['utc']}）：{x['subject']}"]
    full = {k: check_row(D, k, "MLP_dense_full_float32_grads_vs_float64_value_same_inputs") for k in ("adapter_cpu", "adapter_cuda")}
    blk = {k: check_row(D, k, "MLP_dense_blocked_checkpointed_float32_grads_vs_float64_value_same_inputs") for k in ("adapter_cpu", "adapter_cuda")}
    f64 = {k: check_row(D, k, "MLP_dense_blocked_checkpointed_vs_full_grads_float64") for k in ("adapter_cpu", "adapter_cuda")}
    npairs = check_row(D, "adapter_cpu", "MLP_dense_blocked_checkpointed_vs_full_grads_float32").get("n_pairs")
    only_fp32 = [j for j, Lg in sorted(D["logs"].items())
                 if [f for p in Lg["parts"] for f in p["required_failures"]] == ["MLP_dense_blocked_checkpointed_vs_full_grads_float32"]]
    L += [f"  - 内容：fp32 的 dense-MLP 门控改为“分块 / checkpoint 的 fp32 实现 vs 同一输入的 fp64 值”（必需）；fp32 全图实现与 fp32 分块 vs 全图之差保留为诊断；"
          f"fp64 仍门控分块 vs 全图（{'、'.join(f'{k} {v['status']}' for k, v in f64.items())}）。容差未改：fp64 atol {fe(tvd['float64']['atol'], 0)} / rtol "
          f"{fe(tvd['float64']['rtol'], 0)}，fp32 atol {fe(tvd['float32']['atol'], 0)} / rtol {fe(tvd['float32']['rtol'], 0)}。",
          f"  - 证据（当前 T01）：CPU 上 fp32 全图（对 {npairs} 对求和）对 fp64 的最大误差 {fe(full['adapter_cpu']['max_abs_err'])}，分块 fp32 为 "
          f"{fe(blk['adapter_cpu']['max_abs_err'])}；CUDA 上分别为 {fe(full['adapter_cuda']['max_abs_err'])} 与 {fe(blk['adapter_cuda']['max_abs_err'])}。",
          f"  - 影响：作业 {'、'.join(only_fp32) or '—'} 的唯一必需失败由此消除；当时排队的作业 {'、'.join(x['job'] for x in jobs_without_log(D)) or '—'} "
          "因 pin 过期被取消（§7）。", ""]
    ad = (REPO / "src" / "v10" / "adapter.py").read_text()
    clip = re.search(r"clip_norm: float = ([\d.]+)\s+# (.*)", ad)
    warm = re.search(r"# (warm start of the next batch[^\n]*)", ad)
    head = D["protocol"]["head"]
    cap = D["t2"]["cuda"]["protocol"]["m1_cap_binding_tolerance"]
    L += ["已登记的实现选择（编码会话按包“常规实现选择可自行处理”所作，未经 owner 决定，列出以便审阅）：",
          f"- M1：`M1Config` 保留协议值 {fe(head['M1_gap_absolute_target'], 0)}；T02 以 tolerance {fe(cap, 0)}（迭代预算每批用满）计时，另留一行协议记录不进 BUDGET"
          f"（{commit_with(D, 'M1Config keeps the protocol value')}）。"
          + (f"训练中 W 在 batch 间热启动（src/v10/adapter.py：“{warm.group(1)}”）。" if warm else ""),
          (f"- clip：`clip_norm` = {clip.group(1)}，作用范围按 src/v10/adapter.py 注释“{clip.group(2)}”（T00 差异 D7 要求登记 W 是否进入 clip norm）。" if clip else "- clip：未找到 clip_norm 定义。"),
          f"- 可检测性检查只在 fp64 sharpened_x10 状态门控，host_init 与 fp32 行作为诊断（{commit_with(D, 'detectability checks gate only in fp64')}）。",
          f"- 计时配置：训练步与配准以 K{train_ks(D)[0]} 为主（BUDGET 单元映射 {c(D['budget']['per_unit_projection']['MF_VCS_M0']['train_config'])} 等），"
          f"{'/'.join('K' + k for k in train_ks(D)[1])} 另测；"
          "配准计时用未训练的固定模型（BUDGET assumptions：搜索成本与权重无关）。", ""]
    return L


def sec9(D) -> list[str]:
    B = D["budget"]["resources"]
    sl = D["slurm"]
    q = D["queue"]
    gpu = sl["t01_t02_gpu.sbatch"]
    cpu = sl["t01_cpu.sbatch"]
    lg = D["logs"]
    gj = [j for j in result_jobs(D) if "gpu" in lg[j]["header"]["script"]]
    gstart = lg[gj[0]]["header"]["start"] if gj else "—"
    first_cuda = min((p["started_utc"] for k, p in D["t01_parts"].items() if k.endswith("_cuda")), default="—")
    qs = B["queue_snapshot"]
    L = ["## 9 队列与优先级处理", "",
         f"- 规则（BUDGET.json `resources`）：每用户 GPU 上限 {B['per_user_gpu_cap']}，normal-QOS 提交上限 {B['normal_qos_submit_cap']}；"
         f"“{B['v10_queue_rule_2026_10_16']}”。`slurm/v10/submit.sh` 强制：V10 已有 ≥ {sl['v10_max']} 个排队/运行作业或 normal-QOS ≥ {sl['normal_max']} 时拒绝提交，"
         "并把代码 pin 写入作业；作业头在代码改变、未 pin 或脚本与 HEAD 不符时拒绝运行。",
         f"- V10 作业资源：GPU 作业 partition {gpu.get('partition')}、--exclude={gpu.get('exclude')}、--nice={gpu.get('nice')}、--time={gpu.get('time')}；"
         f"CPU 作业 partition {cpu.get('partition')}、--nice={cpu.get('nice')}、--time={cpu.get('time')}。V10 与 V9-N2 共用每用户 GPU 上限，V9 作业未取消。",
         f"- {MAIN_SESSION['attribution']}：第一阶段为让 V10 GPU 作业（--time={gpu.get('time')}）启动，主会话临时把排队中的 V9 作业 nice 最多调高 {MAIN_SESSION['nice_raise_max']}，"
         f"并在该作业启动时（{MAIN_SESSION['nice_restored_utc']}）恢复原值。可核对的时间：作业 {gj[0] if gj else '—'} 作业头开始 {gstart}，"
         f"首个 CUDA part 开始 {first_cuda}（T01 JSON）。",
         f"- BUDGET 生成时的队列快照（{qs['utc']}）：normal-QOS {qs['normal_qos_count']}，GPU 分区上运行中（全部 QOS）{qs['running_gpu_jobs']}，排队 {qs['pending']}。",
         ("- 本报告生成时：未查询队列（--no-squeue）。" if q is None else f"- 本报告生成时：squeue 失败（{q['error']}）。" if "error" in q else
          f"- 本报告生成时（{q['utc']}）：V10 作业 {q['v10']} 个；normal-QOS {q['normal']}；v9n2_* 运行 {q['v9n2_running']}、排队 {q['v9n2_pending']}。"),
         "- 本报告未提交、取消、hold 或修改任何作业。", ""]
    return L


def sec10(D) -> list[str]:
    t2c = D["t2"]["cuda"]
    env = t2c["environment"]
    B = D["budget"]
    v9 = B["v9_reference_costs"]
    g1 = v9.get("V9-G1-vcs-frac1-eps0-s901", {})
    g2 = v9.get("V9-G2-vcs-frac1-eps0-s901", {})
    mlp = B["measured"]["s_per_registration_median"].get("MLP_perm1_host")
    reg = t2c["parts"]["register"]
    pm = re.search(r"partition ([\w,]+)", D["host_map"]["sbatch_environment"]["reg_g1_sbatch_resources"])
    others = [x for x in (pm.group(1).split(",") if pm else []) if x != env.get("slurm_partition")]
    L = ["## 10 限制", "",
         f"1. **只测了一种 GPU**：{env['gpu_name']}（{env['host']}，作业 {env['slurm_job_id']}）。V9 使用的其他 GPU 分区 {'、'.join(others) or '—'} 未测；V9 的同类数字来自不同代码与时间，不是受控比较"
         f"（V9-G1 vcs 在 {g1.get('device_job_per_T00_audit')}：{fs(g1.get('s_per_registration_dev'))} s/次 DEV；T02 宿主 MLP_perm1 在 L40S：{fs(mlp)} s/次，未训练模型）。",
         f"2. **EPFL 成本是代理**：按与 BF-SHG 相同的每窗口成本计；V9-G2 vcs 在 {g2.get('device_job_per_T00_audit')} 实测 {fs(g2.get('s_per_registration_dev'))} s/次（DEV）。"
         "EPFL 的 CAL 调参未计入 BUDGET。",
         f"3. **M1 成本是上界**：BUDGET 取 {c(B['per_unit_projection']['MF_VCS_M1']['train_config'])}（迭代上限用满）；其他迭代上限另有实测（§4.6、§5.3）；"
         "M1 在 max_iter 维度上的调参未计入 BUDGET。",
         f"4. **配准计时**：未训练的固定模型，{len(reg['cores'])} 个 DEV core × {len(reg['perturbation_indices'])} 个扰动，每能量 n = "
         f"{len(reg['cores']) * len(reg['perturbation_indices'])}；假定搜索成本与权重无关（BUDGET assumptions）。",
         f"5. **训练步**：只测宿主 batch（K{train_ks(D)[0]} 为主，{'/'.join('K' + k for k in train_ks(D)[1])} 另测）；完整训练、checkpoint 读写、数据加载、CAL 选择、"
         "排队与失败重跑都未计入（BUDGET assumptions）。",
         "6. **数值头**：随机合成 simplex 特征、单场景、固定排列，不是真实 descriptor；峰值显存含进程基线；CPU 计时有未诊断的成段跳变（§4.4），CPU 不测显存。",
         "7. **T02 未测**：InfoNCE 能量、CoMIR、RuLSIF、MIND、MI/NMI、MLP perm8 配准能量、任何 EPFL 计时、TF32 关闭时的成本、其他 GPU 型号、多 GPU。",
         "8. **T00 只读元数据**：未重算图像 hash；BF-SHG 的患者 / 组织块独立性未验证（core 是可核验的最细单位）。",
         "9. **没有任务结果**：没有配准误差、成功率、S 估计误差或训练曲线；本报告的任何数字都不支持性能、加速或新颖性结论。", ""]
    return L


def sec11(D) -> list[str]:
    tracked = [f for f in git("ls-files", "--", "v10/T00", "v10/T01", "v10/T02").splitlines() if f]
    rows = []
    for f in tracked:
        p = REPO / f
        rows.append((c(f), p.stat().st_size, c(sha256_file(p)[:16]), "是" if f in PACKAGE_REQUIRED else ""))
    missing = [f for f in PACKAGE_REQUIRED if f not in tracked]
    check(not missing, f"package-required attachments missing: {missing}")
    L = ["## 11 文件清单", "", "结果文件（已提交；“包要求”= 「首批回传格式」列出的附件）：", ""]
    L += md_table(["文件", "字节", "sha256[:16]", "包要求"], rows, "lrll") + [""]
    ext = []
    for j, Lg in sorted(D["logs"].items()):
        ext.append((c(Lg["path"]), Lg["bytes"], c(Lg["sha256"][:16]), f"作业 {j} 日志"))
    for dtype, h in D["t2"]["cuda"]["parts"]["head"].items():
        if isinstance(h, dict) and "report" in h:
            ok = dtype in D["bench_raw"]
            ext.append((c(h["report"]), Path(h["report"]).stat().st_size if Path(h["report"]).exists() else "—", c(h["report_sha256"][:16]),
                        "包 benchmark 原始 JSON（" + ("sha256 一致" if ok else "不可读或不一致") + "）"))
    if D["per_step"]:
        ps = Path(D["per_step"]["path"])
        ext.append((c(str(ps)), ps.stat().st_size, c(D["per_step"]["sha256"][:16]), "T02 训练逐步记录（§6.1）"))
    L += ["本报告读取的仓库外文件（未提交；日志与原始 run 目录原样保留）：", ""]
    L += md_table(["文件", "字节", "sha256[:16]", "用途"], ext, "lrll") + [""]
    L += ["原始 part 与 run 目录：`/projects/EEG-foundation-model/yinghao/outputs_v10/T01/{parts,runs,host_replay}/`、"
          "`/projects/EEG-foundation-model/yinghao/outputs_v10/T02/runs/`；T00 审计证据：`/projects/EEG-foundation-model/yinghao/outputs_v10/T00/`。",
          f"本报告：`v10/{REPORT_DEFAULT.name}`；生成器：`{SELF_REL}`（sha256 {D['git']['self_sha256'][:16]}…，{D['git']['self_lines']} 行）。"
          "仓库公开：报告不含数据、图像块、checkpoint 或逐图标签。", ""]
    return L


def render(D) -> str:
    L = []
    for f in (sec_header, sec1, sec2, sec3, sec4, sec5, sec6, sec7, sec8, sec9, sec10, sec11):
        L += f(D)
    return "\n".join(L).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="V10 phase-1 stage report generator (read-only except the report file)")
    ap.add_argument("--output", type=Path, default=REPORT_DEFAULT)
    ap.add_argument("--no-squeue", action="store_true", help="do not query the queue (the queue line says so)")
    args = ap.parse_args()
    D = collect(args)
    text = render(D)
    args.output.write_text(text, encoding="utf-8")
    print(json.dumps({"report": str(args.output), "lines": text.count("\n"), "bytes": len(text.encode("utf-8")),
                      "head": D["git"]["head"][:7], "results_commit": D["git"]["results_commit"][:7]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
