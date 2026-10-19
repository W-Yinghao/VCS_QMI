#!/usr/bin/env python3
"""V10 T03 results report generator -> v10/T03/V10_T03_REPORT_20261017.md (Chinese).

Scope: the whole T03 grid of owner decision V10-phase1-decisions-20261017 section 9 (1212 units: CAL 252, main 600, algo_extra 120,
capacity 240) plus the 7 smoke units, as committed in
  v10/T03/smoke   results commit 90897ba (job 1043854)
  v10/T03/CAL     results commit 112b732 (job 1043862)
  v10/T03/formal  results commit 1c13749 (job 1043864)
Location: v10/tools, outside the T03 code hash (src/v10 + scripts/v10_t03) and job pin, like the aggregator.

Inputs (read from the working tree, which must equal the committed state; see the refusals):
  per unit   v10/T03/<block>/runs/<unit>/result.json and evaluator_only/evaluation.json (unit table v10/protocol/T03_UNITS.csv)
  per block  <BLOCK>_RESULTS_MANIFEST.json, gates/T03_GATES.json, checks/T03_UNITS_CHECK.json, checks/check_synthetic_seed*.json,
             the job log logs/v10_t03_{smoke,grid}_<job>.out
  smoke      T03_BUDGET.json (the smoke projection and the phase-1 order of magnitude)
  CAL        selection/T03_CAL_SELECTION.json
  formal     aggregate/evaluator_only/T03_AGGREGATE.json (v10/tools/t03_aggregate.py grid)
  protocol   v10/protocol/V10_PROTOCOL_P2_20261017.json (section t03: conditions, methods, grids, seeds, counts, unit-table pin)
  decision   v10/decisions/V10_PHASE1_DECISIONS_20261017_CN.md (every quoted rule is checked verbatim inside its section)
  git        commit ids / dates / messages of the three results commits; the T03 code hash and job pin recomputed from git objects
             at each job's commit; scripts/v10_t03/t03_common.py at the formal job's commit (identity tolerance)
Every number in the report is computed here from those records.  The cells of sections 4, 5 (decomposition), 6 and 7 that the
aggregate also holds are recomputed from the per-unit records and must equal the aggregate (relative 1e-12); the rendered values are
the aggregate's.  The CAL selection is recomputed from the CAL units' result.json traces (truth-free) and must equal
T03_CAL_SELECTION.json; every formal unit's resolved hyperparameters must equal that selection and its own-CAL checkpoint must be
the argmax of its own trace.  Hand-written: prose, term definitions, table headers and the limitation items of section 8 (whose
decision quotes are checked).  Data-dependent sentences are fixed text whose condition is checked here (the run refuses if it fails).

Refusals (SystemExit, nothing written): uncommitted or untracked changes under the record paths; a block directory changed after its
results commit or a results commit that does not name its job; tracked block files != manifest files (+ the manifest), or a sha256 /
size mismatch; unit table != protocol pin; a unit not COMPLETE / evaluation not PASS / evaluation sha256 != result record / code
identity or job != its block's; any recomputed cell, selection or checkpoint that differs; a missing decision quote.
Section 10 (corrections, 2026-10-19): appended after section 9; the header gains one line saying so.  Everything else must equal, byte
for byte, the report as first published at PUBLISHED_COMMIT (checked; refusal otherwise).  Each item quotes the published sentence
(checked verbatim against the published text), the corrected statement and its evidence; its numbers are computed from the same
records (collect_corrections), plus, for item 1, the exact population tables of the committed package generator v10_synthetic.py
(sha256 = every unit record's generator_sha256; S checked against the recorded S).  The reviewed counts of section 10 are pinned.
Output: no generation time, no live query; two runs give the same bytes.  --check renders in memory and compares with the report as
committed at HEAD (exit 1 on a difference).

    /home/infres/yinwang/CS_QMI/env/bin/python v10/tools/t03_report.py [--output PATH] [--check]
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import difflib
import hashlib
import importlib.util
import io
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[2]
SELF_REL = Path(__file__).resolve().relative_to(REPO).as_posix()
REPORT_DEFAULT = REPO / "v10" / "T03" / "V10_T03_REPORT_20261017.md"
REPORT_REL = REPORT_DEFAULT.relative_to(REPO).as_posix()
DECISION_REL = "v10/decisions/V10_PHASE1_DECISIONS_20261017_CN.md"
PROTOCOL_REL = "v10/protocol/V10_PROTOCOL_P2_20261017.json"
UNITS_REL = "v10/protocol/T03_UNITS.csv"
AGGREGATOR_REL = "v10/tools/t03_aggregate.py"
PHASE1_REPORT_REL = "v10/V10_PHASE1_STAGE_REPORT_20261017.md"
AGGREGATE_REL = "v10/T03/formal/aggregate/evaluator_only/T03_AGGREGATE.json"
SELECTION_REL = "v10/T03/CAL/selection/T03_CAL_SELECTION.json"
BUDGET_REL = "v10/T03/smoke/T03_BUDGET.json"
COMMON_REL = "scripts/v10_t03/t03_common.py"
OBJECTIVES_REL = "scripts/v10_t03/t03_objectives.py"
PACKAGE_GENERATOR_REL = "v10/package_v10_server_v1_20261016/synthetic/src/v10_synthetic.py"
# section 10 (corrections, 2026-10-19): everything before it must stay byte-identical to the report as first published
PUBLISHED_COMMIT = "0518984db074826f2976afdff4b441a7ff190b22"
CORRECTIONS_DATE = "2026-10-19"
BLOCKS = {
    "smoke": {"dir": "v10/T03/smoke", "manifest": "SMOKE_RESULTS_MANIFEST.json", "commit": "90897ba89e0f71d28ae4998df1cc39a64da35219",
              "log": "logs/v10_t03_smoke_{job}.out", "units": ("smoke",), "label": "smoke"},
    "CAL": {"dir": "v10/T03/CAL", "manifest": "CAL_RESULTS_MANIFEST.json", "commit": "112b732d1f3f9325aa502a256c16832874a4f5d0",
            "log": "logs/v10_t03_grid_{job}.out", "units": ("CAL",), "label": "CAL"},
    "formal": {"dir": "v10/T03/formal", "manifest": "FORMAL_RESULTS_MANIFEST.json", "commit": "1c137499c757275670b3664b92880d8561b18859",
               "log": "logs/v10_t03_grid_{job}.out", "units": ("main", "algo_extra", "capacity"), "label": "正式"},
}
UNIT_BLOCK = {u: b for b, s in BLOCKS.items() for u in s["units"]}
RECORD_PATHS = tuple(s["dir"] for s in BLOCKS.values()) + ("v10/protocol", "v10/decisions", AGGREGATOR_REL, PHASE1_REPORT_REL)
STAT_KEYS = ("R", "mean", "rmse", "sd", "se_mean", "mean_ci95_t", "var", "var_ci95_chi2", "rmse_ci95_bootstrap")
CELL_METRICS = {"EVAL_UJ_minus_S": lambda m: m["EVAL_UJ_minus_S"], "pop_mse": lambda m: m["pop_mse"],
                "EVAL_UJ_minus_J_pop": lambda m: m["EVAL_UJ_minus_J_pop"], "EVAL_U_minus_V": lambda m: m["EVAL_U_minus_V"],
                "perm1_var_fit": lambda m: m["perm_conditional_variance"]["fit"]["perm1_var"]}
PHASE1_QUOTE_KEY = "quotes"


# ------------------------------------------------------------------------------------------------ helpers
def fail(msg: str):
    raise SystemExit(f"[t03_report] refused: {msg}")


def check(cond, msg: str):
    if not cond:
        fail(msg)


def git_bytes(*args) -> bytes:
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True)
    if r.returncode != 0:
        fail(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()[:300]}")
    return r.stdout


def git(*args) -> str:
    return git_bytes(*args).decode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    return sha256_bytes(Path(p).read_bytes())


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def close(a, b, rel: float = 1e-12) -> bool:
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b) and all(close(x, y, rel) for x, y in zip(a, b))
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    return a == b or abs(a - b) <= rel * max(abs(a), abs(b))


def utc_of_epoch(sec: str) -> str:
    return dt.datetime.fromtimestamp(int(sec), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fe(x, sig: int = 3) -> str:
    """compact scientific with sig significant digits: -5.76e-3"""
    if x is None:
        return "—"
    x = float(x)
    if x == 0:
        return "0"
    m, e = f"{x:.{sig - 1}e}".split("e")
    return f"{m}e{int(e)}"


def fci(ci, sig: int = 3) -> str:
    return f"[{fe(ci[0], sig)}, {fe(ci[1], sig)}]"


def fsci(ci) -> str:
    """sd interval from a variance interval"""
    return f"[{fe(math.sqrt(ci[0]))}, {fe(math.sqrt(ci[1]))}]"


def fS(x) -> str:
    return "0" if float(x) == 0 else f"{float(x):.4g}"


def fpct(x, d: int = 0) -> str:
    return f"{100 * float(x):.{d}f}%"


def fh(x, d: int = 2) -> str:
    return f"{float(x):.{d}f}"


def flr(lr) -> str:
    m, e = f"{float(lr):.0e}".split("e")
    return f"{m}e{int(e)}"


def has0(ci) -> bool:
    return ci[0] <= 0 <= ci[1]


def yn(b: bool) -> str:
    return "是" if b else "否"


def frac(n: int, d: int) -> str:
    return f"{n}/{d}"


def md_table(header, rows, align) -> list[str]:
    check(len(header) == len(align), f"table header / align length {header}")
    cell = lambda v: str(v).replace("|", "\\|").replace("\n", " ")  # noqa: E731
    out = ["| " + " | ".join(cell(h) for h in header) + " |", "|" + "|".join({"l": ":---", "r": "---:", "c": ":---:"}[a] for a in align) + "|"]
    for r in rows:
        check(len(r) == len(header), f"table row length {r}")
        out.append("| " + " | ".join(cell(v) for v in r) + " |")
    return out


def rstats(values, B: int, seed: int) -> dict:
    """= v10/tools/t03_aggregate.py _stats (same operations and bootstrap draws)."""
    e = np.asarray(values, dtype=np.float64)
    R = len(e)
    out = {"R": R}
    if R == 0:
        return out
    out.update({"mean": float(e.mean()), "rmse": float(np.sqrt(np.mean(e * e)))})
    if R >= 2:
        sd = float(e.std(ddof=1))
        tq = float(sps.t.ppf(0.975, R - 1))
        var = sd * sd
        boot = np.random.default_rng(seed).integers(0, R, (B, R))
        rm = np.sqrt(np.mean(e[boot] ** 2, axis=1))
        out.update({"sd": sd, "se_mean": sd / math.sqrt(R), "mean_ci95_t": [out["mean"] - tq * sd / math.sqrt(R), out["mean"] + tq * sd / math.sqrt(R)],
                    "var": var, "var_ci95_chi2": [(R - 1) * var / float(sps.chi2.ppf(0.975, R - 1)), (R - 1) * var / float(sps.chi2.ppf(0.025, R - 1))],
                    "rmse_ci95_bootstrap": [float(np.percentile(rm, 2.5)), float(np.percentile(rm, 97.5))]})
    return out


def same_stats(mine: dict, agg: dict, label: str) -> None:
    for k in STAT_KEYS:
        if k in mine or k in agg:
            check(k in mine and k in agg and close(mine[k], agg[k]), f"{label} {k}: recomputed {mine.get(k)} != aggregate {agg.get(k)}")


def case_id(family: str, m: int, rho: float) -> str:
    rho = 0.0 if float(rho) == 0.0 else float(rho)
    return f"{family}__m{int(m):03d}__rho{format(rho, '.17g').replace('.', 'p')}"


def cond_label(c) -> str:
    return f"{c[0]} m{c[1]} ρ{c[2]:g}"


# ------------------------------------------------------------------------------------------------ decision quotes
def decision_sections(text: str) -> dict:
    """section number ('9', '9.5', ...) -> its text (heading to the next heading of the same or a higher level)."""
    heads = [(m.start(), len(m.group(1)), m.group(2)) for m in re.finditer(r"^(#{2,3}) (\d+(?:\.\d+)?) ", text, re.M)]
    out = {}
    for i, (pos, lev, num) in enumerate(heads):
        end = next((p for p, l2, _ in heads[i + 1:] if l2 <= lev), len(text))
        out[num] = text[pos:end]
    return out


class Quotes:
    def __init__(self, text: str):
        self.sections = decision_sections(text)
        self.used = []

    def __call__(self, sec: str, frag: str) -> str:
        check(sec in self.sections, f"decision section {sec} not found")
        check(frag in self.sections[sec], f"decision section {sec} does not contain the quote {frag!r}")
        self.used.append((sec, frag))
        return f"决定 §{sec}：“{frag}”"


# ------------------------------------------------------------------------------------------------ git and records
def check_git() -> dict:
    st = git("status", "--porcelain", "--untracked-files=all", "--", *RECORD_PATHS)
    check(st.strip() == "", f"uncommitted or untracked changes under the record paths:\n{st}")
    out = {}
    for b, spec in BLOCKS.items():
        last = git("log", "-1", "--format=%H", "--", spec["dir"]).strip()
        check(last == spec["commit"], f"{spec['dir']}: last commit {last[:7]} != results commit {spec['commit'][:7]} (changed after the results commit)")
        man = load_json(REPO / spec["dir"] / spec["manifest"])
        msg = git("show", "-s", "--format=%B", last)
        check(f"job {man['job_id']}" in msg, f"results commit {last[:7]} does not name job {man['job_id']}")
        out[b] = {"commit": last, "utc": utc_of_epoch(git("show", "-s", "--format=%ct", last).strip()), "job": str(man["job_id"]),
                  "subject": msg.splitlines()[0]}
    return out


def check_manifest(b: str) -> dict:
    spec = BLOCKS[b]
    d = REPO / spec["dir"]
    man = load_json(d / spec["manifest"])
    prefix = spec["dir"] + "/"
    tracked = {p[len(prefix):] for p in git("ls-files", "--", spec["dir"]).splitlines() if p}
    listed = [f["path"] for f in man["files"]]
    check(len(set(listed)) == len(listed), f"{b}: duplicate manifest paths")
    check(tracked == set(listed) | {spec["manifest"]}, f"{b}: tracked files != manifest files + manifest "
          f"(only tracked: {sorted(tracked - set(listed) - {spec['manifest']})[:5]}, only listed: {sorted(set(listed) - tracked)[:5]})")
    nbytes = 0
    for f in man["files"]:
        raw = (d / f["path"]).read_bytes()
        check(sha256_bytes(raw) == f["sha256"] and len(raw) == f["bytes"], f"{b}: {f['path']} sha256 / size != manifest")
        nbytes += len(raw)
    leak = man["leak_scan"]
    check(leak["status"] == "PASS" and not leak["leaks"], f"{b}: leak scan not PASS")
    out = {"files": len(listed), "bytes": nbytes, "job": str(man["job_id"]), "leak_needles": leak["needles"],
           "leak_found_inside": leak["needles_found_in_evaluator_only"], "leaks": len(leak["leaks"]),
           "kept_outside": len(man.get("kept_outside_repository", [])), "acceptance": man.get("acceptance")}
    if man.get("acceptance"):
        check(man["acceptance"]["status_ok"] and man["acceptance"]["denials"] == 0, f"{b}: manifest acceptance not ok")
    return out


def tree_hash_at(commit: str, paths) -> str:
    """= scripts/v10_t03/t03_common.tree_hash at a commit, from git objects."""
    files = sorted((f for f in git("ls-tree", "-r", "--name-only", commit, "--", *paths).splitlines() if f), key=lambda s: s.encode())
    check(files, f"no files under {paths} at {commit[:7]}")
    lines = "".join(f"{sha256_bytes(git_bytes('cat-file', 'blob', f'{commit}:{f}'))}  {f}\n" for f in files)
    return sha256_bytes(lines.encode())[:16]


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    head = dict(t.split("=", 1) for t in lines[0].split() if "=" in t)
    ends = [ln for ln in lines if ln.startswith("[v10_t03] t03_")]
    check(len(ends) == 1, f"{path.name}: expected one end line")
    tail = dict(t.split("=", 1) for t in ends[0].split() if "=" in t)
    tests = re.search(r"^(\d+) passed", text, re.M)
    return {"head": head, "tail": tail, "tests_passed": int(tests.group(1)) if tests else None}


def load_units(proto: dict) -> tuple[dict, list]:
    raw = (REPO / UNITS_REL).read_bytes()
    check(sha256_bytes(raw) == proto["units_file"]["sha256"], "unit table sha256 != protocol P2 t03.units_file.sha256")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    cnt = collections.Counter(r["block"] for r in rows)
    pc = proto["counts"]
    check(cnt["CAL"] == pc["CAL"] and cnt["main"] == pc["main"] and cnt["algo_extra"] == pc["conditional_algorithm_extra"]
          and cnt["capacity"] == pc["capacity"] and sum(cnt[b] for b in ("CAL", "main", "algo_extra", "capacity")) == pc["total_excluding_smoke_and_followup_baselines"],
          f"unit table block counts {dict(cnt)} != protocol counts")
    units = {}
    steps = int(proto["training"]["steps"])
    for r in rows:
        uid = r["unit_id"]
        d = REPO / BLOCKS[UNIT_BLOCK[r["block"]]]["dir"] / "runs" / uid
        res = load_json(d / "result.json")
        evp = d / "evaluator_only" / "evaluation.json"
        ev = load_json(evp)
        check(res["unit_id"] == uid == ev["unit_id"], f"{uid}: unit id mismatch")
        check(res["status"] == "COMPLETE" and res["steps_completed"] == steps, f"{uid}: status {res['status']} / steps {res.get('steps_completed')}")
        check(ev["status"] == "PASS" and all(v == "PASS" for v in ev["checks"].values()), f"{uid}: evaluation not PASS")
        check(res["evaluation"]["sha256"] == sha256_file(evp), f"{uid}: evaluation.json sha256 != result record")
        u = {"row": r, "res": res, "ev": ev, "block": r["block"], "family": r["family"], "m": int(r["m"]), "rho": float(r["rho"]),
             "method": r["method"], "K": None if r["K"] == "NA" else int(r["K"]), "data": int(r["data_root_seed"]), "algo": int(r["algo_seed"])}
        u["cond"] = (u["family"], u["m"], u["rho"])
        check(res["case_id"] == case_id(*u["cond"]) == ev["case_id"], f"{uid}: case id mismatch")
        u["sel"] = ev["selected_metrics"]
        units[uid] = u
    return units, rows


# ------------------------------------------------------------------------------------------------ collection
def collect() -> dict:
    D = {"git": check_git()}
    proto = load_json(REPO / PROTOCOL_REL)["t03"]
    D["proto"] = proto
    D["conds"] = [(c["family"], int(c["m"]), float(c["rho"])) for c in proto["conditions"]]
    D["methods"] = list(proto["methods"])
    D["mf_methods"] = list(proto["capacity"]["methods"])
    D["lrs"] = [float(x) for x in proto["training"]["learning_rates"]]
    D["caps"] = [int(x) for x in proto["training"]["M1_caps"]]
    D["steps"] = list(range(int(proto["training"]["checkpoint_every"]), int(proto["training"]["steps"]) + 1, int(proto["training"]["checkpoint_every"])))
    check(len(D["steps"]) == int(proto["training"]["checkpoints"]), "checkpoint count")
    D["data_seeds"] = [int(s) for s in proto["replicates"]["main"]["data_root_seeds"]]
    D["algo_main"] = int(proto["replicates"]["main"]["algo_seed"])
    D["algo_extra"] = [int(s) for s in proto["replicates"]["conditional_algorithm"]["algo_seeds"]]
    D["algo_data"] = int(proto["replicates"]["conditional_algorithm"]["data_root_seed"])
    D["cap_conds"] = [(c["family"], int(c["m"]), float(c["rho"])) for c in proto["capacity"]["conditions"]]
    D["cap_K"] = [int(k) for k in proto["capacity"]["K"]]
    D["K_main"] = int(proto["MF"]["K_main"])
    D["tuning_seed"] = int(proto["tuning_suite"]["data_root_seed"])
    D["n_obs"] = proto["observations_per_condition"]
    D["manifests"] = {b: check_manifest(b) for b in BLOCKS}
    units, rows = load_units(proto)
    D["units"], D["rows"] = units, rows
    D["n_by_block"] = dict(collections.Counter(r["block"] for r in rows))
    collect_jobs(D)
    collect_gates(D)
    collect_selection(D)
    collect_cells(D)
    collect_diagnostics(D)
    collect_cost(D)
    collect_corrections(D)
    D["quotes_text"] = (REPO / DECISION_REL).read_text(encoding="utf-8")
    return D


def collect_jobs(D) -> None:
    jobs = {}
    for b, spec in BLOCKS.items():
        us = [u for u in D["units"].values() if UNIT_BLOCK[u["block"]] == b]
        ids = {json.dumps({k: u["res"]["provenance"]["identity"][k] for k in ("t03_code_hash", "t03_job_pin", "git_commit", "git_dirty",
                                                                               "t03_code_paths", "t03_pin_paths", "git_branch")}, sort_keys=True) for u in us}
        check(len(ids) == 1, f"{b}: more than one code identity among its units")
        ident = json.loads(ids.pop())
        check(ident["git_dirty"] == "", f"{b}: units ran on a dirty tree")
        envs = {(u["res"]["provenance"]["environment"]["slurm_job_id"], u["res"]["provenance"]["environment"]["slurm_partition"],
                 u["res"]["provenance"]["environment"]["slurm_nodelist"], u["res"]["provenance"]["environment"]["torch_threads"]) for u in us}
        check(len(envs) == 1, f"{b}: units from more than one job / node / thread count")
        job, part, node, threads = envs.pop()
        check(job == D["git"][b]["job"] == D["manifests"][b]["job"], f"{b}: job id of units / manifest / results commit differ")
        L = parse_log(REPO / spec["dir"] / spec["log"].format(job=job))
        h, t = L["head"], L["tail"]
        check(h["job"] == job and h["commit"] == ident["git_commit"] and h["t03_code"] == ident["t03_code_hash"] and h["pin"] == ident["t03_job_pin"]
              and h["expect"] == h["pin"] and h["dirty_files"] == "0" and h["partition"] == part and h["node"] == node,
              f"{b}: job log header disagrees with the unit records")
        rcs = {k: v for k, v in t.items() if k.endswith("_rc")}
        check(all(v in ("0", "NOT_RUN") for v in rcs.values()), f"{b}: job end line rc {rcs}")
        code_at = tree_hash_at(ident["git_commit"], ident["t03_code_paths"])
        pin_at = tree_hash_at(ident["git_commit"], ident["t03_pin_paths"])
        check(code_at == ident["t03_code_hash"] and pin_at == ident["t03_job_pin"], f"{b}: code hash / pin recomputed at {ident['git_commit'][:7]} differ")
        denials = sum(len(u["res"]["truth_barrier"]["denials_during_unit"]) for u in us)
        starts = sorted(u["res"]["started_utc"] for u in us)
        ends = sorted(u["res"]["finished_utc"] for u in us)
        wall = int(t["wall"].rstrip("s"))
        jobs[b] = {"job": job, "partition": part, "node": node, "threads": threads, "cpus": int(h["cpus"]), "mode": h["mode"], "start": h["start"],
                   "end": t["end"], "wall_s": wall, "run_commit": ident["git_commit"], "code": ident["t03_code_hash"], "pin": ident["t03_job_pin"],
                   "code_paths": ident["t03_code_paths"], "pin_paths": ident["t03_pin_paths"], "units": len(us), "denials": denials,
                   "first_unit_start": starts[0], "last_unit_end": ends[-1], "tests_passed": L["tests_passed"], "rcs": rcs,
                   "layout": t.get("layout"), "branch": ident["git_branch"]}
    D["jobs"] = jobs
    codes = {j["code"] for j in jobs.values()}
    check(len(codes) == 1, "T03 code hash differs between the blocks")
    check(jobs["CAL"]["pin"] == jobs["formal"]["pin"], "CAL / formal job pins differ")
    check(len({jb["branch"] for jb in jobs.values()}) == 1, "blocks ran on different branches")
    # the formal job ran at the CAL results commit (the selection was committed before the formal units ran)
    check(jobs["formal"]["run_commit"] == D["git"]["CAL"]["commit"], "formal job did not run at the CAL results commit")
    check(jobs["CAL"]["run_commit"] == D["git"]["smoke"]["commit"], "CAL job did not run at the smoke results commit")
    common = git("show", f"{jobs['formal']['run_commit']}:{COMMON_REL}")
    m = re.search(r"^IDENTITY_ATOL = ([0-9.e+-]+)", common, re.M)
    check(m, "IDENTITY_ATOL not found in t03_common.py")
    D["identity_atol"] = float(m.group(1))
    # pin-path changes between the smoke and the CAL run commits (the code-hash paths must not change); none from CAL to formal
    a, b, f = jobs["smoke"]["run_commit"], jobs["CAL"]["run_commit"], jobs["formal"]["run_commit"]
    pc = {"from": a, "to": b,
          "commits": [ln.split(" ", 1) for ln in git("log", "--format=%H %s", f"{a}..{b}", "--", *jobs["CAL"]["pin_paths"]).splitlines() if ln],
          "files": [ln.split("\t") for ln in git("diff", "--name-status", a, b, "--", *jobs["CAL"]["pin_paths"]).splitlines() if ln]}
    check(not git("diff", "--name-only", a, b, "--", *jobs["CAL"]["code_paths"]).strip(), "code-hash paths changed between the smoke and CAL run commits")
    check(not git("diff", "--name-only", b, f, "--", *jobs["formal"]["pin_paths"]).strip(), "pin paths changed between the CAL and formal run commits")
    check(pc["commits"] and pc["files"], "smoke / CAL pins differ without a pin-path commit")
    D["pin_change"] = pc
    # truth barrier scope: every unit record names the same scope; the CAL group selection runs inside a barrier block (runner code)
    scopes = {u["res"]["truth_barrier"].get("active_during") for u in D["units"].values()}
    check(len(scopes) == 1 and None not in scopes, "truth barrier scope differs between unit records")
    D["barrier_scope"] = scopes.pop()
    # method structure as defined in the terms table: optimiser groups, W rule, negatives per step
    expect = {"MF_VCS_perm1": (("feature_maps", "W"), 1), "MF_VCS_M0": (("feature_maps", "W"), None), "MF_VCS_M1": (("feature_maps",), None),
              "MLP_VCS_perm1": (("mlp_critic",), 1), "MLP_VCS_perm8": (("mlp_critic",), 8)}
    for u in D["units"].values():
        groups, neg = expect[u["method"]]
        r = u["res"]
        check(tuple(g["name"] for g in r["optimizer"]["groups"]) == groups, f"{u['row']['unit_id']}: optimiser groups")
        if u["method"] == "MF_VCS_M1":
            check(r["optimizer"]["W_projection"].startswith("solved head"), f"{u['row']['unit_id']}: M1 W rule")
        if u["method"].startswith("MLP_"):
            check(r["critic"]["architecture"]["negatives_per_step"] == neg and r["critic"]["architecture"]["output"] == "T = tanh f", f"{u['row']['unit_id']}: MLP negatives")
        elif neg:
            check(r["derangements"]["perm"]["draws"] == int(D["proto"]["training"]["steps"]) and not r["derangements"]["perm8_extra"], f"{u['row']['unit_id']}: MF perm1 draws")
        else:
            check(not (r.get("derangements") or {}).get("perm"), f"{u['row']['unit_id']}: exact-moment method drew permutations")
    run_py = git("show", f"{b}:scripts/v10_t03/t03_run.py")
    check('with C.truth_barrier("CAL group selection")' in run_py, "CAL group selection outside a truth-barrier block at the CAL run commit")


def collect_gates(D) -> None:
    g = {}
    for b, spec in BLOCKS.items():
        d = REPO / spec["dir"]
        gates = load_json(d / "gates" / "T03_GATES.json")
        check(gates["identity"]["git_commit"] == D["jobs"][b]["run_commit"], f"{b}: gate record from another commit")
        chk = sorted((d / "checks").glob("check_synthetic_seed*.json"))
        checker = [load_json(p) for p in chk]
        uc = load_json(d / "checks" / "T03_UNITS_CHECK.json")
        g[b] = {"status": gates["status"], "gates": {k: v["status"] for k, v in gates["gates"].items()}, "record": gates,
                "checker": [(c["suite"].rsplit("seed", 1)[1], c["n_checks"], c["n_checks_passed"], c["status"]) for c in checker],
                "units_check": uc}
        check(gates["status"] == "PASS" and all(v == "PASS" for v in g[b]["gates"].values()), f"{b}: gates not PASS")
        check(all(c[1] == c[2] and c[3] == "PASS" for c in g[b]["checker"]), f"{b}: package checker not PASS")
        check(uc["status"] == "PASS" and uc["bytes_identical"] and uc["n_field_differences"] == 0 and uc["rows_file"] == len(D["rows"]),
              f"{b}: unit table check not PASS")
    seeds = {b: sorted(int(c[0]) for c in g[b]["checker"]) for b in g}
    check(seeds["formal"] == D["data_seeds"] and seeds["CAL"] == [D["tuning_seed"]], "checker suites != protocol seeds")
    G = g["formal"]["record"]["gates"]
    g1 = G["G1_enumeration"]["rows"]
    g2 = G["G2_sampler_law"]["rows"]
    D["gate_facts"] = {
        "G1_rows": len(g1), "G1_n": sorted({r["n"] for r in g1}), "G1_critics": sorted({r["critic"] for r in g1}),
        "G1_value_err": max(r["derangement_mean_value_err"] for r in g1), "G1_grad_err": max(r["derangement_mean_grad_err"] for r in g1),
        "G1_tol": G["G1_enumeration"]["tolerance"],
        "G2_n": [r["n"] for r in g2], "G2_draws": [r["draws"] for r in g2], "G2_p_min": min(r["p_value"] for r in g2),
        "G2_z_max": max(r["max_abs_z_offdiag_marginal"] for r in g2), "G2_fixed": sum(r["fixed_points"] for r in g2),
        "G2_pre": G["G2_sampler_law"]["pre_set"],
        "G3": {k: G["G3_rejection_rate"][k] for k in ("n", "draws", "mean_attempts", "expected", "z")},
        "G4_err": max(r["abs_err"] for r in G["G4_perm8_mean"]["rows"]),
        "G5_err": max(max(r["err_U"], r["err_V"]) for r in G["G5_table_U"]["rows"]),
    }
    D["gates"] = g


def selection_key(u) -> tuple:
    return (u["cond"], u["method"])


def collect_selection(D) -> None:
    sel = load_json(REPO / SELECTION_REL)
    check(sel["n_groups"] == sel["n_selected"] == len(D["conds"]) * len(D["methods"]) == len(sel["groups"]), "CAL selection group count")
    check(sel["identity"]["git_commit"] == D["jobs"]["CAL"]["run_commit"] and sel["identity"]["t03_code_hash"] == D["jobs"]["CAL"]["code"],
          "CAL selection identity != CAL job")
    cal = collections.defaultdict(list)
    for u in D["units"].values():
        if u["block"] == "CAL":
            cal[selection_key(u)].append(u)
    lrs, caps = D["lrs"], D["caps"]
    groups = {}
    for g in sel["groups"]:
        key = ((g["family"], int(g["m"]), float(g["rho"])), g["method"])
        check(g["status"] == "SELECTED" and g["case_id"] == case_id(*key[0]), f"selection group {key}: status / case id")
        cands = []
        for u in cal[key]:
            c = u["res"]["config"]
            li = lrs.index(float(c["lr"]))
            ordinal = li * len(caps) + caps.index(int(c["cap"])) if key[1] == "MF_VCS_M1" else li
            cands.append((ordinal, u))
        check(sorted(o for o, _ in cands) == list(range(len(lrs) * (len(caps) if key[1] == "MF_VCS_M1" else 1))), f"{key}: candidate set")
        pts = [(o, int(ck["step"]), ck["CAL_UJ"]) for o, u in cands for ck in u["res"]["checkpoints"]]
        check(all(isinstance(v, float) and math.isfinite(v) for _, _, v in pts), f"{key}: non-finite CAL U-J")
        best = max(v for _, _, v in pts)
        tied = sorted((s, o) for o, s, v in pts if v == best)
        step, ordinal = tied[0]
        win = next(u for o, u in cands if o == ordinal)
        s = g["selected"]
        check(s["ordinal"] == ordinal and s["unit_id"] == win["row"]["unit_id"] and s["step"] == step and s["CAL_UJ"] == best
              and float(s["lr"]) == float(win["res"]["config"]["lr"]) and s["cap"] == win["res"]["config"]["cap"]
              and [(t["step"], t["ordinal"]) for t in g["tied"]] == tied, f"{key}: CAL selection recomputed from the CAL traces differs")
        groups[key] = {"lr": float(s["lr"]), "cap": s["cap"], "step": step, "CAL_UJ": best, "ordinal": ordinal, "n_tied": len(tied),
                       "n_candidates": len(cands), "selected": s}
    D["selection"] = groups
    D["selection_sha256"] = sha256_file(REPO / SELECTION_REL)
    # formal units: hyperparameters = the selection of the same (condition, method); own-CAL checkpoint = argmax of own trace
    own = collections.Counter()
    n_formal = 0
    for u in D["units"].values():
        if u["block"] not in ("main", "algo_extra", "capacity"):
            continue
        n_formal += 1
        gsel = groups[selection_key(u)]
        hr = u["res"]["hyperparameter_resolution"]["group_selection"]["selected"]
        c = u["res"]["config"]
        check(hr == gsel["selected"] and float(c["lr"]) == gsel["lr"] and c["cap"] == gsel["cap"], f"{u['row']['unit_id']}: hyperparameters != CAL selection")
        trace = [(int(ck["step"]), ck["CAL_UJ"]) for ck in u["res"]["checkpoints"]]
        best = max(v for _, v in trace)
        step = min(s for s, v in trace if v == best)
        check(u["res"]["selection"]["own_cal"]["step"] == step == u["ev"]["selected"]["step"], f"{u['row']['unit_id']}: own-CAL checkpoint differs")
        if u["block"] == "main":
            own[(u["method"], step)] += 1
    D["own_steps_main"] = own
    D["n_formal"] = n_formal


def cell_values(units: list[dict], name: str) -> list[float]:
    f = CELL_METRICS[name]
    return [f(u["sel"]) for u in units]


def collect_cells(D) -> None:
    agg = load_json(REPO / AGGREGATE_REL)
    D["aggregate_sha256"] = sha256_file(REPO / AGGREGATE_REL)
    D["aggregator_sha256"] = agg["aggregator_sha256"]
    check(agg["aggregator_sha256"] == sha256_file(REPO / AGGREGATOR_REL), "aggregate written by another aggregator version than the committed one")
    bs = {json.dumps(c["EVAL_UJ_minus_S"]["bootstrap"], sort_keys=True) for c in agg["main"]}
    check(len(bs) == 1, "bootstrap constants differ between cells")
    boot = json.loads(bs.pop())
    B, seed = int(boot["B"]), int(boot["seed"])
    D["boot"] = {"B": B, "seed": seed}
    st = collections.Counter(u["block"] for u in D["units"].values())
    check(agg["status"] == {b: {"COMPLETE": n} for b, n in st.items()} or all(agg["status"][b] == {"COMPLETE": st[b]} for b in st),
          "aggregate status counts != unit records")

    def mk(units: list[dict], rep: str, agg_cell: dict, label: str) -> dict:
        units = sorted(units, key=lambda u: u[rep])
        check([u["row"]["unit_id"] for u in sorted(units, key=lambda u: u["row"]["unit_id"])] == agg_cell["unit_ids"], f"{label}: unit ids != aggregate")
        Ss = {u["ev"]["truth"]["S"] for u in units}
        check(len(Ss) == 1 and close(Ss.copy().pop(), agg_cell["S"]), f"{label}: S differs between replicates or from the aggregate")
        check(agg_cell["n_complete"] == agg_cell["expected"] == len(units), f"{label}: incomplete")
        cell = {"S": agg_cell["S"], "units": units, "reps": [u[rep] for u in units]}
        for name in CELL_METRICS:
            mine = rstats(cell_values(units, name), B, seed)
            same_stats(mine, agg_cell[name], f"{label} {name}")
            cell[name] = agg_cell[name]
        gaps = [u["sel"]["head_gap"]["fw_gap_W"] for u in units if u["sel"]["head_gap"]["status"] == "MEASURED"]
        if gaps:
            check(len(gaps) == len(units), f"{label}: partial head gaps")
            same_stats(rstats(gaps, B, seed), agg_cell["head_fw_gap_fit"], f"{label} head gap")
            cell["gap"] = agg_cell["head_fw_gap_fit"]
            cell["gap_rel_median"] = float(np.median([u["sel"]["head_gap"]["gap_relative_to_zero"] for u in units]))
            cell["polish_lower_mean"] = float(np.mean([u["sel"]["head_gap"]["fp64_polish"]["suboptimality_lower_bound"] for u in units]))
        else:
            check("head_fw_gap_fit" not in agg_cell, f"{label}: aggregate has a head gap for a non-MF cell")
            cell["gap"] = None
        check(agg_cell["identity_all_pass"] and all(u["sel"]["identity_check"] == "PASS" for u in units), f"{label}: identity check")
        cell["n_neg"] = sum(u["sel"]["EVAL_UJ"] < 0 for u in units)
        cell["perm_eval_var_mean"] = float(np.mean([u["sel"]["perm_conditional_variance"]["eval"]["perm1_var"] for u in units]))
        cell["err"] = [u["sel"]["EVAL_UJ_minus_S"] for u in units]
        cell["pop"] = [u["sel"]["pop_mse"] for u in units]
        return cell

    by = collections.defaultdict(list)
    for u in D["units"].values():
        by[u["block"]].append(u)
    # main
    amain = {(c["case_id"], c["method"]): c for c in agg["main"]}
    check(len(amain) == len(D["conds"]) * len(D["methods"]), "aggregate main cell count")
    g = collections.defaultdict(list)
    for u in by["main"]:
        g[selection_key(u)].append(u)
    main = {}
    for cnd in D["conds"]:
        for meth in D["methods"]:
            us = g[(cnd, meth)]
            check(sorted(u["data"] for u in us) == D["data_seeds"] and {u["algo"] for u in us} == {D["algo_main"]}, f"main {cnd} {meth}: replicates")
            main[(cnd, meth)] = mk(us, "data", amain[(case_id(*cnd), meth)], f"main {cnd} {meth}")
    D["main"] = main
    # every method on a data root seed sees the same observations (FIT / CAL / EVAL sample hashes)
    obs = collections.defaultdict(set)
    for u in by["main"] + by["algo_extra"] + by["capacity"]:
        obs[(u["data"],) + u["cond"]].add((u["res"]["data"]["sample_sha256"]["fit"], u["res"]["data"]["sample_sha256"]["cal"], u["ev"]["sample_sha256"]["eval"]))
    check(all(len(v) == 1 for v in obs.values()) and len({v.copy().pop() for v in obs.values()}) == len(obs), "methods of one data root see different observations")
    D["obs_sets"] = len(obs)
    # algo
    aalgo = {(c["case_id"], c["method"]): c for c in agg["algo"]}
    check(len(aalgo) == len(main), "aggregate algo cell count")
    g = collections.defaultdict(list)
    for u in by["main"]:
        if u["data"] == D["algo_data"] and u["algo"] == D["algo_main"]:
            g[selection_key(u)].append(u)
    for u in by["algo_extra"]:
        check(u["data"] == D["algo_data"], "algo_extra unit on another data root")
        g[selection_key(u)].append(u)
    algo = {}
    for key in main:
        us = g[key]
        check(sorted(u["algo"] for u in us) == [D["algo_main"]] + D["algo_extra"], f"algo {key}: seeds")
        ac = aalgo[(case_id(*key[0]), key[1])]
        check(ac["data_root_seed"] == D["algo_data"] and ac["replicate_unit"] == "algorithm seed (fixed data)", f"algo {key}: aggregate unit")
        algo[key] = mk(us, "algo", ac, f"algo {key}")
    D["algo"] = algo
    # capacity
    acap = {(c["case_id"], c["method"], int(c["K"])): c for c in agg["capacity"]}
    check(len(acap) == len(D["cap_conds"]) * len(D["mf_methods"]) * (len(D["cap_K"]) + 1), "aggregate capacity cell count")
    g = collections.defaultdict(list)
    for u in by["capacity"]:
        g[(u["cond"], u["method"], u["K"])].append(u)
    cap = {}
    for cnd in D["cap_conds"]:
        for meth in D["mf_methods"]:
            for K in sorted(D["cap_K"] + [D["K_main"]]):
                us = main[(cnd, meth)]["units"] if K == D["K_main"] else g[(cnd, meth, K)]
                check(sorted(u["data"] for u in us) == D["data_seeds"] and all(u["K"] == K for u in us), f"capacity {cnd} {meth} K{K}: replicates")
                cap[(cnd, meth, K)] = mk(us, "data", acap[(case_id(*cnd), meth, K)], f"capacity {cnd} {meth} K{K}")
    D["capacity"] = cap
    # paired differences per data replicate (main)
    pairs = []
    for p in agg["paired_main"]:
        if (p["a"], p["b"]) not in pairs:
            pairs.append((p["a"], p["b"]))
    D["pairs"] = pairs
    apair = {(p["case_id"], p["a"], p["b"]): p for p in agg["paired_main"]}
    check(len(apair) == len(D["conds"]) * len(pairs), "aggregate paired cell count")
    paired = {}
    for cnd in D["conds"]:
        for a, b in pairs:
            ua, ub = main[(cnd, a)]["units"], main[(cnd, b)]["units"]
            check([u["data"] for u in ua] == [u["data"] for u in ub], "pairing order")
            dsq = [x["sel"]["EVAL_UJ_minus_S"] ** 2 - y["sel"]["EVAL_UJ_minus_S"] ** 2 for x, y in zip(ua, ub)]
            dmse = [x["sel"]["pop_mse"] - y["sel"]["pop_mse"] for x, y in zip(ua, ub)]
            ap = apair[(case_id(*cnd), a, b)]
            check(ap["n_pairs"] == len(dsq) == len(D["data_seeds"]), f"pair {cnd} {a}-{b}: n")
            same_stats(rstats(dsq, B, seed), ap["diff_sq_err_EVAL_UJ"], f"pair {cnd} {a}-{b} sq")
            same_stats(rstats(dmse, B, seed), ap["diff_pop_mse"], f"pair {cnd} {a}-{b} mse")
            paired[(cnd, a, b)] = {"sq": ap["diff_sq_err_EVAL_UJ"], "mse": ap["diff_pop_mse"]}
    D["paired"] = paired
    # capacity: paired K vs K16 (generator-computed; not in the aggregate)
    capd = {}
    for cnd in D["cap_conds"]:
        for meth in D["mf_methods"]:
            base = cap[(cnd, meth, D["K_main"])]["units"]
            for K in D["cap_K"]:
                other = cap[(cnd, meth, K)]["units"]
                check([u["data"] for u in other] == [u["data"] for u in base], "capacity pairing order")
                capd[(cnd, meth, K)] = rstats([x["sel"]["EVAL_UJ_minus_S"] ** 2 - y["sel"]["EVAL_UJ_minus_S"] ** 2 for x, y in zip(other, base)], B, seed)
    D["cap_paired"] = capd
    # saturated FIT table (deduplicated per (data root, condition) over every unit; all copies must agree)
    sat, nunits, conflicts = {}, collections.Counter(), 0
    for u in D["units"].values():
        k = (u["data"],) + u["cond"]
        rec = {m: {f: u["ev"]["saturated_fit_table"][m][f] for f in ("pop_mse", "EVAL_UJ", "FIT_UJ", "CAL_UJ", "J_pop")} for m in ("U", "V")}
        nunits[k] += 1
        if k in sat and sat[k] != rec:
            conflicts += 1
        sat.setdefault(k, rec)
    check(conflicts == 0 and agg["saturated"]["conflicts"] == [], "saturated FIT table copies disagree")
    asat = {(r["data_root_seed"], r["case_id"]): r for r in agg["saturated"]["rows"]}
    check(len(asat) == len(sat), "aggregate saturated row count")
    for k, rec in sat.items():
        ar = asat[(k[0], case_id(*k[1:]))]
        check(all(close(rec[m][f], ar[m][f]) for m in ("U", "V") for f in rec[m]), f"saturated {k} != aggregate")
    satc = {}
    for cnd in D["conds"]:
        S = main[(cnd, D["methods"][0])]["S"]
        rows = [sat[(s,) + cnd] for s in D["data_seeds"]]
        satc[cnd] = {"S": S, "fit_minus_S": rstats([r["U"]["FIT_UJ"] - S for r in rows], B, seed),
                     "eval_err": rstats([r["U"]["EVAL_UJ"] - S for r in rows], B, seed), "pop": rstats([r["U"]["pop_mse"] for r in rows], B, seed),
                     "eval_err_V": rstats([r["V"]["EVAL_UJ"] - S for r in rows], B, seed)}
    D["sat"] = satc
    D["sat_keys"] = len(sat)
    D["sat_units"] = sum(nunits.values())
    D["sat_shared_max"] = max(nunits.values())
    D["sat_dominates"] = sum(u["ev"]["checks"]["saturated_FIT_U_optimum_dominates_checkpoints"] == "PASS" for u in D["units"].values())
    # cost cells of the aggregate
    acost = {(c["block"], c["method"]): c for c in agg["cost"]}
    g = collections.defaultdict(list)
    for u in D["units"].values():
        g[(u["block"], u["method"])].append(u["res"]["cost"]["wall_seconds_unit_total"])
    check(set(g) == set(acost), "aggregate cost keys")
    for k, v in g.items():
        check(acost[k]["units"] == len(v) and close(acost[k]["wall_seconds_sum"], float(np.sum(v)), 1e-9), f"cost {k} != aggregate")


def collect_diagnostics(D) -> None:
    units = list(D["units"].values())
    # identity S - J_pop(T) = E_M(T - eta)^2 over every evaluated table (selected, every checkpoint, saturated U / V)
    res = []
    for u in units:
        ev = u["ev"]
        res.append(ev["selected_metrics"]["identity_residual"])
        res.extend(r["identity_residual"] for r in ev["trajectory_diagnostic"]["rows"])
        res.extend(ev["saturated_fit_table"][m]["identity_residual"] for m in ("U", "V"))
        check(ev["selected_metrics"]["identity_check"] == "PASS", "identity")
    D["identity"] = {"n": len(res), "max_abs": max(abs(r) for r in res), "units": len(units)}
    check(D["identity"]["max_abs"] <= D["identity_atol"], "identity residual above the tolerance")
    # negative EVAL U-J (kept)
    neg = collections.Counter()
    tot = collections.Counter()
    for u in units:
        if u["block"] == "main":
            k = "S0" if u["ev"]["truth"]["S"] == 0 else "S+"
            tot[k] += 1
            neg[k] += u["sel"]["EVAL_UJ"] < 0
    D["neg"] = {"S0": (neg["S0"], tot["S0"]), "S+": (neg["S+"], tot["S+"])}
    # no clipping: the recorded error is EVAL U-J - S exactly (the selected checkpoint and the saturated tables)
    for u in units:
        S = u["ev"]["truth"]["S"]
        check(u["sel"]["EVAL_UJ_minus_S"] == u["sel"]["EVAL_UJ"] - S and u["sel"]["EVAL_UJ_minus_J_pop"] == u["sel"]["EVAL_UJ"] - u["sel"]["J_pop"]
              and u["sel"]["S"] == S, f"{u['row']['unit_id']}: EVAL error is not EVAL U-J - S")
    # permutation z of the perm1 mean vs the exact U-J (formal units, FIT and EVAL)
    z = [u["sel"]["perm_conditional_variance"][r]["z_mean_vs_exact"] for u in units if u["block"] in ("main", "algo_extra", "capacity") for r in ("fit", "eval")]
    check(all(x is not None for x in z), "missing permutation z")
    R = {u["sel"]["perm_conditional_variance"][r]["R"] for u in units if u["block"] in ("main", "algo_extra", "capacity") for r in ("fit", "eval")}
    check(len(R) == 1, "permutation R differs")
    za = np.asarray(z, dtype=np.float64)
    D["perm_z"] = {"n": len(z), "R": R.pop(), "mean": float(za.mean()), "sd": float(za.std(ddof=1)), "max_abs": float(np.abs(za).max()),
                   "n_gt3": int(np.sum(np.abs(za) > 3)), "expected_gt3": float(len(z) * 2 * sps.norm.sf(3.0))}
    # derangement draws of training (every perm unit) vs the expected attempts
    att = [(u["res"]["derangements"]["perm"]["attempts"], u["res"]["derangements"]["perm"]["draws"]) for u in units
           if u["res"].get("derangements") and u["res"]["derangements"].get("perm")]
    ex = [(u["res"]["derangements"]["perm8_extra"]["attempts"], u["res"]["derangements"]["perm8_extra"]["draws"]) for u in units
          if u["res"].get("derangements") and u["res"]["derangements"].get("perm8_extra")]
    a = sum(x for x, _ in att) + sum(x for x, _ in ex)
    d = sum(y for _, y in att) + sum(y for _, y in ex)
    D["derangements"] = {"units": len(att), "draws": d, "attempts": a, "mean_attempts": a / d}
    # U / V on the same frozen critic (main units)
    uv = {}
    for cnd in D["conds"]:
        vals = [abs(u["sel"]["EVAL_U_minus_V"]) for meth in D["methods"] for u in D["main"][(cnd, meth)]["units"]]
        ratio = max(max(abs(u["sel"]["EVAL_U_minus_V"]) for u in D["main"][(cnd, meth)]["units"]) / D["main"][(cnd, meth)]["EVAL_UJ_minus_S"]["sd"]
                    for meth in D["methods"])
        uv[cnd] = {"max": max(vals), "mean": float(np.mean(vals)), "max_over_sd": ratio}
    D["uv"] = uv
    # head-gap polish statuses (MF formal units)
    hg = [u["sel"]["head_gap"]["fp64_polish"] for u in units if u["block"] in ("main", "algo_extra", "capacity") and u["sel"]["head_gap"]["status"] == "MEASURED"]
    D["polish"] = dict(collections.Counter(p["status"] for p in hg))
    D["polish_iters"] = sorted({p["iterations"] for p in hg})
    check(len(D["polish_iters"]) == 1, "fp64 polish iteration counts differ")
    # M1 inner solver (formal units)
    m1 = collections.defaultdict(lambda: {"units": 0, "stop": collections.Counter(), "conv": [], "fp64_trig": 0, "fp64_dis": 0, "it_med": []})
    for u in units:
        if u["block"] in ("main", "algo_extra", "capacity") and u["method"] == "MF_VCS_M1":
            r = u["res"]["m1"]
            e = m1[int(r["cap"])]
            e["units"] += 1
            e["stop"].update(r["stop_reasons"])
            e["conv"].append(r["converged_fraction"])
            e["fp64_trig"] += r["fp64_diagnostic"]["triggered_steps"]
            e["fp64_dis"] += r["fp64_diagnostic"]["disagreements"]
            e["it_med"].append(r["iterations"]["median"])
            check(r["rule"] == "relative_zero_head" and r["first_step_W_start_is_zero"] and not r["require_converged"], "M1 rule record")
            check(int(r["cap"]) == D["selection"][selection_key(u)]["cap"], "M1 cap != selection")
            e["tol"] = r["relative_tolerance"]
    D["m1"] = dict(m1)
    # step distribution of the CAL group selections
    D["cal_steps"] = collections.Counter((k[1], g["step"]) for k, g in D["selection"].items())


def collect_cost(D) -> None:
    units = list(D["units"].values())
    by = collections.defaultdict(lambda: {"n": 0, "wall": 0.0, "cpu": 0.0, "eval": 0.0})
    meth = collections.defaultdict(lambda: {"n": 0, "wall": 0.0})
    for u in units:
        c = u["res"]["cost"]
        e = by[UNIT_BLOCK[u["block"]]]
        e["n"] += 1
        e["wall"] += c["wall_seconds_unit_total"]
        e["cpu"] += c["cpu_seconds_unit_total"]
        e["eval"] += c["wall_seconds_evaluation"]
        if u["block"] != "smoke":
            meth[u["method"]]["n"] += 1
            meth[u["method"]]["wall"] += c["wall_seconds_unit_total"]
    D["cost_blocks"] = dict(by)
    D["cost_methods"] = dict(meth)
    budget = load_json(REPO / BUDGET_REL)
    D["budget_sha256"] = sha256_file(REPO / BUDGET_REL)
    threads = {j["threads"] for b, j in D["jobs"].items() if b != "smoke"}
    check(len(threads) == 1, "grid jobs with different thread counts")
    thr = threads.pop()
    lay = [L for L in budget["layouts"] if L["threads_per_process"] == thr and L["status"] == "MEASURED_PROJECTION"]
    check(len(lay) == 1, "no unique budget layout at the grid jobs' thread count")
    L = lay[0]
    cpus = {j["cpus"] for b, j in D["jobs"].items() if b != "smoke"}
    check(len(cpus) == 1, "grid jobs with different CPU counts")
    ncpu = cpus.pop()
    wall = [w for w in L["wall_hours_at_concurrency"] if w["jobs"] == 1 and w["cpus_per_job"] == ncpu]
    check(len(wall) == 1, "no budget wall entry for one job at the grid CPU count")
    for frag in ("m = 8 and K = 8 / 32 rows charged the m = 32 / K = 16 cost", "memory-bandwidth contention beyond the measured 8-CPU layout is not measured",
                 "queue time excluded"):
        check(any(frag in s for s in budget["assumptions"]), f"budget assumption {frag!r} not recorded")
    om = budget["phase1_order_of_magnitude"]
    p1 = (REPO / PHASE1_REPORT_REL).read_text(encoding="utf-8")
    check(om["quotes_found_in_report"] and all(q in p1 for q in om[PHASE1_QUOTE_KEY]), "phase-1 order-of-magnitude quotes not found in the phase-1 report")
    D["budget"] = {"layout": L, "wall": wall[0], "om": om, "ncpu": ncpu, "threads": thr}


# ------------------------------------------------------------------------------------------------ section 10: corrections (2026-10-19)
def load_package_generator(D):
    """The package generator v10_synthetic.py (committed, clean, sha256 = every unit record's generator_sha256), imported from its
    file for the exact population tables of section 10 item 1 (float64 arithmetic, no sampling)."""
    check(git("status", "--porcelain", "--untracked-files=all", "--", PACKAGE_GENERATOR_REL).strip() == "",
          f"{PACKAGE_GENERATOR_REL}: uncommitted or untracked changes")
    path = REPO / PACKAGE_GENERATOR_REL
    sha = sha256_file(path)
    rec = {u["res"]["provenance"]["identity"]["generator_sha256"] for u in D["units"].values()}
    check(rec == {sha}, f"{PACKAGE_GENERATOR_REL}: sha256 {sha[:16]} != the generator_sha256 of the unit records {sorted(rec)}")
    spec = importlib.util.spec_from_file_location("v10_synthetic_package", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, sha


def block_witness(p, q, eta, K: int) -> tuple[float, float]:
    """A K-feature MF table in the closure of the MF class: the m states split in order into K equal groups, a(x) = b(x) = one-hot of
    the group (the limit of softmax features), W = the M-weighted mean of eta over each group block.  (E_M(T - eta)^2, max |W|)."""
    m = p.shape[0]
    check(m % K == 0, "witness: m is not a multiple of K")
    g = np.arange(m) // (m // K)
    w = 0.5 * (p + q)
    W = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            blk = np.ix_(g == i, g == j)
            mass = float(w[blk].sum())
            W[i, j] = float((w[blk] * eta[blk]).sum()) / mass if mass > 0 else 0.0
    T = W[g][:, g]
    return float(np.sum(w * (T - eta) ** 2)), float(np.abs(W).max())


def _mm(v) -> tuple[float, float]:
    v = list(v)
    return min(v), max(v)


def fpct_up(x: float, d: int = 2) -> str:
    """percentage rounded up to d decimals (for an upper bound stated as "≤")"""
    return f"{math.ceil(100 * float(x) * 10 ** d) / 10 ** d:.{d}f}%"


def collect_corrections(D) -> None:
    """Every number of section 10, from the records (item 1 also from the package generator's exact population tables); each
    data-dependent phrase of section 10 is checked here and the counts reviewed on 2026-10-19 are pinned."""
    conds, methods, mf = D["conds"], D["methods"], D["mf_methods"]
    mlp = [m for m in methods if m not in mf]
    main, capc, sel = D["main"], D["capacity"], D["selection"]
    K = D["K_main"]
    lr_hi, lr_lo, s_last = max(D["lrs"]), min(D["lrs"]), max(D["steps"])
    units = list(D["units"].values())
    formal = [u for u in units if u["block"] in ("main", "algo_extra", "capacity")]
    run = D["jobs"]["formal"]["run_commit"]
    dmp = [c for c in conds if c[0] == "diagonal_mixture" and c[2] > 0]
    tbp = [c for c in conds if c[0] == "two_block" and c[2] > 0]
    c0 = [c for c in conds if c[2] == 0]
    X = {}
    # 1  "not the minimum error of the K16 MF class": representational support
    arch = {json.dumps({k: u["res"]["critic"]["architecture"][k] for k in ("features", "head", "input", "output")}, sort_keys=True)
            for u in units if u["method"] in mf}
    check(arch == {json.dumps({"features": "Linear + softmax per side", "head": "W in [-1, 1]^{K x K}, zeros at start",
                               "input": "one-hot integer observations", "output": "T = a(x)^T W b(y) (no tanh)"}, sort_keys=True)},
          "MF architecture records differ from the class described in section 10 item 1")
    small = [c for c in dmp if c[1] <= K]
    large = [c for c in dmp if c[1] > K]
    kge = [k for k in D["cap_K"] if all(k >= c[1] for c in large)]
    check(small and large and kge and all(c in D["cap_conds"] for c in large), "section 10 item 1: condition / capacity structure")
    gen, gen_sha = load_package_generator(D)
    wit = {}
    for c in large:
        t = gen.exact_truth(*c)
        p, q, eta = (np.asarray(t[k], dtype=np.float64) for k in ("p", "q", "eta"))
        Ss = {u["ev"]["truth"]["S"] for mth in methods for u in main[(c, mth)]["units"]}
        check(Ss == {float(t["S"])}, f"package exact_truth S of {c} != the recorded S")
        check(float(np.abs(eta).max()) <= 1.0, "eta outside [-1, 1]")
        e, wmax = block_witness(p, q, eta, K)
        trained = _mm(main[(c, mth)]["pop_mse"]["mean"] for mth in mf)
        check(wmax <= 1.0 and 0 < e < trained[0], f"section 10 item 1: the K{K} construction of {c} is not below the trained values")
        wit[c] = {"pop": e, "trained": trained, "group": c[1] // K}
    X["c1"] = {"m_small": sorted({c[1] for c in small}), "m_large": sorted({c[1] for c in large}), "kge": kge, "witness": wit,
               "pop_small": _mm(main[(c, mth)]["pop_mse"]["mean"] for c in small for mth in mf),
               "pop_kge": _mm(capc[(c, mth, k)]["pop_mse"]["mean"] for c in large for mth in mf for k in kge), "gen_sha": gen_sha}
    # 2  grid-edge selections, MF and MLP (recomputed CAL selection, section 3)
    mf_hi = [(c, m) for c in conds for m in mf if sel[(c, m)]["lr"] == lr_hi]
    mf_both = [(c, m) for (c, m) in mf_hi if sel[(c, m)]["step"] == s_last]
    check(all(sel[(c, m)]["lr"] == lr_hi for c in conds for m in mf if sel[(c, m)]["step"] == s_last), "an MF last-step group below lr max")
    mlp_lo = [(c, m) for c in conds for m in mlp if sel[(c, m)]["lr"] == lr_lo]
    mlp_pos = [(c, m) for c in conds if c[2] > 0 for m in mlp]
    mlp_lo_pos = [k for k in mlp_pos if k in mlp_lo]
    mlp_exc = [k for k in mlp_pos if k not in mlp_lo]
    full_hi = [c for c in conds if all((c, m) in mf_hi for m in mf)]
    full_both = [c for c in conds if all((c, m) in mf_both for m in mf)]
    rest_both = [(c, m) for (c, m) in mf_both if c not in full_both]
    exc_lr = {sel[k]["lr"] for k in mlp_exc}
    check(len(full_hi) * len(mf) == len(mf_hi), "section 10 item 2: an lr-max condition without every MF method")
    check(full_both == dmp, "section 10 item 2: the both-edge conditions are not the diagonal_mixture rho > 0 conditions")
    check(len(exc_lr) == 1, "section 10 item 2: MLP rho > 0 exceptions at more than one lr")
    check((len(mf_hi), len(mf) * len(conds), len(mf_both), len(mlp_lo), len(mlp) * len(conds), len(mlp_lo_pos), len(mlp_pos))
          == (18, 36, 13, 21, 24, 14, 16), "section 10 item 2: grid-edge counts changed")
    X["c2"] = {"mf_hi": mf_hi, "mf_n": len(mf) * len(conds), "mf_both": mf_both, "full_hi": full_hi, "full_both": full_both,
               "rest_both": rest_both, "mlp_lo": mlp_lo, "mlp_n": len(mlp) * len(conds), "mlp_lo_pos": mlp_lo_pos, "mlp_pos": mlp_pos,
               "mlp_exc": mlp_exc, "exc_lr": exc_lr.pop()}
    # 3  expected sign of the error: E[e | T] = -E_M(T - eta)^2 for a critic chosen without EVAL
    run_py = git("show", f"{run}:scripts/v10_t03/t03_run.py")
    common = git("show", f"{run}:{COMMON_REL}")
    obj = git("show", f"{run}:{OBJECTIVES_REL}")
    check('with C.truth_barrier(f"train+CAL {cfg.unit_id}"):' in run_py, "train+CAL outside a truth-barrier block at the formal run commit")
    check("def truth_barrier(phase: str, *, allow_eval: bool = False):" in common, "truth_barrier does not default to allow_eval=False")
    for frag in ("q_hat_U = (N_x N_y - N_xy) / (n (n - 1))", "J_hat = sum_xy [ p_hat (T - T^2/2) - q_hat (T + T^2/2) ]"):
        check(frag in obj, f"{OBJECTIVES_REL} at the formal run commit lacks {frag!r}")
    check(D["jobs"]["formal"]["denials"] == 0, "truth-barrier denials in the formal block")
    main_units = [u for c in conds for m in methods for u in main[(c, m)]["units"]]
    check(len(main_units) == D["n_by_block"]["main"] and all(u["sel"]["pop_mse"] > 0 for u in main_units), "a main unit with E_M(T - eta)^2 = 0")
    neg0 = sum(main[(c, m)]["EVAL_UJ_minus_S"]["mean"] < 0 for c in c0 for m in methods)
    excl0 = [(c, m) for c in c0 for m in methods if not has0(main[(c, m)]["EVAL_UJ_minus_S"]["mean_ci95_t"])]
    check(all(main[k]["EVAL_UJ_minus_S"]["mean_ci95_t"][1] < 0 for k in excl0) and (neg0, len(c0) * len(methods), len(excl0)) == (18, 20, 2),
          "section 10 item 3a: rho = 0 counts changed")
    t75 = [c for c in tbp if c[2] == 0.75]
    b75 = [main[(c, m)]["EVAL_UJ_minus_S"]["mean"] for c in t75 for m in methods]
    e75 = [-main[(c, m)]["pop_mse"]["mean"] for c in t75 for m in methods]
    s75 = [main[(c, m)]["EVAL_UJ_minus_J_pop"]["mean"] for c in t75 for m in methods]
    pos75 = [c for c in t75 if all(main[(c, m)]["EVAL_UJ_minus_J_pop"]["mean_ci95_t"][0] > 0 for m in methods)]
    check(all(b > 0 for b in b75) and all(s > 0 for s in s75) and pos75, "section 10 item 3b: two_block rho 0.75 signs")
    check(all(main[(c, m)]["EVAL_UJ_minus_S"]["mean"] < 0 for c in dmp for m in methods), "section 10 item 3c: a diagonal rho > 0 bias >= 0")
    X["c3"] = {"neg0": neg0, "n0": len(c0) * len(methods), "excl0": excl0, "n75": len(b75), "b75": _mm(b75), "e75": _mm(e75), "s75": _mm(s75),
               "pos75": pos75, "n_main": len(main_units),
               "ratio_mlp": _mm(main[(c, m)]["pop_mse"]["mean"] / -main[(c, m)]["EVAL_UJ_minus_S"]["mean"] for c in dmp for m in mlp),
               "pop_mf_dm": _mm(main[(c, m)]["pop_mse"]["mean"] for c in dmp for m in mf)}
    # 4  shared samples and diagnostic derangement streams within a condition
    streams = collections.defaultdict(set)
    for u in formal:
        for role in ("fit", "eval"):
            streams[(u["data"],) + u["cond"] + (role,)].add(json.dumps(u["sel"]["perm_conditional_variance"][role]["stream"], sort_keys=True))
    check(D["obs_sets"] == len(conds) * len(D["data_seeds"]), "observation sets != conditions x data roots")
    check(all(len(v) == 1 for v in streams.values()) and len(streams) == 2 * D["obs_sets"]
          and len({next(iter(v)) for v in streams.values()}) == len(streams), "diagnostic derangement streams not shared one per (data root, condition, role)")
    n75 = sum(has0(main[(c, m)]["EVAL_UJ_minus_S"]["mean_ci95_t"]) for c in t75 for m in methods)
    n0 = sum(has0(main[(c, m)]["EVAL_UJ_minus_S"]["mean_ci95_t"]) for c in c0 for m in methods)
    nsamp = sum(has0(main[(c, m)]["EVAL_UJ_minus_J_pop"]["mean_ci95_t"]) for c in conds for m in methods)
    check(all(main[(c, m)]["EVAL_UJ_minus_S"]["mean_ci95_t"][1] < 0 for c in dmp for m in methods), "section 10 item 4: diagonal rho > 0 count")
    X["c4"] = {"obs": D["obs_sets"], "streams": len(streams), "n75": n75, "n75_of": len(t75) * len(methods), "ndm": len(dmp) * len(methods),
               "n0": n0, "n0_of": len(c0) * len(methods), "nsamp": nsamp, "nsamp_of": len(conds) * len(methods), "nz": D["perm_z"]["n"],
               "n_cond": len(conds), "n_data": len(D["data_seeds"])}
    # 5  two_block rho > 0: RMSE per rho and the cross-family paired difference
    rr = {}
    for rho in sorted({c[2] for c in tbp}):
        cs = [c for c in tbp if c[2] == rho]
        pc = [D["paired"][(c, "MF_VCS_M0", "MLP_VCS_perm1")]["sq"]["mean_ci95_t"] for c in cs]
        kind = {"neg" if ci[1] < 0 else "pos" if ci[0] > 0 else "zero" for ci in pc}
        check(len(kind) == 1, f"section 10 item 5: paired M0 - MLP_perm1 at two_block rho {rho} not homogeneous")
        rr[rho] = {"mf": _mm(main[(c, m)]["EVAL_UJ_minus_S"]["rmse"] for c in cs for m in mf),
                   "mlp": _mm(main[(c, m)]["EVAL_UJ_minus_S"]["rmse"] for c in cs for m in mlp), "n": len(cs), "kind": kind.pop()}
    check([rr[r]["kind"] for r in sorted(rr)] == ["neg", "zero"], "section 10 item 5: paired-difference pattern changed")
    X["c5"] = rr
    # 6  fp64 diagnostic disagreements by rho and cap (formal M1 units)
    fb = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    nm1 = 0
    for u in formal:
        if u["method"] == "MF_VCS_M1":
            nm1 += 1
            r = u["res"]["m1"]
            e = fb[u["rho"]][int(r["cap"])]
            e[0] += r["fp64_diagnostic"]["triggered_steps"]
            e[1] += r["fp64_diagnostic"]["disagreements"]
    tot = sum(v[1] for d in fb.values() for v in d.values())
    check(tot == sum(e["fp64_dis"] for e in D["m1"].values()) and sum(v[0] for d in fb.values() for v in d.values()) == sum(e["fp64_trig"] for e in D["m1"].values()),
          "section 10 item 6: fp64 diagnostic totals != section 5.3")
    check(tot == 6845 and sum(v[1] for v in fb[0.0].values()) == 6233, "section 10 item 6: disagreement counts changed")
    X["c6"] = {"by": {rho: dict(sorted(d.items())) for rho, d in sorted(fb.items())}, "tot": tot, "n_m1": nm1}
    # 7  identity checks vs distinct tables
    modes = {tuple(sorted(k for k in u["ev"]["saturated_fit_table"] if k != "note")) for u in units}
    check(modes == {("U", "V")}, "saturated table references")
    n_modes = len(modes.pop())
    per_unit = {1 + len(u["ev"]["trajectory_diagnostic"]["rows"]) + n_modes for u in units}
    sel_in_traj = sum(any(r["step"] == u["ev"]["selected"]["step"] and all(r[f] == u["sel"][f] for f in ("pop_mse", "J_pop", "EVAL_UJ", "FIT_UJ", "CAL_UJ"))
                          for r in u["ev"]["trajectory_diagnostic"]["rows"]) for u in units)
    n_traj = sum(len(u["ev"]["trajectory_diagnostic"]["rows"]) for u in units)
    check(len(per_unit) == 1 and D["identity"]["n"] == len(units) * per_unit.copy().pop() and sel_in_traj == len(units),
          "section 10 item 7: identity check layout")
    X["c7"] = {"per_unit": per_unit.pop(), "n_traj_ckpt": n_traj // len(units), "sel_in_traj": sel_in_traj, "n_units": len(units),
               "n_traj": n_traj, "sat_keys": D["sat_keys"], "distinct": n_traj + n_modes * D["sat_keys"], "checks": D["identity"]["n"],
               "n_sat_modes": n_modes}
    check(n_traj // len(units) * len(units) == n_traj, "trajectory rows differ between units")
    # 8  head gap: fp32 column vs fp64 bound; checkpoint (post-update) gap vs the M1 solver's pre-update gap
    hg = [u["sel"]["head_gap"] for u in formal if u["method"] in mf]
    check(all(h["status"] == "MEASURED" for h in hg) and len(hg) == sum(D["polish"].values()), "section 10 item 8: head-gap records")
    rel = max(abs(h["fw_gap_W"] - h["fp64"]["fw_gap_W"]) / h["fp64"]["fw_gap_W"] for h in hg)
    nb = sum(h["fp64_polish"]["suboptimality_upper_bound_fw_gap"] == h["fp64"]["fw_gap_W"]
             and h["fp64_polish"]["suboptimality_lower_bound"] <= h["fp64"]["fw_gap_W"] for h in hg)
    sc_eval = {h["scope"] for h in hg}
    sc_m1 = {u["res"]["m1"]["gap_scope"] for u in formal if u["method"] == "MF_VCS_M1"}
    check(nb == len(hg) and len(sc_eval) == 1 and "post outer update" in next(iter(sc_eval)), "section 10 item 8: fp64 bound / evaluator scope")
    check(len(sc_m1) == 1 and next(iter(sc_m1)).startswith("inner_pre_outer_batch"), "section 10 item 8: M1 gap scope")
    gm1 = sum(main[(c, "MF_VCS_M1")]["gap"]["mean"] < main[(c, "MF_VCS_M0")]["gap"]["mean"] for c in conds)
    X["c8"] = {"rel": rel, "n": len(hg), "nb": nb, "m1_scope": next(iter(sc_m1)).split(" ", 1)[0], "gm1": gm1, "n_cond": len(conds)}
    # 9  section 5.5 column "V 参考 EVAL − S 均值" = EVAL U-J of the V-reference saturated table
    satfull = {}
    for u in units:
        k = (u["data"],) + u["cond"]
        rec = json.dumps(u["ev"]["saturated_fit_table"], sort_keys=True)
        check(satfull.setdefault(k, rec) == rec, f"saturated table copies of {k} differ")
    vrows = {k: json.loads(v)["V"] for k, v in satfull.items() if k[0] in D["data_seeds"]}
    check(len(vrows) == D["obs_sets"], "section 10 item 9: formal saturated rows")
    for c in conds:
        S = main[(c, methods[0])]["S"]
        mine = float(np.mean([vrows[(s,) + c]["EVAL_UJ"] - S for s in D["data_seeds"]]))
        check(close(mine, D["sat"][c]["eval_err_V"]["mean"]), f"section 10 item 9: {c} column != V-table EVAL U-J - S")
    shift = {}
    for c in conds:
        col = float(np.mean([vrows[(s,) + c]["EVAL_UJ"] - main[(c, methods[0])]["S"] for s in D["data_seeds"]]))
        d = float(np.mean([vrows[(s,) + c]["EVAL_UJ"] - vrows[(s,) + c]["EVAL_VJ"] for s in D["data_seeds"]]))
        check(col != 0, "section 10 item 9: zero column value")
        shift[c] = (abs(d), abs(d) / abs(col))
    ca = max(shift, key=lambda c: shift[c][0])
    cr = max(shift, key=lambda c: shift[c][1])
    X["c9"] = {"n": len(vrows), "abs": shift[ca][0], "abs_c": ca, "rel": shift[cr][1], "rel_c": cr}
    # 10  R = 3 of the algorithm table
    ra = {a["EVAL_UJ_minus_S"]["R"] for a in D["algo"].values()}
    ci = next(iter(D["algo"].values()))["EVAL_UJ_minus_S"]["var_ci95_chi2"]
    nalg = sum(D["algo"][k]["EVAL_UJ_minus_S"]["sd"] > main[k]["EVAL_UJ_minus_S"]["sd"] for k in main)
    check(len(ra) == 1, "algo cells with different R")
    X["c10"] = {"R": ra.pop(), "ratio": math.sqrt(ci[1] / ci[0]), "nalg": nalg, "n": len(main)}
    # 11  report date
    rdate = D["git"]["formal"]["utc"][:10]
    check(rdate.replace("-", "") in REPORT_REL, "section 10 item 11: report date != the formal results commit date")
    X["c11"] = {"date": rdate}
    # 12  training implementation (no cell-count aggregation) and the per-method cost of section 2.4
    frag = "used for CAL selection and every readout; training itself evaluates the critic row by row, as the natural host does"
    check(frag in obj and "complete U moments; one scene, weight 1" in obj, f"{OBJECTIVES_REL} at the formal run commit: training layout")
    negs = {}
    for u in units:
        r = u["res"]
        if u["method"].startswith("MLP_"):
            v = r["critic"]["architecture"]["negatives_per_step"]
        elif (r.get("derangements") or {}).get("perm"):
            v = r["derangements"]["perm"]["draws"] // int(D["proto"]["training"]["steps"])
        else:
            v = None
        negs.setdefault(u["method"], set()).add(v)
        check(r["data"]["n"]["fit"] == int(D["n_obs"]["FIT"]), f"{u['row']['unit_id']}: FIT n")
    check(all(len(v) == 1 for v in negs.values()), "negatives per step differ within a method")
    negs = {m: v.pop() for m, v in negs.items()}
    check(negs["MF_VCS_M0"] is None and negs["MF_VCS_M1"] is None and all(negs[m] for m in ("MF_VCS_perm1", "MLP_VCS_perm1", "MLP_VCS_perm8")),
          "section 10 item 12: negatives per step")
    per = {m: v["wall"] / v["n"] for m, v in D["cost_methods"].items()}
    X["c12"] = {"frag": frag, "negs": negs, "n_fit": int(D["n_obs"]["FIT"]), "per": _mm(per.values())}
    # every integer count stated in section 10, as reviewed on 2026-10-19: a changed count refuses the run
    stated = {
        "1": (X["c1"]["m_small"], X["c1"]["m_large"], X["c1"]["kge"], sorted({v["group"] for v in wit.values()}), K),
        "2": (len(mf_hi), X["c2"]["mf_n"], len(mf_both), len(full_hi), len(full_both), len(rest_both), len(mlp_lo), X["c2"]["mlp_n"],
              len(mlp_lo_pos), len(mlp_pos), len(mlp_exc), D["n_by_block"]["CAL"], len(sel)),
        "3": (neg0, len(c0) * len(methods), len(excl0), len(b75), len(pos75), len(main_units), len(D["data_seeds"])),
        "4": (D["obs_sets"], len(streams), n75, len(t75) * len(methods), len(dmp) * len(methods), n0, nsamp, len(conds) * len(methods),
              D["perm_z"]["n"], D["perm_z"]["R"], len(conds), len(D["data_seeds"])),
        "5": tuple((r, X["c5"][r]["n"], X["c5"][r]["kind"]) for r in sorted(X["c5"])),
        "6": (tot, nm1, tuple((rho, tuple(v[1] for v in d.values()), tuple(d)) for rho, d in X["c6"]["by"].items())),
        "7": (X["c7"]["checks"], X["c7"]["per_unit"], X["c7"]["n_traj_ckpt"], X["c7"]["n_sat_modes"], X["c7"]["sel_in_traj"], X["c7"]["n_units"],
              X["c7"]["sat_keys"], X["c7"]["n_traj"], X["c7"]["distinct"]),
        "8": (X["c8"]["n"], X["c8"]["nb"], X["c8"]["gm1"], X["c8"]["n_cond"]),
        "9": (X["c9"]["n"],),
        "10": (X["c10"]["R"], X["c10"]["nalg"], X["c10"]["n"], round(X["c10"]["ratio"])),
        "12": (tuple(sorted((m, v) for m, v in negs.items() if v)), X["c12"]["n_fit"]),
    }
    reviewed = {
        "1": ([8], [32], [32], [2], 16),
        "2": (18, 36, 13, 6, 4, 1, 21, 24, 14, 16, 2, 252, 60),
        "3": (18, 20, 2, 10, 1, 600, 10),
        "4": (120, 240, 10, 10, 20, 18, 48, 60, 1920, 512, 12, 10),
        "5": ((0.25, 2, "neg"), (0.75, 2, "zero")),
        "6": (6845, 224, ((0.0, (22, 6211), (10, 100)), (0.25, (0, 612), (10, 100)), (0.75, (0, 0), (10, 30)))),
        "7": (13409, 11, 8, 2, 1219, 1219, 133, 9752, 10018),
        "8": (672, 672, 9, 12),
        "9": (120,),
        "10": (3, 13, 60, 12),
        "12": ((("MF_VCS_perm1", 1), ("MLP_VCS_perm1", 1), ("MLP_VCS_perm8", 8)), 4096),
    }
    for k in reviewed:
        check(stated[k] == reviewed[k], f"section 10 item {k}: stated counts {stated[k]} != the reviewed counts {reviewed[k]}")
    D["corr"] = X


def sec_corrections(D, Q, published: str) -> list[str]:
    X = D["corr"]

    def orig(frag: str) -> str:
        check(frag in published, f"section 10 quotes a sentence not in the published sections 1-9: {frag!r}")
        return f"“{frag}”"

    def item(n: int, title: str, original: str, corrected: str, evidence: str) -> list[str]:
        return [f"{n}. **{title}**", f"   - 原文{original}", f"   - 更正：{corrected}", f"   - 依据：{evidence}"]

    lab = cond_label
    rng = lambda t: f"{fe(t[0])}–{fe(t[1])}"  # noqa: E731
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = (X[f"c{i}"] for i in range(1, 13))
    K = D["K_main"]
    out = ["## 10 更正（2026-10-19）", "",
           f"本节于 2026-10-19 追加，逐条补充 §1、§5、§8 中若干语句的依据、限定与澄清。§1–§9 保持首次发布时的原文（生成器核对其与提交 `{PUBLISHED_COMMIT[:7]}` 中的版本"
           "逐字节相同），全部结果不变。每条给出原文、更正后的表述与依据；数值均由生成器从已提交记录计算，第 1 条的构造另由已提交的包生成器 "
           f"`{PACKAGE_GENERATOR_REL}`（sha256 `{c1['gen_sha'][:16]}`…，等于全部单元记录中的 generator_sha256）的 exact_truth 计算；依赖数据的语句其成立条件由生成器核对，"
           "不成立则拒绝生成。", ""]
    w = c1["witness"]
    wl = sorted(w)
    out += item(1, f"“不是 K{K} MF 函数类可达到的最小误差”的依据",
                "（§1 diagonal_mixture、ρ > 0 条）：" + orig(f"选择落在固定步数与学习率网格的边界，这些 MF 数值是本训练规则下的结果，不是 K{K} MF 函数类可达到的最小误差") + "。",
                f"选择落在网格边界只说明冻结网格没有包住 CAL 最优点；网格外（更大学习率、更长训练）未测量，不能由此推出函数类可达到的最小误差。"
                f"“不是 K{K} MF 函数类可达到的最小误差”本身成立，但依据是函数类的表示能力，见下。",
                "MF 单元记录的结构为每侧 Linear + softmax 特征、one-hot 输入、T = a(x)ᵀ W b(y)、W ∈ [−1, 1]^(K×K)；softmax 特征可任意接近 one-hot，且 η ∈ [−1, 1]。"
                f"m = {'、'.join(map(str, c1['m_small']))} 时 K{K} ≥ m，函数类闭包含 η，E_M(T − η)² 的下确界为 0，而所选 K{K} MF critic 的 E_M(T − η)² 均值为 {rng(c1['pop_small'])}。"
                f"m = {'、'.join(map(str, c1['m_large']))} 时，同条件 {'、'.join(f'K{k}' for k in c1['kge'])}（K ≥ m，闭包同样含 η）训练所得 E_M(T − η)² 均值仍为 {rng(c1['pop_kge'])}（§7）；"
                f"另把 {wl[0][1]} 个状态按顺序每 {w[wl[0]]['group']} 个分为 {K} 组、a(x) 与 b(y) 取所在组的 one-hot、W 取 M 加权的组内 η 均值，这一 K{K} 构造的 E_M(T − η)² 为 "
                + " / ".join(f"{fe(w[c]['pop'])}（{lab(c)}）" for c in wl) + "，低于训练所得的 " + " / ".join(rng(w[c]["trained"]) for c in wl)
                + "。该构造只给出函数类误差的上界，不是训练结果。")
    out += item(2, "网格边界的选择不限于 diagonal_mixture ρ > 0 的 MF 组，也包括 MLP",
                "（§8）：" + orig("diagonal_mixture ρ > 0 的 MF 选择落在学习率与步数网格边界（§1、§3）") + "；§1 diagonal_mixture、ρ > 0 条只对 MF 作此说明。",
                f"MF {c2['mf_n']} 组中 {len(c2['mf_hi'])} 组选中学习率上端 {flr(max(D['lrs']))}（{'、'.join(lab(c) for c in c2['full_hi'])} 的全部 {len(D['mf_methods'])} 个 MF 方法），"
                f"其中 {len(c2['mf_both'])} 组同时选中最后一个 checkpoint（{max(D['steps'])} 步）：diagonal_mixture ρ > 0 的 {len(c2['full_both'])} 个条件 × {len(D['mf_methods'])} 个 MF 方法，加 "
                + "、".join(f"{lab(c)} {m}" for c, m in c2["rest_both"]) + f"。MLP {c2['mlp_n']} 组中 {len(c2['mlp_lo'])} 组选中学习率下端 {flr(min(D['lrs']))}，ρ > 0 的 "
                f"{len(c2['mlp_pos'])} 组中 {len(c2['mlp_lo_pos'])} 组（其余为 " + "、".join(f"{lab(c)} {m}" for c, m in c2["mlp_exc"]) + f"，选 {flr(c2['exc_lr'])}）。"
                "网格外（更大或更小的学习率、更长训练）未测量，MF 与 MLP 的数值都只代表冻结网格内的结果；网格按决定 §9.2 冻结，未作事后扩展。",
                f"由 {D['n_by_block']['CAL']} 个 CAL 单元的检查点 CAL U-J 重算的 {len(D['selection'])} 组选择（§3.1，与选模记录逐组一致）。")
    e0 = "；".join(f"{lab(c)} {m}" for c, m in c3["excl0"])
    out += item(3, "误差的期望符号：每格的期望偏差都为负",
                f"（§1 ρ = 0 条）：" + orig(f"{c4['n0']}/{c4['n0_of']} 格偏差区间含 0，其余 {len(c3['excl0'])} 格（{e0}）偏差为负")
                + "；（§1 two_block、ρ > 0 条）：" + orig(f"ρ = 0.75 的 {c4['n75']}/{c4['n75_of']} 格偏差区间含 0")
                + "；（§1 diagonal_mixture、ρ > 0 条）：" + orig("MF 的偏差主要是总体逼近误差") + "；（§8）：" + orig("主读数为 U") + "。",
                f"所选 critic T 不依赖本重复的 EVAL 样本（训练用 FIT，checkpoint 由本重复的 CAL 选，超参数来自独立调参套），完整经验 U-J 对固定 T 无偏，故每格 E[e | T] = −E_M(T − η)² ≤ 0；{c3['n_main']} 个主单元的 E_M(T − η)² 均 > 0，"
                f"所以每个方法、每个条件的期望偏差都严格为负。(a) “区间含 0”只表示 R = {len(D['data_seeds'])} 下没有分辨出，不表示无偏；ρ = 0 的 {c3['n0']} 格中 {c3['neg0']} 格点估计为负，"
                f"原文所列 {len(c3['excl0'])} 格是区间整体低于 0。(b) two_block ρ0.75 的 {c3['n75']} 格点估计均为正（{rng(c3['b75'])}），而这些格期望偏差的估计 −E_M(T − η)² 均值为 "
                f"{fe(c3['e75'][0])} 至 {fe(c3['e75'][1])}；正值来自 EVAL 抽样误差（{c3['n75']} 格均值 {rng(c3['s75'])}）：同一条件的 {len(D['methods'])} 个方法共用同一组 "
                f"{len(D['data_seeds'])} 套 EVAL 样本（第 4 条），{'、'.join(lab(c) for c in c3['pos75'])} 的 {len(D['methods'])} 个方法抽样误差区间都整体高于 0。"
                f"(c) “偏差主要是总体逼近误差”在期望上对每个方法都成立；E_M(T − η)² 均值 / |偏差| 只反映 {len(D['data_seeds'])} 个重复的均值中 EVAL 抽样误差所占的份量（同条件 "
                f"{len(D['methods']) - len(D['mf_methods'])} 个 MLP 方法为 {c3['ratio_mlp'][0]:.2f}–{c3['ratio_mlp'][1]:.2f}）。该句的实际含义是 diagonal_mixture ρ > 0 的 MF 组 E_M(T − η)² 本身大（均值 {rng(c3['pop_mf_dm'])}）。"
                "(d) 经训练与选模的 critic 的 EVAL U-J 期望为 S − E[E_M(T − η)²] ≤ S，主读数 U 同样不是 S 的无偏估计（§8 所引决定 §9.1 的句子针对 V 公式）。",
                "运行提交的 `scripts/v10_t03/t03_run.py` 中训练与 CAL 处于 `truth_barrier` 内，`scripts/v10_t03/t03_common.py` 的 `truth_barrier` 默认 `allow_eval=False`，"
                "EVAL 样本在此期间不可读取（拒绝 0 次，§2.3）；`scripts/v10_t03/t03_objectives.py` 的读数为 Ĵ = Σ_xy [p̂ (T − T²/2) − q̂ (T + T²/2)]，"
                "q̂_U = (N_x N_y − N_xy)/(n(n − 1))，p̂ 与 q̂_U 是 P 与 Q = P_X P_Y 单元概率的无偏估计，Ĵ 对二者线性。数值取自 §4.1 与 §5.2。")
    out += item(4, "同一条件内各方法共用样本与排列流：跨方法的格计数不是独立证据",
                "（§1 精度条）：" + orig("区间逐格给出，未做多重比较校正") + "——该条未说明样本共用。",
                f"同一（数据 root，条件）的全部方法（含算法重复与容量单元）使用同一套 FIT/CAL/EVAL 样本，评估用的 {D['perm_z']['R']} 个诊断排列也按（数据 root，条件，FIT/EVAL）共用。"
                f"因此同一条件内跨方法的格计数——§1 的“{c4['n75']}/{c4['n75_of']} 格”“全部 {c4['ndm']} 格”“{c4['n0']}/{c4['n0_of']} 格”“{c4['nsamp']}/{c4['nsamp_of']} 格”与 "
                f"{c4['nz']} 个 z 值——不是独立证据；独立单位只有 {c4['n_cond']} 个条件 × {c4['n_data']} 套数据 = {c4['obs']} 套。§4.2 的配对差正是利用这种共用；跨方法的格计数应按条件读。",
                f"{c4['obs']} 套观测的 FIT/CAL/EVAL 样本 sha256 逐一核对（§4）；正式单元的诊断排列流记录共 {c4['streams']} 个（{c4['obs']} 套 × FIT/EVAL），每个在其全部单元中相同。")
    r5 = c5
    rk = sorted(r5)
    kind_txt = {"neg": "为负", "zero": "含 0", "pos": "为正"}
    out += item(5, "MF 与 MLP 的合并范围",
                "（§1 two_block、ρ > 0 条）：" + orig(f"K{K} MF 三法 RMSE {fe(min(r5[r]['mf'][0] for r in rk))}–{fe(max(r5[r]['mf'][1] for r in rk))}，MLP 两法 "
                                                    f"{fe(min(r5[r]['mlp'][0] for r in rk))}–{fe(max(r5[r]['mlp'][1] for r in rk))}")
                + "；§1 标题为" + orig("结论摘要") + ("。" if "\n## 1 结论摘要\n" in published else fail("section 1 heading changed")),
                "§1 是测量结果的摘要，不是方法优劣结论（§1 精度条，决定 §5）。合并条件的范围掩盖了条件差异，按 ρ 分开：" + f"{len(D['mf_methods'])} 个 K{K} MF 方法的 RMSE 在 "
                + "、".join(f"ρ{r:g} 为 {rng(r5[r]['mf'])}" for r in rk) + f"；{len(D['methods']) - len(D['mf_methods'])} 个 MLP 方法在 " + "、".join(f"ρ{r:g} 为 {rng(r5[r]['mlp'])}" for r in rk)
                + "。逐条件的配对差 MF_VCS_M0 − MLP_VCS_perm1（§4.2）" + "、".join(f"在 two_block ρ{r:g} 的 {r5[r]['n']} 个条件{kind_txt[r5[r]['kind']]}" for r in rk)
                + "；方法间比较以 §4.2 的逐条件配对差为准。",
                "§4.1 与 §4.2。")
    by6 = c6["by"]
    out += item(6, "fp64 诊断与决定 §2.3 的条件",
                "（§8）：" + orig("按决定 §2.3 这只是诊断，训练仍用 fp32，其对所选 critic 的影响未单独测量") + "。",
                Q("2.3", "对接近 fp32 分辨能力的参考值，保留一个同输入 fp64 数值诊断；没有实际失败证据时不把主训练整体改成 fp64")
                + f"。{c6['tot']} 个不一致步按 ρ：" + "，".join(f"ρ = {rho:g} 为 {sum(v[1] for v in d.values())} 步（" + "，".join(f"cap {cap}：{v[1]}" for cap, v in d.items()) + "）"
                                                          for rho, d in by6.items())
                + "。这些不一致是否构成“实际失败证据”，本报告不作判定，留待 owner；训练仍用 fp32，其对所选 critic 的影响未单独测量。",
                f"{c6['n_m1']} 个正式 MF_VCS_M1 单元记录的 fp64_diagnostic（§5.3）。")
    out += item(7, f"“{c7['checks']} 个评估表”是核对次数",
                "（§5.1）：" + orig(f"{c7['n_units']} 个单元共 {c7['checks']} 个表") + "；（§1 误差分解条）：" + orig(f"在 {c7['checks']} 个评估表上成立") + "。",
                f"{c7['checks']} 是恒等式的核对次数：每个单元 {c7['per_unit']} 次（所选 checkpoint 的表、{c7['n_traj_ckpt']} 个轨迹 checkpoint 的表与 {c7['n_sat_modes']} 张饱和表 U/V，各核对一次）。"
                f"所选 checkpoint 的表在 {c7['sel_in_traj']}/{c7['n_units']} 个单元中就是轨迹里的一张；饱和表按（数据 root，条件）共 {c7['sat_keys']} 组、每组 {c7['n_sat_modes']} 张，在同组单元间重复。"
                f"不同的表共 {c7['n_traj']} + {c7['n_sat_modes'] * c7['sat_keys']} = {c7['distinct']} 张；最大残差与全部 PASS 的结论不变。",
                "各单元 evaluation.json 的 selected_metrics、trajectory_diagnostic 与 saturated_fit_table。")
    out += item(8, "头 gap 的计算位置与 fp32 / fp64",
                "（§5.2 表注）：" + orig("头 gap 在所选 checkpoint 的特征上、外层更新之后计算；FW gap 是头部次优值的上界") + "；" + orig("故真实次优值位于 [polish 下界, FW gap] 之间") + "。",
                f"(a) §5.2 与 §7 表中的 FW gap 为 fp32 值；同输入 fp64 值与之相差不超过 {fpct_up(c8['rel'])}（{c8['n']} 个正式 MF 单元），区间上界取 fp64 FW gap，"
                f"{c8['nb']}/{c8['n']} 个单元的 polish 下界不超过 fp64 FW gap。(b) 这是所选 checkpoint 特征（外层更新之后）上的诊断量，不是 M1 训练中 solver 的停止 gap；后者是训练日志中的量（"
                + Q("2.3", "头 gap 对应本步外层更新前的特征") + f"；M1 单元记录 gap_scope 为 {c8['m1_scope']}）。§1 误差分解条中"
                + orig(f"所选 checkpoint 处 MF_VCS_M1 的头 gap 均值低于 MF_VCS_M0 的条件为 {c8['gm1']}/{c8['n_cond']}") + "比较的是前者。",
                "各正式 MF 单元 evaluation.json 的 head_gap（fp32、fp64、fp64_polish）与 MF_VCS_M1 单元 result.json 的 m1.gap_scope。")
    out += item(9, "§5.5 “V 参考 EVAL − S 均值”列的含义",
                "（§5.5 表头）：" + orig("V 参考 EVAL − S 均值") + "。",
                f"该列是用 q̂^V 构造的饱和表在 EVAL 上的完整 U-J − S，不是 EVAL V-J − S。若改用 EVAL V-J − S，列均值逐条件最多相差 {fe(c9['abs'])}（{lab(c9['abs_c'])}），"
                f"相对该列数值最多 {fpct_up(c9['rel'], 1)}（{lab(c9['rel_c'])}）；表中数值本身不变。",
                "各单元 evaluation.json 的 saturated_fit_table.V（EVAL_UJ 与 EVAL_VJ；同组副本逐字段一致）。")
    out += item(10, f"§1 算法波动的 R = {c10['R']} 限定",
                 "（§1 算法波动与容量条）：" + orig(f"固定首套数据、R = {c10['R']} 个算法 seed 的 sd 大于同格数据重复 sd 的有 {c10['nalg']}/{c10['n']} 格") + "。",
                 f"补充 §6 的限定：R = {c10['R']} 时 sd 的 95% χ² 区间上限约为下限的 {c10['ratio']:.0f} 倍，{c10['nalg']}/{c10['n']} 只作粗略比较。",
                 "§6。")
    out += item(11, "RuLSIF 状态的日期",
                 "（§8）：" + orig("RuLSIF / 核精确矩尚未实现，本网格不含其二次风险对照") + "。",
                 f"原句指本报告日期（{c11['date']}）的状态；所指基线见"
                 + Q("5", "实现所声明参考下的精确矩；V9 的 RFF ridge perm1 不充当该基线")
                 + "。此后的实现进展以相应记录为准，不在本报告范围内；本网格不含 RuLSIF 对照这一点不变。",
                 f"报告文件名与正式结果提交 `{D['git']['formal']['commit'][:7]}` 的日期（{c11['date']}）。")
    ng = c12["negs"]
    out += item(12, "§2.4 成本与训练实现方式",
                 "（§8）：" + orig("§2.4 的成本只是本网格、本 CPU 布局下的实测") + "。",
                 f"补充实现方式：本网格训练没有按单元频数聚合。排列方法每步按样本行计算若干组 n = {c12['n_fit']} 个负样本（"
                 + "、".join(f"{m} {ng[m]} 组" for m in D["methods"] if ng.get(m)) + "），"
                 + " / ".join(m for m in D["methods"] if ng.get(m, 0) is None) + " 由样本行的特征计算完整 U 矩；按单元频数的表统计只用于 CAL 选模与各项读数。"
                 f"§2.4 按方法的每单元耗时（{c12['per'][0]:.1f}–{c12['per'][1]:.1f} s）反映这一实现，不是函数类之间的计算成本比较（"
                 + Q("9.5", "离散场景中 MLP 同样可以按单元频数聚合") + "）。",
                 f"运行提交的 `{OBJECTIVES_REL}`：“{c12['frag']}”；单元记录的每步负样本数（MLP 的 negatives_per_step、MF_VCS_perm1 的排列抽取数）。")
    out.append("")
    return out


# ------------------------------------------------------------------------------------------------ rendering
def render(D) -> str:
    Q = Quotes(D["quotes_text"])
    head = sec_header(D)
    body = []
    body += sec_terms(D)
    body += sec_summary(D, Q)
    body += sec_impl(D, Q)
    body += sec_cal(D, Q)
    body += sec_main(D, Q)
    body += sec_decomp(D, Q)
    body += sec_algo(D, Q)
    body += sec_capacity(D, Q)
    body += sec_limits(D, Q)
    body += sec_files(D)
    # the report as first published (header + terms + sections 1-9) must be reproduced byte for byte; section 10 is appended
    published = "\n".join(head + body).rstrip("\n") + "\n"
    check(published.encode("utf-8") == git_bytes("show", f"{PUBLISHED_COMMIT}:{REPORT_REL}"),
          f"sections before 10 differ from the report published at {PUBLISHED_COMMIT[:7]}")
    note = (f"**{CORRECTIONS_DATE} 追加 §10 更正**：逐条补充 §1、§5、§8 若干语句的依据、限定与澄清。除本行外，§10 之前的内容与首次发布的提交 "
            f"`{PUBLISHED_COMMIT[:7]}` 逐字节相同（生成器核对），全部结果未改动；§10 的数值同样由生成器从记录计算并核对。")
    out = head[:2] + [note, ""] + head[2:] + body + sec_corrections(D, Q, published)
    return "\n".join(out).rstrip("\n") + "\n"


def sec_header(D) -> list[str]:
    g = D["git"]
    ids = "、".join(f"{BLOCKS[b]['label']} `{g[b]['commit'][:7]}`（{g[b]['utc']}，作业 {g[b]['job']}）" for b in BLOCKS)
    return [
        "# V10 T03 合成真值实验结果报告（U 模式；owner 决定 2026-10-17 §9）",
        "",
        f"生成器 `{SELF_REL}`，分支 `{D['jobs']['formal']['branch']}`。只读已提交的 T03 记录，结果提交为 {ids}。生成器核对：记录路径（"
        + "、".join(f"`{p}`" for p in RECORD_PATHS) + "）无未提交或未跟踪的改动；三个块目录自各自的结果提交后未再改动；每个块目录的已跟踪文件"
        "恰为其结果 manifest 所列文件加 manifest 本身，sha256 与字节数逐一相符；单元表 sha256 等于协议 P2 的 pin。报告不含生成时刻，也不做在线查询；"
        f"两次运行逐字节相同；`/home/infres/yinwang/CS_QMI/env/bin/python {SELF_REL} --check` 与 HEAD 中已提交的本报告比较。",
        "",
        f"来源：数值、表格与计数都由生成器从记录计算。§4–§7 中聚合文件 `{AGGREGATE_REL}`（sha256 `{D['aggregate_sha256'][:16]}`…，由已提交的 "
        f"`{AGGREGATOR_REL}` 生成，sha256 `{D['aggregator_sha256'][:16]}`…）也含有的格值，生成器另从逐单元记录重算并逐格核对一致（相对误差 ≤ 1e-12），"
        "表中显示聚合值；CAL 选模由 CAL 单元的 result.json（不含真值）重算并与选模记录逐组核对。手写内容只有：文字说明、术语定义、表头，以及 §8 的限制条目"
        "（其中引用的决定原文由生成器在所注小节内逐字核对）。依赖数据的语句为固定文字，其成立条件由生成器核对，不成立则拒绝生成。",
        "",
    ]


def sec_terms(D) -> list[str]:
    n = D["n_obs"]
    rows = [
        ("方法", f"MF：每侧 Linear + softmax 特征 a(x)、b(y)（主表 K = {D['K_main']}），T = a(x)ᵀ W b(y)，W ∈ [−1, 1]^(K×K)。MF_VCS_perm1：每步 1 个均匀无固定点排列作负样本，"
                 "特征与 W 由 AdamW 更新；MF_VCS_M0：完整 U 矩目标，特征与 W 由 AdamW 更新；MF_VCS_M1：完整 U 矩目标，W 每步由头 solver 求解（相对零头 gap 规则与 cap），"
                 "特征由 AdamW 更新。MLP：两侧 one-hot 拼接的三层 MLP，T = tanh f；MLP_VCS_perm1 / perm8：每步 1 / 8 个均匀无固定点排列（生成器按单元记录核对）"),
        ("S", "包生成器给出的总体真值。对任一 critic 表 T，S − J_pop(T) = E_M(T − η)² ≥ 0，所以 S 是 J_pop 的最大值，在 T = η 处取到"),
        ("p、q、M、η", "p 为联合分布 P_XY 的单元概率，q 为 Q = P_X P_Y；M = (P + Q)/2；η = (p − q)/(p + q)"),
        ("J(T)", "E_P T − E_Q T − (E_P T² + E_Q T²)/2；J_pop 为总体值"),
        ("完整经验 U-J", "用样本内全部 i ≠ j 配对作负参考 Q̂_U 的经验 J；负值保留"),
        ("EVAL / CAL / FIT U-J", f"同一冻结 critic 在 EVAL（{n['EVAL']} 个观测）/ CAL（{n['CAL']}）/ FIT（{n['FIT']}）样本上的完整经验 U-J"),
        ("误差 e、偏差、sd、RMSE", "每个重复的 e = EVAL U-J − S；偏差 = e 的均值；sd = e 的样本标准差；RMSE = √(e² 的均值)"),
        ("数据重复", f"一个 data root seed 生成的一整套独立 FIT/CAL/EVAL；主表每格 {len(D['data_seeds'])} 个，算法 seed 固定"),
        ("算法重复", "同一套数据只换算法 seed（初始化与排列流），只用于 §6"),
        ("总体逼近误差", "E_M(T − η)²，evaluator 枚举全部 m² 个单元计算"),
        ("EVAL 抽样误差", "EVAL U-J − J_pop(T)；于是 e = EVAL 抽样误差 − 总体逼近误差"),
        ("头 gap", "MF 固定特征经验头问题（FIT 全批、U 模式、ridge 0、特征取所选 checkpoint）在 W 处的 Frank–Wolfe gap，是头部次优值的上界；"
                  "G(0) 为 W = 0 时的 gap；fp64 polish 给出次优值下界"),
        ("排列条件方差", f"固定所选 critic 与同一份基础样本，用 {D['perm_z']['R']} 个均匀无固定点排列（同一数据上各方法共用的排列流）各算一次 perm1 估计所得的方差"),
        ("饱和表", "只用 FIT 计数构造的全表 T*（决定 §9.5 公式，分母为零取零），作为 FIT 全表函数类的经验最优参考"),
        ("区间", "偏差：95% t 区间；sd：由方差 95% χ² 区间开方；RMSE：百分位 bootstrap（B = "
                f"{D['boot']['B']}，seed {D['boot']['seed']}，按重复重采样）。“含 0”= 区间包含 0"),
    ]
    return ["## 术语", ""] + md_table(["术语", "本报告中的含义"], rows, "ll") + [""]


def _ranges(D, conds, methods, f):
    vals = [f(D["main"][(c, m)]) for c in conds for m in methods]
    return min(vals), max(vals)


def sec_summary(D, Q) -> list[str]:
    conds, methods, mf = D["conds"], D["methods"], D["mf_methods"]
    mlp = [m for m in methods if m not in mf]
    main = D["main"]
    R = len(D["data_seeds"])
    c0 = [c for c in conds if c[2] == 0]
    tbp = [c for c in conds if c[0] == "two_block" and c[2] > 0]
    dmp = [c for c in conds if c[0] == "diagonal_mixture" and c[2] > 0]
    check(all(main[(c, methods[0])]["S"] == 0 for c in c0) and all(main[(c, methods[0])]["S"] > 0 for c in tbp + dmp), "S = 0 exactly at rho = 0")
    out = ["## 1 结论摘要", ""]
    nb = D["n_by_block"]
    out.append(f"- **测量对象**：包生成器的 {len(conds)} 个离散合成条件（two_block、diagonal_mixture × m = 8、32 × ρ = 0、0.25、0.75）上，"
               f"{len(methods)} 种 critic 训练方法（{'、'.join(methods)}）所选冻结 critic 的 EVAL U-J 相对总体真值 S 的误差。主表每格 R = {R} 个数据重复"
               f"（data root {D['data_seeds'][0]}–{D['data_seeds'][-1]}，算法 seed {D['algo_main']}）；另有固定首套数据的算法重复（seed "
               f"{'、'.join(map(str, D['algo_extra']))}）与容量诊断（K {'、'.join(map(str, D['cap_K']))}）。网格 {D['proto']['counts']['total_excluding_smoke_and_followup_baselines']} 个单元"
               f"（CAL {nb['CAL']}、主 {nb['main']}、算法 {nb['algo_extra']}、容量 {nb['capacity']}）与 smoke {nb['smoke']} 个单元全部 COMPLETE，评估全部 PASS，真值隔离拒绝 0 次。")
    out.append(f"- **精度**：每格只有 R = {R} 个数据重复，偏差的 t 区间（自由度 {R - 1}）、sd 的 χ² 区间和 RMSE 的 bootstrap 区间都较宽；R = {R} 时百分位 "
               "bootstrap 只是粗略范围（小样本下通常偏窄）。区间逐格给出，未做多重比较校正。以下差异只描述本合成网格在固定训练规则下的测量，"
               f"不是方法优劣结论（{Q('5', '必需基线补齐之前不发布方法优劣、强基线优势或新颖性比较结论')}）。")
    # rho = 0
    mx_b = max(abs(main[(c, m)]["EVAL_UJ_minus_S"]["mean"]) for c in c0 for m in methods)
    mx_r = max(main[(c, m)]["EVAL_UJ_minus_S"]["rmse"] for c in c0 for m in methods)
    n0 = sum(has0(main[(c, m)]["EVAL_UJ_minus_S"]["mean_ci95_t"]) for c in c0 for m in methods)
    excl0 = [(c, m) for c in c0 for m in methods if not has0(main[(c, m)]["EVAL_UJ_minus_S"]["mean_ci95_t"])]
    check(all(main[k]["EVAL_UJ_minus_S"]["mean"] < 0 for k in excl0), "rho0 cells excluding 0 are all negative")
    ng, nt = D["neg"]["S0"]
    ng1, nt1 = D["neg"]["S+"]
    out.append(f"- **ρ = 0（S = 0）的 {len(c0) * len(methods)} 格**：|偏差| ≤ {fe(mx_b)}，RMSE ≤ {fe(mx_r)}；{n0}/{len(c0) * len(methods)} 格偏差区间含 0，"
               f"其余 {len(excl0)} 格（{'；'.join(f'{cond_label(c)} {m}' for c, m in excl0)}）偏差为负。{ng}/{nt} 个 ρ = 0 主单元与 {ng1}/{nt1} 个 ρ > 0 主单元的 "
               f"EVAL U-J 为负，全部保留（{Q('9.5', '主 EVAL U-J 的负数保留，不截成零')}）。")
    # two_block rho > 0
    lo_mf, hi_mf = _ranges(D, tbp, mf, lambda c: c["EVAL_UJ_minus_S"]["rmse"])
    lo_ml, hi_ml = _ranges(D, tbp, mlp, lambda c: c["EVAL_UJ_minus_S"]["rmse"])
    t75 = [c for c in tbp if c[2] == 0.75]
    t25 = [c for c in tbp if c[2] == 0.25]
    n75 = sum(has0(main[(c, m)]["EVAL_UJ_minus_S"]["mean_ci95_t"]) for c in t75 for m in methods)
    n25 = sum(main[(c, m)]["EVAL_UJ_minus_S"]["mean_ci95_t"][1] < 0 for c in t25 for m in methods)
    out.append(f"- **two_block、ρ > 0**：K16 MF 三法 RMSE {fe(lo_mf)}–{fe(hi_mf)}，MLP 两法 {fe(lo_ml)}–{fe(hi_ml)}。ρ = 0.75 的 {frac(n75, len(t75) * len(methods))} 格偏差区间含 0；"
               f"ρ = 0.25 的 {frac(n25, len(t25) * len(methods))} 格偏差为负且区间不含 0（S = {fS(main[(t25[0], methods[0])]['S'])}）。")
    # diagonal_mixture rho > 0
    b_mf = _ranges(D, dmp, mf, lambda c: c["EVAL_UJ_minus_S"]["mean"])
    r_mf = _ranges(D, dmp, mf, lambda c: c["EVAL_UJ_minus_S"]["mean"] / c["S"])
    b_ml = _ranges(D, dmp, mlp, lambda c: c["EVAL_UJ_minus_S"]["mean"])
    r_ml = _ranges(D, dmp, mlp, lambda c: c["EVAL_UJ_minus_S"]["mean"] / c["S"])
    share = {(c, m): main[(c, m)]["pop_mse"]["mean"] / -main[(c, m)]["EVAL_UJ_minus_S"]["mean"] for c in dmp for m in mf}
    sh_hi = [v for v in share.values() if v >= 0.9]
    sh_lo = {k: v for k, v in share.items() if v < 0.9}
    lo_conds = sorted({k[0] for k in sh_lo}, key=conds.index)
    check(b_mf[1] < 0 and b_ml[1] < 0 and sh_hi and min(share.values()) >= 0.5,
          "diagonal rho>0: MF bias negative and mainly the population approximation error")
    if sh_lo:
        check(all(main[k]["EVAL_UJ_minus_J_pop"]["mean"] < 0 for k in sh_lo), "low-share cells: negative sampling-error mean")
    share_txt = (f"E_M(T − η)² 均值 / |偏差| 在 {len(sh_hi)} 组为 {min(sh_hi):.2f}–{max(sh_hi):.2f}"
                 + (f"，在 {'、'.join(cond_label(c) for c in lo_conds)} 的 {len(sh_lo)} 组为 {min(sh_lo.values()):.2f}–{max(sh_lo.values()):.2f}"
                    "（差额为同号的 EVAL 抽样误差均值，§5.2）" if sh_lo else ""))
    allci = all(main[(c, m)]["EVAL_UJ_minus_S"]["mean_ci95_t"][1] < 0 for c in dmp for m in methods)
    check(allci, "diagonal rho>0: every bias interval below 0")
    bnd = [(c, m) for c in dmp for m in mf if D["selection"][(c, m)]["lr"] == max(D["lrs"]) and D["selection"][(c, m)]["step"] == max(D["steps"])]
    own_last = sum(u["ev"]["selected"]["step"] == max(D["steps"]) for c in dmp for m in mf for u in main[(c, m)]["units"])
    own_n = len(dmp) * len(mf) * R
    out.append(f"- **diagonal_mixture、ρ > 0**：全部 {len(dmp) * len(methods)} 格偏差为负且区间不含 0。K16 MF 三法偏差 {fe(b_mf[0])} 至 {fe(b_mf[1])}（相对 S 为 "
               f"{fpct(r_mf[0])} 至 {fpct(r_mf[1])}），MLP 两法 {fe(b_ml[0])} 至 {fe(b_ml[1])}（{fpct(r_ml[0])} 至 {fpct(r_ml[1])}）。MF 的偏差主要是总体逼近误差："
               f"{share_txt}。这 {len(dmp) * len(mf)} 个 MF 组的 CAL 选模中 {frac(len(bnd), len(dmp) * len(mf))} 选中学习率网格上端 "
               f"{flr(max(D['lrs']))} 与最后一个 checkpoint（{max(D['steps'])} 步），正式重复自身 CAL 选中最后 checkpoint 的为 {own_last}/{own_n}：选择落在固定步数与学习率网格的边界，"
               "这些 MF 数值是本训练规则下的结果，不是 K16 MF 函数类可达到的最小误差。")
    # paired
    parts = []
    for a, b in D["pairs"]:
        cells = [D["paired"][(c, a, b)]["sq"] for c in conds]
        neg_ = sum(x["mean_ci95_t"][1] < 0 for x in cells)
        pos_ = sum(x["mean_ci95_t"][0] > 0 for x in cells)
        parts.append(f"{a} − {b}：{len(conds) - neg_ - pos_} 个条件含 0，{neg_} 个为负、{pos_} 个为正")
    out.append(f"- **方法配对**（同一数据重复上的平方误差差 e_a² − e_b²，t 区间；负 = a 的平方误差较小）：" + "；".join(parts) + "。条件与数值见 §4.2。")
    # decomposition
    nsamp = sum(has0(main[(c, m)]["EVAL_UJ_minus_J_pop"]["mean_ci95_t"]) for c in conds for m in methods)
    ex = collections.defaultdict(list)
    for c in conds:
        for m in methods:
            ci = main[(c, m)]["EVAL_UJ_minus_J_pop"]["mean_ci95_t"]
            if not has0(ci):
                ex[c].append(1 if ci[0] > 0 else -1)
    full = [c for c in conds if len(ex.get(c, [])) == len(methods) and len(set(ex[c])) == 1]
    n_ex_conds = sum(1 for c in conds if ex.get(c))
    samp_txt = (f"；不含 0 的 {sum(len(v) for v in ex.values())} 格分布在 {n_ex_conds} 个条件中，其中 {'、'.join(cond_label(c) for c in full)} 的 "
                f"{len(methods)} 个方法全部同号——同一数据重复的各方法共用同一 EVAL 样本，抽样误差在方法间相关，这些格不是独立事件") if full else ""
    gm1 = sum(main[(c, "MF_VCS_M1")]["gap"]["mean"] < main[(c, "MF_VCS_M0")]["gap"]["mean"] for c in conds)
    pr = [math.sqrt(main[(c, m)]["perm_eval_var_mean"]) / main[(c, m)]["EVAL_UJ_minus_J_pop"]["sd"] for c in conds for m in methods]
    pz = D["perm_z"]
    out.append(f"- **误差分解**：E_M(T − η)² = S − J_pop(T) 在 {D['identity']['n']} 个评估表上成立（最大 |残差| {fe(D['identity']['max_abs'], 2)} ≤ {fe(D['identity_atol'], 1)}）。"
               f"EVAL 抽样误差的均值区间含 0 的有 {nsamp}/{len(conds) * len(methods)} 格{samp_txt}。perm1 均值相对完整 U-J 的 z 值（{pz['n']} 个）均值 {pz['mean']:.3f}、sd {pz['sd']:.3f}，"
               f"|z| > 3 的 {pz['n_gt3']} 个（标准正态期望 {pz['expected_gt3']:.1f} 个）；若改用单个 perm1 排列读数，其在 EVAL 上的排列条件 sd 为同格 EVAL 抽样 sd 的 "
               f"{min(pr):.2f}–{max(pr):.2f} 倍（主读数用完整 U-J，无此项）。同一冻结 critic 的 |EVAL U − V| 最大 {fe(max(v['max'] for v in D['uv'].values()))}，"
               f"不超过同格 e 的 sd 的 {fpct(max(v['max_over_sd'] for v in D['uv'].values()), 1)}。所选 checkpoint 处 MF_VCS_M1 的头 gap 均值低于 MF_VCS_M0 的条件为 {gm1}/{len(conds)}。")
    # saturated
    s8 = [D["sat"][c] for c in conds if c[1] == 8]
    s32 = [D["sat"][c] for c in conds if c[1] == 32]
    e8 = (min(x["eval_err"]["mean"] for x in s8), max(x["eval_err"]["mean"] for x in s8))
    e32 = (min(x["eval_err"]["mean"] for x in s32), max(x["eval_err"]["mean"] for x in s32))
    f32 = (min(x["fit_minus_S"]["mean"] for x in s32), max(x["fit_minus_S"]["mean"] for x in s32))
    check(f32[0] > 0 and e32[1] < 0, "saturated m32: FIT above S and EVAL below S")
    out.append(f"- **饱和表**（FIT 全表函数类的经验最优参考，非训练所得）：m = 32 时 EVAL U-J − S 均值 {fe(e32[0])} 至 {fe(e32[1])}，同一表的 FIT U-J − S 为 "
               f"{fe(f32[0])} 至 {fe(f32[1])}（FIT 上偏高、EVAL 上偏低）；m = 8 时 EVAL 均值 {fe(e8[0])} 至 {fe(e8[1])}。{D['sat_units']} 个单元记录中的饱和表按（数据 root，条件）归并为 {D['sat_keys']} 张，同组副本完全一致（冲突 0）。")
    # algo and capacity
    nalg = sum(D["algo"][k]["EVAL_UJ_minus_S"]["sd"] > main[k]["EVAL_UJ_minus_S"]["sd"] for k in main)
    cp = D["cap_paired"]
    Kmin, Kmax = min(D["cap_K"]), max(D["cap_K"])
    dm_c = [c for c in D["cap_conds"] if c[0] == "diagonal_mixture"]
    tb_c = [c for c in D["cap_conds"] if c[0] == "two_block"]
    neg_dm = {K: sum(cp[(c, m, K)]["mean_ci95_t"][1] < 0 for c in dm_c for m in mf) for K in D["cap_K"]}
    pos_dm = {K: sum(cp[(c, m, K)]["mean_ci95_t"][0] > 0 for c in dm_c for m in mf) for K in D["cap_K"]}
    inc_tb = sum(has0(cp[(c, m, K)]["mean_ci95_t"]) for c in tb_c for m in mf for K in D["cap_K"])
    out.append(f"- **算法波动与容量**：固定首套数据、R = {1 + len(D['algo_extra'])} 个算法 seed 的 sd 大于同格数据重复 sd 的有 {nalg}/{len(main)} 格（两者分表，不合并）。"
               f"固定训练规则下的容量响应（与 K16 配对的平方误差差，负 = 该 K 的平方误差较小）：diagonal_mixture 的 {len(dm_c) * len(mf)} 组中 K{Kmax} 有 {neg_dm[Kmax]} 组为负、{pos_dm[Kmax]} 组为正，"
               f"K{Kmin} 有 {pos_dm[Kmin]} 组为正、{neg_dm[Kmin]} 组为负；two_block 的 {len(tb_c) * len(mf) * len(D['cap_K'])} 个对比中 {inc_tb} 个含 0（§7）。")
    # cost
    j = D["jobs"]
    grid_wall_s = j["CAL"]["wall_s"] + j["formal"]["wall_s"]
    alloc = (j["CAL"]["cpus"] * j["CAL"]["wall_s"] + j["formal"]["cpus"] * j["formal"]["wall_s"]) / 3600
    ph = (D["cost_blocks"]["CAL"]["wall"] + D["cost_blocks"]["formal"]["wall"]) / 3600
    L = D["budget"]["layout"]
    om = D["budget"]["om"]["scaled_to_current_grid"]["cpu_hours_by_cap"]
    check(alloc <= 10 * om["100"], "over-budget rule")
    out.append(f"- **成本**：网格 {len(BLOCKS) - 1} 个作业（CAL、正式）共 {grid_wall_s} s 墙钟（{fh(grid_wall_s / 3600, 3)} h），分配 {fh(alloc)} CPU-h；单元墙钟合计 {fh(ph)} 进程小时，"
               f"为 smoke 投影（{fh(L['process_hours']['low'])}–{fh(L['process_hours']['high'])}）的 {ph / L['process_hours']['high']:.2f}–{ph / L['process_hours']['low']:.2f} 倍；"
               f"低于第一阶段数量级换算到本网格的 {om['10']:.1f}–{om['100']:.1f} CPU-h，未触发超预算规则（§2.4）。")
    out.append("")
    return out


def sec_impl(D, Q) -> list[str]:
    j, g = D["jobs"], D["git"]
    out = ["## 2 实施与来源", "", "### 2.1 作业与代码身份", ""]
    rows = []
    for b in BLOCKS:
        J = j[b]
        rows.append((BLOCKS[b]["label"], J["job"], f"{J['partition']}，{J['node']}", f"{J['cpus']}；{J['mode']}；torch 线程 {J['threads']}",
                     f"`{J['run_commit'][:7]}`", f"`{J['code']}` / `{J['pin']}`", f"{J['start']} – {J['end']}", J["wall_s"], J["units"],
                     f"`{g[b]['commit'][:7]}`"))
    out += md_table(["块", "作业", "分区，节点", "CPU；模式；线程", "运行提交", "T03 code hash / 作业 pin", "作业起止（UTC）", "墙钟 s", "单元", "结果提交"],
                    rows, "lllllllrrl")
    out += ["", f"code hash 覆盖 {' + '.join(j['formal']['code_paths'])} 的已跟踪文件，作业 pin 另含 {'、'.join(p for p in j['formal']['pin_paths'] if p not in j['formal']['code_paths'])}；"
            "两者均由生成器在各运行提交的 git 对象上重算，等于作业日志头与每个单元的 provenance。三个块的 code hash 相同；smoke 与 CAL 的 pin 不同："
            f"两个运行提交 `{D['pin_change']['from'][:7]}` → `{D['pin_change']['to'][:7]}` 之间 pin 路径只有 "
            + "、".join(f"{ {'A': '新增', 'M': '修改', 'D': '删除'}.get(st[0], st)} `{fn}`" for st, fn in D["pin_change"]["files"]) + "（提交 " + "、".join(f"`{h[:7]}`" for h, _ in D["pin_change"]["commits"])
            + "），code hash 路径无改动；CAL 与正式运行提交之间 pin 路径无改动。CAL 作业运行于 smoke 结果提交，正式作业运行于 CAL 结果提交，"
            "所以 CAL 选模记录在正式单元运行前已提交。全部单元在干净工作树上运行（git_dirty 为空，作业头 dirty_files=0）。"
            f"smoke 作业内单元测试 {j['smoke']['tests_passed']} 项通过。T03 在独立分支 `{j['formal']['branch']}` 上运行（{Q('10', '并行开展的 T03 与基线开发使用独立 worktree 或固定提交')}），"
            f"顺序为 smoke → CAL → 正式（{Q('10', '并行编写 T03，先生成检查与 smoke，再按已定网格实施')}）。", ""]
    # gates
    gf = D["gate_facts"]
    out += ["### 2.2 门控与检查", ""]
    G = D["gates"]
    gl = {"G1_enumeration": f"小 n 穷举（n = {'、'.join(map(str, gf['G1_n']))}；{'、'.join(gf['G1_critics'])}）：全部无固定点排列的平均值与梯度 = 完整 U；"
                            f"最大误差 值 {fe(gf['G1_value_err'], 2)}、梯度 {fe(gf['G1_grad_err'], 2)}（容差 atol {fe(gf['G1_tol']['atol'], 1)}）",
          "G2_sampler_law": f"抽样分布：n = {'、'.join(map(str, gf['G2_n']))} 各 {'、'.join(map(str, gf['G2_draws']))} 次，固定点 {gf['G2_fixed']} 个，χ² 最小 p = "
                            f"{gf['G2_p_min']:.3f}（预设 ≥ {gf['G2_pre']['chi2_p_min']}），边缘 |z| 最大 {gf['G2_z_max']:.2f}（预设 ≤ {gf['G2_pre']['max_abs_z']}）",
          "G3_rejection_rate": f"FIT 规模 n = {gf['G3']['n']} 的拒绝率：{gf['G3']['draws']} 次平均尝试 {gf['G3']['mean_attempts']:.4f}（期望 {gf['G3']['expected']:.4f}，z = {gf['G3']['z']:.2f}）",
          "G4_perm8_mean": f"perm8 = 其 8 个 perm1 目标的均值：最大误差 {fe(gf['G4_err'], 2)}",
          "G5_table_U": f"计数表 U/V J = 后端 U 矩（MF）/ 显式全配对（MLP）：最大误差 {fe(gf['G5_err'], 2)}"}
    rows = [(k.split("_")[0], *(G[b]["gates"][k] for b in BLOCKS), gl[k]) for k in G["formal"]["gates"]]
    rows.append(("包检查器", *(f"{G[b]['checker'][0][2]}/{G[b]['checker'][0][1]} × {len(G[b]['checker'])} 套" if len({c[1:] for c in G[b]['checker']}) == 1 else "见记录"
                               for b in BLOCKS), "包 synthetic/tests/check_synthetic.py 对每套生成数据的全部检查（种子：smoke "
                 f"{G['smoke']['checker'][0][0]}、CAL {G['CAL']['checker'][0][0]}、正式 {G['formal']['checker'][0][0]}–{G['formal']['checker'][-1][0]}）"))
    rows.append(("单元表", *(G[b]["units_check"]["status"] for b in BLOCKS), f"{G['formal']['units_check']['rows_file']} 行与决定 §9 的独立推导逐字节相同，sha256 = 协议 pin"))
    out += md_table(["检查", *(BLOCKS[b]["label"] for b in BLOCKS), "内容与要点（正式作业记录）"], rows, "lllll")
    out += ["", f"均匀无固定点排列：训练中 {D['derangements']['units']} 个排列方法单元共抽取 {D['derangements']['draws']} 个，平均每个尝试 "
            f"{D['derangements']['mean_attempts']:.4f} 次均匀全排列（期望约 e = {math.e:.4f}）。", ""]
    # truth barrier
    out += ["### 2.3 真值隔离", "", Q("9.2", "truth 仅交 evaluator；训练和 CAL 选择只读样本") + "。", ""]
    M = D["manifests"]
    rows = [(BLOCKS[b]["label"], j[b]["units"], j[b]["denials"], M[b]["leak_needles"], M[b]["leak_found_inside"], M[b]["leaks"], M[b]["files"], M[b]["kept_outside"])
            for b in BLOCKS]
    out += md_table(["块", "单元", "真值隔离拒绝", "泄漏扫描真值数", "仅在 evaluator_only 中找到", "evaluator_only 之外", "已提交文件", "留在 /projects 的文件（sha256 在 manifest）"],
                    rows, "lrrrrrrr")
    out += ["", f"每个单元记录的隔离范围为“{D['barrier_scope']}”；CAL 组选择在运行提交的 t03_run.py 中同样处于隔离块内，出现拒绝的组会记为 BLOCKED，"
            f"本次 {len(D['selection'])}/{len(D['selection'])} 组为 SELECTED。评估与包检查器在隔离外运行，其含真值输出只写在 evaluator_only 路径下。"
            "泄漏扫描以各单元的 S、J_pop、EVAL U-J 等值的 repr 在每个复制文件中搜索。", ""]
    # cost
    out += ["### 2.4 实际成本与投影", ""]
    cb = D["cost_blocks"]
    L, W, om = D["budget"]["layout"], D["budget"]["wall"], D["budget"]["om"]
    grid_alloc = sum(j[b]["cpus"] * j[b]["wall_s"] for b in ("CAL", "formal")) / 3600
    grid_wall = sum(j[b]["wall_s"] for b in ("CAL", "formal"))
    grid_ph = (cb["CAL"]["wall"] + cb["formal"]["wall"]) / 3600
    grid_cpu = (cb["CAL"]["cpu"] + cb["formal"]["cpu"]) / 3600
    rows = []
    for b in BLOCKS:
        rows.append((BLOCKS[b]["label"], cb[b]["n"], j[b]["wall_s"], fh(j[b]["cpus"] * j[b]["wall_s"] / 3600, 3), fh(cb[b]["wall"] / 3600, 3), fh(cb[b]["cpu"] / 3600, 3),
                     fh(cb[b]["eval"] / 3600, 3)))
    rows.append(("网格合计（CAL + 正式）", cb["CAL"]["n"] + cb["formal"]["n"], grid_wall, fh(grid_alloc, 3), fh(grid_ph, 3), fh(grid_cpu, 3), fh((cb["CAL"]["eval"] + cb["formal"]["eval"]) / 3600, 3)))
    out += md_table(["块", "单元", "作业墙钟 s", "分配 CPU-h（CPU 数 × 墙钟）", "单元墙钟合计 h（进程小时）", "单元 CPU 时间合计 h", "其中评估墙钟 h"], rows, "lrrrrrr")
    sc = om["scaled_to_current_grid"]["cpu_hours_by_cap"]
    out += ["", f"投影（`{BUDGET_REL}`，布局 {L['name']}：{L['processes']} 进程 × {L['threads_per_process']} 线程，与网格作业每进程 {D['budget']['threads']} 线程相同）："
            f"网格 {fh(L['process_hours']['low'])}–{fh(L['process_hours']['high'])} 进程小时 = CPU-h（M1 cap 下界–上界），一个 {D['budget']['ncpu']} CPU 作业的 LPT 墙钟 "
            f"{fh(W['wall_hours']['low'], 3)}–{fh(W['wall_hours']['high'], 3)} h。实测：单元墙钟合计 {fh(grid_ph)} 进程小时（投影的 {grid_ph / L['process_hours']['high']:.2f}–"
            f"{grid_ph / L['process_hours']['low']:.2f} 倍）；两个作业墙钟合计 {fh(grid_wall / 3600, 3)} h，分配 {fh(grid_alloc)} CPU-h。第一阶段报告 §5.1 的粗估（头部代理，"
            f"未实测）按单元数换算到本网格为 {sc['10']:.1f} / {sc['30']:.1f} / {sc['100']:.1f} CPU-h（M1 cap 10 / 30 / 100）；实测分配 CPU-h 为其 cap 10 情形的 "
            f"{grid_alloc / sc['10']:.3f} 倍。{Q('9.4', '若实际资源超过原数量级，先记录新预算及可完成范围')}；实测未超过，规则未触发。", ""]
    pm = L["by_method_process_hours"]
    rows = []
    for m in D["methods"]:
        cm = D["cost_methods"][m]
        rows.append((m, cm["n"], fh(cm["wall"] / 3600, 3), f"{fh(pm['low'][m], 3)}–{fh(pm['high'][m], 3)}", f"{cm['wall'] / cm['n']:.1f}"))
    out += md_table(["方法", "网格单元", "实测进程小时", "投影进程小时", "实测每单元墙钟 s"], rows, "lrrrr")
    hi = max(D["methods"], key=lambda m: D["cost_methods"][m]["wall"] / 3600 - pm["high"][m])
    out += ["", f"实测超出投影最多的是 {hi}（{fh(D['cost_methods'][hi]['wall'] / 3600)} vs {fh(pm['high'][hi])} 进程小时）。投影记录的假设：每单元成本取 "
            f"{L['allocated_cpus']} CPU 分配内实测的同方法 smoke 单元，更大进程数下的内存带宽争用未测量；m = 8 与 K8/K32 单元按 m = 32、K16 计；排队时间不计。", ""]
    return out


def sec_cal(D, Q) -> list[str]:
    S, conds, methods = D["selection"], D["conds"], D["methods"]
    out = ["## 3 CAL 选模", "", f"调参套为独立数据 root seed {D['tuning_seed']}、算法 seed {D['tuning_seed']}。{Q('9.2', '只最大化独立 CAL 的完整经验 U-J；并列取更早 checkpoint，再取上述冻结候选顺序')}。"
           f"生成器从 {D['n_by_block']['CAL']} 个 CAL 单元 result.json 的检查点 CAL U-J（不含真值）重算全部 {len(S)} 组选择，与 `{SELECTION_REL}`（sha256 `{D['selection_sha256'][:16]}`…）逐组一致；"
           f"CAL 单元真值隔离拒绝 {D['jobs']['CAL']['denials']} 次。正式的 {D['n_formal']} 个单元所用 lr / cap 均等于本组选择（容量单元继承同条件同方法的 K16 选择），"
           "每个正式单元的 checkpoint 只按自身 CAL U-J 选取（生成器逐单元重算一致）。", "", "### 3.1 每组选中的候选", "",
           "格式：lr / 选中步数；MF_VCS_M1 为 lr / cap / 步数。", ""]
    rows = []
    for c in conds:
        r = [cond_label(c)]
        for m in methods:
            g = S[(c, m)]
            r.append(f"{flr(g['lr'])} / cap {g['cap']} / {g['step']}" if m == "MF_VCS_M1" else f"{flr(g['lr'])} / {g['step']}")
        rows.append(r)
    out += md_table(["条件", *methods], rows, "l" + "l" * len(methods))
    nt = sum(g["n_tied"] > 1 for g in S.values())
    out += ["", f"并列：{nt}/{len(S)} 组出现多个（候选，checkpoint）CAL U-J 完全相等。", "", "### 3.2 各学习率、cap 与步数的胜出次数", ""]
    rows = []
    for m in methods:
        gs = [S[(c, m)] for c in conds]
        rows.append((m, *(sum(g["lr"] == lr for g in gs) for lr in D["lrs"]),
                     "、".join(f"cap {cap}: {sum(g['cap'] == cap for g in gs)}" for cap in D["caps"]) if m == "MF_VCS_M1" else "—"))
    out += md_table(["方法", *(f"lr {flr(x)}" for x in D["lrs"]), "M1 cap"], rows, "lrrrl")
    out += [""]
    rows = []
    for m in methods:
        rows.append((m, *(D["cal_steps"][(m, s)] for s in D["steps"]), *(D["own_steps_main"][(m, s)] for s in D["steps"])))
    hdr = ["方法", *(f"CAL {s}" for s in D["steps"]), *(f"主 {s}" for s in D["steps"])]
    out += [f"选中 checkpoint 步数：“CAL s”= 调参套各组（每方法 {len(conds)} 组）选中步数 s 的组数；“主 s”= 主实验正式单元（每方法 "
            f"{len(conds) * len(D['data_seeds'])} 个）自身 CAL 选中步数 s 的单元数。", ""]
    out += md_table(hdr, rows, "l" + "r" * (len(hdr) - 1))
    out += [""]
    return out


def sec_main(D, Q) -> list[str]:
    conds, methods, main = D["conds"], D["methods"], D["main"]
    R = len(D["data_seeds"])
    out = ["## 4 主表（决定 §9.3）", "", Q("9.3", "以 data replicate 为主配对单位计算 bias、variance、RMSE，并报告有限重复数的不确定性；条件算法波动独立成表") + "。"
           f"每格 R = {R} 个数据重复（data root {D['data_seeds'][0]}–{D['data_seeds'][-1]}，算法 seed {D['algo_main']}）；同一数据 seed 与条件的全部方法（含算法与容量单元）"
           f"共用同一套观测（FIT/CAL/EVAL 样本 sha256 逐一核对，{D['obs_sets']} 套）。", "",
           "### 4.1 EVAL U-J − S", ""]
    rows = []
    for c in conds:
        for m in methods:
            e = main[(c, m)]["EVAL_UJ_minus_S"]
            rows.append((cond_label(c), m, fS(main[(c, m)]["S"]), fe(e["mean"]), fci(e["mean_ci95_t"]), yn(has0(e["mean_ci95_t"])), fe(e["sd"]), fsci(e["var_ci95_chi2"]),
                         fe(e["rmse"]), fci(e["rmse_ci95_bootstrap"]), f"{main[(c, m)]['n_neg']}/{R}"))
    out += md_table(["条件", "方法", "S", "偏差", "95% t", "含 0", "sd", "95% χ²", "RMSE", "95% bootstrap", "EVAL U-J < 0"], rows, "llrrlcrlrlr")
    out += ["", f"S 在同一条件的 {R} 个数据重复间相同（生成器核对）。“EVAL U-J < 0”为该格中 EVAL U-J 为负的重复数（负值均保留在偏差、sd 与 RMSE 中）。", "",
            "### 4.2 方法配对差（每个数据重复配对）", "",
            f"Δ平方误差 = 同一数据重复上 e_a² − e_b² 的均值；ΔE_M(T − η)² = 两法总体逼近误差之差的均值；区间均为 95% t（R = {R} 对）。负值表示 a 的值较小。", ""]
    rows = []
    for c in conds:
        for a, b in D["pairs"]:
            p = D["paired"][(c, a, b)]
            rows.append((cond_label(c), f"{a} − {b}", fe(p["sq"]["mean"]), fci(p["sq"]["mean_ci95_t"]), yn(has0(p["sq"]["mean_ci95_t"])),
                         fe(p["mse"]["mean"]), fci(p["mse"]["mean_ci95_t"]), yn(has0(p["mse"]["mean_ci95_t"]))))
    out += md_table(["条件", "配对 a − b", "Δ平方误差", "95% t", "含 0", "ΔE_M(T − η)²", "95% t", "含 0"], rows, "llrlcrlc")
    out += [""]
    return out


def sec_decomp(D, Q) -> list[str]:
    conds, methods, main = D["conds"], D["methods"], D["main"]
    idt = D["identity"]
    out = ["## 5 误差分解（决定 §9.5）", "", "### 5.1 恒等式核对", "",
           f"evaluator 对每个评估表（所选 checkpoint、全部 checkpoint 轨迹与饱和表 U/V）枚举 m² 个单元，核对 E_M(T − η)² = S − J_pop(T)："
           f"{idt['units']} 个单元共 {idt['n']} 个表，最大 |残差| {fe(idt['max_abs'], 2)}，容差 {fe(D['identity_atol'], 1)}（包检查器的容差），全部 PASS。"
           "由此每个重复的误差分解为 e = EVAL 抽样误差 − 总体逼近误差。", "", f"### 5.2 分解表（主实验，每格 {len(D['data_seeds'])} 个数据重复）", "",
           "列：总体逼近误差均值与 95% t 区间；EVAL 抽样误差均值、95% t 区间与 sd；排列条件 sd = √(各重复排列条件方差的均值)，分 FIT 与 EVAL；"
           "头 gap（仅 MF）：FW gap 均值与 95% t 区间、FW gap / G(0) 的中位数、fp64 polish 给出的次优值下界均值。非负量的 t 区间可延伸到 0 以下，按原值列出。", ""]
    rows = []
    for c in conds:
        for m in methods:
            x = main[(c, m)]
            g = x["gap"]
            rows.append((cond_label(c), m, fe(x["pop_mse"]["mean"]), fci(x["pop_mse"]["mean_ci95_t"]), fe(x["EVAL_UJ_minus_J_pop"]["mean"]),
                         fci(x["EVAL_UJ_minus_J_pop"]["mean_ci95_t"]), fe(x["EVAL_UJ_minus_J_pop"]["sd"]), fe(math.sqrt(x["perm1_var_fit"]["mean"])),
                         fe(math.sqrt(x["perm_eval_var_mean"])),
                         fe(g["mean"]) if g else "—", fci(g["mean_ci95_t"]) if g else "—", fe(x["gap_rel_median"], 2) if g else "—",
                         fe(x["polish_lower_mean"]) if g else "—"))
    out += md_table(["条件", "方法", "E_M(T − η)²", "95% t", "EVAL 抽样误差", "95% t", "抽样 sd", "排列 sd FIT", "排列 sd EVAL", "FW gap", "95% t",
                     "gap / G(0) 中位数", "polish 下界"], rows, "llrlrlrrrrlrr")
    pol = D["polish"]
    out += ["", f"头 gap 在所选 checkpoint 的特征上、外层更新之后计算；FW gap 是头部次优值的上界，polish 下界是从该 W 出发 fp64 迭代 {D['polish_iters'][0]} 步所得的改进量，"
            f"正式 MF 单元的 polish 状态：{'、'.join(f'{k} {v}' for k, v in sorted(pol.items()))}，故真实次优值位于 [polish 下界, FW gap] 之间。", "",
            "### 5.3 M1 内层求解（正式单元）", ""]
    rows = []
    for cap in sorted(D["m1"]):
        e = D["m1"][cap]
        n = sum(e["stop"].values())
        rows.append((cap, e["units"], n, fpct(e["stop"]["gap_tolerance"] / n, 1), fpct(e["stop"]["iteration_budget_exhausted"] / n, 1), e["stop"]["zero_reference_gap"],
                     f"{min(e['conv']):.3f}", f"{float(np.median(e['it_med'])):g}", e["fp64_trig"], e["fp64_dis"]))
    tol = {e["tol"] for e in D["m1"].values()}
    out += md_table(["选中 cap", "单元", "内层求解（步）", "达到相对 gap", "达到 cap", "零参考 gap", "单元内达到相对 gap 的比例（最小）", "每步迭代数中位数（单元中位数的中位数）",
                     "fp64 诊断触发步", "其中停止判断不一致"], rows, "rrrrrrrrrr")
    out += ["", f"每个训练步做一次内层求解；相对 gap 阈值为当前批 W = 0 参考 gap 的 {fpct(tol.pop())}（决定 §2.1），达到 cap 的求解按近似解记录并保留实际 gap。"
            f"fp64 诊断：{Q('2.3', '对接近 fp32 分辨能力的参考值，保留一个同输入 fp64 数值诊断')}；“停止判断不一致”为该步 fp32 与同输入 fp64 对是否达到相对 gap 的判断不同的步数。"
            "诊断不改变训练。", "",
            "### 5.4 排列条件方差与 U/V 诊断", "",
            f"排列条件方差：每个正式单元的 FIT 与 EVAL 各用 {D['perm_z']['R']} 个均匀无固定点排列计算 perm1 估计，其均值与完整 U-J 之差的 z 值共 {D['perm_z']['n']} 个："
            f"均值 {D['perm_z']['mean']:.3f}、sd {D['perm_z']['sd']:.3f}、最大 |z| {D['perm_z']['max_abs']:.2f}，|z| > 3 的 {D['perm_z']['n_gt3']} 个（标准正态期望 "
            f"{D['perm_z']['expected_gt3']:.1f}）。perm1 的条件 sd 见 5.2。", "",
            f"U/V（{Q('9.1', '不另开第二套 V 训练网格')}）：同一冻结 critic、同一 EVAL 样本上 EVAL U-J − EVAL V-J 的绝对值；"
            "“/ sd”为格内最大 |U − V| 与该格 e 的 sd 之比的最大值。", ""]
    rows = [(cond_label(c), fe(D["uv"][c]["mean"]), fe(D["uv"][c]["max"]), fe(D["uv"][c]["max_over_sd"], 2)) for c in conds]
    out += md_table(["条件", f"|U − V| 均值（{len(methods) * len(D['data_seeds'])} 个主单元）", "最大", "最大 / sd"], rows, "lrrr")
    out += ["", "### 5.5 FIT 计数饱和表", "", Q("9.5", "它作为 FIT 全表函数类的经验最优参考，不使用真实 p/q 训练，也不冒充已训练的 MF 神经特征") + "。"
            f"同一（数据 root，条件）的饱和表只依赖 FIT 计数，{D['sat_units']} 个单元记录的副本归并为 {D['sat_keys']} 张、冲突 0（单张最多 {D['sat_shared_max']} 份副本）；"
            f"其 FIT U-J 不低于本单元每个 checkpoint 的检查 {D['sat_dominates']}/{len(D['units'])} PASS。下表为 {len(D['data_seeds'])} 个正式数据重复（U 参考）；末列为 §4.1 同条件 {len(methods)} 个训练方法的 RMSE 范围，仅作对照。", ""]
    rows = []
    for c in conds:
        s = D["sat"][c]
        rr = [main[(c, m)]["EVAL_UJ_minus_S"]["rmse"] for m in methods]
        rows.append((cond_label(c), fS(s["S"]), fe(s["fit_minus_S"]["mean"]), fe(s["eval_err"]["mean"]), fci(s["eval_err"]["mean_ci95_t"]), fe(s["eval_err"]["rmse"]),
                     fe(s["pop"]["mean"]), fe(s["eval_err_V"]["mean"]), f"{fe(min(rr))}–{fe(max(rr))}"))
    out += md_table(["条件", "S", "FIT U-J − S 均值", "EVAL U-J − S 均值", "95% t", "RMSE", "E_M(T − η)²", "V 参考 EVAL − S 均值", "训练方法 RMSE（§4.1）"], rows, "lrrrlrrrl")
    out += ["", Q("9.5", "不要以这个小支持集实验推断 MF 对任何 MLP 都有普遍计算优势") + "。", ""]
    return out


def sec_algo(D, Q) -> list[str]:
    conds, methods = D["conds"], D["methods"]
    out = ["## 6 条件算法波动（固定数据 root " + str(D["algo_data"]) + "）", "",
           Q("9.3", "不与 10 个 data 重复合并为 12 个同分布重复") + f"。每格为同一套数据上算法 seed {D['algo_main']}（即主实验该数据 root 的单元）、"
           f"{'、'.join(map(str, D['algo_extra']))} 的 R = 3 个值；超参数同主实验，checkpoint 各按自身 CAL 选。末列为 §4.1 同格的数据重复 sd，仅作对照、不合并。", ""]
    rows = []
    for c in conds:
        for m in methods:
            a = D["algo"][(c, m)]
            e = a["EVAL_UJ_minus_S"]
            rows.append((cond_label(c), m, " / ".join(fe(v) for v in a["err"]), fe(e["mean"]), fe(e["sd"]), fsci(e["var_ci95_chi2"]),
                         fe(D["main"][(c, m)]["EVAL_UJ_minus_S"]["sd"])))
    out += md_table(["条件", "方法", f"e（algo {' / '.join(map(str, [D['algo_main']] + D['algo_extra']))}）", "均值", "算法 sd", "95% χ²", "数据重复 sd（§4.1）"], rows, "lllrrlr")
    ra = {a["EVAL_UJ_minus_S"]["R"] for a in D["algo"].values()}
    check(len(ra) == 1, "algo cells with different R")
    any_cell = next(iter(D["algo"].values()))["EVAL_UJ_minus_S"]["var_ci95_chi2"]
    out += ["", f"R = {ra.pop()} 时 sd 的 95% χ² 区间上限约为下限的 {math.sqrt(any_cell[1] / any_cell[0]):.0f} 倍，算法 sd 只能作粗略比较。", ""]
    return out


def sec_capacity(D, Q) -> list[str]:
    cap = D["capacity"]
    out = ["## 7 容量诊断：固定训练规则下的容量响应（决定 §9.4）", "",
           Q("9.4", "包含优化因素，不称为每个 K 独立充分优化后的最优容量曲线") + f"。条件为 {'、'.join(cond_label(c) for c in D['cap_conds'])}；三个 MF 方法的 K "
           f"{'、'.join(map(str, D['cap_K']))} 继承各自 K{D['K_main']} 的 lr / cap，同为 {len(D['data_seeds'])} 个数据重复、算法 seed {D['algo_main']}，checkpoint 按自身 CAL U-J 选取；"
           f"K{D['K_main']} 行即主实验单元。末两列为同一数据重复上与 K{D['K_main']} 的平方误差差（e_K² − e_K16² 的均值，95% t）。", ""]
    rows = []
    for c in D["cap_conds"]:
        for m in D["mf_methods"]:
            for K in sorted(D["cap_K"] + [D["K_main"]]):
                x = cap[(c, m, K)]
                e = x["EVAL_UJ_minus_S"]
                d = D["cap_paired"].get((c, m, K))
                rows.append((cond_label(c), m, K, fe(e["mean"]), fci(e["mean_ci95_t"]), fe(e["rmse"]), fci(e["rmse_ci95_bootstrap"]), fe(x["pop_mse"]["mean"]),
                             fe(x["gap"]["mean"]), fe(d["mean"]) if d else "—", fci(d["mean_ci95_t"]) if d else "—"))
    out += md_table(["条件", "方法", "K", "偏差", "95% t", "RMSE", "95% bootstrap", "E_M(T − η)²", "FW gap", "Δ平方误差 vs K16", "95% t"], rows, "llrrlrlrrrl")
    out += [""]
    return out


def sec_limits(D, Q) -> list[str]:
    cap_m = sorted({c[1] for c in D["cap_conds"]})
    dis = sum(e["fp64_dis"] for e in D["m1"].values())
    trig = sum(e["fp64_trig"] for e in D["m1"].values())
    return ["## 8 未完成与限制", "",
            f"- RuLSIF / 核精确矩尚未实现，本网格不含其二次风险对照：{Q('9.5', 'RuLSIF 精确矩实现并核对后补入二次风险比较')}；{Q('5', 'RuLSIF 精确矩也应作为后续 T03 的二次风险对照')}。",
            f"- 不对 InfoNCE、DV、NWJ 等非 S 输出计算相对 S 的 RMSE（{Q('9.5', '非 S 输出的 InfoNCE、DV、NWJ 等不按 S 真值计算 RMSE')}）；本网格也未训练这些方法。",
            f"- 不据这一小支持集实验对 MF 相对 MLP 的计算优势作一般结论（{Q('9.5', '离散场景中 MLP 同样可以按单元频数聚合')}）；§2.4 的成本只是本网格、本 CPU 布局下的实测。",
            f"- 不作方法优劣、强基线优势或新颖性结论（决定 §5）；§4.2 的配对差只是本网格在固定训练规则下的测量。",
            f"- V 只是同一冻结 critic 上的 U/V 评估诊断（{Q('9.1', '这个有限样本公式不证明经训练及选模后的最大 J 是 S 的无偏估计')}）；主读数为 U。",
            f"- 容量诊断只覆盖 {len(D['cap_conds'])} 个 m = {'、'.join(map(str, cap_m))} 条件、{len(D['mf_methods'])} 个 MF 方法，且继承 K{D['K_main']} 的超参数；"
            "它是固定训练规则下的容量响应（决定 §9.4），不是最优容量曲线。",
            f"- 区间以 {len(D['data_seeds'])} 个数据重复为单位逐格计算，未做多重比较校正；§6 的算法重复只有 {1 + len(D['algo_extra'])} 个 seed、只在一套数据上。",
            "- diagonal_mixture ρ > 0 的 MF 选择落在学习率与步数网格边界（§1、§3）；更大学习率或更长训练下的结果未测量，网格按决定 §9.2 冻结，未作事后扩展。",
            f"- M1 的同输入 fp64 诊断在 {trig} 个触发步中有 {dis} 步（{fpct(dis / trig, 1)}）停止判断与 fp32 不一致（§5.3）；按决定 §2.3 这只是诊断，训练仍用 fp32，"
            "其对所选 critic 的影响未单独测量。",
            "- 本报告只含合成真值实验；自然配准（BF、EPFL）的结果不在其中。",
            ""]


def sec_files(D) -> list[str]:
    rows = []
    for b, spec in BLOCKS.items():
        M = D["manifests"][b]
        rows.append((f"`{spec['dir']}`", BLOCKS[b]["label"], f"{M['files']} 个文件 + `{spec['manifest']}`", f"{M['bytes']}", f"`{D['git'][b]['commit'][:7]}`"))
    out = ["## 9 记录文件", ""] + md_table(["目录", "块", "已提交文件", "字节", "结果提交"], rows, "lllrl")
    out += ["", f"主要汇总记录：`{AGGREGATE_REL}`、`{SELECTION_REL}`、`{BUDGET_REL}`；单元表 `{UNITS_REL}`；协议 `{PROTOCOL_REL}`（t03 节）；决定 `{DECISION_REL}`。"
            "checkpoints.pt 与 train_steps.json.gz 留在 /projects/EEG-foundation-model/yinghao/outputs_v10/T03，其 sha256 见各块 manifest。", ""]
    return out


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V10 T03 results report generator (writes only the report)")
    ap.add_argument("--output", type=Path, default=REPORT_DEFAULT)
    ap.add_argument("--check", action="store_true", help="render in memory and compare with the report committed at HEAD")
    a = ap.parse_args(argv)
    D = collect()
    text = render(D)
    data = text.encode("utf-8")
    if a.check:
        committed = git_bytes("show", f"HEAD:{REPORT_REL}")
        if committed != data:
            diff = list(difflib.unified_diff(committed.decode("utf-8").splitlines(), text.splitlines(), "HEAD", "generated", lineterm="", n=0))
            print("\n".join(diff[:60]))
            print(json.dumps({"check": "DIFFERENT", "committed_sha256": sha256_bytes(committed)[:16], "generated_sha256": sha256_bytes(data)[:16]}))
            return 1
        print(json.dumps({"check": "IDENTICAL", "sha256": sha256_bytes(data)[:16], "bytes": len(data)}))
        return 0
    a.output.write_bytes(data)
    print(json.dumps({"report": str(a.output), "lines": text.count("\n"), "bytes": len(data), "sha256": sha256_bytes(data)[:16]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
