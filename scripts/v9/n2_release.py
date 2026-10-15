#!/usr/bin/env python
"""V9-N2 release orchestrator — login node, long-running (coordinator's execution order 2026-10-15; no protocol change).

Order of submission (normal-QOS count `squeue -u $USER -h -o %q | grep -c normal` < CAP before every sbatch, so the count stays <= CAP;
nice 400 comes from slurm/v9/n2_unit.sbatch):
  0. gate    the first run of formal group g01 (--gate-job) has written its first checkpoint with a finite loss and dev rSum >= --gate-rsum
             (progress.jsonl, else last.pt), with no Traceback in the job log.  A failed gate stops everything (abort event; a human decides).
  1. tuning  seed 917 groups, planned by scripts/v9/n2_plan_units.py from seed 917's OWN precheck merge groups once the shared-stage job
             (--tuning-precheck-job) ended with exit=0; submitted in planner order (x1 pairs first, then {0.5, 2} per (method, rho)).
  2. formal  the remaining formal runs (formal_units.json minus g01), one job = runs of ONE (rho, merged-member set) key, <= --per-job runs.
             At each free slot: the first planned run whose seed-917 tuning is COMPLETE (all three multipliers done) for every member is
             submitted at the selected multiplier (frozen protocol: "chosen on seed 917 and frozen before the formal seeds"); if no planned run
             is complete, the first remaining key is submitted at x1 and the frozen contingency applies.  Selection = highest dev rSum of
             seed 917; ties -> 1, else the smaller multiplier.  A merged run whose members select different multipliers is split by multiplier
             (one run only when coefficients AND lr are identical).
  3. rerun   once every (method, rho) selection is complete: x1 formal runs whose members select a multiplier other than 1 are re-run at
             it (contingency); the x1 run stays attributed to the members that selected 1 and is reported as a sensitivity row otherwise.
  wave 2  NCR_A / L2RM_A (patched official code, slurm/v9/n2_official.sbatch, ONE run per job: NCR has no co-training resume) join the same
             queues once reports/V9/V9_N2_units/wave2_enabled.json exists (written after the official smoke passed and fork A's timings landed):
             their seed-917 tuning runs go to the tuning queue (ahead of any formal run not yet submitted), their formal runs to the formal list
             under the same selected / x1 rule.  The contingency is computed per run as soon as its members' selections are complete.
A job that left the queue with runs lacking done.json (official formal: result.json too) is resubmitted once (units resume from last.pt and skip when done); a second failure is
logged as given_up for a human.  Ledger (append-only, no results): reports/V9/V9_N2_units/launch_ledger.jsonl; state + status:
slurm_logs/v9_n2_release_{state,status}.json; job ids for slurm/v9/n2_watch.sh: slurm_logs/v9_n2_feed_ids.txt.

usage (login node):
  nohup setsid /home/infres/yinwang/CS_QMI/env/bin/python scripts/v9/n2_release.py run --gate-job 1042201 \\
      --gate-group reports/V9/V9_N2_units/formal_g01.txt --tuning-precheck-job 1042200 >/dev/null 2>&1 &
  python scripts/v9/n2_release.py status
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MULTS = (0.5, 1.0, 2.0)
ARMS = ("Unweighted_A", "VCS_Weight", "BCE_Weight", "RuLSIF_Weight", "Nystrom_Weight", "RankMass")
OFFICIAL = {"NCR_A": "ncr", "L2RM_A": "l2rm"}
ORDER = ARMS + tuple(OFFICIAL)
SGR_EXPORT = ("ALL,V9_SGR_PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python,"
              "DATA_ROOT=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data,"
              "VOCAB=/projects/EEG-foundation-model/yinghao/code_v9/SGRAF_py3/vocab,"
              "OUT_ROOT=/projects/EEG-foundation-model/yinghao/outputs_v9/n2,HOST_LOCK=reports/V9/HOST_LOCK.yaml")
DEF_OUT = "/projects/EEG-foundation-model/yinghao/outputs_v9/n2"
DEF_LOGS = "/home/infres/yinwang/CS_QMI/slurm_logs"
DEF_UNITS = str(REPO / "reports/V9/V9_N2_units")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- pure decision logic (tests/v9/test_n2_release.py)
def rkey(r: dict) -> tuple:
    return (float(r["rho"]), tuple(r["members"]))


def unit_dir(out: Path, r: dict) -> Path:
    return Path(out) / r["dataset"] / f"rho{float(r['rho']):g}" / f"s{int(r['seed'])}" / f"unit_{r['method']}_lr{float(r['lr_mult']):g}"


def family(r: dict) -> str:
    return "official" if r["method"] in OFFICIAL else "sgr"


def run_complete(out: Path, r: dict, kind: str) -> bool:
    """sgr: done.json; official tuning: done.json (dev best); official formal / rerun: done.json and result.json (official test collect)."""
    if run_done(out, r) is None:
        return False
    return not (family(r) == "official" and kind in ("formal", "rerun")) or (unit_dir(out, r) / "result.json").exists()


def run_done(out: Path, r: dict) -> dict | None:
    p = unit_dir(out, r) / "done.json"
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except (OSError, ValueError):
        return None


def selection(out: Path, tuning_runs: list) -> dict:
    """{(method, rho): {"rsum": {mult: dev rSum | None}, "complete": bool, "selected": mult | None}} for every member of every tuning run."""
    table: dict = {}
    for r in tuning_runs:
        d = run_done(out, r)
        rs = None if d is None else float(d["best"]["rsum"])
        for m in r["members"]:
            table.setdefault((m, float(r["rho"])), {mu: None for mu in MULTS})[float(r["lr_mult"])] = rs
    res = {}
    for k, rs in table.items():
        complete = all(v is not None for v in rs.values())
        sel = None
        if complete:
            best = max(rs.values())
            cands = [mu for mu in MULTS if rs[mu] == best]
            sel = 1.0 if 1.0 in cands else min(cands)
        res[k] = {"rsum": rs, "complete": complete, "selected": sel}
    return res


def sel_json(sel: dict, keys=None) -> dict:
    return {f"{m}|rho{rho:g}": {"dev_rsum_by_mult": {f"{mu:g}": v for mu, v in s["rsum"].items()}, "complete": s["complete"], "selected": s["selected"]}
            for (m, rho), s in sorted(sel.items()) if keys is None or (m, rho) in keys}


def complete_for(r: dict, sel: dict) -> bool:
    return all(sel.get((m, float(r["rho"])), {}).get("complete", False) for m in r["members"])


def split_by_selection(r: dict, sel: dict) -> list:
    by: dict = {}
    for m in r["members"]:
        by.setdefault(sel[(m, float(r["rho"]))]["selected"], []).append(m)
    return [dict(r, method=sorted(ms)[0], lr_mult=float(mu), members=sorted(ms), fixed=True) for mu, ms in sorted(by.items())]


def next_formal_job(pending: list, sel: dict, per_job):
    """-> (job_runs, case, consumed_indices, leftover_fixed_runs); one job = runs of a single (rho, members) key; per_job: int or run -> int."""
    pj = per_job if callable(per_job) else (lambda _r: per_job)
    fixed = [j for j, q in enumerate(pending) if q.get("fixed")]
    if fixed:
        k0 = rkey(pending[fixed[0]])
        same = [j for j in fixed if rkey(pending[j]) == k0][:pj(pending[fixed[0]])]
        return [pending[j] for j in same], "selected", same, []
    for i, r in enumerate(pending):
        if complete_for(r, sel):
            n = pj(r)
            parts = split_by_selection(r, sel)
            if len(parts) >= n:
                return parts[:n], "selected", [i], parts[n:]
            job, used = list(parts), [i]
            for j, q in enumerate(pending):
                if j != i and rkey(q) == rkey(r) and complete_for(q, sel):
                    qp = split_by_selection(q, sel)
                    if len(job) + len(qp) <= n:
                        job += qp
                        used.append(j)
                    if len(job) >= n:
                        break
            return job, "selected", used, []
    k0 = rkey(pending[0])
    same = [j for j, q in enumerate(pending) if rkey(q) == k0][:pj(pending[0])]
    return [dict(pending[j], lr_mult=1.0) for j in same], "x1_contingency", same, []


def contingency_runs(x1_runs: list, sel: dict) -> list:
    out = []
    for r in x1_runs:
        out += [p for p in split_by_selection(r, sel) if p["lr_mult"] != 1.0]
    return out


def chunk_by_key(runs: list, per_job) -> list:
    pj = per_job if callable(per_job) else (lambda _r: per_job)
    groups: dict = {}
    for r in runs:
        groups.setdefault((rkey(r), float(r["lr_mult"])), []).append(r)
    jobs = []
    for _, rs in groups.items():
        rs = sorted(rs, key=lambda q: int(q["seed"]))
        n = pj(rs[0])
        jobs += [rs[k:k + n] for k in range(0, len(rs), n)]
    return jobs


def order_key(r: dict) -> tuple:
    return (not r.get("fixed"), -float(r["rho"]), ORDER.index(r["method"]) if r["method"] in ORDER else 99, tuple(r["members"]), int(r["seed"]))


def group_line(r: dict) -> str:
    return f"{r['dataset']} {float(r['rho']):g} {r['method']} {int(r['seed'])} {float(r['lr_mult']):g} {','.join(r['members'])}"


def parse_group_file(p: Path, members_lookup=None) -> list:
    runs = []
    for ln in Path(p).read_text().splitlines():
        f = ln.split()
        if not f:
            continue
        r = {"dataset": f[0], "rho": float(f[1]), "method": f[2], "seed": int(f[3]), "lr_mult": float(f[4]) if len(f) > 4 else 1.0}
        if len(f) > 5:
            r["members"] = sorted(f[5].split(","))
        elif members_lookup is not None:
            r["members"] = members_lookup(r)
        else:
            r["members"] = [r["method"]]
        runs.append(r)
    return runs


def plain(r: dict) -> dict:
    return {k: r[k] for k in ("dataset", "rho", "seed", "method", "lr_mult", "members")}


# ---------------------------------------------------------------- cluster access (replaced by a fake in the tests)
class Cluster:
    def __init__(self, user: str, logs: Path):
        self.user, self.logs = user, Path(logs)

    def my_jobs(self):
        r = subprocess.run(["squeue", "-u", self.user, "-h", "-o", "%i"], capture_output=True, text=True)
        return None if r.returncode != 0 else {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}

    def normal_count(self):
        r = subprocess.run(["squeue", "-u", self.user, "-h", "-o", "%q"], capture_output=True, text=True)
        return None if r.returncode != 0 else sum(1 for ln in r.stdout.splitlines() if ln.strip() == "normal")

    def submit(self, name: str, group_file: Path, formal: bool):
        cmd = ["sbatch", "--parsable", f"--job-name={name}", f"--export={SGR_EXPORT}{',FORMAL=1' if formal else ''}",
               str(REPO / "slurm/v9/n2_unit.sbatch"), "group", str(group_file)]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        jid = r.stdout.strip().split(";")[0]
        return (jid if r.returncode == 0 and jid.isdigit() else None), (r.stderr or r.stdout)[-400:]

    def submit_official(self, name: str, r: dict, kind: str, sbatch_args: list):
        cmd = ["sbatch", "--parsable", f"--job-name={name}", *sbatch_args, str(REPO / "slurm/v9/n2_official.sbatch"), "unit", OFFICIAL[r["method"]], r["dataset"],
               f"{float(r['rho']):g}", str(int(r["seed"])), f"{float(r['lr_mult']):g}", "tuning" if kind == "tuning" else "formal"]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        jid = res.stdout.strip().split(";")[0]
        return (jid if res.returncode == 0 and jid.isdigit() else None), (res.stderr or res.stdout)[-400:]

    def log_of(self, job: str):
        c = sorted(self.logs.glob(f"*_{job}.out"))
        return c[0] if c else None


# ---------------------------------------------------------------- orchestrator
class Orchestrator:
    def __init__(self, cluster, out: Path, units: Path, logs: Path, gate_job: str, gate_group: Path, tuning_precheck_job: str, per_job: int = 2,
                 cap: int = 27, gate_rsum: float = 30.0, dataset: str = "f30k", tuning_seed: int = 917, python: str = sys.executable,
                 precheck_record_dir: Path | None = None):
        self.cluster, self.out, self.units, self.logs = cluster, Path(out), Path(units), Path(logs)
        self.gate_job, self.gate_group, self.tp_job = str(gate_job), Path(gate_group), str(tuning_precheck_job)
        self.per_job, self.cap, self.gate_rsum, self.dataset, self.tuning_seed, self.python = per_job, cap, gate_rsum, dataset, tuning_seed, python
        self.precheck_record_dir = precheck_record_dir
        self.state_path = self.logs / "v9_n2_release_state.json"
        self.status_path = self.logs / "v9_n2_release_status.json"
        self.ledger = self.units / "launch_ledger.jsonl"
        self.ids = self.logs / "v9_n2_feed_ids.txt"
        self.release_dir = self.units / "release"
        self.release_dir.mkdir(parents=True, exist_ok=True)
        self._last_err = ("", 0.0)
        self.st = self._load()

    # -- persistence
    def _load(self) -> dict:
        if self.state_path.exists():
            st = json.loads(self.state_path.read_text())
            st["tuning_queue"] = [q if isinstance(q, dict) else {"family": "sgr", "group_file": q} for q in st["tuning_queue"]]   # v1 state: group-file strings
            st.setdefault("x1_processed", []); st.setdefault("wave2", None); st.setdefault("resubmit_queue", [])
            for j in st["jobs"]:
                j.setdefault("family", family(j["runs"][0]))
            return st
        plan = json.loads((self.units / "formal_units.json").read_text())
        lookup = {(float(r["rho"]), int(r["seed"]), r["method"]): sorted(r["members"]) for r in plan["runs"]}
        gate_runs = parse_group_file(self.gate_group, lambda r: lookup[(r["rho"], r["seed"], r["method"])])
        gate_keys = {(r["rho"], r["seed"], r["method"]) for r in gate_runs}
        pending = [dict(plain(r), lr_mult=1.0) for r in plan["runs"] if (float(r["rho"]), int(r["seed"]), r["method"]) not in gate_keys]
        pending.sort(key=lambda r: (-float(r["rho"]), ARMS.index(r["method"]), tuple(r["members"]), int(r["seed"])))
        st = {"created": utc(), "gate": "wait", "tuning_planned": False, "tuning_runs": [], "tuning_queue": [], "formal_pending": pending,
              "contingency_computed": False, "rerun_queue": [], "jobs": [], "counters": {"t": 0, "f": 1, "r": 0}, "done": False, "blocked": "",
              "x1_processed": [], "wave2": None, "resubmit_queue": []}
        st["jobs"].append({"job": self.gate_job, "name": "v9n2_f01", "kind": "formal", "case": "x1_contingency", "attempt": 1, "status": "queued", "family": "sgr",
                           "group_file": str(self.gate_group), "runs": [plain(r) for r in gate_runs], "submitted_utc": "",
                           "note": "gate group submitted before any seed-917 tuning existed (frozen contingency applies)"})
        return st

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.st, indent=1))
        tmp.replace(self.state_path)
        jobs = self.st["jobs"]
        status = {"utc": utc(), "gate": self.st["gate"], "tuning_planned": self.st["tuning_planned"], "tuning_queue": len(self.st["tuning_queue"]),
                  "formal_pending_runs": len(self.st["formal_pending"]), "rerun_queue": len(self.st["rerun_queue"]),
                  "x1_runs_processed": len(self.st["x1_processed"]), "wave2_loaded": self.st["wave2"] is not None, "blocked": self.st["blocked"], "done": self.st["done"],
                  "jobs": {s: sum(1 for j in jobs if j["status"] == s) for s in sorted({j["status"] for j in jobs})},
                  "cases": {c: sum(1 for j in jobs if j["case"] == c) for c in sorted({j["case"] for j in jobs})}}
        self.status_path.write_text(json.dumps(status, indent=1))

    def event(self, ev: str, **kw) -> None:
        rec = {"utc": utc(), "event": ev, **kw}
        with open(self.ledger, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def error(self, e: Exception) -> None:
        msg = repr(e)[:500]
        if msg != self._last_err[0] or time.time() - self._last_err[1] > 3600:
            self.event("error", error=msg)
            self._last_err = (msg, time.time())

    def commit(self) -> str:
        r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip()

    # -- gate and tuning plan
    def gate_status(self, mine: set):
        runs = parse_group_file(self.gate_group)
        d = unit_dir(self.out, runs[0])
        rec = None
        p = d / "progress.jsonl"
        try:
            if p.exists() and p.read_text().strip():
                rec = json.loads(p.read_text().splitlines()[0])
            elif (d / "last.pt").exists():
                import torch
                ck = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
                e = ck["log"]["epochs"][0]
                rec = {"epoch": e["epoch"], "phase": e.get("phase"), "mean_loss": e["mean_loss"], "dev_rsum": e["dev"]["rsum"], "seconds": e["seconds"]}
        except Exception as ex:                                  # partially visible file on NFS: try again next cycle
            self.error(ex)
            rec = None
        log = self.cluster.log_of(self.gate_job)
        txt = log.read_text(errors="replace") if log is not None and log.exists() else ""
        if "Traceback" in txt:
            return "fail", {"why": "Traceback in the gate job log", "log": str(log)}
        if rec is not None:
            ok = math.isfinite(float(rec["mean_loss"])) and float(rec["dev_rsum"]) >= self.gate_rsum
            return ("pass" if ok else "fail"), {"first_epoch": rec, "gate_rsum": self.gate_rsum, "unit": str(d)}
        if self.gate_job not in mine:
            return "fail", {"why": "gate job left the queue without a first checkpoint", "log": str(log)}
        return "wait", {}

    def plan_tuning(self, mine: set) -> str:
        if self.tp_job in mine:
            return "wait"
        log = self.cluster.log_of(self.tp_job)
        txt = log.read_text(errors="replace") if log is not None and log.exists() else ""
        if "exit=0" not in txt:
            return "fail"
        r = subprocess.run([self.python, str(REPO / "scripts/v9/n2_plan_units.py"), "--precheck-dir", str(self.out / self.dataset), "--dataset", self.dataset,
                            "--seeds", str(self.tuning_seed), "--kind", "tuning", "--dest", str(self.units), "--per-job", str(self.per_job)],
                           capture_output=True, text=True, cwd=REPO)
        if r.returncode != 0:
            self.event("tuning_plan_failed", stdout=r.stdout[-1500:], stderr=r.stderr[-1500:])
            return "fail"
        plan = json.loads((self.units / "tuning_units.json").read_text())
        self.st["tuning_runs"] = [plain(x) for x in plan["runs"]] + self.st["tuning_runs"]                 # wave-2 runs (if already loaded) stay
        self.st["tuning_queue"] = [{"family": "sgr", "group_file": str(g)} for g in sorted(self.units.glob("tuning_g*.txt"))] + self.st["tuning_queue"]
        self.st["tuning_planned"] = True
        copied = []
        if self.precheck_record_dir is not None:
            for rho in (0.4, 0.0):
                for suf in ("json", "md"):
                    src = self.out / self.dataset / f"rho{rho:g}" / f"s{self.tuning_seed}" / "precheck" / f"precheck_{self.dataset}_rho{rho:g}_s{self.tuning_seed}.{suf}"
                    if src.exists():
                        shutil.copy2(src, self.precheck_record_dir / src.name)
                        copied.append(src.name)
        self.event("tuning_planned", precheck_job=self.tp_job, n_runs=plan["n_runs"], n_arm_rows=plan["n_arm_rows_covered"], groups=[q.get("group_file") for q in self.st["tuning_queue"]],
                   planner_stdout=r.stdout.strip()[-1500:], precheck_tables_copied=copied)
        return "ok"

    # -- job bookkeeping
    def runs_done(self, runs: list, kind: str) -> bool:
        return all(run_complete(self.out, r, kind) for r in runs)

    def refresh(self, mine: set) -> None:
        for j in list(self.st["jobs"]):
            if j["status"] != "queued" or j["job"] in mine:
                continue
            if self.runs_done(j["runs"], j["kind"]):
                j["status"] = "ok"
                j["ended_utc"] = utc()
                self.event("ended_ok", job=j["job"], name=j["name"], kind=j["kind"])
                continue
            j["ended_utc"] = utc()
            missing = [plain(r) for r in j["runs"] if not run_complete(self.out, r, j["kind"])]
            if j["attempt"] >= 2:
                j["status"] = "given_up"
                self.event("given_up", job=j["job"], name=j["name"], missing=missing)
                continue
            j["status"] = "incomplete"
            self.st.setdefault("resubmit_queue", []).append({"of": j["job"], "name": j["name"], "kind": j["kind"], "case": j["case"],
                                                             "group_file": j["group_file"], "runs": j["runs"], "attempt": j["attempt"] + 1})
            self.event("incomplete", job=j["job"], name=j["name"], missing=missing)

    def per_job_of(self, r: dict) -> int:
        return 1 if family(r) == "official" else self.per_job

    def _submit(self, name: str, kind: str, case: str, gfile: Path, runs: list, attempt: int = 1, **extra) -> bool:
        n = self.cluster.normal_count()
        if n is None or n >= self.cap:
            return False
        if family(runs[0]) == "official":
            assert len(runs) == 1, "official baselines: one run per job"
            jid, msg = self.cluster.submit_official(name, runs[0], kind, (self.st.get("wave2") or {}).get("sbatch_args", []))
        else:
            jid, msg = self.cluster.submit(name, gfile, formal=kind in ("formal", "rerun"))
        if jid is None:
            self.event("submit_failed", name=name, group_file=str(gfile), message=msg)
            return False
        rec = {"job": jid, "name": name, "kind": kind, "case": case, "attempt": attempt, "status": "queued", "group_file": str(gfile),
               "runs": [plain(r) for r in runs], "family": family(runs[0]), "submitted_utc": utc(), "commit": self.commit(), "normal_count_before": n, **extra}
        self.st["jobs"].append(rec)
        with open(self.ids, "a") as fh:
            fh.write(f"{jid} {name}\n")
        self.event("submitted", **{k: v for k, v in rec.items() if k != "status"})
        self.save()
        return True

    def maybe_load_wave2(self) -> None:
        f = self.units / "wave2_enabled.json"
        if self.st["wave2"] is not None or not f.exists():
            return
        cfg = json.loads(f.read_text())
        meths, rhos = cfg.get("methods", list(OFFICIAL)), [float(x) for x in cfg.get("rhos", [0.4, 0.0])]
        tseed, fseeds = int(cfg.get("tuning_seed", self.tuning_seed)), [int(x) for x in cfg.get("formal_seeds", [901, 902, 903])]
        mk = lambda m, rho, seed, mu: {"dataset": self.dataset, "rho": rho, "seed": seed, "method": m, "lr_mult": mu, "members": [m]}
        tuning = [mk(m, rho, tseed, mu) for rho in sorted(rhos, reverse=True) for mu in (1.0, 0.5, 2.0) for m in meths]
        formal = [mk(m, rho, s, 1.0) for rho in rhos for m in meths for s in fseeds]
        self.st["tuning_runs"] += tuning
        self.st["tuning_queue"] += [{"family": "official", "run": r} for r in tuning]
        self.st["formal_pending"] = sorted(self.st["formal_pending"] + formal, key=order_key)
        self.st["wave2"] = {"loaded_utc": utc(), "config": cfg, "sbatch_args": cfg.get("sbatch_args", []), "n_tuning": len(tuning), "n_formal": len(formal)}
        self.event("wave2_loaded", config=cfg, n_tuning=len(tuning), n_formal=len(formal))

    def process_contingency(self, sel: dict) -> None:
        """x1 formal runs whose members' selections are complete: queue the members that selected another multiplier (frozen contingency)."""
        done = set(self.st["x1_processed"])
        new = []
        for j in self.st["jobs"]:
            if j["kind"] != "formal" or not j["case"].endswith("x1_contingency"):
                continue
            for r in j["runs"]:
                ident = json.dumps(plain(r), sort_keys=True)
                if ident in done or not complete_for(r, sel):
                    continue
                parts = [q for q in split_by_selection(r, sel) if q["lr_mult"] != 1.0]
                new += parts
                done.add(ident)
                self.event("contingency", run=plain(r), reruns=[plain(q) for q in parts], selection=sel_json(sel, {(m, float(r["rho"])) for m in r["members"]}))
        self.st["x1_processed"] = sorted(done)
        if new:
            self.st["rerun_queue"] += [[plain(q) for q in grp] for grp in chunk_by_key(new, self.per_job_of)]

    def tuning_complete(self, sel: dict) -> bool:
        need = {(m, float(r["rho"])) for r in self.st["tuning_runs"] for m in r["members"]}
        return bool(need) and all(sel.get(k, {}).get("complete") for k in need)

    def submit_ready(self) -> None:
        while True:
            n = self.cluster.normal_count()
            if n is None or n >= self.cap:
                return
            rq = self.st.get("resubmit_queue", [])
            if rq:
                it = rq[0]
                if not self._submit(it["name"] + f"_a{it['attempt']}", it["kind"], "resubmit:" + it["case"], Path(it["group_file"]), it["runs"],
                                    attempt=it["attempt"], resubmit_of=it["of"]):
                    return
                rq.pop(0)
                for j in self.st["jobs"]:
                    if j["job"] == it["of"]:
                        j["status"] = "resubmitted"
                continue
            if self.st["tuning_queue"]:
                item = self.st["tuning_queue"][0]
                self.st["counters"]["t"] += 1
                name = f"v9n2_t{self.st['counters']['t']:02d}"
                if item["family"] == "sgr":
                    g = Path(item["group_file"])
                    runs = parse_group_file(g)
                else:
                    runs = [item["run"]]
                    g = self.release_dir / f"{name}.txt"
                    g.write_text(group_line(item["run"]) + "\n")
                if not self._submit(name, "tuning", "tuning", g, runs):
                    self.st["counters"]["t"] -= 1
                    return
                self.st["tuning_queue"].pop(0)
                continue
            sel = selection(self.out, self.st["tuning_runs"])
            self.process_contingency(sel)
            if self.st["formal_pending"]:
                pend = self.st["formal_pending"]
                job, case, used, leftover = next_formal_job(pend, sel, self.per_job_of)
                self.st["counters"]["f"] += 1
                name = f"v9n2_f{self.st['counters']['f']:02d}"
                gfile = self.release_dir / f"{name}.txt"
                gfile.write_text("".join(group_line(r) + "\n" for r in job))
                keys = {(m, float(r["rho"])) for r in job for m in r["members"]}
                snap = sel_json(sel, keys)
                if not self._submit(name, "formal", case, gfile, job, selection=snap,
                                    rule="selected multiplier (seed-917 dev rSum; all three multipliers done for every member)" if case == "selected"
                                    else "x1: seed-917 tuning not complete for every member at submission; frozen contingency applies"):
                    self.st["counters"]["f"] -= 1
                    return
                self.st["formal_pending"] = leftover + [q for k, q in enumerate(pend) if k not in used]
                continue
            if self.st["rerun_queue"]:
                grp = self.st["rerun_queue"][0]
                self.st["counters"]["r"] += 1
                name = f"v9n2_r{self.st['counters']['r']:02d}"
                gfile = self.release_dir / f"{name}.txt"
                gfile.write_text("".join(group_line(r) + "\n" for r in grp))
                keys = {(m, float(r["rho"])) for r in grp for m in r["members"]}
                if not self._submit(name, "rerun", "contingency_rerun", gfile, grp, selection=sel_json(sel, keys)):
                    self.st["counters"]["r"] -= 1
                    return
                self.st["rerun_queue"].pop(0)
                continue
            return

    def finished(self) -> bool:
        st = self.st
        wave2_settled = st["wave2"] is not None or (self.units / "wave2_disabled.json").exists()
        x1 = {json.dumps(plain(r), sort_keys=True) for j in st["jobs"] if j["kind"] == "formal" and j["case"].endswith("x1_contingency") for r in j["runs"]}
        if not (st["gate"] == "pass" and st["tuning_planned"] and wave2_settled and not st["tuning_queue"] and not st["formal_pending"]
                and x1 <= set(st["x1_processed"]) and not st["rerun_queue"] and not st.get("resubmit_queue")):
            return False
        return all(j["status"] in ("ok", "given_up", "resubmitted") for j in st["jobs"])

    def step(self) -> str:
        st = self.st
        if st["done"]:
            return "done"
        mine = self.cluster.my_jobs()
        if mine is None:
            return "squeue unavailable"
        self.refresh(mine)
        if st["gate"] == "wait":
            g, info = self.gate_status(mine)
            if g == "pass":
                st["gate"] = "pass"
                self.event("gate_passed", **info)
            elif g == "fail":
                st["gate"] = "fail"
                st["done"] = True
                self.event("abort", reason="gate failed", **info)
                self.save()
                return "abort"
        if not st["tuning_planned"]:
            t = self.plan_tuning(mine)
            if t == "fail":
                st["done"] = True
                self.event("abort", reason=f"seed-917 shared stage (job {self.tp_job}) did not end with exit=0 or its plan failed")
                self.save()
                return "abort"
        self.maybe_load_wave2()
        if st["gate"] == "pass" and st["tuning_planned"]:
            self.submit_ready()
            blocked = [j["name"] for j in st["jobs"] if j["status"] == "given_up" and j["kind"] == "tuning"]
            msg = f"tuning jobs given up: {blocked}; selections of their (method, rho) cannot complete — a human decides" if blocked else ""
            if msg and msg != st["blocked"]:
                self.event("blocked", reason=msg)
            st["blocked"] = msg
        if self.finished():
            st["done"] = True
            self.event("finished", jobs=len(st["jobs"]), cases={c: sum(1 for j in st["jobs"] if j["case"] == c) for c in sorted({j["case"] for j in st["jobs"]})})
        self.save()
        return "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gate-job", required=True); r.add_argument("--gate-group", required=True); r.add_argument("--tuning-precheck-job", required=True)
    r.add_argument("--out", default=DEF_OUT); r.add_argument("--logs", default=DEF_LOGS); r.add_argument("--units", default=DEF_UNITS)
    r.add_argument("--per-job", type=int, default=2); r.add_argument("--cap", type=int, default=27); r.add_argument("--gate-rsum", type=float, default=30.0)
    r.add_argument("--poll", type=int, default=60); r.add_argument("--max-days", type=float, default=14.0)
    s = sub.add_parser("status"); s.add_argument("--logs", default=DEF_LOGS)
    a = ap.parse_args()
    if a.cmd == "status":
        print(Path(a.logs, "v9_n2_release_status.json").read_text())
        return 0
    orch = Orchestrator(Cluster(os.environ.get("USER", "yinwang"), Path(a.logs)), Path(a.out), Path(a.units), Path(a.logs), a.gate_job, REPO / a.gate_group
                        if not Path(a.gate_group).is_absolute() else Path(a.gate_group), a.tuning_precheck_job, per_job=a.per_job, cap=a.cap,
                        gate_rsum=a.gate_rsum, precheck_record_dir=REPO / "reports/V9/V9_N2_precheck")
    orch.event("start", pid=os.getpid(), args=vars(a), commit=orch.commit())
    orch.save()
    t0 = time.time()
    while not orch.st["done"] and time.time() - t0 < a.max_days * 86400:
        try:
            orch.step()
        except Exception as e:                                   # transient NFS / scheduler errors: log (rate-limited) and keep going
            orch.error(e)
        time.sleep(a.poll)
    orch.event("exit", done=orch.st["done"], pid=os.getpid())
    return 0 if orch.st["done"] and orch.st["gate"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
