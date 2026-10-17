"""V9-N2 release orchestrator: decision logic + a simulated release against a fake cluster (no Slurm, no GPU)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "v9"))
sys.path.insert(0, str(ROOT / "src"))
import n2_release as R  # noqa: E402

ARMS = R.ARMS


def _sel(table):
    return {k: {"rsum": v, "complete": all(x is not None for x in v.values()),
                "selected": None if any(x is None for x in v.values()) else (1.0 if v[1.0] == max(v.values()) else min(m for m in R.MULTS if v[m] == max(v.values())))}
            for k, v in table.items()}


def test_selection_ties_and_incomplete(tmp_path):
    runs = []
    for mu, rs in ((0.5, 400.0), (1.0, 400.0), (2.0, 390.0)):          # tie 0.5 vs 1 -> 1
        r = {"dataset": "f30k", "rho": 0.4, "seed": 917, "method": "VCS_Weight", "lr_mult": mu, "members": ["VCS_Weight"]}; runs.append(r)
        d = R.unit_dir(tmp_path, r); d.mkdir(parents=True); (d / "done.json").write_text(json.dumps({"best": {"rsum": rs, "epoch": 1}}))
    for mu, rs in ((0.5, 410.0), (1.0, 400.0), (2.0, 410.0)):          # tie 0.5 vs 2 (not 1) -> the smaller
        r = {"dataset": "f30k", "rho": 0.4, "seed": 917, "method": "BCE_Weight", "lr_mult": mu, "members": ["BCE_Weight"]}; runs.append(r)
        d = R.unit_dir(tmp_path, r); d.mkdir(parents=True); (d / "done.json").write_text(json.dumps({"best": {"rsum": rs, "epoch": 1}}))
    runs.append({"dataset": "f30k", "rho": 0.4, "seed": 917, "method": "RankMass", "lr_mult": 1.0, "members": ["RankMass"]})   # no done.json
    s = R.selection(tmp_path, runs)
    assert s[("VCS_Weight", 0.4)]["selected"] == 1.0 and s[("BCE_Weight", 0.4)]["selected"] == 0.5
    assert not s[("RankMass", 0.4)]["complete"] and s[("RankMass", 0.4)]["selected"] is None


def test_split_and_next_formal_job():
    sel = _sel({("A", 0.0): {0.5: 1, 1.0: 2, 2.0: 0}, ("B", 0.0): {0.5: 3, 1.0: 2, 2.0: 0}, ("C", 0.4): {0.5: None, 1.0: 1, 2.0: 1}})
    run = {"dataset": "f30k", "rho": 0.0, "seed": 901, "method": "A", "lr_mult": 1.0, "members": ["A", "B"]}
    parts = R.split_by_selection(run, sel)
    assert [(p["method"], p["lr_mult"], p["members"]) for p in parts] == [("B", 0.5, ["B"]), ("A", 1.0, ["A"])]
    c = {"dataset": "f30k", "rho": 0.4, "seed": 902, "method": "C", "lr_mult": 1.0, "members": ["C"]}
    c2 = dict(c, seed=903)
    job, case, used, left = R.next_formal_job([c, c2, run], sel, 2)               # C incomplete, merged run complete -> selected, split fills the job
    assert case == "selected" and used == [2] and len(job) == 2 and left == []
    job, case, used, left = R.next_formal_job([c, c2], sel, 2)                     # nothing complete -> x1 for the first key, both seeds
    assert case == "x1_contingency" and used == [0, 1] and all(r["lr_mult"] == 1.0 for r in job)
    job, case, used, left = R.next_formal_job([run], sel, 1)                       # split larger than the job -> leftover fixed part
    assert case == "selected" and len(job) == 1 and len(left) == 1 and left[0]["fixed"]
    assert [p["lr_mult"] for p in R.contingency_runs([run], sel)] == [0.5]          # only the member selecting != 1 is re-run


class FakeCluster:
    def __init__(self, logs: Path, out: Path, base: int, cap: int, rsum_fn):
        self.logs, self.out, self.base, self.cap, self.rsum_fn = logs, out, base, cap, rsum_fn
        self.queue, self.next_id, self.submitted, self.max_seen = [], 5000, [], 0

    def my_jobs(self):
        return set(self.queue) | {"1042201"} if self.gate_running else set(self.queue)

    gate_running = True

    def normal_count(self):
        return self.base + len(self.queue)

    def submit(self, name, group_file, formal):
        assert self.normal_count() < self.cap, "submitted at or above the cap"
        jid = str(self.next_id); self.next_id += 1; self.queue.append(jid)
        self.submitted.append({"job": jid, "name": name, "group_file": str(group_file), "formal": formal, "lines": Path(group_file).read_text().splitlines(), "official": False})
        self.max_seen = max(self.max_seen, self.normal_count())
        return jid, ""

    def submit_official(self, name, r, kind, sbatch_args):
        assert self.normal_count() < self.cap, "submitted at or above the cap"
        jid = str(self.next_id); self.next_id += 1; self.queue.append(jid)
        self.submitted.append({"job": jid, "name": name, "official": True, "kind": kind, "lines": [R.group_line(r)], "args": sbatch_args})
        self.max_seen = max(self.max_seen, self.normal_count())
        return jid, ""

    def log_of(self, job):
        p = self.logs / f"x_{job}.out"
        return p if p.exists() else None

    def finish(self, k=1, fail=()):
        for jid in self.queue[:k]:
            sub = next(s for s in self.submitted if s["job"] == jid)
            for ln in sub["lines"]:
                f = ln.split(); r = {"dataset": f[0], "rho": float(f[1]), "method": f[2], "seed": int(f[3]), "lr_mult": float(f[4]), "members": f[5].split(",")}
                if jid in fail:
                    continue
                d = R.unit_dir(self.out, r); d.mkdir(parents=True, exist_ok=True)
                (d / "done.json").write_text(json.dumps({"best": {"rsum": self.rsum_fn(r), "epoch": 3}}))
                if sub["official"] and sub["kind"] == "formal":
                    (d / "result.json").write_text("{}")
            (self.logs / f"x_{jid}.out").write_text("exit=0\n" if jid not in fail else "exit=1\n")
        self.queue = self.queue[k:]


def _layout(tmp_path, groups917):
    out, units, logs = tmp_path / "out", tmp_path / "units", tmp_path / "logs"
    for p in (out, units, logs):
        p.mkdir()
    shutil.copy(ROOT / "reports/V9/V9_N2_units/formal_units.json", units)
    shutil.copy(ROOT / "reports/V9/V9_N2_units/formal_g01.txt", units)
    for rho, groups in groups917.items():
        d = out / "f30k" / f"rho{rho:g}" / "s917" / "precheck"; d.mkdir(parents=True)
        (d / f"precheck_f30k_rho{rho:g}_s917.json").write_text(json.dumps({"formal_block_allowed": True, "refusal_reason": "", "loss_coefficients": {"groups": groups}}))
    (logs / "x_1042200.out").write_text("[precheck-exit] rc=0\nexit=0 end=now\n")
    g = out / "f30k/rho0.4/s901/unit_Unweighted_A_lr1"; g.mkdir(parents=True)
    (g / "progress.jsonl").write_text(json.dumps({"epoch": 0, "phase": "warmup_mean_hinge", "mean_loss": 3.1, "dev_rsum": 210.0, "seconds": 420.0}) + "\n")
    return out, units, logs


def _orch(fc, out, units, logs, per_job=2):
    return R.Orchestrator(fc, out, units, logs, "1042201", units / "formal_g01.txt", "1042200", per_job=per_job, cap=27, gate_rsum=30.0,
                          precheck_record_dir=None)


@pytest.mark.parametrize("scenario", ["merged_same", "merged_differs", "with_wave2"])
def test_simulated_release(tmp_path, scenario):
    if scenario in ("merged_same", "with_wave2"):               # seed 917 merges the six arms at rho 0 like seeds 901 / 903
        g0 = [sorted(ARMS), ["PlainFilter"]]
        sel_fn = {("VCS_Weight", 0.4): 0.5, ("BCE_Weight", 0.4): 2.0, ("NCR_A", 0.4): 2.0}
    else:                                                       # seed 917 keeps Nystrom separate at rho 0 and it selects 0.5 -> formal s901 / s903 merged runs split
        g0 = [sorted(set(ARMS) - {"Nystrom_Weight"}), ["Nystrom_Weight"], ["PlainFilter"]]
        sel_fn = {("Nystrom_Weight", 0.0): 0.5, ("Unweighted_A", 0.4): 2.0}
    out, units, logs = _layout(tmp_path, {0.4: [[m] for m in ARMS] + [["PlainFilter"]], 0.0: g0})

    def rsum(r):                                                # dev rSum peaks at the scenario's multiplier (default 1)
        best = 1.0
        for m in r["members"]:
            best = sel_fn.get((m, float(r["rho"])), best)
        return 400.0 - 10.0 * abs(__import__("math").log2(r["lr_mult"] / best))

    fc = FakeCluster(logs, out, base=22, cap=27, rsum_fn=rsum)
    o = _orch(fc, out, units, logs)
    if scenario != "with_wave2":
        (units / "wave2_disabled.json").write_text("{}")
    for it in range(600):
        if scenario == "with_wave2" and it == 12:               # enabled while wave-1 formal runs are being released
            (units / "wave2_enabled.json").write_text(json.dumps({"sbatch_args": ["--partition=RTX6000PRO,H100"], "reason": "test"}))
        o.step()
        if o.st["done"]:
            break
        if it == 3:
            fc.gate_running = False                             # the gate job keeps running a while, then ends with both runs done
            for meth in ("Unweighted_A", "VCS_Weight"):
                d = out / f"f30k/rho0.4/s901/unit_{meth}_lr1"; d.mkdir(parents=True, exist_ok=True); (d / "done.json").write_text(json.dumps({"best": {"rsum": 380.0, "epoch": 9}}))
        fc.finish(k=1)
    assert o.st["done"] and o.st["gate"] == "pass", o.st
    assert fc.max_seen <= 27
    names = [s["name"] for s in fc.submitted if not s["official"]]
    first_formal = min(i for i, n in enumerate(names) if n.startswith("v9n2_f"))
    assert all(n.startswith("v9n2_t") for n in names[:first_formal]) and not any(n.startswith("v9n2_t") for n in names[first_formal:])
    assert all(len(ln.split()) == 6 for s in fc.submitted for ln in s["lines"])
    if scenario == "with_wave2":
        off = [s for s in fc.submitted if s["official"]]
        assert len([s for s in off if s["kind"] == "tuning"]) == 12 and all(len(s["lines"]) == 1 and s["args"] == ["--partition=RTX6000PRO,H100"] for s in off)
        i_last_off_tuning = max(i for i, s in enumerate(fc.submitted) if s["official"] and s["kind"] == "tuning")
        i_first_off_formal = min(i for i, s in enumerate(fc.submitted) if s["official"] and s["kind"] != "tuning")
        assert i_last_off_tuning < i_first_off_formal                     # tuning before formal within wave 2 as well
        off_formal = [r for j in o.st["jobs"] if j["kind"] == "formal" and j["family"] == "official" for r in j["runs"]]
        assert sorted({(r["method"], r["rho"], r["seed"]) for r in off_formal}) == sorted({(m, rho, s) for m in ("NCR_A", "L2RM_A") for rho in (0.4, 0.0) for s in (901, 902, 903)})
    sel = R.selection(out, o.st["tuning_runs"])
    formal = [j for j in o.st["jobs"] if j["kind"] == "formal"]
    covered = {}
    for j in formal:
        for r in j["runs"]:
            covered.setdefault((r["rho"], r["seed"]), set()).update(r["members"])
            if j["case"] == "selected":
                assert all(sel[(m, r["rho"])]["selected"] == r["lr_mult"] for m in r["members"])
            else:
                assert r["lr_mult"] == 1.0
    plan = json.loads((units / "formal_units.json").read_text())["runs"]
    for r in plan:
        assert set(r["members"]) <= covered[(float(r["rho"]), int(r["seed"]))]
    assert {j["case"] for j in formal} >= {"x1_contingency"}
    x1 = [r for j in formal if j["case"].endswith("x1_contingency") for r in j["runs"]]
    want = sorted((r["rho"], r["seed"], m) for r in x1 for m in r["members"] if sel[(m, r["rho"])]["selected"] != 1.0)
    got = sorted((r["rho"], r["seed"], m) for j in o.st["jobs"] if j["kind"] == "rerun" for r in j["runs"] for m in r["members"])
    assert want == got and want, (want, got)
    ev = [json.loads(ln)["event"] for ln in (units / "launch_ledger.jsonl").read_text().splitlines()]
    assert {"gate_passed", "tuning_planned", "contingency", "finished"} <= set(ev)
    if scenario == "merged_differs":
        assert any(r["members"] == ["Nystrom_Weight"] and r["rho"] == 0.0 and r["lr_mult"] == 0.5 for j in o.st["jobs"] if j["kind"] in ("formal", "rerun") for r in j["runs"])


def test_gate_failure_aborts(tmp_path):
    out, units, logs = _layout(tmp_path, {0.4: [[m] for m in ARMS], 0.0: [sorted(ARMS)]})
    (out / "f30k/rho0.4/s901/unit_Unweighted_A_lr1/progress.jsonl").write_text(json.dumps({"epoch": 0, "mean_loss": 51.2, "dev_rsum": 0.6, "seconds": 400.0}) + "\n")
    fc = FakeCluster(logs, out, base=20, cap=27, rsum_fn=lambda r: 1.0)
    o = _orch(fc, out, units, logs)
    assert o.step() == "abort" and o.st["done"] and o.st["gate"] == "fail" and fc.submitted == []


def test_incomplete_job_resubmitted_once(tmp_path):
    out, units, logs = _layout(tmp_path, {0.4: [[m] for m in ARMS], 0.0: [sorted(ARMS)]})
    fc = FakeCluster(logs, out, base=24, cap=27, rsum_fn=lambda r: 400.0)
    o = _orch(fc, out, units, logs)
    o.step()                                                    # gate passes, tuning planned, 3 tuning jobs submitted (24 -> 27)
    first = fc.queue[0]
    fc.finish(k=1, fail=(first,))                               # first tuning job ends without done.json
    o.step()
    js = {j["job"]: j for j in o.st["jobs"]}
    assert js[first]["status"] == "resubmitted" and any(j.get("resubmit_of") == first and j["attempt"] == 2 for j in o.st["jobs"])


def test_run_unit_refuses_members_outside_the_coefficient_group(tmp_path):
    from v9.n2.pipeline import run_unit
    d = tmp_path / "f30k/rho0/s901/precheck"; d.mkdir(parents=True)
    (d / "precheck_f30k_rho0_s901.json").write_text(json.dumps({"formal_block_allowed": True, "loss_coefficients": {"groups": [["BCE_Weight", "VCS_Weight"], ["PlainFilter"]], "merged": {}}}))
    with pytest.raises(ValueError, match="coefficient group"):
        run_unit(None, None, tmp_path, "f30k", 0.0, "BCE_Weight", 901, members=["BCE_Weight", "PlainFilter"])
    with pytest.raises(Exception) as ei:                        # valid attribution passes the check and fails later only on the dummy inputs
        run_unit(None, None, tmp_path, "f30k", 0.0, "BCE_Weight", 901, members=["VCS_Weight", "BCE_Weight"])
    assert "coefficient group" not in str(ei.value)


def test_v1_state_migrates(tmp_path):
    """an orchestrator state written before the wave-2 extension (tuning queue of group-file strings) loads and keeps working."""
    out, units, logs = _layout(tmp_path, {0.4: [[m] for m in ARMS], 0.0: [sorted(ARMS)]})
    fc = FakeCluster(logs, out, base=26, cap=27, rsum_fn=lambda r: 400.0)
    o = _orch(fc, out, units, logs)
    o.step()                                                    # gate + plan; one tuning job (26 -> 27)
    st = json.loads(o.state_path.read_text())
    st["tuning_queue"] = [q["group_file"] for q in st["tuning_queue"]]
    for k in ("x1_processed", "wave2", "resubmit_queue"):
        st.pop(k, None)
    o.state_path.write_text(json.dumps(st))
    o2 = _orch(fc, out, units, logs)
    assert all(isinstance(q, dict) and q["family"] == "sgr" for q in o2.st["tuning_queue"]) and o2.st["wave2"] is None
    fc.finish(k=1); o2.step()
    assert len(fc.submitted) == 2


def test_wave2_methods_added_one_at_a_time(tmp_path):
    """L2RM_A enabled first (its timing landed), NCR_A later: each method's 6 tuning runs are queued when it is listed, never twice."""
    out, units, logs = _layout(tmp_path, {0.4: [[m] for m in ARMS], 0.0: [sorted(ARMS)]})
    fc = FakeCluster(logs, out, base=0, cap=27, rsum_fn=lambda r: 400.0)
    o = _orch(fc, out, units, logs)
    (units / "wave2_enabled.json").write_text(json.dumps({"methods": ["L2RM_A"], "sbatch_args_by_method": {"L2RM_A": ["--partition=RTX6000PRO,H100,L40S"]}}))
    o.step()
    off = [s for s in fc.submitted if s["official"]]
    assert len(off) == 6 and all("L2RM_A" in s["lines"][0] and s["args"] == ["--partition=RTX6000PRO,H100,L40S"] for s in off)
    fc.finish(k=len(fc.queue))                                  # the queue drains (cap 27 reached in the first step)
    (units / "wave2_enabled.json").write_text(json.dumps({"methods": ["L2RM_A", "NCR_A"], "sbatch_args_by_method": {"L2RM_A": ["--partition=RTX6000PRO,H100,L40S"], "NCR_A": ["--partition=RTX6000PRO,H100"]}}))
    n_before = len(fc.submitted)
    o.step(); o.step()
    assert all("NCR_A" in s["lines"][0] for s in fc.submitted[n_before:n_before + 6])   # NCR_A tuning ahead of the formal runs still pending
    off = [s for s in fc.submitted if s["official"]]
    tun = [s for s in off if s["kind"] == "tuning"]
    assert len([s for s in tun if "NCR_A" in s["lines"][0]]) == 6 and len([s for s in tun if "L2RM_A" in s["lines"][0]]) == 6
    assert all(s["args"] == ["--partition=RTX6000PRO,H100"] for s in off if "NCR_A" in s["lines"][0])
    n_off_formal = len([r for r in o.st["formal_pending"] if r["method"] in ("NCR_A", "L2RM_A")]) + len([r for j in o.st["jobs"] if j["kind"] == "formal" and j["family"] == "official" for r in j["runs"]])
    assert o.st["wave2"]["methods_loaded"] == ["L2RM_A", "NCR_A"] and n_off_formal == 12


def test_wave1_pause_keeps_cancelled_jobs_and_lets_wave2_flow(tmp_path):
    """main session 2026-10-17: pause wave 1, cancel its pending jobs, keep them for unchanged resubmission; wave 2 keeps flowing."""
    out, units, logs = _layout(tmp_path, {0.4: [[m] for m in ARMS], 0.0: [sorted(ARMS)]})
    fc = FakeCluster(logs, out, base=0, cap=27, rsum_fn=lambda r: 400.0)
    o = _orch(fc, out, units, logs)
    o.step()                                                    # gate + tuning plan + first wave-1 submissions (27 queued)
    pending = list(fc.queue[-5:])                               # the last 5 submitted jobs play "pending"
    (units / "wave1_paused.json").write_text(json.dumps({"reason": "test", "cancelled_jobs": pending}))
    o.step()                                                    # pause registered before any cancel
    n_before = len(fc.submitted)
    fc.queue = [j for j in fc.queue if j not in pending]        # scancel of the pending wave-1 jobs (no done.json)
    o.step()
    js = {j["job"]: j for j in o.st["jobs"]}
    assert all(js[p]["status"] == "paused" for p in pending) and len(o.st["paused_groups"]) == 5
    assert len(fc.submitted) == n_before                        # nothing resubmitted, no new wave-1 work while paused
    (units / "wave2_enabled.json").write_text(json.dumps({"methods": ["L2RM_A"]}))
    o.step()
    new = fc.submitted[n_before:]
    assert new and all(s["official"] for s in new)              # wave 2 flows while wave 1 is paused
    files = {js[p]["group_file"] for p in pending}
    (units / "wave1_paused.json").unlink()
    fc.queue = []                                               # room for the resumed groups
    o.step()
    resumed = [j for j in o.st["jobs"] if j["case"].startswith("resume:")]
    assert {j["group_file"] for j in resumed} == files and all(js[p]["status"] == "resubmitted" for p in pending)
    ev = [json.loads(ln)["event"] for ln in (units / "launch_ledger.jsonl").read_text().splitlines()]
    assert {"wave1_paused", "paused_job_kept", "wave1_resumed"} <= set(ev)
