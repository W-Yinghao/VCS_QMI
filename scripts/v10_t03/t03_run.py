"""V10 T03 unit runner driven by v10/protocol/T03_UNITS.csv (decision section 9; the protocol agent's file, sha256-pinned).

  t03_run.py units --check                  the CSV against its pin and the independent derivation of decision section 9
  t03_run.py gen --root-seed S              the complete suite of data root seed S + the package checker on it (checks/)
  t03_run.py gate                           the uniform-derangement / U gates (gates/T03_GATES.json)
  t03_run.py run (--unit-ids A,B | --block smoke) [--shard k/N] [--threads T] [--retry-failed]
  t03_run.py select                         the tuning-suite CAL selection table (selection/T03_CAL_SELECTION.json)
  t03_run.py status                         status per block
Outputs (V10_T03_OUT_ROOT, default /projects/EEG-foundation-model/yinghao/outputs_v10/T03):
  data/seed<S>/                 package-layout suite + T03_SUITE_RECORD.json (generator and sample hashes)
  checks/check_synthetic_seed<S>.json   truth-free package-checker summary (status, n_checks, hashes): the suite gate reads this
  checks/evaluator_only/check_synthetic_seed<S>.json   the checker's full record (its rows carry truth values: evaluator-only)
  gates/T03_GATES.json
  runs/<unit>/result.json       sample-only record: config, resolved hyperparameters, data / stream / init hashes, CAL trace,
                                own-CAL selection, cost, provenance, status COMPLETE / FAIL / BLOCKED
  runs/<unit>/train_steps.json.gz, runs/<unit>/checkpoints.pt (all 8 states), runs/<unit>/evaluator_only/evaluation.json
A unit's prerequisites (else BLOCKED, nothing trained): its suite checked (PASS); the gates PASS under the same T03 code hash; every
depends_on unit COMPLETE under the same T03 code hash; formal rows: their tuning-suite group selection.  Training and CAL readout /
selection run inside the truth barrier; the evaluator runs after it.  A unit whose barrier block recorded any denial is FAIL (stage
truth_barrier), even when the refused call was swallowed; it is not evaluated.  FAIL records are kept (a retry moves them to attempts/).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch

import t03_common as C
import t03_data as D
import t03_evaluate as E
import t03_select as S
import t03_train as TR
import t03_units as U

STATUSES = ("COMPLETE", "FAIL", "BLOCKED", "NOT_RUN")


# ------------------------------------------------------------------------------------------------------------ paths / records
def runs_dir(root: Path | None = None) -> Path:
    return (root or C.out_root()) / "runs"


def unit_dir(uid: str, root: Path | None = None) -> Path:
    return runs_dir(root) / uid


def read_result(uid: str, root: Path | None = None) -> dict | None:
    p = unit_dir(uid, root) / "result.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def unit_status(uid: str, root: Path | None = None) -> str:
    r = read_result(uid, root)
    return r["status"] if r else "NOT_RUN"


def check_path(seed: int, root: Path | None = None) -> Path:
    """The truth-free checker summary (the suite gate); the full record is full_check_path()."""
    return (root or C.out_root()) / "checks" / f"check_synthetic_seed{seed}.json"


def full_check_path(summary: Path) -> Path:
    """The package checker's full record beside a summary, under evaluator_only (its rows carry table / analytic S values)."""
    return summary.parent / C.TRUTH_DIR_NAME / summary.name


def gates_path(root: Path | None = None) -> Path:
    return (root or C.out_root()) / "gates" / "T03_GATES.json"


class Blocked(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------------------ suite + checker
def run_package_checker(suite: Path, out_json: Path) -> dict:
    """The package synthetic/tests/check_synthetic.py run_checks (sha256-pinned, code unchanged) on a suite.  Evaluator-side (it reads
    truth): never inside the barrier.  Its full record - the checker rows carry table / analytic S values - goes to
    full_check_path(out_json) under evaluator_only; out_json gets the truth-free summary (status, check counts, the name of a failed
    check, hashes) that the suite gate reads.  The checker's module-level ROOT only places its temporary regeneration directory; it
    is pointed at a temporary directory inside the evaluator-only directory so nothing is written into the package tree."""
    if C.barrier_active():
        raise C.TruthAccessError("the package checker reads truth; not inside the barrier")
    C.verify_pinned_file(C.CHECKER_PATH, C.CHECKER_SHA256, "package checker")
    D.load_generator()                                               # the checker's `from v10_synthetic import ...` reuses the pinned module
    spec = importlib.util.spec_from_file_location("v10_t03_check_synthetic", C.CHECKER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    full = full_check_path(out_json)
    full.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rec = {"schema": "v10_t03.package_check.v1", "checker": str(C.CHECKER_PATH.relative_to(C.REPO)), "checker_sha256": C.CHECKER_SHA256,
           "suite": str(suite), "redirect": "module ROOT -> a temporary directory beside this record (regeneration scratch only)"}
    failed_name = None
    with tempfile.TemporaryDirectory(prefix=".checker_", dir=full.parent) as tmp:
        (Path(tmp) / "validation").mkdir()
        mod.ROOT = Path(tmp)
        try:
            res = mod.run_checks(suite)
            rec.update({"status": "PASS" if res["status"] == "passed" else "FAIL", "result": res})
        except AssertionError as e:
            rec.update({"status": "FAIL", "failed_check": str(e)})
            try:                                                     # the checker raises AssertionError(json.dumps(row))
                failed_name = json.loads(str(e)).get("name")
            except (ValueError, AttributeError):
                failed_name = "<unparsed>"
    rec["seconds"] = time.perf_counter() - t0
    rec["utc"] = C.utc_now()
    full_sha = C.write_json_atomic(full, rec)
    res = rec.get("result") or {}
    summary = {"schema": "v10_t03.package_check_summary.v1", "checker": rec["checker"], "checker_sha256": C.CHECKER_SHA256, "suite": str(suite),
               "status": rec["status"], "n_checks": res.get("n_checks"), "n_cases": res.get("n_cases"),
               "n_checks_passed": sum(1 for r in res.get("checks", []) if r.get("passed") is True) if res else None,
               "failed_check_name": failed_name, "seconds": rec["seconds"], "utc": rec["utc"],
               "full_record": {"path": str(full.relative_to(out_json.parent)), "sha256": full_sha,
                               "note": "the checker rows carry truth values (table / analytic S): evaluator-only"}}
    C.write_json_atomic(out_json, summary)
    return summary


def ensure_checked_suite(seed: int, root: Path | None = None) -> tuple[Path, dict, dict]:
    """(suite directory, suite record, truth-free checker summary); the checker reruns when the suite manifest changed."""
    record = D.ensure_suite(seed, root)
    suite = D.suite_dir(seed, root)
    cp = check_path(seed, root)
    chk = json.loads(cp.read_text()) if cp.exists() else None
    if chk is None or chk.get("schema") != "v10_t03.package_check_summary.v1" or chk.get("suite_manifest_sha256") != record["manifest_sha256"]:
        chk = run_package_checker(suite, cp)
        chk["suite_manifest_sha256"] = record["manifest_sha256"]
        C.write_json_atomic(cp, chk)
    return suite, record, chk


# ------------------------------------------------------------------------------------------------------------ resolution
def _dep_results(row: dict, ident: dict, root: Path | None) -> dict[str, dict]:
    out = {}
    for d in row["depends_on"]:
        r = read_result(d, root)
        if r is None or r["status"] != "COMPLETE":
            raise Blocked(f"dependency {d} is {r['status'] if r else 'NOT_RUN'}")
        if r["provenance"]["identity"]["t03_code_hash"] != ident["t03_code_hash"]:
            raise Blocked(f"dependency {d} ran under T03 code hash {r['provenance']['identity']['t03_code_hash']} != {ident['t03_code_hash']}")
        out[d] = r
    return out


def group_selection(method: str, deps: dict[str, dict]) -> dict:
    cands = []
    for uid, r in deps.items():
        cfg = r["config"]
        if cfg["method"] != method or r["config"]["block"] != "CAL":
            raise Blocked(f"{uid} is not a CAL candidate of {method}")
        cands.append({"ordinal": S.candidate_ordinal(method, cfg["lr"], cfg["cap"]), "unit_id": uid, "lr": cfg["lr"], "cap": cfg["cap"],
                      "trace": r["checkpoints"]})
    expected = [o for o, _, _ in S.candidates_for(method)]
    if sorted(c["ordinal"] for c in cands) != expected:
        raise Blocked(f"candidate set {sorted(c['ordinal'] for c in cands)} != the frozen order {expected}")
    try:
        return S.select_group(cands)
    except S.SelectionBlocked as e:
        raise Blocked(str(e)) from e


def resolve(row: dict, ident: dict, root: Path | None) -> tuple[float, int | None, dict]:
    """(lr, cap, resolution record) of a parsed unit row."""
    deps = _dep_results(row, ident, root)
    if row["lr"] not in ("CAL", "CAL_K16"):
        return float(row["lr"]), row["cap"], {"source": "unit table", "dependencies_complete": sorted(deps)}
    first = deps[row["depends_on"][0]]["config"]
    if (first["family"], first["m"], float(first["rho"])) != (row["family"], row["m"], row["rho"]):
        raise Blocked("dependencies belong to another condition")
    sel = group_selection(row["method"], deps)
    lr, cap = float(sel["selected"]["lr"]), sel["selected"]["cap"]
    return lr, (int(cap) if cap is not None else None), {
        "source": "tuning-suite CAL selection (data root 1017, algo 1017) of the same (condition, method)"
                  + (" at K16, inherited by this capacity row" if row["lr"] == "CAL_K16" else ""),
        "group_selection": sel, "dependency_result_sha256": {d: C.sha256_file(unit_dir(d, root) / "result.json") for d in sorted(deps)}}


def gates_ok(ident: dict, root: Path | None) -> dict:
    p = gates_path(root)
    if not p.exists():
        raise Blocked("gates not run (t03_run.py gate)")
    g = json.loads(p.read_text())
    if g["status"] != "PASS" or g["identity"]["t03_code_hash"] != ident["t03_code_hash"]:
        raise Blocked(f"gates {g['status']} under T03 code hash {g['identity']['t03_code_hash']} (current {ident['t03_code_hash']})")
    return {"path": str(p), "sha256": C.sha256_file(p), "status": g["status"]}


# ------------------------------------------------------------------------------------------------------------ one unit
def _archive_attempt(d: Path) -> str | None:
    if not d.exists() or not any(d.iterdir()):
        return None
    att = d / "attempts"
    k = 1 + max([int(p.name) for p in att.iterdir() if p.name.isdigit()], default=0) if att.exists() else 1
    dest = att / str(k)
    dest.mkdir(parents=True)
    for p in list(d.iterdir()):
        if p.name != "attempts":
            shutil.move(str(p), str(dest / p.name))
    return str(dest)


def execute_unit(cfg: TR.UnitConfig, *, suite: Path, record: dict, out: Path, ident: dict, env: dict, extra: dict) -> dict:
    """Train (inside the truth barrier) + own-CAL selection + checkpoints + result.json, then the evaluator.  Returns the result."""
    t0, cpu0 = time.perf_counter(), time.process_time()
    out.mkdir(parents=True, exist_ok=True)
    base = {"schema": "v10_t03.unit_result.v1", "unit_id": cfg.unit_id, "block": cfg.block, "config": None, "status": "FAIL",
            "started_utc": C.utc_now(), "provenance": {"identity": ident, "environment": env}, **extra}
    stage = "train"
    n_denied0 = C.denial_count()
    try:
        with C.truth_barrier(f"train+CAL {cfg.unit_id}"):
            t_load = time.perf_counter()
            smp = D.load_case_samples(record, suite, cfg.case_id, ("fit", "cal"))
            t_load = time.perf_counter() - t_load
            rec, log, ckpts = TR.train_unit(cfg, smp["fit"], smp["cal"])
            sel = S.select_checkpoint(rec["checkpoints"], cfg.unit_id)
            t_save = time.perf_counter()
            torch.save({"format": "v10_t03.checkpoints.v1", "unit_id": cfg.unit_id, "states": ckpts,
                        "meta": {"method": cfg.method, "case_id": cfg.case_id, "steps": sorted(ckpts)}}, out / "checkpoints.pt")
            steps_sha = C.write_json_gz_atomic(out / "train_steps.json.gz", log)
            t_save = time.perf_counter() - t_save
            denied = C.barrier_denials()[n_denied0:]                   # this unit's own barrier block only
        if denied:                                                   # a refused (even if swallowed) access fails the unit
            stage = "truth_barrier"
            raise C.TruthAccessError(f"{len(denied)} truth-barrier denial(s) recorded during sample loading / training / CAL readout; "
                                     f"first: {denied[0]['event']} of {denied[0]['path']} ({denied[0]['reason']})")
        result = {**base, **rec, "data": {"suite": str(suite), "case_id": cfg.case_id, "generator_sha256": record["generator_sha256"],
                  "suite_manifest_sha256": record["manifest_sha256"], "numpy_version_generation": record["numpy_version"],
                  "sample_sha256": {r: smp[r].sample_sha256 for r in ("fit", "cal")},
                  "eval_sample_sha256_record": record["cases"][cfg.case_id]["roles"]["eval"]["sample_sha256"],
                  "n": {r: smp[r].n for r in ("fit", "cal")}},
                  "selection": {"scope": {"CAL": "unit level only; the tuning-suite choice is the group selection (t03_run.py select)",
                                          "smoke": "unit level; smoke outputs never enter any selection",
                                          }.get(cfg.block, "OWN_CAL_UJ_CKPT: this replicate's own CAL samples"), "own_cal": sel},
                  "artifacts": {"checkpoints_sha256": C.sha256_file(out / "checkpoints.pt"), "train_steps_sha256": steps_sha},
                  "truth_barrier": {"active_during": "sample loading, training, CAL readout, own-CAL selection", "denials_during_unit": denied}}
        result["cost"].update({"wall_seconds_sample_load": t_load, "wall_seconds_artifact_save": t_save})
        result["status"] = "TRAINED"
        C.write_json_atomic(out / "result.json", result)
        stage = "evaluate"
        t_ev = time.perf_counter()
        ev = E.evaluate_unit(result, out, suite, record)
        ev_sha = C.write_json_atomic(out / C.TRUTH_DIR_NAME / "evaluation.json", ev)
        result["evaluation"] = {"path": f"{C.TRUTH_DIR_NAME}/evaluation.json", "sha256": ev_sha, "status": ev["status"],
                                "note": "evaluator-only (truth-based) record; the selection code never reads it"}
        result["cost"]["wall_seconds_evaluation"] = time.perf_counter() - t_ev
        result["cost"]["wall_seconds_unit_total"] = time.perf_counter() - t0
        result["cost"]["cpu_seconds_unit_total"] = time.process_time() - cpu0
        result["status"] = "COMPLETE" if ev["status"] == "PASS" else "FAIL"
        if result["status"] == "FAIL":
            result["failure"] = {"stage": "evaluate", "reason": "evaluation checks failed", "checks": ev["checks"]}
        result["finished_utc"] = C.utc_now()
        C.write_json_atomic(out / "result.json", result)
        return result
    except Exception as exc:                                          # noqa: BLE001 - every failure is recorded and kept
        fail = {**base, "config": TR.asdict(cfg), "status": "FAIL", "finished_utc": C.utc_now(),
                "failure": {"stage": stage, "type": type(exc).__name__, "message": str(exc)[:2000], "traceback": traceback.format_exc()[-8000:]},
                "cost": {"wall_seconds_unit_total": time.perf_counter() - t0},
                "truth_barrier": {"denials_during_unit": C.barrier_denials()[n_denied0:]}}
        prev = out / "result.json"
        if prev.exists():
            fail["partial_result_sha256"] = C.sha256_file(prev)
            shutil.copy2(prev, out / "result.partial.json")
        C.write_json_atomic(out / "result.json", fail)
        return fail


def run_unit(raw_row: dict, *, ident: dict, env: dict, device: str = "cpu", dtype: str = "float32", retry_failed: bool = False,
             root: Path | None = None) -> str:
    row = U.parse(raw_row)
    uid = row["unit_id"]
    d = unit_dir(uid, root)
    prev = read_result(uid, root)
    if prev is not None and prev["status"] == "COMPLETE":
        print(f"[t03] {uid} COMPLETE (kept)", flush=True)
        return "COMPLETE"
    if prev is not None and prev["status"] == "FAIL" and not retry_failed:
        print(f"[t03] {uid} FAIL kept (use --retry-failed to rerun; the record moves to attempts/)", flush=True)
        return "FAIL"
    if prev is not None and prev["status"] == "BLOCKED":            # nothing was trained: the new attempt replaces the record
        (d / "result.json").unlink()
        archived = None
    else:                                                            # FAIL (retry), TRAINED (crash before evaluation) or partial files
        archived = _archive_attempt(d)
    extra = {"unit_row": raw_row, "previous_attempt": archived}
    try:
        suite, record, chk = ensure_checked_suite(row["data_root_seed"], root)
        if chk["status"] != "PASS":
            raise Blocked(f"suite seed{row['data_root_seed']} failed the package checker")
        g = gates_ok(ident, root)
        n_denied0 = C.denial_count()
        with C.truth_barrier(f"hyperparameter resolution {uid}"):   # the tuning-suite selection reads result.json records only
            lr, cap, resolution = resolve(row, ident, root)
        if C.denial_count() != n_denied0:
            raise C.TruthAccessError(f"truth-barrier denial(s) during hyperparameter resolution: {C.barrier_denials()[n_denied0:]}")
    except Blocked as b:
        rec = {"schema": "v10_t03.unit_result.v1", "unit_id": uid, "block": row["block"], "status": "BLOCKED", "reason": str(b),
               "utc": C.utc_now(), "provenance": {"identity": ident}, **extra}
        C.write_json_atomic(d / "result.json", rec)
        print(f"[t03] {uid} BLOCKED: {b}", flush=True)
        return "BLOCKED"
    except Exception as exc:                                          # noqa: BLE001 - e.g. a suite that fails its hash verification
        rec = {"schema": "v10_t03.unit_result.v1", "unit_id": uid, "block": row["block"], "status": "FAIL", "utc": C.utc_now(),
               "failure": {"stage": "prerequisites", "type": type(exc).__name__, "message": str(exc)[:2000],
                           "traceback": traceback.format_exc()[-8000:]}, "provenance": {"identity": ident}, **extra}
        C.write_json_atomic(d / "result.json", rec)
        print(f"[t03] {uid} FAIL (prerequisites): {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        return "FAIL"
    cfg = TR.UnitConfig(unit_id=uid, block=row["block"], method=row["method"], family=row["family"], m=row["m"], rho=row["rho"], K=row["K"],
                        lr=lr, cap=cap, data_root_seed=row["data_root_seed"], algo_seed=row["algo_seed"], steps=row["steps"],
                        ckpt_every=row["ckpt_every"], dtype=dtype, device=device, selection_rule=row["selection_rule"])
    extra.update({"hyperparameter_resolution": resolution, "prerequisites": {"suite_check": {"path": str(check_path(row['data_root_seed'], root)),
                  "status": chk["status"]}, "gates": g}})
    print(f"[t03] {uid} {row['block']} {cfg.case_id} {cfg.method} K={cfg.K} lr={cfg.lr:g} cap={cfg.cap} data={cfg.data_root_seed} "
          f"algo={cfg.algo_seed} start {C.utc_now()}", flush=True)
    res = execute_unit(cfg, suite=suite, record=record, out=d, ident=ident, env=env, extra=extra)
    c = res.get("cost", {})
    print(f"[t03] {uid} {res['status']} wall={c.get('wall_seconds_unit_total', float('nan')):.1f}s "
          f"train={c.get('wall_seconds_train_unit', float('nan')):.1f}s"
          + (f" failure={res['failure']['stage']}:{res['failure'].get('type', '')}: {res['failure'].get('message', res['failure'].get('reason'))[:300]}"
             if res["status"] == "FAIL" else ""), flush=True)
    return res["status"]


# ------------------------------------------------------------------------------------------------------------ commands
def cmd_units(a) -> int:
    res = U.check_against_derivation()
    print(json.dumps({k: v for k, v in res.items() if k != "field_differences"}, indent=1))
    if a.output:
        C.write_json_atomic(a.output, res)
    return 0 if res["status"] == "PASS" else 1


def cmd_gen(a) -> int:
    for seed in a.root_seed:
        suite, record, chk = ensure_checked_suite(seed)
        print(json.dumps({"root_seed": seed, "suite": str(suite), "manifest_sha256": record["manifest_sha256"], "n_cases": record["n_cases"],
                          "check": chk["status"], "check_n": chk.get("n_checks"), "failed_check_name": chk.get("failed_check_name")}))
        if chk["status"] != "PASS":
            return 1
    return 0


def cmd_gate(a) -> int:
    ident = C.identity()
    res = {**__import__("t03_gates").run_gates(), "identity": ident, "environment": C.environment("cpu"), "utc": C.utc_now()}
    C.write_json_atomic(gates_path(), res)
    print(json.dumps({"status": res["status"], **{k: v["status"] for k, v in res["gates"].items()}, "seconds": res["seconds"]}))
    return 0 if res["status"] == "PASS" else 1


def run_threads(requested: int | None) -> int:
    """torch intra-op threads of a run process: --threads, else SLURM_CPUS_PER_TASK, else the torch default."""
    t = requested if requested is not None else int(os.environ.get("SLURM_CPUS_PER_TASK", torch.get_num_threads()))
    if isinstance(t, bool) or int(t) != t or t < 1:
        raise SystemExit(f"[t03] bad thread count {t!r}")
    return int(t)


def cmd_run(a) -> int:
    C.check_job_pin()
    rows = U.load_rows()
    chk = U.check_against_derivation()
    if chk["status"] != "PASS":
        raise SystemExit(f"[t03] REFUSED: {C.UNITS_CSV} does not match its pin / the decision-9 derivation")
    if a.unit_ids:
        want = a.unit_ids.split(",")
        by_id = {r["unit_id"]: r for r in rows}
        missing = [u for u in want if u not in by_id]
        if missing:
            raise SystemExit(f"[t03] unknown unit ids {missing}")
        todo = [by_id[u] for u in want]
    else:
        todo = [r for r in rows if r["block"] == a.block]
    if a.shard:                                                      # k/N: every N-th unit of the selection, offset k (0-based)
        k, n = (int(v) for v in a.shard.split("/"))
        if not 0 <= k < n:
            raise SystemExit(f"[t03] bad --shard {a.shard}")
        todo = todo[k::n]
    if a.limit:
        todo = todo[: a.limit]
    torch.set_num_threads(run_threads(a.threads))
    ident, env = C.identity(), C.environment(a.device)
    if ident["git_dirty"] and os.environ.get(C.PIN_ENV):
        raise SystemExit(f"[t03] REFUSED: uncommitted changes in pinned paths:\n{ident['git_dirty']}")
    t0 = time.perf_counter()
    tally = {s: 0 for s in STATUSES}
    for r in todo:
        st = run_unit(r, ident=ident, env=env, device=a.device, dtype=a.dtype, retry_failed=a.retry_failed)
        tally[st] = tally.get(st, 0) + 1
    print(f"[t03] run done units={len(todo)} {tally} wall={time.perf_counter() - t0:.1f}s", flush=True)
    return 0 if tally.get("FAIL", 0) == 0 and tally.get("BLOCKED", 0) == 0 else 1


def cmd_select(a) -> int:
    rows = [U.parse(r) for r in U.load_rows()]
    ident = C.identity()
    groups = {}
    for r in rows:
        if r["block"] == "CAL":
            groups.setdefault((r["family"], r["m"], r["rho"], r["method"]), []).append(r["unit_id"])
    out = []
    with C.truth_barrier("CAL group selection"):
        for (f, m, rho, method), ids in groups.items():
            ent = {"case_id": C.case_id(f, m, rho), "family": f, "m": m, "rho": rho, "method": method, "candidates": ids}
            n_denied0 = C.denial_count()
            try:
                deps = _dep_results({"depends_on": ids}, ident, None)
                sel = {"status": "SELECTED", **group_selection(method, deps)}
            except (Blocked, C.TruthAccessError) as b:
                sel = {"status": "BLOCKED", "reason": f"{type(b).__name__}: {b}"}
            if C.denial_count() != n_denied0:                        # a refused (even if swallowed) access blocks the group
                sel = {"status": "BLOCKED", "reason": "truth-barrier denial(s) during the group selection",
                       "denials": C.barrier_denials()[n_denied0:]}
            ent.update(sel)
            out.append(ent)
    res = {"schema": "v10_t03.cal_selection.v1", "identity": ident, "utc": C.utc_now(), "groups": out,
           "n_selected": sum(e["status"] == "SELECTED" for e in out), "n_groups": len(out)}
    p = C.write_json_atomic(C.out_root() / "selection" / "T03_CAL_SELECTION.json", res)
    print(json.dumps({"n_groups": res["n_groups"], "n_selected": res["n_selected"], "sha256": p}))
    return 0


def cmd_status(a) -> int:
    rows = U.load_rows()
    tab = {}
    for r in rows:
        tab.setdefault(r["block"], {s: 0 for s in STATUSES + ("TRAINED",)})
        tab[r["block"]][unit_status(r["unit_id"])] = tab[r["block"]].get(unit_status(r["unit_id"]), 0) + 1
    print(json.dumps(tab, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("units")
    u.add_argument("--check", action="store_true")
    u.add_argument("--output", type=Path)
    g = sub.add_parser("gen")
    g.add_argument("--root-seed", type=int, nargs="+", required=True)
    sub.add_parser("gate")
    r = sub.add_parser("run")
    grp = r.add_mutually_exclusive_group(required=True)
    grp.add_argument("--unit-ids")
    grp.add_argument("--block", choices=tuple(U.BLOCK_COUNTS))
    r.add_argument("--limit", type=int)
    r.add_argument("--shard", help="k/N: run every N-th selected unit starting at k (0-based), for parallel processes of one job")
    r.add_argument("--threads", type=int, help="torch intra-op threads of this process (default SLURM_CPUS_PER_TASK, else torch's)")
    r.add_argument("--device", default="cpu")
    r.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    r.add_argument("--retry-failed", action="store_true")
    sub.add_parser("select")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    return {"units": cmd_units, "gen": cmd_gen, "gate": cmd_gate, "run": cmd_run, "select": cmd_select, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
