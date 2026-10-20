"""V10 TEST_REPEAT134 evaluator of the frozen final checkpoints (GPU, Slurm only).

    evaluate.py gate <gate_id>    DEV-reproduction gate: re-evaluate the gate unit's frozen checkpoint on DEV32 on the gate's
                                  GPU type and require bitwise agreement with its recorded DEV rows and DEV32 metrics.
    evaluate.py test <unit_key>   TEST_REPEAT134: evaluate one frozen final checkpoint on the 134 TEST cores, on the GPU type
                                  that produced its recorded DEV32, after that (family, GPU type) gate PASSed and the TEST
                                  inputs are locked.

Science is imported unchanged from the unit's frozen execution source, verified byte-identical first:
  bf1   handoff/v10_bf1_20261017/source scripts/v10_bf1/run.py: freeze(), new_trainer, load_checkpoint, evaluate
  supp  the supplement / RuLSIF_exact_v2 source scripts/v10_supplement/run.py: source inventory == release, new_trainer,
        CP.load, evaluate.  Its RegistrationSession admits CAL/DEV only, so the runner's global is re-bound to
        TestRepeatSession, whose __init__ is the frozen one with the role gate changed (checked against the frozen source).
The TEST stage adds only this driver, TestRepeatGuard (tr_common) and the records it writes.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib.util
import json
import os
import signal
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as C  # noqa: E402


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------- session (supp only)
def make_session(base, A):
    import torch

    class TestRepeatSession(base):
        """The frozen RegistrationSession; only the role gate differs (DEV for the gate, TEST_REPEAT for the read)."""

        def __init__(self, family, trainer, guard, role, cores, device, candidate):
            if role not in ("DEV", "TEST_REPEAT"):
                raise PermissionError("TEST_REPEAT stage evaluates DEV (gate) or TEST_REPEAT only")
            self.family, self.trainer, self.device = family, trainer, torch.device(device)
            self.candidate = candidate
            self.perturbations = {}
            for core in cores:
                guard.log_rule(core, {role}, "V10 supplemental frozen perturbation rule")
                self.perturbations[core] = A.Q.perturbations_p2(core)
            self.inputs = A.GuardedInputs(guard, {role}, "V10 supplemental " + role,
                                          self.perturbations, self.device)
            self.core, self.reference = None, None

    return TestRepeatSession


def init_body(source: str, cls: str) -> tuple:
    """(first statement, remaining statements) of cls.__init__ as ast dumps."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return ast.dump(item.body[0]), [ast.dump(s) for s in item.body[1:]], ast.dump(item.args)
    raise RuntimeError(f"{cls}.__init__ not found")


def check_session_replica(registration_py: Path) -> dict:
    frozen_first, frozen_rest, frozen_args = init_body(registration_py.read_text(), "RegistrationSession")
    mine_first, mine_rest, mine_args = init_body(Path(__file__).read_text(), "TestRepeatSession")
    gate = ast.dump(ast.parse('if role not in ("CAL", "DEV"):\n'
                              '    raise PermissionError("Only currently activated CAL/DEV are supported")').body[0])
    C.require(frozen_first == gate, "frozen RegistrationSession no longer starts with its CAL/DEV role gate")
    C.require(frozen_rest == mine_rest and frozen_args == mine_args,
              "TestRepeatSession.__init__ differs from the frozen RegistrationSession.__init__ beyond the role gate")
    return {"registration_py": str(registration_py), "sha256": C.sha(registration_py), "body_statements": len(frozen_rest)}


# ------------------------------------------------------------------------------------------------------ frozen sources
def load_source(u: dict):
    code = u["code"]
    src = Path(code["source_root"])
    if code["kind"] == "bf1":
        os.environ["V10_BF1_EXPECT_FREEZE"] = code["freeze"]["sha256"]
        sys.path.insert(0, str(src / "src"))
        sys.path.insert(0, str(src / "scripts" / "v10_bf1"))
        import_path("common", src / "scripts/v10_bf1/common.py")
        run = import_path("bf1_run", src / "scripts/v10_bf1/run.py")
        frozen = run.freeze()                    # every frozen source file and run input byte-identical, or it raises
        C.require(frozen["source_commit"] == code["source_commit"], "BF1 freeze names another commit")
        C.require(run.A.code_hash() == code["core_code_hash"] and run.A.host_code_hash() == code["host_code_hash"],
                  "accepted BF core / host code identity changed")
        return run, {"kind": "bf1", "freeze_sha256": code["freeze"]["sha256"], "source_commit": frozen["source_commit"]}
    C.require(code["kind"] == "supp", "unknown source kind")
    C.require(C.sha(code["release"]["path"]) == code["release"]["sha256"], "execution release changed")
    rel = C.read(code["release"]["path"])
    sys.path.insert(0, str(src / "src"))
    sys.path.insert(0, str(src / "scripts" / "v10_supplement"))
    run = import_path("supp_run", src / "scripts/v10_supplement/run.py")
    from v10.supplement import contract
    inventory = contract.source_inventory()
    C.require(inventory == rel["source_files"] and contract.digest(inventory) == rel["code_sha256"] == code["code_sha256"],
              "supplement source is not byte-identical to its execution release")
    run.initialize_science()
    replica = check_session_replica(src / "src/v10/supplement/registration.py")
    run.RegistrationSession = make_session(run.RegistrationSession, run.A)
    return run, {"kind": "supp", "release_sha256": code["release"]["sha256"], "code_sha256": rel["code_sha256"],
                 "session_replica": replica}


def environment(A, precision: dict, gpu_type: str) -> dict:
    env = A.environment("cuda")
    actual = C.GPU_TYPES.get(env.get("gpu_name"))
    C.require(actual == gpu_type, f"GPU {env.get('gpu_name')} is not the required {gpu_type}")
    for key in ("torch", "cuda_runtime", "cudnn", "torch_threads"):
        C.require(env[key] == precision["environment"][key], "runtime mismatch: " + key)
    C.require(A.precision_settings(env["precision"]) == precision["settings"], "precision settings differ from the protocol")
    C.require(env["precision"]["env"] == precision["env"], "precision environment variables differ from the protocol")
    env["gpu_type"] = actual
    return env


def verify_unit_files(u: dict) -> Path:
    d = Path(u["unit_dir"])
    C.require(C.sha(d / "COMPLETE.json") == u["complete_sha256"], f"{u['key']}: COMPLETE.json changed")
    C.require(C.read(d / "COMPLETE.json")["identity"] == u["identity"], f"{u['key']}: identity changed")
    C.require(C.sha(d / u["checkpoint"]["path"]) == u["checkpoint"]["sha256"], f"{u['key']}: frozen checkpoint changed")
    C.require(C.sha(d / "DEV.jsonl") == u["dev"]["rows_sha256"], f"{u['key']}: recorded DEV rows changed")
    return d


def evaluate_unit(run, u: dict, role: str, test_guard, out_path: Path, budget) -> tuple:
    """Rebuild the frozen trainer exactly as its final unit did, load the selected checkpoint (identity + sha), evaluate."""
    ident, d = u["identity"], Path(u["unit_dir"])
    data_guard = run.A.DataGuard()
    if u["code"]["kind"] == "bf1":
        run.check_data({"FIT", "DEV"} if role == "DEV" else {"FIT"})
        trainer = run.new_trainer(ident, data_guard, "cuda")
        C.require(json.loads(json.dumps(dataclasses.asdict(trainer.cfg))) == C.read(d / "CONFIG.json"),
                  "resolved BF1 configuration differs")
        meta = run.load_checkpoint(trainer, d / u["checkpoint"]["path"], ident)
    else:
        rel = C.read(u["code"]["release"]["path"])
        lock_path = Path(u["code"]["release"]["path"]).parent / rel["data_lock"]["path"]
        C.require(C.sha(lock_path) == rel["data_lock"]["sha256"] == ident["data_lock_sha256"], "data lock changed")
        recipe = run.schedule(ident["family"])
        roles = {"CAL"} if recipe["kind"] == "training_free" else {"FIT", "CAL"}
        run.check_data(C.read(lock_path), data_guard, roles | ({"DEV"} if role == "DEV" else set()))
        trainer = run.new_trainer(ident["family"], ident["candidate"], ident["seed"], data_guard, "cuda")
        C.require(run.CP.canonical_config(trainer) == C.read(d / "CONFIG.json"), "resolved configuration differs")
        meta = run.CP.load(trainer, d / u["checkpoint"]["path"], ident)
    C.require(meta["sha256"] == u["checkpoint"]["sha256"] and meta["step"] == u["checkpoint"]["step"],
              "loaded checkpoint is not the frozen selected checkpoint")
    guard = data_guard if role == "DEV" else test_guard
    cores = guard.included(role)
    if role == "DEV":
        C.require(cores == u["dev"]["cores"], "DEV32 core list differs from the record")
    if u["code"]["kind"] == "bf1":
        summary = run.evaluate(trainer, guard, role, cores, range(10), out_path, ident, meta, budget)
    else:
        summary = run.evaluate(trainer, guard, role, cores, out_path, ident, meta, budget)
    return summary, {"data_guard": data_guard.reads, "test_guard": getattr(test_guard, "reads", [])}


# ------------------------------------------------------------------------------------------------------------- stages
def pins() -> dict:
    job = C.require_slurm()
    act = os.environ.get("V10_TR_EXPECT_ACTIVATION")
    commit = os.environ.get("V10_TR_EXPECT_COMMIT")
    C.require(act and commit, "activation and commit pins are required at submission")
    rec = C.records(act)
    rec["code"] = C.verify_code(rec["activation"], commit)
    rec["job"] = job
    return rec


def stage_gate(gate_id: str) -> int:
    rec = pins()
    g = C.gate_spec(rec["protocol"], gate_id)
    u = C.unit(rec["manifest"], g["unit"])
    C.require(u["family"] == g["family"] and u["hardware"]["gpu_type"] == g["gpu_type"], "gate unit / GPU type mismatch")
    verify_unit_files(u)
    out = C.BUNDLE / "gate" / gate_id / rec["job"]
    C.require(not out.exists(), "gate output exists; never overwritten")
    out.mkdir(parents=True)
    run, source = load_source(u)
    env = environment(run.A, rec["protocol"]["precision"], g["gpu_type"])
    C.write(out / "ENV.json", env)
    budget = C.Budget(out / "HEARTBEAT.jsonl", gate_id)
    record = {"schema": "v10.test_repeat134.dev_gate.v2", "gate_id": gate_id, "family": g["family"], "gpu_type": g["gpu_type"],
              "unit": u["key"], "job": rec["job"], "activation_sha256": rec["activation_sha256"],
              "code_commit": rec["code"]["commit"], "source": source, "criterion": rec["protocol"]["dev_gate"]["criterion_by_gpu_type"][g["gpu_type"]],
              "dev32_recorded": u["dev"]["DEV32"], "test_read": False, "gpu_name": env["gpu_name"], "host": env["host"]}
    try:
        summary, reads = evaluate_unit(run, u, "DEV", None, out / "DEV.jsonl", budget)
    except RuntimeError as e:
        if "onfiguration" not in str(e):
            raise
        verdict = C.gate_verdict(g["gpu_type"], False, False, False, {}, rebuild_error=str(e))
        C.write(out / "GATE.json", {**record, "status": verdict["status"], "verdict": verdict, "utc": C.utc()})
        print("GATE", gate_id, verdict["status"], verdict["reason"], flush=True)
        return 3
    C.write(out / "READS.json", reads)
    roles = sorted({r["role"] for r in reads["data_guard"]})
    recorded = C.lines(Path(u["unit_dir"]) / "DEV.jsonl")
    new = C.lines(out / "DEV.jsonl")
    cmp = C.compare_rows(recorded, new)
    diag = C.gate_diagnostics(recorded, new, C.read(Path(u["unit_dir"]) / "DEV.identity.json"), C.read(out / "DEV.identity.json"))
    metrics_equal = summary["metrics"] == u["dev"]["DEV32"]
    verdict = C.gate_verdict(g["gpu_type"], cmp["status"] == "PASS", metrics_equal, set(roles) <= {"FIT", "CAL", "DEV"}, diag)
    C.write(out / "GATE.json", {**record, "status": verdict["status"], "verdict": verdict, "rows": cmp, "diagnostics": diag,
                                "dev32_reevaluated": summary["metrics"], "metrics_equal": metrics_equal,
                                "data_roles_read": roles, "utc": C.utc()})
    print("GATE", gate_id, verdict["status"], cmp["n_mismatches"], diag["abs_delta_err_full"]["max"], flush=True)
    return 0 if verdict["status"] in C.PASSING[g["gpu_type"]] else 3


BLOCKING_ERRORS = ("onfiguration", "changed", "differs", "mismatch")   # deterministic identity / config / input failures


def dev_pass(run, u: dict, out: Path, budget, protocol: dict) -> dict:
    """v4: the same-job DEV32 re-evaluation and the in-job check (a), (e), (f) against the recorded DEV32 rows."""
    try:
        summary, reads = evaluate_unit(run, u, "DEV", None, out / "DEV_REEVAL.jsonl", budget)
    except RuntimeError as e:
        if not any(t in str(e) for t in BLOCKING_ERRORS):
            raise                                      # transient: no BLOCKED record, the unit stays resumable
        return {"check": C.in_job_check({}, False, rebuild_error=str(e))}
    d = Path(u["unit_dir"])
    recorded, new = C.lines(d / "DEV.jsonl"), C.lines(out / "DEV_REEVAL.jsonl")
    diag = C.gate_diagnostics(recorded, new, C.read(d / "DEV.identity.json"), C.read(out / "DEV_REEVAL.identity.json"))
    roles = sorted({r["role"] for r in reads["data_guard"]})
    dev24 = set(protocol["sets"]["DEV24_cores"])
    ref = {"check": C.in_job_check(diag, set(roles) <= {"FIT", "CAL", "DEV"}), "DEV32": summary["metrics"],
           "DEV24": C.metrics([r for r in new if r["core"] in dev24]),
           "per_core": {c: C.metrics([r for r in new if r["core"] == c]) for c in u["dev"]["cores"]},
           "rows_sha256": C.sha(out / "DEV_REEVAL.jsonl"), "data_roles_read": roles,
           "recorded": {"DEV32": u["dev"]["DEV32"], "DEV24": u["dev"]["DEV24"], "gpu_name": u["hardware"]["gpu_name"]},
           "vs_recorded": diag}
    C.write(out / "DEV_REEVAL.json", ref)
    return ref


def stage_test(key: str) -> int:
    rec = pins()
    man, prot = rec["manifest"], rec["protocol"]
    u = C.unit(man, key)
    plan = C.test_plan(rec, key)
    verify_unit_files(u)
    gpu_type = plan["gpu_type"]
    C.require(gpu_type == u["hardware"]["gpu_type"], "v4 evaluates each unit on the GPU type of its recorded DEV32")
    lock_record = C.input_lock(rec["activation_sha256"])
    out = C.BUNDLE / "results" / "TEST_REPEAT" / key
    with C.lock(out / "RUN.lock"):
        if (out / "COMPLETE.json").exists():
            C.require(C.read(out / "COMPLETE.json")["status"] == "COMPLETE", "existing TEST record is not COMPLETE")
            print("ALREADY_COMPLETE", key, flush=True)
            return 0
        C.require(not (out / "BLOCKED.json").exists(), "this unit is BLOCKED by the in-job check; TEST is never read for it")
        run, source = load_source(u)
        env = environment(run.A, prot["precision"], gpu_type)
        C.write(out / f"ENV_{rec['job']}.json", env)
        budget = C.Budget(out / "HEARTBEAT.jsonl", key)
        ref = dev_pass(run, u, out, budget, prot)
        if ref["check"]["status"] != "PASS":
            C.write(out / "BLOCKED.json", {"status": "BLOCKED", "unit": key, "job": rec["job"], "utc": C.utc(),
                                           "check": ref["check"], "test_read": False,
                                           "diagnostics": ref.get("vs_recorded")})
            print("BLOCKED", key, ref["check"]["reason"], flush=True)
            return 3
        guard = C.TestRepeatGuard(man, lock_record, rec["activation_sha256"])
        try:
            summary, reads = evaluate_unit(run, u, C.ROLE, guard, out / "TEST_REPEAT.jsonl", budget)
        except BaseException:
            C.write(out / f"READS_INCOMPLETE_{rec['job']}.json", {"test_guard": guard.reads})
            raise
        C.write(out / f"READS_{rec['job']}.json", reads)
        C.require({r["role"] for r in reads["data_guard"]} <= {"FIT", "CAL"} and {r["role"] for r in reads["test_guard"]} == {C.ROLE},
                  "TEST job read data outside FIT / CAL (model rebuild) and TEST_REPEAT")
        rows = C.lines(out / "TEST_REPEAT.jsonl")
        C.require(len(rows) == 1340 and {(r["core"], r["perturbation"]) for r in rows}
                  == {(c, i) for c in man["test"]["cores"] for i in range(10)}, "incomplete TEST_REPEAT block")
        C.require(all(r["role"] == C.ROLE and r["checkpoint_sha256"] == u["checkpoint"]["sha256"] for r in rows),
                  "mixed role or checkpoint in TEST rows")
        retained = set(man["test"]["sensitivity112"]["cores"])
        jobs = sorted(p.stem[len("ENV_"):] for p in out.glob("ENV_*.json"))
        result = {
            "schema": "v10.test_repeat134.unit.v4", "status": "COMPLETE", "scope": "evaluator_only",
            "label": prot["reporting"]["label"], "unit": key, "family": u["family"], "seed": u["seed"], "tier": u["tier"],
            "row_label": u["row_label"], "identity": u["identity"], "checkpoint": u["checkpoint"],
            "TEST134": C.metrics(rows), "TEST112": C.metrics([r for r in rows if r["core"] in retained]),
            "per_core": {c: C.metrics([r for r in rows if r["core"] == c]) for c in man["test"]["cores"]},
            "summary_metrics": summary["metrics"],
            "dev_reevaluation": {k: v for k, v in ref.items() if k != "per_core"},
            "dev_reevaluation_per_core": ref["per_core"], "jobs": jobs, "single_job": len(jobs) == 1,
            "input_lock_sha256": C.sha(C.BUNDLE / "INPUT_LOCK.json"), "activation_sha256": rec["activation_sha256"],
            "code_commit": rec["code"]["commit"], "source": source, "job": rec["job"], "utc": C.utc(),
            "hardware": {"gpu_name": env["gpu_name"], "gpu_type": gpu_type, "host": env["host"],
                         "partition": env["slurm_partition"], "recorded_dev32_gpu": u["hardware"]["gpu_name"]},
            "artifacts": {p.name: C.sha(p) for p in sorted(out.iterdir())
                          if p.is_file() and p.name not in ("RUN.lock", "COMPLETE.json", "HEARTBEAT.jsonl")}}
        C.require(result["TEST134"] == summary["metrics"], "summary metrics differ from the recomputation")
        C.write(out / "COMPLETE.json", result)
    print("TEST_COMPLETE", key, result["TEST134"], flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("gate", "test"))
    ap.add_argument("target")
    a = ap.parse_args(argv)

    def terminated(signum, frame):
        raise InterruptedError(f"Slurm signal {signum}; completed rows are kept and the unit resumes")
    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGUSR1, terminated)
    try:
        return stage_gate(a.target) if a.stage == "gate" else stage_test(a.target)
    except BaseException as e:
        C.write(C.BUNDLE / "failures" / f"{os.environ.get('SLURM_JOB_ID', 'local')}.json",
                {"utc": C.utc(), "stage": a.stage, "target": a.target, "exception": repr(e),
                 "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
