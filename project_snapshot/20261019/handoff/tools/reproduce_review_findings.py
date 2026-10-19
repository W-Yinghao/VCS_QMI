#!/usr/bin/env python3
"""Reproduce orchestration defects with AST-extracted functions and inert stubs.

No training-module imports, result changes, cluster calls, or torch execution.
This is a narrow control-flow demonstration, not an integration test.
"""
import ast
import contextlib
import collections
import hashlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ssl_pilot_v10_t03/scripts/v10_t03/t03_run.py"
source = SOURCE.read_text()
tree = ast.parse(source)


def extract(name, namespace, path=SOURCE):
    parsed = tree if path == SOURCE else ast.parse(path.read_text())
    node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name], node.lineno


class Blocked(RuntimeError):
    pass


saved = {"status": "COMPLETE", "config": {"dtype": "float32"},
         "provenance": {"identity": {"t03_code_hash": "old-code"}}}
namespace = {"Path": Path, "U": SimpleNamespace(parse=lambda r: r),
             "unit_dir": lambda *args: Path("/unused"), "read_result": lambda *args: saved}
run_unit, line = extract("run_unit", namespace)
with contextlib.redirect_stdout(io.StringIO()):
    returned = run_unit({"unit_id": "SAME_ID"}, ident={"t03_code_hash": "new-code"}, env={}, dtype="float64")
assert returned == "COMPLETE"
findings = [{"id": "CR-01", "line": line, "reproduced": True,
             "trigger": "Existing COMPLETE result with old code and float32; request new code and float64 using same unit ID.",
             "actual": returned, "expected": "Refuse reuse or validate matching identity/configuration and artifacts."}]

captured = {}


def reject(*args):
    raise Blocked("dependency missing")


def capture(path, value):
    captured.update(value)
    return "inert-write-stub"


row = {"unit_id": "CAL_1", "block": "CAL", "family": "two_block", "m": 8, "rho": 0, "method": "MF_VCS_M0"}
common = SimpleNamespace(identity=lambda: {}, truth_barrier=lambda _: contextlib.nullcontext(), denial_count=lambda: 0,
                         case_id=lambda *args: "case", utc_now=lambda: "stub", out_root=lambda: Path("/unused"),
                         write_json_atomic=capture, TruthAccessError=PermissionError)
namespace = {"U": SimpleNamespace(load_rows=lambda: [row], parse=lambda r: r), "C": common,
             "Blocked": Blocked, "_dep_results": reject, "json": json}
cmd_select, line = extract("cmd_select", namespace)
with contextlib.redirect_stdout(io.StringIO()):
    returned = cmd_select(None)
assert returned == 0 and captured["n_selected"] == 0 and captured["n_groups"] == 1
findings.append({"id": "CR-03", "line": line, "reproduced": True,
                 "trigger": "All CAL groups are BLOCKED by missing dependencies.",
                 "actual": {"exit_code": returned, "n_selected": captured["n_selected"], "n_groups": captured["n_groups"]},
                 "expected": "Nonzero exit code when any required group is not SELECTED."})
AGG = ROOT / "ssl_pilot_v10_t03/v10/tools/t03_aggregate.py"
with tempfile.TemporaryDirectory(prefix="cs_qmi_review_") as tmp:
    p = Path(tmp) / "runs/U1"
    (p / "evaluator_only").mkdir(parents=True)
    (p / "result.json").write_text(json.dumps({"status": "COMPLETE", "evaluation": {"sha256": "wrong-digest"}}))
    metrics = {k: 0.0 for k in ("S", "EVAL_UJ_minus_S", "pop_mse", "EVAL_UJ_minus_J_pop", "EVAL_U_minus_V")}
    metrics.update(identity_check="PASS", perm_conditional_variance={"fit": {"perm1_var": 0.0}})
    (p / "evaluator_only/evaluation.json").write_text(json.dumps({"unit_id": "DIFFERENT_UNIT", "status": "FAIL", "selected_metrics": metrics}))
    ns = {"Path": Path, "json": json, "collections": collections,
          "U": SimpleNamespace(load_rows=lambda: [{"unit_id": "U1", "data_root_seed": "1"}]),
          "C": SimpleNamespace(TRUTH_DIR_NAME="evaluator_only"), "_sel": lambda e: e["selected_metrics"],
          "_stats": lambda values, *args: {"values": values}}
    loader, line = extract("load_all", ns, AGG)
    cell, _ = extract("_cell", ns, AGG)
    loaded = loader(Path(tmp))
    counted = cell(list(loaded.values()))
    assert counted["n_complete"] == 1
    findings.append({"id": "CR-02", "source": str(AGG.relative_to(ROOT)), "source_sha256": hashlib.sha256(AGG.read_bytes()).hexdigest(),
                     "line": line, "reproduced": True,
                     "trigger": "COMPLETE result with wrong evaluation digest, evaluator status FAIL, and a different evaluator unit ID.",
                     "actual": {"n_complete": counted["n_complete"]},
                     "expected": "Refuse aggregation until evaluator digest, status and unit identity match."})
report = {"schema": "cs_qmi.handoff.control_flow_reproduction.v1", "scope": "AST-isolated functions with inert stubs; not a training rerun",
          "source": str(SOURCE.relative_to(ROOT)), "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "findings": findings}
dest = ROOT / "handoff/evidence/REVIEW_REPRODUCTIONS.json"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"reproduced": [x["id"] for x in findings]}))
