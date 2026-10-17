"""Truth isolation (decision 9.2): training and CAL code cannot read evaluator-only material - enforced at run time by the audit-hook
barrier (Python-level file APIs, process creation) plus the wrapped torch C-level readers, checked statically on the training / CAL
modules (no C-level reader that bypasses the audit hook), by the refusal of the evaluator and the package checker to run inside the
barrier, and by the runner failing any unit whose barrier block recorded a denial."""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

import t03_common as C
import t03_data as D
import t03_evaluate as E
import t03_run as R
import t03_train as TR

TRAINING_SIDE = ("t03_data.py", "t03_objectives.py", "t03_models.py", "t03_train.py", "t03_select.py", "t03_units.py")
FORBIDDEN_TOKENS = ("load_evaluation_truth", "exact_truth", "population_j", "truth.npz", "truth.json", "TRUTH_DIR_NAME", "evaluator_only",
                    "t03_evaluate", "eval/samples", "check_synthetic", "run_checks")
# readers that bypass Python's audit events (the barrier cannot see them at run time): never referenced by a training-side module
C_LEVEL_READERS = ("from_file", "ctypes", "cffi", "h5py", "pyarrow", "mmap", "jit", "PyTorchFileReader", "import_ir_module",
                   "_load_jit_module_from_file", "_load_mobile_module_from_file", "CDLL", "dlopen")


def _case(suite: Path) -> Path:
    return suite / C.case_id("two_block", 8, 0.25)


def test_barrier_denies_truth_and_eval_allows_fit_cal(tiny_suite):
    suite, record = tiny_suite
    case = _case(suite)
    truth = case / "evaluator_only" / "truth.npz"
    assert truth.exists()
    with C.truth_barrier("test"):
        for fn in (lambda: D.GEN.load_evaluation_truth(case), lambda: np.load(truth), lambda: open(case / "evaluator_only" / "truth.json"),
                   lambda: truth.read_bytes(), lambda: os.open(truth, os.O_RDONLY), lambda: D.GEN.load_samples(case, "eval"),
                   lambda: shutil.copyfile(truth, case / "fit" / "copy.npz"), lambda: os.link(truth, case / "fit" / "hard.npz"),
                   lambda: subprocess.run(["cat", str(truth)], check=False)):
            with pytest.raises(C.TruthAccessError):
                fn()
        assert set(D.GEN.load_samples(case, "fit")) == {"x", "y", "base_id"}
        assert D.load_role(case, "cal", 8).n == 48
    assert not (case / "fit" / "copy.npz").exists() and not (case / "fit" / "hard.npz").exists()
    assert set(D.GEN.load_evaluation_truth(case)) >= {"p", "q", "eta", "S"}   # outside the barrier the evaluator may read it
    assert any(d["reason"] == "evaluator_only" for d in C.barrier_denials())


def test_generator_truth_apis_refuse_inside_the_barrier(tiny_suite):
    suite, _ = tiny_suite
    with C.truth_barrier("test"):
        with pytest.raises(C.TruthAccessError):
            D.GEN.exact_truth("two_block", 8, 0.25)                         # analytic truth, no file involved
    assert D.GEN.exact_truth("two_block", 8, 0.25)["S"] == pytest.approx(0.25 ** 2 / (4 - 0.25 ** 2))
    assert D.GEN.load_evaluation_truth.v10_t03_guarded and D.GEN.exact_truth.v10_t03_guarded


def test_barrier_follows_symlinks(tiny_suite, tmp_path):
    suite, _ = tiny_suite
    link = tmp_path / "innocent.npz"
    os.symlink(_case(suite) / "evaluator_only" / "truth.npz", link)
    with C.truth_barrier("test"):
        with pytest.raises(C.TruthAccessError):
            np.load(link)


def test_barrier_nesting_keeps_the_strictest_rule(tiny_suite):
    suite, _ = tiny_suite
    case = _case(suite)
    with C.truth_barrier("outer"):
        with C.truth_barrier("inner", allow_eval=True):
            with pytest.raises(C.TruthAccessError):
                D.GEN.load_samples(case, "eval")
        assert C.barrier_active()
    assert not C.barrier_active()
    assert set(D.GEN.load_samples(case, "eval")) == {"x", "y", "base_id"}


def test_train_refuses_outside_and_evaluator_refuses_inside(tiny_suite):
    suite, record = tiny_suite
    cid = C.case_id("two_block", 8, 0.25)
    smp = D.load_case_samples(record, suite, cid)
    cfg = TR.UnitConfig(unit_id="X", block="smoke", method="MF_VCS_M0", family="two_block", m=8, rho=0.25, K=4, lr=1e-3, cap=None,
                        data_root_seed=7, algo_seed=1, steps=2, ckpt_every=1)
    with pytest.raises(RuntimeError, match="truth_barrier"):
        TR.train_unit(cfg, smp["fit"], smp["cal"])
    with C.truth_barrier("test"):
        with pytest.raises(C.TruthAccessError):
            E.load_truth(suite / cid)
        with pytest.raises(C.TruthAccessError):
            E.evaluate_unit({}, suite, suite, record)


def _tokens(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                yield a.name
            if isinstance(node, ast.ImportFrom) and node.module:
                yield node.module


@pytest.mark.parametrize("name", TRAINING_SIDE)
def test_training_side_modules_never_reference_truth(name):
    path = C.HERE / name
    toks = set(_tokens(path))
    hits = sorted({t for t in toks for f in FORBIDDEN_TOKENS if f in t})
    assert not hits, f"{name} references evaluator-only material: {hits}"
    dotted = {part for t in toks for part in re.split(r"[.\s]", t)}
    c_level = sorted(set(C_LEVEL_READERS) & dotted)
    assert not c_level, f"{name} references a reader the truth barrier cannot see at run time: {c_level}"


def _calls(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            f = node.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


def test_only_the_evaluator_and_the_checker_wrapper_read_truth():
    """Direct truth readers: the evaluator (generator truth APIs) and t03_run.run_package_checker (the package checker's run_checks,
    which reads every truth file of a suite); both refuse to run inside the barrier (tests below / test_t03_runner)."""
    readers = [p.name for p in sorted(C.HERE.glob("t03_*.py")) if {"load_evaluation_truth", "exact_truth"} & _calls(p)]
    assert readers == ["t03_evaluate.py"]
    checker_callers = [p.name for p in sorted(C.HERE.glob("t03_*.py")) if "run_checks" in _calls(p)]
    assert checker_callers == ["t03_run.py"]
    src = (C.HERE / "t03_run.py").read_text(encoding="utf-8")
    fn = src[src.index("def run_package_checker"):src.index("def ensure_checked_suite")]
    assert fn.index("C.barrier_active()") < fn.index("run_checks(") and "full_check_path(out_json)" in fn


def test_package_checker_refuses_inside_the_barrier(tiny_suite, out_root):
    suite, _ = tiny_suite
    with C.truth_barrier("test"):
        with pytest.raises(C.TruthAccessError):
            R.run_package_checker(suite, out_root / "checks" / "x.json")
    assert not (out_root / "checks" / "x.json").exists()


def test_barrier_wraps_the_torch_c_level_readers(tiny_suite):
    """torch.from_file / UntypedStorage.from_file (and the TypedStorage classes through it) / torch.ops.aten.from_file raise no audit
    event: they are wrapped on barrier entry.  Allowed paths (FIT samples) still read."""
    suite, _ = tiny_suite
    case = _case(suite)
    truth = case / "evaluator_only" / "truth.npz"
    fit = case / "fit" / "samples.npz"
    attempts = {"torch.from_file": lambda f: torch.from_file(str(f), size=4, dtype=torch.uint8),
                "UntypedStorage.from_file": lambda f: torch.UntypedStorage.from_file(str(f), False, 4),
                "ByteStorage.from_file": lambda f: torch.ByteStorage.from_file(str(f), False, 4),
                "aten.from_file": lambda f: torch.ops.aten.from_file(str(f), False, 4, dtype=torch.uint8),
                "aten.from_file.default": lambda f: torch.ops.aten.from_file.default(str(f), False, 4, dtype=torch.uint8),
                "from_file(filename=...)": lambda f: torch.from_file(filename=str(f), size=4, dtype=torch.uint8)}
    n0 = C.denial_count()
    with C.truth_barrier("test"):
        for name, fn in attempts.items():
            with pytest.raises(C.TruthAccessError):
                fn(truth)
        assert bytes(torch.from_file(str(fit), size=4, dtype=torch.uint8).tolist()) == b"PK\x03\x04"     # allowed path
        assert len(torch.UntypedStorage.from_file(str(fit), False, 4)) == 4
    assert C.denial_count() - n0 == len(attempts)
    assert {d["event"] for d in C.barrier_denials()[n0:]} >= {"torch.from_file", "torch.UntypedStorage.from_file", "torch.ops.aten.from_file",
                                                               "torch.ops.aten.from_file.default"}
    assert bytes(torch.from_file(str(truth), size=4, dtype=torch.uint8).tolist()) == b"PK\x03\x04"       # outside: unchanged behaviour
    assert all(getattr(f, "v10_t03_guarded", False) for f in (torch.from_file, torch.UntypedStorage.from_file, torch.ops.aten.from_file))


def test_barrier_refuses_tampering_with_truth(tiny_suite, tmp_path):
    suite, _ = tiny_suite
    case = _case(suite)
    truth = case / "evaluator_only" / "truth.npz"
    before = C.sha256_file(truth)
    with C.truth_barrier("test"):
        for fn in (lambda: os.remove(truth), lambda: os.chmod(truth, 0o600), lambda: os.utime(truth, (0, 0)),
                   lambda: os.truncate(truth, 0), lambda: shutil.rmtree(case / "evaluator_only"), lambda: os.rename(truth, tmp_path / "t.npz"),
                   lambda: sqlite3.connect(str(case / "evaluator_only" / "x.db"))):
            with pytest.raises(C.TruthAccessError):
                fn()
    assert C.sha256_file(truth) == before and not (case / "evaluator_only" / "x.db").exists()


def test_a_swallowed_denial_fails_the_unit(tiny_suite, out_root, monkeypatch):
    """Training code that catches the refusal (TruthAccessError is a PermissionError) still leaves a recorded denial: the unit is FAIL
    at stage truth_barrier, it is not evaluated, and the failure record lists the denial."""
    suite, record = tiny_suite
    truth = _case(suite) / "evaluator_only" / "truth.npz"
    orig = TR.train_unit

    def sneaky(*a, **k):
        try:
            np.load(truth)
        except PermissionError:
            pass
        return orig(*a, **k)
    monkeypatch.setattr(TR, "train_unit", sneaky)
    cfg = TR.UnitConfig(unit_id="SNEAKY", block="smoke", method="MF_VCS_M0", family="two_block", m=8, rho=0.25, K=4, lr=1e-3, cap=None,
                        data_root_seed=7, algo_seed=1, steps=2, ckpt_every=1)
    res = R.execute_unit(cfg, suite=suite, record=record, out=R.unit_dir("SNEAKY"), ident={"t03_code_hash": "test"}, env={}, extra={})
    assert res["status"] == "FAIL" and res["failure"]["stage"] == "truth_barrier"
    assert [d["path"] for d in res["truth_barrier"]["denials_during_unit"]] == [str(truth)]
    assert not (R.unit_dir("SNEAKY") / "evaluator_only").exists()
    assert json.loads((R.unit_dir("SNEAKY") / "result.json").read_text())["status"] == "FAIL"
