"""Truth isolation (decision 9.2): training and CAL code cannot read evaluator-only material - enforced at run time by the audit-hook
barrier, checked statically on the training / CAL modules, and by the refusal of the evaluator to run inside the barrier."""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import t03_common as C
import t03_data as D
import t03_evaluate as E
import t03_train as TR

TRAINING_SIDE = ("t03_data.py", "t03_objectives.py", "t03_models.py", "t03_train.py", "t03_select.py", "t03_units.py")
FORBIDDEN_TOKENS = ("load_evaluation_truth", "exact_truth", "population_j", "truth.npz", "truth.json", "TRUTH_DIR_NAME", "evaluator_only",
                    "t03_evaluate", "eval/samples")


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
    hits = sorted({t for t in _tokens(path) for f in FORBIDDEN_TOKENS if f in t})
    assert not hits, f"{name} references evaluator-only material: {hits}"


def _calls(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            f = node.func
            out.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return out


def test_only_the_evaluator_reads_truth():
    readers = [p.name for p in sorted(C.HERE.glob("t03_*.py")) if {"load_evaluation_truth", "exact_truth"} & _calls(p)]
    assert readers == ["t03_evaluate.py"]
