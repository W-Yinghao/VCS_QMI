"""V10 T03 unit tests (login node, CPU, seconds): fixtures for tiny synthetic suites under pytest's tmp_path.

Never touches outputs_v10, BF-SHG data, TEST or EPFL: every test sets V10_T03_OUT_ROOT to a temporary directory.
  /home/infres/yinwang/CS_QMI/env/bin/python -m pytest -q tests/v10_t03
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "scripts" / "v10_t03", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
torch.set_num_threads(1)

TINY = {"fit": 96, "cal": 48, "eval": 160}


@pytest.fixture()
def out_root(tmp_path, monkeypatch):
    root = tmp_path / "T03"
    monkeypatch.setenv("V10_T03_OUT_ROOT", str(root))
    return root


@pytest.fixture()
def tiny_sizes(monkeypatch):
    import t03_common as C
    monkeypatch.setattr(C, "ROLE_SIZES", dict(TINY))
    return dict(TINY)


@pytest.fixture()
def tiny_suite(out_root, tiny_sizes):
    import t03_data as D
    record = D.ensure_suite(7, out_root)
    return D.suite_dir(7, out_root), record
