"""TEST_REPEAT134 v6: the CoMIR source check with two trainers in one process (CPU; FIT data only, no TEST file).

    cd ssl_pilot_v10_test_repeat && OMP_NUM_THREADS=1 /home/infres/yinwang/CS_QMI/env/bin/python -m pytest -q tests/v10_test_repeat
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("V10_TR_BUNDLE", tempfile.mkdtemp(prefix="tr134_test_bundle_"))   # never the real bundle
sys.path.insert(0, str(REPO / "scripts/v10_test_repeat"))
import tr_common as C  # noqa: E402
import build_freeze as B  # noqa: E402

PY = "/home/infres/yinwang/CS_QMI/env/bin/python"
REAL_BUNDLE = C.HAND / "v10_test_repeat134_20261020"


def test_two_comir_trainers_in_one_process():
    """The frozen check refuses the second CoMIR trainer (the v5 failure); with comir_check_v6 it is built, while a namespace
    package outside the pinned root, one without __path__ and a regular module outside the root are still refused."""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env.pop("V10_TR_BUNDLE", None)
    r = subprocess.run([PY, str(Path(__file__).with_name("comir_two_trainers.py")), str(REPO)], env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads([x for x in r.stdout.splitlines() if x.startswith("RESULT ")][-1][len("RESULT "):])
    assert out["first_trainer"] == "built"
    assert out["second_trainer_v5"] == B.COMIR_ERROR                                 # v5: the second trainer is refused
    assert out["models_modules"]["models.layers"] is True                            # the namespace subpackage
    assert out["second_trainer_v6"] == "built"                                       # v6: built
    for case in ("namespace_outside_root", "namespace_without_path", "regular_outside_root"):
        assert out[case] == [B.COMIR_ERROR, True]                                    # still refused; sys.modules restored
    assert out["sys_modules_restored_after_pass"] is True


def test_v6_records():
    if not (REAL_BUNDLE / C.ACTIVATION).exists():
        pytest.skip("v6 not frozen yet")
    act, dec = C.read(REAL_BUNDLE / C.ACTIVATION), C.read(REAL_BUNDLE / C.DECISION_V6)
    assert dec["chosen_option_verbatim"] == "修复后补跑这 3 个单元 (Recommended)" and dec == B.decision_v6()
    assert act["v6"]["units"] == list(B.V6_UNITS) and act["v6"]["input_lock_activation_sha256"] == B.V5_ACTIVATION_SHA
    assert act["supersedes"] == {"activation_sha256": B.V5_ACTIVATION_SHA, "commit": B.V5_COMMIT,
                                 "for": "the 3 CoMIR_native units only; the inputs and the other 39 units stay under v5"}
    am = C.read(REAL_BUNDLE / C.AMENDMENT_V6)
    assert all(a["test_guard_reads"] == 0 and a["dev_reevaluation_vs_recorded"]["n_mismatches"] == 0
               for a in am["failure"]["attempts"].values())
