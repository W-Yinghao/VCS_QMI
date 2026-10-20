"""EPFL data gate (src/v10/epfl/guard.py): admission only under a pinned activation record, stage roles, refusals.  Standard library only;
reads the pinned V10 / REG-P3 manifests and (one test) the 10 CAL cache files' bytes for the data-lock check.  No image is decoded."""
import copy
import csv
import hashlib
import io
import json
import shutil
from pathlib import Path

import pytest

from v10.epfl import guard as GD

REC = GD.ACTIVATION_RECORD


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def guard():
    lock = json.loads((GD.REPO / "v10" / "epfl" / "EPFL_DATA_LOCK.json").read_text())
    return GD.EPFLDataGuard(record=REC, expect_record_sha256=sha(REC), data_lock=lock)


def write_record(tmp_path, **changes):
    rec = json.loads(REC.read_text()); rec.update(changes)
    p = tmp_path / "record.json"; p.write_text(json.dumps(rec)); return p


def test_record_must_be_pinned(tmp_path):
    with pytest.raises(PermissionError):
        GD.EPFLDataGuard(record=REC, expect_record_sha256=None)
    with pytest.raises(PermissionError):
        GD.EPFLDataGuard(record=REC, expect_record_sha256="0" * 64)
    p = write_record(tmp_path, purpose="edited after pinning")
    with pytest.raises(PermissionError):
        GD.EPFLDataGuard(record=p, expect_record_sha256=sha(REC))


def test_committed_record_is_gate_checks_fit_cal(guard):
    assert guard.stage == "GATE_CHECKS" and guard.admitted == {"FIT", "CAL"}
    assert len(guard.included("FIT")) == 336 and len(guard.included("CAL")) == 10 and len(guard.included("DEV")) == 74
    assert len(guard.held()) == 57


def test_fit_cal_admitted_dev_refused(guard):
    fit, cal, dev = guard.included("FIT")[0], guard.included("CAL")[0], guard.included("DEV")[0]
    assert guard.check(fit, {"FIT"}, "test") == GD.P3_ROOT / "cache" / f"{fit}.npz"
    assert guard.check(cal, {"CAL"}, "test").name == f"{cal}.npz"
    for roles in ({"DEV"}, {"FIT", "DEV"}, {"CAL", "DEV"}):
        with pytest.raises(PermissionError):
            guard.check(dev, roles, "test")
    with pytest.raises(PermissionError):            # a DEV window is not readable as FIT / CAL either
        guard.check(dev, {"FIT", "CAL"}, "test")


def test_excluded_pairs_bf_records_and_unknown_ids_refused(guard):
    for w in ("water_0083", "country_0015", "country_0007", "urban_0086"):     # 2 alignment, 2 size exclusions
        with pytest.raises(PermissionError):
            guard.check(w, {"FIT", "CAL"}, "test")
    for bf in ("1B_A2", "1B_A1", "2B_B2"):           # BF FIT / BF held TEST cores: never through the EPFL gate
        with pytest.raises(PermissionError):
            guard.check(bf, {"FIT"}, "test")
    for bad in ("country_9999", "nocategory", "", "country/0000"):
        with pytest.raises(PermissionError):
            guard.check(bad, {"FIT"}, "test")


def test_role_mismatch_and_unknown_roles(guard):
    cal = guard.included("CAL")[0]
    with pytest.raises(PermissionError):
        guard.check(cal, {"FIT"}, "test")
    for roles in (set(), {"TEST"}, {"FIT", "TEST"}, {"CONFIRM"}):
        with pytest.raises(PermissionError):
            guard.check(guard.included("FIT")[0], roles, "test")


def test_reads_are_logged_and_rules_have_no_path(guard):
    n = len(guard.reads); w = guard.included("CAL")[1]
    guard.check(w, {"CAL"}, "logged read"); guard.log_rule(w, {"CAL"}, "logged rule")
    a, b = guard.reads[n], guard.reads[n + 1]
    assert a["window"] == w and a["record_id"] == GD.record_id(w) and a["role"] == "CAL" and a["kind"] == "read" and a["path"]
    assert b["kind"] == "rule" and b["path"] is None and b["stage"] == "GATE_CHECKS"


def test_unit_and_id_mapping(guard):
    assert guard.unit_of("country_0025") == "country/0001+country/0025+country/0032"
    assert GD.window_id("street/0021") == "street_0021" and GD.record_id("oldbuilding_0006") == "oldbuilding/0006"


def test_record_validation_rules():
    base = json.loads(REC.read_text())
    GD.validate_record(base)
    bad = []
    r = copy.deepcopy(base); r["schema"] = "x"; bad.append(r)
    r = copy.deepcopy(base); r["dataset"] = "BF_SHG"; bad.append(r)
    r = copy.deepcopy(base); r["admitted_roles"] = ["FIT", "CAL", "DEV"]; bad.append(r)
    r = copy.deepcopy(base); r["stage"] = "OPEN"; bad.append(r)
    r = copy.deepcopy(base); r["manifest_sha256"] = "0" * 64; bad.append(r)
    r = copy.deepcopy(base); r["gates"]["item2"]["status"] = "FAIL"; bad.append(r)
    r = copy.deepcopy(base); r["gates"]["item4"]["status"] = "FAIL"; bad.append(r)
    r = copy.deepcopy(base); r["stage"] = "ACTIVATED"; r["admitted_roles"] = ["FIT", "CAL", "DEV"]; bad.append(r)   # pending item 4, no item 5
    for r in bad:
        with pytest.raises(PermissionError):
            GD.validate_record(r)


def activated(base):
    g = GD.GROUP_RECORD_V2
    r = copy.deepcopy(base); r.update({"stage": "ACTIVATED", "admitted_roles": ["FIT", "CAL", "DEV"], "owner_decision": "fixture",
                                       "frozen_epfl_protocol_sha256": "f" * 64, "frozen_group_record_sha256": sha(g),
                                       "frozen_group_record": {"path": str(g), "sha256": sha(g)}})
    r["gates"]["item4"]["status"] = "PASS"; r["gates"]["item5"] = {"status": "PASS"}; r["gates"]["item5_timing"] = {"status": "MEASURED"}
    return r


def test_activated_stage_admits_dev_only_with_every_condition(tmp_path):
    base = json.loads(REC.read_text()); r = activated(base); GD.validate_record(r)
    for key in ("owner_decision", "frozen_epfl_protocol_sha256", "frozen_group_record_sha256", "frozen_group_record"):
        x = copy.deepcopy(r); x[key] = None
        with pytest.raises(PermissionError):
            GD.validate_record(x)
    x = copy.deepcopy(r); x["gates"]["item5"]["status"] = "FAIL"
    with pytest.raises(PermissionError):
        GD.validate_record(x)
    p = tmp_path / "activated.json"; p.write_text(json.dumps(r))
    g = GD.EPFLDataGuard(record=p, expect_record_sha256=sha(p))
    dev = g.included("DEV")[0]
    assert g.check(dev, {"DEV"}, "fixture").name == f"{dev}.npz"
    with pytest.raises(PermissionError):
        g.check("water_0083", {"DEV"}, "fixture")      # excluded pairs stay refused at every stage


def test_forbidden_path_refused_on_a_fixture_manifest(tmp_path):
    raw = GD.V10_MANIFEST.read_text(); rows = list(csv.DictReader(io.StringIO(raw)))
    for r in rows:
        if r["record_id"] == "country/0000":
            r["fixed_path"] = "epfl_rgb_nir/closed/country/0000_rgb.tiff"
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    m = tmp_path / "manifest.csv"; m.write_text(buf.getvalue())
    g = GD.EPFLDataGuard(record=REC, expect_record_sha256=sha(REC), manifest=m, expect_manifest_sha256=sha(m))
    with pytest.raises(PermissionError):
        g.check("country_0000", {"FIT"}, "forbidden path fixture")
    with pytest.raises(RuntimeError):                    # the default pin refuses the fixture manifest
        GD.EPFLDataGuard(record=REC, expect_record_sha256=sha(REC), manifest=m)


def test_data_lock_cal_files(guard):
    assert guard.verify_data_lock({"CAL"}) == {"checked": 10, "roles": ["CAL"]}
    with pytest.raises(PermissionError):
        guard.verify_data_lock({"DEV"})


def test_lock_and_record_agree():
    rec = json.loads(REC.read_text()); lock = GD.REPO / "v10" / "epfl" / "EPFL_DATA_LOCK.json"
    assert rec["data_lock_sha256"] == sha(lock)
    windows = json.loads(lock.read_text())["windows"]
    assert len(windows) == 420 and sum(w["role"] == "FIT" for w in windows) == 336 and sum(w["role"] == "DEV" for w in windows) == 74
