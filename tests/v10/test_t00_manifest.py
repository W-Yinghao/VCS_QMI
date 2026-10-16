"""V10 T00: invariants of the committed data manifest (v10/T00) and the builder's metadata-only guard.

Reads only the files in v10/T00/ and the package template; never touches data, archives or caches.
  /home/infres/yinwang/CS_QMI/env/bin/python -m pytest -q tests/v10/test_t00_manifest.py
"""
from __future__ import annotations

import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
T00 = REPO / "v10" / "T00"
TEMPLATE = REPO / "v10" / "package_v10_server_v1_20261016" / "configs" / "DATA_MANIFEST_TEMPLATE.csv"
BUILDER = REPO / "scripts" / "v10" / "t00_build_manifest.py"


def rows():
    with open(T00 / "V10_DATA_MANIFEST.csv", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def summary():
    return json.loads((T00 / "MANIFEST_BUILD_SUMMARY.json").read_text(encoding="utf-8"))


def builder():
    spec = importlib.util.spec_from_file_location("t00_build_manifest", BUILDER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_columns_equal_package_template():
    with open(TEMPLATE, newline="", encoding="utf-8-sig") as f:
        template = next(csv.reader(f))
    cols, _ = rows()
    assert cols == template


def test_sha256_file_matches():
    for line in (T00 / "MANIFEST_SHA256.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((T00 / name).read_bytes()).hexdigest() == digest, name


def test_counts_and_roles():
    _, rs = rows()
    c = collections.Counter((r["dataset"], r["inclusion"], r["role"] or "-", r["original_split"]) for r in rs)
    assert c[("BF_SHG", "include", "FIT", "train")] == 32 and c[("BF_SHG", "include", "CAL", "train")] == 8
    assert c[("BF_SHG", "include", "DEV", "validation1")] == 25 and c[("BF_SHG", "include", "DEV", "validation2")] == 7
    assert c[("BF_SHG", "exclude", "DEV", "test")] == 134
    assert sum(n for (d, *_), n in c.items() if d == "BF_SHG") == 206
    assert sum(n for (d, *_), n in c.items() if d == "EPFL_RGB_NIR") == 477
    epfl_inc = collections.Counter(r["role"] for r in rs if r["dataset"] == "EPFL_RGB_NIR" and r["inclusion"] == "include")
    assert epfl_inc == {"FIT": 336, "CAL": 10, "DEV": 74}
    assert sum(1 for r in rs if r["dataset"] == "EPFL_RGB_NIR" and r["inclusion"] == "exclude") == 57
    groups = {(r["dataset"], r["group_id"]) for r in rs if r["dataset"] == "EPFL_RGB_NIR" and r["inclusion"] == "include"}
    assert len(groups) == 411
    assert summary()["manifest"]["rows_by_dataset_inclusion_role_split"] == {"|".join(k): n for k, n in sorted(c.items())}


def test_no_confirm_every_row_exposed():
    _, rs = rows()
    assert not any(r["role"] == "CONFIRM" for r in rs)
    assert {r["exposure"] for r in rs} == {"used_for_design"}


def test_group_single_role_and_unique_paths():
    _, rs = rows()
    g = collections.defaultdict(set)
    for r in rs:
        if r["role"]:
            g[(r["dataset"], r["group_id"])].add(r["role"])
    assert all(len(v) == 1 for v in g.values())
    paths = [r[s] for r in rs for s in ("fixed_path", "moving_path")]
    assert len(paths) == len(set(paths))


def test_phase1_hold_on_test_cores():
    _, rs = rows()
    for r in rs:
        if r["inclusion"] == "include":
            for s in ("fixed_path", "moving_path"):
                assert "/closed/" not in r[s] and "outputs_v9/reg_test" not in r[s], r["record_id"]
        if r["original_split"] == "test":
            assert r["inclusion"] == "exclude" and r["exclusion_reason"].startswith("HOLD") and r["role"] == "DEV"


def test_every_exclusion_has_reason():
    _, rs = rows()
    assert all(r["exclusion_reason"].strip() for r in rs if r["inclusion"] == "exclude")
    assert all(r["inclusion"] in ("include", "exclude") for r in rs)


def test_build_checks_all_pass():
    s = summary()
    assert s["checks_failed"] == [] and all(c["status"] == "PASS" for c in s["checks"])
    assert s["confirm_assessment"]["eligible_groups"] == 0


@pytest.mark.parametrize("name", ["x.tif", "x.tiff", "test.zip", "c.npz", "m.pt", "closed/list.csv"])
def test_guard_refuses_data_reads(tmp_path, name):
    p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"not data")
    with pytest.raises(SystemExit):
        builder().read_meta(p)


def test_guard_allows_metadata(tmp_path):
    p = tmp_path / "ok.json"; p.write_text("{}")
    assert builder().read_meta(p) == b"{}"
