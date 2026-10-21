"""CPU tests of the TEST_REPEAT134 input diagnostic.  No TEST image or label: the data-touching test runs the diagnostic on
one DEV core (REG-P1 validation1 files) with a planted 1-ULP difference; the record test reads only bundle metadata.

    cd ssl_pilot_v10_test_repeat && OMP_NUM_THREADS=1 /home/infres/yinwang/CS_QMI/env/bin/python -m pytest -q tests/v10_test_repeat
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("V10_TR_BUNDLE", tempfile.mkdtemp(prefix="tr134_test_bundle_"))   # never the real bundle
os.environ.pop("SLURM_JOB_ID", None)
sys.path.insert(0, str(REPO / "scripts/v10_test_repeat"))
import tr_common as C  # noqa: E402
import diagnose_inputs as DI  # noqa: E402
import build_diag_authorization as BA  # noqa: E402

PY = "/home/infres/yinwang/CS_QMI/env/bin/python"
BF1_SRC = C.HAND / "v10_bf1_20261017" / "source"
HOST_HASH = "6c6f094684fbc50a"
REAL_BUNDLE = C.HAND / "v10_test_repeat134_20261020"
DEV_DATA = Path("/projects/EEG-foundation-model/yinghao/data/bf_shg_4550300/validation1")


def test_array_stats_synthetic():
    rng = np.random.default_rng(0)
    b = rng.normal(size=(8, 9)).astype(np.float32)
    assert DI.array_stats(b.copy(), b)["identical"] is True
    a = b.copy()
    for (i, j) in ((0, 0), (3, 4), (7, 8)):
        a[i, j] = np.nextafter(a[i, j], np.float32(np.inf))
    s = DI.array_stats(a, b)
    assert (s["identical"], s["n_differ"], s["ulp"]["max"], s["ulp"]["eq1"]) == (False, 3, 1, 3)
    assert s["differing_indices"] == [[0, 0], [3, 4], [7, 8]] and s["differing_index_range"] == [[0, 7], [0, 8]]
    rgb = rng.normal(size=(6, 5, 3)).astype(np.float32)
    rgb2 = rgb.copy()
    rgb2[2, 3, 1] += 0.5
    s = DI.array_stats(rgb2, rgb)
    assert s["channel_axis"] == 2 and [c["n_differ"] for c in s["per_channel"]] == [0, 1, 0]
    assert s["max_abs"] == pytest.approx(0.5, abs=1e-6) and s["ulp"]["gt16"] == 1
    n = b.copy()
    n[1, 1] = np.nan
    assert DI.array_stats(n, n.copy())["identical"] is True                     # NaN at the same place counts as equal
    m = n.copy()
    m[1, 1] = 0.0
    assert DI.array_stats(m, n)["nan_mismatch"] == 1
    assert DI.array_stats(b[:4], b)["comparable"] is False
    f = DI.array_stats((b.astype(np.float64) * 1.001 + 0.01).astype(np.float32), b)["affine_fit_a_on_b"]
    assert f["slope_minus_1"] == pytest.approx(0.001, abs=1e-5) and f["intercept"] == pytest.approx(0.01, abs=1e-5)
    C.dumps(s)                                                                 # finite, JSON-safe


def _dev_core():
    for p in sorted(DEV_DATA.glob("*_WB.tif")):
        core = p.name[: -len("_WB.tif")]
        if (DEV_DATA / f"{core}_MI.tif").exists():
            return core
    pytest.skip("no DEV core")


def test_diagnose_fixture_on_dev(tmp_path, monkeypatch):
    """One DEV core prepared once as the 'V10' root, copied as the 'V9' root with two shg_log elements moved by 1 ULP;
    diagnose() must report exactly that, reproduce the 'V10' file in fresh child processes and leave no rebuilt copy."""
    core = _dev_core()
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / f"{core}_BF.tif").symlink_to(DEV_DATA / f"{core}_WB.tif")
    (raw / f"{core}_SHG.tif").symlink_to(DEV_DATA / f"{core}_MI.tif")
    shas = (C.sha(DEV_DATA / f"{core}_WB.tif"), C.sha(DEV_DATA / f"{core}_MI.tif"))
    table = {core: {"bf_sha256": shas[0], "shg_sha256": shas[1]}}
    code = ("import json, sys; sys.path.insert(0, %r); import prepare_inputs as PI; from pathlib import Path; "
            "P, h = PI.host_modules(Path(%r), %r); PI.prepare(P, json.loads(%r), Path(%r), Path(%r))"
            % (str(REPO / "scripts/v10_test_repeat"), str(BF1_SRC), HOST_HASH, json.dumps(table), str(raw), str(tmp_path / "v10")))
    subprocess.run([PY, "-c", code], env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True, capture_output=True)
    shutil.copytree(tmp_path / "v10", tmp_path / "v9")
    p9 = tmp_path / "v9" / "cache" / f"{core}.npz"
    with np.load(p9) as z:
        arrays = {k: z[k].copy() for k in z.files}
    for (i, j) in ((200, 201), (300, 17)):
        arrays["shg_log"][i, j] = np.nextafter(arrays["shg_log"][i, j], np.float32(np.inf))
    np.savez_compressed(p9, **arrays)
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    import time
    now = int(time.time())
    windows = {"v9_g1_prepare": {"job": "a", "start_unix": now - 900, "end_unix": now + 900},
               "v10_prepare": {"job": "b", "start_unix": now - 9000, "end_unix": now - 8000}}
    res = DI.diagnose(core, {"fixture_inventory": shas}, raw, tmp_path / "v10", tmp_path / "v9", tmp_path, "4242",
                      BF1_SRC, HOST_HASH, windows)
    s = res["summary"]
    assert s["raw_members_match"] == {"fixture_inventory": True, "v9_g1_prepared_manifest": True, "v10_prepared_manifest": True}
    assert list(s["v10_vs_v9_differing_arrays"]) == ["shg_log"]
    assert s["v10_vs_v9_differing_arrays"]["shg_log"]["n_differ"] == 2 and s["v10_vs_v9_differing_arrays"]["shg_log"]["max_ulp"] == 1
    assert res["v10_vs_v9_g1"]["arrays"]["shg_log"]["differing_indices"] == [[200, 201], [300, 17]]
    assert s["rebuild_deterministic"] is True and {"default_1", "default_2"} <= set(s["rebuilds_equal_to_v10"])
    assert s["rebuilds_equal_to_v9_g1"] == []
    assert s["v9_manifest_pins_cache_sha256"] is False and res["provenance"]["prepared_manifests_byte_identical"] is True
    assert s["v9_cache_written_in_original_sequence"] is True and s["v9_cache_inside_v9_g1_prepare_window"] is True
    m10 = res["provenance"]["v10_rebuild"]["all_cache_mtimes"]
    assert (m10["core_inside_window"], m10["n_outside_window"], m10["window"]["margin_seconds"]) == (False, 1, DI.MARGIN)
    disp = {t: r["dispatch"]["NPY_DISABLE_CPU_FEATURES"] for t, r in res["rebuilds"].items()}
    assert disp == {"default_1": "", "default_2": "", "no_X86_V4": "X86_V4", "baseline_X86_V2": "X86_V3 X86_V4"}
    assert res["rebuilds"]["baseline_X86_V2"]["dispatch"]["add_float64"].startswith("baseline")
    if not s["node_has_X86_V4"]:                                                 # e.g. the login node: X86_V4 off already
        assert "no_X86_V4" in s["rebuilds_equal_to_v10"]
    assert not list(tmp_path.glob("v10t_diag_4242_*"))                          # rebuilt copies removed
    C.dumps(res)


def test_expected_hashes_real_inventories_agree():
    """Metadata only: the 1B_D8 row of the TEST_REPEAT134 manifest, V9-G1's frozen member CSV and the V10 data manifest
    (whose sha256_moving value carries a note) name the same two member sha256 values."""
    rec = {"manifest": C.read(REAL_BUNDLE / C.MANIFEST)}
    e = DI.expected_hashes(rec, "1B_D8")
    assert set(e) == {"v10_test_repeat_manifest", "v9_g1_frozen_member_csv", "v10_data_manifest"}
    assert len(set(e.values())) == 1 and all(len(h) == 64 for h in next(iter(e.values())))


def test_expected_hashes_parses_annotated_values(tmp_path, monkeypatch):
    h1, h2 = "1" * 64, "2" * 64
    csv_p, dm_p = tmp_path / "cores.csv", tmp_path / "repo" / "dm.csv"
    dm_p.parent.mkdir()
    csv_p.write_text(f"core,bf_sha256,shg_sha256\nX1,{h1},{h2}\n")
    dm_p.write_text("dataset,record_id,source_version\n"
                    f"BF_SHG,X1,members a / b (note); sha256_fixed={h1}; sha256_moving={h2} (V9-G1 frozen list)\n")
    monkeypatch.setattr(C, "REPO", tmp_path / "repo")
    rec = {"manifest": {"test": {"raw": [{"core": "X1", "bf_sha256": h1, "shg_sha256": h2}],
                                 "source": {"v9_member_list": {"path": str(csv_p), "sha256": C.sha(csv_p)},
                                            "data_manifest": {"path": "dm.csv", "sha256": C.sha(dm_p)}}}}}
    assert set(DI.expected_hashes(rec, "X1").values()) == {(h1, h2)}
    dm_p.write_text("dataset,record_id,source_version\nBF_SHG,X1,sha256_fixed=abc\n")
    rec["manifest"]["test"]["source"]["data_manifest"]["sha256"] = C.sha(dm_p)
    with pytest.raises(RuntimeError):
        DI.expected_hashes(rec, "X1")
    csv_p.write_text("changed\n")
    with pytest.raises(RuntimeError):                                          # pinned inventory changed
        DI.expected_hashes(rec, "X1")


def test_rebuild_child_refuses_outside_its_job(tmp_path, monkeypatch):
    spec = {"core": "X", "bf_sha256": "0", "shg_sha256": "0", "raw_root": str(tmp_path), "dest": str(tmp_path / "x" / "d"),
            "host_source_root": str(BF1_SRC), "host_code_hash": HOST_HASH}
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError):
        DI.rebuild_child(spec)
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    monkeypatch.setenv("V10_TR_DIAG_PARENT", "77")
    with pytest.raises(RuntimeError, match="job-local"):
        DI.rebuild_child(spec)


def diag_env(monkeypatch, tmp_path, *, cores=("X1",), lock_status="FAIL_V9_CACHE_MISMATCH", exec_root=None):
    """A fake bundle and execution repo: v4 records and code checks stubbed, the authorization and lock real files."""
    bundle, repo = tmp_path / "bundle", tmp_path / "repo"
    (repo / C.RECORDS).mkdir(parents=True)
    bundle.mkdir()
    for rel in BA.CODE:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(f"code {rel}\n")
    lock = {"status": lock_status, "job": "1", "activation_sha256": "a" * 64, "files": [{"core": "X1", "sha256": "f" * 64}],
            "v9_cache_comparison": {"differing": ["X1"]}}
    C.write(bundle / "INPUT_LOCK.json", lock)
    auth = {"v4_activation_sha256": "a" * 64, "failed_input_lock_sha256": C.sha(bundle / "INPUT_LOCK.json"), "cores": list(cores),
            "scope": "s", "code": {"files": {rel: C.sha(repo / rel) for rel in BA.CODE}, "execution_root": str(exec_root or repo)}}
    text = C.dumps(auth)
    (bundle / DI.AUTHORIZATION).write_text(text)
    (repo / C.RECORDS / DI.AUTHORIZATION).write_text(text)
    monkeypatch.setattr(C, "BUNDLE", bundle)
    monkeypatch.setattr(C, "REPO", repo)
    monkeypatch.setattr(C, "committed_bytes", lambda rel: (repo / rel).read_bytes())
    monkeypatch.setattr(C, "git", lambda *a, **k: "")
    rec = {"activation": {}, "activation_sha256": "a" * 64,
           "protocol": {"inputs": {"root": str(tmp_path / "v10"), "raw_root": str(tmp_path / "raw"),
                                   "v9_prepared_root": str(tmp_path / "v9"), "host_source_root": str(BF1_SRC),
                                   "host_code_hash": HOST_HASH}}, "manifest": {}}
    monkeypatch.setattr(C, "records", lambda pin=None: rec)
    monkeypatch.setattr(C, "verify_code", lambda a, c: {"commit": c})
    monkeypatch.setenv("SLURM_JOB_ID", "555")
    monkeypatch.setenv("V10_TR_EXPECT_ACTIVATION", "a" * 64)
    monkeypatch.setenv("V10_TR_EXPECT_COMMIT", "c" * 40)
    monkeypatch.setenv("V10_TR_EXPECT_DIAG", C.sha_bytes(text.encode()))
    (tmp_path / "v10" / "cache").mkdir(parents=True)
    return bundle, repo


class Reached(Exception):
    pass


def test_diag_main_reaches_diagnosis_only_with_every_pin(monkeypatch, tmp_path):
    bundle, repo = diag_env(monkeypatch, tmp_path)
    monkeypatch.setattr(C, "sha", lambda p: ("f" * 64 if str(p).endswith("X1.npz") else C.sha_bytes(Path(p).read_bytes())))
    monkeypatch.setattr(DI, "expected_hashes", lambda rec, core: {"m": ("b", "s")})
    monkeypatch.setattr(DI, "diagnose", lambda *a, **k: (_ for _ in ()).throw(Reached(a[0])))
    with pytest.raises(Reached):
        DI.main(["X1"])


def test_check_pins_outside_slurm_reads_no_test_file(monkeypatch, tmp_path):
    diag_env(monkeypatch, tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID")
    seen, real_sha = [], C.sha
    monkeypatch.setattr(C, "sha", lambda p: (seen.append(str(p)), real_sha(p))[1])
    monkeypatch.setattr(DI, "diagnose", lambda *a, **k: (_ for _ in ()).throw(Reached()))
    assert DI.main(["--check-pins", "X1"]) == 0
    assert seen and not any(f"/{d}/" in s for s in seen for d in ("v10", "v9", "raw"))
    with pytest.raises(RuntimeError):
        DI.main(["--check-pins", "X2"])
    with pytest.raises(RuntimeError):                                          # the diagnosis itself still needs Slurm
        DI.main(["X1"])


@pytest.mark.parametrize("what", ["no_slurm", "no_pin", "auth_pin", "scope", "lock_changed", "lock_pass", "exec_root",
                                  "code_changed", "output_exists", "v10_file"])
def test_diag_main_refuses(monkeypatch, tmp_path, what):
    bundle, repo = diag_env(monkeypatch, tmp_path, lock_status="PASS" if what == "lock_pass" else "FAIL_V9_CACHE_MISMATCH",
                            exec_root=tmp_path / "elsewhere" if what == "exec_root" else None)
    real_sha = C.sha
    monkeypatch.setattr(C, "sha", lambda p: ("f" * 64 if str(p).endswith("X1.npz") and what != "v10_file" else real_sha(p)))
    monkeypatch.setattr(DI, "diagnose", lambda *a, **k: (_ for _ in ()).throw(Reached()))
    monkeypatch.setattr(DI, "expected_hashes", lambda rec, core: {})
    core = "X1"
    if what == "no_slurm":
        monkeypatch.delenv("SLURM_JOB_ID")
    elif what == "no_pin":
        monkeypatch.delenv("V10_TR_EXPECT_DIAG")
    elif what == "auth_pin":
        monkeypatch.setenv("V10_TR_EXPECT_DIAG", "0" * 64)
    elif what == "scope":
        core = "X2"
    elif what == "lock_changed":
        C.write(bundle / "INPUT_LOCK.json", {**C.read(bundle / "INPUT_LOCK.json"), "job": "2"})
    elif what == "code_changed":
        (repo / "scripts/v10_test_repeat/diagnose_inputs.py").write_text("changed\n")
    elif what == "output_exists":
        (bundle / "diagnostics" / "X1" / "555").mkdir(parents=True)
    if what == "v10_file":
        (tmp_path / "v10" / "cache" / "X1.npz").write_bytes(b"x")
    with pytest.raises(RuntimeError):
        DI.main([core])


def test_authorization_record_real_bundle(monkeypatch):
    """Read-only against the real bundle (metadata: activation, failed lock, prepare receipt; the V9-G1 prepare log)."""
    if not (REAL_BUNDLE / "INPUT_LOCK.json").exists():
        pytest.skip("no INPUT_LOCK")
    monkeypatch.setattr(C, "BUNDLE", REAL_BUNDLE)
    r1, r2 = BA.record(), BA.record()
    assert C.dumps(r1) == C.dumps(r2)
    assert r1["cores"] == ["1B_D8"] and r1["v4_activation_sha256"] == C.sha(REAL_BUNDLE / C.V4_ACTIVATION)
    assert r1["failed_input_lock_sha256"] == C.sha(REAL_BUNDLE / "INPUT_LOCK.json")
    w9, w10 = r1["windows"]["v9_g1_prepare"], r1["windows"]["v10_prepare"]
    assert (w9["node"], w9["start_utc"], w9["end_utc"]) == ("nodecpu10", "2026-10-15T23:36:07Z", "2026-10-15T23:41:35Z")
    assert (w10["job"], w10["node"]) == ("1051404", "node57") and w10["end_unix"] > w10["start_unix"]
    assert set(r1["code"]["files"]) == set(BA.CODE) and r1["code"]["execution_root"].endswith("exec_v4diag_20261020")
    frozen = C.read(REAL_BUNDLE / BA.NAME)                                          # the record the diagnostic job ran under
    assert frozen == C.read(REPO / C.RECORDS / BA.NAME) and C.sha(REAL_BUNDLE / BA.NAME) == DIAG_AUTH_SHA
    assert all(frozen["code"]["files"][rel] == C.sha_bytes(git_blob(DIAG_COMMIT, rel)) for rel in BA.CODE)
    assert {k: v for k, v in r1.items() if k != "code"} == {k: v for k, v in frozen.items() if k != "code"}


def git_blob(commit: str, rel: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=REPO, capture_output=True, check=True).stdout


DIAG_COMMIT, DIAG_AUTH_SHA = "af7c0798c886886db18ce6973c5a0111bd97c305", "fe4e6093dfbaf85ece802d620346ec66bc6e37972230e66dee373a0ba5ad660a"


def test_v4_pinned_code_unchanged():
    """v4's pins name the code of the v4 freeze commit (v5 changes the working tree, not v4)."""
    act = C.read(REAL_BUNDLE / "TEST_REPEAT134_ACTIVATION_V4.json")
    assert set(act["code"]["files"]) == {"scripts/v10_test_repeat/tr_common.py", "scripts/v10_test_repeat/evaluate.py",
                                         "scripts/v10_test_repeat/prepare_inputs.py", "slurm/v10_test_repeat/gpu.sbatch"}
    assert all(C.sha_bytes(git_blob("827099d95702204352b589e4a9d1801429f68080", rel)) == v for rel, v in act["code"]["files"].items())
    ex = C.CS / "ssl_pilot_v10_test_repeat_exec_v4_20261020"
    assert all(C.sha(ex / rel) == v for rel, v in act["code"]["files"].items())


def test_diag_sbatch():
    p = REPO / "slurm/v10_test_repeat/diag.sbatch"
    subprocess.run(["bash", "-n", str(p)], check=True)
    text = p.read_text()
    for needle in ("--partition=cpu-high", "--gres=gpu:0", "V10_TR_EXPECT_ACTIVATION:?", "V10_TR_EXPECT_COMMIT:?",
                   "V10_TR_EXPECT_DIAG:?", "ssl_pilot_v10_test_repeat_exec_v4diag_20261020", '"diagnostics" / "exits"',
                   "unset NPY_DISABLE_CPU_FEATURES"):
        assert needle in text
    assert "gpu:1" not in text
