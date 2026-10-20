"""Seed-1001 L40S hardware replicate: same-unit guard, separate root, agreement record. CPU only, no data."""
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/v10_supplement"))
import rulsif_v2_hwrep as H  # noqa: E402
from v10.supplement import contract as C  # noqa: E402


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def row(core, i, err, **extra):
    return {"core": core, "perturbation": i, "err_full": err, "theta_hat": 1.0, "t_hat": [2.0, 3.0], "F_hat": 0.02,
            "F_true": 0.01, "rot_err_deg": 0.5, "trans_err_full": 1.0, **extra}


def test_compare_counts_identical_values_differences_and_threshold_flips():
    primary = {("a", i): row("a", i, e) for i, e in enumerate((5.0, 8.9, 41.0, 100.0))}
    same = {k: dict(v) for k, v in primary.items()}
    result = H.compare(primary, same)
    assert result["err_full"]["identical_fraction"] == 1 and result["err_full"]["max_abs_difference"] == 0
    assert result["success42_flips"] == result["success9_flips"] == 0 and result["t_hat"]["identical_fraction"] == 1
    moved = {k: dict(v) for k, v in primary.items()}
    moved[("a", 1)]["err_full"] = 9.2                      # crosses 9
    moved[("a", 2)]["err_full"] = 43.0                     # crosses 42
    moved[("a", 3)]["t_hat"] = [2.0, 3.5]
    result = H.compare(primary, moved)
    assert result["err_full"]["identical_fraction"] == 0.5
    assert result["err_full"]["max_abs_difference"] == pytest.approx(2.0)
    assert result["success9_flips"] == 1 and result["success42_flips"] == 1
    assert result["t_hat"]["max_abs_difference"] == pytest.approx(0.5)
    with pytest.raises(RuntimeError):
        H.compare(primary, {k: v for k, v in moved.items() if k != ("a", 0)})


def test_delta_allows_only_the_diagnostic_mode():
    base = {"src/v10/supplement/rulsif_v2.py": "1", "scripts/v10_supplement/run.py": "2", "tests/v10/test_supplement.py": "3",
            "slurm/v10_supplement/run.sbatch": "4", "v10/supplement/RULSIF_V2_AMENDMENT_20261019.json": "5"}
    mode = {**base, "scripts/v10_supplement/run.py": "9", "scripts/v10_supplement/rulsif_v2_hwrep.py": "8",
            "tests/v10/test_rulsif_v2_hwrep.py": "7"}
    change = H.delta(base, mode)
    assert change["allowlisted_only"] and change["science_identical"]
    for bad in ({**mode, "src/v10/supplement/rulsif_v2.py": "x"}, {**mode, "slurm/v10_supplement/run.sbatch": "x"},
                {**mode, "scripts/v10_supplement/validate.py": "x"}, {**mode, "tests/v10/test_supplement.py": "x"},
                {k: v for k, v in mode.items() if not k.startswith("v10/")}):
        change = H.delta(base, bad)
        assert not (change["allowlisted_only"] and change["science_identical"]), bad


@pytest.fixture
def units(tmp_path, monkeypatch):
    primary, unit = tmp_path / "primary/seed_1001", tmp_path / "rep/results/FINAL_DEV/fam/seed_1001__L40S_replicate"
    monkeypatch.setattr(H, "PRIMARY", primary); monkeypatch.setattr(H, "UNIT", unit)
    monkeypatch.setattr(H, "ROOT", tmp_path / "rep"); monkeypatch.setattr(H, "AGREEMENT", tmp_path / "rep/AGREEMENT.json")
    ident = {"release_sha256": H.XHW_RELEASE_SHA256, "seed": 1001, "phase": "FINAL_DEV"}
    dump(primary / "IDENTITY.json", ident)
    gate = dump(tmp_path / "gate.json", {"job": "1049287"})
    return primary, unit, ident, {"l40s_gate": {"path": str(gate)}}


def test_replicate_is_the_same_unit_in_its_own_directory_on_l40s(units):
    primary, unit, ident, release = units
    env = {"gpu_name": "NVIDIA L40S", "host": "node50", "slurm_nodelist": "node50", "slurm_partition": "L40S", "slurm_job_id": "5"}
    with pytest.raises(RuntimeError, match="differs from the primary"):
        H.prepare_replicate({**ident, "seed": 1002}, env, release)
    with pytest.raises(RuntimeError, match="L40S only"):
        H.prepare_replicate(ident, {**env, "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}, release)
    directory, hardware = H.prepare_replicate(ident, env, release)
    assert directory == unit and hardware["gpu_name"] == "NVIDIA L40S" and hardware["gate_job"] == "1049287"
    assert C.read(unit / "DIAGNOSTIC.json")["kind"] == H.STATUS and C.read(unit / "HARDWARE.json")["node"] == "node50"
    assert not (primary / "HARDWARE.json").exists()
    dump(unit / "IDENTITY.json", {**ident, "seed": 7})
    with pytest.raises(RuntimeError, match="another unit"):
        H.prepare_replicate(ident, env, release)


def test_agreement_record_against_the_primary_run(units):
    primary, unit, ident, release = units
    dev = [row(f"c{k}", i, 5.0 + k + i) for k in range(3) for i in range(2)]
    cal = [row("cal", i, 10.0 * i) for i in range(4)]
    m = lambda rs: {"n": len(rs), "success42": sum(r["err_full"] < 42 for r in rs) / len(rs),
                    "success9": sum(r["err_full"] < 9 for r in rs) / len(rs),
                    "median_err": sorted(r["err_full"] for r in rs)[len(rs) // 2]}
    for directory, gpu, job, shift in ((primary, "NVIDIA RTX PRO 6000 Blackwell Server Edition", "1050357", 0.0),
                                       (unit, "NVIDIA L40S", "9", 0.5)):
        rows_dev = [dict(r, err_full=r["err_full"] + (shift if r["core"] == "c2" else 0.0)) for r in dev]
        jsonl(directory / "DEV.jsonl", rows_dev); jsonl(directory / "CAL/step_0001.jsonl", cal)
        dump(directory / "COMPLETE.json", {"identity": ident, "job": job, "DEV32": m(rows_dev), "DEV24": m(rows_dev[:4])})
        dump(directory / "HARDWARE.json", {"gpu_name": gpu})
        dump(directory / "checkpoints/step_0001.json", {"model_sha256": "same"})
    result = H.agreement()
    dev32 = result["DEV32"]["per_registration"]
    assert dev32["n"] == 6 and dev32["err_full"]["identical_fraction"] == pytest.approx(4 / 6)
    assert dev32["err_full"]["max_abs_difference"] == pytest.approx(0.5)
    assert result["CAL8"]["per_registration"]["err_full"]["identical_fraction"] == 1
    assert result["DEV32"]["replicate_minus_primary"]["median_err"] == pytest.approx(
        result["DEV32"]["replicate"]["median_err"] - result["DEV32"]["primary"]["median_err"])
    assert result["fit"]["model_sha256_equal"] and result["primary"]["gpu"].startswith("NVIDIA RTX")
    assert C.read(H.AGREEMENT)["replicate"]["job"] == "9"
