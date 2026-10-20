"""Job matrix, sbatch hygiene, D1 selection rule and the mechanical grid of summarize.py (synthetic records)."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from v10.supplement.records import selection_key
from v10_sensitivity import common as C

SCRIPTS = C.REPO / "scripts/v10_sensitivity"


def load(name):
    spec = importlib.util.spec_from_file_location("v10d_" + name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_job_matrix_and_sbatch_hygiene():
    jobs = load("prepare_bundle").JOBS
    pre = C.prereg()
    assert len(jobs) == 9 and len({j["name"] for j in jobs}) == 9
    assert all(j["name"].startswith("v10d_") and "v9" not in j["name"] for j in jobs)
    assert [j["args"][0] for j in jobs if j["diagnostic"] == "d1"] == pre["D1"]["units"]
    assert all(j["partition"] == "L40S" for j in jobs if j["diagnostic"] in ("d1", "d2"))
    assert {(j["args"][0], j["partition"]) for j in jobs if j["diagnostic"] == "d3"} == \
        {(j["family"], j["partition"]) for j in pre["D3"]["jobs"]}
    for name in ("gpu.sbatch", "cpu_check.sbatch"):
        path = C.REPO / "slurm/v10_sensitivity" / name
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text()
        assert "PYTHONDONTWRITEBYTECODE=1" in text and "--nice=0" in text and "--qos=normal" in text
    gpu = (C.REPO / "slurm/v10_sensitivity/gpu.sbatch").read_text()
    assert "--exclude=node51,node60" in gpu and "nvidia-smi" in gpu and "INFRA_CUDA_UNAVAILABLE" in gpu
    assert "unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE" in gpu and "CUBLAS_WORKSPACE_CONFIG=:4096:8" in gpu
    assert "OMP_NUM_THREADS=4" in gpu  # the frozen torch_threads check expects 4
    assert "cpu-high" in (C.REPO / "slurm/v10_sensitivity/cpu_check.sbatch").read_text()


def test_d1_selection_is_items_1_to_4_with_earlier_step_ties():
    def r(step, s42, s9, med):
        return {"step": step, "metrics": {"n": 80, "success42": s42, "success9": s9, "median_err": med}}
    rows = [r(64, 0.75, 0.75, 3.0), r(128, 0.875, 0.5, 9.0), r(192, 0.875, 0.625, 4.0),
            r(256, 0.875, 0.625, 4.0), r(736, 0.875, 0.625, 3.9)]
    assert min(rows, key=selection_key)["step"] == 736  # median breaks the 42/9 tie
    rows[-1]["metrics"]["median_err"] = 4.0
    assert min(rows, key=selection_key)["step"] == 192  # full tie: the earlier checkpoint


def dev_rows(fail_cores, n_cores=32, theta=5.0):
    rows = []
    for c in range(n_cores):
        core = f"1B_Z{c:02d}"
        for i in range(10):
            err = 100.0 if core in fail_cores else 2.0
            rows.append({"core": core, "perturbation": i, "err_full": err, "theta": theta * (i % 7) - 15,
                         "theta_hat": 0.0, "t_hat": [0.0, 0.0], "F_hat": 0.0, "F_true": 1.0})
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def complete(directory, extra):
    artifacts = {str(p.relative_to(directory)): C.sha(p) for p in directory.rglob("*") if p.is_file()}
    C.write(directory / "COMPLETE.json", {"status": "COMPLETE", "artifacts": artifacts, **extra})


@pytest.mark.parametrize("case", ["G1", "G2", "G3"])
def test_summarize_applies_the_d1_grid_mechanically(tmp_path, case):
    summarize = load("summarize")
    pre = C.prereg()
    frozen_base = tmp_path / "frozen"
    pre["frozen_inputs"]["comir_release"]["path"] = str(frozen_base / "RELEASE.json")
    out = tmp_path / "out"
    metric = {"n": 240, "success42": 0.8, "success9": 0.5, "median_err": 5.0}
    cal_rows = [{"epoch": e, "step": 32 * e, "metrics": {"n": 80, "success42": 0.75, "success9": 0.7, "median_err": 3.0}}
                for e in pre["D1"]["epochs"]]
    for key in ("CAL_1017", "FINAL_1001", "FINAL_1002", "FINAL_1003"):
        unit = frozen_base / pre["frozen_inputs"]["comir_release"]["units"][key]["dir"]
        unit.mkdir(parents=True)
        selected_step = 736 if case == "G1" or key == "CAL_1017" else 320
        directory = out / "D1" / key
        directory.mkdir(parents=True)
        extra = {"selected": {"epoch": selected_step // 32, "step": selected_step}, "cal_rows": cal_rows,
                 "qc_replay_cal_0736": "IDENTICAL"}
        if key != "CAL_1017":
            frozen_fail = {f"1B_Z{c:02d}" for c in range(8)}
            write_rows(unit / "DEV.jsonl", dev_rows(frozen_fail))
            C.write(unit / "COMPLETE.json", {"DEV24": metric})
            better = {"G1": frozen_fail, "G2": set(),
                      "G3": (frozen_fail - {"1B_Z07"}) | {"1B_Z08"}}[case]
            write_rows(directory / "DEV.jsonl", dev_rows(better))
            extra["DEV24"] = metric
        complete(directory, extra)
    report = summarize.d1(pre, out)
    assert report["status"] == "COMPLETE"
    expected = {"G1": "G1_all_select_23", "G2": "G2_material_gain", "G3": "G3_no_material_change"}[case]
    assert report["grid_cell"] == expected, report["paired_success42_mean_over_seeds"]
    assert report["gap_clause"].startswith("checkpoint timing accounts" if case == "G2" else "gap not explained")
    assert report["paired_success42_mean_over_seeds"]["mean"] == {"G1": 0.0, "G2": 0.25, "G3": 0.0}[case]


def test_summarize_reports_incomplete_instead_of_classifying(tmp_path):
    summarize = load("summarize")
    report = summarize.d1(C.prereg(), tmp_path)
    assert report["status"] == "INCOMPLETE" and "grid_cell" not in report and len(report["missing"]) == 4
    assert summarize.d2(C.prereg(), tmp_path) == {"status": "INCOMPLETE"}


FIT = ["2B_F4", "2B_H4", "2B_H9", "1B_D5", "1B_G3", "2B_Q1"]
CAL = ["1B_G5", "1B_H2", "1B_I5", "2B_Q2"]
DEV = list(C.DEV24_EXCLUDED) + [f"1B_Z{i:02d}" for i in range(24)]


class FakeGuard:
    def __init__(self):
        self.reads = [{"role": "CAL"}]

    def included(self, role):
        return {"FIT": FIT, "CAL": CAL, "DEV": DEV}[role]


def fake_runner(records, label=None, session_name="RegistrationSession"):
    class R:
        RegistrationSession = type(session_name, (), {})

        class CP:
            @staticmethod
            def load(trainer, path, identity):
                return {"step": int(Path(path).stem.split("_")[1]), "sha256": "s"}

        @staticmethod
        def check_data(lock, guard, roles):
            assert roles == {"DEV"}
            guard.reads.append({"role": "DEV"})

        @staticmethod
        def evaluate(trainer, guard, role, cores, path, ident, checkpoint, allocation):
            rows = [{"core": c, "perturbation": i, "err_full": 3.0 if i else 60.0, "theta": 25.0 if i == 9 else 1.0,
                     "theta_hat": 0.0, "t_hat": [0.0, 0.0], "F_hat": 0.0, "F_true": 0.0,
                     **({"variant": label} if label else {})} for c in cores for i in range(10)]
            write_rows(path, rows)
            return {"metrics": records.metrics(rows)}
    return R


def test_d1_dev_block_only_at_the_written_selection(tmp_path):
    from v10.supplement import records
    d1 = load("d1_comir_checkpoints")
    steps = C.prereg()["D1"]["steps"]
    out, unit = tmp_path / "out", tmp_path / "unit"
    chosen = {"step": 320, "epoch": 10}
    C.write(out / "SELECTION.json", {"selected": chosen})
    R = fake_runner(records)
    extra = d1.evaluate_dev(R, records, None, FakeGuard(), None, unit, {}, {}, chosen, steps, out, None, CAL)
    assert extra["DEV_epoch"] == 10 and extra["DEV32"]["n"] == 320 and extra["DEV24"]["n"] == 240
    assert len(extra["per_core"]) == 32 and "QC_REPLAY_DEV" not in extra
    with pytest.raises(RuntimeError, match="written CAL selection"):
        d1.evaluate_dev(R, records, None, FakeGuard(), None, unit, {}, {}, {"step": 384, "epoch": 12}, steps, out, None, CAL)
    chosen = {"step": 736, "epoch": 23}
    C.write(out / "SELECTION.json", {"selected": chosen})
    (out / "DEV.jsonl").unlink()
    R.evaluate(None, None, "DEV", DEV, unit / "DEV.jsonl", None, None, None)  # frozen rows = the same rows
    extra = d1.evaluate_dev(R, records, None, FakeGuard(), None, unit, {}, {}, chosen, steps, out, None, CAL)
    assert extra["QC_REPLAY_DEV"]["status"] == "IDENTICAL" and (out / "QC_REPLAY_DEV_0736.json").exists()


def test_d2_dev_block_requires_the_variant_session_and_labels(tmp_path):
    from v10.supplement import records
    d2 = load("d2_mind_wtd")
    variant = C.prereg()["D2"]["variant"]
    with pytest.raises(RuntimeError, match="variant session"):
        d2.evaluate_variant_dev(fake_runner(records, variant), records, variant, None, FakeGuard(), None, {}, {}, None,
                                CAL, tmp_path)
    with pytest.raises(RuntimeError, match="variant label"):
        d2.evaluate_variant_dev(fake_runner(records, None, "WarpThenDescribeSession"), records, variant, None,
                                FakeGuard(), None, {}, {}, None, CAL, tmp_path / "a")
    extra = d2.evaluate_variant_dev(fake_runner(records, variant, "WarpThenDescribeSession"), records, variant, None,
                                    FakeGuard(), None, {}, {}, None, CAL, tmp_path / "b")
    assert extra["DEV32"]["n"] == 320 and extra["DEV24"]["n"] == 240 and extra["dev_strata"]["20-30"]["n"] == 32


THETAS = (-29.0, -25.0, -21.0, -15.0, -12.0, -5.0, 0.0, 4.0, 13.0, 27.0)


def mind_rows(fail):
    """fail(core_index, theta) -> bool; 32 cores x the 10 fixed angles."""
    return [{"core": f"1B_Z{c:02d}", "perturbation": i, "theta": t, "err_full": 80.0 if fail(c, t) else 2.0,
             "theta_hat": 0.0, "t_hat": [0.0, 0.0], "F_hat": 1.0 if fail(c, t) else 0.0, "F_true": 0.5}
            for c in range(32) for i, t in enumerate(THETAS)]


@pytest.mark.parametrize("case", ["G1", "G3", "G4"])
def test_summarize_applies_the_d2_grid_mechanically(tmp_path, case):
    summarize = load("summarize")
    pre = C.prereg()
    frozen_base = tmp_path / "frozen"
    pre["frozen_inputs"]["mind_release"]["path"] = str(frozen_base / "RELEASE.json")
    unit = frozen_base / pre["frozen_inputs"]["mind_release"]["unit"]["dir"]
    frozen_fail = lambda c, t: abs(t) >= 20 and c % 2 == 0
    write_rows(unit / "DEV.jsonl", mind_rows(frozen_fail))
    write_rows(unit / "CAL/step_0000.jsonl", mind_rows(frozen_fail)[:80])
    metric = {"n": 240, "success42": 0.7, "success9": 0.4, "median_err": 14.0}
    C.write(unit / "COMPLETE.json", {"DEV24": metric})
    variant_fail = {"G1": lambda c, t: False, "G3": frozen_fail,
                    "G4": lambda c, t: frozen_fail(c, t) or (abs(t) < 10 and c < 24)}[case]
    directory = tmp_path / "out/D2" / pre["D2"]["variant"]
    write_rows(directory / "VARIANT/DEV.jsonl", mind_rows(variant_fail))
    write_rows(directory / "VARIANT/CAL_step_0000.jsonl", mind_rows(variant_fail)[:80])
    complete(directory, {"replay_frozen_cal": {"status": "IDENTICAL"}, "DEV24": metric})
    report = summarize.d2(pre, tmp_path / "out")
    assert report["status"] == "COMPLETE" and report["frozen"]["DEV32_strata"]["20-30"]["success42"] == 0.5
    assert report["grid_cell"] == {"G1": "G1_orientation_explains", "G3": "G3_not_orientation",
                                   "G4": "G3_not_orientation"}[case]
    assert report["G4_overall_worse"] is (case == "G4")
    assert report["frozen"]["failures"]["energy_optimum_wrong"] == report["frozen"]["failures"]["failures"] == 64
