"""EPFL results analysis: the post-hoc evaluation-noise section (EPFL_EVAL_NOISE_20261020).  Recorded vs re-evaluated DEV per final, the
per-family noise floor, the between-family differences of the seed means with the below-noise flag, the PENDING / FAIL / UNVERIFIED states.
CPU; reads result JSON / JSONL only, no image (the re-evaluations are stand-ins built from the recorded rows)."""
import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

from v10.epfl import production as PR

REPO = PR.REPO
PARENT = PR.PROJECT / "handoff" / "v10_epfl_production_20261019"
L40S = "NVIDIA L40S"
UNITS = (("MF_VCS_M1", 1002), ("MI_native", 1001))


def load(name):
    spec = importlib.util.spec_from_file_location(f"epfl_{name}", REPO / "scripts" / "v10_epfl" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _group():
    return json.loads((REPO / "v10" / "epfl" / "EPFL_GROUP_RECORD_V2_FROZEN.json").read_text())


def _bundle(tmp_path, monkeypatch, an):
    specs = {}
    for fam, seed in UNITS:
        u = PARENT / "results" / "FINAL_DEV" / fam / f"seed_{seed}"; c = json.loads((u / "COMPLETE.json").read_text())
        specs[f"{fam}/{seed}"] = {"family": fam, "seed": seed, "recorded_unit": str(u), "out": str(tmp_path / "results" / fam / f"seed_{seed}"),
                                  "complete_sha256": sha(u / "COMPLETE.json"), "recorded_job": c["job"], "gpu_name": L40S,
                                  "selected_step": c["selected"]["step"], "selected_checkpoint_sha256": c["selected_checkpoint_sha256"]}
    rel = {"schema": an.EVAL_NOISE_SCHEMA, "reference_commit": "x" * 40, "recorded_release": {"sha256": "r" * 64}, "units": specs}
    p = tmp_path / "RELEASE_xxxxxxxx.json"; p.write_text(json.dumps(rel))
    monkeypatch.setattr(an, "EVAL_NOISE", tmp_path)
    return p, rel


def _complete(tmp_path, an, rel_path, spec, rows, job, checks_ok=True, gpu=L40S, kind=None):
    out = Path(spec["out"]); out.mkdir(parents=True, exist_ok=True)
    (out / "DEV.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    rec = json.loads((Path(spec["recorded_unit"]) / "COMPLETE.json").read_text())
    c = {"status": "COMPLETE", "identity": {**rec["identity"], "diagnostic": {"kind": kind or an.EVN_LABEL, "release_sha256": sha(rel_path)}},
         "hardware": {"gpu_name": gpu}, "checks": [{"check": "x", "status": "PASS" if checks_ok else "FAIL"}], "job": job,
         "artifacts": {"DEV.jsonl": sha(out / "DEV.jsonl")}}
    (out / "COMPLETE.json").write_text(json.dumps(c))
    (tmp_path / "exits").mkdir(exist_ok=True)
    (tmp_path / "exits" / f"{job}.json").write_text(json.dumps({"job": job, "exit_code": 0, "allocated_seconds": 1800.0}))


def _families(an, inp, group, sets_units):
    out = {}
    for fam, seeds in (("MF_VCS_M1", (1001, 1002, 1003)), ("MI_native", (1001,))):
        finals = {s: an.final_record(inp, PARENT / "results" / "FINAL_DEV" / fam / f"seed_{s}", group, {}, "x") for s in seeds}
        out[fam] = {"summary": an.family_summary(finals, sets_units)}
    return out


def _recorded(fam, seed):
    return [json.loads(x) for x in (PARENT / "results" / "FINAL_DEV" / fam / f"seed_{seed}" / "DEV.jsonl").read_text().splitlines() if x.strip()]


def test_pending_fail_and_no_release(tmp_path, monkeypatch):
    an = load("analyze"); group = _group(); sets_units = {n: sorted(group["sensitivity_sets"][n]["units"]) for n in an.SETS}
    monkeypatch.setattr(an, "EVAL_NOISE", tmp_path / "none")
    assert an.eval_noise(an.Inputs(), group, sets_units, {}, {}, [])[0] == {"status": "NO_RELEASE"}
    p, rel = _bundle(tmp_path, monkeypatch, an)
    out = Path(rel["units"]["MI_native/1001"]["out"]); out.mkdir(parents=True); (out / "FAIL_77.json").write_text(json.dumps({"status": "FAIL"}))
    inp = an.Inputs(); problems = []
    res, files = an.eval_noise(inp, group, sets_units, _families(an, inp, group, sets_units), {}, problems)
    assert res["status"] == "PENDING" and files == {} and not problems
    assert res["units"]["MF_VCS_M1/1002"]["status"] == "PENDING" and res["units"]["MI_native/1001"]["status"].startswith("FAIL (no result used)")
    assert all(v["status"] == "PENDING" for v in res["family_floors"].values())
    rows = res["between_family_differences"]["DEV72_primary"]["success9"]
    assert len(rows) == 1 and rows[0]["observed_repeat_change_max"] is None and "below_noise" not in rows[0]
    assert "not a calibrated noise distribution" in res["statement"]


def test_complete_floors_and_between_family(tmp_path, monkeypatch):
    an = load("analyze"); group = _group(); sets_units = {n: sorted(group["sensitivity_sets"][n]["units"]) for n in an.SETS}
    p, rel = _bundle(tmp_path, monkeypatch, an)
    m1 = _recorded("MF_VCS_M1", 1002)
    _complete(tmp_path, an, p, rel["units"]["MF_VCS_M1/1002"], copy.deepcopy(m1), "901")                # a bitwise repeat
    mi = _recorded("MI_native", 1001); noisy = copy.deepcopy(mi)
    for r in noisy[::7]:                                                                                # search noise: estimates and errors move
        r["err_full"] = r["err_full"] + 12.0; r["theta_hat"] = r["theta_hat"] + 0.01
    _complete(tmp_path, an, p, rel["units"]["MI_native/1001"], noisy, "902")
    inp = an.Inputs(); problems = []
    fams = _families(an, inp, group, sets_units)
    res, files = an.eval_noise(inp, group, sets_units, fams, {}, problems)
    assert res["status"] == "COMPLETE" and not problems, problems
    a = res["units"]["MF_VCS_M1/1002"]
    assert a["per_registration_agreement"]["row_bitwise_equal"] == 740 and a["per_registration_agreement"]["F_true_equal"] == 740
    assert all(a["unit_macro_difference"][n]["abs_difference"] == {"success42": 0.0, "success9": 0.0, "median_err_full": 0.0} for n in an.SETS)
    b = res["units"]["MI_native/1001"]; ag = b["per_registration_agreement"]
    assert ag["row_bitwise_equal"] == 740 - len(noisy[::7]) and ag["F_true_equal"] == 740 and abs(ag["delta_err_full"]["max_abs"] - 12.0) < 1e-9
    first = an.SETS[0]
    s_old = an.set_stats(mi, group["sensitivity_sets"][first], group["window_unit"])["unit_macro"]
    s_new = an.set_stats(noisy, group["sensitivity_sets"][first], group["window_unit"])["unit_macro"]
    for k in ("success42", "success9", "median_err_full"):
        assert abs(b["unit_macro_difference"][first]["difference_rerun_minus_primary"][k] - (s_new[k] - s_old[k])) < 1e-12
    fl = res["family_floors"]
    assert fl["MF_VCS_M1"]["status"] == "COMPLETE" and fl["MF_VCS_M1"]["floor_max_abs"][first]["success9"] == 0.0
    assert abs(fl["MI_native"]["floor_max_abs"][first]["success9"] - abs(s_new["success9"] - s_old["success9"])) < 1e-12
    row = res["between_family_differences"][first]["success9"][0]
    want = fams[row["a"]]["summary"]["seed_mean"][first]["unit_macro"]["success9"] - fams[row["b"]]["summary"]["seed_mean"][first]["unit_macro"]["success9"]
    assert abs(row["difference"] - want) < 1e-12 and row["observed_repeat_change_max"] == max(fl["MF_VCS_M1"]["floor_max_abs"][first]["success9"],
                                                                                 fl["MI_native"]["floor_max_abs"][first]["success9"])
    f = b["per_registration_file"]
    assert f["rows"] == 740 and Path(f["path"]).parent == tmp_path / "analysis" and hashlib.sha256(files[f["path"]].encode()).hexdigest() == f["sha256"]
    md = an.markdown({"families": {}, "sensitivity_deltas_vs_primary": {}, "rulsif_v2": {}, "disclosures": {}, "problems": [],
                      "bootstrap": {"B": an.B_BOOT, "seed": an.BOOT_SEED}, "evaluation_noise": res})
    assert "不是已标定的噪声分布" in md and "不可解释" not in md and "| MI_native | 1001 | L40S | COMPLETE |" in md


def test_unverified_and_ftrue_change_are_flagged(tmp_path, monkeypatch):
    an = load("analyze"); group = _group(); sets_units = {n: sorted(group["sensitivity_sets"][n]["units"]) for n in an.SETS}
    p, rel = _bundle(tmp_path, monkeypatch, an)
    _complete(tmp_path, an, p, rel["units"]["MF_VCS_M1/1002"], _recorded("MF_VCS_M1", 1002), "901", gpu="NVIDIA H100 NVL")   # wrong GPU
    ft = _recorded("MI_native", 1001); ft[0]["F_true"] += 1e-7
    _complete(tmp_path, an, p, rel["units"]["MI_native/1001"], ft, "902")
    inp = an.Inputs(); problems = []
    res, _ = an.eval_noise(inp, group, sets_units, _families(an, inp, group, sets_units), {}, problems)
    assert res["units"]["MF_VCS_M1/1002"]["status"] == "UNVERIFIED (not used)" and res["family_floors"]["MF_VCS_M1"]["status"] == "PENDING"
    assert any("MF_VCS_M1/1002" in x for x in problems) and any("F_true differs in 1 rows" in x for x in problems)
    (Path(rel["units"]["MI_native/1001"]["out"]) / "FAIL_903.json").write_text(json.dumps({"status": "FAIL"}))   # a FAIL record is sticky
    res, _ = an.eval_noise(an.Inputs(), group, sets_units, _families(an, an.Inputs(), group, sets_units), {}, [])
    assert res["units"]["MI_native/1001"]["status"] == "FAIL (no result used): FAIL_903"
    shutil.rmtree(tmp_path / "results")
