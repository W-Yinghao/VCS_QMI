"""EPFL post-hoc hardware diagnostics: the RTX-vs-L40S replicates (EPFL_HW_REPLICATE_20261020) and their L40S-vs-L40S control
(EPFL_HW_CONTROL_20261020).  The release checks (L40S only, the version's units, own results root never inside the primary one nor the
replicates' root, the parent CAL selection pinned, the primary COMPLETE.json hashes pinned, the expected primary GPU), the runner path (CAL
selection read from the parent results, the diagnostic label in the identity, no write into the primary results), and the analysis (per
registration, unit-macro differences with the families' bootstrap draws, agreement, pending state, the side-by-side |delta| comparison).
CPU; reads result JSON / JSONL only, no image."""
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from v10.epfl import hardware as HW, production as PR

REPO = PR.REPO
PARENT = PR.PROJECT / "handoff" / "v10_epfl_production_20261019"
HW2 = PR.PROJECT / "handoff" / "v10_epfl_production_hw2_20261020"
RTX = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
L40S = "NVIDIA L40S"
OWNER = {"2026-10-19": "final 放开，CAL 留 L40S", "2026-10-20": "什么快什么来吧"}
CONTROL_SOURCE_COMMIT = "095ffcb9238d037a29af19b882c34fdad7cbafb8"                      # the control release 5247ce5c (built from it)
CONTROL_RELEASE_SHA256 = "5247ce5c0407e77ad3d40fe8dd4760801cb88f4410ecac5da355ffe197d96920"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"epfl_{name}", REPO / "scripts" / "v10_epfl" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def _group():
    return json.loads((REPO / "v10" / "epfl" / "EPFL_GROUP_RECORD_V2_FROZEN.json").read_text())


def test_replicate_rule_and_records():
    assert HW.REPLICATE_UNITS == {"MF_VCS_M1": [1001], "MLP_VCS_perm8": [1001]}
    assert HW.REPLICATE_PRIMARY == {"MF_VCS_M1/1001": "1050507", "MLP_VCS_perm8/1001": "1050524"}
    assert (HW.REPLICATE_PARTITION, HW.REPLICATE_EXCLUDE) == ("L40S", "node51,node60") and HW.CAL_GPU == L40S
    assert sha(HW2 / "RELEASE.json") == HW.HW2_RELEASE_SHA256 and sha(HW2 / "hw_replicates" / "RELEASE.json") == HW.REPLICATE_RELEASE_SHA256
    rec = json.loads(HW.REPLICATE_RECORD.read_text())
    assert rec["owner_rules_verbatim"] == OWNER and rec["label"].startswith(HW.REPLICATE_LABEL) and "post hoc" in rec["label"]
    assert rec["request_verbatim"].startswith("Two EPFL finals ran on RTX PRO 6000") and rec["units"] == {k: list(v) for k, v in HW.REPLICATE_UNITS.items()}
    for fam in HW.REPLICATE_UNITS:
        assert rec["motivating_values_DEV72_primary_unit_macro"][fam]["1001"]["gpu_name"] == RTX
    review = json.loads(HW.DELTA_REVIEW_REPLICATE.read_text())
    sci = HW.science_identity()
    assert review["science_identity"] == {f: v[0] for f, v in sci.items()} and review["science_files_byte_identical"] is True
    assert review["science_identity_equal_to_base_release"] and review["numerics_identity_equal_to_base_release"]
    assert review["numerics_identical_to_item5_gate_commit"] and review["base_source_commit"] == HW.HW2_SOURCE_COMMIT
    upto = subprocess.run(["git", "-C", str(REPO), "diff", "--name-only", HW.HW2_SOURCE_COMMIT, HW.REPLICATE_SOURCE_COMMIT],
                          capture_output=True, text=True).stdout.split()
    assert upto and set(upto) <= set(review["changed_files_since_base_source"])                  # every change up to ca2bc60a's source
    assert all(f not in sci for f in review["changed_files_since_base_source"])                   # none is a science file


def test_control_rule_and_records():
    assert HW.CONTROL_UNITS == {"MF_VCS_M1": [1002], "MLP_VCS_perm8": [1002]}
    assert HW.CONTROL_PRIMARY == {"MF_VCS_M1/1002": "1050508", "MLP_VCS_perm8/1002": "1050531"}
    assert set(HW.DIAGNOSTICS) == {HW.REPLICATE_SCHEMA, HW.CONTROL_SCHEMA}
    assert HW.DIAGNOSTICS[HW.CONTROL_SCHEMA]["controls_for"] == HW.REPLICATE_RELEASE_SHA256
    rec = json.loads(HW.CONTROL_RECORD.read_text())
    assert rec["owner_rules_verbatim"] == OWNER and rec["label"].startswith(HW.REPLICATE_LABEL) and "L40S-VS-L40S CONTROL" in rec["label"]
    assert "Make no claim beyond those numbers." in rec["request_verbatim"] and rec["units"] == {k: list(v) for k, v in HW.CONTROL_UNITS.items()}
    assert rec["controls_for"]["release_sha256"] == HW.REPLICATE_RELEASE_SHA256
    assert all(v["gpu_name"] == L40S and v["job"] == HW.CONTROL_PRIMARY[k] for k, v in rec["primary"].items())
    review = json.loads(HW.DELTA_REVIEW_CONTROL.read_text())
    sci = HW.science_identity()
    assert review["science_identity"] == {f: v[0] for f, v in sci.items()} and review["science_identity_equal_to_replicate_release"]
    assert review["science_identity_equal_to_base_release"] and review["numerics_identical_to_item5_gate_commit"]
    assert review["base_source_commit"] == HW.REPLICATE_SOURCE_COMMIT
    upto = subprocess.run(["git", "-C", str(REPO), "diff", "--name-only", HW.REPLICATE_SOURCE_COMMIT, CONTROL_SOURCE_COMMIT],
                          capture_output=True, text=True).stdout.split()
    assert upto and set(upto) <= set(review["changed_files_since_base_source"])                  # every change up to the control's source
    assert all(f not in sci for f in review["changed_files_since_base_source"])
    assert sha(HW2 / "hw_controls" / "RELEASE.json") == CONTROL_RELEASE_SHA256
    assert json.loads((HW2 / "hw_controls" / "RELEASE.json").read_text())["source_commit"] == CONTROL_SOURCE_COMMIT


def fixture_diag(tmp_path, schema):
    """A diagnostic release of the given schema in tmp_path/<rep|ctl>/ built like release_hw.build_diagnostic (current HEAD and inventory)."""
    spec = HW.DIAGNOSTICS[schema]; rh = load_script("release_hw")
    bundle = tmp_path / ("rep" if schema == HW.REPLICATE_SCHEMA else "ctl"); bundle.mkdir(exist_ok=True)
    par = json.loads((PARENT / "RELEASE.json").read_text()); parent_results = Path(par["results_dir"])
    inv = PR.source_inventory()
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    pins, pairs, fams = {}, {}, {}
    for fam, seeds in spec["units"].items():
        fams[fam] = copy.deepcopy(par["families"][fam])
        for s in seeds:
            c = json.loads((parent_results / "FINAL_DEV" / fam / f"seed_{s}" / "COMPLETE.json").read_text())
            pins[fam] = c["identity"]["selection_dependency_sha256"]
            pairs[f"{fam}/{s}"] = {**rh.primary_pair(parent_results, fam, s, pins[fam], c["identity"]["candidate"], spec),
                                   "replicate_unit": str(bundle / "results" / "FINAL_DEV" / fam / f"seed_{s}"), "replicate_gpu": L40S,
                                   "time_limit_l40s": par["families"][fam]["time_limits"]["final"]}
    controls = {"controls_for": {"path": str(HW2 / "hw_replicates" / "RELEASE.json"), "sha256": spec["controls_for"]}} if spec["controls_for"] else {}
    rel = {"schema": schema, "status": HW.RELEASE_STATUS_HW, "role": spec["role"], "label": HW.REPLICATE_LABEL + ": test", "source_commit": head,
           "source_root": str(REPO), "source_files": inv, "code_sha256": PR.digest(inv), "numerics_identity": PR.numerics_identity(),
           "science_identity": HW.science_identity(), **controls,
           "parent_release": {"path": str(PARENT / "RELEASE.json"), "sha256": HW.PARENT_RELEASE_SHA256},
           "base_release": {"path": str(HW2 / "RELEASE.json"), "sha256": HW.HW2_RELEASE_SHA256},
           "cal_release": {"path": str(PARENT / "RELEASE.json"), "sha256": HW.PARENT_RELEASE_SHA256},
           "results_root": str(bundle / "results"), "cal_results_root": str(parent_results),
           **{k: par[k] for k in ("frozen_protocol", "activation_record", "group_record", "data_lock", "item5_gate", "item5_timing", "delta_review")},
           **{k: {"path": str(p.relative_to(REPO)), "sha256": sha(p)} for k, p in
              (("hardware_amendment", HW.AMENDMENT), ("xhw_gate_amendment", HW.XHW_AMENDMENT), (spec["record_key"], spec["record"]),
               (spec["review_key"], spec["review"]))},
           "phases": ["FINAL_DEV"], "final_gpus": [L40S], "partition": "L40S", "exclude": "node51,node60",
           "units": {k: list(v) for k, v in spec["units"].items()}, "families": fams, "selection_pins": pins, "pairs": pairs,
           "owner_rules_verbatim": OWNER}
    p = bundle / "RELEASE.json"; p.write_text(json.dumps(rel)); return p, rel


def test_primary_pairs_are_the_expected_finals(tmp_path):
    hw2 = json.loads((HW2 / "RELEASE.json").read_text())
    for schema, gpu in ((HW.REPLICATE_SCHEMA, RTX), (HW.CONTROL_SCHEMA, L40S)):
        p, rel = fixture_diag(tmp_path, schema)
        for key, pair in rel["pairs"].items():
            assert pair["primary_job"] == HW.DIAGNOSTICS[schema]["primary"][key] and pair["primary_gpu"] == gpu
            assert pair["primary_release_sha256"] == HW.HW2_RELEASE_SHA256 and pair["primary_complete_sha256"] == sha(pair["primary_complete"])
            assert pair["selection_dependency_sha256"] == rel["selection_pins"][pair["family"]]
        assert rel["selection_pins"]["MLP_VCS_perm8"] == hw2["selection_pins"]["MLP_VCS_perm8"]   # the same CAL selection as d9d67cf2
    rh = load_script("release_hw")
    parent_results = PARENT / "results"
    c = json.loads((parent_results / "FINAL_DEV" / "MF_VCS_M1" / "seed_1001" / "COMPLETE.json").read_text())
    with pytest.raises(RuntimeError):                                       # an RTX unit is never accepted as an L40S control primary
        rh.primary_pair(parent_results, "MF_VCS_M1", 1001, c["identity"]["selection_dependency_sha256"], c["identity"]["candidate"],
                        {**HW.DIAGNOSTICS[HW.CONTROL_SCHEMA], "primary": {"MF_VCS_M1/1001": "1050507"}})


def _accepts_and_refuses(tmp_path, schema, units, other_units, mutations):
    from v10.epfl import host as H
    p, rel = fixture_diag(tmp_path, schema)
    kw = {"candidates_fn": H.production_candidates, "schedule_fn": H.production_schedule}
    for fam, seeds in units.items():
        assert HW.verify_release_replicate(p, sha(p), fam, seeds[0], **kw)["schema"] == schema
    fam0, seed0 = next(iter(units)), next(iter(units.values()))[0]
    with pytest.raises(RuntimeError, match="finals only"):
        HW.verify_release_replicate(p, sha(p), fam0, seed0, phase="cal", **kw)
    for fam, seed in other_units:
        with pytest.raises(RuntimeError, match="not a unit of this diagnostic release"):
            HW.verify_release_replicate(p, sha(p), fam, seed, **kw)
    primary = rel["cal_results_root"]
    common = (lambda r: r.update(results_root=primary), lambda r: r.update(results_root=primary + "/FINAL_DEV"),
              lambda r: r.update(cal_results_root=r["results_root"]), lambda r: r.update(final_gpus=list(HW.FINAL_GPUS)),
              lambda r: r.update(partition="RTX6000PRO,L40S,H100"), lambda r: r.update(exclude="node51,node53,node60"),
              lambda r: r["units"][fam0].append(1003), lambda r: r["pairs"][f"{fam0}/{seed0}"].update(primary_complete_sha256="0" * 64),
              lambda r: r["selection_pins"].update({fam0: "0" * 64}), lambda r: r.update(label="hardware replicate"),
              lambda r: r.update(role="OTHER"), lambda r: r["base_release"].update(sha256="0" * 64),
              lambda r: r.update(schema=HW.RELEASE_SCHEMA_HW), lambda r: r["families"][fam0].update(candidates=[]),
              lambda r: r["science_identity"].update({"src/v10/adapter.py": ["x", "x"]}), lambda r: r.update(source_commit="deadbeef"),
              lambda r: r["pairs"].pop(f"{fam0}/{seed0}"))
    for mutate in common + tuple(mutations):
        r = copy.deepcopy(rel); mutate(r); p.write_text(json.dumps(r))
        with pytest.raises(RuntimeError):
            HW.verify_release_replicate(p, sha(p), fam0, seed0, **kw)
    p.write_text(json.dumps(rel)); (p.parent / "PAUSE").write_text("")
    with pytest.raises(RuntimeError, match="paused"):
        HW.verify_release_replicate(p, sha(p), fam0, seed0, **kw)


def test_replicate_release_accepts_and_refuses(tmp_path):
    _accepts_and_refuses(tmp_path, HW.REPLICATE_SCHEMA, HW.REPLICATE_UNITS, (("MF_VCS_M1", 1002), ("MF_VCS_perm1", 1001), ("MLP_VCS_perm8", 1003)),
                         (lambda r: r["pairs"]["MLP_VCS_perm8/1001"].update(primary_job="1050531"),
                          lambda r: r["pairs"]["MF_VCS_M1/1001"].update(primary_gpu=L40S),
                          lambda r: r["replicate_record"].update(sha256="0" * 64)))


def test_control_release_accepts_and_refuses(tmp_path):
    rep_root = json.loads((HW2 / "hw_replicates" / "RELEASE.json").read_text())["results_root"]
    _accepts_and_refuses(tmp_path, HW.CONTROL_SCHEMA, HW.CONTROL_UNITS, (("MF_VCS_M1", 1001), ("MLP_VCS_perm8", 1001), ("MLP_VCS_perm8", 1003)),
                         (lambda r: r["pairs"]["MLP_VCS_perm8/1002"].update(primary_job="1050539"),
                          lambda r: r["pairs"]["MF_VCS_M1/1002"].update(primary_gpu=RTX),
                          lambda r: r["controls_for"].update(sha256="0" * 64), lambda r: r.pop("controls_for"),
                          lambda r: r.update(results_root=rep_root), lambda r: r.update(schema=HW.REPLICATE_SCHEMA),
                          lambda r: r.update(units={k: list(v) for k, v in HW.REPLICATE_UNITS.items()}),
                          lambda r: r["control_record"].update(sha256="0" * 64), lambda r: r.pop("delta_review_control")))


class _Ctx:
    def __init__(self, hardware, replicate=None):
        self.hardware, self.replicate, self.schedule_fn = hardware, replicate, (lambda f: {"steps": 1})


def test_runner_rerun_path(tmp_path, monkeypatch):
    run = load_script("run")
    seen = {}

    def sel(root, fam, rsha, d, g, ctx, require_exit=True):
        seen.update(root=Path(root), rsha=rsha, require_exit=require_exit)
        return {"selected": {"candidate": {"order": 0}}}, "SELSHA"
    monkeypatch.setattr(run, "checked_selection", sel)
    monkeypatch.setattr(run, "run_unit", lambda ctx, unit, ident, alloc, env: {"unit": Path(unit), "identity": ident})
    label = {"kind": HW.REPLICATE_LABEL, "role": "L40S_VS_L40S_CONTROL", "replicate_of": {"job": "1050508"}}
    l40s = {"gpu_name": L40S, "node": "node50"}
    primary, own = tmp_path / "primary" / "results", tmp_path / "hw_controls" / "results"
    out = run.run_final(_Ctx(l40s, label), own, "MF_VCS_M1", 1002, "CTLREL", "d", "g", None, {}, cal_release_sha="PARENT",
                        selection_pin="SELSHA", cal_root=primary)
    assert seen == {"root": primary, "rsha": "PARENT", "require_exit": True}                       # CAL selection read from the parent
    assert out["unit"] == own / "FINAL_DEV" / "MF_VCS_M1" / "seed_1002"                             # the unit lives in the rerun root
    ident = out["identity"]
    assert ident["diagnostic"] == label and ident["hardware"] == {"gpu_name": L40S} and ident["release_sha256"] == "CTLREL"
    assert ident["selection_dependency_sha256"] == "SELSHA" and ident["seed"] == 1002
    for root, cal_root in ((primary, primary), (primary / "inner", primary), (own, None)):            # never into the primary results
        with pytest.raises(RuntimeError):
            run.run_final(_Ctx(l40s, label), root, "MF_VCS_M1", 1002, "CTLREL", "d", "g", None, {}, cal_release_sha="PARENT", cal_root=cal_root)
    with pytest.raises(RuntimeError, match="pinned"):
        run.run_final(_Ctx(l40s, label), own, "MF_VCS_M1", 1002, "CTLREL", "d", "g", None, {}, cal_release_sha="PARENT", selection_pin="OTHER",
                      cal_root=primary)
    plain = run.run_final(_Ctx(l40s), primary, "MF_VCS_M1", 1002, "HWREL", "d", "g", None, {}, cal_release_sha="PARENT")
    assert "diagnostic" not in plain["identity"]                                                    # the hardware release path is unchanged


def _rows(fam, seed):
    return [json.loads(x) for x in (PARENT / "results" / "FINAL_DEV" / fam / f"seed_{seed}" / "DEV.jsonl").read_text().splitlines() if x.strip()]


def test_pair_functions_on_result_rows():
    import numpy as np
    an = load_script("analyze")
    group = _group()
    sets_units = {n: sorted(group["sensitivity_sets"][n]["units"]) for n in an.SETS}
    draws = an.boot_draws(sets_units)
    first = an.SETS[0]
    assert np.array_equal(draws[first], np.random.default_rng(an.BOOT_SEED).integers(0, len(sets_units[first]), size=(an.B_BOOT, len(sets_units[first]))))
    a = _rows("MF_VCS_M1", 1001)
    ag = an.agreement(an.per_registration("MF_VCS_M1", 1001, a, copy.deepcopy(a)))
    assert ag["n_registrations"] == 740 and ag["estimate_bitwise_equal"] == 740 and ag["delta_err_full"]["max_abs"] == 0.0
    assert ag["success42"]["agreement"] == 1.0 and ag["success9"]["primary_only"] == 0 and ag["success9"]["rerun_only"] == 0 and ag["success9"]["disagree"] == 0
    st = an.set_stats(a, group["sensitivity_sets"][first], group["window_unit"])["_units"]
    d0 = an.unit_macro_difference(st, st, sets_units[first], draws[first])
    assert d0["difference_rerun_minus_primary"] == {"success42": 0.0, "success9": 0.0, "median_err_full": 0.0} == d0["abs_difference"]
    assert d0["bootstrap95"]["success9"] == [0.0, 0.0] and d0["units_differing"]["success9"] == 0
    b = _rows("MF_VCS_M1", 1002)          # another seed: exercises the arithmetic (not a hardware pair)
    rows = an.per_registration("MF_VCS_M1", 1001, a, b)
    ag = an.agreement(rows)
    t9 = ag["success9"]
    assert t9["both"] + t9["neither"] + t9["primary_only"] + t9["rerun_only"] == 740 and t9["disagree"] == t9["primary_only"] + t9["rerun_only"]
    assert ag["estimate_bitwise_equal"] < 740
    sb = an.set_stats(b, group["sensitivity_sets"][first], group["window_unit"])
    d = an.unit_macro_difference(st, sb["_units"], sets_units[first], draws[first])
    sa = an.set_stats(a, group["sensitivity_sets"][first], group["window_unit"])
    for k in ("success42", "success9", "median_err_full"):
        assert abs(d["difference_rerun_minus_primary"][k] - (sb["unit_macro"][k] - sa["unit_macro"][k])) < 1e-12
        assert d["abs_difference"][k] == abs(d["difference_rerun_minus_primary"][k]) and d["bootstrap95"][k][0] <= d["bootstrap95"][k][1]
    assert abs(sum(r["success9_rerun"] for r in rows) / 740 - sb["registration_level"]["success9"]) < 1e-12
    bad = copy.deepcopy(b); bad[0]["theta"] += 1.0
    with pytest.raises(RuntimeError, match="truth differs"):
        an.per_registration("MF_VCS_M1", 1001, a, bad)


def _diag_bundle(tmp_path, monkeypatch, an, schema, complete, source_seed=None):
    """A fixture diagnostic root and, if complete, rerun units made of a primary unit's JSON files (no checkpoint archive is copied) with the
    rerun identity: the unit's own files (a stand-in for a bitwise-identical rerun) or, with source_seed, another seed's files (a stand-in
    for a rerun that differs)."""
    p, rel = fixture_diag(tmp_path, schema)
    monkeypatch.setattr(an, "REPLICATES" if schema == HW.REPLICATE_SCHEMA else "CONTROLS", p.parent)
    if complete:
        rsha = sha(p)
        for key, pair in rel["pairs"].items():
            prim = Path(pair["primary_unit"]); src = prim.parent / f"seed_{source_seed}" if source_seed else prim
            dst = Path(pair["replicate_unit"]); (dst / "checkpoints").mkdir(parents=True)
            for f in ("DEV.jsonl", "TRAIN_STEPS.jsonl"):
                shutil.copy(src / f, dst / f)
            for f in (src / "checkpoints").glob("step_*.json"):
                shutil.copy(f, dst / "checkpoints" / f.name)
            c = json.loads((src / "COMPLETE.json").read_text())
            c["identity"] = {**json.loads((prim / "COMPLETE.json").read_text())["identity"], "release_sha256": rsha, "hardware": {"gpu_name": L40S},
                             "diagnostic": {"kind": HW.REPLICATE_LABEL, "role": HW.DIAGNOSTICS[schema]["role"], "replicate_of": {"job": pair["primary_job"]}}}
            c["hardware"] = {"gpu_name": L40S, "node": "node50", "partition": "L40S"}; c["job"] = "9" + pair["primary_job"]
            (dst / "COMPLETE.json").write_text(json.dumps(c))
            (p.parent / "exits").mkdir(exist_ok=True)
            (p.parent / "exits" / f"{c['job']}.json").write_text(json.dumps({"job": c["job"], "exit_code": 0, "allocated_seconds": 3600.0, "gpu_name": L40S}))
    return rel


def _families(an, inp, group):
    out = {}
    for fam in HW.REPLICATE_UNITS:
        finals = {s: an.final_record(inp, PARENT / "results" / "FINAL_DEV" / fam / f"seed_{s}", group, {}, "x") for s in (1001, 1002, 1003)}
        out[fam] = {"finals": {str(s): an.strip(f) for s, f in finals.items()}, "strata": an.strata(finals)}
    return out


def test_analysis_pending(tmp_path, monkeypatch):
    an = load_script("analyze"); group = _group()
    sets_units = {n: sorted(group["sensitivity_sets"][n]["units"]) for n in an.SETS}
    for schema, fn in ((HW.REPLICATE_SCHEMA, an.replicate_pairs), (HW.CONTROL_SCHEMA, an.control_pairs)):
        rel = _diag_bundle(tmp_path, monkeypatch, an, schema, complete=False)
        problems = []
        res, files = fn(an.Inputs(), group, sets_units, {}, problems)
        assert res["status"] == "PENDING" and files == {} and not problems and res["role"] == HW.DIAGNOSTICS[schema]["role"]
        assert all(v["status"].startswith("PENDING") for v in res["pairs"].values()) and sorted(res["pairs"]) == sorted(rel["pairs"])
        assert res["owner_rules_verbatim"] == OWNER
    cmp_ = an.replicate_vs_control(*(fn(an.Inputs(), group, sets_units, {}, [])[0] for fn in (an.replicate_pairs, an.control_pairs)))
    assert set(cmp_["families"]) == set(HW.REPLICATE_UNITS) and all(set(v) == {"rtx_vs_l40s", "l40s_vs_l40s"} for v in cmp_["families"].values())
    assert all("abs_unit_macro_difference" not in r for v in cmp_["families"].values() for r in v.values())
    assert "no claim beyond these numbers" in cmp_["statement"]
    nr = an.replicate_vs_control({"status": "NO_RELEASE"}, {"status": "NO_RELEASE"})
    assert nr["families"] == {}


def test_analysis_complete_pairs_and_side_by_side(tmp_path, monkeypatch):
    an = load_script("analyze"); group = _group()
    sets_units = {n: sorted(group["sensitivity_sets"][n]["units"]) for n in an.SETS}
    rel_rep = _diag_bundle(tmp_path, monkeypatch, an, HW.REPLICATE_SCHEMA, complete=True)               # bitwise-identical stand-in
    rel_ctl = _diag_bundle(tmp_path, monkeypatch, an, HW.CONTROL_SCHEMA, complete=True, source_seed=1003)  # differing stand-in
    inp = an.Inputs(); problems = []
    fams = _families(an, inp, group)
    rep, rep_files = an.replicate_pairs(inp, group, sets_units, fams, problems)
    ctl, ctl_files = an.control_pairs(inp, group, sets_units, fams, problems)
    assert rep["status"] == "COMPLETE" and ctl["status"] == "COMPLETE" and not problems, problems
    for key, v in rep["pairs"].items():
        assert all(v["identity_checks"]["same_unit_fields"].values()) and v["identity_checks"]["gpu_l40s"]
        assert v["per_registration_agreement"]["estimate_bitwise_equal"] == 740 and v["training_log"]["first_step_loss_differs"] is None
        assert all(c["model_sha256_equal"] for c in v["cal_trajectory"]) and v["selected_step"]["primary"] == v["selected_step"]["rerun"]
        for name in an.SETS:
            assert v["unit_macro_difference"][name]["abs_difference"] == {"success42": 0.0, "success9": 0.0, "median_err_full": 0.0}
        ctx = v["seed_context_DEV72_primary"]
        assert set(ctx["l40s_seeds"]) == {"1001_rerun", "1002", "1003"}
        assert abs(ctx["primary_minus_mean_of_primary_l40s_seeds"]["success9"] - ctx["rerun_minus_mean_of_primary_l40s_seeds"]["success9"]) < 1e-12
        f = v["per_registration_file"]
        assert f["rows"] == 740 and hashlib.sha256(rep_files[f["path"]].encode()).hexdigest() == f["sha256"] and Path(f["path"]).parent == tmp_path / "rep" / "analysis"
    first = an.SETS[0]
    for key, v in ctl["pairs"].items():
        fam = v["family"]
        s2 = an.set_stats(_rows(fam, 1002), group["sensitivity_sets"][first], group["window_unit"])["unit_macro"]
        s3 = an.set_stats(_rows(fam, 1003), group["sensitivity_sets"][first], group["window_unit"])["unit_macro"]
        for k in ("success42", "success9", "median_err_full"):
            assert abs(v["unit_macro_difference"][first]["abs_difference"][k] - abs(s3[k] - s2[k])) < 1e-12
        assert "seed_context_DEV72_primary" not in v and v["training_log"]["first_step_loss_differs"] is not None
        assert Path(v["per_registration_file"]["path"]).parent == tmp_path / "ctl" / "analysis"
    cmp_ = an.replicate_vs_control(rep, ctl)
    for fam in HW.REPLICATE_UNITS:
        r, c = cmp_["families"][fam]["rtx_vs_l40s"], cmp_["families"][fam]["l40s_vs_l40s"]
        assert (r["seed"], c["seed"]) == (1001, 1002) and r["primary_gpu"] == RTX and c["primary_gpu"] == L40S
        assert r["abs_unit_macro_difference"][first] == {"success42": 0.0, "success9": 0.0, "median_err_full": 0.0}
        assert c["abs_unit_macro_difference"][first] == ctl["pairs"][f"{fam}/1002"]["unit_macro_difference"][first]["abs_difference"]
        ag = ctl["pairs"][f"{fam}/1002"]["per_registration_agreement"]
        assert c["per_registration"]["disagree9"] == ag["success9"]["disagree"] and c["per_registration"]["n"] == 740
        assert r["per_registration"]["disagree42"] == 0 and r["per_registration"]["max_abs_delta_err_full"] == 0.0
    md = an.markdown({"families": {}, "sensitivity_deltas_vs_primary": {}, "rulsif_v2": {}, "disclosures": {}, "problems": [],
                      "bootstrap": {"B": an.B_BOOT, "seed": an.BOOT_SEED}, "hardware_replicates": rep, "hardware_controls": ctl,
                      "hardware_replicate_vs_control": cmp_})
    assert "|Δ(RTX vs L40S)| 与 |Δ(L40S vs L40S)| 并列" in md and md.count("| L40S vs L40S | 1002 |") == 4
    bad_unit = Path(rel_ctl["pairs"]["MF_VCS_M1/1002"]["replicate_unit"]) / "COMPLETE.json"
    bad = json.loads(bad_unit.read_text()); bad["identity"]["hardware"] = {"gpu_name": RTX}; bad_unit.write_text(json.dumps(bad))
    problems = []
    an.control_pairs(an.Inputs(), group, sets_units, fams, problems)
    assert any("MF_VCS_M1/1002" in p for p in problems)                                              # a non-L40S rerun is flagged
    assert sorted(rel_rep["pairs"]) == ["MF_VCS_M1/1001", "MLP_VCS_perm8/1001"]


@pytest.mark.parametrize("schema,fam,seed,job", [(HW.REPLICATE_SCHEMA, "MF_VCS_M1", 1001, "1050507"),
                                                 (HW.CONTROL_SCHEMA, "MF_VCS_M1", 1002, "1050508")])
def test_production_entry_glue(tmp_path, monkeypatch, schema, fam, seed, job):
    """production() on a fixture diagnostic release up to run_final (no GPU, no data read): the L40S-only refusal, the release's own results
    root and ledger, the parent CAL root / release / pinned selection, the diagnostic label and the hardware record handed to run_final."""
    import argparse
    import time
    from v10.epfl import guard as GD, runtime as RT
    run = load_script("run")
    p, rel = fixture_diag(tmp_path, schema); bundle = p.parent
    for k, v in (("SLURM_JOB_ID", "4242"), ("V10_EPFL_JOB_START", str(time.time())), ("V10_EPFL_EXPECT_RELEASE", sha(p))):
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(run.signal, "signal", lambda *a: None)
    monkeypatch.setattr(GD.EPFLDataGuard, "verify_data_lock", lambda self, roles: None)
    seen = {}

    def final(ctx, root, family, seed, release_sha, data_sha, group_sha, allocation, env, **kw):
        seen.update(root=Path(root), family=family, seed=seed, release_sha=release_sha, replicate=ctx.replicate, hardware=ctx.hardware, **kw)
    monkeypatch.setattr(run, "run_final", final)
    gpu = {"name": L40S}
    monkeypatch.setattr(RT, "runtime_environment", lambda A: {"gpu_name": gpu["name"], "slurm_nodelist": "node50", "slurm_partition": "L40S",
                                                               "slurm_job_id": "4242"})
    assert run.production(argparse.Namespace(phase="final", family=fam, seed=seed, release=p)) == 0
    assert seen["root"] == bundle / "results" and seen["cal_root"] == Path(rel["cal_results_root"]) and seen["release_sha"] == sha(p)
    assert seen["cal_release_sha"] == HW.PARENT_RELEASE_SHA256 and seen["selection_pin"] == rel["selection_pins"][fam]
    assert seen["replicate"] == HW.replicate_label(rel, fam, seed) and seen["replicate"]["replicate_of"]["job"] == job
    assert seen["replicate"]["role"] == HW.DIAGNOSTICS[schema]["role"]
    assert seen["hardware"]["gpu_name"] == L40S and seen["hardware"]["xhw_gate"] is None
    ledger = [json.loads(x) for x in (bundle / "COST_LEDGER.jsonl").read_text().splitlines()]
    assert [r["event"] for r in ledger] == ["START", "HARDWARE", "END"] and ledger[-1]["status"] == "COMPLETE" and ledger[-1]["gpu_name"] == L40S
    assert ledger[0]["phase"] == "FINAL_DEV" and ledger[0]["job"] == "4242"                       # the release's own ledger (tmp bundle)
    gpu["name"] = RTX                                                                              # any other GPU type is refused
    seen.clear()
    with pytest.raises(RuntimeError, match="L40S only"):
        run.production(argparse.Namespace(phase="final", family="MLP_VCS_perm8", seed=seed, release=p))
    assert not seen and (bundle / "failures" / "4242.json").exists()
    with pytest.raises(RuntimeError, match="finals only"):
        run.production(argparse.Namespace(phase="cal", family=fam, seed=None, release=p))
