"""T03 constants vs protocol P2 / the owner's contract, pinned inputs, and the unit table vs an independent derivation."""
from __future__ import annotations

import collections
import json


import t03_common as C
import t03_data as D
import t03_select as S
import t03_units as U

P2 = json.loads(C.PROTOCOL_PATH.read_text(encoding="utf-8"))["t03"]
CONTRACT = json.loads((C.REPO / "v10" / "decisions" / "V10_PROTOCOL_DECISIONS_20261017.json").read_text(encoding="utf-8"))["T03"]


def test_conditions_and_sizes():
    conds = [(c["family"], c["m"], c["rho"]) for c in P2["conditions"]]
    assert conds == U.conditions()
    assert [c["case_id"] for c in P2["conditions"]] == [C.case_id(*c) for c in conds]
    assert P2["observations_per_condition"] == {"FIT": 4096, "CAL": 2048, "EVAL": 8192}
    assert {k.upper(): v for k, v in C.ROLE_SIZES.items()} == P2["observations_per_condition"]
    assert CONTRACT["conditions"] == {"families": list(C.FAMILIES), "m": list(C.MS), "rho": list(C.RHOS), "count": 12}


def test_methods_models_training():
    assert P2["methods"] == list(C.METHODS) == CONTRACT["main_families"]
    assert (P2["MF"]["K_main"], P2["MF"]["temperature"], P2["MF"]["W_initial"]) == (C.K_MAIN, C.TEMPERATURE, "zeros")
    assert CONTRACT["MLP"]["hidden_dims"] == [C.MLP_HIDDEN, C.MLP_HIDDEN] and CONTRACT["MLP"]["critic_transform"] == "tanh"
    tr = P2["training"]
    assert (tr["steps"], tr["checkpoint_every"], tr["checkpoints"]) == (C.STEPS, C.CKPT_EVERY, C.STEPS // C.CKPT_EVERY)
    assert (tr["non_W_weight_decay"], tr["W_weight_decay"], tr["ridge"], tr["clip_norm"]) == (C.NON_W_WEIGHT_DECAY, C.W_WEIGHT_DECAY, C.RIDGE, C.CLIP_NORM)
    assert tr["learning_rates"] == list(C.LEARNING_RATES) and tr["M1_caps"] == list(C.M1_CAPS) and tr["optimizer"] == "AdamW"


def test_seeds_and_capacity():
    assert (P2["tuning_suite"]["data_root_seed"], P2["tuning_suite"]["algo_seed"]) == (C.TUNING_SEED, C.TUNING_SEED)
    assert P2["replicates"]["main"]["data_root_seeds"] == list(C.MAIN_DATA_SEEDS) and P2["replicates"]["main"]["algo_seed"] == C.MAIN_ALGO_SEED
    assert P2["replicates"]["conditional_algorithm"]["algo_seeds"] == list(C.EXTRA_ALGO_SEEDS)
    assert P2["replicates"]["conditional_algorithm"]["data_root_seed"] == C.MAIN_DATA_SEEDS[0]
    assert [(c["family"], c["m"], c["rho"]) for c in P2["capacity"]["conditions"]] == list(C.CAPACITY_CONDITIONS)
    assert P2["capacity"]["methods"] == list(C.MF_METHODS) and P2["capacity"]["K"] == list(C.K_CAPACITY)
    assert P2["counts"] == {"main": 600, "conditional_algorithm_extra": 120, "capacity": 240, "final_and_diagnostic": 960, "CAL": 252,
                            "total_excluding_smoke_and_followup_baselines": 1212}


def test_pinned_inputs():
    assert P2["generator"]["sha256"] == C.GENERATOR_SHA256 == C.sha256_file(C.GENERATOR_PATH)
    man = json.loads((C.PACKAGE / "MANIFEST_SHA256.json").read_text())
    assert man["synthetic/src/v10_synthetic.py"] == C.GENERATOR_SHA256
    assert man["synthetic/tests/check_synthetic.py"] == C.CHECKER_SHA256 == C.sha256_file(C.CHECKER_PATH)
    assert P2["units_file"]["sha256"] == C.UNITS_CSV_SHA256 == C.sha256_file(C.UNITS_CSV)


def test_codes_match_generator():
    assert C.FAMILY_CODE == D.GEN.FAMILIES and C.ROLE_CODE == D.GEN.ROLES
    for f, m, r in U.conditions():
        assert C.case_id(f, m, r) == D.GEN.case_id(f, m, r)
        assert C.CASE_DIR_RE.match(C.case_id(f, m, r))


def test_units_csv_equals_independent_derivation():
    res = U.check_against_derivation()
    assert res["status"] == "PASS", res["field_differences"][:5]
    assert res["bytes_identical"] and res["rows_file"] == 1219
    rows = U.load_rows()
    assert dict(collections.Counter(r["block"] for r in rows)) == U.BLOCK_COUNTS
    assert sum(v for k, v in U.BLOCK_COUNTS.items() if k != "smoke") == 1212


def test_units_dependencies_and_candidate_order():
    rows = [U.parse(r) for r in U.load_rows()]
    by = {r["unit_id"]: r for r in rows}
    smoke = [r["unit_id"] for r in rows if r["block"] == "smoke"]
    for r in rows:
        if r["block"] == "CAL":
            assert r["depends_on"] == smoke and r["data_root_seed"] == r["algo_seed"] == 1017
        elif r["block"] in ("main", "algo_extra", "capacity"):
            deps = [by[d] for d in r["depends_on"]]
            assert all(d["block"] == "CAL" and d["method"] == r["method"] and (d["family"], d["m"], d["rho"]) == (r["family"], r["m"], r["rho"])
                       for d in deps)
            assert [S.candidate_ordinal(d["method"], d["lr"], d["cap"]) for d in deps] == [o for o, _, _ in S.candidates_for(r["method"])]
            assert all(d["K"] == C.K_MAIN for d in deps if d["method"] in C.MF_METHODS)
            assert r["lr"] == ("CAL_K16" if r["block"] == "capacity" else "CAL")
    cal = [r for r in rows if r["block"] == "CAL"]
    for i in range(0, len(cal), 21):                                  # condition -> method -> lr -> cap
        blk = cal[i:i + 21]
        assert [r["method"] for r in blk] == [m for m in C.METHODS for _ in S.candidates_for(m)]
        assert [(r["lr"], r["cap"]) for r in blk] == [(lr, cap) for m in C.METHODS for _, lr, cap in S.candidates_for(m)]
