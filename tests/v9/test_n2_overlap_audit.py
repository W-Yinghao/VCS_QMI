"""Tests for scripts/v9/n2_overlap_audit.py (review item U07): the counting logic on a hand-built example with known answers, detection of
links the real code never makes (a Q donor inside U, a Q link across the split), the reproduction of the pipeline's held-out split and Q
re-pairing against the pipeline's own compute_scores (teacher stubbed), and the CLI end to end on a synthetic cell.  CPU only, no data.
The last two need src/v9/n2 (the repository, or V9_N2_SRC) and torch; they are skipped otherwise."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("n2_overlap_audit", ROOT / "scripts" / "v9" / "n2_overlap_audit.py")
AUD = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(AUD)

# U images 0..7; held-out (independent batch) H = {4, 5, 7}; estimation E = {0, 1, 2, 3, 6}.  Rows: (pair, observed image, caption-source image).
U_PAIRS = [("p0", "0", "0"), ("p1", "0", "4"), ("p2", "4", "1"), ("p3", "1", "2"), ("p4", "5", "5"),
           ("p5", "2", "3"), ("p6", "3", "3"), ("p7", "5", "4"), ("p8", "6", "6"), ("p9", "7", "7")]
HELD = {"4", "5", "7"}
REF_Q = [("reference_moments", "r1", "r2"), ("reference_moments", "r2", "r3"), ("reference_moments", "r3", "r1")]
ANCHORS = {"r1": "reference_moments", "r2": "reference_moments", "r3": "reference_moments", "t1": "teacher_warmup"}


def test_counts_on_hand_built_example():
    r = AUD.count_overlap(U_PAIRS, HELD, REF_Q, ANCHORS)
    g = r["groups"]
    assert g["E"] == {"n_pairs": 6, "n_images": 5, "n_mismatched_pairs": 3} and g["H"] == {"n_pairs": 4, "n_images": 3, "n_mismatched_pairs": 2}
    lk = r["caption_source_links"]                                          # p2: H<-E, p1: E<-H, p7: H<-H, p3 + p5: E<-E
    assert (lk["H<-E"], lk["E<-H"], lk["H<-H"], lk["E<-E"], lk["source_outside_U_groups"], lk["cross_group_total"], lk["n_mismatched_total"]) == (1, 1, 1, 2, 0, 2, 5)
    assert lk["cross_share_of_mismatched"] == 2 / 5
    b = r["images_in_both_groups"]                                          # image 4 (H) has a caption in E (p1); image 1 (E) has one in H (p2)
    assert (b["total"], b["H_images_with_a_caption_in_E"], b["E_images_with_a_caption_in_H"]) == (2, 1, 1)
    assert (b["H_clusters_holding_an_E_caption"], b["E_clusters_holding_an_H_caption"], b["images_on_any_cross_link"]) == (1, 1, 3)   # clusters 4 / 0; {0, 1, 4}
    gr = r["graph"]                                                         # components {0..5} (spans H and E), {6}, {7}, {r1, r2, r3}, {t1}
    assert (gr["n_nodes"], gr["n_link_edges"], gr["n_components"], gr["n_components_with_U_images"], gr["n_singleton_U_images"]) == (12, 8, 5, 3, 2)
    big = {"n_images": 6, "n_H_images": 2, "n_E_images": 4, "n_anchor_images": 0, "n_U_pairs_H": 3, "n_U_pairs_E": 5}
    assert gr["largest_component"] == big and gr["largest_component_with_U_images"] == big
    assert gr["n_components_spanning_both_groups"] == 1 and gr["spanning_components"] == {k: v for k, v in big.items() if k != "n_images"}
    assert gr["H_images_in_a_component_with_another_H_image"] == 2 and gr["n_components_with_2plus_H_images"] == 1
    assert r["q_donors"]["reference_moments"] == {"n_rows": 3, "donor_in_same_role": 3, "donor_in_other_anchor_role": 0, "donor_in_U": 0, "crossing_H_E": 0,
                                                  "receiver_outside_role": 0, "self_donor": 0, "n_distinct_donor_images": 3, "n_receiving_images": 3}
    pv = r["provenance_units"]                                              # H call: r1..r3 (P and Q) + image 4 (clusters R4, R5); E call: r1..r3 + images 2, 3
    assert pv["call_H"] == {"n_clusters": {"P": 3, "Q": 3, "R": 3}, "units_in_more_than_one_cluster": 4, "clusters_sharing_a_unit": {"P": 3, "Q": 3, "R": 2}}
    assert pv["call_E"] == {"n_clusters": {"P": 3, "Q": 3, "R": 5}, "units_in_more_than_one_cluster": 5, "clusters_sharing_a_unit": {"P": 3, "Q": 3, "R": 3}}
    assert pv["shared_between_calls"] == {"identical_P_clusters": 3, "identical_Q_clusters": 3, "R_units_in_both_calls": 2}


def test_clean_split_has_no_links():
    clean = [(p, o, o) for p, o, _ in U_PAIRS]                              # rho 0: every caption on its own image
    r = AUD.count_overlap(clean, HELD, REF_Q, ANCHORS)
    assert r["caption_source_links"]["cross_group_total"] == 0 and r["images_in_both_groups"]["total"] == 0
    assert r["graph"]["n_components_spanning_both_groups"] == 0 and r["graph"]["largest_component_with_U_images"]["n_images"] == 1
    assert r["graph"]["largest_component"]["n_anchor_images"] == 3            # the reference Q component
    assert r["provenance_units"]["call_H"]["clusters_sharing_a_unit"]["R"] == 0 and r["provenance_units"]["call_H"]["clusters_sharing_a_unit"]["Q"] == 3


def test_links_the_pipeline_never_makes_are_detected():
    q = REF_Q + [("reference_moments", "r1", "4"), ("critic_fit", "6", "7")]   # a reference donor inside U (H) and a Q link from E to H
    r = AUD.count_overlap(U_PAIRS, HELD, q, ANCHORS)
    qr, qf = r["q_donors"]["reference_moments"], r["q_donors"]["critic_fit"]
    assert (qr["n_rows"], qr["donor_in_same_role"], qr["donor_in_U"], qr["crossing_H_E"]) == (4, 3, 1, 0)
    assert (qf["n_rows"], qf["donor_in_U"], qf["crossing_H_E"], qf["receiver_outside_role"]) == (1, 1, 1, 1)
    gr = r["graph"]                                                         # {0..5} + {r1, r2, r3} merge through r1 - 4; {6, 7} becomes a spanning component
    assert gr["n_components_spanning_both_groups"] == 2 and gr["largest_component"]["n_images"] == 9 and gr["largest_component"]["n_anchor_images"] == 3
    assert (gr["spanning_components"]["n_H_images"], gr["spanning_components"]["n_E_images"]) == (3, 5)


def test_heldout_rule_is_seeded_and_sized():
    imgs = [str(i) for i in range(100) for _ in range(5)]
    h1, h2, h3 = AUD.heldout_images(imgs, 901), AUD.heldout_images(imgs, 901), AUD.heldout_images(imgs, 902)
    assert h1 == h2 and h1 != h3 and len(h1) == 20 and h1 <= set(imgs) and AUD.heldout_images(["1", "2", "3", "4"], 901) == set()
    assert AUD.caption_source("144999") == "28999" and AUD.caption_source(5) == "1"


def _pipeline_src():
    for cand in (os.environ.get("V9_N2_SRC"), str(ROOT / "src")):
        if cand and (Path(cand) / "v9" / "n2" / "pipeline.py").exists():
            return Path(cand)
    return None


def _import_src():
    src = _pipeline_src()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


@pytest.mark.skipif(_pipeline_src() is None, reason="needs src/v9/n2 (repository) or V9_N2_SRC")
@pytest.mark.parametrize("rho", [0.4, 0.0])
def test_reproduction_equals_pipeline_code(rho):
    pytest.importorskip("torch")
    src = _import_src()
    from v9.n2 import noise as NZ, roles as RL
    n = 60; cap_img = {c: c // 5 for c in range(5 * n)}                   # the precomp layout: caption c belongs to image c // 5
    roles = RL.assign_roles([str(i) for i in range(n)], 5, 0.5, min_sizes={k: 1 for k in RL.ANCHOR_ROLES})
    pairs = NZ.make_pair_table(cap_img, RL.role_of_image(roles)); obs, labels, man = NZ.inject_mismatch(pairs, rho, 5)
    U = [o for o in obs if o["role"] == "U"]
    held = AUD.heldout_images([o["observed_image_id"] for o in U], roles["seed"]); est = np.array([o["observed_image_id"] not in held for o in U])
    q = AUD.q_rows_by_role(obs, roles["seed"])
    xc = AUD.pipeline_crosscheck(src, obs, roles, held, est, q)
    assert xc["status"] == "ran" and xc["all_equal"], xc
    assert len(held) == round(0.2 * len({o["observed_image_id"] for o in U})) and 0 < int((~est).sum()) < len(U)
    assert all(AUD.caption_source(o["caption_id"]) == labels[o["pair_id"]]["original_image_id"] for o in obs)   # rule = the pair table's own image
    assert sum(o["observed_image_id"] != AUD.caption_source(o["caption_id"]) for o in obs) == man["n_mismatched"]


@pytest.mark.skipif(_pipeline_src() is None, reason="needs src/v9/n2 (repository) or V9_N2_SRC")
def test_cli_end_to_end_on_a_synthetic_cell(tmp_path):
    src = _import_src()
    from v9.n2 import noise as NZ, roles as RL
    n = 60; cap_img = {c: c // 5 for c in range(5 * n)}
    roles = RL.assign_roles([str(i) for i in range(n)], 7, 0.5, min_sizes={k: 1 for k in RL.ANCHOR_ROLES})
    cell = tmp_path / "fixture" / "rho0.4" / "s7"; cell.mkdir(parents=True); RL.write_roles(roles, cell / "roles.json")
    pairs = NZ.make_pair_table(cap_img, RL.role_of_image(roles)); obs, lab, man = NZ.inject_mismatch(pairs, 0.4, 7); NZ.write_noise(obs, lab, man, cell, "pairs")
    out = tmp_path / "audit"
    rc = AUD.main(["--out-root", str(tmp_path), "--dataset", "fixture", "--rhos", "0.4", "--seeds", "7", "--out-dir", str(out), "--pipeline-src", str(src)])
    res = json.loads((out / "overlap_audit.json").read_text()); c = res["cells"]["rho0.4/s7"]
    assert rc == 0 and c["all_input_checks_pass"] and c["pipeline_crosscheck"].get("all_equal", True) and res["meta"]["eval_only_opened"] is False
    lk = c["caption_source_links"]
    assert lk["n_mismatched_total"] == man["n_mismatched"] and lk["H<-E"] == lk["E<-H"]                 # permutation among the selected pairs
    assert c["groups"]["H"]["n_images"] == round(0.2 * roles["n_U"]) and c["heldout"]["saved_list_files"] == []
    assert (out / "OVERLAP_AUDIT.md").read_text().startswith("# V9-N2 overlap audit")
