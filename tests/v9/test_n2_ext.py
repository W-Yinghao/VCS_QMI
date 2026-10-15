"""V9-N2 extensions (owner 2026-10-15): precheck gate, loss-coefficient merge rule, independent-batch rho interval, official export, patches."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from v9.n2 import precheck as PC, weights as W  # noqa: E402

CLONES = {"NCR": ["/projects/EEG-foundation-model/yinghao/code_v9/2021-NeurIPS-NCR", "/tmp/claude-34987/-home-infres-yinwang-CS-QMI/3ef2737e-e5a7-4b67-9f0b-53c6d6caa16c/scratchpad/v9_code/NCR"],
          "L2RM": ["/projects/EEG-foundation-model/yinghao/code_v9/L2RM", "/tmp/claude-34987/-home-infres-yinwang-CS-QMI/3ef2737e-e5a7-4b67-9f0b-53c6d6caa16c/scratchpad/v9_code/L2RM"]}


def _clone(name):
    for c in CLONES[name]:
        if Path(c, ".git").exists():
            return c
    return None


def test_auroc_and_separability_thresholds():
    rng = np.random.default_rng(0)
    assert abs(PC.auroc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) - 1.0) < 1e-12 and abs(PC.auroc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) - 0.5) < 1e-12
    sep = PC.teacher_separability(rng.normal(0.7, 0.1, 500), rng.normal(0.3, 0.1, 500)); assert not sep["degenerate"] and 0.95 < sep["auroc"] < 1.0
    deg = PC.teacher_separability(np.linspace(0.9, 1.0, 200), np.linspace(0.0, 0.1, 200)); assert deg["degenerate"] and "above max" in deg["reason"]
    flat = PC.teacher_separability(rng.normal(0.5, 0.1, 500), rng.normal(0.5, 0.1, 500)); assert flat["degenerate"] and "below min" in flat["reason"]
    assert 0 <= sep["overlap_coefficient"] <= 1 and set(sep["P"]) >= {"q5", "q50", "q95"}


def test_weight_tables_pairwise():
    rng = np.random.default_rng(1); w1 = rng.uniform(0, 1, 100); w2 = w1.copy(); w3 = rng.uniform(0, 1, 100)
    t = PC.weight_tables({"A": w1, "B": w2, "C": w3})
    assert t["pairwise"]["A|B"]["share_identical_1e-6"] == 1.0 and abs(t["pairwise"]["A|B"]["spearman"] - 1.0) < 1e-12 and t["pairwise"]["A|C"]["share_identical_1e-6"] < 0.05
    assert t["per_method"]["A"]["ess"] > 0 and abs(t["per_method"]["A"]["total_mass"] - w1.sum()) < 1e-9


def test_loss_coefficient_merge_rule():
    rows = [{"pair_id": f"a{i}", "role": "critic_fit"} for i in range(5)] + [{"pair_id": f"u{i}", "role": "U"} for i in range(20)]
    base = {f"u{i}": 0.5 for i in range(20)}; same = dict(base); other = dict(base); other["u3"] = 0.5 + 1e-9
    r = PC.loss_coefficients(rows, {"X": base, "Y": same, "Z": other}, seed=901)
    assert r["merged"] == {"Y": {"implementation_identical_to": "X", "condition": r["merged"]["Y"]["condition"], "seed": 901}} and ["X", "Y"] in r["groups"] and ["Z"] in r["groups"]
    c = r["coefficients"]["X"]; assert abs(c["mean"] - 1.0) < 1e-12               # coefficients average to 1 by construction


def test_rho_independent_batch_reported():
    rng = np.random.default_rng(2); sP, sQ = np.clip(rng.normal(0.7, 0.15, 600), 0, 1), np.clip(rng.normal(0.3, 0.15, 600), 0, 1)
    rho = 0.3; C = rng.random(1500) < 1 - rho; sU = np.where(C, np.clip(rng.normal(0.7, 0.15, 1500), 0, 1), np.clip(rng.normal(0.3, 0.15, 1500), 0, 1))
    imgs = [f"i{k // 5}" for k in range(1500)]; est = np.array([int(i[1:]) % 5 != 0 for i in imgs])
    s = {"fit_P": sP, "fit_Q": sQ, "sel_P": sP[:200], "sel_Q": sQ[:200], "ref_P": sP, "ref_Q": sQ, "U": sU, "U_img": imgs, "U_est_mask": est,
         "ref_P_img": [f"p{k}" for k in range(600)], "ref_Q_img": [f"q{k}" for k in range(600)]}
    res = W.weights_for_method("VCS_Weight", s, steps=150); r = res["record"]["rho"]
    assert r["n_U"] == int(est.sum()) and "independent_batch" in r and r["independent_batch"]["n_pairs"] == int((~est).sum())
    assert abs(r["projected"] - rho) < 0.06 and r["independent_batch"]["interval"]["status"] in ("bounded_interval", "full_parameter_space") and len(res["w"]) == 1500


def test_official_export_matches_table(tmp_path):
    from v9.n2 import noise as NZ, roles as RL
    from v9.n2.fixtures import make_fixture
    fx = make_fixture(n_images=40, seed=4); roles = RL.assign_roles([str(i) for i in range(40)], 4, 0.25, min_sizes={k: 1 for k in RL.ANCHOR_ROLES})
    pairs = NZ.make_pair_table(fx["caption_image"], RL.role_of_image(roles)); obs, lab, man = NZ.inject_mismatch(pairs, 0.4, 4); NZ.write_noise(obs, lab, man, tmp_path, "pairs"); RL.write_roles(roles, tmp_path / "roles.json")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/v9/n2_official.py"), "export", "--shared", str(tmp_path), "--data-root", str(tmp_path / "none"), "--out", str(tmp_path / "unit")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t2i = np.load(tmp_path / "unit/v9_observed_t2i.npy"); am = np.load(tmp_path / "unit/v9_anchor_mask.npy")
    assert len(t2i) == 200 and t2i.dtype == np.int64 and am.sum() == sum(1 for o in obs if o["role"] != "U")
    for o in obs:
        assert t2i[int(o["caption_id"])] == int(o["observed_image_id"]) and am[int(o["caption_id"])] == (o["role"] != "U")
    assert all(t2i[c] == c // 5 for c in np.where(am)[0])                                   # anchors are clean


@pytest.mark.parametrize("name", ["NCR", "L2RM"])
def test_patches_apply_to_clones(name):
    c = _clone(name)
    if c is None:
        pytest.skip("clone not available")
    r = subprocess.run(["git", "apply", "--check", str(ROOT / "v9/patches" / f"{name.lower()}_anchor_assisted.diff")], cwd=c, capture_output=True, text=True); assert r.returncode == 0, r.stderr


def test_degradation_check_fixture(tmp_path):
    n, l = _clone("NCR"), _clone("L2RM")
    if n is None or l is None:
        pytest.skip("clones not available")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/v9/n2_degradation.py"), "--ncr", n, "--l2rm", l, "--out", str(tmp_path / "d.json")], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    d = json.loads((tmp_path / "d.json").read_text()); assert d["pass"] and d["ncr"]["fixed_batch_eval_loss_equal"] and d["l2rm"]["split_labeled_equal"]


@pytest.mark.parametrize("rel", ["reports/V9/HOST_LOCK.yaml", "reports/V9/V9_N2_PROTOCOL_DRAFT.yaml", "reports/V9/V9_N2_PROTOCOL_FROZEN*.yaml"])
def test_n2_yaml_files_parse(rel):
    """HOST_LOCK and the N2 protocol must be loadable YAML (precheck 1042125 died on a HOST_LOCK ScannerError).  The frozen G1 / N1 protocols are text-only and hashed: not covered."""
    import glob
    import yaml
    files = sorted(glob.glob(str(ROOT / rel)))
    if not files:
        pytest.skip(f"{rel}: not present yet")
    for f in files:
        d = yaml.safe_load(Path(f).read_text()); assert isinstance(d, dict) and d, f


def test_student_warmup_phase_then_hard_negatives(tmp_path):
    """HOST_LOCK 57366c5: the weighted student runs the warm-up (mean) hinge for warmup_epochs, then max-violation; both phases logged."""
    from functools import partial
    from v9.n2 import noise as NZ, roles as RL
    from v9.n2.fixtures import collate_pairs, make_fixture
    from v9.n2.teacher import HostConfig
    from v9.n2.train import WeightedStudent
    fx = make_fixture(n_images=24, seed=8); roles = RL.assign_roles([str(i) for i in range(24)], 8, 0.25, min_sizes={k: 1 for k in RL.ANCHOR_ROLES})
    pairs = NZ.make_pair_table(fx["caption_image"], RL.role_of_image(roles)); obs, _, _ = NZ.inject_mismatch(pairs, 0.0, 8)
    cfg = HostConfig(vocab_size=fx["vocab"], img_dim=fx["img_dim"], word_dim=8, embed_size=16, sim_dim=8, num_region=fx["n_region"], batch_size=8, num_epochs=3, warmup_epochs=2)
    dev = make_fixture(n_images=6, seed=9); st = WeightedStudent(cfg, obs, {r["pair_id"]: 1.0 for r in obs if r["role"] == "U"}, partial(collate_pairs, fx["images"], fx["captions"]), dev["images"], [dev["captions"][i] for i in range(30)], tmp_path / "u", 0)
    st.fit(); log = json.loads((tmp_path / "u" / "train_log.json").read_text())
    assert [e["phase"] for e in log["epochs"]] == ["warmup_mean_hinge", "warmup_mean_hinge", "max_violation"] and "57366c5" in json.loads((tmp_path / "u" / "run_manifest.json").read_text())["loss"]
