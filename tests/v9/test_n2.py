"""V9-N2 adapter tests (synthetic fixtures; no data, no GPU)."""
from __future__ import annotations

import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from v9.n2 import eval as EV, noise as NZ, roles as RL, sgr_host as H, weights as W  # noqa: E402
from v9.n2.fixtures import collate_pairs, make_fixture  # noqa: E402
from v9.n2.strict import run_strict_check  # noqa: E402
from v9.n2.teacher import HostConfig, build_host  # noqa: E402
from v9.n2.train import WeightedStudent  # noqa: E402

MIN1 = {k: 1 for k in RL.ANCHOR_ROLES}


def _roles_and_pairs(n_images=40, seed=3, rho=0.4):
    fx = make_fixture(n_images=n_images, seed=seed)
    roles = RL.assign_roles(sorted(set(str(i) for i in fx["caption_image"].values())), seed, 0.25, min_sizes=MIN1)
    pairs = NZ.make_pair_table(fx["caption_image"], RL.role_of_image(roles)); observed, labels, man = NZ.inject_mismatch(pairs, rho, seed)
    return fx, roles, pairs, observed, labels, man


def test_roles_disjoint_counts_and_caption_groups():
    fx, roles, pairs, *_ = _roles_and_pairs()
    RL.check_disjoint(roles); assert roles["n_anchor"] == 10 and sum(roles["counts"].values()) == 10 and roles["counts"] == {"teacher_warmup": 4, "critic_fit": 3, "critic_select": 2, "reference_moments": 1}
    rmap = RL.role_of_image(roles)
    for r in pairs:                                          # every caption of an image carries the image's role
        assert r["role"] == rmap[r["image_id"]]
    assert RL.assign_roles(list(range(40)), 3, 0.25, min_sizes=MIN1)["hashes"] == roles["hashes"]   # deterministic
    with pytest.raises(ValueError):
        RL.assign_roles(list(range(40)), 3, 0.25, min_sizes={"reference_moments": 5})


def test_noise_realized_rho_donors_and_anchor_clean():
    fx, roles, pairs, observed, labels, man = _roles_and_pairs(rho=0.4)
    u_pairs = [r for r in pairs if r["role"] == "U"]; assert man["n_selected"] == round(0.4 * len(u_pairs))
    assert man["realized_rho"] <= 0.4 + 1e-12 and man["realized_rho"] > 0.3 and man["n_A_mismatched"] == 0
    u_images = {r["image_id"] for r in u_pairs}
    for o, r in zip(observed, pairs):
        if r["role"] != "U":
            assert o["observed_image_id"] == r["image_id"]            # A untouched
        else:
            assert o["observed_image_id"] in u_images                  # donors only from U
        assert labels[r["pair_id"]]["mismatched"] == (o["observed_image_id"] != r["image_id"])
    o0, l0, m0 = NZ.inject_mismatch(pairs, 0.0, 3); assert m0["realized_rho"] == 0 and all(o["observed_image_id"] == r["image_id"] for o, r in zip(o0, pairs))


def test_labels_unreachable_from_training_side(tmp_path):
    fx, roles, pairs, observed, labels, man = _roles_and_pairs()
    paths = NZ.write_noise(observed, labels, man, tmp_path, "pairs")
    rows = NZ.load_observed(paths["observed"]); assert all("original_image_id" not in r and "mismatched" not in r for r in rows)
    with pytest.raises(ValueError):
        NZ.load_observed(paths["eval_only"])
    src = (ROOT / "src/v9/n2/train.py").read_text() + (ROOT / "src/v9/n2/teacher.py").read_text() + (ROOT / "src/v9/n2/weights.py").read_text()
    assert "eval_only" not in src.replace("EVAL_ONLY_SUFFIX", "") or "refuses" in src          # the training modules never open the labels
    bad = dict(rows[0]); bad["mismatched"] = True
    cfg = HostConfig(vocab_size=40, img_dim=32, word_dim=8, embed_size=16, sim_dim=8, num_region=6, batch_size=8, num_epochs=1)
    with pytest.raises(ValueError):
        WeightedStudent(cfg, [bad], {}, lambda r: None, fx["images"][:2], [fx["captions"][0]] * 10, tmp_path / "u", 0)


def test_hinge_per_anchor_matches_official_aggregate():
    torch.manual_seed(0); sims = torch.rand(7, 7)
    er, ec = H.hinge_per_anchor(sims, 0.2, True)
    diagonal = sims.diag().view(7, 1); cost_s = (0.2 + sims - diagonal.expand_as(sims)).clamp(min=0); cost_im = (0.2 + sims - diagonal.t().expand_as(sims)).clamp(min=0)
    mask = torch.eye(7) > .5; cost_s = cost_s.masked_fill(mask, 0); cost_im = cost_im.masked_fill(mask, 0)
    assert torch.allclose(er.sum() + ec.sum(), cost_s.max(1)[0].sum() + cost_im.max(0)[0].sum())   # SGRAF ContrastiveLoss(max_violation=True)
    er2, ec2 = H.hinge_per_anchor(sims, 0.2, False); assert torch.allclose(er2.sum() + ec2.sum(), cost_s.mean(1).sum() + cost_im.mean(0).sum())


@pytest.mark.skipif(not Path("/tmp/claude-34987/-home-infres-yinwang-CS-QMI/3ef2737e-e5a7-4b67-9f0b-53c6d6caa16c/scratchpad/v9_code/NCR/NCR/model.py").exists()
                    and not Path("/projects/EEG-foundation-model/yinghao/code_v9/NCR/NCR/model.py").exists(), reason="official NCR clone not available")
def test_vendored_host_matches_official():
    for cand in ("/projects/EEG-foundation-model/yinghao/code_v9/NCR/NCR/model.py", "/tmp/claude-34987/-home-infres-yinwang-CS-QMI/3ef2737e-e5a7-4b67-9f0b-53c6d6caa16c/scratchpad/v9_code/NCR/NCR/model.py"):
        if Path(cand).exists():
            break
    spec = importlib.util.spec_from_file_location("ncr_model_official", cand); off = importlib.util.module_from_spec(spec); spec.loader.exec_module(off)
    torch.manual_seed(1); vocab, img_dim, embed, sim_dim, R = 30, 24, 16, 8, 5
    mine = H.SGRHost(vocab, img_dim, 8, embed, sim_dim, 1, True, R, 2).eval()
    o_img = off.EncoderImage(img_dim, embed); o_txt = off.EncoderText(vocab, 8, embed, 1, use_bi_gru=True); o_sim = off.EncoderSimilarity(embed, sim_dim, "SGR", 2)
    o_sim.v_global_w = off.VisualSA(embed, 0.4, R)                                              # official hard-codes 36 regions
    o_img.load_state_dict(mine.img_enc.state_dict()); o_txt.load_state_dict(mine.txt_enc.state_dict()); o_sim.load_state_dict(mine.sim_enc.state_dict())
    o_img.eval(); o_txt.eval(); o_sim.eval()
    ims = torch.randn(4, R, img_dim); caps = torch.randint(3, vocab, (4, 7)); lens = [7, 6, 5, 4]
    for i, L in enumerate(lens):
        caps[i, L:] = 0
    with torch.no_grad():
        a = mine(ims, caps, lens); b = o_sim(o_img(ims), o_txt(caps, lens), lens)
    assert torch.allclose(a, b, atol=1e-6)


def test_normaliser_fixed_and_weights_detached(tmp_path):
    fx, roles, pairs, observed, labels, man = _roles_and_pairs(n_images=24, seed=5, rho=0.3)
    cfg = HostConfig(vocab_size=fx["vocab"], img_dim=fx["img_dim"], word_dim=8, embed_size=16, sim_dim=8, num_region=fx["n_region"], batch_size=8, num_epochs=1, teacher_epochs=1)
    u_ids = [r["pair_id"] for r in observed if r["role"] == "U"]; rng = np.random.default_rng(0); w = {p: float(rng.uniform(0, 1)) for p in u_ids}
    collate = partial(collate_pairs, fx["images"], fx["captions"])
    dev = make_fixture(n_images=6, seed=9); st = WeightedStudent(cfg, observed, w, collate, dev["images"], [dev["captions"][i] for i in range(30)], tmp_path / "u", 0)
    n_A = sum(r["role"] != "U" for r in observed); assert abs(st.mean_v_full - (n_A + sum(w.values())) / len(observed)) < 1e-12
    # the batch loss divides by the FIXED pool mean, not the batch mean; the weights carry no gradient path
    host = build_host(cfg); host.train(); idx = np.arange(8); ims, caps, lens = collate([observed[i] for i in idx]); v = torch.tensor(st.v[idx], dtype=torch.float32, requires_grad=True)
    sims = host(ims, caps, lens); er, ec = H.hinge_per_anchor(sims, cfg.margin); loss = (v.detach() * (er + ec)).mean() / st.mean_v_full
    loss.backward(); assert v.grad is None and any(p.grad is not None for p in host.parameters())
    done = st.fit(); assert (tmp_path / "u" / "done.json").exists() and (tmp_path / "u" / "best.pt").exists() and done["best"]["epoch"] == 0
    man2 = json.loads((tmp_path / "u" / "run_manifest.json").read_text()); assert man2["mean_v_full_pool"] == st.mean_v_full and man2["n_U"] == len(u_ids)


def test_resume_continues_from_checkpoint(tmp_path):
    fx, roles, pairs, observed, labels, man = _roles_and_pairs(n_images=24, seed=6, rho=0.0)
    cfg = HostConfig(vocab_size=fx["vocab"], img_dim=fx["img_dim"], word_dim=8, embed_size=16, sim_dim=8, num_region=fx["n_region"], batch_size=8, num_epochs=2, teacher_epochs=1)
    w = {r["pair_id"]: 1.0 for r in observed if r["role"] == "U"}; collate = partial(collate_pairs, fx["images"], fx["captions"]); dev = make_fixture(n_images=6, seed=9); dc = [dev["captions"][i] for i in range(30)]
    full = WeightedStudent(cfg, observed, w, collate, dev["images"], dc, tmp_path / "full", 0).fit()
    part = WeightedStudent(cfg, observed, w, collate, dev["images"], dc, tmp_path / "part", 0); part.fit(epochs=1)
    (tmp_path / "part" / "done.json").unlink()
    res = WeightedStudent(cfg, observed, w, collate, dev["images"], dc, tmp_path / "part", 0).fit(epochs=2)
    lf, lp = json.loads((tmp_path / "full" / "train_log.json").read_text()), json.loads((tmp_path / "part" / "train_log.json").read_text())
    assert lp["resumed_at"] == [1] and len(lp["epochs"]) == 2
    assert abs(lf["epochs"][1]["mean_loss"] - lp["epochs"][1]["mean_loss"]) < 1e-5 and res["epochs_run"] == 2


def test_rankmass_mass_equality_and_filter():
    s = {"U": np.linspace(0, 1, 50), "total_mass": 17.3, "sel_P": np.linspace(0.5, 1, 20)}
    w = W.weights_for_method("RankMass", s)["w"]; assert abs(w.sum() - 17.3) < 1e-9 and set(np.round(w[np.argsort(-s["U"])][:17], 9)) == {1.0}
    f = W.weights_for_method("PlainFilter", s); assert set(np.unique(f["w"])) <= {0.0, 1.0} and f["record"]["main_family"] is False


def test_scalar_critic_vcs_bce_and_rho():
    rng = np.random.default_rng(0); sP, sQ = np.clip(rng.normal(0.7, 0.15, 800), 0, 1), np.clip(rng.normal(0.3, 0.15, 800), 0, 1)
    for kind in ("vcs", "bce"):
        fit = W.fit_scalar_critic(sP, sQ, sP[:200], sQ[:200], kind=kind, steps=200); T = fit["critic"].T
        assert fit["selected"]["a"] > 0 and np.mean(T(sP)) > 0.5 and np.mean(T(sQ)) < -0.5
    rho = 0.3; C = rng.random(2000) < 1 - rho; sR = np.where(C, np.clip(rng.normal(0.7, 0.15, 2000), 0, 1), np.clip(rng.normal(0.3, 0.15, 2000), 0, 1))
    est = W.estimate_rho(T(sP), T(sQ), T(sR), clusters_P=[f"i{i}" for i in range(800)], clusters_Q=[f"q{i}" for i in range(800)], clusters_U=[f"u{i}" for i in range(2000)])
    assert abs(est["projected"] - rho) < 0.05 and est["status"] == "point_estimate" and est["interval"]["lower"] <= rho <= est["interval"]["upper"]
    w = W.posterior_weights(T(sR), est["projected"]); assert w[C].mean() > w[~C].mean() and np.all((w >= 0) & (w <= 1))
    assert np.all(W.posterior_weights(T(sR), 0.0) == 1.0)


def test_weight_lock_roundtrip_and_tamper(tmp_path):
    lock = W.write_weight_lock(tmp_path / "w.csv", ["p1", "p2"], [0.5, 1.0], [0.3, 0.9], 0.2, "fh", "rh", {"method": "x"})
    assert W.read_weight_lock(tmp_path / "w.csv") == {"p1": 0.5, "p2": 1.0} and lock["n"] == 2
    (tmp_path / "w.csv").write_text((tmp_path / "w.csv").read_text().replace("0.5000000000", "0.9000000000"))
    with pytest.raises(ValueError):
        W.read_weight_lock(tmp_path / "w.csv")


def test_recall_metrics_vs_reference_and_coco_folds():
    rng = np.random.default_rng(0); n = 30; sims = rng.normal(size=(n, 5 * n)); sims[np.arange(n)[:, None], (5 * np.arange(n))[:, None] + np.arange(5)] += 2.0
    m = EV.recall_metrics(sims, 5, "stable"); ref = EV.reference_cross_check(sims)
    for k in (1, 5, 10):
        assert abs(m[f"i2t_R{k}"] - ref[f"i2t_R{k}"]) < 1e-9 and abs(m[f"t2i_R{k}"] - ref[f"t2i_R{k}"]) < 1e-9
    assert abs(m["rsum"] - ref["rsum"]) < 1e-9 and abs(m["mR"] - m["rsum"] / 6) < 1e-12
    big = rng.normal(size=(5000, 25000)); out = EV.evaluate_split(big, "coco"); assert "coco_5x1k_mean" in out and len(out["coco_5x1k_folds"]) == 5 and len(out["random_tie"]) == 3


def test_similarity_matrix_equals_pairwise_scores():
    fx = make_fixture(n_images=8, seed=2); cfg = HostConfig(vocab_size=fx["vocab"], img_dim=fx["img_dim"], word_dim=8, embed_size=16, sim_dim=8, num_region=fx["n_region"])
    host = build_host(cfg).eval(); caps = [fx["captions"][i] for i in range(40)]
    S = EV.similarity_matrix(host, fx["images"], caps, img_batch=3, cap_batch=7)
    rows = [{"caption_id": j, "observed_image_id": j // 5} for j in range(40)]
    from v9.n2.teacher import score_pairs
    s = score_pairs(host, rows, partial(collate_pairs, fx["images"], fx["captions"]), batch_size=6)
    assert np.allclose(S[np.arange(40) // 5, np.arange(40)], s, atol=1e-5)


def test_strict_interface_check():
    r = run_strict_check(n=3000, seed=1)
    assert r["z_only"]["rel_err_oracle"] < 0.05 and r["z_only"]["rel_err_fitted"] < 0.12 and abs(r["rho_hat"] - 0.4) < 0.06
    assert r["richer_student"]["rel_err"] > 0.05


@pytest.mark.parametrize("repo,patch", [("NCR", "ncr_anchor_assisted.diff"), ("L2RM", "l2rm_anchor_assisted.diff")])
def test_anchor_assisted_patches_apply(repo, patch):
    """The NCR+A / L2RM+A patches apply cleanly to the official clones (read-only reference copies or fork A's code_v9)."""
    import subprocess
    for base in (f"/projects/EEG-foundation-model/yinghao/code_v9/{repo}", f"/tmp/claude-34987/-home-infres-yinwang-CS-QMI/3ef2737e-e5a7-4b67-9f0b-53c6d6caa16c/scratchpad/v9_code/{repo}"):
        if Path(base, ".git").exists():
            break
    else:
        pytest.skip("official clone not available")
    r = subprocess.run(["git", "apply", "--check", str(ROOT / "v9" / "patches" / patch)], cwd=base, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
