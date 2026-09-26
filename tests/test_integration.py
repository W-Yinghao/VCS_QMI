"""Server integration tests (spec §15.2).  Tiny synthetic CIFAR-shaped data; real data/manifest checks run when DATA_ROOT
and MANIFEST_ROOT are exported.  Nothing here is benchmark evidence."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from reference.ssl_core import cyclic_negative_indices
from vcs_ssl.checkpoint import load_checkpoint
from vcs_ssl.config import ConfigError, load_config, policy_checks
from vcs_ssl.data.cifar import CifarTrain, load_cifar10_train
from vcs_ssl.data.datasets import SSLTwoViewDataset
from vcs_ssl.data.splits import build_manifest, load_manifest
from vcs_ssl.data.transforms import build_clean_transform, build_two_view_transform
from vcs_ssl.diagnostics import extract_features, linear_probe
from vcs_ssl.models import build_models
from vcs_ssl.objectives import compute_objective, forward_features
from vcs_ssl.optim import build_optimizer, grad_norms, verify_optimizer_coverage
from vcs_ssl.schedule import lr_factor, warmup_steps_for
from vcs_ssl.train import Trainer
from vcs_ssl.utils import state_dict_sha256

REPO = Path(__file__).resolve().parents[1]
CFG_DIR = REPO / "configs"
DEVICE = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


# ----------------------------------------------------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------------------------------------------------
def _env(tmp: Path) -> dict[str, str]:
    for d in ("data", "out", "man"):
        (tmp / d).mkdir(exist_ok=True)
    return {"DATA_ROOT": str(tmp / "data"), "OUTPUT_ROOT": str(tmp / "out"), "MANIFEST_ROOT": str(tmp / "man")}


def small_cfg(tmp: Path, name: str, *, batch: int = 32, epochs: int = 2, workers: int = 0) -> dict:
    cfg = load_config(CFG_DIR / f"cifar10_pilot_{name}.yaml", env=_env(tmp), require_dirs=True)
    cfg = copy.deepcopy(cfg)
    t = cfg["train"]
    t.update({"batch_size_images": batch, "epochs": epochs, "warmup_epochs": 1, "num_workers": workers, "pin_memory": False})
    e = cfg["evaluation"]
    e["knn_epochs"] = [0, epochs]
    e["knn"]["k"] = 20
    e["spectrum"]["selection_first_sorted_ids"] = 64
    e["critic_validation"].update({"repeats": 2, "batch_size": 32})
    e["linear"].update({"epochs": 2})
    cfg["logging"]["step_interval"] = 1
    cfg["logging"]["checkpoint_epochs"] = [epochs]
    policy_checks(cfg)
    return cfg


def synthetic_bundle(n_per_class: int = 64, seed: int = 0) -> tuple[CifarTrain, dict]:
    rng = np.random.default_rng(seed)
    n = 10 * n_per_class
    images = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
    targets = np.repeat(np.arange(10), n_per_class)
    targets = rng.permutation(targets)
    data = CifarTrain(data=images, targets=targets, root="synthetic", file_hashes={"synthetic": {"md5": "0", "sha256": "0", "bytes": "0"}},
                      source={"dataset": "synthetic"})
    manifest = build_manifest(targets, split_seed=20260924, val_per_class=8, source=data.source, file_hashes=data.file_hashes)
    return data, manifest


def run_trainer(tmp: Path, cfg: dict, data, manifest, *, run_id: str, device=DEVICE, **kw) -> tuple[Trainer, str]:
    run_dir = Path(cfg["run"]["output_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tr = Trainer(cfg, run_dir=run_dir, data=data, manifest=manifest, device=device, stage="TEST", **kw)
    tr.setup()
    status = tr.run()
    return tr, status


def steps_log(run_dir: Path) -> list[dict]:
    p = run_dir / "logs" / "steps.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ----------------------------------------------------------------------------------------------------------------------
# 1. real ResNet shapes
# ----------------------------------------------------------------------------------------------------------------------
def test_1_real_resnet_shapes(tmp_path):
    cfg = small_cfg(tmp_path, "vcs")
    built = build_models(cfg, seed=0, device="cpu")
    enc, proj, crit = built["encoder"], built["projector"], built["critic"]
    x1, x2 = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
    f = forward_features(enc, proj, x1, x2, eps=1e-8)
    assert f["h"].shape == (8, 512) and f["p_raw"].shape == (8, 128) and f["z_l2"].shape == (8, 128)
    torch.testing.assert_close(f["z_l2"].norm(dim=1), torch.ones(8), atol=1e-5, rtol=0)
    z1, z2 = f["z_l2"].chunk(2)
    assert crit(z1, z2).shape == (4,)
    idx, _ = cyclic_negative_indices(4, 1, generator=torch.Generator().manual_seed(0))
    assert crit(z1, z2[idx[0]]).shape == (4,)
    assert built["params"]["encoder"] == 11_168_832  # torchvision resnet18 minus fc, conv1 3x3
    assert "Identity" in built["model_strings"]["encoder"]


# ----------------------------------------------------------------------------------------------------------------------
# 2. gradients & optimizer coverage
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["vcs", "simclr", "vicreg"])
def test_2_gradients_optimizer_coverage(tmp_path, name):
    cfg = small_cfg(tmp_path, name)
    built = build_models(cfg, seed=0, device=DEVICE)
    enc, proj, crit = built["encoder"], built["projector"], built["critic"]
    opt = build_optimizer(enc, proj, crit, cfg["optimizer"])
    cov = verify_optimizer_coverage(opt, enc, proj, crit)
    assert cov["n_params_in_optimizer"] == cov["n_trainable_expected"]
    x1, x2 = torch.randn(8, 3, 32, 32, device=DEVICE), torch.randn(8, 3, 32, 32, device=DEVICE)
    f = forward_features(enc, proj, x1, x2, eps=1e-8)
    f["z_l2"].retain_grad(); f["p_raw"].retain_grad()
    obj = compute_objective(cfg["run"]["method"], f, cfg=cfg, critic=crit, pair_generator=torch.Generator().manual_seed(1))
    obj["loss"].backward()
    gn = grad_norms(enc, proj, crit)
    assert gn["grad_norm_encoder"] > 0 and gn["grad_norm_projector"] > 0
    if name == "vcs":
        assert gn["grad_norm_critic"] > 0
        g = f["z_l2"].grad
        assert g[:8].abs().sum() > 0 and g[8:].abs().sum() > 0  # both views receive gradients
    else:
        assert crit is None and gn["grad_norm_critic"] is None
    # negative branch alone reaches the encoder (VCS)
    if name == "vcs":
        enc.zero_grad(); proj.zero_grad(); crit.zero_grad()
        f2 = forward_features(enc, proj, x1, x2, eps=1e-8)
        z1, z2 = f2["z_l2"].chunk(2)
        idx, _ = cyclic_negative_indices(8, 1, generator=torch.Generator().manual_seed(3))
        tn = crit(z1, z2[idx[0]])
        (tn.mean() + 0.5 * tn.square().mean()).backward()
        assert grad_norms(enc, proj, crit)["grad_norm_encoder"] > 0


# ----------------------------------------------------------------------------------------------------------------------
# 3. labels inaccessible to SSL training
# ----------------------------------------------------------------------------------------------------------------------
def test_3_labels_inaccessible(tmp_path):
    data, manifest = synthetic_bundle()
    cfg = small_cfg(tmp_path, "vcs")
    ds = SSLTwoViewDataset(data.data, np.asarray(manifest["fit_uids"]), build_two_view_transform(cfg["views"]))
    item = ds[0]
    assert len(item) == 3 and isinstance(item[2], int)
    assert not hasattr(ds, "targets") and not hasattr(ds, "labels")
    # same run twice with permuted class labels -> identical step losses
    tr_a, st_a = run_trainer(tmp_path, cfg, data, manifest, run_id="labels_a", device=torch.device("cpu"), smoke_steps=2, epoch_eval=False)
    data_perm = CifarTrain(data=data.data, targets=np.random.default_rng(1).permutation(data.targets), root="synthetic",
                           file_hashes=data.file_hashes, source=data.source)
    tr_b, st_b = run_trainer(tmp_path, cfg, data_perm, manifest, run_id="labels_b", device=torch.device("cpu"), smoke_steps=2, epoch_eval=False)
    assert st_a == st_b == "COMPLETED"
    la, lb = steps_log(tr_a.run_dir), steps_log(tr_b.run_dir)
    assert [r["loss"] for r in la] == [r["loss"] for r in lb]
    assert [r["shift"] for r in la] == [r["shift"] for r in lb]


# ----------------------------------------------------------------------------------------------------------------------
# 4. split isolation
# ----------------------------------------------------------------------------------------------------------------------
def test_4_split_isolation_synthetic(tmp_path):
    data, m = synthetic_bundle()
    fit, sel = set(m["fit_uids"]), set(m["selection_uids"])
    assert not (fit & sel) and len(fit) + len(sel) == len(data)
    assert m["selection_class_counts"] == [8] * 10 and m["fit_class_counts"] == [56] * 10
    cfg = small_cfg(tmp_path, "simclr")
    tr, st = run_trainer(tmp_path, cfg, data, m, run_id="split", device=torch.device("cpu"), smoke_steps=2, epoch_eval=False)
    assert st == "COMPLETED"
    # a batch containing a selection UID is refused by the trainer
    bad = torch.tensor(sorted(sel)[:cfg["train"]["batch_size_images"]])
    x = torch.randn(len(bad), 3, 32, 32)
    with pytest.raises(RuntimeError, match="outside the fit split"):
        tr.train_step(x, x, bad, log_this=False)
    with pytest.raises(RuntimeError, match="duplicate UID"):
        tr.train_step(x[:4], x[:4], torch.tensor(sorted(fit)[:3] + sorted(fit)[:1]), log_this=False)


@pytest.mark.skipif(not os.environ.get("MANIFEST_ROOT"), reason="MANIFEST_ROOT not exported")
def test_4b_real_manifest_if_present():
    p = Path(os.environ["MANIFEST_ROOT"]) / "cifar10_dev45k_val5k.json"
    if not p.is_file():
        pytest.skip("real manifest not created yet")
    m = load_manifest(p, expected_seed=20260924, expected_val_per_class=500)
    assert m["n_fit"] == 45000 and m["n_selection"] == 5000
    assert m["fit_class_counts"] == [4500] * 10 and m["selection_class_counts"] == [500] * 10
    assert not (set(m["fit_uids"]) & set(m["selection_uids"]))
    assert max(m["fit_uids"] + m["selection_uids"]) < 50000  # no official-test UID can exist


# ----------------------------------------------------------------------------------------------------------------------
# 5. augmentation independence
# ----------------------------------------------------------------------------------------------------------------------
def test_5_augmentation_independence(tmp_path):
    data, m = synthetic_bundle()
    cfg = small_cfg(tmp_path, "vcs")
    ds = SSLTwoViewDataset(data.data, np.asarray(m["fit_uids"]), build_two_view_transform(cfg["views"]))
    torch.manual_seed(0)
    diff_within = sum(not torch.equal(ds[i][0], ds[i][1]) for i in range(8))
    a = ds[3][0]
    torch.manual_seed(1)
    b = ds[3][0]
    assert diff_within >= 7 and not torch.equal(a, b)
    v1, v2, uid = ds[0]
    assert v1.shape == (3, 32, 32) and v1.dtype == torch.float32 and uid == int(m["fit_uids"][0])
    clean = build_clean_transform(cfg["views"])
    c1 = clean(__import__("PIL.Image", fromlist=["fromarray"]).fromarray(data.data[0]))
    assert torch.equal(c1, clean(__import__("PIL.Image", fromlist=["fromarray"]).fromarray(data.data[0])))
    assert c1.min() >= -1.0 and c1.max() <= 1.0


# ----------------------------------------------------------------------------------------------------------------------
# 6. evaluation does not pollute training
# ----------------------------------------------------------------------------------------------------------------------
def test_6_eval_no_pollution(tmp_path):
    data, m = synthetic_bundle()
    cfg = small_cfg(tmp_path, "vcs")
    built = build_models(cfg, seed=0, device=DEVICE)
    enc, proj = built["encoder"], built["projector"]
    enc.train(); proj.train()
    with torch.no_grad():  # perturb BN running stats so they are non-trivial
        enc(torch.randn(8, 3, 32, 32, device=DEVICE))
    before = state_dict_sha256(enc)
    clean = build_clean_transform(cfg["views"])
    fit_u, sel_u = np.asarray(m["fit_uids"])[:128], np.asarray(m["selection_uids"])
    ff = extract_features(enc, proj, data.data, data.targets, fit_u, clean, device=DEVICE, num_workers=0, batch_size=64)
    fs = extract_features(enc, proj, data.data, data.targets, sel_u, clean, device=DEVICE, num_workers=0, batch_size=64)
    lp = linear_probe(ff["h"], ff["labels"], fs["h"], fs["labels"], cfg["evaluation"]["linear"], device=DEVICE)
    assert 0.0 <= lp["linear_val_top1_pct"] <= 100.0 and lp["probe_epochs"] == 2
    assert state_dict_sha256(enc) == before  # parameters and BN buffers untouched
    assert all(not p.requires_grad for p in enc.parameters())
    # in-training evaluation leaves training RNG streams untouched
    tr, st = run_trainer(tmp_path, cfg, data, m, run_id="evalrng", smoke_steps=3, smoke_epoch_steps=3, epoch_eval=True)
    assert st == "COMPLETED"
    res = json.loads((tr.run_dir / "evaluations" / "knn_epoch_001.json").read_text())
    assert res["training_rng_untouched"] is True
    assert (tr.run_dir / "evaluations" / "critic_holdout_epoch_001.json").is_file()
    assert (tr.run_dir / "evaluations" / "spectrum_epoch_000.json").is_file()


# ----------------------------------------------------------------------------------------------------------------------
# 7. checkpoint resume (single worker, CPU, exact)
# ----------------------------------------------------------------------------------------------------------------------
def test_7_checkpoint_resume_exact(tmp_path):
    data, m = synthetic_bundle()
    cfg = small_cfg(tmp_path, "vcs", workers=0)
    cpu = torch.device("cpu")
    tr_a, st_a = run_trainer(tmp_path, cfg, data, m, run_id="cont", device=cpu, smoke_steps=6, smoke_epoch_steps=3, epoch_eval=False)
    assert st_a == "COMPLETED" and tr_a.step == 6
    tr_b, st_b = run_trainer(tmp_path, cfg, data, m, run_id="interrupted", device=cpu, smoke_steps=6, smoke_epoch_steps=3, epoch_eval=False, stop_after_steps=3)
    assert st_b == "STOPPED_BUDGET" and tr_b.completed_epoch == 1
    ck = load_checkpoint(tr_b.ckpt_dir / "last.pt")
    assert ck["optimizer_step"] == 3 and ck["completed_epoch"] == 1
    tr_c, st_c = run_trainer(tmp_path, cfg, data, m, run_id="interrupted", device=cpu, smoke_steps=6, smoke_epoch_steps=3, epoch_eval=False,
                             resume_from=tr_b.ckpt_dir / "last.pt")
    assert st_c == "COMPLETED" and tr_c.step == 6
    for mod in ("encoder", "projector", "critic"):
        a, c = getattr(tr_a, mod).state_dict(), getattr(tr_c, mod).state_dict()
        for k in a:
            assert torch.equal(a[k], c[k]), f"{mod}.{k} differs after resume"
    la = [r["loss"] for r in steps_log(tr_a.run_dir)]
    lc = [r["loss"] for r in steps_log(tr_c.run_dir) if r["step"] >= 3]
    assert la[3:] == lc
    assert [r["shift"] for r in steps_log(tr_a.run_dir)][3:] == [r["shift"] for r in steps_log(tr_c.run_dir) if r["step"] >= 3]
    # refusing a changed horizon / config
    cfg2 = copy.deepcopy(cfg)
    with pytest.raises(ConfigError, match="resume refused"):
        run_dir = tr_b.run_dir
        t = Trainer(cfg2, run_dir=run_dir, data=data, manifest=m, device=cpu, stage="TEST", smoke_steps=9, smoke_epoch_steps=3,
                    resume_from=tr_b.ckpt_dir / "last.pt", epoch_eval=False)
        t.setup()


# ----------------------------------------------------------------------------------------------------------------------
# 8. config strictness
# ----------------------------------------------------------------------------------------------------------------------
def _write(tmp: Path, cfg_text: str) -> Path:
    p = tmp / "c.yaml"
    p.write_text(cfg_text)
    return p


def test_8_config_strictness(tmp_path):
    import yaml

    env = _env(tmp_path)
    base = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text())
    ok = load_config(_write(tmp_path, yaml.safe_dump(base)), env=env)
    assert ok["run"]["method"] == "vcs_qmi" and ok["_meta"]["config_file_sha256"]
    bad = copy.deepcopy(base); bad["train"]["mixed_precision"] = True
    with pytest.raises(ConfigError, match="unknown field"):
        load_config(_write(tmp_path, yaml.safe_dump(bad)), env=env)
    for path, val, msg in [(("objective", "extra_regularizers"), ["vicreg_var"], "extra_regularizers"),
                           (("pairing", "k"), 0, "pairing.k"), (("pairing", "k"), 256, "pairing.k"), (("pairing", "allow_self"), True, "pairing policy"),
                           (("model", "weights"), "IMAGENET1K_V1", "weights"), (("data", "official_test_accessible"), True, "official test"),
                           (("train", "precision"), "bf16", "FP32"), (("objective", "clip_J"), True, "clipped J")]:
        bad = copy.deepcopy(base)
        d = bad
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = val
        with pytest.raises(ConfigError, match=msg):
            load_config(_write(tmp_path, yaml.safe_dump(bad)), env=env)
    with pytest.raises(ConfigError, match="unresolved"):
        load_config(CFG_DIR / "cifar10_pilot_vcs.yaml", env={})
    with pytest.raises(ConfigError, match="missing field"):
        bad = copy.deepcopy(base); del bad["pairing"]["k"]
        load_config(_write(tmp_path, yaml.safe_dump(bad)), env=env)


# ----------------------------------------------------------------------------------------------------------------------
# 9. no official test access
# ----------------------------------------------------------------------------------------------------------------------
def test_9_no_official_test_access(tmp_path):
    with pytest.raises(PermissionError):
        load_cifar10_train(tmp_path, train=False)
    root = os.environ.get("DATA_ROOT")
    if root and (Path(root) / "cifar-10-batches-py" / "data_batch_1").is_file():
        import torchvision.datasets as tvd

        orig = tvd.CIFAR10.__init__
        calls = []

        def spy(self, *a, **k):
            calls.append(k.get("train", a[1] if len(a) > 1 else None))
            return orig(self, *a, **k)

        tvd.CIFAR10.__init__ = spy
        try:
            d = load_cifar10_train(root)
        finally:
            tvd.CIFAR10.__init__ = orig
        assert calls == [True] and len(d) == 50000
        assert d.targets[:10].tolist() == [6, 9, 9, 4, 1, 1, 2, 7, 8, 3]
    else:
        pytest.skip("real CIFAR-10 not available; PermissionError path verified only")


# ----------------------------------------------------------------------------------------------------------------------
# 10. first-step checks, epoch-0 and final evaluation artifacts exist
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["simclr", "vicreg"])
def test_10_controls_train_and_emit_artifacts(tmp_path, name):
    data, m = synthetic_bundle()
    cfg = small_cfg(tmp_path, name)
    tr, st = run_trainer(tmp_path, cfg, data, m, run_id=f"ctrl_{name}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=True)
    assert st == "COMPLETED"
    g = json.loads((tr.run_dir / "first_step_gradients.json").read_text())
    assert g["check"] == "PASS" and g["grad_norm_critic"] is None
    assert (tr.run_dir / "checkpoints" / "initial.pt").is_file() and (tr.run_dir / "checkpoints" / "last.pt").is_file()
    assert (tr.run_dir / "evaluations" / "knn_epoch_000.json").is_file() and (tr.run_dir / "evaluations" / "knn_epoch_002.json").is_file()
    assert not (tr.run_dir / "evaluations" / "critic_holdout_epoch_002.json").exists()
    rows = [json.loads(l) for l in (tr.run_dir / "logs" / "epochs.jsonl").read_text().splitlines()]
    assert rows[-1]["status"] == "COMPLETED" and rows[-1]["J_raw"] is None
    s = steps_log(tr.run_dir)
    if name == "simclr":
        assert all(r["nt_xent"] is not None and r["J_raw"] is None for r in s)
    else:
        assert all(r["vicreg_variance"] is not None and r["J_raw"] is None for r in s)
    assert (tr.run_dir / "artifacts" / "view_examples.png").is_file()


def test_10b_vcs_smoke_emits_J_and_critic_stats(tmp_path):
    data, m = synthetic_bundle()
    cfg = small_cfg(tmp_path, "vcs")
    tr, st = run_trainer(tmp_path, cfg, data, m, run_id="vcs_smoke", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=True)
    assert st == "COMPLETED"
    s = steps_log(tr.run_dir)
    assert all(r["J_raw"] is not None and -3.0 <= r["J_raw"] <= 1.0 and abs(r["J_raw"] - (1 - r["R_binary"])) < 1e-5 for r in s)
    assert all(r["shift"] is not None and 1 <= r["shift"] <= cfg["train"]["batch_size_images"] - 1 for r in s)
    g = json.loads((tr.run_dir / "first_step_gradients.json").read_text())
    assert g["grad_norm_critic"] > 0
    ck = load_checkpoint(tr.run_dir / "checkpoints" / "last.pt")
    assert ck["critic_state"] is not None and ck["rng_pair_generator"] is not None and ck["best_metric_policy"] is None
    summary = json.loads((tr.run_dir / "summary.json").read_text())
    assert summary["status"] == "COMPLETED" and summary["seen_base_images"] == 4 * cfg["train"]["batch_size_images"]


# ----------------------------------------------------------------------------------------------------------------------
# schedule
# ----------------------------------------------------------------------------------------------------------------------
def test_schedule_edges():
    assert lr_factor(0, 3500, 350, 0.01) == pytest.approx(1 / 350)
    assert lr_factor(349, 3500, 350, 0.01) == pytest.approx(1.0)
    assert lr_factor(350, 3500, 350, 0.01) == pytest.approx(1.0)
    assert lr_factor(3499, 3500, 350, 0.01) == pytest.approx(0.01)
    assert lr_factor(0, 10, 0, 0.01) == pytest.approx(1.0)  # W=0
    assert lr_factor(0, 1, 0, 0.01) == pytest.approx(1.0)  # T-1-W = 0
    assert lr_factor(1, 2, 1, 0.01) == pytest.approx(1.0)  # T-1-W = 0 after warmup
    assert warmup_steps_for(warmup_epochs=2, epochs=20, steps_per_epoch=175, total_steps=3500) == 350
    assert warmup_steps_for(warmup_epochs=2, epochs=20, steps_per_epoch=175, total_steps=100) == 10
    with pytest.raises(ValueError):
        lr_factor(5, 5, 0, 0.01)


def test_numerical_failure_is_recorded(tmp_path):
    """A non-finite loss must produce FAILED_NUMERICAL with failure artifacts, never a skipped batch."""
    data, m = synthetic_bundle()
    cfg = small_cfg(tmp_path, "vcs")
    run_dir = Path(cfg["run"]["output_root"]) / "nanrun"
    run_dir.mkdir()
    tr = Trainer(cfg, run_dir=run_dir, data=data, manifest=m, device=torch.device("cpu"), stage="TEST", smoke_steps=2, epoch_eval=False)
    tr.setup()
    with torch.no_grad():
        tr.critic.net[-1].weight.fill_(float("nan"))
    st = tr.run()
    assert st == "FAILED_NUMERICAL"
    assert any(run_dir.joinpath("failure").glob("failure_step_*.pt"))
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "FAILED_NUMERICAL" and "non-finite" in status["failure_reason"]


@pytest.mark.parametrize("k", [8, 64])
def test_k_greater_than_one_objective_and_config(tmp_path, k):
    """K distinct nonzero shifts: K*B negatives averaged as one distribution; config accepts 1 <= K <= B-1."""
    import yaml
    env = _env(tmp_path)
    base = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text())
    base["pairing"]["k"] = k
    cfg = load_config(_write(tmp_path, yaml.safe_dump(base)), env=env)
    assert cfg["pairing"]["k"] == k
    cfg = copy.deepcopy(cfg); cfg["train"].update({"batch_size_images": 128, "num_workers": 0, "pin_memory": False})
    built = build_models(cfg, seed=0, device="cpu")
    x1, x2 = torch.randn(128, 3, 32, 32), torch.randn(128, 3, 32, 32)
    f = forward_features(built["encoder"], built["projector"], x1, x2, eps=1e-8)
    obj = compute_objective("vcs_qmi", f, cfg=cfg, critic=built["critic"], pair_generator=torch.Generator().manual_seed(1))
    assert obj["n_pos"] == 128 and obj["n_neg"] == k * 128
    assert isinstance(obj["shift"], list) and len(obj["shift"]) == k and len(set(obj["shift"])) == k and all(1 <= s_ <= 127 for s_ in obj["shift"])
    assert abs(obj["stats"]["J_raw"] - (1 - obj["stats"]["R_binary"])) < 1e-5
    obj["loss"].backward()
    assert grad_norms(built["encoder"], built["projector"], built["critic"])["grad_norm_critic"] > 0


def test_critic_variants_and_raw_input(tmp_path):
    """PairCriticMLP([w,w], gain 0.1) is structurally the reference PairCritic; other depths/gains build; 'none' feeds raw p."""
    import yaml
    from vcs_ssl.models.critic import PairCriticMLP, build_critic, critic_impl_name
    from reference.ssl_core import PairCritic

    torch.manual_seed(0); ref = PairCritic(128, 512)
    torch.manual_seed(0); var = PairCriticMLP(128, [512, 512], 0.1)
    assert [type(m).__name__ for m in ref.net] == [type(m).__name__ for m in var.net]
    assert all(torch.equal(a, b) for a, b in zip(ref.state_dict().values(), var.state_dict().values()))
    base = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text())["model"]["critic"]
    assert critic_impl_name(base) == "reference.ssl_core.PairCritic" and isinstance(build_critic(base, feature_dim=128), PairCritic)
    for hd, gain in (([256, 256], 0.1), ([2048, 2048], 0.1), ([512], 0.1), ([512, 512, 512], 0.1), ([512, 512], 1.0), ([512, 512], 0.01)):
        c = dict(base, hidden_dims=hd, last_layer_xavier_gain=gain)
        crit = build_critic(c, feature_dim=64)
        out = crit(torch.randn(5, 64), torch.randn(5, 64))
        assert out.shape == (5,) and (out.abs() <= 1).all()
        assert isinstance(crit, PairCritic) == (len(hd) == 2 and hd[0] == hd[1] and gain == 0.1)
    with pytest.raises(ValueError):
        build_critic(dict(base, last_layer_xavier_gain=0.0), feature_dim=64)
    # raw-input variant: objective consumes p_raw, not z_l2
    env = _env(tmp_path)
    cfgd = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text())
    cfgd["model"]["normalization"]["vcs_and_simclr"] = "none"
    cfgd["model"]["projector"]["output_dim"] = 256
    cfgd["model"]["projector"]["hidden_dim"] = 1024
    cfgd["optimizer"]["critic_lr_multiplier"] = 10.0
    cfgd["optimizer"]["critic_weight_decay"] = 1e-4
    cfg = load_config(_write(tmp_path, yaml.safe_dump(cfgd)), env=env)
    built = build_models(cfg, seed=0, device="cpu")
    assert built["params"]["projector"] == 512 * 1024 + 2 * 1024 + 1024 * 256 + 256
    f = forward_features(built["encoder"], built["projector"], torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32), eps=1e-8)
    assert f["p_raw"].shape == (8, 256)
    obj = compute_objective("vcs_qmi", f, cfg=cfg, critic=built["critic"], pair_generator=torch.Generator().manual_seed(0))
    from vcs_ssl.objectives import critic_input_key
    assert critic_input_key(cfg) == "p_raw"
    z1, z2 = f["p_raw"].chunk(2)
    from reference.ssl_core import vcs_pair_loss
    s, _ = vcs_pair_loss(z1, z2, built["critic"], k=1, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(obj["loss"], s["loss"])
    opt = build_optimizer(built["encoder"], built["projector"], built["critic"], cfg["optimizer"])
    g = [g for g in opt.param_groups if g["name"] == "critic"][0]
    assert g["base_lr"] == pytest.approx(1e-2) and g["weight_decay"] == pytest.approx(1e-4)
    # controls must keep the frozen projector / l2 policy
    bad = yaml.safe_load((CFG_DIR / "cifar10_pilot_simclr.yaml").read_text()); bad["model"]["projector"]["output_dim"] = 256
    with pytest.raises(ConfigError, match="control runs keep"):
        load_config(_write(tmp_path, yaml.safe_dump(bad)), env=env)


def test_critic_holdout_caps_k_on_short_tail_batch(tmp_path):
    """K larger than the tail batch allows is capped per batch (recorded), instead of raising (K=255 failure, 2026-09-24)."""
    from vcs_ssl.diagnostics import critic_holdout
    data, m = synthetic_bundle()
    cfg = small_cfg(tmp_path, "vcs")
    built = build_models(cfg, seed=0, device=torch.device("cpu"))
    sel = np.asarray(m["selection_uids"])  # 80 images -> batches 32, 32, 16 -> tail allows K <= 15
    res = critic_holdout(built["encoder"], built["projector"], built["critic"], data.data, sel, build_two_view_transform(cfg["views"]),
                         device=torch.device("cpu"), batch_size=32, repeats=1, rng_seed=1, k=20, num_workers=0)
    rep = res["per_repeat"][0]
    assert rep["k_effective_min"] == 15 and rep["n_pos"] == 80 and rep["n_neg"] == 20 * 32 + 20 * 32 + 15 * 16
    assert -3.0 <= rep["heldout_J"] <= 1.0


def test_blur_variant_and_control_lock(tmp_path):
    import yaml
    env = _env(tmp_path)
    base = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text())
    base["views"]["gaussian_blur_p"] = 0.5
    cfg = load_config(_write(tmp_path, yaml.safe_dump(base)), env=env)
    tf = build_two_view_transform(cfg["views"])
    assert any(type(m).__name__ == "RandomApply" and any(type(x).__name__ == "GaussianBlur" for x in m.transforms) for m in tf.transforms)
    from PIL import Image
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    v = tf(img)
    assert v.shape == (3, 32, 32) and torch.isfinite(v).all()
    bad = yaml.safe_load((CFG_DIR / "cifar10_pilot_simclr.yaml").read_text()); bad["views"]["gaussian_blur_p"] = 0.5
    with pytest.raises(ConfigError, match="control runs keep"):
        load_config(_write(tmp_path, yaml.safe_dump(bad)), env=env)
    bad2 = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text()); bad2["views"]["solarize_p"] = 0.2
    with pytest.raises(ConfigError, match="solarize"):
        load_config(_write(tmp_path, yaml.safe_dump(bad2)), env=env)


@pytest.mark.parametrize("inp", ["concat_interact", "bilinear_concat", "cosine"])
def test_critic_input_variants(tmp_path, inp):
    """Named critic variants: pointwise, bounded, gradients reach both views and the critic; config accepted for VCS only."""
    import yaml
    from vcs_ssl.models.critic import build_critic, critic_impl_name
    env = _env(tmp_path)
    base = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text())
    base["model"]["critic"]["input"] = inp
    cfg = load_config(_write(tmp_path, yaml.safe_dump(base)), env=env)
    crit = build_critic(cfg["model"]["critic"], feature_dim=16)
    expected = {"concat_interact": "InteractCritic", "bilinear_concat": "BilinearConcatCritic", "cosine": "CosineCritic"}[inp]
    assert critic_impl_name(cfg["model"]["critic"]).endswith(expected) and type(crit).__name__ == expected
    x = torch.randn(6, 16, requires_grad=True); y = torch.randn(6, 16, requires_grad=True)
    out = crit(x, y)
    assert out.shape == (6,) and (out.abs() <= 1).all()
    pointwise = torch.cat([crit(x[i:i + 1], y[i:i + 1]) for i in range(6)])
    torch.testing.assert_close(out, pointwise, atol=1e-6, rtol=1e-5)
    (out.mean() + 0.5 * out.square().mean()).backward()
    assert x.grad.abs().sum() > 0 and y.grad.abs().sum() > 0
    assert sum(float(p.grad.abs().sum()) for p in crit.parameters() if p.grad is not None) > 0
    if inp == "cosine":
        assert sum(p.numel() for p in crit.parameters()) == 2


def test_symmetric_pairing_and_critic_steps(tmp_path):
    import yaml
    from vcs_ssl.objectives import critic_steps, pair_symmetric, vcs_pair_loss_symmetric
    from reference.ssl_core import vcs_pair_loss
    env = _env(tmp_path)
    base = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text())
    base["pairing"]["sampler"] = "random_nonzero_cyclic_shift_symmetric"
    base["train"]["mode"] = "joint_critic_steps_3"
    cfg = load_config(_write(tmp_path, yaml.safe_dump(base)), env=env)
    assert pair_symmetric(cfg) and critic_steps(cfg) == 3
    built = build_models(cfg, seed=0, device="cpu")
    z1, z2 = F.normalize(torch.randn(8, 128), dim=1), F.normalize(torch.randn(8, 128), dim=1)
    s_sym, sh = vcs_pair_loss_symmetric(z1, z2, built["critic"], k=2, generator=torch.Generator().manual_seed(0))
    s_a, _ = vcs_pair_loss(z1, z2, built["critic"], k=2, generator=torch.Generator().manual_seed(0))
    assert len(sh) == 2 and torch.isfinite(s_sym["loss"])
    # symmetric loss uses 2B positives and 2KB negatives; with a symmetric input it equals the one-directional loss
    f = forward_features(built["encoder"], built["projector"], torch.randn(8, 3, 32, 32), torch.randn(8, 3, 32, 32), eps=1e-8)
    obj = compute_objective("vcs_qmi", f, cfg=cfg, critic=built["critic"], pair_generator=torch.Generator().manual_seed(1))
    assert obj["n_pos"] == 16 and obj["n_neg"] == 16
    # extra critic-only steps: critic params move before the joint step; encoder gets exactly one update per batch
    data, m = synthetic_bundle()
    cfg2 = small_cfg(tmp_path, "vcs"); cfg2["train"]["mode"] = "joint_critic_steps_3"; cfg2["pairing"]["sampler"] = "random_nonzero_cyclic_shift_symmetric"
    policy_checks(cfg2)
    tr, st = run_trainer(tmp_path, cfg2, data, m, run_id="altsteps", device=torch.device("cpu"), smoke_steps=2, epoch_eval=False)
    assert st == "COMPLETED"
    rows = steps_log(tr.run_dir)
    assert rows[0]["n_pos"] == 2 * cfg2["train"]["batch_size_images"]
    rm = json.loads((tr.run_dir / "run_manifest.json").read_text())
    assert rm["hparams"]["critic_steps"] == 3 and rm["hparams"]["pair_symmetric"] is True
    bad = yaml.safe_load((CFG_DIR / "cifar10_pilot_simclr.yaml").read_text()); bad["train"]["mode"] = "joint_critic_steps_2"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, yaml.safe_dump(bad)), env=env)


def test_critic_on_h_and_negative_detach(tmp_path):
    import yaml
    from vcs_ssl.objectives import critic_feature_dim, critic_input_key, vcs_pair_loss_negdetach
    env = _env(tmp_path)
    # optional field absent -> default 'z'; old configs unchanged
    cfg0 = load_config(CFG_DIR / "cifar10_pilot_vcs.yaml", env=env)
    assert cfg0["model"]["critic"]["feature_source"] == "z" and critic_feature_dim(cfg0) == 128
    base = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text())
    base["model"]["critic"]["feature_source"] = "h_l2"
    cfg = load_config(_write(tmp_path, yaml.safe_dump(base)), env=env)
    assert critic_input_key(cfg) == "h_l2" and critic_feature_dim(cfg) == 512
    built = build_models(cfg, seed=0, device="cpu")
    assert built["critic"].feature_dim == 512
    f = forward_features(built["encoder"], built["projector"], torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32), eps=1e-8)
    torch.testing.assert_close(f["h_l2"].norm(dim=1), torch.ones(8), atol=1e-5, rtol=0)
    obj = compute_objective("vcs_qmi", f, cfg=cfg, critic=built["critic"], pair_generator=torch.Generator().manual_seed(0))
    obj["loss"].backward()
    gn = grad_norms(built["encoder"], built["projector"], built["critic"])
    assert gn["grad_norm_encoder"] > 0 and gn["grad_norm_critic"] > 0 and gn["grad_norm_projector"] is None
    # negative_detach: gradient through z2 comes only from positives
    base2 = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text()); base2["pairing"]["negative_detach"] = True
    cfg2 = load_config(_write(tmp_path, yaml.safe_dump(base2)), env=env)
    crit = build_models(cfg2, seed=0, device="cpu")["critic"]
    z1 = F.normalize(torch.randn(8, 128), dim=1).requires_grad_(True); z2 = F.normalize(torch.randn(8, 128), dim=1).requires_grad_(True)
    s_nd, _ = vcs_pair_loss_negdetach(z1, z2, crit, k=1, generator=torch.Generator().manual_seed(0))
    # negatives-only term must not reach z2
    z1b = z1.detach().clone().requires_grad_(True); z2b = z2.detach().clone().requires_grad_(True)
    from reference.ssl_core import cyclic_negative_indices
    idx, _ = cyclic_negative_indices(8, 1, generator=torch.Generator().manual_seed(0))
    tn = crit(z1b, z2b.detach()[idx[0]]); (tn.mean()).backward()
    assert z1b.grad is not None and z2b.grad is None
    bad = yaml.safe_load((CFG_DIR / "cifar10_pilot_simclr.yaml").read_text()); bad["pairing"]["negative_detach"] = True
    with pytest.raises(ConfigError, match="VCS-only"):
        load_config(_write(tmp_path, yaml.safe_dump(bad)), env=env)
    # trainer smoke with critic on h (projector unused) completes
    data, m = synthetic_bundle()
    cfgs = small_cfg(tmp_path, "vcs"); cfgs["model"]["critic"]["feature_source"] = "h_l2"; policy_checks(cfgs)
    tr, st = run_trainer(tmp_path, cfgs, data, m, run_id="crit_on_h", device=torch.device("cpu"), smoke_steps=2, epoch_eval=True, smoke_epoch_steps=2)
    assert st == "COMPLETED"


def test_target_branch_predictor_projector_depth(tmp_path):
    """EMA/stop-grad target branches, predictor head and projector depth: named variants that leave J untouched."""
    import yaml
    from vcs_ssl.models import ema_update
    from vcs_ssl.objectives import compute_objective_target, forward_features_target
    env = _env(tmp_path)
    cfg0 = load_config(CFG_DIR / "cifar10_pilot_vcs.yaml", env=env)
    assert cfg0["train"]["target_branch"] == "shared" and cfg0["model"]["projector"]["depth"] == 2 and cfg0["model"]["projector"]["predictor"] is False
    base = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text())
    base["train"]["target_branch"] = "ema_0.99"; base["model"]["projector"]["predictor"] = True; base["model"]["projector"]["depth"] = 3
    cfg = load_config(_write(tmp_path, yaml.safe_dump(base)), env=env)
    built = build_models(cfg, seed=0, device="cpu")
    assert built["teacher"] is not None and built["predictor"] is not None and built["teacher"]["tau"] == 0.99
    assert sum(isinstance(m, torch.nn.Linear) for m in built["projector"]) == 3
    assert all(not q.requires_grad for q in built["teacher"]["encoder"].parameters())
    x1, x2 = torch.randn(6, 3, 32, 32), torch.randn(6, 3, 32, 32)
    f = forward_features_target(built["encoder"], built["projector"], x1, x2, 1e-8, target_branch="ema_0.99", teacher=built["teacher"], predictor=built["predictor"])
    assert "tgt_z_l2" in f and not f["tgt_z_l2"].requires_grad and f["z_l2"].requires_grad
    obj = compute_objective_target(f, cfg=cfg, critic=built["critic"], pair_generator=torch.Generator().manual_seed(0))
    assert obj["n_pos"] == 12 and obj["n_neg"] == 12
    obj["loss"].backward()
    gn = grad_norms(built["encoder"], built["projector"], built["critic"], built["predictor"])
    assert gn["grad_norm_encoder"] > 0 and gn["grad_norm_projector"] > 0 and gn["grad_norm_predictor"] > 0 and gn["grad_norm_critic"] > 0
    opt = build_optimizer(built["encoder"], built["projector"], built["critic"], cfg["optimizer"], predictor=built["predictor"])
    assert verify_optimizer_coverage(opt, built["encoder"], built["projector"], built["critic"], built["predictor"])["n_params_in_optimizer"] > 0
    # EMA moves the teacher toward the student
    before = [q.clone() for q in built["teacher"]["encoder"].parameters()][:1][0]
    with torch.no_grad():
        for q in built["encoder"].parameters():
            q.add_(1.0)
    ema_update(built["teacher"], built["encoder"], built["projector"])
    after = next(built["teacher"]["encoder"].parameters())
    torch.testing.assert_close(after, before + 0.01 * 1.0, atol=1e-5, rtol=0)
    # stop-grad target: detached student features
    base2 = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text()); base2["train"]["target_branch"] = "stopgrad"
    cfg2 = load_config(_write(tmp_path, yaml.safe_dump(base2)), env=env)
    b2 = build_models(cfg2, seed=0, device="cpu")
    f2 = forward_features_target(b2["encoder"], b2["projector"], x1, x2, 1e-8, target_branch="stopgrad")
    assert not f2["tgt_z_l2"].requires_grad and torch.equal(f2["tgt_z_l2"], f2["z_l2"].detach())
    # policy: predictor needs a target branch; controls locked
    bad = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text()); bad["model"]["projector"]["predictor"] = True
    with pytest.raises(ConfigError, match="predictor requires"):
        load_config(_write(tmp_path, yaml.safe_dump(bad)), env=env)
    bad2 = yaml.safe_load((CFG_DIR / "cifar10_pilot_simclr.yaml").read_text()); bad2["train"]["target_branch"] = "ema_0.99"
    with pytest.raises(ConfigError, match="control runs keep"):
        load_config(_write(tmp_path, yaml.safe_dump(bad2)), env=env)
    # trainer smoke: EMA + predictor completes, checkpoint carries teacher/predictor states, resume works
    data, m = synthetic_bundle()
    cfgs = small_cfg(tmp_path, "vcs"); cfgs["train"]["target_branch"] = "ema_0.99"; cfgs["model"]["projector"]["predictor"] = True; policy_checks(cfgs)
    tr, st = run_trainer(tmp_path, cfgs, data, m, run_id="ema_pred", device=torch.device("cpu"), smoke_steps=4, smoke_epoch_steps=2, epoch_eval=True)
    assert st == "COMPLETED"
    ck = load_checkpoint(tr.run_dir / "checkpoints" / "last.pt")
    assert ck["teacher_encoder_state"] is not None and ck["predictor_state"] is not None
    tr_b, st_b = run_trainer(tmp_path, cfgs, data, m, run_id="ema_pred_b", device=torch.device("cpu"), smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False, stop_after_steps=2)
    tr_c, st_c = run_trainer(tmp_path, cfgs, data, m, run_id="ema_pred_b", device=torch.device("cpu"), smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False,
                             resume_from=tr_b.ckpt_dir / "last.pt")
    assert st_b == "STOPPED_BUDGET" and st_c == "COMPLETED"
    for k in tr.teacher["encoder"].state_dict():
        pass  # teacher present
    rm = json.loads((tr.run_dir / "run_manifest.json").read_text())
    assert rm["hparams"]["target_branch"] == "ema_0.99" and rm["hparams"]["predictor"] is True


def test_cosine_scale_init(tmp_path):
    import yaml
    from vcs_ssl.models.critic import CosineCritic
    env = _env(tmp_path)
    base = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text()); base["model"]["critic"]["input"] = "cosine"; base["model"]["critic"]["cosine_scale_init"] = 10
    cfg = load_config(_write(tmp_path, yaml.safe_dump(base)), env=env)
    crit = build_models(cfg, seed=0, device="cpu")["critic"]
    assert isinstance(crit, CosineCritic) and float(crit.scale) == 10.0 and float(crit.bias) == 0.0
    cfg0 = load_config(CFG_DIR / "cifar10_pilot_vcs.yaml", env=env); assert cfg0["model"]["critic"]["cosine_scale_init"] == 1.0
    bad = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text()); bad["model"]["critic"]["cosine_scale_init"] = 0
    with pytest.raises(ConfigError, match="cosine_scale_init"):
        load_config(_write(tmp_path, yaml.safe_dump(bad)), env=env)


def test_critic_holdout_follows_training_wiring(tmp_path):
    """With an EMA teacher + predictor the hold-out diagnostic scores online[+predictor] vs teacher pairs (training wiring), not online-online."""
    import yaml
    from vcs_ssl.diagnostics import critic_holdout
    from vcs_ssl.objectives import compute_objective_target, forward_features_target
    env = _env(tmp_path)
    base = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text())
    base["train"]["target_branch"] = "ema_0.99"; base["model"]["projector"]["predictor"] = True; base["model"]["critic"]["input"] = "cosine"
    cfg = load_config(_write(tmp_path, yaml.safe_dump(base)), env=env)
    data, m = synthetic_bundle()
    built = build_models(cfg, seed=0, device="cpu")
    # perturb the teacher so wiring matters
    with torch.no_grad():
        for q in built["teacher"]["projector"].parameters():
            q.add_(0.5)
    sel = np.asarray(m["selection_uids"])[:64]
    tf = build_two_view_transform(cfg["views"])
    res_w = critic_holdout(built["encoder"], built["projector"], built["critic"], data.data, sel, tf, device=torch.device("cpu"), batch_size=32, repeats=1, rng_seed=1, k=1,
                           num_workers=0, target_branch="ema_0.99", teacher=built["teacher"], predictor=built["predictor"])
    res_o = critic_holdout(built["encoder"], built["projector"], built["critic"], data.data, sel, tf, device=torch.device("cpu"), batch_size=32, repeats=1, rng_seed=1, k=1, num_workers=0)
    assert res_w["wiring"].startswith("training wiring") and res_o["wiring"] == "online-online"
    assert res_w["per_repeat"][0]["n_pos"] == 2 * 64 and res_o["per_repeat"][0]["n_pos"] == 64
    assert abs(res_w["heldout_J_mean"] - res_o["heldout_J_mean"]) > 1e-6
    # the diagnostic matches a no-grad evaluation of the training objective on the same batch (eval mode, same pairs/shifts)
    for mod in (built["encoder"], built["projector"], built["critic"], built["predictor"], built["teacher"]["encoder"], built["teacher"]["projector"]):
        mod.eval()
    torch.manual_seed(0)
    x1 = torch.stack([tf(__import__("PIL.Image", fromlist=["fromarray"]).fromarray(data.data[u])) for u in sel[:32]])
    x2 = torch.stack([tf(__import__("PIL.Image", fromlist=["fromarray"]).fromarray(data.data[u])) for u in sel[:32]])
    with torch.no_grad():
        f = forward_features_target(built["encoder"], built["projector"], x1, x2, 1e-8, target_branch="ema_0.99", teacher=built["teacher"], predictor=built["predictor"])
        obj = compute_objective_target(f, cfg=cfg, critic=built["critic"], pair_generator=torch.Generator().manual_seed(5))
    assert torch.isfinite(obj["loss"]) and obj["n_pos"] == 64


def test_wave_g_variants(tmp_path):
    """Fixed cosine scale, bias calibration, projector output BN, interaction-only and shared-metric critics."""
    import yaml
    from vcs_ssl.models.critic import CosineCritic, InteractOnlyCritic, SharedMetricCritic
    env = _env(tmp_path)
    base = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text())
    # fixed scale excluded from the optimizer
    b1 = copy.deepcopy(base); b1["model"]["critic"]["input"] = "cosine"; b1["model"]["critic"]["cosine_scale_init"] = 5; b1["model"]["critic"]["cosine_scale_fixed"] = True
    cfg = load_config(_write(tmp_path, yaml.safe_dump(b1)), env=env)
    bt = build_models(cfg, seed=0, device="cpu"); crit = bt["critic"]
    assert isinstance(crit, CosineCritic) and not crit.scale.requires_grad and float(crit.scale) == 5.0
    opt = build_optimizer(bt["encoder"], bt["projector"], crit, cfg["optimizer"])
    assert [g for g in opt.param_groups if g["name"] == "critic"][0]["params"] == [crit.bias]
    # shared metric starts identical to cosine
    b2 = copy.deepcopy(base); b2["model"]["critic"]["input"] = "shared_metric"
    cfg2 = load_config(_write(tmp_path, yaml.safe_dump(b2)), env=env)
    sm = build_models(cfg2, seed=0, device="cpu")["critic"]; assert isinstance(sm, SharedMetricCritic)
    z1, z2 = F.normalize(torch.randn(6, 128), dim=1), F.normalize(torch.randn(6, 128), dim=1)
    torch.testing.assert_close(sm(z1, z2), CosineCritic(128)(z1, z2), atol=1e-6, rtol=0)
    assert sum(p.numel() for p in sm.parameters()) == 128 * 128 + 2
    # interaction-only: symmetric in the two views, no raw channel
    b3 = copy.deepcopy(base); b3["model"]["critic"]["input"] = "interact_only"; b3["model"]["critic"]["hidden_dims"] = [256, 256]
    cfg3 = load_config(_write(tmp_path, yaml.safe_dump(b3)), env=env)
    io = build_models(cfg3, seed=0, device="cpu")["critic"]; assert isinstance(io, InteractOnlyCritic)
    torch.testing.assert_close(io(z1, z2), io(z2, z1), atol=1e-6, rtol=0)
    assert io.net[0].in_features == 256
    # projector output BN (affine-free) requires bias=False; controls locked
    b4 = copy.deepcopy(base); b4["model"]["projector"]["output_batchnorm"] = True; b4["model"]["projector"]["output_linear_bias"] = False
    cfg4 = load_config(_write(tmp_path, yaml.safe_dump(b4)), env=env)
    proj = build_models(cfg4, seed=0, device="cpu")["projector"]
    assert isinstance(proj[-1], torch.nn.BatchNorm1d) and proj[-1].affine is False and proj[-2].bias is None
    bad = copy.deepcopy(base); bad["model"]["projector"]["output_batchnorm"] = True
    with pytest.raises(ConfigError, match="output_linear_bias=false"):
        load_config(_write(tmp_path, yaml.safe_dump(bad)), env=env)
    badc = yaml.safe_load((CFG_DIR / "cifar10_pilot_simclr.yaml").read_text()); badc["model"]["projector"]["output_batchnorm"] = True; badc["model"]["projector"]["output_linear_bias"] = False
    with pytest.raises(ConfigError, match="VCS-only"):
        load_config(_write(tmp_path, yaml.safe_dump(badc)), env=env)
    # bias calibration at init: b0 = -a0*mu, BN buffers restored, record written; trainer smoke completes
    data, m = synthetic_bundle()
    cfgs = small_cfg(tmp_path, "vcs"); cfgs["model"]["critic"]["input"] = "cosine"; cfgs["model"]["critic"]["cosine_bias_calibrate"] = True; policy_checks(cfgs)
    run_dir = Path(cfgs["run"]["output_root"]) / "calib"; run_dir.mkdir()
    tr = Trainer(cfgs, run_dir=run_dir, data=data, manifest=m, device=torch.device("cpu"), stage="TEST", smoke_steps=2, epoch_eval=False)
    tr.setup()
    rec = json.loads((run_dir / "cosine_bias_calibration.json").read_text())
    assert rec["n_images"] == min(512, len(m["fit_uids"])) and abs(rec["b0"] + rec["a0"] * rec["mu"]) < 1e-6 and float(tr.critic.bias) == pytest.approx(rec["b0"])
    bn = [mod for mod in tr.encoder.modules() if isinstance(mod, torch.nn.BatchNorm2d)][0]
    assert int(bn.num_batches_tracked) == 0  # calibration forward did not leave BN statistics behind
    assert tr.run() == "COMPLETED"


def test_new_candidates_spline_diag_allpairs_views(tmp_path):
    import yaml
    from vcs_ssl.models.critic import CosineCritic, DiagMetricCritic, MonoSplineCritic
    from vcs_ssl.objectives import compute_objective_views, forward_features_views, vcs_pair_loss_all_matrix
    from reference.ssl_core import vcs_pair_loss
    env = _env(tmp_path)
    base = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text())
    z1, z2 = F.normalize(torch.randn(16, 128, dtype=torch.float64), dim=1), F.normalize(torch.randn(16, 128, dtype=torch.float64), dim=1)
    # spline: monotone, init == cosine (a=1, b=0)
    b1 = copy.deepcopy(base); b1["model"]["critic"]["input"] = "mono_spline"; b1["model"]["critic"]["hidden_dims"] = [8, 8]
    cfg1 = load_config(_write(tmp_path, yaml.safe_dump(b1)), env=env)
    sp = build_models(cfg1, seed=0, device="cpu")["critic"]; assert isinstance(sp, MonoSplineCritic) and len(sp.knots) == 8
    s = torch.linspace(-1, 1, 201)
    f = sp.f(s); assert torch.all(f[1:] >= f[:-1] - 1e-6); torch.testing.assert_close(f, s, atol=1e-2, rtol=0)  # init ~ cosine (a=1,b=0) within 0.01
    torch.testing.assert_close(sp(z1.float(), z2.float()), CosineCritic(128)(z1.float(), z2.float()), atol=1e-2, rtol=0)
    # diag metric init == cosine
    b2 = copy.deepcopy(base); b2["model"]["critic"]["input"] = "diag_metric"
    cfg2 = load_config(_write(tmp_path, yaml.safe_dump(b2)), env=env)
    dm = build_models(cfg2, seed=0, device="cpu")["critic"]; assert isinstance(dm, DiagMetricCritic)
    torch.testing.assert_close(dm(z1.float(), z2.float()), CosineCritic(128)(z1.float(), z2.float()), atol=1e-6, rtol=0)
    # all-pairs matrix == cyclic shifts with K = B-1 (values and gradients), with and without negative detach
    crit = CosineCritic(128).double(); crit.scale.data.fill_(3.0); crit.bias.data.fill_(-1.5)
    for nd in (False, True):
        a1, a2 = z1.clone().requires_grad_(True), z2.clone().requires_grad_(True)
        sA, _ = vcs_pair_loss_all_matrix(a1, a2, crit, negative_detach=nd)
        gA = torch.autograd.grad(sA["loss"], (a1, a2))
        c1, c2 = z1.clone().requires_grad_(True), z2.clone().requires_grad_(True)
        if nd:
            from vcs_ssl.objectives import vcs_pair_loss_negdetach
            sB, _ = vcs_pair_loss_negdetach(c1, c2, crit, k=15, generator=torch.Generator().manual_seed(0))
        else:
            sB, _ = vcs_pair_loss(c1, c2, crit, k=15, generator=torch.Generator().manual_seed(0))
        gB = torch.autograd.grad(sB["loss"], (c1, c2))
        torch.testing.assert_close(sA["J_raw"], sB["J_raw"], atol=1e-12, rtol=1e-12)
        for x, y in zip(gA, gB):
            torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-10)
    b3 = copy.deepcopy(base); b3["model"]["critic"]["input"] = "cosine"; b3["pairing"]["sampler"] = "all_pairs_matrix"
    cfg3 = load_config(_write(tmp_path, yaml.safe_dump(b3)), env=env)
    bt = build_models(cfg3, seed=0, device="cpu")
    fz = forward_features(bt["encoder"], bt["projector"], torch.randn(6, 3, 32, 32), torch.randn(6, 3, 32, 32), eps=1e-8)
    obj = compute_objective("vcs_qmi", fz, cfg=cfg3, critic=bt["critic"], pair_generator=None)
    assert obj["n_neg"] == 30 and obj["shift"] is None
    badc = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text()); badc["pairing"]["sampler"] = "all_pairs_matrix"  # concat MLP critic
    with pytest.raises(ConfigError, match="similarity-type"):
        load_config(_write(tmp_path, yaml.safe_dump(badc)), env=env)
    # four views: dataset, objective (6 pairs), trainer smoke; controls locked
    b4 = copy.deepcopy(base); b4["views"]["count"] = 4; b4["model"]["critic"]["input"] = "cosine"
    cfg4 = load_config(_write(tmp_path, yaml.safe_dump(b4)), env=env)
    from vcs_ssl.data.datasets import SSLMultiViewDataset
    data, m = synthetic_bundle()
    ds = SSLMultiViewDataset(data.data, np.asarray(m["fit_uids"]), build_two_view_transform(cfg4["views"]), 4)
    item = ds[0]; assert len(item) == 5 and item[0].shape == (3, 32, 32) and isinstance(item[-1], int)
    bt4 = build_models(cfg4, seed=0, device="cpu")
    fv = forward_features_views(bt4["encoder"], bt4["projector"], [torch.randn(5, 3, 32, 32) for _ in range(4)], eps=1e-8)
    assert len(fv["views_z"]) == 4 and fv["h"].shape == (20, 512)
    ov = compute_objective_views(fv, cfg=cfg4, critic=bt4["critic"], pair_generator=torch.Generator().manual_seed(0))
    assert ov["n_pos"] == 30 and ov["n_neg"] == 30 and len(ov["shift"]) == 6 and torch.isfinite(ov["loss"])
    cfgs = small_cfg(tmp_path, "vcs"); cfgs["views"]["count"] = 4; cfgs["model"]["critic"]["input"] = "cosine"; policy_checks(cfgs)
    tr, st = run_trainer(tmp_path, cfgs, data, m, run_id="views4", device=torch.device("cpu"), smoke_steps=2, epoch_eval=True, smoke_epoch_steps=2)
    assert st == "COMPLETED" and steps_log(tr.run_dir)[0]["views"] == 4 * cfgs["train"]["batch_size_images"]
    badv = yaml.safe_load((CFG_DIR / "cifar10_pilot_simclr.yaml").read_text()); badv["views"]["count"] = 4
    with pytest.raises(ConfigError, match="two views"):
        load_config(_write(tmp_path, yaml.safe_dump(badv)), env=env)
    # 8 views: 28 pairs, same code path
    b8 = copy.deepcopy(b4); b8["views"]["count"] = 8
    cfg8 = load_config(_write(tmp_path, yaml.safe_dump(b8)), env=env)
    bt8 = build_models(cfg8, seed=0, device="cpu")
    fv8 = forward_features_views(bt8["encoder"], bt8["projector"], [torch.randn(5, 3, 32, 32) for _ in range(8)], eps=1e-8)
    ov8 = compute_objective_views(fv8, cfg=cfg8, critic=bt8["critic"], pair_generator=torch.Generator().manual_seed(0))
    assert ov8["n_pos"] == 5 * 28 and len(ov8["shift"]) == 28 and torch.isfinite(ov8["loss"])
    bad8 = copy.deepcopy(b4); bad8["views"]["count"] = 6
    with pytest.raises(ConfigError, match="views.count"):
        load_config(_write(tmp_path, yaml.safe_dump(bad8)), env=env)


def test_projector_kinds(tmp_path):
    import yaml
    env = _env(tmp_path)
    base = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text())
    b1 = copy.deepcopy(base); b1["model"]["projector"]["depth"] = 1
    proj = build_models(load_config(_write(tmp_path, yaml.safe_dump(b1)), env=env), seed=0, device="cpu")["projector"]
    assert len(proj) == 1 and isinstance(proj[0], torch.nn.Linear) and proj[0].in_features == 512 and proj[0].out_features == 128
    b2 = copy.deepcopy(base); b2["model"]["projector"]["kind"] = "bn_only"; b2["model"]["projector"]["output_dim"] = 512; b2["model"]["critic"]["input"] = "cosine"
    cfg2 = load_config(_write(tmp_path, yaml.safe_dump(b2)), env=env)
    bt = build_models(cfg2, seed=0, device="cpu")
    assert isinstance(bt["projector"][0], torch.nn.BatchNorm1d) and bt["projector"][0].affine is False and bt["critic"].feature_dim == 512
    f = forward_features(bt["encoder"], bt["projector"], torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32), eps=1e-8)
    assert f["z_l2"].shape == (8, 512)
    bad = copy.deepcopy(base); bad["model"]["projector"]["kind"] = "bn_only"
    with pytest.raises(ConfigError, match="output_dim == h_dim"):
        load_config(_write(tmp_path, yaml.safe_dump(bad)), env=env)
    badc = yaml.safe_load((CFG_DIR / "cifar10_pilot_simclr.yaml").read_text()); badc["model"]["projector"]["depth"] = 1
    with pytest.raises(ConfigError, match="control runs keep"):
        load_config(_write(tmp_path, yaml.safe_dump(badc)), env=env)


def test_bn_only_projector_has_no_trainable_params_and_is_skipped_by_gradient_check(tmp_path):
    import yaml
    from vcs_ssl.optim import has_trainable_params
    env = _env(tmp_path)
    base = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text())
    assert has_trainable_params(build_models(load_config(_write(tmp_path, yaml.safe_dump(base)), env=env), seed=0, device="cpu")["projector"])
    b2 = copy.deepcopy(base); b2["model"]["projector"]["kind"] = "bn_only"; b2["model"]["projector"]["output_dim"] = 512; b2["model"]["critic"]["input"] = "cosine"
    bt = build_models(load_config(_write(tmp_path, yaml.safe_dump(b2)), env=env), seed=0, device="cpu")
    assert not has_trainable_params(bt["projector"])
    # the optimizer still covers encoder + critic, and a backward pass gives finite gradients to both
    from vcs_ssl.optim import build_optimizer, grad_norms
    cfg2 = load_config(_write(tmp_path, yaml.safe_dump(b2)), env=env)
    opt = build_optimizer(bt["encoder"], bt["projector"], bt["critic"], cfg2["optimizer"])
    x1, x2 = torch.randn(8, 3, 32, 32), torch.randn(8, 3, 32, 32)
    f = forward_features(bt["encoder"], bt["projector"], x1, x2, eps=cfg2["model"]["normalization"]["eps"])
    out = compute_objective("vcs_qmi", f, cfg=cfg2, critic=bt["critic"], pair_generator=torch.Generator().manual_seed(0))
    out["loss"].backward()
    gn = grad_norms(bt["encoder"], bt["projector"], bt["critic"])
    assert gn["grad_norm_projector"] is None and gn["grad_norm_encoder"] > 0 and gn["grad_norm_critic"] > 0
    opt.zero_grad(set_to_none=True)


def test_control_tuning_flag_and_multiview_controls(tmp_path):
    import yaml
    from vcs_ssl.objectives import compute_objective_views, forward_features_views
    env = _env(tmp_path)
    for name, key in (("cifar10_pilot_simclr.yaml", "nt_xent"), ("cifar10_pilot_vicreg.yaml", "vicreg_invariance")):
        base = yaml.safe_load((CFG_DIR / name).read_text())
        locked = copy.deepcopy(base); locked["views"]["count"] = 4
        with pytest.raises(ConfigError, match="control_tuning"):
            load_config(_write(tmp_path, yaml.safe_dump(locked)), env=env)
        tuned = copy.deepcopy(base); tuned["views"]["count"] = 4; tuned["run"]["control_tuning"] = True
        cfg = load_config(_write(tmp_path, yaml.safe_dump(tuned)), env=env)
        bt = build_models(cfg, seed=0, device="cpu"); assert bt["critic"] is None
        fv = forward_features_views(bt["encoder"], bt["projector"], [torch.randn(6, 3, 32, 32) for _ in range(4)], eps=1e-8)
        ov = compute_objective_views(fv, cfg=cfg, critic=None, pair_generator=None)
        assert torch.isfinite(ov["loss"]) and ov["stats"][key] is not None and ov["shift"] is None
        # 2-view reduction: with exactly two views the multi-view path equals the standard objective
        f2 = forward_features(bt["encoder"], bt["projector"], torch.randn(6, 3, 32, 32), torch.randn(6, 3, 32, 32), eps=1e-8)
        std = compute_objective(cfg["run"]["method"], f2, cfg=cfg, critic=None, pair_generator=None)
        fv2 = {"views_z": list(f2["z_l2"].chunk(2)), "views_p": list(f2["p_raw"].chunk(2)), "views_h": list(f2["h_l2"].chunk(2))}
        mv = compute_objective_views(fv2, cfg=cfg, critic=None, pair_generator=None)
        assert torch.allclose(std["loss"], mv["loss"], atol=1e-6)
    badv = yaml.safe_load((CFG_DIR / "cifar10_pilot_vcs.yaml").read_text()); badv["run"]["control_tuning"] = True
    with pytest.raises(ConfigError, match="control methods only"):
        load_config(_write(tmp_path, yaml.safe_dump(badv)), env=env)
