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
                           (("pairing", "k"), 0, "pairing.k"), (("pairing", "k"), 256, "pairing.k"), (("pairing", "negative_detach"), True, "pairing policy"),
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
