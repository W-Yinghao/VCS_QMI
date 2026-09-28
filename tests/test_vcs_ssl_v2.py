"""S line (Server Spec v2 §3.2, §3.4, §6, §12): the CS-K-native objective, the S-Kernel (RFF-tanh) critic, the CIFAR-100 path and the
n_classes plumbing.  Synthetic data only; nothing here is benchmark evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vcs_ssl.checkpoint import load_checkpoint  # noqa: E402
from vcs_ssl.config import ConfigError, load_config, policy_checks  # noqa: E402
from vcs_ssl.data import cifar as cifar_mod  # noqa: E402
from vcs_ssl.data.cifar import CifarTrain, load_cifar100_train, load_train_partition, n_classes_of  # noqa: E402
from vcs_ssl.data.splits import SPLIT_ALGORITHM, build_manifest  # noqa: E402
from vcs_ssl.diagnostics import knn_eval, linear_probe  # noqa: E402
from vcs_ssl.evaluate import evaluate_run  # noqa: E402
from vcs_ssl.kernel_cs import KERNEL_CS_STAT_KEYS, kernel_cs_from_log_grams, kernel_cs_pair_loss, log_gaussian_gram, median_pairwise_distance  # noqa: E402
from vcs_ssl.models.critic import RFFTanhCritic, build_critic, critic_impl_name  # noqa: E402
from vcs_ssl.objectives import compute_objective, compute_objective_views  # noqa: E402
from vcs_ssl.train import Trainer  # noqa: E402

CFG_DIR = REPO / "configs"
DEVICE = torch.device("cpu")


# ----------------------------------------------------------------------------------------------------------------------- helpers
def _env(tmp: Path) -> dict[str, str]:
    for d in ("data", "out", "man"):
        (tmp / d).mkdir(exist_ok=True)
    return {"DATA_ROOT": str(tmp / "data"), "OUTPUT_ROOT": str(tmp / "out"), "MANIFEST_ROOT": str(tmp / "man")}


def _base(name: str = "vcs") -> dict:
    return yaml.safe_load((CFG_DIR / f"cifar10_pilot_{name}.yaml").read_text())


def _load(tmp: Path, d: dict, fname: str = "c.yaml") -> dict:
    p = tmp / fname
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return load_config(p, env=_env(tmp))


def _small(cfg: dict, *, batch: int = 32, epochs: int = 2) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["train"].update({"batch_size_images": batch, "epochs": epochs, "warmup_epochs": 1, "num_workers": 0, "pin_memory": False})
    e = cfg["evaluation"]
    e["knn_epochs"] = [0, epochs]; e["knn"]["k"] = 20; e["spectrum"]["selection_first_sorted_ids"] = 64
    e["critic_validation"].update({"repeats": 2, "batch_size": 32}); e["linear"].update({"epochs": 2})
    cfg["logging"]["step_interval"] = 1; cfg["logging"]["checkpoint_epochs"] = [epochs]
    if cfg["pairing"]["k"] > batch - 1:
        cfg["pairing"]["k"] = batch - 1
    policy_checks(cfg)
    return cfg


def rff(d: dict, m: int = 256, bw: float = 1.0) -> dict:
    d["model"]["critic"]["input"] = "rff_tanh"; d["model"]["critic"]["rff_features"] = m; d["model"]["critic"]["rff_bandwidth_multiple"] = bw
    return d


def kcs(d: dict, bw: float = 1.0) -> dict:
    d["run"]["method"] = "cs_kernel_native"; d["model"]["critic"]["enabled"] = False
    d["objective"]["target"] = "classical_kernel_CS_QMI"; d["objective"]["loss"] = "negative_kernel_cs"
    d["objective"]["kernel_cs_bandwidth_multiple"] = bw; d["objective"]["kernel_cs_chunk"] = 0
    d["pairing"]["k"] = 1; d["pairing"]["negative_detach"] = False; d["evaluation"]["critic_validation"]["enabled"] = False
    return d


def c100(d: dict) -> dict:
    d["data"]["name"] = "cifar100"; d["data"]["val_per_class"] = 50; d["data"]["manifest"] = "${MANIFEST_ROOT}/cifar100_dev45k_val5k.json"
    return d


def synthetic(n_per_class: int = 64, n_classes: int = 10, seed: int = 0, val_per_class: int = 8) -> tuple[CifarTrain, dict]:
    rng = np.random.default_rng(seed)
    n = n_classes * n_per_class
    images = rng.integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
    targets = rng.permutation(np.repeat(np.arange(n_classes), n_per_class))
    data = CifarTrain(data=images, targets=targets, root="synthetic", file_hashes={"synthetic": {"md5": "0", "sha256": "0", "bytes": "0"}},
                      source={"dataset": f"synthetic-{n_classes}"})
    return data, build_manifest(targets, split_seed=20260924, val_per_class=val_per_class, source=data.source, file_hashes=data.file_hashes)


def run_trainer(cfg: dict, data, manifest, *, run_id: str, **kw) -> tuple[Trainer, str]:
    run_dir = Path(cfg["run"]["output_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tr = Trainer(cfg, run_dir=run_dir, data=data, manifest=manifest, device=DEVICE, stage="TEST", **kw)
    tr.setup()
    return tr, tr.run()


def _feats(b: int = 24, d: int = 128, seed: int = 0, grad: bool = False) -> dict:
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(2 * b, d, generator=g)
    p.requires_grad_(grad)
    return {"h": torch.randn(2 * b, 512, generator=g), "p_raw": p, "z_l2": F.normalize(p, dim=1), "h_l2": F.normalize(torch.randn(2 * b, 512, generator=g), dim=1)}


# ----------------------------------------------------------------------------------------------------------------------- config policy
def test_config_policy_new_methods_and_datasets(tmp_path):
    ok = _load(tmp_path, rff(_base()))
    assert ok["model"]["critic"]["input"] == "rff_tanh" and ok["model"]["critic"]["rff_features"] == 256
    with pytest.raises(ConfigError):
        _load(tmp_path, rff(_base(), m=0))
    with pytest.raises(ConfigError):
        _load(tmp_path, rff(_base(), bw=0.0))
    bad = rff(_base()); bad["model"]["critic"]["feature_source"] = "h_l2"
    with pytest.raises(ConfigError):
        _load(tmp_path, bad)
    okk = _load(tmp_path, kcs(_base()))
    assert okk["run"]["method"] == "cs_kernel_native" and okk["objective"]["kernel_cs_chunk"] == 0
    four = kcs(_base()); four["views"]["count"] = 4
    assert _load(tmp_path, four)["views"]["count"] == 4  # native multi-view averaging, no control_tuning flag needed
    for mut in (lambda d: d["objective"].__setitem__("kernel_cs_bandwidth_multiple", 0.0),
                lambda d: d["objective"].__setitem__("target", "mixture_reference_I_VQ"),
                lambda d: d["objective"].__setitem__("loss", "negative_J"),
                lambda d: d["model"]["normalization"].__setitem__("vcs_and_simclr", "none"),
                lambda d: d["model"]["critic"].__setitem__("enabled", True),
                lambda d: d["pairing"].__setitem__("negative_detach", True),
                lambda d: d["objective"].__setitem__("kernel_cs_chunk", -1)):
        d = kcs(_base()); mut(d)
        with pytest.raises(ConfigError):
            _load(tmp_path, d)
    c = _load(tmp_path, c100(_base()))
    assert c["data"]["name"] == "cifar100" and c["data"]["val_per_class"] == 50
    for mut in (lambda d: d["data"].__setitem__("val_per_class", 0), lambda d: d["data"].__setitem__("manifest", "${MANIFEST_ROOT}/cifar10_dev45k_val5k.json"),
                lambda d: d["data"].__setitem__("name", "cifar20"), lambda d: d["data"].__setitem__("official_test_accessible", True),
                lambda d: d["data"].__setitem__("split", "dev40k_val10k")):
        d = c100(_base()); mut(d)
        with pytest.raises(ConfigError):
            _load(tmp_path, d)
    d = _base(); d["data"]["manifest"] = "${MANIFEST_ROOT}/cifar100_dev45k_val5k.json"  # cifar10 config must not point at the cifar100 manifest
    with pytest.raises(ConfigError):
        _load(tmp_path, d)


def test_generated_s_line_configs_pass_policy(tmp_path):
    reg = json.loads((CFG_DIR / "S_LINE_SHA256.json").read_text())
    env = {k: os.environ.get(k) or v for k, v in _env(tmp_path).items()}
    n = 0
    for st, block in reg["stages"].items():
        for u in block["units"]:
            p = REPO / u["config"]
            assert p.is_file(), p
            assert hashlib.sha256(p.read_bytes()).hexdigest() == u["sha256"], f"{p} changed since generation"
            raw = yaml.safe_load(p.read_text())
            if raw["data"]["name"] == "cifar100" and not Path(raw["data"]["root"]).is_dir():
                continue  # data link not created yet on this machine
            cfg = load_config(p, env=env, require_dirs=Path(raw["data"]["root"] if "${" not in raw["data"]["root"] else env["DATA_ROOT"]).is_dir())
            assert cfg["data"]["official_test_accessible"] is False and cfg["evaluation"]["official_test_enabled"] is False
            if "skernel" in u["run_id"]:
                assert cfg["model"]["critic"]["input"] == "rff_tanh" and cfg["pairing"]["negative_detach"] is True and cfg["pairing"]["k"] == 8
            if "kcs" in u["run_id"]:
                assert cfg["run"]["method"] == "cs_kernel_native" and cfg["model"]["critic"]["enabled"] is False
            if st == "P89_s4_aug_interaction":
                assert cfg["views"]["random_resized_crop"]["scale"][0] == 0.08 and cfg["views"]["color_jitter"]["brightness"] == 0.8 and cfg["views"]["count"] == 4
            if st == "P91_cifar100":
                assert cfg["data"]["name"] == "cifar100" and cfg["data"]["val_per_class"] == 50 and "cifar100_" in Path(cfg["data"]["manifest"]).name
            n += 1
    assert n >= 20


# ----------------------------------------------------------------------------------------------------------------------- RFF critic
def test_rff_critic_forward_grad_and_state_roundtrip():
    torch.manual_seed(0)
    c = RFFTanhCritic(16, 64, 1.0)
    assert int(c.calibrated) == 0 and float(c.sigma) == 1.0
    assert sum(p.numel() for p in c.parameters() if p.requires_grad) == 65  # theta: m weights + intercept
    assert all(n.startswith("readout") for n, _ in c.named_parameters())  # Omega0 / phase are fixed buffers
    s = c.set_bandwidth(0.9)
    assert s == pytest.approx(0.9) and int(c.calibrated) == 1
    z1 = F.normalize(torch.randn(10, 16), dim=1).requires_grad_(True)
    z2 = F.normalize(torch.randn(10, 16), dim=1).requires_grad_(True)
    t = c(z1, z2)
    assert t.shape == (10,) and bool((t.abs() < 1).all())
    t.sum().backward()
    assert z1.grad is not None and z2.grad is not None and float(z1.grad.norm()) > 0 and float(c.readout.weight.grad.norm()) > 0
    c2 = RFFTanhCritic(16, 64, 1.0)
    c2.load_state_dict(c.state_dict())
    assert float(c2.sigma) == pytest.approx(0.9) and torch.equal(c2.omega0, c.omega0) and torch.allclose(c2(z1, z2), t)
    with pytest.raises(ValueError):
        c.set_bandwidth(0.0)
    with pytest.raises(ValueError):
        c(z1[:, :8], z2[:, :8])
    cc = {"enabled": True, "input": "rff_tanh", "hidden_dims": [512, 512], "activation": "relu", "output": "tanh", "batchnorm": False, "dropout": 0.0,
          "last_layer_xavier_gain": 0.1, "last_layer_bias": 0.0, "rff_features": 128, "rff_bandwidth_multiple": 2.0}
    assert critic_impl_name(cc).endswith("RFFTanhCritic")
    b = build_critic(cc, feature_dim=32)
    assert isinstance(b, RFFTanhCritic) and b.n_features == 128 and b.bandwidth_multiple == 2.0 and b.feature_dim == 32


# ----------------------------------------------------------------------------------------------------------------------- kernel CS
def test_kernel_cs_extremes_gradcheck_chunk_and_estim_equivalence():
    B = 7
    ones = torch.zeros(B, B, dtype=torch.float64)  # log of the all-ones kernel
    assert abs(float(kernel_cs_from_log_grams(ones, ones)["D_CS"])) < 1e-12
    eye = torch.full((B, B), -math.inf, dtype=torch.float64); eye.fill_diagonal_(0.0)  # log of the identity kernel
    out = kernel_cs_from_log_grams(eye, eye)
    assert float(out["D_CS"]) == pytest.approx(math.log(B), abs=1e-12)  # plug-in diagonal effect, not dependence
    assert float(out["underflow_frac"]) == pytest.approx((B * B - B) / (B * B))
    g = torch.Generator().manual_seed(3)
    z1 = F.normalize(torch.randn(6, 3, generator=g, dtype=torch.float64), dim=1).requires_grad_(True)
    z2 = F.normalize(torch.randn(6, 3, generator=g, dtype=torch.float64), dim=1).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a, b: kernel_cs_pair_loss(a, b, sigma=0.7)["loss"], (z1, z2), atol=1e-6, rtol=1e-5)
    x = F.normalize(torch.randn(50, 5, generator=g, dtype=torch.float64), dim=1); y = F.normalize(torch.randn(50, 5, generator=g, dtype=torch.float64), dim=1)
    dense = kernel_cs_pair_loss(x, y, sigma=0.8); chunked = kernel_cs_pair_loss(x, y, sigma=0.8, chunk=7)
    assert float(dense["kernel_cs"]) == pytest.approx(float(chunked["kernel_cs"]), abs=1e-12)
    for k in KERNEL_CS_STAT_KEYS:
        assert k in dense and torch.is_tensor(dense[k])
    from vcs_estim.kernel_cs import kernel_cs_native  # noqa: PLC0415  (E-line module, read-only use)
    ref = kernel_cs_native(x, y, 0.8, 0.8, effective=False)
    assert float(ref["D_CS"]) == pytest.approx(float(dense["kernel_cs"]), abs=1e-9)
    xa = x.clone().requires_grad_(True); ya = y.clone().requires_grad_(True)
    kernel_cs_pair_loss(xa, ya, sigma=0.8)["loss"].backward()
    assert float(xa.grad.norm()) > 0 and float(ya.grad.norm()) > 0  # native autodiff through both sides, no detach
    d = torch.cdist(x, x); iu = torch.triu_indices(50, 50, 1)
    assert median_pairwise_distance(x) == pytest.approx(float(d[iu[0], iu[1]].median()))
    with pytest.raises(ValueError):
        log_gaussian_gram(x, 0.0)


def test_objective_paths_for_new_methods(tmp_path):
    cfg_k = _load(tmp_path, kcs(_base()), "k.yaml")
    f = _feats(grad=True)
    out = compute_objective("cs_kernel_native", f, cfg=cfg_k, kernel_sigma=0.8)
    assert torch.isfinite(out["loss"]) and out["stats"]["kernel_cs"] is not None and out["stats"]["kcs_sigma"] == 0.8 and out["stats"]["J_raw"] is None
    assert out["n_pos"] == 24 and out["n_neg"] == 24 * 23 and out["shift"] is None
    out["loss"].backward(); assert float(f["p_raw"].grad.norm()) > 0
    with pytest.raises(ValueError):
        compute_objective("cs_kernel_native", _feats(), cfg=cfg_k, kernel_sigma=None)
    cfg_k4 = _load(tmp_path, (lambda d: (d["views"].__setitem__("count", 4), d)[1])(kcs(_base())), "k4.yaml")
    zs = [F.normalize(torch.randn(24, 128), dim=1) for _ in range(4)]
    ov = compute_objective_views({"views_z": zs, "views_p": zs, "views_h": zs}, cfg=cfg_k4, critic=None, pair_generator=None, kernel_sigma=0.8)
    assert ov["n_pos"] == 24 * 6 and torch.isfinite(ov["loss"]) and ov["stats"]["kernel_cs"] is not None
    cfg_r = _load(tmp_path, rff(_base()), "r.yaml")
    crit = build_critic(cfg_r["model"]["critic"], feature_dim=128); crit.set_bandwidth(1.0)
    f2 = _feats(grad=True)
    o = compute_objective("vcs_qmi", f2, cfg=cfg_r, critic=crit, pair_generator=torch.Generator().manual_seed(0))
    assert o["stats"]["J_raw"] is not None and -3 <= o["stats"]["J_raw"] <= 1 and o["n_neg"] == 24 * cfg_r["pairing"]["k"]
    o["loss"].backward(); assert float(f2["p_raw"].grad.norm()) > 0 and float(crit.readout.weight.grad.norm()) > 0


# ----------------------------------------------------------------------------------------------------------------------- CIFAR-100 loader
def _fake_c100(tmp: Path, n_classes: int = 100, per_class: int = 3, seed: int = 1):
    base = tmp / "cifar-100-python"; base.mkdir()
    rng = np.random.default_rng(seed)
    n = n_classes * per_class
    data = rng.integers(0, 256, size=(n, 3072), dtype=np.uint8)
    fine = rng.permutation(np.repeat(np.arange(n_classes), per_class))
    with open(base / "train", "wb") as fh:
        pickle.dump({"data": data, "fine_labels": fine.tolist(), "coarse_labels": [0] * n, "filenames": [f"{i}.png" for i in range(n)]}, fh)
    with open(base / "meta", "wb") as fh:
        pickle.dump({"fine_label_names": [f"c{i}" for i in range(n_classes)], "coarse_label_names": ["x"]}, fh)
    with open(base / "test", "wb") as fh:
        pickle.dump({"data": data[:10], "fine_labels": fine[:10].tolist()}, fh)
    md5 = {f: hashlib.md5((base / f).read_bytes()).hexdigest() for f in ("train", "meta", "test")}
    return fine, md5


def _patch_c100(monkeypatch, md5: dict, fine: np.ndarray, n_classes: int, per_class: int) -> None:
    from torchvision.datasets import CIFAR100  # noqa: PLC0415
    monkeypatch.setattr(cifar_mod, "CIFAR100_TRAIN", ("train", md5["train"]))
    monkeypatch.setattr(cifar_mod, "CIFAR100_META", ("meta", md5["meta"]))
    monkeypatch.setattr(cifar_mod, "CIFAR100_TEST", ("test", md5["test"]))
    monkeypatch.setattr(cifar_mod, "CIFAR100_N_TRAIN", n_classes * per_class)
    monkeypatch.setattr(cifar_mod, "CIFAR100_N_CLASSES", n_classes)
    monkeypatch.setattr(cifar_mod, "CIFAR100_PER_CLASS", per_class)
    monkeypatch.setattr(cifar_mod, "CIFAR100_FIRST10_LABELS", fine[:10].tolist())
    monkeypatch.setattr(CIFAR100, "train_list", [["train", md5["train"]]])
    monkeypatch.setattr(CIFAR100, "test_list", [["test", md5["test"]]])
    monkeypatch.setattr(CIFAR100, "meta", {**CIFAR100.meta, "md5": md5["meta"]})


def test_cifar100_loader_synthetic(tmp_path, monkeypatch):
    fine, md5 = _fake_c100(tmp_path)
    _patch_c100(monkeypatch, md5, fine, 100, 3)
    d = load_cifar100_train(tmp_path)
    assert d.data.shape == (300, 32, 32, 3) and d.data.dtype == np.uint8 and np.array_equal(d.targets, fine)
    assert set(d.file_hashes) == {"train", "meta", "test_UNREAD"} and d.file_hashes["train"]["md5"] == md5["train"]
    assert "training partition only" in d.source["dataset"] and len(d.source["classes"]) == 100 and n_classes_of(d) == 100
    assert np.array_equal(load_train_partition("cifar100", tmp_path).targets, fine)
    with pytest.raises(ValueError):
        load_train_partition("cifar20", tmp_path)
    monkeypatch.setattr(cifar_mod, "CIFAR100_FIRST10_LABELS", [0] * 10)
    with pytest.raises(ValueError, match="sentinel"):
        load_cifar100_train(tmp_path)
    monkeypatch.setattr(cifar_mod, "CIFAR100_FIRST10_LABELS", fine[:10].tolist())
    monkeypatch.setattr(cifar_mod, "CIFAR100_PER_CLASS", 4)
    with pytest.raises(ValueError, match="class counts"):
        load_cifar100_train(tmp_path)
    monkeypatch.setattr(cifar_mod, "CIFAR100_PER_CLASS", 3)
    monkeypatch.setattr(cifar_mod, "CIFAR100_TRAIN", ("train", "0" * 32))
    with pytest.raises(ValueError, match="md5"):
        load_cifar100_train(tmp_path)
    (tmp_path / "cifar-100-python" / "meta").unlink()
    monkeypatch.setattr(cifar_mod, "CIFAR100_TRAIN", ("train", md5["train"]))
    with pytest.raises(FileNotFoundError):
        load_cifar100_train(tmp_path)


def test_manifest_100_classes_and_split_text():
    rng = np.random.default_rng(0)
    t = rng.permutation(np.repeat(np.arange(100), 60))
    m = build_manifest(t, split_seed=20260924, val_per_class=50, source={"dataset": "CIFAR-100 python version, official training partition only (fine labels)"}, file_hashes={})
    assert m["n_classes"] == 100 and m["n_selection"] == 5000 and m["n_fit"] == 1000 and set(m["selection_class_counts"]) == {50} and set(m["fit_class_counts"]) == {10}
    assert "0..99" in m["split_algorithm"] and "0..9 " not in m["split_algorithm"] and "CIFAR-100" in m["uid_definition"]
    from vcs_ssl.data.splits import load_manifest, write_manifest  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    with tempfile.TemporaryDirectory() as td:
        write_manifest(m, Path(td) / "cifar100_dev45k_val5k.json")
        assert load_manifest(Path(td) / "cifar100_dev45k_val5k.json", expected_seed=20260924, expected_val_per_class=50)["n_classes"] == 100
        with pytest.raises(ValueError, match="val_per_class"):  # a cifar10-style config value (500) is refused against the cifar100 manifest
            load_manifest(Path(td) / "cifar100_dev45k_val5k.json", expected_seed=20260924, expected_val_per_class=500)
    t10 = rng.permutation(np.repeat(np.arange(10), 20))
    m10 = build_manifest(t10, split_seed=20260924, val_per_class=5, source={"dataset": "x"}, file_hashes={})
    assert m10["split_algorithm"] == SPLIT_ALGORITHM and m10["n_classes"] == 10  # CIFAR-10 manifest text (and hash) unchanged


def test_knn_and_linear_probe_n_classes_plumbing():
    g = torch.Generator().manual_seed(0)
    hb, hq = torch.randn(400, 32, generator=g), torch.randn(100, 32, generator=g)
    yb, yq = torch.randint(0, 100, (400,), generator=g), torch.randint(0, 100, (100,), generator=g)
    r = knn_eval(hb, yb, hq, yq, k=5, n_classes=100)
    assert 0.0 <= r["knn_val_top1_pct"] <= 100.0 and r["n_query"] == 100
    with pytest.raises((RuntimeError, IndexError)):
        knn_eval(hb, yb, hq, yq, k=5)  # default 10 votes cannot hold 100 labels: the plumbing is required
    lcfg = _base()["evaluation"]["linear"]; lcfg["epochs"] = 1
    lp = linear_probe(hb, yb, hq, yq, lcfg, device=DEVICE, n_classes=100)
    assert lp["probe_epochs"] == 1 and 0.0 <= lp["linear_val_top1_pct"] <= 100.0 and lp["n_selection"] == 100


# ----------------------------------------------------------------------------------------------------------------------- trainer smokes
@pytest.mark.parametrize("kind,views", [("rff", 2), ("rff", 4), ("kcs", 2), ("kcs", 4)])
def test_trainer_smoke_rff_and_kernel_cs(tmp_path, kind, views):
    data, m = synthetic()
    d = rff(_base()) if kind == "rff" else kcs(_base())
    d["views"]["count"] = views
    cfg = _small(_load(tmp_path, d, f"{kind}{views}.yaml"))
    tr, st = run_trainer(cfg, data, m, run_id=f"{kind}{views}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=(views == 2))
    assert st == "COMPLETED", tr.failure_reason
    cal = json.loads((tr.run_dir / "bandwidth_calibration.json").read_text())
    rm = json.loads((tr.run_dir / "run_manifest.json").read_text())
    ck = load_checkpoint(tr.run_dir / "checkpoints" / "last.pt")
    g = json.loads((tr.run_dir / "first_step_gradients.json").read_text())
    steps = [json.loads(l) for l in (tr.run_dir / "logs" / "steps.jsonl").read_text().splitlines() if l.strip()]
    assert cal["sigma"] > 0 and cal["median_pair_distance"] > 0 and rm["bandwidth_calibration"]["sigma"] == cal["sigma"] and rm["n_classes"] == 10
    assert g["check"] == "PASS" and g["grad_norm_encoder"] > 0 and g["grad_norm_projector"] > 0
    if kind == "rff":
        assert cal["kind"] == "rff_tanh" and cal["trainable_params"] == 257 and g["grad_norm_critic"] > 0
        assert float(ck["critic_state"]["sigma"]) == pytest.approx(cal["sigma"]) and int(ck["critic_state"]["calibrated"]) == 1 and ck["kernel_sigma"] is None
        assert all(r["J_raw"] is not None and abs(r["J_raw"] - (1 - r["R_binary"])) < 1e-5 for r in steps)
        if views == 2:
            ch = json.loads((tr.run_dir / "evaluations" / "critic_holdout_epoch_002.json").read_text())
            assert -3 <= ch["heldout_J_mean"] <= 1
    else:
        assert cal["kind"] == "cs_kernel_native" and cal["bandwidth_multiple"] == 1.0 and cal["sigma"] == pytest.approx(cal["median_pair_distance"])
        assert ck["critic_state"] is None and ck["kernel_sigma"] == pytest.approx(cal["sigma"]) and g["grad_norm_critic"] is None
        assert all(r["kernel_cs"] is not None and r["J_raw"] is None and r["kcs_sigma"] == pytest.approx(cal["sigma"]) for r in steps)
        assert rm["K"] is None and "kernel CS" in rm["pair_sampling"]
    if views == 2:
        knn = json.loads((tr.run_dir / "evaluations" / "knn_epoch_002.json").read_text())
        assert 0 <= knn["knn_val_top1_pct"] <= 100


def test_kernel_cs_resume_keeps_and_checks_sigma(tmp_path):
    data, m = synthetic()
    cfg = _small(_load(tmp_path, kcs(_base()), "kr.yaml"))
    tr, st = run_trainer(cfg, data, m, run_id="kres", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False, stop_after_steps=2)
    assert st == "STOPPED_BUDGET"
    ck1 = load_checkpoint(tr.run_dir / "checkpoints" / "last.pt")
    assert ck1["optimizer_step"] == 2 and ck1["kernel_sigma"] > 0
    tr2, st2 = run_trainer(cfg, data, m, run_id="kres", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False, resume_from=tr.run_dir / "checkpoints" / "last.pt")
    assert st2 == "COMPLETED" and tr2.step == 4 and tr2.kernel_sigma == pytest.approx(ck1["kernel_sigma"])
    ck2 = load_checkpoint(tr.run_dir / "checkpoints" / "last.pt")
    assert ck2["kernel_sigma"] == ck1["kernel_sigma"]
    bad = dict(ck1); bad["kernel_sigma"] = ck1["kernel_sigma"] * 1.5
    torch.save(bad, tr.run_dir / "checkpoints" / "tampered.pt")
    tr3 = Trainer(cfg, run_dir=tr.run_dir, data=data, manifest=m, device=DEVICE, stage="TEST", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False,
                  resume_from=tr.run_dir / "checkpoints" / "tampered.pt")
    with pytest.raises(ConfigError, match="kernel_sigma"):
        tr3.setup()


def test_trainer_and_evaluate_synthetic_100_classes(tmp_path):
    data, m = synthetic(n_per_class=56, n_classes=100, val_per_class=50)
    assert m["n_classes"] == 100 and m["n_selection"] == 5000
    cfg = _small(_load(tmp_path, c100(_base()), "c100.yaml"))
    cfg["evaluation"]["critic_validation"].update({"repeats": 1, "batch_size": 256})
    tr, st = run_trainer(cfg, data, m, run_id="c100", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=True)
    assert st == "COMPLETED", tr.failure_reason
    assert tr.n_classes == 100
    knn = json.loads((tr.run_dir / "evaluations" / "knn_epoch_002.json").read_text())
    assert knn["knn"]["n_bank"] == 600 and knn["knn"]["n_query"] == 5000 and 0 <= knn["knn_val_top1_pct"] <= 100
    res = evaluate_run(tr.run_dir, "last.pt", protocol="pilot", device=DEVICE, num_workers=0, data=data)
    assert res["n_classes"] == 100 and res["dataset"] == "cifar100" and 0 <= res["linear_val_top1_pct"] <= 100 and res["linear"]["n_fit"] == 600
    with pytest.raises(PermissionError, match="CIFAR-10 protocol only"):
        evaluate_run(tr.run_dir, "last.pt", protocol="final_official_test", device=DEVICE, num_workers=0, standin_selection=True, data=data)
    # a 10-class manifest with 100-class data (or vice versa) is refused
    data10, m10 = synthetic()
    with pytest.raises(ConfigError, match="n_classes"):
        Trainer(cfg, run_dir=tr.run_dir, data=data, manifest=m10, device=DEVICE, stage="TEST", smoke_steps=2, smoke_epoch_steps=1)
