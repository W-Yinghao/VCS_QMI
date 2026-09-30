"""P100 (owner 2026-09-30): momentum-encoder key queue for the 4-view VCS recipe and tuned SimCLR.  Synthetic data only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO, REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from reference.ssl_core import simclr_nt_xent  # noqa: E402
from vcs_ssl.checkpoint import load_checkpoint  # noqa: E402
from vcs_ssl.config import ConfigError, load_config  # noqa: E402
from vcs_ssl.models.critic import CosineCritic  # noqa: E402
from vcs_ssl.objectives import NegativeQueue, compute_objective_views, simclr_nt_xent_plus_queue, vcs_pair_loss_momentum_queue  # noqa: E402
from test_vcs_ssl_v2 import _base, _env, _load, _small, run_trainer, synthetic  # noqa: E402

D = 128


def mq(d: dict, method: str = "vcs") -> dict:
    if method == "vcs":
        c = d["model"]["critic"]; c["input"] = "cosine"; c["cosine_scale_init"] = 5.0; c["hidden_dims"] = [512, 512]
        d["pairing"]["negative_detach"] = True
    else:
        d["run"]["control_tuning"] = True
    d["views"]["count"] = 4
    d["pairing"].update({"negative_source": "queue", "queue": True, "queue_size": 64, "momentum_encoder": True, "momentum_m": 0.9})
    return d


def _cfg(tmp_path, method="vcs", name="c.yaml", epochs=2):
    d = mq(_base("vcs" if method == "vcs" else "simclr"), method)
    return _small(_load(tmp_path, d, name), epochs=epochs)


# ------------------------------------------------------------------------------------------------------------ units
def test_queue_fifo_size_and_fast_sampler():
    q = NegativeQueue(8, 3)
    for i in range(5):
        q.enqueue(torch.full((3, 3), float(i)))
    assert q.filled == 8
    f = q.features()
    assert f.shape == (8, 3) and float(f[0, 0]) == 2.0 and float(f[-1, 0]) == 4.0  # oldest kept = batch 2 (last 2 rows), newest last
    g = torch.Generator().manual_seed(0)
    idx = q.sample_indices_fast(5, 4, g)
    assert idx.shape == (4, 5) and int(idx.min()) >= 0 and int(idx.max()) < 8


def test_nt_xent_plus_empty_queue_equals_reference_and_keys_no_grad():
    torch.manual_seed(0)
    a, b = torch.randn(16, D, requires_grad=True), torch.randn(16, D, requires_grad=True)
    ref = simclr_nt_xent(a, b, temperature=0.2)
    got = simclr_nt_xent_plus_queue(a, b, torch.zeros(0, D), temperature=0.2)
    assert torch.allclose(ref, got, atol=1e-6)
    keys = torch.randn(32, D, requires_grad=True)
    simclr_nt_xent_plus_queue(a, b, keys, temperature=0.2).backward()
    assert keys.grad is None and a.grad is not None
    q = NegativeQueue(32, D); q.enqueue(torch.randn(32, D))
    crit = CosineCritic(D, scale_init=5.0)
    z1, z2 = F.normalize(torch.randn(16, D), dim=1).requires_grad_(True), F.normalize(torch.randn(16, D), dim=1).requires_grad_(True)
    s = vcs_pair_loss_momentum_queue(z1, z2, crit, k=4, queue=q, generator=torch.Generator().manual_seed(1))
    s["loss"].backward()
    assert q.buffer.grad is None and torch.isfinite(s["loss"])


def test_views_objective_fallback_then_queue():
    torch.manual_seed(0)
    views = [F.normalize(torch.randn(8, D), dim=1) for _ in range(4)]
    feats = {"views_z": views, "views_p": views, "views_h": views}
    cfg = {"run": {"method": "vcs_qmi"}, "objective": {"loss": "negative_J", "simclr_temperature": 0.2},
           "pairing": {"k": 4, "negative_detach": True, "sampler": "random_nonzero_cyclic_shift"}, "model": {"critic": {"feature_source": "z", "input": "cosine"}, "normalization": {"vcs_and_simclr": "l2"}}}
    crit = CosineCritic(D, scale_init=5.0)
    q = NegativeQueue(16, D)
    out = compute_objective_views(feats, cfg=cfg, critic=crit, pair_generator=torch.Generator().manual_seed(0), queue=q)
    assert out["stats"]["queue_fallback"] == 1.0
    q.enqueue(torch.randn(16, D))
    out = compute_objective_views(feats, cfg=cfg, critic=crit, pair_generator=torch.Generator().manual_seed(0), queue=q)
    assert out["stats"]["queue_fallback"] == 0.0 and out["n_neg"] == 8 * 4 * 6


def test_config_policy_and_old_hashes(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg["pairing"]["momentum_encoder"] is True
    d = mq(_base("vcs")); d["views"]["count"] = 2
    with pytest.raises(ConfigError):
        _load(tmp_path, d, "bad2v.yaml")
    d = mq(_base("vcs")); d["pairing"]["momentum_m"] = 1.0
    with pytest.raises(ConfigError):
        _load(tmp_path, d, "badm.yaml")
    d = _base("vcs"); d["pairing"]["momentum_m"] = 0.9
    with pytest.raises(ConfigError):
        _load(tmp_path, d, "badm2.yaml")
    rc = load_config(REPO / "configs" / "cifar10_hpK_a5_views4_800ep_vcs_seed0.yaml", env=_env(tmp_path))
    assert "momentum_encoder" not in rc["pairing"] and "momentum_m" not in rc["pairing"]


# ------------------------------------------------------------------------------------------------------------ trainer
@pytest.mark.parametrize("method", ["vcs", "simclr"])
def test_trainer_ema_update_exact_and_keys(tmp_path, method):
    data, m = synthetic()
    cfg = _cfg(tmp_path, method, f"t{method}.yaml")
    run_dir = Path(cfg["run"]["output_root"]) / f"e{method}"; run_dir.mkdir(parents=True, exist_ok=True)
    from vcs_ssl.train import Trainer
    tr = Trainer(cfg, run_dir=run_dir, data=data, manifest=m, device=torch.device("cpu"), stage="TEST", smoke_steps=3, smoke_epoch_steps=3, epoch_eval=False)
    tr.setup()
    assert all(not p.requires_grad for mod in tr.key_model.values() for p in mod.parameters())
    assert all(torch.equal(a, b) for a, b in zip(tr.key_model["encoder"].parameters(), tr.encoder.parameters()))
    kq = [p.detach().clone() for p in tr.key_model["encoder"].parameters()]
    it = iter(tr.loader)
    batch = next(it)
    views, uids = list(batch[:-1]), batch[-1]
    out = tr.train_step(views[0], views[1], uids, True, extra_views=views[2:])
    assert out["queue_fallback"] == 1.0 and tr.neg_queue.filled == cfg["train"]["batch_size_images"]
    mm = cfg["pairing"]["momentum_m"]
    for k0, kn, q in zip(kq, tr.key_model["encoder"].parameters(), tr.encoder.parameters()):
        assert torch.allclose(kn, mm * k0 + (1 - mm) * q.detach(), atol=1e-6)
    out2 = tr.train_step(views[0], views[1], uids, True, extra_views=views[2:])
    assert out2["queue_fallback"] == 0.0


@pytest.mark.parametrize("method", ["vcs", "simclr"])
def test_stop_resume_bit_exact_queue_and_ema(tmp_path, method):
    data, m = synthetic()
    cfg = _cfg(tmp_path, method, f"s{method}.yaml")
    tr, st = run_trainer(cfg, data, m, run_id=f"s{method}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False, stop_after_steps=2)
    assert st == "STOPPED_BUDGET"
    ck = load_checkpoint(tr.run_dir / "checkpoints" / "last.pt")
    assert ck["momentum_key_encoder_state"] is not None and ck["neg_queue_state"]["count"] > 0
    tr2, st2 = run_trainer(cfg, data, m, run_id=f"s{method}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False, resume_from=tr.run_dir / "checkpoints" / "last.pt")
    assert st2 == "COMPLETED" and tr2.step == 4
    tr3, st3 = run_trainer(cfg, data, m, run_id=f"u{method}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False)
    a = load_checkpoint(tr2.run_dir / "checkpoints" / "last.pt"); b = load_checkpoint(tr3.run_dir / "checkpoints" / "last.pt")
    assert torch.equal(a["neg_queue_state"]["buffer"], b["neg_queue_state"]["buffer"]) and a["neg_queue_state"]["ptr"] == b["neg_queue_state"]["ptr"]
    for k in a["momentum_key_encoder_state"]:
        assert torch.equal(a["momentum_key_encoder_state"][k], b["momentum_key_encoder_state"][k]), k
    for k in a["encoder_state"]:
        assert torch.equal(a["encoder_state"][k], b["encoder_state"][k]), k
    rm = json.loads((tr3.run_dir / "run_manifest.json").read_text())
    assert rm["momentum_queue"]["m"] == cfg["pairing"]["momentum_m"]
