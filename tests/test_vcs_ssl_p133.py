"""P133 (owner 2026-10-04): MoCo-consistent momentum-key queue — every pair is (online query, momentum key); P = different views of one
image, Q = keys of other images (current batch + uid-masked queue).  Synthetic data only."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO, REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vcs_ssl.checkpoint import load_checkpoint  # noqa: E402
from vcs_ssl.config import ConfigError, load_config  # noqa: E402
from vcs_ssl.models.critic import FixedCosineCritic  # noqa: E402
from vcs_ssl.objectives import (KeyUidQueue, compute_objective_moco_consistent, moco_consistent_infonce, moco_consistent_scores,  # noqa: E402
                                moco_consistent_vcs)
from test_vcs_ssl_v2 import _base, _load, _small, run_trainer, synthetic  # noqa: E402

D = 16


def p133(d: dict, method: str = "vcs") -> dict:
    if method == "vcs":
        c = d["model"]["critic"]
        c.update({"input": "cosine", "affine_mode": "fixed", "cosine_scale_init": 2.0, "cosine_bias_init": -1.0})
        d["pairing"]["negative_detach"] = False
    else:
        d["run"]["control_tuning"] = True
    d["views"]["count"] = 4
    d["pairing"].update({"negative_source": "queue", "queue": True, "queue_size": 64, "momentum_encoder": True, "momentum_m": 0.9, "moco_consistent": True})
    return d


def _cfg(tmp_path, method="vcs", name="c.yaml", epochs=2):
    return _small(_load(tmp_path, p133(_base("vcs" if method == "vcs" else "simclr"), method), name), epochs=epochs)


def _views(V=4, B=6, seed=0, grad=False):
    g = torch.Generator().manual_seed(seed)
    q = [F.normalize(torch.randn(B, D, generator=g), dim=1) for _ in range(V)]
    k = [F.normalize(torch.randn(B, D, generator=g), dim=1) for _ in range(V)]
    if grad:
        q = [x.requires_grad_(True) for x in q]
    return q, k


# ------------------------------------------------------------------------------------------------------------ units
def test_key_uid_queue_fifo_alignment_and_state():
    q = KeyUidQueue(5, 2)
    for i in range(4):
        q.enqueue(torch.full((2, 2), float(i)), torch.tensor([10 * i, 10 * i + 1]))
    assert q.filled == 5
    f, u = q.features(), q.uids()
    assert f[:, 0].tolist() == [1.0, 2.0, 2.0, 3.0, 3.0] and u.tolist() == [11, 20, 21, 30, 31]  # oldest first, keys and uids aligned
    st = q.state_dict()
    q2 = KeyUidQueue(5, 2); q2.load_state_dict(st)
    assert torch.equal(q2.features(), f) and torch.equal(q2.uids(), u)
    q.enqueue(torch.full((7, 2), 9.0), torch.arange(100, 107))  # longer than the ring: newest 5 survive
    assert q.uids().tolist() == [102, 103, 104, 105, 106] and float(q.features()[0, 0]) == 9.0


def test_index_sets_differ_only_in_image_identity():
    V, B = 4, 6
    qv, kv = _views(V, B)
    buids = torch.tensor([10, 11, 12, 13, 14, 15])
    queue = KeyUidQueue(8, D)
    queue.enqueue(F.normalize(torch.randn(8, D), dim=1), torch.tensor([3, 12, 4, 5, 15, 6, 7, 8]))  # uid 12 and 15 are in this batch
    s_pos, C_in, mask_in, C_q, mask_q = moco_consistent_scores(qv, kv, buids, queue)
    row_uid = buids.repeat(V)                       # view-major query tokens
    col_uid_in = buids.repeat(V)                    # view-major batch keys
    col_uid_q = queue.uids()
    # P: every positive pairs a query with a key of the SAME image and a DIFFERENT view; values are exact dot products
    for i in range(V):
        js = [j for j in range(V) if j != i]
        for r, j in enumerate(js):
            assert torch.allclose(s_pos[i, r], (qv[i] * kv[j]).sum(-1))
    # Q: every valid negative has a different image uid than its query; every same-uid entry is excluded
    assert torch.equal(mask_in, row_uid[:, None] != col_uid_in[None, :])
    assert torch.equal(mask_q, row_uid[:, None] != col_uid_q[None, :])
    assert int(mask_in.sum()) == V * B * V * (B - 1)
    assert int((~mask_q).sum()) == 2 * V  # tokens of images 12 and 15 (V each) lose their own queued key
    # both P and Q keys come from the same key tensors (the momentum keys): C_in columns are exactly the batch keys
    assert torch.allclose(C_in, torch.cat(qv) @ torch.cat(kv).T)


def test_vcs_and_infonce_match_bruteforce():
    V, B = 4, 5
    qv, kv = _views(V, B, seed=1)
    buids = torch.arange(B)
    queue = KeyUidQueue(6, D); queue.enqueue(F.normalize(torch.randn(6, D), dim=1), torch.tensor([7, 8, 2, 9, 10, 11]))
    crit = FixedCosineCritic(D, 2.0, -1.0)
    qk, qu = queue.features(), queue.uids()
    tp, tq, ce = [], [], []
    for i in range(V):
        for b in range(B):
            negs = [float((qv[i][b] * kv[v][bb]).sum()) for v in range(V) for bb in range(B) if bb != b]
            negs += [float((qv[i][b] * qk[n]).sum()) for n in range(len(qk)) if int(qu[n]) != b]
            tq += [torch.tanh(torch.tensor(2 * s - 1.0)) for s in negs]
            for j in range(V):
                if j == i:
                    continue
                sp = float((qv[i][b] * kv[j][b]).sum())
                tp.append(torch.tanh(torch.tensor(2 * sp - 1.0)))
                logits = torch.tensor([sp] + negs) / 0.2
                ce.append(-torch.log_softmax(logits, 0)[0])
    tp, tq = torch.stack(tp), torch.stack(tq)
    j_ref = tp.mean() - tq.mean() - 0.5 * tp.square().mean() - 0.5 * tq.square().mean()
    out = moco_consistent_vcs(qv, kv, buids, queue, crit)
    assert torch.allclose(out["J_raw"], j_ref, atol=1e-5) and out["n_pos"] == V * (V - 1) * B and out["n_neg"] == len(tq)
    nce = moco_consistent_infonce(qv, kv, buids, queue, 0.2)
    assert torch.allclose(nce["loss"], torch.stack(ce).mean(), atol=1e-5)


def test_keys_get_no_gradient_queries_do():
    qv, kv = _views(grad=True)
    kv = [k.clone().requires_grad_(True) for k in kv]
    queue = KeyUidQueue(8, D); queue.enqueue(F.normalize(torch.randn(8, D), dim=1), torch.arange(100, 108))
    out = moco_consistent_vcs(qv, kv, torch.arange(6), queue, FixedCosineCritic(D, 2.0, -1.0))
    out["loss"].backward()
    assert all(k.grad is None for k in kv) and all(q.grad is not None and q.grad.abs().sum() > 0 for q in qv)


def test_constant_online_map_cannot_reach_positive_J_and_p100_construction_can():
    """Gate core (synthetic): with a constant online map q = c, P and Q scores are c·k for keys of the same marginal law, so J = -E[T^2] <= 0
    up to sampling error, for every c (random and adversarial).  Positive control: the P100 construction (positives online-online,
    negatives online vs keys of another encoder) lets the same constant map reach J > 0."""
    torch.manual_seed(0)
    V, B, Q = 4, 256, 4096
    mu = F.normalize(torch.randn(D), dim=0)
    def keys(n):  # a concentrated key law (the hardest case for a constant map)
        return F.normalize(mu + 0.3 * torch.randn(n, D), dim=1)
    kv = [keys(B) for _ in range(V)]
    queue = KeyUidQueue(Q, D); queue.enqueue(keys(Q), torch.arange(10_000, 10_000 + Q))
    crit = FixedCosineCritic(D, 2.0, -1.0)
    pos_all = torch.cat(kv); qk = queue.features()
    cands = [F.normalize(torch.randn(D), dim=0) for _ in range(20)] + [mu, F.normalize(pos_all.mean(0) - qk.mean(0), dim=0)]
    worst = max(float(moco_consistent_vcs([c.expand(B, D).clone() for _ in range(V)], kv, torch.arange(B), queue, crit)["J_raw"]) for c in cands)
    assert worst <= 0.0
    # P100 construction with the same constant map: positives online-online (s = 1), negatives vs keys of a different (random) encoder
    c = mu
    other = F.normalize(torch.randn(Q, D), dim=1)
    tp = torch.tanh(2.0 * torch.ones(B) - 1.0); tq = torch.tanh(2.0 * (other @ c) - 1.0)
    j_p100 = tp.mean() - tq.mean() - 0.5 * tp.square().mean() - 0.5 * tq.square().mean()
    assert float(j_p100) > 0.5


def test_config_policy(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg["pairing"]["moco_consistent"] is True
    for mutate in (lambda d: d["pairing"].update({"momentum_encoder": False, "momentum_m": None}) or d["pairing"].pop("momentum_m"),
                   lambda d: d["views"].update({"count": 2}),
                   lambda d: d["pairing"].update({"pair_scope": "all_view_tokens"}),
                   lambda d: d["model"]["critic"].update({"affine_mode": "learned"}),
                   lambda d: d["pairing"].update({"moco_consistent": False})):
        d = p133(_base("vcs")); mutate(d)
        with pytest.raises(ConfigError):
            _load(tmp_path, d, "bad.yaml")
    assert _cfg(tmp_path, "simclr", "s.yaml")["pairing"]["moco_consistent"] is True


REAL_RUNS = {"cifar10_hpY_AP3_views4_800ep_seed0.yaml": "P107_AP3_views4_800ep_seed0",
             "cifar10_hpQ_vcs_mq_views4_800ep_seed0.yaml": "P100_vcs_mq_views4_800ep_seed0",
             "cifar10_hpAF_simclr_croponly_views4_800ep_seed0.yaml": "P115_simclr_croponly_views4_800ep_seed0"}
# (P41's stored hash predates the kernel_cs / rff defaults added on 2026-09-28 — pre-existing schema drift, unrelated to P133 — so a recent
#  SimCLR run is used as the SimCLR reference.)


@pytest.mark.skipif(not os.environ.get("OUTPUT_ROOT"), reason="needs the cluster environment (slurm/common.sh)")
def test_old_config_hashes_unchanged():
    for cfgname, run in REAL_RUNS.items():
        rc = load_config(REPO / "configs" / cfgname)
        man = Path(os.environ["OUTPUT_ROOT"]) / run / "run_manifest.json"
        if not man.is_file():
            pytest.skip(f"{man} missing")
        assert "moco_consistent" not in rc["pairing"]
        assert rc["_meta"]["config_hash"] == json.loads(man.read_text())["config_hash"], cfgname


# ------------------------------------------------------------------------------------------------------------ trainer
@pytest.mark.parametrize("method", ["vcs", "simclr"])
def test_trainer_eval_keys_ema_params_and_buffers_queue_uids(tmp_path, method):
    data, m = synthetic()
    cfg = _cfg(tmp_path, method, f"t{method}.yaml")
    run_dir = Path(cfg["run"]["output_root"]) / f"e{method}"; run_dir.mkdir(parents=True, exist_ok=True)
    from vcs_ssl.train import Trainer
    tr = Trainer(cfg, run_dir=run_dir, data=data, manifest=m, device=torch.device("cpu"), stage="TEST", smoke_steps=3, smoke_epoch_steps=3, epoch_eval=False)
    tr.setup()
    assert tr.moco and tr.neg_queue is None and tr.p133_queue is not None
    kp0 = [p.detach().clone() for p in tr.key_model["encoder"].parameters()]
    kb0 = {n: b.detach().clone() for n, b in tr.key_model["encoder"].named_buffers()}
    batch = next(iter(tr.loader))
    views, uids = list(batch[:-1]), batch[-1]
    keys = tr.p133_momentum_keys(views)  # eval mode: computing keys must not touch the key encoder's buffers
    assert all(torch.equal(b, kb0[n]) for n, b in tr.key_model["encoder"].named_buffers())
    assert len(keys) == 4 and torch.allclose(keys[0].norm(dim=1), torch.ones(len(uids)), atol=1e-5)
    out = tr.train_step(views[0], views[1], uids, True, extra_views=views[2:])  # first step: empty queue -> batch negatives only
    assert out["p133_n_neg_queue"] == 0.0 and torch.isfinite(torch.tensor(out["loss"]))
    assert tr.p133_queue.filled == len(uids) and torch.equal(tr.p133_queue.uids(), uids.long())
    mm = cfg["pairing"]["momentum_m"]
    for k0, kn, q in zip(kp0, tr.key_model["encoder"].parameters(), tr.encoder.parameters()):
        assert torch.allclose(kn, mm * k0 + (1 - mm) * q.detach(), atol=1e-6)
    for (n, kb), (_, qb) in zip(tr.key_model["encoder"].named_buffers(), tr.encoder.named_buffers()):
        if kb.dtype.is_floating_point:
            assert torch.allclose(kb, mm * kb0[n] + (1 - mm) * qb, atol=1e-6), n
        else:
            assert torch.equal(kb, qb), n
    out2 = tr.train_step(views[0], views[1], uids, True, extra_views=views[2:])  # same images again: their queued keys are uid-masked
    n = len(uids)  # each of the 4n query tokens masks exactly its own image's queued key; the other n - 1 queued keys are valid negatives
    assert out2["p133_n_queue_uid_masked"] == 4 * n and out2["p133_n_neg_queue"] == 4 * n * (n - 1)


@pytest.mark.parametrize("method", ["vcs", "simclr"])
def test_stop_resume_bit_exact(tmp_path, method):
    data, m = synthetic()
    cfg = _cfg(tmp_path, method, f"s{method}.yaml")
    tr, st = run_trainer(cfg, data, m, run_id=f"s{method}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False, stop_after_steps=2)
    assert st == "STOPPED_BUDGET"
    ck = load_checkpoint(tr.run_dir / "checkpoints" / "last.pt")
    assert ck["p133_queue_state"]["keys"]["count"] > 0 and ck["neg_queue_state"] is None
    tr2, st2 = run_trainer(cfg, data, m, run_id=f"s{method}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False, resume_from=tr.run_dir / "checkpoints" / "last.pt")
    assert st2 == "COMPLETED" and tr2.step == 4
    tr3, st3 = run_trainer(cfg, data, m, run_id=f"u{method}", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False)
    a = load_checkpoint(tr2.run_dir / "checkpoints" / "last.pt"); b = load_checkpoint(tr3.run_dir / "checkpoints" / "last.pt")
    assert torch.equal(a["p133_queue_state"]["keys"]["buffer"], b["p133_queue_state"]["keys"]["buffer"])
    assert torch.equal(a["p133_queue_state"]["uids"], b["p133_queue_state"]["uids"])
    for k in a["momentum_key_encoder_state"]:
        assert torch.equal(a["momentum_key_encoder_state"][k], b["momentum_key_encoder_state"][k]), k
    for k in a["encoder_state"]:
        assert torch.equal(a["encoder_state"][k], b["encoder_state"][k]), k
    rm = json.loads((tr3.run_dir / "run_manifest.json").read_text())
    assert rm["p133_moco_consistent"]["m"] == cfg["pairing"]["momentum_m"] and "momentum_queue" not in rm


# ------------------------------------------------------------------------------------------------------------ variant (a): no queue
def test_noqueue_variant_policy(tmp_path):
    d = p133(_base("vcs")); d["pairing"]["moco_use_queue"] = False
    assert _load(tmp_path, d, "nq.yaml")["pairing"]["moco_use_queue"] is False
    d = _base("vcs"); d["pairing"]["moco_use_queue"] = False  # orphan field without the P133 marker
    with pytest.raises(ConfigError):
        _load(tmp_path, d, "nq_bad.yaml")


def test_noqueue_variant_trainer_and_resume(tmp_path):
    data, m = synthetic()
    d = p133(_base("vcs")); d["pairing"]["moco_use_queue"] = False
    cfg = _small(_load(tmp_path, d, "nqt.yaml"), epochs=2)
    tr, st = run_trainer(cfg, data, m, run_id="nq", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False, stop_after_steps=2)
    assert st == "STOPPED_BUDGET" and tr.p133_queue is None and tr.neg_queue is None
    ck = load_checkpoint(tr.run_dir / "checkpoints" / "last.pt")
    assert "p133_queue_state" not in ck and ck["momentum_key_encoder_state"] is not None
    tr2, st2 = run_trainer(cfg, data, m, run_id="nq", smoke_steps=4, smoke_epoch_steps=2, epoch_eval=False, resume_from=tr.run_dir / "checkpoints" / "last.pt")
    assert st2 == "COMPLETED" and tr2.step == 4
    rm = json.loads((tr2.run_dir / "run_manifest.json").read_text())
    assert rm["p133_moco_consistent"]["use_queue"] is False and rm["p133_moco_consistent"]["queue_size"] is None
    rows = [json.loads(l) for l in (tr2.run_dir / "logs" / "steps.jsonl").read_text().splitlines() if l.strip()]
    assert all(r.get("p133_n_neg_queue", 0.0) == 0.0 for r in rows)
