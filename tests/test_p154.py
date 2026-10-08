"""P154 ImageNet-100 trainer: split, schedule, weight-decay groups, loss glue against independent references, view bookkeeping, resume
equivalence and the evaluation path (synthetic data, CPU)."""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
import p154_in100 as P  # noqa: E402


def test_split_counts_disjoint_deterministic():
    wnids = P.CLASS_FILE.read_text().split()
    rows = [(f"train/{w}/{w}_{i}.JPEG", w) for w in wnids for i in range(60)] + [(f"train/n0/x_{i}.JPEG", "n00000000") for i in range(30)]
    tr, va, cls = P.in100_split(rows)
    assert len(tr) == 100 * 10 and len(va) == 100 * 50 and not set(tr) & set(va)
    assert sorted(cls.values()) == list(range(100)) and list(cls) == sorted(wnids)
    assert all(sum(1 for r in va if r[1] == w) == 50 for w in wnids)
    tr2, va2, _ = P.in100_split(list(reversed(rows)))          # input order does not matter
    assert tr2 == tr and va2 == va and P.split_digest(tr, va) == P.split_digest(tr2, va2)


def test_lr_schedule():
    spe, ep = 10, 20; base, w = P.RECIPE["lr"], P.RECIPE["warmup_epochs"] * 10
    assert P.lr_at(0, spe, ep) == pytest.approx(base / w) and P.lr_at(w - 1, spe, ep) == pytest.approx(base)
    assert P.lr_at(ep * spe - 1, spe, ep) == pytest.approx(base * P.RECIPE["min_lr_ratio"])
    post = [P.lr_at(s, spe, ep) for s in range(w, ep * spe)]
    assert all(a >= b - 1e-15 for a, b in zip(post, post[1:]))


def test_param_groups():
    enc, proj = P.build(0)
    g = P.param_groups(enc, proj, wd=1e-4)
    assert all(p.ndim >= 2 for p in g[0]["params"]) and all(p.ndim < 2 for p in g[1]["params"])
    assert g[0]["weight_decay"] == 1e-4 and g[1]["weight_decay"] == 0.0
    assert sum(p.numel() for gr in g for p in gr["params"]) == sum(p.numel() for m in (enc, proj) for p in m.parameters())


def _ref_tokens(zs):
    V, B = len(zs), zs[0].shape[0]
    t = torch.stack([F.normalize(z.double(), dim=-1) for z in zs])          # [V, B, d]
    f = 2.0 * torch.einsum("vid,wjd->viwj", t, t) - 1.0                    # A-P3 scorer on cosine
    same = torch.eye(B, dtype=torch.bool)[None, :, None, :].expand(V, B, V, B)
    diag = (torch.eye(V, dtype=torch.bool)[:, None, :, None] & torch.eye(B, dtype=torch.bool)[None, :, None, :])
    return f[same & ~diag], f[~same]


def test_vcs_and_js_loss_match_reference():
    g = torch.Generator().manual_seed(0); zs = [torch.randn(8, 16, generator=g) for _ in range(4)]
    fp, fq = _ref_tokens(zs); tp, tq = torch.tanh(fp), torch.tanh(fq)
    J = (tp - 0.5 * tp ** 2).mean() + (-tq - 0.5 * tq ** 2).mean()
    out = P.ssl_loss("vcs", zs)
    assert float(out["loss"]) == pytest.approx(-float(J), abs=1e-5) and float(out["J_global"]) == pytest.approx(float(J), abs=1e-5)
    js = F.softplus(-2 * fp).mean() + F.softplus(2 * fq).mean()
    assert float(P.ssl_loss("js", zs)["loss"]) == pytest.approx(float(js), abs=1e-5)


def test_simclr_loss_matches_reference():
    g = torch.Generator().manual_seed(1); zs = [torch.randn(8, 16, generator=g) for _ in range(3)]
    tot, pairs = 0.0, 0
    for a in range(3):
        for c in range(a + 1, 3):
            z = F.normalize(torch.cat([zs[a], zs[c]]).double(), dim=-1); lg = z @ z.T / 0.2
            lg.fill_diagonal_(-math.inf); tgt = torch.cat([torch.arange(8, 16), torch.arange(8)])
            tot += float(F.cross_entropy(lg, tgt)); pairs += 1
    assert float(P.ssl_loss("simclr", zs)["loss"]) == pytest.approx(tot / pairs, abs=1e-5)


def test_forward_views_bookkeeping():
    x = torch.randint(0, 256, (5, 4, 3, 8, 8), dtype=torch.uint8)
    enc = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Identity()); proj = torch.nn.Identity()
    zs = P.forward_views(enc, proj, x, lambda t: t.float(), torch.device("cpu"), amp=False)
    assert len(zs) == 4 and all(z.shape == (5, 3 * 64) for z in zs)
    for v in range(4):
        for i in range(5):
            assert torch.equal(zs[v][i], x[i, v].float().flatten())


def _args(tmp, **kw):
    base = dict(run_id="t", workers=0, fake=True, fake_n=64, cpu=True, method="vcs", seed=0, epochs=2, max_steps_per_epoch=2, fake_batch=16,
                stop_after_epochs=None, checkpoint="epoch_2.pt")
    base.update(kw); return SimpleNamespace(**base)


def test_resume_equals_uninterrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "OUT_ROOT", tmp_path)
    P.train(_args(tmp_path, run_id="full"))
    P.train(_args(tmp_path, run_id="split", stop_after_epochs=1)); P.train(_args(tmp_path, run_id="split"))
    a = torch.load(tmp_path / "full" / "checkpoints" / "last.pt", weights_only=False); b = torch.load(tmp_path / "split" / "checkpoints" / "last.pt", weights_only=False)
    assert a["completed_epoch"] == b["completed_epoch"] == 2 and a["step"] == b["step"] == 4
    for k in a["encoder"]:
        assert torch.equal(a["encoder"][k], b["encoder"][k]), k
    log = [json.loads(l) for l in open(tmp_path / "split" / "logs" / "epochs.jsonl")]
    assert [r["epoch"] for r in log] == [1, 2] and all(np.isfinite(r["loss"]) for r in log)


def test_methods_train_and_eval(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "OUT_ROOT", tmp_path)
    for m in ("js", "simclr"):
        P.train(_args(tmp_path, run_id=m, method=m, epochs=1, max_steps_per_epoch=1))
    P.train(_args(tmp_path, run_id="ev")); P.evaluate(_args(tmp_path, run_id="ev"))
    r = json.load(open(tmp_path / "ev" / "evaluations" / "evaluation_epoch_2.json"))
    assert 0 <= r["linear_val_top1_pct"] <= 100 and 0 <= r["knn_val_top1_pct"] <= 100 and r["official_val_used"] is False
