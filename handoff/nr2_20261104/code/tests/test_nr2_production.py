"""Production-interface tests against the package reference contract (synthetic data, CPU, float64)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
import pytest
from nr2.proposal_reference import proposal, log_reference_ratio, sampled_rows
from nr2.reference import Marginals, ReferenceSampler, log_w, step_loss, build_base_topk


def toy(n_img=12, n_txt=40, n_rec=60, seed=0):
    g = np.random.default_rng(seed)
    im = g.integers(0, n_img, n_rec); im[:n_img] = np.arange(n_img)          # full image support, nonuniform counts
    tx = g.integers(0, n_txt, n_rec); tx[:n_txt] = np.arange(n_txt)
    marg = Marginals(im, tx, n_img, n_txt)
    top_i2t = np.stack([g.choice(n_txt, 5, replace=False) for _ in range(n_img)]).astype(np.int32)
    top_t2i = np.stack([g.choice(n_img, 5, replace=False) for _ in range(n_txt)]).astype(np.int32)
    return marg, top_i2t, top_t2i


def test_log_w_equals_reference_ratio():
    marg, ti, tt = toy()
    for d, top, pc, nq in (('i2t', ti, marg.p_txt, 12), ('t2i', tt, marg.p_img, 40)):
        for x in range(nq):
            y = np.arange(len(pc))
            ref = log_reference_ratio(pc, top[x], y, .5)
            got = log_w(np.full(len(y), x), y, top, pc, .5)
            assert np.allclose(got, ref, atol=1e-13, rtol=0)
            assert np.isclose(proposal(pc, top[x], .5).sum(), 1, atol=1e-12)


def test_sampler_frequency_matches_exact_mixture():
    marg, ti, tt = toy()
    s = ReferenceSampler(marg, ti, tt, .5, 7, n_query=1, n_ref=63)
    x = np.array([3]); counts = np.zeros(len(marg.p_txt))
    for _ in range(4000):
        y = s._draw_ref('i2t', x); np.add.at(counts, y.reshape(-1), 1)
    n = counts.sum(); q = proposal(marg.p_txt, ti[3], .5)
    z = (counts - n*q) / np.sqrt(n*q*(1-q))
    assert np.abs(z).max() < 5


def test_positive_draws_identical_across_modes_and_reference_coupled():
    marg, ti, tt = toy()
    a = ReferenceSampler(marg, ti, tt, 1.0, 11).draw(); b = ReferenceSampler(marg, ti, tt, .5, 11).draw()
    for d in a:
        assert np.array_equal(a[d]['records'], b[d]['records']) and np.array_equal(a[d]['cand'][:, 0], b[d]['cand'][:, 0])
        assert np.all(a[d]['log_w'] == 0)
        # coupled: wherever b drew the marginal component, the candidate equals a's marginal draw
        same = (a[d]['cand'][:, 1:] == b[d]['cand'][:, 1:]).mean()
        assert same > .4


def test_observed_mask_and_p_slot():
    marg, ti, tt = toy()
    b = ReferenceSampler(marg, ti, tt, .5, 3).draw()
    rel = set(zip(marg.im.tolist(), marg.tx.tolist()))
    for d in b:
        q, c, o = b[d]['query'], b[d]['cand'], b[d]['observed']
        for i in range(len(q)):
            for k in range(c.shape[1]):
                pair = (q[i], c[i, k]) if d == 'i2t' else (c[i, k], q[i])
                assert o[i, k] == (pair in rel)
        assert o[:, 0].all()


def test_step_loss_matches_reference_rows():
    marg, ti, tt = toy()
    b = ReferenceSampler(marg, ti, tt, .5, 5).draw()
    g = torch.Generator().manual_seed(0)
    sc = {d: torch.randn(b[d]['cand'].shape, dtype=torch.float64, generator=g) for d in b}
    for obj in ('J', 'BCE', 'NCE'):
        for mode in ('U', 'HIW', 'HREF'):
            if mode == 'HIW' and obj == 'NCE': continue
            got = float(step_loss(sc, b, obj, mode))
            exp = 0.
            for d in b:
                lw = torch.as_tensor(b[d]['log_w'], dtype=torch.float64)
                m = torch.as_tensor(b[d]['observed']) if obj == 'NCE' else None
                if mode == 'HREF':
                    exp += .5*float(sampled_rows(sc[d]-lw, torch.zeros_like(lw), obj, 'U', m).mean())
                elif mode == 'HIW':
                    t = torch.tanh(sc[d]/2); w = torch.exp(-lw[:, 1:])
                    if obj == 'J':
                        v = -(t[:, 0]-t[:, 0]**2/2) + (w*(t[:, 1:]+t[:, 1:]**2/2)).mean(1)
                    else:
                        v = torch.nn.functional.softplus(-sc[d][:, 0]) + (w*torch.nn.functional.softplus(sc[d][:, 1:])).mean(1)
                    exp += .5*float(v.mean())
                else:
                    exp += .5*float(sampled_rows(sc[d], torch.zeros_like(lw), obj, 'U', m).mean())
            assert got == pytest.approx(exp, abs=1e-12)


def test_alpha_one_all_modes_equal():
    marg, ti, tt = toy()
    b = ReferenceSampler(marg, ti, tt, 1.0, 5).draw()
    sc = {d: torch.randn(b[d]['cand'].shape, dtype=torch.float64) for d in b}
    for obj in ('J', 'BCE'):
        v = [float(step_loss(sc, b, obj, m)) for m in ('U', 'HIW', 'HREF')]
        assert v[0] == v[1] == v[2]


def test_topk_chunk_reorder_rename_invariance():
    g = np.random.default_rng(1)
    q, gal = g.normal(size=(37, 8)), g.normal(size=(300, 8)); ids = [f'g{i}' for i in range(300)]
    a, _ = build_base_topk(q, gal, ids, k=10, chunk=5, device='cpu')
    b, _ = build_base_topk(q, gal, ids, k=10, chunk=1000, device='cpu')
    assert np.array_equal(a, b)
    brute = np.argsort(-(q/np.linalg.norm(q, axis=1, keepdims=True)) @ (gal/np.linalg.norm(gal, axis=1, keepdims=True)).T, 1)[:, :10]
    assert np.array_equal(np.sort(a, 1), np.sort(brute, 1))
    perm = g.permutation(300)
    c, _ = build_base_topk(q, gal[perm], [f'renamed{i}' for i in perm], k=10, chunk=7, device='cpu')
    assert np.array_equal(perm[c], a)
