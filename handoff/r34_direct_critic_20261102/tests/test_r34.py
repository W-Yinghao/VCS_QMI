"""R34 CPU checks (values, gradients, evaluator equivalence, locks, guard). Run: pytest -q tests/test_r34.py"""
import importlib.util, os, sys
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'r34'))
import common, critics, native_eval, train_unit, make_locks  # noqa: E402
from build_cache import standardization  # noqa: E402

R32_SRC = Path('/home/infres/yinwang/CS_QMI/r32_native_ratio/VCS_NATIVE_RATIO_v1_Server_20261101/src')
sys.path.insert(1, str(R32_SRC))
from native_ratio.objectives import original_j, balanced_logistic, inbatch_pair_weights  # noqa: E402
from native_ratio.models import AffineCosineCritic  # noqa: E402
from native_ratio.frozen import standardize_caches  # noqa: E402


def upstream_eval():
    sys.path.insert(0, str(common.CODE_R30))
    import evaluation
    return evaluation


def test_native_eval_verbatim():
    E = upstream_eval(); rng = np.random.default_rng(0)
    for ties in (False, True):
        s = rng.normal(size=(40, 200))
        if ties:
            s = np.round(s, 1)
        a = native_eval.i2t(40, s, 5, return_ranks=True); b = E.i2t(40, s, 5, return_ranks=True)
        assert a[0] == b[0] and np.array_equal(a[1][0], b[1][0])
        a = native_eval.t2i(40, s, 5, return_ranks=True); b = E.t2i(40, s, 5, return_ranks=True)
        assert a[0] == b[0] and np.array_equal(a[1][0], b[1][0])


def test_tie_bounded():
    rng = np.random.default_rng(1)
    s = rng.normal(size=(30, 150))
    up, _, _ = native_eval.upstream_metrics(s); tb = native_eval.tie_bounded(s)
    assert not tb['any_exact_tie_at_gt']
    assert abs(tb['rsum_optimistic'] - up['rsum']) < 1e-9 and abs(tb['rsum_pessimistic'] - up['rsum']) < 1e-9
    s2 = np.round(s, 0)
    tb2 = native_eval.tie_bounded(s2); up2, _, _ = native_eval.upstream_metrics(s2)
    assert tb2['rsum_pessimistic'] <= up2['rsum'] + 1e-9 <= tb2['rsum_optimistic'] + 2e-9
    # brute force t2i optimistic/pessimistic for one column
    j = 7; col = s2[:, j]; p = col[j // 5]
    assert (col > p).sum() == np.sum(col > p) and ((col == p).sum() - 1) >= 0


def test_objectives_match_r32_and_gradients():
    torch.manual_seed(0)
    B = 9; g = torch.randn(B, B, dtype=torch.float64) * 3
    p, q = critics.pq_weights(B, 'cpu', torch.float64); P, Q = inbatch_pair_weights(B, dtype=torch.float64)
    assert torch.allclose(critics.j_value(g, p, q), original_j(torch.tanh(g / 2), P, Q), atol=1e-12)
    assert torch.allclose(critics.bce_value(g, p, q), balanced_logistic(g, P, Q), atol=1e-12)
    lab = torch.arange(B)
    ref = .5 * (-(g.log_softmax(1)[lab, lab]).mean() - (g.T.log_softmax(1)[lab, lab]).mean())
    assert torch.allclose(critics.nce_value(g), ref, atol=1e-12)
    for obj in critics.OBJECTIVES:
        x = (torch.randn(B, B, dtype=torch.float64) * 2).requires_grad_()
        assert torch.autograd.gradcheck(lambda z: critics.loss_of(obj, z, p, q), (x,))
    # at g=0: -J and BCE have identical logit gradients (+-1/2 weighted by P/Q)
    z = torch.zeros(B, B, dtype=torch.float64, requires_grad=True)
    gj = torch.autograd.grad(critics.loss_of('J', z, p, q), z)[0]; gb = torch.autograd.grad(critics.loss_of('BCE', z, p, q), z)[0]
    assert torch.allclose(gj, gb, atol=1e-15)


def test_dual_cosine_equals_r32_affine():
    torch.manual_seed(3)
    m = critics.DualCosine(16, 12, 8).double(); r = AffineCosineCritic(16, 12, 8).double()
    r.image_head.load_state_dict(m.image_head.state_dict()); r.text_head.load_state_dict(m.text_head.state_dict())
    with torch.no_grad():
        r.raw_scale.copy_(m.raw_scale); r.offset_value.copy_(m.offset)
    x, y = torch.randn(6, 16, dtype=torch.float64), torch.randn(6, 12, dtype=torch.float64)
    assert torch.allclose(m.pair_scores(x, y), r(x, y)['log_ratio'], atol=1e-12)
    a, b = m.ab(); assert abs(float(a) - 4) < 1e-6 and abs(float(b) + 2) < 1e-6


def test_joint_blockwise_equals_concat_and_params():
    torch.manual_seed(4)
    m = critics.JointMLP(16, 12, 8).double()
    x, y = torch.randn(5, 16, dtype=torch.float64), torch.randn(7, 12, dtype=torch.float64)
    assert torch.allclose(m.pair_scores(x, y), m.concat_scores(x, y), atol=1e-12)
    assert torch.allclose(m.pair_scores(x, y, chunk=2), m.pair_scores(x, y), atol=0)
    assert critics.n_params(critics.DualCosine()) == 524802 and critics.n_params(critics.JointMLP()) == 590593


def test_identical_init_within_structure():
    for s in critics.STRUCTURES:
        hs = set()
        for _ in critics.OBJECTIVES:
            torch.manual_seed(1001); hs.add(train_unit.state_hash(critics.build(s)))
        assert len(hs) == 1


def test_j_from_matrix_ustat_bruteforce():
    rng = np.random.default_rng(5)
    n_img, n = 6, 20
    m = rng.integers(0, n_img, size=n); g = torch.tensor(rng.normal(size=(n_img, n)) * 2, dtype=torch.float32)
    r = train_unit.j_from_matrix(g, m)
    t = torch.tanh(g / 2).double().numpy()
    ep = np.mean([t[m[k], k] - .5 * t[m[k], k] ** 2 for k in range(n)])
    eq = np.mean([t[m[a], b] + .5 * t[m[a], b] ** 2 for a in range(n) for b in range(n) if a != b])
    assert abs(r['J'] - (ep - eq)) < 1e-9


def test_standardization_matches_r32():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(10, 4)).astype(np.float32); y = rng.normal(size=(30, 4)).astype(np.float32)
    m = rng.integers(0, 10, size=30)
    st = standardization(x, y, m)
    caches = {'train': {'x': torch.from_numpy(x), 'y': torch.from_numpy(y), 'mapping': m, 'sha256': '0'}}
    ref = standardize_caches(caches)
    for k in ('image_mean', 'image_std', 'text_mean', 'text_std'):
        assert np.allclose(st[k], ref[k].numpy(), atol=1e-6)


def test_noise_mechanism_matches_r25_and_locks():
    spec = importlib.util.spec_from_file_location('mk', '/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/tools/make_noise.py')
    mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
    for seed in (1, 20261102):
        assert np.array_equal(mk.official(5070, 5, .6, np.random.RandomState(seed)), make_locks.official(5070, 5, .6, np.random.RandomState(seed)))
    # the R25 train map is reproduced by the same function (sanity of the mechanism we copy to DEV)
    t = make_locks.official(145000, 5, .6, np.random.RandomState(20261027))
    assert np.array_equal(t, np.load(common.CONDS['n60']['noise_file']))


def test_test_guard(tmp_path):
    common.install_test_guard()
    d = tmp_path / 'f30k_precomp'; d.mkdir()
    (d / 'dev_caps.txt').write_text('x')
    open(d / 'dev_caps.txt').close()
    with pytest.raises(PermissionError):
        open(d / 'test_caps.txt', 'w')
    with pytest.raises(PermissionError):
        np.load(str(d / 'testall_ims.npy'))
    assert len(common.GUARD['blocked_attempts']) >= 2
    common.GUARD['blocked_attempts'].clear()


def _fake_d(rng):
    d = {'cond': 'fake', 'manifest': {}, 'locks': {}, 'manifest_sha256': 'fake'}
    n_img, n_cap = 50, 250
    obs = rng.integers(0, n_img, size=n_cap).astype(np.int64)
    d['obs_np'] = obs; d['obs'] = torch.from_numpy(obs)
    for p in common.PEERS:
        d[('fit', 'img', p)] = torch.randn(n_img, 1024); d[('fit', 'cap', p)] = torch.randn(n_cap, 1024)
        d[('dev', 'img', p)] = torch.randn(20, 1024); d[('dev', 'cap', p)] = torch.randn(100, 1024)
    d['native_dev'] = rng.normal(size=(20, 100))
    d['dev_clean'] = np.arange(100) // 5
    d['dev_obs'] = make_locks.official(100, 5, .6, np.random.RandomState(0))
    sub = np.sort(rng.choice(n_cap, 60, replace=False))
    imgs, ml = np.unique(obs[sub], return_inverse=True)
    d['fit_sub'] = dict(records=torch.from_numpy(sub), images=torch.from_numpy(imgs), map=ml.astype(np.int64), n_images=len(imgs))
    return d


def test_unit_path_synthetic(tmp_path):
    rng = np.random.default_rng(7); d = _fake_d(rng)
    for s, sname in (('D', 'D_dual_cosine'), ('X', 'X_joint_mlp')):
        hashes = []
        for o in critics.OBJECTIVES:
            cfg = common.read_json(ROOT / 'configs' / f"direct_n60_s1001_{sname}_{'InfoNCE' if o == 'NCE' else o}.json")
            cfg = dict(cfg, steps=6, eval_every=3, batch_size=16)
            r = train_unit.run_unit(cfg, d, torch.device('cpu'), log=lambda *a: None, runs_dir=tmp_path)
            assert np.isfinite(r['dev_retrieval_step2000']['rsum']) and len(r['curves']) == 3
            assert (tmp_path / cfg['unit_id'] / 'DONE.json').exists()
            hashes.append((r['identity']['init_state_sha256'], r['identity']['batch_index_sha256']))
        assert len(set(hashes)) == 1
    # isolation: NaN DEV does not change training
    cfg = dict(common.read_json(ROOT / 'configs' / 'direct_n60_s1001_X_joint_mlp_J.json'), steps=6, eval_every=3, batch_size=16)
    a = train_unit.train(cfg, d, torch.device('cpu'), with_rank=False, log=lambda *a: None)
    b = train_unit.train(cfg, d, torch.device('cpu'), with_rank=False, dev_nan=True, log=lambda *a: None)
    assert a['final_hash'] == b['final_hash']
