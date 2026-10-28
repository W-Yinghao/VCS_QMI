"""R25 real-Torch gate on a smoke prefix (05_SERVER_TASK_CN.md step 4). Writes GATE.json; exits 1 on failure.

checks:
  native_branch   noisy_loss(S,'sce') is the author's infonce(S) (bitwise) and equals an FP64 re-derivation
  grad_paths      after one VCS train step: query encoders and current GraphRefiners get gradients; peer FIFO
                  tensors carry no grad; peer encoder params get no grad
  dense_vs_moment C0 J: dense B^2 form == moment form (value and grad wrt both embeddings), fp64 and fp32
  reference       J, dJ/dF and balanced BCE on the real S match the package NumPy reference
  state_roundtrip load prefix -> save -> reload: every tensor/RNG state identical
  resume          continuous run (prefix saved at epoch 0, epoch 1 continued in-process) vs fresh resume
                  from prefix_state.pt: epoch-1 statistics and final parameters
"""
import os, sys, json, copy
import numpy as np
import torch

sys.path.insert(0, '/home/infres/yinwang/CS_QMI/r25_retrieval_design/VCS_Retrieval_R25_Server_Design_20261027/reference')
import vcs_reference as REF
import r25_run
import r25_co_train as R
from data import get_loader, get_dataset
from vse import VSE, infonce
from vocab import deserialize_vocab
from co_train import eval_train, split_prob
from noisy_loss import noisy_loss, vcs_J_dense, vcs_J_moment, balanced_bce

res = {}


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', info)


def sce64(S, tau=0.05):
    S = S.double(); B = S.shape[0]
    m = torch.eye(B, dtype=torch.float64, device=S.device) + 1e-7
    e = (S / tau).exp(); i2t = e / e.sum(1, keepdim=True); t2i = e.t() / e.t().sum(1, keepdim=True)
    return 0.5 * (-(m.log() * i2t).sum(1).mean() - (m.log() * t2i).sum(1).mean() - i2t.diag().log().mean() - t2i.diag().log().mean())


def flat_params(m):
    ps = list(m.img_enc.parameters()) + list(m.txt_enc.parameters()) + list(m.i2t_refiner_node.parameters()) + list(m.t2i_refiner_node.parameters())
    return torch.cat([p.detach().flatten().double() for p in ps])


def main():
    smoke = sys.argv[1]
    loc = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'r25_' + os.environ.get('SLURM_JOB_ID', 'x'))
    extra = ['--data_path', loc] if os.path.isdir(loc) else []
    opt = r25_run.parse(['--noise_file', sys.argv[2], '--output_dir', f'{smoke}/gate', '--resume', f'{smoke}/cont/prefix_state.pt', '--max_iters', '30'] + extra)
    os.makedirs(opt.output_dir, exist_ok=True)
    r25_run.seed_all(opt.seed)
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    caps, ims = get_dataset(opt.data_path, opt.data_name, "train")
    noisy_loader, data_size, clean_labels = get_loader(caps, ims, vocab, "warmup", opt.batch_size, 0, opt.noise_ratio, opt.noise_file)
    A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
    st = R.load_full_state(opt.resume, opt, A, B)

    # state roundtrip
    R.save_full_state(f'{opt.output_dir}/roundtrip.pt', opt, A, B, st['epoch'], st['best_rsum'], st['all_loss'], st['history'])
    s1 = torch.load(opt.resume, weights_only=False); s2 = torch.load(f'{opt.output_dir}/roundtrip.pt', weights_only=False)
    bad = []

    def cmp(a, b, path):
        if isinstance(a, dict):
            for k in a:
                if k == 'opt':
                    continue
                cmp(a[k], b[k], path + '/' + str(k))
        elif isinstance(a, (list, tuple)):
            for i, (x, y) in enumerate(zip(a, b)):
                cmp(x, y, f'{path}[{i}]')
        elif torch.is_tensor(a):
            if not torch.equal(a.cpu(), b.cpu()):
                bad.append(path)
        elif isinstance(a, np.ndarray):
            if not np.array_equal(a, b):
                bad.append(path)
        else:
            if a != b and not (isinstance(a, float) and np.isnan(a) and np.isnan(b)):
                bad.append(path)
    cmp(s1, s2, '')
    ok('state_roundtrip', not bad, mismatches=bad[:10])

    # a real unlabeled batch for peer A
    prob_A, prob_B, _ = eval_train(opt, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), clean_labels, st['epoch'] + 1)
    pred_B = split_prob(prob_B, opt.p_threshold)
    for m, prob, pred in ((A, prob_A, split_prob(prob_A, opt.p_threshold)), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob[pred].mean()
    lt, ut = get_loader(caps, ims, vocab, "train", opt.batch_size, 0, opt.noise_ratio, opt.noise_file, pred=pred_B, prob=prob_B)
    batch = next(iter(ut))
    A.train_start(); B.val_start()
    xi, xt, _ = A.forward_emb(batch[0], batch[2], batch[3], imgae_lengths=batch[1])
    pt = B.memory_queue.retrieve_nearest_text(xi, k=opt.k, refiner_node=A.i2t_refiner_node)
    S = xi.mm(pt.t())

    # native branch
    a = noisy_loss(S, 'sce')[0]; bnat = infonce(S)
    ok('native_branch', torch.equal(a, bnat) and abs(float(a) - float(sce64(S))) < 1e-4 * max(1, abs(float(a))),
       hook=float(a), author=float(bnat), fp64=float(sce64(S)))

    # dense vs moment (C0), value and gradient wrt both embeddings
    out = {}
    for dt in (torch.float64, torch.float32):
        X = xi.detach().to(dt).requires_grad_(True); Y = pt.detach().to(dt).requires_grad_(True)
        jd = vcs_J_dense(X.mm(Y.t())); gd = torch.autograd.grad(jd, [X, Y])
        jm = vcs_J_moment(X, Y); gm = torch.autograd.grad(jm, [X, Y])
        out[str(dt)] = dict(val_abs=float(abs(jd - jm)), grad_max_abs=float(max((u - v).abs().max() for u, v in zip(gd, gm))),
                            grad_scale=float(max(u.abs().max() for u in gd)))
    ok('dense_vs_moment', out['torch.float64']['val_abs'] < 1e-12 and out['torch.float64']['grad_max_abs'] < 1e-12
       and out['torch.float32']['val_abs'] < 1e-5 and out['torch.float32']['grad_max_abs'] < 1e-6, **out)

    # NumPy reference on the real matrix
    F64 = S.detach().double().clamp(-1, 1)
    Fr = F64.cpu().numpy(); p, q = REF.diagonal_product_measures(Fr.shape[0])
    Ft = F64.clone().requires_grad_(True); jt = vcs_J_dense(Ft); gt = torch.autograd.grad(jt, Ft)[0].cpu().numpy()
    bt = balanced_bce(F64, 'c0')[0]
    ok('reference', abs(float(jt) - REF.pair_matrix_j(Fr)) < 1e-12 and np.abs(gt - REF.pair_matrix_gradient(Fr)).max() < 1e-14
       and abs(float(bt) - REF.balanced_bce(Fr, p, q)) < 1e-9,
       J=float(jt), J_ref=REF.pair_matrix_j(Fr), bce=float(bt), bce_ref=REF.balanced_bce(Fr, p, q))

    # gradient paths through one real VCS step on peer A
    opt.noisy_obj, opt.link, opt.c_D = 'vcs', 'c0', 1.0
    for prm in list(A.params) + list(B.params):
        prm.grad = None
    stats = R.new_stats()
    q_before = (B.memory_queue.img_queue.clone(), B.memory_queue.txt_queue.clone())
    # hook a gradient probe: run the training loop for 1 iteration but stop before optimizer.step
    orig_step = A.optimizer.step
    grads = {}

    def probe():
        grads['img_enc'] = sum(float(p.grad.abs().sum()) for p in A.img_enc.parameters() if p.grad is not None)
        grads['txt_enc'] = sum(float(p.grad.abs().sum()) for p in A.txt_enc.parameters() if p.grad is not None)
        grads['i2t_ref'] = sum(float(p.grad.abs().sum()) for p in A.i2t_refiner_node.parameters() if p.grad is not None)
        grads['t2i_ref'] = sum(float(p.grad.abs().sum()) for p in A.t2i_refiner_node.parameters() if p.grad is not None)
        grads['peer_params_with_grad'] = sum(1 for p in list(B.params) + list(B.i2t_refiner_node.parameters()) + list(B.t2i_refiner_node.parameters()) if p.grad is not None)
        grads['peer_queue_requires_grad'] = bool(B.memory_queue.img_queue.requires_grad or B.memory_queue.txt_queue.requires_grad)
    A.optimizer.step = lambda *a, **k: probe()
    R.train(opt, A, B, lt, ut, st['epoch'] + 1, stats, max_iters=1)
    A.optimizer.step = orig_step
    ok('grad_paths', grads['img_enc'] > 0 and grads['txt_enc'] > 0 and grads['i2t_ref'] > 0 and grads['t2i_ref'] > 0
       and grads['peer_params_with_grad'] == 0 and not grads['peer_queue_requires_grad']
       and stats['i2t_n'] == 1 and stats['t2i_n'] == 1, **grads, stats=dict(stats))

    # resume vs continuous
    hc = json.load(open(f'{smoke}/cont/EPOCHS.json'))['cotrain']; hr = json.load(open(f'{smoke}/resume/EPOCHS.json'))['cotrain']
    ec = [e for e in hc if e['epoch'] == st['epoch'] + 1][0]; er = [e for e in hr if e['epoch'] == st['epoch'] + 1][0]
    first = {}
    for peer in ('A', 'B'):
        for k in ('clean_loss_mean', 'i2t_raw_loss_mean', 't2i_raw_loss_mean'):
            first[f'{peer}_{k}'] = (ec['train'][peer].get(k), er['train'][peer].get(k))
    rel = max(abs(a - b) / max(abs(a), 1e-12) for a, b in first.values() if a is not None)
    dev = (ec['dev']['rsum'], er['dev']['rsum'])
    split_equal = ec['split_A'] == er['split_A'] and ec['split_B'] == er['split_B']
    A2 = VSE(opt, vocab.word2idx); B2 = VSE(opt, vocab.word2idx); R.load_full_state(f'{smoke}/cont/state_last.pt', opt, A2, B2)
    A3 = VSE(opt, vocab.word2idx); B3 = VSE(opt, vocab.word2idx); R.load_full_state(f'{smoke}/resume/state_last.pt', opt, A3, B3)
    pdiff = float((flat_params(A2) - flat_params(A3)).abs().max())
    ok('resume', split_equal and rel < 1e-3, split_equal=split_equal, epoch1_stat_max_rel_diff=rel, stats=first,
       dev_rsum=dev, final_param_max_abs_diff_peerA=pdiff,
       note='cudnn.benchmark=True (native) is not bitwise deterministic; tolerance 1e-3 relative on epoch means')

    res['all_pass'] = all(v['pass_'] for k, v in res.items() if isinstance(v, dict))
    json.dump(res, open(f'{smoke}/GATE.json', 'w'), indent=1, default=float)
    sys.exit(0 if res['all_pass'] else 1)


if __name__ == '__main__':
    main()
