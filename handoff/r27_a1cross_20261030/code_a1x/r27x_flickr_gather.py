"""R27-A1X Flickr equivalence gate for the host image gather. Writes OUT.json.
Usage: r27x_flickr_gather.py OUT.json DATA_PATH STATE NOISE_FILE V12_DIR R27_CODE_DIR WORKDIR

STATE = a finished R27 Flickr A1 state_last.pt (read only; both queues ID-bearing).
  gather_equal     over 20 real A1 steps (host-gather mode) + 5 steps (GPU-resident mode): the gathered image rows equal the
                   ORIGINAL code_v12 path Ctx.gpu_images()[cols] bitwise, and the production re-encoded candidates Y equal
                   code_v12's reencode() (original module, loaded from V12_DIR) bitwise, both directions
  a1_steps_equal   3 real A1 steps from the same state and split in separate processes: R27 formal code/ (original A1),
                   code_a1x host-gather, code_a1x GPU-resident (twice: nondeterminism floor); encoder+refiner params compared
  timing           ms/step over 30 iterations from the same state, GPU-resident vs host gather
"""
import os, sys, json, copy, hashlib, time, glob, subprocess, importlib.util
import numpy as np
import torch

import r25_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
from r27_interface import Ctx
from data import get_dataset
from vse import VSE
from vocab import deserialize_vocab
from co_train import eval_train, split_prob

res = {}
HERE = os.path.dirname(os.path.abspath(__file__))


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:3000], flush=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def main():
    out, dpath, state, noise, v12dir, codedir, work = sys.argv[1:8]
    os.makedirs(work, exist_ok=True)
    # the original module imports vse / data: they must be byte-identical between the snapshots
    same = {f: sha(os.path.join(v12dir, f)) == sha(os.path.join(HERE, f)) for f in ('vse.py', 'data.py', 'model.py')}
    assert all(same.values()), same
    spec = importlib.util.spec_from_file_location('v12_iface', os.path.join(v12dir, 'r27_interface.py'))
    V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
    assert not hasattr(V.Ctx, 'image_rows')

    opt = r25_run.parse(['--noise_file', noise, '--output_dir', work, '--resume', state, '--data_path', dpath])
    opt.arm, opt.lam = 'a1_iface', 0.0
    R27.install()
    r25_run.seed_all(opt.seed)
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)); vocab.add_word('<mask>')
    opt.vocab_size = len(vocab)
    caps, ims = get_dataset(opt.data_path, opt.data_name, "train")
    noisy_loader, data_size, clean_labels = R.get_loader(caps, ims, vocab, "warmup", opt.batch_size, 0, opt.noise_ratio, opt.noise_file)
    V.Ctx.set(caps, ims, vocab)
    A = VSE(opt, vocab.word2idx); B = VSE(opt, vocab.word2idx)
    st = R.load_full_state(state, opt, A, B)
    ep = st['epoch'] + 1
    prob_A, prob_B, _ = eval_train(opt, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), clean_labels, ep)
    pred_A = split_prob(prob_A, opt.p_threshold); pred_B = split_prob(prob_B, opt.p_threshold)
    for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob[pred].mean()
    npz = os.path.join(work, 'split.npz')
    np.savez(npz, pred_B=np.asarray(pred_B), prob_B=np.asarray(prob_B), acpA=float(A.avg_clean_prob), acpB=float(B.avg_clean_prob))
    lt, ut = R.get_loader(caps, ims, vocab, "train", opt.batch_size, 0, opt.noise_ratio, opt.noise_file, pred=pred_B, prob=prob_B)
    snap = copy.deepcopy(R.model_full_state(A)), copy.deepcopy(R.model_full_state(B)), R.rng_state()

    def restore():
        for m, s0 in ((A, snap[0]), (B, snap[1])):
            s = copy.deepcopy(s0)
            m.img_enc.load_state_dict(s['img_enc']); m.txt_enc.load_state_dict(s['txt_enc']); m.optimizer.load_state_dict(s['optimizer'])
            m.i2t_refiner_node.load_state_dict(s['i2t_refiner']); m.t2i_refiner_node.load_state_dict(s['t2i_refiner'])
            q = m.memory_queue; q.img_queue.copy_(s['queue']['img']); q.txt_queue.copy_(s['queue']['txt']); q.ptr, q.is_full = s['queue']['ptr'], s['queue']['is_full']
            I.load_ids(q, s['r27_queue_ids'])
        R.set_rng_state(copy.deepcopy(snap[2]))
        R27._PEER.clear(); R27._PEER[id(A)] = 0; R27._PEER[id(B)] = 1

    # ---- gather / re-encoding equivalence vs the original code_v12 path
    eq = {}
    for gm, n in (('cpu', 20), ('gpu', 5)):
        restore()
        Ctx.gather_mode = gm
        c = dict(t2i_calls=0, i2t_calls=0, rows_bad=0, Y_vs_v12_bad=0, Y_repeat_bad=0, max_abs_Y_vs_v12=0.0, atoms=0)

        def probe(branch, d):
            S, Y, mean = d['S'], d['Y'], d['mean']
            net = A   # R27.train(opt, A, B, ...): A is the current peer
            Yv = V.reencode(net, branch, S['cols'], mean)
            Yr = I.reencode(net, branch, S['cols'], mean)
            c['Y_vs_v12_bad'] += int(not torch.equal(Y, Yv)); c['Y_repeat_bad'] += int(not torch.equal(Y, Yr))
            c['max_abs_Y_vs_v12'] = max(c['max_abs_Y_vs_v12'], float((Y - Yv).abs().max()))
            c['atoms'] += int(S['cols'].numel())
            if branch == 't2i':
                c['t2i_calls'] += 1
                c['rows_bad'] += int(not torch.equal(Ctx.image_rows(S['cols']), V.Ctx.gpu_images()[S['cols']]))
            else:
                c['i2t_calls'] += 1
        R27.train(opt, A, B, lt, ut, ep, R.new_stats(), max_iters=n, probe=probe)
        eq[gm] = c
    Ctx.gather_mode = None
    ok('gather_equal', all(eq[g]['rows_bad'] == 0 and eq[g]['Y_vs_v12_bad'] == 0 and eq[g]['t2i_calls'] >= eq[g]['i2t_calls'] > 0 for g in eq),
       note='original = code_v12 r27_interface (Ctx.gpu_images()[cols] / reencode), loaded from ' + v12dir, **eq)

    # ---- timing, same state, 30 iterations, both gather modes
    tim = {}
    for gm in ('gpu', 'cpu'):
        restore()
        Ctx.gather_mode = gm
        s = R.new_stats()
        torch.cuda.synchronize(); t = time.time()
        R27.train(opt, A, B, lt, ut, ep, s, max_iters=30)
        torch.cuda.synchronize()
        tim[gm] = dict(ms_per_step=1000 * (time.time() - t) / 30, reenc_t2i_s=s.get('time_reenc_t2i', 0) / max(s.get('time_n', 1), 1),
                       n_atoms_t2i_mean=s.get('t2i_n_atoms', 0) / max(s.get('t2i_n', 1), 1))
    Ctx.gather_mode = None
    ok('timing', True, device=torch.cuda.get_device_name(0), peak_alloc_GiB=torch.cuda.max_memory_allocated() / 2**30, **tim)
    del A, B
    torch.cuda.empty_cache()

    # ---- 3 real A1 steps in separate processes: code/ (formal R27 A1) vs code_a1x
    py = sys.executable
    script = os.path.join(HERE, 'r27x_onestep.py')
    outs = {}
    for tag, cdir, gm in (('r27_code', codedir, None), ('a1x_cpu', HERE, 'cpu'), ('a1x_gpu', HERE, 'gpu'), ('a1x_gpu_repeat', HERE, 'gpu')):
        o = os.path.join(work, f'onestep_{tag}.pt')
        env = dict(os.environ)
        env.pop('R27X_IMG_GATHER', None)
        if gm:
            env['R27X_IMG_GATHER'] = gm
        cmd = [py, '-B', '-c', f"import sys; sys.path.insert(0, {cdir!r}); exec(compile(open({script!r}).read(), {script!r}, 'exec'))",
               cdir, state, npz, dpath, noise, o, '3']
        r = subprocess.run(cmd, cwd=cdir, env=env, capture_output=True, text=True)
        print(tag, r.returncode, r.stdout[-500:], r.stderr[-2000:], flush=True)
        assert r.returncode == 0, tag
        outs[tag] = torch.load(o, weights_only=False)
        os.remove(o)
    ref = outs['r27_code']
    cmp = {}
    for tag in ('a1x_cpu', 'a1x_gpu', 'a1x_gpu_repeat'):
        cmp[tag] = dict(A_max_abs=float((outs[tag]['A'] - ref['A']).abs().max()), A_bitwise=bool(torch.equal(outs[tag]['A'], ref['A'])),
                        B_bitwise=bool(torch.equal(outs[tag]['B'], ref['B'])), gather=outs[tag]['gather'],
                        raw_loss=(outs[tag]['raw_i2t'], outs[tag]['raw_t2i']))
    floor = float((outs['a1x_gpu']['A'] - outs['a1x_gpu_repeat']['A']).abs().max())
    ok('a1_steps_equal', all(v['A_max_abs'] <= 1e-4 and v['B_bitwise'] for v in cmp.values()), r27_code_raw_loss=(ref['raw_i2t'], ref['raw_t2i']),
       nondeterminism_floor_gpu_repeat=floor, note='3 steps of model A; tolerance 1e-4 = host resume tolerance (cudnn.benchmark)', **cmp)

    res['code_sha256'] = {os.path.basename(f): sha(f) for f in sorted(glob.glob(os.path.join(HERE, '*.py')))}
    res['state'] = state
    res['job'] = os.environ.get('SLURM_JOB_ID')
    res['all_pass'] = all(v['pass_'] for v in res.values() if isinstance(v, dict) and 'pass_' in v)
    json.dump(res, open(out, 'w'), indent=1, default=float)
    print('FLICKR_GATHER all_pass', res['all_pass'])
    sys.exit(0 if res['all_pass'] else 1)


if __name__ == '__main__':
    main()
