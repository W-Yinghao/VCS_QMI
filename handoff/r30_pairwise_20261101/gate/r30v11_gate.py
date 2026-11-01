"""R30 v1.1 real-data GPU gate (docs/SERVER_PLAN_CN.md §8). Usage: r30v11_gate.py SMOKE_DIR DATA_PATH CODE_DIR REF_DIR PKG_DIR
-> SMOKE_DIR/GPU_GATE.json. Runs after sbatch/r30v11_smoke.sbatch (reduced to the gate minimum): smoke parents
wp_f60_s1001_H_e5 and wp_f60_s1001_L_e5 (2 warmup epochs x 30 it), one smoke calibration (H parent, 4 batches/peer), three real
branches with matrix unit ids (30 it per co-train epoch, milestone 1): H0 1 epoch, LN (L parent) 1 epoch, PV 2 epochs. All eight
objective/scale settings are exercised in-process from the same parent (fork identity, isolation steps, math).
Technical checks only:
  smoke_ran            parents (read-only file, sha, epoch -1, no refiner/queue, fingerprint re-computed == PARENT.json), calibs
                       (all_ok, parent unchanged), branches (DONE, FORK_REPORT ok, forked from the parent at epoch -1, registered
                       objective/scale/coefficient, R30 diagnostics, annotated history, METRICS.jsonl)
  fork_identity        (§8) all branches of one parent report the same parent fingerprint == PARENT.json; in-process: the same
                       parent forked under all 8 objective/scale settings gives identical model/optimizer/RNG fingerprints
  epoch_milestone      completed-epoch checkpoint read-only, raw = completed-1, history/metrics/ranks consistent, RNG unchanged by
                       the rank pass; METRICS accepted by the package tools/report_stage.read_metrics
  h_boundary_bitwise   (§8) in-process warmup_main(H) -> in-memory M + parent file; path N = M continued with native A1
                       r27_co_train.train; path R = fresh models + fork + R30 train(H). GMM, 3+3 steps: params / AdamW / RNG /
                       queues+IDs / counters / LR bitwise
  l_boundary_bitwise   same for the L source: path N = R29 train(l_listwise_vcs) continuing the in-memory L warmup; path R = fork
                       + R30 train(L, coefficient 1)
  isolation_boundary / isolation_filled  one step per objective from the same state (boundary-derived; filled-queue H0 smoke
                       state): same batch, RNG, noisy branch, other peer untouched, pre-clip grad minus main grad equal, scale rule
  calibration_inprocess  calibration peer_batches() on the same state: same batch, main-term norms == isolation main gradients
  calibration_records  smoke COEFFICIENTS arithmetic, ids, params unchanged, .grad untouched, parent sha/fingerprint unchanged
  math_gpu / pv_margin_gpu / rules_gpu  production H/L/N/B/PN/PV/PB terms vs the package reference on a real batch; PV margin
                       never reversed; multi-positive / duplicates / ties / transpose / no-negative on GPU
  resume_roundtrip     (§8) PV / Ls / N branch states: 3 steps -> save -> 3 live vs fresh R30 resume + 3 bitwise; refusals
  truth_isolation      _t2i_index poisoned in every gate loader, warmup _labels poisoned, labeled/unlabeled _labels scrambled
  host_unchanged       code_r29 files byte-identical; native eval_train / save_full_state underneath
  timing_memory        report only
"""
import os, sys, json, copy, hashlib, time, glob, math, stat, importlib.util
import numpy as np
import torch

CODE, REFDIR, PKG = sys.argv[3], sys.argv[4], sys.argv[5]
sys.path.insert(0, CODE)
_spec = importlib.util.spec_from_file_location('objective_checks', os.path.join(REFDIR, 'objective_checks.py'))
REF = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(REF)
torch.set_default_dtype(torch.float32)       # the reference module sets float64 globally at import
import r25_run, r30_run
import r25_co_train as R
import r27_co_train as R27
import r27_interface as I
import r29_co_train as R29
import r29_align as AL
import r30_align as A30
import r30_co_train as R30
import r30_calibrate as CAL
import r30_state as ST
import co_train
from data import get_dataset
from vocab import deserialize_vocab
sys.path.insert(0, os.path.join(PKG, 'tools'))
import planlib, report_stage
from pathlib import Path

res = {}
R29CODE = '/home/infres/yinwang/CS_QMI/handoff/r29_listwise_20261101/code_r29'
N60, N0 = os.environ['R30_NOISE60'], os.environ['R30_NOISE0']
P60H, P60L = 'wp_f60_s1001_H_e5', 'wp_f60_s1001_L_e5'
# label -> (objective, scale)
COMBOS = {'H': ('H', 'native'), 'L': ('L', '1'), 'Ls': ('L', 'fit_static'), 'PN': ('PN', 'fit_static'), 'PV': ('PV', 'fit_static'),
          'PB': ('PB', 'fit_static'), 'N': ('N', '1'), 'B': ('B', '1')}
SMOKE_UNITS = {'r30_f60_s1001_H0': (P60H, 'H', 1), 'r30v11_f60_s1001_LN': (P60L, 'N', 1), 'r30_f60_s1001_PV': (P60H, 'PV', 2)}


def ok(name, cond, **info):
    res[name] = dict(pass_=bool(cond), **info)
    print(name, 'PASS' if cond else 'FAIL', json.dumps(info, default=float)[:3000], flush=True)


def h(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()


sha = R30.sha256


class Poison:
    def _boom(self, *a, **k):
        raise RuntimeError('R30 gate: injected-noise truth was read')
    __getitem__ = __array__ = __len__ = __iter__ = _boom


def rng_eq(a, b):
    return (a['python'] == b['python'] and all(np.array_equal(x, y) for x, y in zip(a['numpy'][1:3], b['numpy'][1:3]))
            and torch.equal(a['torch'], b['torch']) and all(torch.equal(x, y) for x, y in zip(a['cuda'], b['cuda'])))


def own(m):
    ps = list(m.params)
    for name in ('i2t_refiner_node', 't2i_refiner_node'):
        if hasattr(m, name):
            ps += list(getattr(m, name).parameters())
    return ps


def flat(ps):
    return torch.cat([p.detach().flatten().double() for p in ps]).cpu()


def adam_flat(m):
    xs = [v.detach().flatten().double().cpu() for p in own(m) for k, v in m.optimizer.state.get(p, {}).items() if torch.is_tensor(v)]   # incl. CPU 'step'
    return torch.cat(xs).cpu() if xs else torch.zeros(0, dtype=torch.float64)


def qhash(m):
    if not hasattr(m, 'memory_queue'):
        return None
    q = m.memory_queue
    return h(q.img_queue) + h(q.txt_queue) + f'|{q.ptr}|{q.is_full}|' + ''.join(h(v) for v in getattr(q, 'r27_ids', {}).values())


def gdiff(a, b):
    m = 0.0
    for x, y in zip(a, b):
        if (x is None) != (y is None):
            return float('inf')
        if x is not None:
            m = max(m, float((x - y).abs().max()))
    return m


def maxdiff(a, b):
    return float((a - b).abs().max()) if a.numel() else 0.0


def state_record(A, B, stats=None):
    return dict(pA=flat(own(A)), pB=flat(own(B)), adA=adam_flat(A), adB=adam_flat(B), rng=R.rng_state(), qA=qhash(A), qB=qhash(B),
                eit=(A.Eiters, B.Eiters), acp=(float(A.avg_clean_prob), float(B.avg_clean_prob)),
                lrA=[g['lr'] for g in A.optimizer.param_groups], lrB=[g['lr'] for g in B.optimizer.param_groups], stats=stats)


def compare_records(X, Y):
    d = dict(pA=maxdiff(X['pA'], Y['pA']), pB=maxdiff(X['pB'], Y['pB']), adA=maxdiff(X['adA'], Y['adA']), adB=maxdiff(X['adB'], Y['adB']),
             sizes_equal=all(X[k].numel() == Y[k].numel() for k in ('pA', 'pB', 'adA', 'adB')),
             rng_equal=rng_eq(X['rng'], Y['rng']), queues_equal=(X['qA'] == Y['qA'] and X['qB'] == Y['qB']),
             eiters_equal=X['eit'] == Y['eit'], acp_equal=X['acp'] == Y['acp'], lr_equal=(X['lrA'] == Y['lrA'] and X['lrB'] == Y['lrB']))
    d['bitwise'] = (d['pA'] == 0 and d['pB'] == 0 and d['adA'] == 0 and d['adB'] == 0 and d['sizes_equal'] and d['rng_equal']
                    and d['queues_equal'] and d['eiters_equal'] and d['acp_equal'] and d['lr_equal'])
    return d


def readonly(p):
    return not (os.stat(p).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def main():
    smoke, dpath = sys.argv[1], sys.argv[2]
    G = f"{smoke}/gate_{os.environ.get('SLURM_JOB_ID', 'local')}"
    os.makedirs(G, exist_ok=True)
    VS = {}

    def base(noise_file, ratio, out, epochs=2):
        return ['--data_path', dpath, '--data_name', 'f30k_precomp', '--seed', '1001', '--k', '5', '--log_step', '10', '--noise_file', noise_file,
                '--noise_ratio', ratio, '--max_iters', '30', '--warmup_epoch', '2', '--num_epochs', str(epochs), '--prefix_epoch', '-1', '--output_dir', out]
    base60 = lambda out, epochs=2: base(N60, '0.6', out, epochs)
    PREC = {pid: json.load(open(f'{smoke}/parents/{pid}/PARENT.json')) for pid in (P60H, P60L)}
    CALJ = {pid: json.load(open(f'{smoke}/calib/{pid}/COEFFICIENTS_smoke.json')) for pid in (P60H,)}
    COEF = {lab: (None if sc == 'native' else 1.0 if sc == '1' else CALJ[P60H]['coefficients'][lab]['c']) for lab, (o, sc) in COMBOS.items()}
    matrix = {r['unit_id']: r for r in planlib.load_matrix(Path(PKG))}

    def unit_opt(pid, label, out, mode='fork', resume=None, unit_id='gate_unit', epochs=2, noise=(N60, '0.6'), milestone=1, coef=None):
        o, sc = COMBOS[label]
        c = COEF[label] if coef is None else coef
        a = ['--r30_stage', 'unit', '--r30_unit_id', unit_id, '--r30_source_warmup', PREC[pid]['warmup_source'], '--r30_objective', o,
             '--r30_scale', sc, '--r30_mode', mode, '--r30_parent', PREC[pid]['state_path'], '--r30_parent_sha', PREC[pid]['sha256'],
             '--r30_parent_id', pid, '--r30_parent_fingerprint', PREC[pid]['training_state_fingerprint'], '--r30_milestone', str(milestone)]
        if sc == 'fit_static':
            a += ['--r30_coef', repr(c)]
        if resume:
            a += ['--resume', resume]
        os.makedirs(out, exist_ok=True)
        opt_ = r30_run.parse(a + base(noise[0], noise[1], out, epochs))
        opt_.vocab_size = VS.get('n')
        return opt_

    # ------------------------------------------------------------------ smoke_ran / fork records / milestone (file based)
    info, good = {}, True
    for pid, rec in PREC.items():
        p = rec['state_path']
        st = torch.load(p, map_location='cpu', weights_only=False)
        fp = ST.fingerprint(st)
        e = dict(sha_ok=sha(p) == rec['sha256'], readonly=readonly(p), epoch=st['epoch'], stage=st['opt'].get('r30_stage'),
                 no_refiner=('i2t_refiner' not in st['model_A'] and 'queue' not in st['model_A']), n_warm=len(st['history']['warmup']),
                 all_loss=st['all_loss'] == [[], []], best=st['best_rsum'], fp_ok=fp['overall'] == rec['training_state_fingerprint'],
                 source=st['opt']['r30_source_warmup'], r29_arm=st['opt']['r29_arm'], data=bool(rec['data']['data_split_digest']),
                 diag=os.path.exists(f'{smoke}/parents/{pid}/R29_WARMUP_DIAG.json'), dev=[x['rsum'] for x in rec['warmup_dev']])
        e['ok'] = (e['sha_ok'] and e['readonly'] and e['epoch'] == -1 and e['stage'] == 'boundary' and e['no_refiner'] and e['n_warm'] == 2
                   and e['all_loss'] and e['best'] == 0 and e['fp_ok'] and e['r29_arm'] == R30.SOURCE_ARM[e['source']] and e['data'] and e['diag']
                   and bool(np.isfinite(e['dev']).all()))
        good &= e['ok']; info[pid] = e
        del st
    for pid, c in CALJ.items():
        e = dict(all_ok=c['all_ok'], parent_unchanged=c['parent_unchanged'], coefs={k: c['coefficients'][k]['c'] for k in c['coefficients']},
                 parent_sha_ok=c['parent_sha256'] == PREC[pid]['sha256'], fork_report=json.load(open(f'{smoke}/calib/{pid}/CALIBRATION_FORK_REPORT.json'))['ok'])
        e['ok'] = e['all_ok'] and e['parent_unchanged'] and e['parent_sha_ok'] and e['fork_report']
        good &= e['ok']; info['calib_' + pid] = e
    fork_fps, ms, metrics_lines = {}, {}, []
    for uid, (pid, lab, nep) in SMOKE_UNITS.items():
        d = f'{smoke}/runs/{uid}'
        e = {'done': os.path.exists(f'{d}/DONE.json')}
        if e['done']:
            hist = json.load(open(f'{d}/EPOCHS.json'))
            run = json.load(open(sorted(glob.glob(f'{d}/RUN_*.json'))[0]))['opt']
            fr = json.load(open(f'{d}/FORK_REPORT.json'))
            mrow = matrix[uid]
            rs = [c['dev']['rsum'] for c in hist['cotrain']]
            tr = [c['train'][p] for c in hist['cotrain'] for p in ('A', 'B')]
            o, sc = COMBOS[lab]
            want_coef = None if sc == 'native' else 1.0 if sc == '1' else CALJ[pid]['coefficients'][lab]['c']
            e.update(n_cotrain=len(hist['cotrain']), rsum=rs, resumed_from=hist.get('resumed_from'),
                     finite=bool(np.isfinite(rs).all() and all(np.isfinite(t.get('clean_loss_mean', np.nan)) for t in tr)),
                     coef_ok=run['r30_coef'] == want_coef, term_ok=(run['r30_objective'], run['r30_scale']) == (o, sc) == (mrow['continuation_objective'], {'native': 'native', '1': '1', 'fit_static': 'fit_static'}[mrow['scale_rule']]),
                     source_ok=run['r30_source_warmup'] == mrow['warmup_source'] == PREC[pid]['warmup_source'] and mrow['parent_id'] == pid,
                     fork_ok=fr['ok'] and fr['components_equal'] and fr['parent_fingerprint'] == PREC[pid]['training_state_fingerprint'],
                     fork_allowed_diffs=sorted(fr['opt_differences']),
                     r30_keys=all('r30_main_raw_mean' in t and 'r30_clip_rate' in t and 'r30_i2t_invalid_frac' in t and len(t.get('r30_trace', [])) > 0 for t in tr),
                     annotated=[(c.get('raw_epoch'), c.get('completed_cotrain_epochs'), c.get('updates_cumulative')) for c in hist['cotrain']],
                     main_raw=[t.get('r30_main_raw_mean') for t in tr], main_scaled=[t.get('r30_main_scaled_mean') for t in tr], clip=[t.get('r30_clip_rate') for t in tr])
            fork_fps.setdefault(pid, set()).add(fr['parent_fingerprint'])
            ups = [x[2] for x in e['annotated']]
            e['ok'] = (e['n_cotrain'] == nep and e['finite'] and e['coef_ok'] and e['term_ok'] and e['source_ok'] and e['fork_ok'] and e['r30_keys']
                       and (e['resumed_from'] or {}).get('epoch') == -1 and (e['resumed_from'] or {}).get('path') == PREC[pid]['state_path']
                       and [x[:2] for x in e['annotated']] == [(i, i + 1) for i in range(nep)] and ups[0] > 0 and all(b > a for a, b in zip(ups, ups[1:])))
            # milestone 1 (completed 1 = raw 0)
            ck, er = f'{d}/state_completed1.pt', json.load(open(f'{d}/E1_RECORD.json'))
            cst = torch.load(ck, map_location='cpu', mmap=True, weights_only=False)
            ms[uid] = dict(readonly=readonly(ck) and readonly(f'{d}/E1_RECORD.json'), raw=cst['epoch'], ck_hist=len(cst['history']['cotrain']),
                           record=(er['completed_cotrain_epochs'], er['raw_epoch']), dev_equal=er['dev'] == hist['cotrain'][0]['dev'],
                           recomputed_equal=er['recomputed_equal'], rng_unchanged=er['rng_unchanged_by_rank_pass'], ck_sha=sha(ck) == er['checkpoint_sha256'],
                           ranks=os.path.exists(er['ranks_file']) and sha(er['ranks_file']) == er['ranks_sha256'],
                           ck_fp_differs_from_last=ST.fingerprint(cst)['overall'] != ST.fingerprint(torch.load(f'{d}/state_last.pt', map_location='cpu', weights_only=False))['overall'],
                           rng_consumed_before_restore=er.get('rng_consumed_by_rank_pass_before_restore'), n_epochs=nep)
            # milestone at the last epoch (1-epoch branch): checkpoint == final state; milestone then continue (2 epochs): differs
            ms[uid]['ok'] = (ms[uid]['readonly'] and ms[uid]['raw'] == 0 and ms[uid]['ck_hist'] == 1 and ms[uid]['record'] == (1, 0) and ms[uid]['dev_equal']
                             and ms[uid]['recomputed_equal'] and ms[uid]['rng_unchanged'] and ms[uid]['ck_sha'] and ms[uid]['ranks']
                             and ms[uid]['ck_fp_differs_from_last'] == (nep > 1))
            metrics_lines += open(f'{d}/METRICS.jsonl').read().splitlines()
            pk = glob.glob(f'{d}/PEAK_*.json')
            e['peak'] = json.load(open(pk[0])) if pk else None
        else:
            e['ok'] = False
        good &= e['ok']; info[uid] = e
    ok('smoke_ran', good, **info)
    mp = f'{G}/SMOKE_METRICS.jsonl'
    open(mp, 'w').write('\n'.join(metrics_lines) + '\n')
    try:
        rd = report_stage.read_metrics(Path(mp), list(matrix.values())); reader = True
    except Exception as ex:
        rd, reader = {}, repr(ex)
    ok('epoch_milestone', all(v['ok'] for v in ms.values()) and len(ms) == len(SMOKE_UNITS) and reader is True and len(rd) == sum(1 + v[2] for v in SMOKE_UNITS.values()),
       milestones=ms, package_reader=reader, metric_rows=len(rd))

    # ------------------------------------------------------------------ data / loaders (gate-owned, truth poisoned)
    R29.install()
    vocab = deserialize_vocab(os.path.join('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab', 'f30k_precomp_vocab.json'))
    vocab.add_word('<mask>')
    w2i = vocab.word2idx
    VS['n'] = len(vocab)
    caps, ims = get_dataset(dpath, 'f30k_precomp', 'train')
    noisy_loader, data_size, _t = R.get_loader(caps, ims, vocab, 'warmup', 128, 0, 0.6, N60)
    del _t
    noisy_loader.dataset._t2i_index = Poison(); noisy_loader.dataset._labels = Poison()

    def loaders(pred, prob, scramble=False):
        lt, ut = R.get_loader(caps, ims, vocab, 'train', 128, 0, 0.6, N60, pred=pred, prob=prob)
        lt.dataset._t2i_index = Poison(); ut.dataset._t2i_index = Poison()
        if scramble:
            lt.dataset._labels = [1 - int(x) for x in lt.dataset._labels]; ut.dataset._labels = [1 - int(x) for x in ut.dataset._labels]
        return lt, ut

    def inproc_warmup(source, tag):
        """exact warmup_main in-process (truth poisoned) -> captured in-memory models + parent record."""
        R29.install()
        ow = r30_run.parse(['--r30_stage', 'warmup', '--r30_parent_id', P60H if source == 'H' else P60L, '--r30_source_warmup', source]
                           + base60(f'{G}/warm_{tag}'))
        cap = {}
        orig_sb, orig_gl = R30.save_boundary, R.get_loader

        def sb(o, A, B, best, al, hist):
            rec = orig_sb(o, A, B, best, al, hist)
            cap.update(A=A, B=B, all_loss=al, rec=rec, rng=R.rng_state())
            return rec

        def gl(*a, **k):
            out = orig_gl(*a, **k)
            if a[3] == 'warmup':
                out[0].dataset._t2i_index = Poison(); out[0].dataset._labels = Poison()
            return out
        R30.save_boundary, R.get_loader = sb, gl
        r25_run.seed_all(ow.seed)
        R30.warmup_main(ow)
        R30.save_boundary, R.get_loader = orig_sb, orig_gl
        R30.install()
        cap['vocab_size'] = ow.vocab_size
        return cap

    def gate_opt(cap, label, out, source):
        o, sc = COMBOS[label]
        a = ['--r30_stage', 'unit', '--r30_unit_id', 'gate_unit', '--r30_source_warmup', source, '--r30_objective', o, '--r30_scale', sc,
             '--r30_mode', 'fork', '--r30_parent', cap['rec']['state_path'], '--r30_parent_sha', cap['rec']['sha256'],
             '--r30_parent_id', cap['rec']['parent_id'], '--r30_parent_fingerprint', cap['rec']['training_state_fingerprint'], '--r30_milestone', '1']
        if sc == 'fit_static':
            a += ['--r30_coef', repr(COEF[label])]
        os.makedirs(out, exist_ok=True)
        opt = r30_run.parse(a + base60(out))
        opt.vocab_size = cap['vocab_size']
        return opt

    def prelude(o, A, B, all_loss):
        R27._PEER.clear()
        pa, proba, pb, probb = CAL.epoch0_prelude(o, A, B, noisy_loader, data_size, all_loss, 0)
        return dict(pred_A=pa, prob_A=proba, pred_B=pb, prob_B=probb)

    def steps(o, A, B, pp, fn, n):
        st = {}
        for name, net, net2, pred, prob in (('A', A, B, pp['pred_B'], pp['prob_B']), ('B', B, A, pp['pred_A'], pp['prob_A'])):
            lt, ut = loaders(pred, prob)
            s = R.new_stats()
            fn(o, net, net2, lt, ut, 0, s, max_iters=n)
            st[name] = {k: s.get(k) for k in ('clean_loss', 'steps', 'i2t_raw_loss', 't2i_raw_loss', 'i2t_n', 't2i_n', 'i2t_no_proto', 't2i_no_proto')}
        return st

    def boundary_bitwise(source, native_fn, label, opt_native_arm):
        cap = inproc_warmup(source, source)
        opt = gate_opt(cap, label, f'{G}/u_{source}', source)
        A, B = cap['A'], cap['B']
        R.set_rng_state(cap['rng'])
        o_n = copy.copy(opt); o_n.r29_arm = opt_native_arm
        ppN = prelude(o_n, A, B, cap['all_loss'])
        stN = steps(o_n, A, B, ppN, native_fn, 3)
        recN = state_record(A, B, stN)
        del A, B, cap['A'], cap['B']
        torch.cuda.empty_cache()
        A2, B2 = R.VSE(opt, w2i), R.VSE(opt, w2i)
        B2.load_state_dict(A2.state_dict())
        stb = R.load_full_state(cap['rec']['state_path'], opt, A2, B2)          # fork (identity rule + FORK_REPORT)
        fr = json.load(open(f'{G}/u_{source}/FORK_REPORT.json'))
        ppR = prelude(opt, A2, B2, stb['all_loss'])
        snap = dict(A=copy.deepcopy(R.model_full_state(A2)), B=copy.deepcopy(R.model_full_state(B2)), rng=R.rng_state())
        stR = steps(opt, A2, B2, ppR, R30.train, 3)
        recR = state_record(A2, B2, stR)
        d = compare_records(recN, recR)
        d.update(prob_equal=bool(np.array_equal(ppN['prob_A'], ppR['prob_A']) and np.array_equal(ppN['prob_B'], ppR['prob_B'])),
                 pred_equal=bool(np.array_equal(ppN['pred_A'], ppR['pred_A']) and np.array_equal(ppN['pred_B'], ppR['pred_B'])),
                 stats_equal=stN == stR, stats_native=stN, noisy_ran=(stN['B'].get('i2t_n') or 0) >= 1 and (stN['B'].get('t2i_n') or 0) >= 1,
                 fork_report_ok=fr['ok'], parent_sha=cap['rec']['sha256'])
        del A2, B2
        torch.cuda.empty_cache()
        return d, cap, opt, ppR, snap

    # ------------------------------------------------------------------ h_boundary_bitwise (H source)
    d, capH, opt, ppR, snapB = boundary_bitwise('H', R27.train, 'H', None)
    ok('h_boundary_bitwise', d['bitwise'] and d['prob_equal'] and d['pred_equal'] and d['stats_equal'] and d['noisy_ran'] and d['fork_report_ok'], **d)
    sm = torch.load(PREC[P60H]['state_path'], map_location='cpu', weights_only=False); ip = torch.load(capH['rec']['state_path'], map_location='cpu', weights_only=False)
    xproc = max(float((v.double() - ip[k1][k2][k3].double()).abs().max()) for k1 in ('model_A', 'model_B') for k2 in ('img_enc', 'txt_enc')
                for k3, v in sm[k1][k2].items() if torch.is_floating_point(v))
    res['parent_crossprocess'] = dict(report_only=True, param_maxabsdiff=xproc, bitwise=xproc == 0.0,
                                      dev_smoke=[x['rsum'] for x in PREC[P60H]['warmup_dev']], dev_inproc=[x['rsum'] for x in capH['rec']['warmup_dev']])
    del sm, ip

    # ------------------------------------------------------------------ fork_identity in-process: same parent, all objectives
    fps, good = {}, True
    for lab in COMBOS:
        o = gate_opt(capH, lab, f'{G}/fork_{lab}', 'H')
        A3, B3 = R.VSE(o, w2i), R.VSE(o, w2i)
        st = R.load_full_state(capH['rec']['state_path'], o, A3, B3)
        fps[lab] = ST.fingerprint(ST.live_state(R, A3, B3, st['epoch'], st['best_rsum'], st['all_loss']))['overall']
        del A3, B3
    torch.cuda.empty_cache()
    ok('fork_identity', len(set(fps.values())) == 1 and fps['H'] == capH['rec']['training_state_fingerprint']
       and all(len(v) == 1 and next(iter(v)) == PREC[k]['training_state_fingerprint'] for k, v in fork_fps.items()),
       inprocess_live_fingerprints=fps, parent_fingerprint=capH['rec']['training_state_fingerprint'],
       smoke_branch_parent_fingerprints={k: sorted(v) for k, v in fork_fps.items()})

    # ------------------------------------------------------------------ isolation hooks
    hk = {}
    SCORE = []
    orig_main, orig_clip = A30.main_term, R30.clip_grad_norm_

    def main_wrap(o, net, sims, obs):
        l1, raw = orig_main(o, net, sims, obs)
        if hk.get('want'):
            sd = sims.detach()
            SCORE.append(dict(objective=o.r30_objective, scale=o.r30_scale, autocast=torch.is_autocast_enabled(), sims_dtype=str(sims.dtype),
                              loss_dtype=str(l1.dtype), raw_dtype=str(raw.dtype), param_dtypes=sorted({str(p.dtype) for p in net.params}),
                              smin=float(sd.min()), smax=float(sd.max()), diag_mean=float(sd.diagonal().mean()), off_mean=float((sd.sum() - sd.diagonal().sum()) / (sd.numel() - sd.shape[0])),
                              multipos_rows=int(((obs[:, None] == obs[None, :]).sum(1) > 1).sum()), n=int(sd.shape[0])))
            g = torch.autograd.grad(l1, list(net.params), retain_graph=True, allow_unused=True)
            hk.update(g_main=[None if x is None else x.detach().clone() for x in g], obs=h(obs), sims=sims.detach().clone(),
                      l1=float(l1.detach()), raw=float(raw.detach()), same_obj=l1 is raw)
        return l1, raw

    def clip_wrap(params, mx):
        if hk.get('want'):
            hk['g_pre'] = [None if p.grad is None else p.grad.detach().clone() for p in params]
            hk['want'] = False
        return orig_clip(params, mx)

    def restore_enc(m, s):
        m.img_enc.load_state_dict(s['img_enc']); m.txt_enc.load_state_dict(s['txt_enc'])
        m.optimizer.load_state_dict(copy.deepcopy(s['optimizer']))
        m.Eiters = s['Eiters']; m.avg_clean_prob = s['avg_clean_prob']

    def one_step(o, A, B, lt, ut, lab, n=1, want=True):
        o.r30_objective, o.r30_scale = COMBOS[lab]; o.r30_coef = COEF[lab]
        c, seen = {}, {}
        orig_step = A.optimizer.step

        def step_probe(*a, **k):
            if 'enc' not in c:
                c['enc'] = [None if p.grad is None else p.grad.detach().clone() for p in A.params]
                c['peer_with_grad'] = sum(1 for p in own(B) if p.grad is not None)
            return orig_step(*a, **k)

        def probe(branch, dd):
            seen[branch] = dict(cols=h(dd['S']['cols']), A=h(dd['S']['A']), l_nat=float(dd['l_nat'].detach()))
        A.optimizer.step = step_probe
        hk.clear(); hk['want'] = want
        s = R.new_stats()
        R30.train(o, A, B, lt, ut, 0, s, max_iters=n, probe=probe)
        A.optimizer.step = orig_step
        c.update(params=flat(own(A)), rng=R.rng_state(), qB=qhash(B), hk=dict(hk), seen=seen, stats=dict(s))
        return c

    def iso_eval(results, ctx, tol=2e-3):
        g_rest = {a: [None if x is None else (x - (y if y is not None else 0)) for x, y in zip(r['hk']['g_pre'], r['hk']['g_main'])] for a, r in results.items()}
        gscale = max(float(x.abs().max()) for x in g_rest['H'] if x is not None)
        href = results['H']
        gmax = lambda gs: max(float(x.abs().max()) for x in gs if x is not None)
        tr0 = lambda r: r['stats']['r30_trace'][0]
        d, good = {}, True
        for a, r in results.items():
            raw, l1 = r['hk']['raw'], r['hk']['l1']
            scale_ok = (r['hk']['same_obj'] and l1 == raw) if COMBOS[a][1] != 'fit_static' else (abs(l1 - COEF[a] * raw) <= 1e-5 * max(abs(l1), 1e-12))
            diff = gdiff(g_rest[a], g_rest['H'])
            tot = max(gmax(r['hk']['g_pre']), gmax(href['hk']['g_pre']))
            e = dict(obs_equal=r['hk']['obs'] == href['hk']['obs'], rng_equal=rng_eq(r['rng'], href['rng']), peer_with_grad=r['peer_with_grad'],
                     qB_equal=r['qB'] == href['qB'], noisy_equal=r['seen'] == href['seen'], noisy_branches=sorted(r['seen']),
                     intra_value_equal=tr0(r)['intra'] == tr0(href)['intra'],
                     intra_grad_norm_rel=abs(tr0(r)['g_intra'] - tr0(href)['g_intra']) / max(tr0(href)['g_intra'], 1e-30),
                     g_rest_absdiff=diff, g_rest_rel_to_residual=diff / gscale, g_pre_max=gmax(r['hk']['g_pre']), g_main_max=gmax(r['hk']['g_main']),
                     g_rest_rel_to_total=diff / tot, l1=l1, raw=raw, coef=COEF[a], scale_ok=scale_ok,
                     g_main=sum(float((x.double() ** 2).sum()) for x in r['hk']['g_main'] if x is not None) ** 0.5)
            # bitwise: batch, RNG, other peer, queue, noisy-branch candidates/targets/loss, intra-loss value; cancellation-free: the
            # directly computed intra-term gradient norm; g_pre - g_main is a float32 difference of the summed backward and the separate
            # main-term gradient, so its residual is judged relative to the total gradient magnitude
            e['ok'] = (e['obs_equal'] and e['rng_equal'] and e['peer_with_grad'] == 0 and e['qB_equal'] and e['noisy_equal'] and e['intra_value_equal']
                       and e['intra_grad_norm_rel'] <= 1e-5 and e['g_rest_rel_to_total'] <= tol and e['scale_ok'])
            good &= e['ok']; d[a] = e
        if ctx == 'filled':
            good &= all(len(v['noisy_branches']) == 2 for v in d.values())
        return good, d, gscale

    # ------------------------------------------------------------------ isolation from the boundary-derived state (fork path)
    A30.main_term, R30.clip_grad_norm_ = main_wrap, clip_wrap
    isoB = {}
    for lab in COMBOS:
        A3, B3 = R.VSE(opt, w2i), R.VSE(opt, w2i)
        restore_enc(A3, snapB['A']); restore_enc(B3, snapB['B']); R.set_rng_state(snapB['rng']); R27._PEER.clear()
        lt, ut = loaders(ppR['pred_B'], ppR['prob_B'])
        isoB[lab] = one_step(opt, A3, B3, lt, ut, lab)
        del A3, B3
    # same steps with cuDNN TF32 disabled (native flag: torch default allow_tf32=True): separates TF32 rounding from real differences
    tf32_native = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    isoB32 = {}
    for lab in COMBOS:
        A3, B3 = R.VSE(opt, w2i), R.VSE(opt, w2i)
        restore_enc(A3, snapB['A']); restore_enc(B3, snapB['B']); R.set_rng_state(snapB['rng']); R27._PEER.clear()
        lt, ut = loaders(ppR['pred_B'], ppR['prob_B'])
        isoB32[lab] = one_step(opt, A3, B3, lt, ut, lab)
        del A3, B3
    torch.backends.cudnn.allow_tf32 = tf32_native
    A30.main_term, R30.clip_grad_norm_ = orig_main, orig_clip
    good, d, gscale = iso_eval(isoB, 'boundary')
    ok('isolation_boundary', good, g_scale=gscale, tolerance_rel_to_total=2e-3, cudnn_allow_tf32=tf32_native,
       note='native flags (cuDNN TF32 allowed, torch default, as in every native A1/R29 run): g_pre - g_main residual limited by TF32 rounding', **d)
    good32, d32, _ = iso_eval(isoB32, 'boundary', tol=1e-4)
    ok('isolation_boundary_fp32', good32, tolerance_rel_to_total=1e-4, cudnn_allow_tf32=False,
       note='diagnostic pass with cuDNN TF32 disabled; strict float32 tolerance', **d32)
    del isoB32

    # ------------------------------------------------------------------ calibration in-process (same state, same first batch)
    A3, B3 = R.VSE(opt, w2i), R.VSE(opt, w2i)
    restore_enc(A3, snapB['A']); restore_enc(B3, snapB['B']); R.set_rng_state(snapB['rng']); R27._PEER.clear()
    lt, ut = loaders(ppR['pred_B'], ppR['prob_B'])
    encd = lambda: [ST.digest([[p.detach() for p in m.params], m.Eiters, R.model_full_state(m)['optimizer']['state']]) for m in (A3, B3)]
    bufd = lambda: [ST.digest([b.detach() for mod in (m.img_enc, m.txt_enc) for b in mod.buffers()]) for m in (A3, B3)]
    fp0, bf0 = encd(), bufd()
    recs, _, _ = CAL.peer_batches(opt, A3, B3, lt, ut, 1, 'A')
    sq = recs[0]['sqnorm']
    obs_b = torch.as_tensor(np.asarray(lt.dataset.t2i_index, dtype=np.int64)[np.asarray(recs[0]['ids'], dtype=np.int64)], device='cuda')
    sqm = {a: sum(float((x.double() ** 2).sum()) for x in isoB[a]['hk']['g_main'] if x is not None) for a in COMBOS}
    want = {'H': sq['H'], 'L': sq['L'], 'Ls': COEF['Ls'] ** 2 * sq['L'], 'PN': COEF['PN'] ** 2 * sq['PN'], 'PV': COEF['PV'] ** 2 * sq['PV'], 'PB': COEF['PB'] ** 2 * sq['PB']}
    rel = {a: abs(sqm[a] - want[a]) / max(want[a], 1e-30) for a in want}
    enc_unchanged = fp0 == encd()
    bn_buffers_changed = bf0 != bufd()          # expected: train-mode forward updates BatchNorm running stats of this discarded copy
    sims_same = all(torch.equal(isoB['H']['hk']['sims'], isoB[a]['hk']['sims']) for a in COMBOS)
    d = dict(rel=rel, sq_calib=sq, sq_isolation=sqm, sims_equal_across_objectives=sims_same, obs_equal=h(obs_b) == isoB['H']['hk']['obs'],
             grads_untouched=all(p.grad is None for p in A3.params), params_optimizer_counters_unchanged=enc_unchanged,
             bn_running_stats_changed_on_the_copy=bn_buffers_changed, batch_ids_sha=recs[0]['ids_sha256'],
             note='the calibration runs in its own process on a discarded copy of the read-only parent; train-mode BatchNorm uses batch '
                  'statistics, so the gradients do not depend on the running stats it updates; parent sha256 + fingerprint verified unchanged (calibration_records)')
    ok('calibration_inprocess', all(v <= 1e-4 for v in rel.values()) and sims_same and d['obs_equal'] and d['grads_untouched'] and enc_unchanged, **d)
    del A3, B3, recs
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ math on a real batch (peer A's first labeled batch)
    S32 = isoB['H']['hk']['sims'].float()
    math_, good = {}, True
    ob = obs_b.cpu()
    for kind, lab in (('nce', 'PN'), ('vcs', 'PV'), ('bce', 'PB')):
        x32 = S32.clone().requires_grad_(); x64 = S32.double().clone().requires_grad_()
        v32 = A30.pair_two_way(kind, x32, obs_b); g32 = torch.autograd.grad(v32, x32)[0]
        v64 = A30.pair_two_way(kind, x64, obs_b); g64 = torch.autograd.grad(v64, x64)[0]
        xr = S32.double().cpu().requires_grad_()
        r1, _ = REF.hard_pair_rows(xr, ob, kind, A30.TAU); r2, _ = REF.hard_pair_rows(xr.t(), ob, kind, A30.TAU)
        vr = r1.sum() + r2.sum(); gr = torch.autograd.grad(vr, xr)[0]
        idx32 = A30.hard_pair_rows(S32, obs_b, kind)[2].cpu()
        pos = ob[:, None].eq(ob[None, :])
        idxr = xr.detach().masked_fill(pos, float('-inf')).argmax(1)
        gs = float(gr.abs().max())
        e = dict(v64_err=abs(float(v64) - float(vr)) / abs(float(vr)), g64_err=float((g64.cpu() - gr).abs().max()) / gs,
                 v32_rel=abs(float(v32) - float(vr)) / abs(float(vr)), g32_rel=float((g32.double().cpu() - gr).abs().max()) / gs,
                 idx_equal=bool(torch.equal(idx32, idxr)), value=float(vr), multipos_rows=int((pos.sum(1) > 1).sum()))
        e['ok'] = e['v64_err'] < 1e-10 and e['g64_err'] < 1e-10 and e['v32_rel'] < 1e-4 and e['g32_rel'] < 1e-4 and e['idx_equal']
        good &= e['ok']; math_[lab] = e
    Aobs = AL.target_from_obs(ob, torch.float64)
    for obj, kind in (('L', 'vcs'), ('N', 'nce'), ('B', 'bce')):
        xl32 = S32.clone().requires_grad_()
        vl32 = A30.raw_main(obj, None, xl32, obs_b); gl32 = torch.autograd.grad(vl32, xl32)[0]
        xr = S32.double().cpu().requires_grad_()
        vlr = REF.list_loss(xr / A30.TAU, Aobs, kind).sum() + REF.list_loss(xr.t() / A30.TAU, Aobs.t(), kind).sum()
        glr = torch.autograd.grad(vlr, xr)[0]
        e = dict(v32_rel=abs(float(vl32) - float(vlr)) / abs(float(vlr)), g32_rel=float((gl32.double().cpu() - glr).abs().max()) / float(glr.abs().max()))
        e['ok'] = e['v32_rel'] < 1e-4 and e['g32_rel'] < 1e-4
        good &= e['ok']; math_[obj] = e
    t = torch.tensor([[0.0, 3.0, 1.0, 3.0], [3.0, 0.0, 3.0, 1.0], [1.0, 2.0, 0.0, 2.0], [2.0, 2.0, 2.0, 0.0]], device='cuda')
    math_['cuda_argmax_tie_first'] = A30.hard_pair_rows(t, torch.arange(4, device='cuda'), 'vcs')[2].tolist() == [1, 0, 1, 0]
    good &= math_['cuda_argmax_tie_first']
    ok('math_gpu', good, **math_)

    mg = torch.linspace(-60, 60, 24001, device='cuda', requires_grad=True)
    g = torch.autograd.grad(A30.pair_loss(mg, 'vcs').sum(), mg)[0].double()
    p = mg.detach().double().sigmoid()
    analytic = -8 * p * (1 - p) ** 2 * ((1 + 2 * p) ** -3 + (3 - 2 * p) ** -3)
    rev = (g > 0) & (analytic.abs() >= 1e-6)
    x32 = S32.clone().requires_grad_()
    sc = {}
    for dname in ('i2t', 't2i'):
        M = x32 if dname == 'i2t' else x32.t()
        rows, valid, idx, pos = A30.hard_pair_rows(M, obs_b, 'vcs')
        gg = torch.autograd.grad(rows.sum(), x32)[0]
        gg = gg if dname == 'i2t' else gg.t()
        sel = torch.zeros_like(pos); sel[torch.arange(pos.shape[0], device='cuda'), idx] = True; sel &= valid[:, None]
        other = ~pos & ~sel
        mm = ((M - M.gather(1, idx[:, None])) / A30.TAU).detach().double()
        pp = mm.sigmoid()
        an = (-8 * pp * (1 - pp) ** 2 * ((1 + 2 * pp) ** -3 + (3 - 2 * pp) ** -3))
        pos_v = pos & valid[:, None]
        sc[dname] = dict(pos_reversed=int(((gg > 0) & pos_v & (an.abs() >= 1e-6)).sum()), pos_max=float(gg[pos_v].max()),
                         neg_min=float(gg[sel].min()), neg_reversed=int(((gg < 0) & sel).sum()), other_max_abs=float(gg[other].abs().max()),
                         comparisons=int(pos_v.sum()), saturated=int((pos_v & (an.abs() < 1e-6)).sum()))
    good = int(rev.sum()) == 0 and all(v['pos_reversed'] == 0 and v['neg_reversed'] == 0 and v['other_max_abs'] == 0 for v in sc.values())
    ok('pv_margin_gpu', good, grid_positive=int((g > 0).sum()), grid_reversed_unsaturated=int(rev.sum()), grid_max=float(g.max()), score_level=sc)

    rr, good = {}, True
    gen = torch.Generator().manual_seed(30)
    cases = {'multipos': torch.tensor([0, 0, 1, 2, 2, 2, 3, 4]), 'distinct': torch.arange(8), 'noneg': torch.zeros(8, dtype=torch.long)}
    for cname, ids in cases.items():
        Sx = torch.randn(8, 8, generator=gen, dtype=torch.float64) * 0.3
        if cname == 'multipos':
            Sx[:, 1] = Sx[:, 0]
            Sx[0, 3] = Sx[0, 6] = 2.0
        for kind in ('nce', 'vcs', 'bce'):
            xg = Sx.float().cuda().requires_grad_()
            rg, vg, ig, _ = A30.hard_pair_rows(xg, ids.cuda(), kind); rgt, vgt, _, _ = A30.hard_pair_rows(xg.t(), ids.cuda(), kind)
            gg = torch.autograd.grad(rg.sum() + rgt.sum(), xg)[0]
            xc = Sx.float().double().requires_grad_()
            q1, w1 = REF.hard_pair_rows(xc, ids, kind, A30.TAU); q2, w2 = REF.hard_pair_rows(xc.t(), ids, kind, A30.TAU)
            gc = torch.autograd.grad(q1.sum() + q2.sum(), xc)[0]
            gsc = max(float(gc.abs().max()), 1e-30)
            rsc = max(float(q1.abs().max()), float(q2.abs().max()), 1.0)
            e = dict(rows=float(max((rg.double().cpu() - q1).abs().max(), (rgt.double().cpu() - q2).abs().max())) / rsc,
                     grad_rel=float((gg.double().cpu() - gc).abs().max()) / gsc,
                     valid_equal=bool(torch.equal(vg.cpu(), w1) and torch.equal(vgt.cpu(), w2)), finite=bool(torch.isfinite(gg).all()), n_invalid=int((~vg).sum()))
            if cname == 'multipos':
                e['tie_first'] = int(ig[0]) == 3
            e['ok'] = e['rows'] < 1e-5 and (e['grad_rel'] < 1e-4 or gsc == 1e-30) and e['valid_equal'] and e['finite'] and e.get('tie_first', True)
            if cname == 'noneg':
                e['ok'] &= e['n_invalid'] == 8 and float(gg.abs().max()) == 0.0
            good &= e['ok']; rr[f'{cname}_{kind}'] = e
    ok('rules_gpu', good, **rr)
    del isoB
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ l_boundary_bitwise (L source; native = R29 train l_listwise_vcs)
    d, capL, _, _, _ = boundary_bitwise('L', R29.train, 'L', 'l_listwise_vcs')
    ok('l_boundary_bitwise', d['bitwise'] and d['prob_equal'] and d['pred_equal'] and d['stats_equal'] and d['noisy_ran'] and d['fork_report_ok'], **d)

    # ------------------------------------------------------------------ filled-queue context (H0 smoke branch state)
    os_ = unit_opt(P60H, 'H', f'{G}/s', unit_id='r30_f60_s1001_H0', mode='resume', resume=f'{smoke}/runs/r30_f60_s1001_H0/state_last.pt')
    os_.vocab_size = capH['vocab_size']
    A, B = R.VSE(os_, w2i), R.VSE(os_, w2i)
    st = R27.load_full_state(f'{smoke}/runs/r30_f60_s1001_H0/state_last.pt', os_, A, B)     # mechanics (identity rule tested below)
    ep = st['epoch'] + 1
    R.adjust_learning_rate(os_, A.optimizer, ep); R.adjust_learning_rate(os_, B.optimizer, ep)
    prob_A, prob_B, _ = co_train.eval_train(os_, A, B, noisy_loader, data_size, copy.deepcopy(st['all_loss']), None, ep)
    pred_A = co_train.split_prob(prob_A, os_.p_threshold); pred_B = co_train.split_prob(prob_B, os_.p_threshold)
    for m, prob, pred in ((A, prob_A, pred_A), (B, prob_B, pred_B)):
        m.avg_clean_prob = prob[pred].mean()

    def snapshot():
        return copy.deepcopy(R.model_full_state(A)), copy.deepcopy(R.model_full_state(B)), R.rng_state()

    def restore(snap):
        for m, s in ((A, snap[0]), (B, snap[1])):
            m.img_enc.load_state_dict(s['img_enc']); m.txt_enc.load_state_dict(s['txt_enc'])
            m.optimizer.load_state_dict(copy.deepcopy(s['optimizer']))
            m.i2t_refiner_node.load_state_dict(s['i2t_refiner']); m.t2i_refiner_node.load_state_dict(s['t2i_refiner'])
            q = m.memory_queue; q.img_queue.copy_(s['queue']['img']); q.txt_queue.copy_(s['queue']['txt']); q.ptr, q.is_full = s['queue']['ptr'], s['queue']['is_full']
            I.load_ids(q, s['r27_queue_ids'])
            m.Eiters = s['Eiters']; m.avg_clean_prob = s['avg_clean_prob']
            for p in own(m):
                p.grad = None
        R.set_rng_state(snap[2])
        R27._PEER.clear(); R27._PEER[id(A)] = 0; R27._PEER[id(B)] = 1

    snap = snapshot()
    A30.main_term, R30.clip_grad_norm_ = main_wrap, clip_wrap
    isoF = {}
    for lab in COMBOS:
        restore(snap)
        lt, ut = loaders(pred_B, prob_B)
        isoF[lab] = one_step(os_, A, B, lt, ut, lab)
    good, d, gscale = iso_eval(isoF, 'filled')
    ok('isolation_filled', good, g_scale=gscale, tolerance_rel_to_total=2e-3, noisy_branch=isoF['H']['seen'], **d)
    scr = {}
    for lab in ('H', 'PV', 'N'):
        restore(snap)
        lt, ut = loaders(pred_B, prob_B, scramble=True)
        c = one_step(os_, A, B, lt, ut, lab)
        scr[lab] = dict(params_maxdiff=maxdiff(c['params'], isoF[lab]['params']), rng_equal=rng_eq(c['rng'], isoF[lab]['rng']), qB_equal=c['qB'] == isoF[lab]['qB'])
    # milestone rank pass (extra DEV eval, consumes torch RNG via the DataLoader base seed) + full RNG restore: next step unchanged
    restore(snap)
    r0 = R.rng_state()
    R30.dev_ranks(os_, [A, B])
    consumed = not rng_eq(r0, R.rng_state())
    R.set_rng_state(r0)
    lt, ut = loaders(pred_B, prob_B)
    c = one_step(os_, A, B, lt, ut, 'H')
    mri = dict(rng_consumed_before_restore=consumed, params_maxdiff=maxdiff(c['params'], isoF['H']['params']),
               rng_equal=rng_eq(c['rng'], isoF['H']['rng']), qB_equal=c['qB'] == isoF['H']['qB'])
    ok('milestone_rank_pass_isolated', mri['params_maxdiff'] == 0 and mri['rng_equal'] and mri['qB_equal'], **mri)
    A30.main_term, R30.clip_grad_norm_ = orig_main, orig_clip
    ok('truth_isolation', all(v['params_maxdiff'] == 0 and v['rng_equal'] and v['qB_equal'] for v in scr.values()),
       poisoned=['warmup dataset _t2i_index/_labels (raise)', 'labeled/unlabeled _t2i_index (raise)', 'eval_train truth argument None'],
       labels_scrambled=scr)
    tm = {}
    for lab in COMBOS:
        restore(snap)
        lt, ut = loaders(pred_B, prob_B)
        os_.r30_objective, os_.r30_scale = COMBOS[lab]; os_.r30_coef = COEF[lab]
        torch.cuda.synchronize(); t0 = time.time()
        R30.train(os_, A, B, lt, ut, 0, R.new_stats(), max_iters=20)
        torch.cuda.synchronize(); tm[lab] = (time.time() - t0) / 20 * 1000
    del A, B, snap, isoF
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ resume roundtrip (production R30 resume) + refusals
    rr, good = {}, True
    for uid, rpid, lab in (('r30_f60_s1001_PV', P60H, 'PV'), ('r30v11_f60_s1001_LN', P60L, 'N')):
        src = f'{smoke}/runs/{uid}/state_last.pt'
        o = unit_opt(rpid, lab, f'{G}/rr', unit_id=uid, mode='resume', resume=f'{G}/rr_{lab}.pt', epochs=6, milestone=99)
        o.vocab_size = capH['vocab_size']
        A1m, B1m = R.VSE(o, w2i), R.VSE(o, w2i)
        stx = R27.load_full_state(src, o, A1m, B1m)
        e = stx['epoch'] + 1

        def stp(Am, Bm, n=3):
            R.adjust_learning_rate(o, Am.optimizer, e); R.adjust_learning_rate(o, Bm.optimizer, e)
            l1, u1 = loaders(pred_B, prob_B)
            R30.train(o, Am, Bm, l1, u1, e, R.new_stats(), max_iters=n)
        stp(A1m, B1m)
        tmp = f'{G}/rr_{lab}.pt'
        R.save_full_state(tmp, o, A1m, B1m, e, 0.0, [[], []], {'warmup': [], 'cotrain': []})
        stp(A1m, B1m)
        X = state_record(A1m, B1m)
        del A1m, B1m
        A2m, B2m = R.VSE(o, w2i), R.VSE(o, w2i)
        R.load_full_state(tmp, o, A2m, B2m)                                                   # production resume path
        stp(A2m, B2m)
        Y = state_record(A2m, B2m)
        del A2m, B2m
        r = compare_records(X, Y)
        r['identity_refusals'] = dict(
            other_objective=_refused(tmp, unit_opt(rpid, 'PB', f'{G}/ref', unit_id=uid, mode='resume', resume=tmp, epochs=6, milestone=99)),
            other_unit=_refused(tmp, unit_opt(rpid, lab, f'{G}/ref', unit_id='r30_f60_s1001_PB', mode='resume', resume=tmp, epochs=6, milestone=99)),
            other_coef=_refused(tmp, unit_opt(rpid, lab, f'{G}/ref', unit_id=uid, mode='resume', resume=tmp, epochs=6, milestone=99, coef=(COEF[lab] or 1.0) * 1.01))
            if COMBOS[lab][1] == 'fit_static' else True,
            other_budget=_refused(tmp, unit_opt(rpid, lab, f'{G}/ref', unit_id=uid, mode='resume', resume=tmp, epochs=8, milestone=99)),
            other_parent=_refused(tmp, unit_opt(P60H if rpid == P60L else P60L, lab, f'{G}/ref', unit_id=uid, mode='resume', resume=tmp, epochs=6, milestone=99)))
        os.remove(tmp)
        r['ok'] = r['bitwise'] and all(r['identity_refusals'].values())
        good &= r['ok']; rr[lab] = r
        torch.cuda.empty_cache()
    pv = f'{smoke}/runs/r30_f60_s1001_PV/state_last.pt'
    formal_base = [x if x != '30' else '0' for x in base60(f'{G}/ref')]

    def fork_opt(pid, lab='PV', extra=None, sha_over=None, base_over=None):
        o_, sc_ = COMBOS[lab]
        a = ['--r30_stage', 'unit', '--r30_unit_id', 'gate_unit', '--r30_source_warmup', PREC[pid]['warmup_source'] if not extra else extra,
             '--r30_objective', o_, '--r30_scale', sc_, '--r30_mode', 'fork', '--r30_parent', PREC[pid]['state_path'],
             '--r30_parent_sha', sha_over or PREC[pid]['sha256'], '--r30_parent_id', pid, '--r30_milestone', '1']
        if sc_ == 'fit_static':
            a += ['--r30_coef', repr(COEF[lab])]
        os.makedirs(f'{G}/ref', exist_ok=True)
        o = r30_run.parse(a + (base_over or base60(f'{G}/ref')))
        o.vocab_size = capH['vocab_size']
        return o
    o_branch_parent = fork_opt(P60H); o_branch_parent.r30_parent = pv; o_branch_parent.r30_parent_sha = sha(pv)
    ref = dict(wrong_parent_sha=_refused(PREC[P60H]['state_path'], fork_opt(P60H, sha_over='0' * 64)),
               wrong_source=_refused(PREC[P60H]['state_path'], fork_opt(P60H, extra='L')),
               smoke_parent_for_formal_unit=_refused(PREC[P60H]['state_path'], fork_opt(P60H, base_over=formal_base)),
               other_cell_parent=_refused(PREC[P60H]['state_path'], fork_opt(P60H, base_over=base(N0, '0.0', f'{G}/ref'))),
               L_parent_for_H_source_unit=_refused(PREC[P60L]['state_path'], fork_opt(P60L, extra='H')),
               branch_state_as_parent=_refused(pv, o_branch_parent),
               parent_as_resume=_refused(PREC[P60H]['state_path'], unit_opt(P60H, 'PV', f'{G}/ref', unit_id='r30_f60_s1001_PV', mode='resume', resume=pv)),
               completed_branch_resume=_refused(pv, unit_opt(P60H, 'PV', f'{G}/ref', unit_id='r30_f60_s1001_PV', mode='resume', resume=pv)),
               parent_accepted=not _refused(PREC[P60H]['state_path'], fork_opt(P60H)))
    good &= all(ref.values())
    ok('resume_roundtrip', good, refusals=ref, **rr)

    # ------------------------------------------------------------------ calibration records (smoke)
    cr, good = {}, True
    for pid, c in CALJ.items():
        sums = {t: sum(b['sqnorm'][t] for b in c['batches']) for t in CAL.TERMS}
        co = CAL.coefficients(sums)
        e = dict(sums_equal=all(abs(sums[t] - c['sum_sqnorm'][t]) <= 1e-12 * max(abs(sums[t]), 1e-300) for t in CAL.TERMS),
                 coefs_equal=all((co[a]['c'] == c['coefficients'][a]['c']) for a in co),
                 ids_sha_ok=all(hashlib.sha256(np.asarray(b['ids'], dtype=np.int64).tobytes()).hexdigest() == b['ids_sha256'] for b in c['batches']),
                 n_pairs=c['n_gradient_pairs'], want_pairs=2 * c['n_batches_per_peer'], peers=sorted({b['peer'] for b in c['batches']}),
                 params_unchanged=c['params_unchanged'], grads_none=c['grads_none_before'] and c['grads_none_after'],
                 parent_unchanged=c['parent_unchanged'] and c['parent_fingerprint_before'] == c['parent_fingerprint_after'] == PREC[pid]['training_state_fingerprint'],
                 statuses={a: c['coefficients'][a]['status'] for a in c['coefficients']}, coefs={a: c['coefficients'][a]['c'] for a in c['coefficients']})
        e['ok'] = (e['sums_equal'] and e['coefs_equal'] and e['ids_sha_ok'] and e['n_pairs'] == e['want_pairs'] and e['peers'] == ['A', 'B']
                   and e['params_unchanged'] and e['grads_none'] and e['parent_unchanged'] and all(e['statuses'][a] == 'OK' for a in A30.CALIBRATED))
        good &= e['ok']; cr[pid] = e
    ok('calibration_records', good, **cr)

    # real dtype / AMP and actual cosine score range at the production main-term call (isolation steps, both contexts)
    tf = dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
              float32_matmul_precision=torch.get_float32_matmul_precision(), cudnn_benchmark=torch.backends.cudnn.benchmark)
    smin, smax = min(x['smin'] for x in SCORE), max(x['smax'] for x in SCORE)
    good = (len(SCORE) >= 2 * len(COMBOS) and not any(x['autocast'] for x in SCORE) and all(x['sims_dtype'] == x['loss_dtype'] == x['raw_dtype'] == 'torch.float32' for x in SCORE)
            and all(x['param_dtypes'] == ['torch.float32'] for x in SCORE) and -1 - 1e-5 <= smin and smax <= 1 + 1e-5 and not tf['matmul_allow_tf32'])
    ok('dtype_and_score_range', good, calls=len(SCORE), autocast_any=any(x['autocast'] for x in SCORE), score_min=smin, score_max=smax,
       implied_pair_margin_range=[(smin - smax) / A30.TAU, (smax - smin) / A30.TAU], backend_flags=tf, per_call=SCORE,
       note='float32, no autocast / GradScaler anywhere in the R30 path; scores are cosines of l2-normalized centered embeddings; '
            'cudnn TF32 flag is the torch default (affects cuDNN kernels), matmul TF32 off')
    same = {f: sha(os.path.join(CODE, f)) == sha(os.path.join(R29CODE, f)) for f in sorted(os.listdir(R29CODE)) if f.endswith('.py')}
    ok('host_unchanged', all(same.values()) and R.eval_train is co_train.eval_train and R30._ORIG_SAVE.__module__ == 'r25_co_train' and R.N is co_train,
       files_identical_to_code_r29=same)
    ok('timing_memory', True, ms_per_cotrain_step=tm, smoke_peaks={k: v.get('peak') for k, v in info.items() if k.startswith('r30')})

    root = os.path.dirname(os.path.abspath(CODE))
    im = json.load(open(os.path.join(CODE, 'INSTALL_MANIFEST.json')))
    code_ok = all(sha(os.path.join(CODE, f)) == hh for f, hh in im['files_sha256'].items()) and \
        sorted(im['files_sha256']) == sorted(f for f in os.listdir(CODE) if f.endswith('.py'))
    pf = os.path.join(root, 'PROTOCOL_FROZEN.json')
    pj = json.load(open(pf)) if os.path.isfile(pf) else {}
    frozen = pj.get('frozen') is True and pj.get('version') == 'R30-v1.1-merged-20261101'
    checks = [k for k, v in res.items() if isinstance(v, dict) and 'pass_' in v]
    af = os.path.join(root, 'PROTOCOL_AMENDMENT_A1.json')
    aj = json.load(open(af)) if os.path.isfile(af) else {}
    amend_ok = aj.get('frozen') is True and pj.get('amendment_a1_sha256') == (sha(af) if os.path.isfile(af) else None)
    out = dict(all_pass=all(res[k]['pass_'] for k in checks) and code_ok and frozen and amend_ok, config_frozen=frozen, code_matches_manifest=code_ok,
               protocol_version=pj.get('version'), protocol_frozen_sha256=sha(pf) if os.path.isfile(pf) else None, code_files_sha256=im['files_sha256'],
               amendment_a1_sha256=sha(af) if os.path.isfile(af) else None, amendment_a1_bound_in_protocol=amend_ok,
               install_manifest_sha256=sha(os.path.join(CODE, 'INSTALL_MANIFEST.json')), reference_sha256=sha(os.path.join(REFDIR, 'objective_checks.py')),
               job=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, smoke_dir=smoke,
               failed=[k for k in checks if not res[k]['pass_']], checks=res, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    json.dump(out, open(f'{smoke}/GPU_GATE.json', 'w'), indent=1, default=float)
    print('ALL_PASS', out['all_pass'], out['failed'], flush=True)
    sys.exit(0 if out['all_pass'] else 1)


def _refused(path, o):
    try:
        R30.check_loadable(path, o)
        return False
    except SystemExit:
        return True


if __name__ == '__main__':
    main()
