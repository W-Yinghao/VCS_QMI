"""Check an R33 smoke (no TEST file is read).  check_smoke.py CELL SMOKE_DIR "ARMS" -> JSON on stdout, exit 0 iff pass."""
import os, sys, json, glob, hashlib, math, re
import numpy as np
import torch

D = '/home/infres/yinwang/CS_QMI/handoff/r33_baselines_20261102'
cell, S, arms = sys.argv[1], sys.argv[2], sys.argv[3].split()
C = json.load(open(f'{D}/CELLS.json'))[cell]
lock = json.load(open(f'{D}/NOISE_LOCK.json'))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
code = {k: {os.path.basename(p): sha(p) for p in glob.glob(f'{D}/{k}/*.py')} for k in ('code_h0', 'code_a1')}
SRC = {'code_h0': '/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/code', 'code_a1': '/home/infres/yinwang/CS_QMI/handoff/r27_a1cross_20261030/code_a1x'}
COMMON = dict(batch_size=128, embed_size=1024, lr_update=15, learning_rate=0.0005, queue_size=65536, p_threshold=0.5, grad_clip=2.0,
              margin=0.2, warmup_epoch=5, k=5, i2t=True, t2i=True, noisy_obj='sce', bi_gru=True, seed=C['seed'],
              data_name=C['data_name'], noise_ratio=C['noise_ratio'], noise_file=C['noise_file'])
out, ok = {'cell': cell, 'job': os.environ.get('SLURM_JOB_ID'), 'units': {}}, True
# code snapshots byte-identical to their sources
out['code_identical_to_source'] = {k: code[k] == {os.path.basename(p): sha(p) for p in glob.glob(f'{SRC[k]}/*.py')} for k in code}
ok &= all(out['code_identical_to_source'].values())
# noise file = lock, realized rate
nf = C['noise_file']; idx = np.load(nf)
out['noise'] = dict(file=nf, sha256=sha(nf), lock_sha256=C['noise_sha256'], realized_mismatch=float((idx != np.arange(len(idx)) // 5).mean()))
ok &= out['noise']['sha256'] == C['noise_sha256'] and abs(out['noise']['realized_mismatch'] - C['noise_ratio']) < 1e-3
# prefix
pd = f'{S}/prefix'; pe = json.load(open(f'{pd}/EPOCHS.json')); pr = json.load(open(glob.glob(f'{pd}/RUN_*.json')[0]))
sp = pe['cotrain'][0]['split_A']; tp = sp['precision'] * sp['n_pred_clean']
out['prefix'] = dict(warmup_epochs=len(pe['warmup']), cotrain=[x['epoch'] for x in pe['cotrain']], done=json.load(open(f'{pd}/PREFIX_DONE.json')),
                     state_last_removed=not os.path.exists(f'{pd}/state_last.pt'), code_sha_ok=pr['env']['code_sha256'] == code['code_h0'],
                     opt_mismatch={k: (pr['opt'].get(k), v) for k, v in COMMON.items() if pr['opt'].get(k) != v},
                     loader_clean_fraction=float(tp / sp['recall']) / len(idx) if sp['recall'] else None)
pc = [out['prefix']['warmup_epochs'] == 5, out['prefix']['cotrain'] == [0], out['prefix']['state_last_removed'],
      out['prefix']['code_sha_ok'], not out['prefix']['opt_mismatch'],
      out['prefix']['loader_clean_fraction'] is not None and abs(1 - out['prefix']['loader_clean_fraction'] - out['noise']['realized_mismatch']) < 1e-6]
out['prefix']['checks'] = pc; ok &= all(pc)
for arm in arms:
    d = f'{S}/{arm}'; r = {}; c = []
    runs = sorted(glob.glob(f'{d}/RUN_*.json')); run = json.load(open(runs[-1])); opt = run['opt']
    exp = dict(COMMON); exp.update(arm='a1_iface', lam=0.0) if arm == 'a1_iface' else None
    r['opt_mismatch'] = {k: (opt.get(k), v) for k, v in exp.items() if opt.get(k) != v}; c.append(not r['opt_mismatch'])
    r['gpu'] = run['env']['gpu']; r['code_sha_ok'] = run['env']['code_sha256'] == code['code_a1' if arm == 'a1_iface' else 'code_h0']; c.append(r['code_sha_ok'])
    pk = glob.glob(f'{d}/PEAK_*.json'); r['peak'] = json.load(open(pk[-1])) if pk else None
    e = json.load(open(f'{d}/EPOCHS.json')); co = e['cotrain']; e1 = json.load(open(f'{d}/EPOCHS_first.json'))
    r['cotrain_epochs'] = [x['epoch'] for x in co]; c.append(r['cotrain_epochs'] == [0, 1, 2, 3])
    r['prefix_epoch0_kept'] = co[0] == pe['cotrain'][0]; c.append(r['prefix_epoch0_kept'])
    r['first_run_records_kept'] = e1['cotrain'] == co[:3]; c.append(r['first_run_records_kept'])
    r['resumed_from'] = e.get('resumed_from'); c.append((r['resumed_from'] or {}).get('epoch') == 2 and r['resumed_from']['path'].endswith(f'{arm}/state_last.pt'))
    r['dev_rsum'] = [round(x['dev']['rsum'], 3) for x in co]; c.append(all(math.isfinite(x['dev']['rsum']) and x['dev']['rsum'] > 0 for x in co))
    att = [l.split('\t') for l in open(f'{d}/ATTEMPTS.tsv').read().strip().split('\n')]
    r['attempts'] = [a[-1] for a in att]; c.append(len(att) == 2 and att[0][-1].endswith('prefix/prefix_state.pt') and att[1][-1].endswith('state_last.pt'))
    done = json.load(open(f'{d}/DONE.json')); r['done'] = done; c.append(done['status'] == 'COMPLETE' and done['epochs_done'] == 4)
    if os.path.exists(f'{d}/model_best.pth.tar'):
        ck = torch.load(f'{d}/model_best.pth.tar', map_location='cpu', weights_only=False)
        r['model_best'] = dict(epoch=ck['epoch'], best_rsum=ck['best_rsum'])
        ent = [x for x in co if x['epoch'] == ck['epoch']]
        c.append(bool(ent) and abs(ent[-1]['dev']['rsum'] - ck['best_rsum']) < 1e-9 and abs(done['best_rsum'] - ck['best_rsum']) < 1e-9)
    else:
        r['model_best'] = 'absent (prefix epoch 0 stayed best within the smoke)'; c.append(max(x['dev']['rsum'] for x in co) == co[0]['dev']['rsum'])
    c.append(os.path.exists(f'{d}/state_last.pt'))
    tr = co[-1]['train']; r['rectification_ran'] = {p: any(k.startswith(('i2t_n', 't2i_n')) for k in tr[p]) for p in tr}; r['train_keys_peer0'] = sorted(tr[sorted(tr)[0]])[:60]
    if arm == 'H0': c.append(all(r['rectification_ran'].values()))  # A1 stats use other names; recorded only
    r['checks'] = c; r['pass'] = all(c); ok &= r['pass']; out['units'][arm] = r
out['pass'] = bool(ok)
print(json.dumps(out, indent=1, default=str))
sys.exit(0 if ok else 1)
