"""Check the R31 smoke outputs (no TEST file is read). check_smoke.py flickr|cc JOBID -> SMOKE_<which>_<job>.json"""
import os, sys, json, glob, hashlib, math, re
import numpy as np
import torch

D = '/home/infres/yinwang/CS_QMI/handoff/r31_in2r_fidelity_20261101'
O = '/projects/EEG-foundation-model/yinghao/outputs_v9/r31_in2r_fidelity_20261101/smoke'
R25CODE = '/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027/code'
OFF02 = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/noise_index/f30k_precomp_0.2.npy'
R25_04 = '/projects/EEG-foundation-model/yinghao/outputs_v9/r25_in2r_20261027/noise/f30k_precomp_0.4.npy'
SHA = {'official_f30k_precomp_0.2': 'cdd8b019e69dc1341316fb089a1d9e2c32ca14a62cdd48d72c6af49ab0120f50',
       'r25_f30k_precomp_0.4': '4926687360301d9052d348c2a75776915e4827bfb2d9c06dcf728dfa704bf739'}
which, job = sys.argv[1], sys.argv[2]
log = open(glob.glob(f'{D}/logs/*_{job}.out')[0]).read()
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
r25 = {os.path.basename(p): sha(p) for p in glob.glob(R25CODE + '/*.py')}
UNITS = {'flickr': [('f20_s1001', 'H0_codecfg'), ('f40_s1001', 'H0_default'), ('f40_s1001', 'H0_default_i2t_t2i')],
         'cc': [('cc_s1001', 'H0_default'), ('cc_s1001', 'H0_default_i2t_t2i')]}[which]
EXP = {'H0_codecfg': dict(warmup_epoch=5, k=5, i2t=True, t2i=True),
       'H0_default': dict(warmup_epoch=1, k=1, i2t=False, t2i=False),
       'H0_default_i2t_t2i': dict(warmup_epoch=1, k=1, i2t=True, t2i=True)}
COMMON = dict(batch_size=128, embed_size=1024, lr_update=15, learning_rate=0.0005, queue_size=65536, p_threshold=0.5,
              grad_clip=2.0, margin=0.2, workers=0, noisy_obj='sce', bi_gru=True, seed=1001)
out, ok = {'job': job, 'units': {}}, True
for cell, arm in UNITS:
    d = f'{O}/{cell}/{arm}'; r = {}; c = []
    runs = sorted(glob.glob(f'{d}/RUN_{job}*.json'))
    run = json.load(open(f'{d}/RUN_{job}.json')); opt = run['opt']
    exp = dict(COMMON, **EXP[arm])
    if cell.startswith('f20'):
        exp.update(data_name='f30k_precomp', noise_ratio=0.2, noise_file=OFF02)
    elif cell.startswith('f40'):
        exp.update(data_name='f30k_precomp', noise_ratio=0.4, noise_file=R25_04)
    else:
        exp.update(data_name='cc152k_precomp', noise_ratio=0, noise_file='none')
    r['opt_mismatch'] = {k: (opt.get(k), v) for k, v in exp.items() if opt.get(k) != v}
    c.append(not r['opt_mismatch'])
    r['opt_used'] = {k: opt[k] for k in sorted(opt) if k not in ('data_path',)}
    r['gpu'] = run['env']['gpu']
    r['code_identical_to_r25'] = run['env']['code_sha256'] == r25; c.append(r['code_identical_to_r25'])
    if exp['noise_ratio']:
        nf = exp['noise_file']; idx = np.load(nf)
        r['noise_file_sha256'] = sha(nf); r['realized_mismatch'] = float((idx != np.arange(len(idx)) // 5).mean())
        c.append(r['noise_file_sha256'] in SHA.values()); c.append(abs(r['realized_mismatch'] - exp['noise_ratio']) < 1e-3)
        r['loader_loaded_noise_file'] = f'=> load noisy index from {nf}' in log; c.append(r['loader_loaded_noise_file'])
    m = re.findall(r'load (\w+) / (\w+) data: (\d+) images, (\d+) captions', log)
    r['data_lines_in_log'] = sorted(set(m))
    e = json.load(open(f'{d}/EPOCHS.json')); co = e['cotrain']
    n_co = 2 if (which == 'flickr' and arm == 'H0_codecfg') else 1
    r['warmup_epochs'] = len(e['warmup']); c.append(len(e['warmup']) == exp['warmup_epoch'])
    r['cotrain_epochs'] = [x['epoch'] for x in co]; c.append(r['cotrain_epochs'] == list(range(n_co)))
    r['dev'] = [{k: x['dev'][k] for k in ('rsum', 'n_images', 'n_captions')} for x in co]
    c.append(all(math.isfinite(x['dev']['rsum']) and x['dev']['rsum'] > 0 for x in co))
    tr = co[-1]['train']
    r['train_stats_last'] = {p: {k: v for k, v in tr[p].items() if k.split('_')[0] in ('i2t', 't2i', 'skip', 'steps')} for p in tr}
    for p in tr:
        has = any(k.startswith(('i2t_n', 't2i_n', 'i2t_no_proto', 't2i_no_proto')) for k in tr[p])
        c.append(has == exp['i2t'])  # rectification branches ran iff i2t/t2i on
    done = json.load(open(f'{d}/DONE.json')); r['done'] = done; c.append(done['status'] == 'COMPLETE')
    ck = torch.load(f'{d}/model_best.pth.tar', map_location='cpu', weights_only=False)
    r['model_best'] = dict(epoch=ck['epoch'], best_rsum=ck['best_rsum'], keys=sorted(ck))
    ent = [x for x in co if x['epoch'] == ck['epoch']]
    c.append(bool(ent) and abs(ent[-1]['dev']['rsum'] - ck['best_rsum']) < 1e-9 and abs(done['best_rsum'] - ck['best_rsum']) < 1e-9)
    c.append(os.path.exists(f'{d}/state_last.pt'))
    if n_co == 2:
        att = [l.split('\t') for l in open(f'{d}/ATTEMPTS.tsv').read().strip().split('\n')]
        e0 = json.load(open(f'{d}/EPOCHS_epoch0.json'))
        r['resume'] = dict(attempts=len(att), second_from=att[-1][-1], resumed_from=e.get('resumed_from'),
                           epoch0_record_kept=e0['cotrain'][0] == co[0], run_jsons=[os.path.basename(p) for p in runs])
        c.append(len(att) == 2 and att[-1][-1].startswith('--resume') and e.get('resumed_from', {}).get('epoch') == 0
                 and e0['cotrain'][0] == co[0])
    r['checks'] = c; r['pass'] = all(c); ok &= r['pass']
    out['units'][f'{cell}/{arm}'] = r
out['pass'] = bool(ok)
json.dump(out, open(f'{D}/SMOKE_{which}_{job}.json', 'w'), indent=1, default=str)
print('SMOKE', which, 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
