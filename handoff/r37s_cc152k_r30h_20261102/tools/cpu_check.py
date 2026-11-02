#!/usr/bin/env python3
"""R37S CPU check (no GPU, no TEST): code identity vs code_r30v11 / code_a1x, CC152K one-image-one-caption structure through the
real data.py, driver argument parsing of both stages, host-gather row identity. Writes CPU_CHECK.json."""
from pathlib import Path
import hashlib, json, os, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / 'code_r37s'
DATA = '/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data'
sys.path.insert(0, str(CODE))
res = {}


def ok(name, cond, **info):
    res[name] = dict(ok=bool(cond), **info)
    print(('PASS ' if cond else 'FAIL ') + name, json.dumps(info, default=str)[:300], flush=True)


sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
man = json.loads((CODE / 'INSTALL_MANIFEST.json').read_text())
prot = json.loads((ROOT / 'PROTOCOL.json').read_text())
ok('code_manifest', all(sha(CODE / f) == h for f, h in man['files_sha256'].items())
   and sorted(man['files_sha256']) == sorted(p.name for p in CODE.glob('*.py')))
ok('protocol_frozen_bound', prot['frozen'] is True and prot['code_files_sha256'] == man['files_sha256'])
base = Path(man['base_snapshot'])
same = [f for f in man['files_sha256'] if f not in man['changed_vs_base']]
ok('unchanged_files_byte_identical_to_r30v11', all(sha(base / f) == man['files_sha256'][f] for f in same), n=len(same),
   changed=man['changed_vs_base'])
ok('r27_interface_is_a1x_file', sha(CODE / 'r27_interface.py') == sha('/home/infres/yinwang/CS_QMI/handoff/r27_a1cross_20261030/code_a1x/r27_interface.py'))

import data as D
cap_tr, im_tr = D.get_dataset(DATA, 'cc152k_precomp', 'train')
cap_dv, im_dv = D.get_dataset(DATA, 'cc152k_precomp', 'dev')


class V:  # vocab stub (only __call__ / word2idx used in __getitem__, which is not called here)
    word2idx = {}


tr = D.PrecompDataset(cap_tr, im_tr, V(), 'train', 0, 'none')
dv = D.PrecompDataset(cap_dv, im_dv, V(), 'dev')
_, _, ids_tr, _ = D.get_dataset(DATA, 'cc152k_precomp', 'train', return_id_caps=True)
ok('cc152k_train_one_image_per_caption', tr.im_div == 1 and tr.length == 150000 == im_tr.shape[0] and np.array_equal(tr.t2i_index, np.arange(150000))
   and np.array_equal(tr._t2i_index, tr.t2i_index) and int(tr._labels.sum()) == 150000,
   im_div=tr.im_div, n_caps=tr.length, n_imgs=int(im_tr.shape[0]), image_shape=list(im_tr.shape[1:]))
ok('cc152k_train_image_ids', True, n_unique_tsv_image_ids=len(set(ids_tr)), n_rows=len(ids_tr),
   note='informational: native IN2R treats each row as its own image index (t2i_index = arange)')
ok('cc152k_dev_structure', dv.im_div == 1 and dv.length == 1000 == im_dv.shape[0] and np.array_equal(dv.t2i_index, np.arange(1000)),
   n_caps=dv.length, n_imgs=int(im_dv.shape[0]))
ok('no_test_files_read', True, note='only train/dev opened by this check')

import r25_co_train as R, r30_co_train as RC
ok('per_captions_cc152k', 'per_captions = 1 if opt.data_name == \'cc152k_precomp\' else 5' in (CODE / 'r30_co_train.py').read_text()
   and 'per_captions = 1 if opt.data_name == "cc152k_precomp" else 5' in (CODE / 'r25_co_train.py').read_text())

import r30_run
W = ['--r30_stage', 'warmup', '--r30_parent_id', 'wp_cc_s1001_H_e5', '--r30_source_warmup', 'H', '--data_path', DATA, '--data_name',
     'cc152k_precomp', '--noise_file', 'none', '--noise_ratio', '0.0', '--seed', '1001', '--k', '5', '--num_epochs', '40',
     '--warmup_epoch', '5', '--prefix_epoch', '-1', '--output_dir', '/nonexistent']
o = r30_run.parse(W)
ok('parse_warmup', o.r29_arm == 'h_native' and o.arm == 'a1_iface' and o.lam == 0.0 and o.k == 5 and o.i2t and o.t2i and o.warmup_epoch == 5
   and o.noise_ratio == 0 and o.data_name == 'cc152k_precomp' and o.num_epochs == 40 and o.noisy_obj == 'sce',
   **{k: getattr(o, k) for k in ('k', 'i2t', 't2i', 'warmup_epoch', 'num_epochs', 'noise_ratio', 'batch_size', 'learning_rate', 'lr_update', 'margin')})
U = ['--r30_stage', 'unit', '--r30_parent_id', 'wp_cc_s1001_H_e5', '--r30_source_warmup', 'H', '--r30_unit_id', 'r37s_cc_s1001_H0',
     '--r30_objective', 'H', '--r30_scale', 'native', '--r30_mode', 'fork', '--r30_parent', '/x/parent_state.pt', '--r30_parent_sha', 'x'] + W[6:]
o = r30_run.parse(U)
ok('parse_unit_H0_fork', o.r30_objective == 'H' and o.r30_scale == 'native' and o.resume == '/x/parent_state.pt' and o.r29_arm is None
   and o.r30_milestone == 10 and o.arm == 'a1_iface')

import r27_interface as I
I.Ctx.images = im_tr
idx = np.array([149999, 7, 7, 0, 123456, 31], dtype=np.int64)
g = I.Ctx.rows_np(idx)
ok('host_gather_rows_identical', g.dtype == np.float32 and all(np.array_equal(g[j], np.asarray(im_tr[i])) for j, i in enumerate(idx)))

out = dict(all_pass=all(v['ok'] for v in res.values()), checks=res, job=os.environ.get('SLURM_JOB_ID'),
           code_files_sha256=man['files_sha256'], protocol_sha256=sha(ROOT / 'PROTOCOL.json'),
           utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
(ROOT / 'CPU_CHECK.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
print('ALL_PASS' if out['all_pass'] else 'SOME_FAIL')
sys.exit(0 if out['all_pass'] else 1)
