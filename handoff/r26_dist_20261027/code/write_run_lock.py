"""Write RUN_LOCK.json for R26 (code/patch/data/noise/cell/scale hashes and roles)."""
import json, hashlib, glob, os, datetime

D = '/home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027'
R25 = '/home/infres/yinwang/CS_QMI/handoff/r25_in2r_20261027'
PKG = '/home/infres/yinwang/CS_QMI/r26_retrieval_arch/VCS_Retrieval_R26_Architecture_20261027'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


pinned = dict(l.split()[::-1] for l in open(f'{D}/R25_CODE_PINNED_SHA256.txt') if len(l.split()) == 2)
code = {os.path.basename(f): sha(f) for f in sorted(glob.glob(f'{D}/code/*.py'))}
r26_new = {k: v for k, v in code.items() if k not in pinned}
changed = {k: (pinned[k], code[k]) for k in pinned if code.get(k) != pinned[k]}
noise = json.load(open(f'{R25}/NOISE_LOCK.json'))['files']
cells = json.load(open(f'{D}/CELLS.json'))
scales = {}
for ds in ('flickr', 'coco', 'cc152k'):
    p = f'{D}/SCALE_{ds}.json'
    if os.path.exists(p):
        s = json.load(open(p))
        scales[ds] = dict(sha256=sha(p), c_J=s['c_J'], prefix=s['prefix'], n_ratio_rows=s['n_ratio_rows'])
lock = dict(
    written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    design=dict(path=f'{D}/DESIGN_CN.md', sha256=sha(f'{D}/DESIGN_CN.md')),
    package=dict(zip='/home/infres/yinwang/CS_QMI/VCS_Retrieval_R26_Architecture_20261027.zip',
                 zip_sha256=sha('/home/infres/yinwang/CS_QMI/VCS_Retrieval_R26_Architecture_20261027.zip'),
                 spec_sha256=sha(f'{PKG}/architecture_spec.json')),
    base=dict(host='IN2R-code 909e53293ec3c633823597b8215ce23a71796d69 via R25 pinned copy', r25_code_pinned=pinned,
              r25_files_modified_in_r26=changed, r25_source_sha256_file=f'{R25}/SOURCE_SHA256.txt'),
    r26_code=r26_new,
    env='/projects/EEG-foundation-model/yinghao/envs/v9_sgr (torch 2.9.0+cu128), float32 no AMP, cudnn.benchmark=True (native)',
    data_root='/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data',
    noise_files={k: v for k, v in noise.items() if k in ('f30k_precomp_0.4', 'f30k_precomp_0.6', 'coco_precomp_0.4')},
    noise_note='R25 NOISE_LOCK (seed 20261027, full training pool, no clean anchors); CC152K real noise, none injected',
    cells=cells, scales=scales,
    roles=dict(train='train split only', select='full DEV (best-rsum epoch, native validate)', test='not read; entry tools/test_eval.py (R25) after lock'),
    arms=dict(d_vcs='full graph + VCS J', d_bce='same graph, balanced BCE', m_vcs='unnormalized mean point + VCS',
              n_vcs='normalized mean point + VCS', d_snce='same graph, marginal-aware soft NCE tau=0.05', H0='native IN2R-code (R25 runs for Flickr)'),
    gates_only='no result threshold gates any submission (user 「不要设置阈值让实验直接不启动」)')
json.dump(lock, open(f'{D}/RUN_LOCK.json', 'w'), indent=1)
print('RUN_LOCK written; r25 files modified:', list(changed))
