"""R38 parent binding (CPU, read-only on parent dirs, caches and preps) -> PARENT_BINDING.json: the 4 frozen parents of R38
(Flickr60 R30-H s1001 = the R36 parent; CC152K R31 H0_default s1001 = the R37C parent; CC152K R31 H0_default s1002/s1003 = new),
model_best sha256, logged best DEV, cache/prep manifests + candidate/trust/calibration hashes. Usage: python -I -B tools/bind_parents.py"""
import glob, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r38c'))
from common import CONDS as CC, sha256, read_json, write_json, utc

R36B = read_json('/home/infres/yinwang/CS_QMI/handoff/r36_supervision_20261102/PARENT_BINDING.json')


def cc(k):
    c = CC[k]; P = Path(c['parent_dir']); cdir = Path(c['cache_dir']); pdir = Path(c['prep_dir'])
    run = read_json(P / c['run_json']); done = read_json(P / 'DONE.json'); ep = read_json(P / 'EPOCHS.json')
    best = max(ep['cotrain'], key=lambda e: e['dev']['rsum'])
    att = [l.rstrip('\n').split('\t') for l in open(P / 'ATTEMPTS.tsv')]
    man = read_json(cdir / 'CACHE_MANIFEST.json'); pm = read_json(pdir / 'PREP_MANIFEST.json')
    return dict(parent_unit=c['h_unit'], dataset='CC152K', noise='real (uninjected), observed relation = native identity', parent_seed=c['parent_seed'],
                parent_dir=str(P), model_best_sha256=sha256(P / 'model_best.pth.tar'), run_json=str(P / c['run_json']), run_json_sha256=sha256(P / c['run_json']),
                recipe={x: run['opt'][x] for x in ('warmup_epoch', 'k', 'i2t', 't2i', 'num_epochs', 'lr_update', 'learning_rate', 'batch_size', 'noise_ratio', 'noisy_obj', 'seed')},
                status=done['status'], epochs_done=done['epochs_done'], best_rsum_logged=done['best_rsum'], best_epoch=best['epoch'], best_epoch_dev=best['dev'],
                train_attempts=[dict(attempt=a[0], utc=a[1], job=a[2], host=a[4], gpu=a[5]) for a in att],
                selection='parent model_best = native DEV-best epoch (R31 rule); not re-selected by any R38 result',
                cache_dir=str(cdir), cache_manifest_sha256=sha256(cdir / 'CACHE_MANIFEST.json'), native_dev_reproduced=man['native_dev']['rsum'],
                native_dev_exact=man['native_dev']['reproduces_checkpoint_best_rsum_exactly'], cache_gpu=man['env']['gpu'],
                prep_dir=str(pdir), prep_manifest_sha256=sha256(pdir / 'PREP_MANIFEST.json'), prep_files={n: r['sha256'] for n, r in pm['files'].items()},
                a0b0={x: pm['calibration_init'][c['prep_key']][x] for x in ('a0', 'b0')}, candidates=pm['candidates'][c['prep_key']],
                trust={x: pm['trust'][c['prep_key']][x] for x in ('n_subset', 'retention')},
                origin='R37C cache + prep (read-only reuse)' if k == 'cc' else 'R38 cache + prep (R37C rules unchanged; locks byte-equal to R37C)')


R37B = read_json('/home/infres/yinwang/CS_QMI/handoff/r37_confirm_20261102/PARENT_BINDING.json')


def r37(k):
    c = R37B['conditions'][k]
    cm = f'/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/cache/{k}/CACHE_MANIFEST.json'
    return dict(c, origin='R37 parent: R37 cache + prep (read-only reuse), binding copied from R37 PARENT_BINDING.json', cache_manifest_sha256_now=sha256(cm),
                prep_manifest_sha256_now=sha256('/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/prep/PREP_MANIFEST.json'))


def main():
    n60 = dict(R36B['conditions']['n60'], origin='R36 parent: R34 cache + R35 prep (read-only reuse), binding copied from R36 PARENT_BINDING.json',
               cache_manifest_sha256_now=sha256('/projects/EEG-foundation-model/yinghao/outputs_v9/r34_direct_critic_20261102/cache/n60/CACHE_MANIFEST.json'),
               prep_manifest_sha256_now=sha256('/projects/EEG-foundation-model/yinghao/outputs_v9/r35_expanded_20261102/prep/PREP_MANIFEST.json'))
    write_json(W / 'PARENT_BINDING.json', dict(created_utc=utc(), line='R38; development experiment, no TEST read',
               parents={'f60_R30_s1001': n60, 'f60_R30_s1002': r37('n60s1002'), 'f60_R30_s1003': r37('n60s1003'),
                        'cc_R31_s1001': cc('cc'), 'cc_R31_s1002': cc('cc_s1002'), 'cc_R31_s1003': cc('cc_s1003')},
               amendment='f60_R30_s1002/s1003 added by the user amendment (completeness), R37 parents reused read-only',
               series_note='Flickr60 R30-H and CC152K R31 H0_default are different parent series; gains reported relative to the same parent only, no absolute cross-dataset comparison. The CC152K R30-style parent (R37S, 1082196) is a separate line and not used here.',
               missing=[]))
    print('PARENT_BINDING written')


if __name__ == '__main__':
    main()
