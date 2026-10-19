"""Single-unit execution of the frozen N2F recipe; scheduling amendment only."""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'v9_full_20261017'))
from common import ROOT, frozen, oracle_pool, read, sha, verify_input, write
from split_common import BUNDLE, unit_path, verify_control, verify_identity, check_old_absent
import fcntl


def verify_completed(path):
    result = read(path / 'full_result.json')
    if result['identity']['freeze_sha256'] != sha(ROOT / 'FROZEN.json'):
        raise ValueError('Completed unit belongs to a different freeze')
    for name, digest in result['artifact_sha256'].items():
        if sha(path / name) != digest:
            raise ValueError(f'Completed artifact changed: {path/name}')


def main(group_id, unit_index):
    m = frozen(); groups = read(ROOT / 'GROUPS.json')
    group = next(g for g in groups if g['id'] == group_id)
    group = {**group, 'units': [group['units'][unit_index]]}
    out = ROOT / 'results' / group_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'DONE.json').exists():
        d = read(out / 'DONE.json')
        if d['freeze_sha256'] != sha(ROOT / 'FROZEN.json'):
            raise ValueError('Group COMPLETE belongs to a different freeze')
        for item in group['units']:
            verify_completed(out / f"{item['method']}_{item['scale']}")
        print('already complete', group_id, flush=True); return
    sys.path.insert(0, str(ROOT / 'source' / 'src'))
    import numpy as np
    import torch
    from v9.n2.data import eval_arrays, load_precomp, load_vocab
    from v9.n2.fixtures import collate_pairs
    from v9.n2.teacher import load_host_lock
    from v9.n2.train import WeightedStudent
    from functools import partial
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('A full training job requires its allocated GPU')
    seed = group['seed']; base = Path(m['old_output']) / 'rho0.4' / f's{seed}'
    obs_path = base / 'pairs.observed_pairs.json'; verify_input(m, obs_path)
    rows = read(obs_path)
    vocab_path = Path(m['vocab']); verify_input(m, vocab_path)
    vocab = load_vocab(vocab_path)
    cfg = load_host_lock(ROOT / 'source/HOST_LOCK.yaml', 'f30k_precomp', len(vocab))
    if cfg.to_verify or cfg.num_epochs != 40 or cfg.warmup_epochs != 5:
        raise ValueError(f'Unfrozen recipe: {cfg}')
    cache = ROOT / 'source/token_cache'
    for p in cache.iterdir():
        if p.is_file() and p.name not in m['cache_names']:
            raise ValueError('Unregistered token cache')
    # No train/dev files may silently change while the long queue is waiting.
    for p, info in m['data_files'].items():
        st = Path(p).stat()
        if st.st_size != info['size'] or st.st_mtime_ns != info['mtime_ns']:
            raise ValueError(f'Data identity changed: {p}')
    train = load_precomp(m['data'], 'train', vocab, cache_dir=cache)
    dev = load_precomp(m['data'], 'dev', vocab, cache_dir=cache)
    if train['n_images'] != 29000 or len(rows) != 145000 or dev['n_images'] != 1014:
        raise ValueError('This is not the registered full dataset')
    dev_images, dev_captions = eval_arrays(dev)
    collate = partial(collate_pairs, train['images'], train['captions'])
    hardware = {'host': platform.node(), 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'python': platform.python_version(), 'job_id': os.environ.get('SLURM_JOB_ID'),
                'deterministic_algorithms': True, 'tf32': False}
    write(BUNDLE / 'units' / group_id / str(unit_index) / ('hardware_' + os.environ['SLURM_JOB_ID'] + '.json'), hardware)
    units = []
    for item in group['units']:
        method, scale = item['method'], item['scale']
        ud = out / f'{method}_{scale}'
        identity = {'group': group_id, 'seed': seed, 'method': method, 'scale': scale, 'epochs': 40,
                    'freeze_sha256': sha(ROOT / 'FROZEN.json'), 'protocol_sha256': m['code_sha256']['PROTOCOL_CN.md'],
                    'oracle': method.startswith('Oracle'), 'selection': 'best full DEV rSum; earliest tie',
                    'dataset': 'f30k full train', 'design_rho': 0.4, 'learning_rate_multiplier': 1.0}
        if (ud / 'identity.json').exists() and read(ud / 'identity.json') != identity:
            raise ValueError(f'Unit identity mismatch: {ud}')
        write(ud / 'identity.json', identity)
        if (ud / 'full_result.json').exists():
            verify_identity(ud, group_id, item)
            verify_completed(ud)
            units.append(str(ud / 'full_result.json'))
            print('SKIP verified complete', method, scale, flush=True)
            continue
        if (ud / 'done.json').exists():
            done = read(ud / 'done.json')
            if done['manifest_sha256'] != sha(ud / 'run_manifest.json'):
                raise ValueError('Stale done marker: run manifest changed')
            for name in ('best.pt', 'last.pt', 'train_log.json'):
                if not (ud/name).is_file() or (ud/name).stat().st_size == 0:
                    raise ValueError(f'Stale done marker: missing {name}')
        if method.startswith('Oracle'):
            lp = base / 'eval_only/pairs.eval_only.json'; verify_input(m, lp)
            pool, weights = oracle_pool(rows, read(lp), method == 'OracleDrop')
            lock = {'sha256': m['input_sha256'][str(lp)]}
        else:
            wp = base / 'weights' / f'{method}.csv'; verify_input(m, wp)
            verify_input(m, str(wp) + '.lock.json')
            lock = read(str(wp) + '.lock.json')
            if lock['sha256'] != m['input_sha256'][str(wp)]:
                raise ValueError('Production weight lock differs from frozen input')
            with wp.open() as f:
                weights = {r['pair_id']: float(r['w']) for r in csv.DictReader(f)}
            pool = rows
        if len(weights) != sum(r['role'] == 'U' for r in pool):
            raise ValueError('Incomplete weight coverage')
        print('START', group_id, method, scale, 'n_pool', len(pool), hardware, flush=True)
        t0 = time.time()
        prior_manifest_sha = sha(ud / 'run_manifest.json') if (ud / 'last.pt').exists() else None
        student = WeightedStudent(cfg, pool, weights, collate, dev_images, dev_captions, ud, seed, 1.0,
                                  device='cuda', weight_lock=lock, formal=False, loss_scale=scale)
        if prior_manifest_sha is not None and sha(ud / 'run_manifest.json') != prior_manifest_sha:
            raise ValueError('Resume recipe differs from the saved checkpoint manifest')
        done = student.fit(40)
        if not np.isfinite(done['best']['rsum']):
            raise ValueError('Nonfinite validation result')
        log = read(ud / 'train_log.json')
        if len(log['epochs']) != 40 or log['epochs'][-1]['epoch'] != 39:
            raise ValueError('Incomplete full training cannot be marked COMPLETE')
        result = {'identity': identity, 'done': done, 'hardware': hardware, 'elapsed_this_attempt_s': time.time()-t0,
                  'n_pool': len(pool), 'n_U': len(weights), 'full_epochs': 40, 'test_read': False,
                  'artifact_sha256': {n:sha(ud/n) for n in ('best.pt','last.pt','train_log.json','run_manifest.json','identity.json')}}
        write(ud / 'full_result.json', result)
        units.append(str(ud / 'full_result.json'))
        del student; gc.collect(); torch.cuda.empty_cache()
        print('COMPLETE', method, scale, done, flush=True)
    write(BUNDLE / 'units' / group_id / str(unit_index) / 'DONE.json',
          {'group': group_id, 'unit_index': unit_index, 'freeze_sha256': sha(ROOT / 'FROZEN.json'),
           'job_id': os.environ['SLURM_JOB_ID'], 'units': units})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', required=True)
    ap.add_argument('--unit-index', required=True, type=int)
    a = ap.parse_args()
    try:
        cfg = verify_control()
        if a.group not in cfg['split_groups'] or a.unit_index not in (0, 1):
            raise ValueError('Unregistered split unit')
        check_old_absent(cfg['split_groups'][a.group]['old_job'])
        ud = unit_path(a.group, a.unit_index)
        ud.mkdir(parents=True, exist_ok=True)
        with (ud / 'RUN.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            main(a.group, a.unit_index)
    except Exception:
        write(BUNDLE / 'units' / a.group / str(a.unit_index) / ('FAILED_' + os.environ.get('SLURM_JOB_ID', 'unknown') + '.json'),
              {'group': a.group, 'unit_index': a.unit_index, 'utc': time.strftime('%FT%TZ', time.gmtime()),
               'job_id': os.environ.get('SLURM_JOB_ID'), 'traceback': traceback.format_exc()})
        raise
