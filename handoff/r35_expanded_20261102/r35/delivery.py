"""R35 first-delivery records (login/CPU; hashing + JSON only): STATE_BINDING.json, SOURCE_HASHES.json,
OLD_TASKS_AND_DEDUP.json. Usage: python -B delivery.py"""
from __future__ import annotations
import csv, glob, json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, OUT, PREP, PACKAGE, R34_ROOT, R34_OUT, R30_RUNS, CODE_R30, CONDS, sha256, write_json, read_json, utc


def main():
    sb = dict(created_utc=utc(), line='R35 EXPANDED (VCS_R35_Expanded_Server_20261102)', package_base_commit='ea5513f92cc9e52e500487961ffe380d199af26a', conditions={})
    for cond, c in CONDS.items():
        hdir = R30_RUNS / c['h_unit']; man = read_json(R34_OUT / 'cache' / cond / 'CACHE_MANIFEST.json')
        cache_ok = {n: sha256(R34_OUT / 'cache' / cond / n) == r['sha256'] for n, r in man['files'].items()}
        sb['conditions'][cond] = dict(
            parent=dict(unit=c['h_unit'], model_best=str(hdir / 'model_best.pth.tar'), model_best_sha256=sha256(hdir / 'model_best.pth.tar'),
                        bound_in_R34_cache=man['parent']['model_best_sha256'], epoch=man['parent']['epoch'], best_rsum=man['parent']['best_rsum'],
                        code_dir=str(CODE_R30), code_sha256=man['parent']['code_sha256']),
            observed_map=dict(path=c['noise_file'], sha256=sha256(c['noise_file']), expected=c['noise_sha256'], realized_mismatch=man['observed_map']['realized_mismatch']),
            r34_cache=dict(dir=str(R34_OUT / 'cache' / cond), manifest_sha256=sha256(R34_OUT / 'cache' / cond / 'CACHE_MANIFEST.json'),
                           all_files_hash_verified=all(cache_ok.values()), n_files=len(cache_ok), native_dev_rsum=man['native_dev']['rsum'],
                           native_reproduces_checkpoint=man['native_dev']['reproduces_checkpoint_best_rsum_exactly'],
                           used=['fit_img/cap_peer{A,B}.npy raw native unit vectors (NOT the R34 FIT standardisation)', 'dev_img/cap_peer{A,B}.npy', 'native_dev_sims.npy']),
            r34_locks=dict(path=str(R34_OUT / 'locks' / 'LOCKS.json'), dev_observed_map_sha256=sha256(read_json(R34_OUT / 'locks' / 'LOCKS.json')['dev_observed_maps'][cond]['path'])))
        sb['conditions'][cond]['parent']['matches_R34_binding'] = sb['conditions'][cond]['parent']['model_best_sha256'] == man['parent']['model_best_sha256']
    if (PREP / 'PREP_MANIFEST.json').exists():
        pm = read_json(PREP / 'PREP_MANIFEST.json')
        sb['r35_prep'] = dict(manifest=str(PREP / 'PREP_MANIFEST.json'), manifest_sha256=sha256(PREP / 'PREP_MANIFEST.json'), files=pm['files'],
                              calibration_init=pm['calibration_init'], trust_n60=pm.get('trust_n60'), candidates=pm['candidates'])
    write_json(ROOT / 'STATE_BINDING.json', sb)

    pkg_man = read_json(PACKAGE / 'MANIFEST_SHA256.json')
    entries = pkg_man.get('files', pkg_man)
    pk = {}
    if isinstance(entries, dict):
        for k, v in entries.items():
            h = v if isinstance(v, str) else v.get('sha256')
            if (PACKAGE / k).exists():
                pk[k] = dict(expected=h, actual=sha256(PACKAGE / k), ok=sha256(PACKAGE / k) == h)
    sh = dict(created_utc=utc(), package=dict(dir=str(PACKAGE), manifest_entries_verified=pk, all_ok=all(v['ok'] for v in pk.values()) if pk else None),
              r35_code={Path(f).name: sha256(f) for f in sorted(glob.glob(str(ROOT / 'r35' / '*.py')))},
              r35_tests={Path(f).name: sha256(f) for f in sorted(glob.glob(str(ROOT / 'tests' / '*.py')))},
              r35_sbatch={Path(f).name: sha256(f) for f in sorted(glob.glob(str(ROOT / 'sbatch' / '*.sbatch')))},
              reused_from_R34=dict(native_eval_py_verbatim=dict(r34=sha256(R34_ROOT / 'r34' / 'native_eval.py'), r35=sha256(ROOT / 'r35' / 'native_eval.py')),
                                   critics_py_read_only_for_6_1=sha256(R34_ROOT / 'r34' / 'critics.py')),
              r30_code_used={f: sha256(CODE_R30 / f) for f in ('vse.py', 'data.py', 'evaluation.py', 'co_train.py', 'vocab.py', 'glove_shim.py')},
              stage_script=sha256('/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/sbatch/stage.sh'))
    write_json(ROOT / 'SOURCE_HASHES.json', sh)

    q = subprocess.run(['squeue', '-u', 'yinwang', '-h', '-o', '%i|%j|%T|%q|%P|%M|%y'], capture_output=True, text=True).stdout.strip().splitlines()
    jobs = [dict(zip(('id', 'name', 'state', 'qos', 'partition', 'time', 'nice'), l.split('|'))) for l in q]
    r34_units = read_json(R34_ROOT / 'configs' / 'UNITS.json')
    r35 = list(csv.DictReader(open(ROOT / 'MATRIX_RESOLVED.csv')))
    r34_keys = {(read_json(R34_ROOT / 'configs' / f'{u}.json')['structure'], read_json(R34_ROOT / 'configs' / f'{u}.json')['objective'],
                 read_json(R34_ROOT / 'configs' / f'{u}.json')['cond'], read_json(R34_ROOT / 'configs' / f'{u}.json')['steps']) for u in r34_units}
    r35_keys = {(r['structure_resolved'], r['objective'], r['cond'], int(r['steps'])) for r in r35}
    od = dict(created_utc=utc(),
              live_queue=dict(r31=[j for j in jobs if j['name'].startswith('r31')], r33=[j for j in jobs if j['name'].startswith('r33')],
                              r35=[j for j in jobs if j['name'].startswith('r35')], other=[j for j in jobs if not j['name'].startswith(('r31', 'r33', 'r35'))]),
              r34=dict(units=r34_units, all_done=all((R34_OUT / 'runs' / u / 'DONE.json').exists() for u in r34_units),
                       structures='D dual-cosine 256 proj / X joint MLP, random init, R34 FIT standardisation, shared critic, 2000 steps, lr1e-3'),
              dedup=dict(r35_unique_unit_ids=len({r['unit_id'] for r in r35}) == 42, overlap_structure_objective_cond_steps_with_R34=sorted(map(str, r34_keys & r35_keys)),
                         note='R35 structures ID/G/L start from the native H score (common calibration), per-peer heads, raw native features, 8000/4000 steps, lr1e-4 wd1e-4 clip2: no R34 unit repeated; no R32 simplex-ratio grid; R30-R34 files only read'),
              r35_existing_dirs_before_first_run=sorted(p.name for p in (OUT / 'runs').glob('r35_*')) if (OUT / 'runs').exists() else [])
    write_json(ROOT / 'OLD_TASKS_AND_DEDUP.json', od)
    print('delivery records written')


if __name__ == '__main__':
    main()
