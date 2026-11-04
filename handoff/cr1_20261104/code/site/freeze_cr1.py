"""Write PROTOCOL_FROZEN_CR1.json (chmod 444). Formal units re-check every hash before running."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104')
CONDS = ('Flickr00', 'Flickr60', 'CC152K_RECOVERED_20261104')
NR1 = '/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'
CCN = '/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--acceptance', required=True); a = ap.parse_args()
    dst = TREE/'PROTOCOL_FROZEN_CR1.json'
    if dst.exists(): raise FileExistsError('Frozen protocol exists; register an amendment instead')
    acc = json.load(open(a.acceptance))
    if not acc['all_pass']: raise RuntimeError('Acceptance not ALL_PASS')
    code = sorted([p for p in (TREE/'code'/'src'/'cr1').glob('*.py')] +
                  [TREE/'code'/'site'/n for n in ('cr1_prod.py', 'eval_student.py', 'graph_diag.py')] +
                  [TREE/'sbatch'/n for n in ('common_env.sh', 'est.sbatch', 'score.sbatch', 'match.sbatch', 'student.sbatch')])
    pkg_src = {p.name: sha256(p) for p in (TREE/'package'/'src'/'cr1').glob('*.py')}
    prod_src = {p.name: sha256(p) for p in (TREE/'code'/'src'/'cr1').glob('*.py')}
    if pkg_src != prod_src: raise RuntimeError('code/src/cr1 is not a byte copy of package src/cr1')
    with open(TREE/'package'/'configs'/'MATRIX.csv') as f: units = [r['unit_id'] for r in csv.DictReader(f)]
    data = {}
    for c in CONDS:
        for p in (OUT/'banks'/c/'fit_bank.npz', OUT/'banks'/c/'fit_bank.binding.json', OUT/c/'prepared'/'PREP.json',
                  OUT/c/'prepared'/'graph.npz', *sorted((OUT/c/'prepared').glob('allowed_E*.npy'))):
            data[str(p)] = sha256(p)
    dev = {'Flickr': {'dev_cache_manifest': sha256(f'{NR1}/cache/dev/manifest.json'), 'truth': sha256(f'{NR1}/evaluator_only/dev_positives.json')},
           'CC152K_RECOVERED_20261104': {'dev_cache_manifest': sha256(f'{CCN}/cache/dev/manifest.json'),
                                         'truth': sha256(f'{CCN}/evaluator_only/dev_positives.json')}}
    fz = {'protocol_id': 'CR1_PRODUCTQ_CROSSFIT_PARTIAL_RELATION_V1', 'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'package_zip_sha256': sha256('/home/infres/yinwang/CS_QMI/VCS_CR1_Server_20261104.zip'),
          'package_manifest_sha256': sha256(TREE/'package'/'PACKAGE_MANIFEST.json'),
          'user_instruction': '/home/infres/yinwang/CS_QMI/VCS_CR1_Server_20261104.zip准备这个包，并行进行',
          'tree_note': 'tree name handoff/cr1_20261104 set by the main session (package suggested handoff/cr1_correspondence_20261104)',
          'code_sha256': {str(p.relative_to(TREE)): sha256(p) for p in code},
          'unit_config_sha256': {u: sha256(TREE/'package'/'configs'/'units'/f'{u}.json') for u in units},
          'matrices_sha256': {n: sha256(TREE/'package'/'configs'/n) for n in ('MATRIX.csv', 'ESTIMATOR_MATRIX.csv', 'TASK_MATRIX.csv', 'GRAPH_MATRIX.csv', 'PLAN.json')},
          'data_sha256': data, 'dev_eval_sha256': dev,
          'dev_evaluator': 'NR1 nr1.evaluate (NR1_BASE_TOP100_ALL_QUERY_V1, float64) via site/eval_student.py bridge',
          'acceptance': {'file': a.acceptance, 'sha256': sha256(a.acceptance), 'all_pass': True},
          'model': {'id': 'google/siglip2-base-patch16-224', 'revision': '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2', 'encoder': 'frozen'},
          'formal_units': {'estimators': 54, 'students': 21, 'score_exports': 12, 'graph_solves': 18},
          'endpoints': {'estimator': 'step 8000 (no DEV-best)', 'student': 'step 8000 primary; 0/250/500/1000/2000/4000 archived'},
          'test_read': False, 'no_result_thresholds': True}
    dst.write_text(json.dumps(fz, indent=1, ensure_ascii=False)+'\n'); os.chmod(dst, 0o444)
    print(sha256(dst))


if __name__ == '__main__': main()
