"""Write PROTOCOL_FROZEN_CR2.json (chmod 444) after acceptance ALL_PASS; formal students re-check it."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, time
from pathlib import Path
TREE = Path(__file__).resolve().parents[2]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr2/CR2_20261104')
CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104')
CR1T = Path('/home/infres/yinwang/CS_QMI/handoff/cr1_20261104')
NR1 = '/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'
CCN = '/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104'
CONDS = ('Flickr00', 'Flickr60', 'CC152K_RECOVERED_20261104')
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--acceptance', required=True); a = ap.parse_args()
    dst = TREE/'PROTOCOL_FROZEN_CR2.json'
    if dst.exists(): raise FileExistsError('frozen exists; register an amendment instead')
    acc = json.load(open(a.acceptance))
    if not acc['all_pass']: raise RuntimeError('acceptance not ALL_PASS')
    pkg = {p.name: sha256(p) for p in (TREE/'package'/'reference').glob('*.py')}
    ref = {p.name: sha256(p) for p in (TREE/'code'/'reference').glob('*.py')}
    if pkg != ref or sha256(TREE/'package'/'scripts'/'build_relation.py') != sha256(TREE/'code'/'site'/'build_relation_pkg.py'):
        raise RuntimeError('code/reference or build_relation_pkg.py is not a byte copy of the package')
    code = [p for p in (TREE/'code'/'reference').glob('*.py')] + [TREE/'code'/'site'/n for n in
            ('cr2_student.py', 'build_relation_pkg.py', 'build_oracle.py', 'audit_cr2.py')] + [TREE/'sbatch'/n for n in ('common_env.sh', 'student.sbatch')]
    code_sha = {str(p.relative_to(TREE)): sha256(p) for p in sorted(code)}
    for p in [CR1T/'code'/'site'/'cr1_prod.py', CR1T/'code'/'site'/'eval_student.py', *sorted((CR1T/'code'/'src'/'cr1').glob('*.py'))]:
        code_sha[str(p)] = sha256(p)
    rows = list(csv.DictReader(open(TREE/'package'/'configs'/'MATRIX.csv')))
    data = {}
    for c in CONDS:
        for p in (CR1/'banks'/c/'fit_bank.npz', CR1/'banks'/c/'fit_bank.binding.json', CR1/c/'prepared'/'PREP.json', CR1/c/'prepared'/'graph.npz'):
            data[str(p)] = sha256(p)
        for s in ('BASE', 'J', 'BCE', 'NCE'):
            for p in (CR1/c/'relations'/f'{s}_CAP.npz', CR1/c/'relations'/f'{s}_CAP.npz.json'): data[str(p)] = sha256(p)
    rel = {}
    for r in rows:
        u, c = r['unit_id'], r['condition']
        p = OUT/('oracle' if r['oracle'] == 'True' else 'relations')/c/f'{u}.npz'
        rc = json.load(open(str(p)+'.json'))
        if rc['relation_sha256'] != sha256(p) or rc['unit_id'] != u: raise RuntimeError(f'relation receipt mismatch {u}')
        data[str(p)] = sha256(p); data[str(p)+'.json'] = sha256(str(p)+'.json'); rel[u] = str(p)
    fz = {'protocol_id': 'CR2_ANCHOR_PRESERVATION_CONCENTRATION_V1', 'evidence_commit': 'bf3c4380897e2b4c67961711af4c4b2778b33045',
          'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'package_zip_sha256': sha256('/home/infres/yinwang/CS_QMI/VCS_CR1_Review_CR2_Server_20261104.zip'),
          'package_manifest_sha256': sha256(TREE/'package'/'PACKAGE_MANIFEST.json'),
          'code_sha256': code_sha, 'unit_config_sha256': {r['unit_id']: sha256(TREE/'package'/'configs'/'units'/f"{r['unit_id']}.json") for r in rows},
          'matrix_sha256': sha256(TREE/'package'/'configs'/'MATRIX.csv'), 'data_sha256': data, 'relation_paths': rel,
          'dev_eval_sha256': {'Flickr': {'dev_cache_manifest': sha256(f'{NR1}/cache/dev/manifest.json'), 'truth': sha256(f'{NR1}/evaluator_only/dev_positives.json')},
                              'CC152K_RECOVERED_20261104': {'dev_cache_manifest': sha256(f'{CCN}/cache/dev/manifest.json'), 'truth': sha256(f'{CCN}/evaluator_only/dev_positives.json')}},
          'acceptance': {'file': a.acceptance, 'sha256': sha256(a.acceptance), 'all_pass': True},
          'reuse': {'CR1_students': 21, 'CR1_frozen_protocol_sha256': sha256(CR1T/'PROTOCOL_FROZEN_CR1.json')},
          'student': 'CR1 train_student algorithm + CSR loader (code/site/cr2_student.py); CR1 eval_student bridge to NR1 evaluator',
          'new_estimator_fits': 0, 'new_student_fits': 32, 'lambda': 0.5, 'test_read': False, 'no_result_thresholds': True}
    dst.write_text(json.dumps(fz, indent=1)+'\n'); os.chmod(dst, 0o444); print(sha256(dst))
if __name__ == '__main__': main()
