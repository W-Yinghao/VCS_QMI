"""PROTOCOL_FROZEN_CR3.json (chmod 444) after acceptance ALL_PASS; records GPU/env lock, graph hashes, reuse decisions."""
import argparse, csv, hashlib, json, os, time
from pathlib import Path
TREE = Path(__file__).resolve().parents[2]
CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104'); CR2 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr2/CR2_20261104')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104'); CR1T = Path('/home/infres/yinwang/CS_QMI/handoff/cr1_20261104')
NR1 = '/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104'; CCN = '/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gpath(c, r):
    if r in ('OBS', 'J_MASS_OBS'): return OUT/'graphs'/c/f'{r}.npz'
    if r == 'FULL_TRUE_ORACLE': return OUT/'oracle'/c/'FULL_TRUE_ORACLE.npz'
    if r == 'KEEP_ORACLE': return CR2/'oracle'/c/f'CR2_{c}_KEEP_ORACLE_s1001.npz'
    return CR2/'relations'/c/f"CR2_{c}_{r.replace('_KEEP_TOP1', '')}_KEEP_TOP1_s1001.npz"
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--acceptance', required=True); ap.add_argument('--receipts', required=True); a = ap.parse_args()
    dst = TREE/'PROTOCOL_FROZEN_CR3.json'
    if dst.exists(): raise FileExistsError('frozen exists')
    acc = json.load(open(a.acceptance)); assert acc['all_pass']
    for sub in ('reference', 'oracle', 'tools'):
        for p in (TREE/'package'/sub).glob('*.py'): assert sha(p) == sha(TREE/'code'/'pkg'/sub/p.name), p
    code = sorted([*(TREE/'code'/'pkg').rglob('*.py')] + [TREE/'code'/'site'/n for n in ('cr3_student.py', 'build_graphs_cr3.py', 'build_full_true_oracle.py')]
                  + [TREE/'sbatch'/n for n in ('common_env.sh', 'student.sbatch')])
    code_sha = {str(p.relative_to(TREE)): sha(p) for p in code if '__pycache__' not in str(p)}
    for p in [CR1T/'code'/'site'/'cr1_prod.py', CR1T/'code'/'site'/'eval_student.py', *sorted((CR1T/'code'/'src'/'cr1').glob('*.py'))]: code_sha[str(p)] = sha(p)
    rows = list(csv.DictReader(open(TREE/'package'/'configs'/'MATRIX.csv')))
    rec = json.load(open(a.receipts)); data = {}; plan = {}
    for c in {r['condition'] for r in rows}:
        for p in (CR1/'banks'/c/'fit_bank.npz', CR1/c/'prepared'/'PREP.json', CR1/c/'prepared'/'graph.npz'): data[str(p)] = sha(p)
    for r in rows:
        p = gpath(r['condition'], r['relation']); data[str(p)] = sha(p)
        if r['status'] == 'NEW': plan[r['unit_id']] = 'RUN_NEW'
        else:
            d = rec[f"{r['condition']}:{r['relation']}"]; plan[r['unit_id']] = 'REUSE' if d['decision'] == 'REUSE' else 'RERUN_ON_LOCK'
    g = acc['checks']['GPU_LOCK']
    fz = {'protocol_id': 'CR3_FIXED_GRAPH_SUPPORT_COMPLETE_V1', 'evidence_commit': 'e08cdf50b31e3cbaff0a7ece6b56d65a59bd621b',
          'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'package_zip_sha256': sha('/home/infres/yinwang/CS_QMI/VCS_CR2_Review_CR3_Server_20261104.zip'), 'package_manifest_sha256': sha(TREE/'package'/'PACKAGE_MANIFEST.json'),
          'code_sha256': code_sha, 'unit_config_sha256': {r['unit_id']: sha(TREE/'package'/'configs'/'units'/f"{r['unit_id']}.json") for r in rows},
          'data_sha256': data, 'graph_paths': {r['unit_id']: str(gpath(r['condition'], r['relation'])) for r in rows},
          'unit_plan': plan, 'reuse_receipts_sha256': sha(a.receipts),
          'gpu_env_lock': {'gpu_name': g['gpu'], 'torch': g['torch'], 'cuda': g['cuda'], 'cudnn': g['cudnn'], 'partition': 'RTX6000PRO', 'exclude': 'node51,node53,node60',
                           'determinism': g['determinism'], 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'},
          'dev_eval_sha256': {'Flickr': {'manifest': sha(f'{NR1}/cache/dev/manifest.json'), 'truth': sha(f'{NR1}/evaluator_only/dev_positives.json')},
                              'CC': {'manifest': sha(f'{CCN}/cache/dev/manifest.json'), 'truth': sha(f'{CCN}/evaluator_only/dev_positives.json')}},
          'acceptance': {'file': a.acceptance, 'sha256': sha(a.acceptance)}, 'new_estimator_fits': 0, 'test_read': False}
    dst.write_text(json.dumps(fz, indent=1)+'\n'); os.chmod(dst, 0o444); print(sha(dst)); print(json.dumps(plan, indent=0))
if __name__ == '__main__': main()
