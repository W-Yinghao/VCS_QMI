"""PROTOCOL_FROZEN_CR4.json (chmod 444) + SOURCE_GRAPH_HASHES.json + REFERENCE_RECEIPTS_FROZEN.json."""
import csv, hashlib, json, os, sys, time
from pathlib import Path
TREE = Path(__file__).resolve().parents[2]; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104'); CR3T = Path('/home/infres/yinwang/CS_QMI/handoff/cr3_20261104')
CR3S = Path('/projects/EEG-foundation-model/yinghao/outputs_cr3/CR3_20261104/students_A1'); OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_cr4/CR4_20261104')
CR1T = Path('/home/infres/yinwang/CS_QMI/handoff/cr1_20261104')
acc = sys.argv[1]; A = json.load(open(acc)); assert A['all_pass']
dst = TREE/'PROTOCOL_FROZEN_CR4.json'
if dst.exists(): raise FileExistsError(dst)
for sub in ('reference', 'tools'):
    for p in (TREE/'package'/sub).glob('*.py'): assert sha(p) == sha(TREE/'code'/'pkg'/sub/p.name), p
assert sha(TREE/'code'/'pkg'/'reference'/'support_complete.py') == sha(CR3T/'code'/'pkg'/'reference'/'support_complete.py')
fz3 = json.load(open(CR3T/'PROTOCOL_FROZEN_CR3.json'))
rows = list(csv.DictReader(open(TREE/'package'/'configs'/'MATRIX.csv'))); units = {}; data = {}; graphs = {}
for r in rows:
    c, rel = r['condition'], r['relation']; cr3u = f'CR3_{c}_{rel}_COMPLETE_s1001'; gp = fz3['graph_paths'][cr3u]
    units[r['unit_id']] = dict(condition=c, relation=rel, graph_path=gp, plan=r['plan'], oracle=r['oracle'] == 'True', result_dir=str(OUT/'students'/c/r['unit_id']))
    data[gp] = sha(gp); graphs[f'{c}:{rel}'] = dict(path=gp, sha256=data[gp], cr3_frozen_sha256=fz3['data_sha256'][gp])
    assert data[gp] == fz3['data_sha256'][gp]
for c in {r['condition'] for r in rows}:
    for p in (CR1/'banks'/c/'fit_bank.npz', CR1/c/'prepared'/'PREP.json', CR1/c/'prepared'/'graph.npz'): data[str(p)] = sha(p)
refs = {}
for r in csv.DictReader(open(TREE/'package'/'configs'/'REFERENCES.csv')):
    d = CR3S/r['condition']/r['unit_id']; m = json.load(open(d/'eval_008000'/'METRICS.json'))
    refs[r['unit_id']] = dict(dir=str(d), step8000_sha256=sha(d/'step_008000.pt'), step250_sha256=sha(d/'step_000250.pt'), rSum_recorded=m['metrics']['rSum'],
                              rSum_package=float(r['rSum']), rSum_match=abs(m['metrics']['rSum']-float(r['rSum'])) < 1e-9, group_env=json.load(open(CR3S/r['condition']/'GROUP_ENV.json')))
assert all(v['rSum_match'] for v in refs.values())
code = sorted([p for p in (TREE/'code'/'pkg').rglob('*.py') if '__pycache__' not in str(p)] + [TREE/'code'/'site'/n for n in ('cr4_student.py', 'cr4_group.py', 'audit_cr4.py')]
              + [TREE/'sbatch'/n for n in ('common_env.sh', 'group_cr4.sbatch')])
cs = {str(p.relative_to(TREE)): sha(p) for p in code}
for p in [CR3T/'code'/'site'/'cr3_student.py', CR1T/'code'/'site'/'cr1_prod.py', CR1T/'code'/'site'/'eval_student.py', *sorted((CR1T/'code'/'src'/'cr1').glob('*.py'))]: cs[str(p)] = sha(p)
fz = {'protocol_id': 'CR4_FIXED_GRAPH_POSITIVE_EXCLUSIVE_V1', 'evidence_commit': 'e156e2f1ccfa354785cd3b2450408dba0038aea9', 'frozen_utc': time.strftime('%FT%TZ', time.gmtime()),
      'package_zip_sha256': sha('/home/infres/yinwang/CS_QMI/VCS_CR3_Review_CR4_Server_20261104.zip'), 'package_manifest_sha256': sha(TREE/'package'/'PACKAGE_MANIFEST.json'),
      'code_sha256': cs, 'unit_config_sha256': {u: sha(TREE/'package'/'configs'/'units'/f'{u}.json') for u in units}, 'units': units, 'data_sha256': data,
      'gpu_policy': 'CR3_A1 principle: one Slurm job per condition, all its cells on that job GPU; partition L40S,RTX6000PRO,H100 (exclude node51,node53,node60); '
                    'CR3 references reused only if the group env equals the CR3 group env and 250-step CE and RECT replays are bitwise, else rerun in the group',
      'acceptance': {'file': acc, 'sha256': sha(acc), 'gpu': A['checks']['GPU_RECORD']['gpu']}, 'new_estimator_fits': 0, 'new_graphs': 0, 'encoder_training': False, 'test_read': False}
dst.write_text(json.dumps(fz, indent=1)+'\n'); os.chmod(dst, 0o444)
(TREE/'SOURCE_GRAPH_HASHES.json').write_text(json.dumps(graphs, indent=1)); (TREE/'REFERENCE_RECEIPTS_FROZEN.json').write_text(json.dumps(refs, indent=1))
print(sha(dst))
