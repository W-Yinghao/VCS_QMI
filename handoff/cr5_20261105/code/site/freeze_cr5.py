"""PROTOCOL_FROZEN_CR5.json (chmod 444)."""
import csv, hashlib, json, os, sys, time
from pathlib import Path
TREE = Path(__file__).resolve().parents[2]; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104'); CR1T = Path('/home/infres/yinwang/CS_QMI/handoff/cr1_20261104')
CR3T = Path('/home/infres/yinwang/CS_QMI/handoff/cr3_20261104'); CR4T = Path('/home/infres/yinwang/CS_QMI/handoff/cr4_20261104')
acc = sys.argv[1]; A = json.load(open(acc)); assert A['all_pass']; dst = TREE/'PROTOCOL_FROZEN_CR5.json'
if dst.exists(): raise FileExistsError(dst)
fz4 = json.load(open(CR4T/'PROTOCOL_FROZEN_CR4.json')); units = {}; data = {}
for r in csv.DictReader(open(TREE/'configs'/'MATRIX.csv')):
    gp = next(i['graph_path'] for i in fz4['units'].values() if i['condition'] == r['condition'] and i['relation'] == r['relation'])
    assert sha(gp) == fz4['data_sha256'][gp]; data[gp] = sha(gp)
    units[r['unit_id']] = dict(condition=r['condition'], relation=r['relation'], consumer=r['consumer'], head_seed=int(r['head_seed']), sampler_seed=int(r['sampler_seed']), plan=r['plan'], graph_path=gp,
                               result_dir=f"/projects/EEG-foundation-model/yinghao/outputs_cr5/CR5_20261105/students/{r['condition']}/{r['unit_id']}")
for c in {u['condition'] for u in units.values()}:
    for p in (CR1/'banks'/c/'fit_bank.npz', CR1/c/'prepared'/'PREP.json', CR1/c/'prepared'/'graph.npz'): data[str(p)] = sha(p)
code = [TREE/'code'/'site'/n for n in ('cr5_student.py', 'cr5_group.py')] + [TREE/'sbatch'/n for n in ('common_env.sh', 'group_cr5.sbatch')]
cs = {str(p.relative_to(TREE)): sha(p) for p in code}
for p in [CR4T/'code'/'site'/'cr4_student.py', *(CR4T/'code'/'pkg'/'reference').glob('*.py'), CR3T/'code'/'site'/'cr3_student.py', CR1T/'code'/'site'/'cr1_prod.py', CR1T/'code'/'site'/'eval_student.py', *sorted((CR1T/'code'/'src'/'cr1').glob('*.py'))]: cs[str(p)] = sha(p)
fz = {'protocol_id': 'CR5_SEED_CONFIRM_RECT_VS_COMPLETE_PC_V1', 'frozen_utc': time.strftime('%FT%TZ', time.gmtime()),
      'authorisation': {'user_quote': '下来这两个你都自己来分析分析结果，设计新一轮的方向，不要让GPU空着', 'designed_by': 'main session',
                        'reason': 'CR4 COMPLETE-PC beat COMPLETE-CE in every non-identity cell and on Flickr60 also beat RECT and OBS-RECT; CC ~RECT; all single seed -> seed confirmation'},
      'matrix': '{Flickr60, CC152K_RECOVERED_20261104} x {OBS, J_MASS_OBS, BASE/J/BCE/NCE_KEEP_TOP1} x {RECT, COMPLETE_PC} x seeds {1002, 1003} = 48; oracle graphs excluded',
      'seeds': {'1002': {'head_seed': 1002, 'sampler_seed': 20261104202}, '1003': {'head_seed': 1003, 'sampler_seed': 20261104203}, 'rule': 'sampler = 20261104201 + (seed - 1001); s1001 = CR4 (PC) and CR4 refs_rerun (RECT)'},
      'code_sha256': cs, 'unit_config_sha256': {u: sha(TREE/'configs'/'units'/f'{u}.json') for u in units}, 'units': units, 'data_sha256': data,
      'gpu_policy': 'one job per condition x seed on one GPU (CR3_A1/CR4_A1); L40S,RTX6000PRO,H100 excl. node51,node53,node60', 'acceptance': {'file': acc, 'sha256': sha(acc), 'gpu': A['checks']['GPU_RECORD']['gpu']},
      'new_graphs': 0, 'new_estimators': 0, 'encoder_training': False, 'test_read': False}
dst.write_text(json.dumps(fz, indent=1)+'\n'); os.chmod(dst, 0o444); print(sha(dst))
