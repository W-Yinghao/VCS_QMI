"""PROTOCOL_FROZEN_CR5_P1105.json (chmod 444): package CR5 cells registered as CR5P1105 (date-suffixed; distinct from the running CR5)."""
import csv, hashlib, json, os, time
from pathlib import Path
TREE = Path(__file__).resolve().parents[2]; sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
CR1 = Path('/projects/EEG-foundation-model/yinghao/outputs_cr1/CR1_20261104'); CR1T = Path('/home/infres/yinwang/CS_QMI/handoff/cr1_20261104')
CR4T = Path('/home/infres/yinwang/CS_QMI/handoff/cr4_20261104'); CR5T = Path('/home/infres/yinwang/CS_QMI/handoff/cr5_20261105')
dst = TREE/'PROTOCOL_FROZEN_CR5_P1105.json'
if dst.exists(): raise FileExistsError(dst)
fz4 = json.load(open(CR4T/'PROTOCOL_FROZEN_CR4.json')); units = {}; data = {}; mapping = {}
for r in csv.DictReader(open(TREE/'package'/'configs'/'MATRIX.csv')):
    if r['family'] != 'CR5_FIXED_GRAPH_CONFIRM': continue
    cfg = json.load(open(TREE/'package'/'configs'/'units'/f"{r['unit_id']}.json"))
    gp = next(i['graph_path'] for i in fz4['units'].values() if i['condition'] == r['condition'] and i['relation'] == r['relation'])
    assert sha(gp) == fz4['data_sha256'][gp]; data[gp] = sha(gp)
    u = f"CR5P1105_{r['condition']}_{r['relation']}_COMPLETE_PC_t{r['student_seed']}"; mapping[r['unit_id']] = u
    units[u] = dict(package_unit_id=r['unit_id'], package_config_sha256=sha(TREE/'package'/'configs'/'units'/f"{r['unit_id']}.json"), condition=r['condition'], relation=r['relation'],
                    student_seed=int(r['student_seed']), sampler_seed=int(r['sampler_seed']), graph_path=gp, objective=cfg['objective'],
                    result_dir=f"/projects/EEG-foundation-model/yinghao/outputs_cr5p/CR5P1105/students/{r['condition']}/{u}")
assert len(units) == 24
for c in {u['condition'] for u in units.values()}:
    for p in (CR1/'banks'/c/'fit_bank.npz', CR1/c/'prepared'/'PREP.json', CR1/c/'prepared'/'graph.npz'): data[str(p)] = sha(p)
cs = {str(p.relative_to(TREE)): sha(p) for p in [TREE/'code'/'site'/'cr5p_group.py', TREE/'sbatch'/'common_env.sh', TREE/'sbatch'/'group_cr5p.sbatch']}
for p in [CR5T/'code'/'site'/'cr5_student.py', CR4T/'code'/'site'/'cr4_student.py', *(CR4T/'code'/'pkg'/'reference').glob('*.py'), CR1T/'code'/'site'/'cr1_prod.py', CR1T/'code'/'site'/'eval_student.py', *sorted((CR1T/'code'/'src'/'cr1').glob('*.py'))]: cs[str(p)] = sha(p)
acc = CR5T/'acceptance'/'ACCEPTANCE_CR5_1089910.json'
fz = {'protocol_id': 'CR5_P1105_FIXED_GRAPH_STUDENT_SEED_PC', 'package': {'zip_sha256': sha('/home/infres/yinwang/CS_QMI/VCS_NR5_CR4_Action_Server_20261105.zip'), 'evidence_commit': 'e397159953d83626a4068ba2da72a47de799c2f3',
      'user_instruction': '组织这一批的实验'}, 'frozen_utc': time.strftime('%FT%TZ', time.gmtime()),
      'dedup': 'package CR5 overlaps but is not byte-equivalent to the running CR5 (RECT+PC, seeds 1002/1003, sampler 20261104202/3); registered under suffix CR5P1105; nothing reused',
      'unit_id_mapping': mapping, 'units': units, 'code_sha256': cs, 'data_sha256': data,
      'trainer': 'cr5_student.train(consumer=COMPLETE_PC): CR4 PC step; new G from condition common a,b, head seed = student seed, zero last residual; AdamW 1e-4->1e-5 cosine, wd 1e-4, clip 2; 8000 steps',
      'trainer_acceptance': {'file': str(acc), 'sha256': sha(acc), 'note': 'CR5 acceptance R2: seed-1001 PC path equals CR4 trainer bitwise; R3/R4 seeds and resume'},
      'gpu_policy': 'one job per condition x student seed on one GPU (L40S, RTX6000PRO, H100; excl. node51,node53,node60)', 'test_read': False, 'new_estimators': 0, 'new_graphs': 0}
dst.write_text(json.dumps(fz, indent=1)+'\n'); os.chmod(dst, 0o444); (TREE/'UNIT_ID_MAPPING.json').write_text(json.dumps(mapping, indent=1)); print(sha(dst))
