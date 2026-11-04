"""R46 configs (main session relaying the user 「现在就开始设计R46吧，我同意你的设计，该提交的CPU，GPUgate smoke test现在就准备」). 30 units:
  A (12, tree r46o)  COCO 80% parent R33 H0 c80 s1001, FIT pool restricted to the fixed 29k / 58k image subsets (cond c80in_s1001_p29k /
     _p58k) x {N-frozen-shared-cosine, N-JOINT} x J/BCE/NCE, 8000 updates; config = the R43-A c80 s1001 config of the same cell with the cond
     swapped (DEV unchanged); no in-unit reference (the pool changes the record stream).
  B (18, trees r46f / r46o)  iterated N supervision, round 2: Flickr60 R30-H s1001 (R39 N-JOINT), Flickr80 IN2R s1001 (R41-C N-JOINT), COCO80
     IN2R s1001 (R43-A N-JOINT) x {R2_REBUILT, R2_CONTINUED} x J/BCE/NCE; each objective starts from its own round-1 step-8000 checkpoint
     (heads, nulls, student; fresh AdamW, the R39 JOINT cosine restarted over 8000 updates; sampler stream [1001, 46], new); REBUILT uses
     the round-2 top-32 tables of that objective's round-1 student (tools/r46_build_r2cand.py), CONTINUED the original parent tables.
     Step 0 must reproduce the round-1 step-8000 DEV ranks exactly.
Writes configs/R46_*.json, configs/UNITS_R46.json, RESOLVED_MATRIX_R46.csv. Usage: python -I -B tools/make_configs_r46.py (v9_sgr env)"""
import copy, csv, json, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'tools'))
import make_configs_r41 as M
from common import sha256, write_json, read_json
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
R2C = OUT / 'r46_parents' / 'r2cand'
QUOTE = '现在就开始设计R46吧，我同意你的设计，该提交的CPU，GPUgate smoke test现在就准备'
B_BASE = {'F60_R30_H1001': ('n60', 'r46f', M.R39 / 'configs' / 'R39_Flickr60_N_joint_global_{o}.json', M.R39R / 'runs' / 'R39_Flickr60_N_joint_global_{o}'),
          'F80_IN2R_H1001': ('f80in', 'r46f', W / 'configs' / 'R41_C_F80_IN2R_H1001_N_joint_{o}.json', OUT / 'runs_r41' / 'R41_C_F80_IN2R_H1001_N_joint_{o}'),
          'C80_IN2R_H1001': ('c80in_s1001', 'r46o', W / 'configs' / 'R43_A_C80_IN2R_H1001_N_joint_{o}.json', OUT / 'runs_r43' / 'R43_A_C80_IN2R_H1001_N_joint_{o}')}
units, rows = [], []


def put(cfg, tree, extra):
    cfg = {k: v for k, v in cfg.items() if k not in ('package_unit', 'package_block', 'job_bundle', 'matrix_source', 'extension', 'teacher_parents', 'r39_template')}
    cfg.update(code_tree=tree, runner=f'{tree}/train_unit.py', line='R46', oracle=False, test_read=False, label='development experiment (no TEST read in this line)',
               authorisation=dict(user_quote=QUOTE, relayed_by='main session'))
    write_json(W / 'configs' / f"{cfg['unit_id']}.json", cfg); units.append(cfg['unit_id'])
    rows.append(dict(unit_id=cfg['unit_id'], block=cfg['block'], tree=tree, cond=cfg['cond'], arm=extra, adaptation=cfg['adaptation'], objective=cfg['objective'], steps=cfg['steps'],
                     init_seed=cfg['init_seed'], sampler_seed=str(cfg['sampler_seed']), init_from=(cfg.get('init_from') or {}).get('unit_id'),
                     candidate_table=(cfg.get('candidate_file') or {}).get('sha256'), config_sha256=sha256(W / 'configs' / f"{cfg['unit_id']}.json")))


for o in M.OBJ:
    for n in (29, 58):
        for kind in ('frozen', 'joint'):
            b = read_json(W / 'configs' / f'R43_A_C80_IN2R_H1001_N_{kind}_{o}.json')
            cfg = dict(copy.deepcopy(b), unit_id=f'R46_A_C80_IN2R_H1001_P{n}K_N_{kind}_{o}', block='A', cond=f'c80in_s1001_p{n}k', reference_unit=None, reference_joint=None,
                       fit_pool=f'fixed {n},000-image subset of the COCO train images (seed 20261146, nested), records whose observed image is in the subset; natural top-32 / N labels / locks / calibration rebuilt within the pool; DEV unchanged',
                       reference_note='pool restriction changes the record stream: no in-unit reference')
            put(cfg, 'r46o', f'P{n}K')
    for pk, (cond, tree, cpath, udir) in B_BASE.items():
        base = read_json(Path(str(cpath).format(o=o))); u1 = Path(str(udir).format(o=o)); assert base['cond'] == cond and base['objective'] == o
        e8 = read_json(u1 / 'eval_step8000.json')
        inf = dict(unit_id=u1.name, ckpt=str(u1 / 'ckpt_step8000.pt'), ckpt_sha256=sha256(u1 / 'ckpt_step8000.pt'), step=8000, dev_ranks=str(u1 / 'dev_ranks_step8000.npz'),
                   dev_ranks_sha256=sha256(u1 / 'dev_ranks_step8000.npz'), eval_rsum=e8['dev_retrieval']['rsum'], round1_result_sha256=sha256(u1 / 'RESULT.json'),
                   round1_gpu=read_json(u1 / 'RESULT.json')['env']['gpu'])
        for arm in ('REBUILT', 'CONTINUED'):
            cfg = M.joint_fields(M.finish(copy.deepcopy(base), f'R46_B_{pk}_R2_{arm}_{o}', 'B', 'both_joint'), base, 'both_joint')
            cfg.update(sampler_seed=[1001, 46], reference_unit=None, reference_joint=None, init_from=inf, round2=arm,
                       seed_rule='round 2: head / null / student loaded from round 1 (init_seed unused for the weights); new sampler stream [1001, 46] for both arms',
                       head_init=f"round-1 head (step 8000 of {u1.name}), not re-initialised",
                       schedule_rule='R39 JOINT common cosine restarted over the 8000 round-2 updates (head 1e-4 -> 1e-5, student 1e-5 -> 1e-6), fresh AdamW',
                       reference_note='round 2: step 0 must reproduce the round-1 step-8000 DEV ranks exactly; REBUILT / CONTINUED share init and sampler seed')
            if arm == 'REBUILT':
                mf = read_json(R2C / f'R2CAND_MANIFEST_{cond}.json')['tables'][o]; assert mf['round1_unit'] == u1.name and mf['round1_ckpt_sha256'] == inf['ckpt_sha256']
                cfg['candidate_file'] = dict(path=mf['path'], sha256=mf['sha256'], rule=mf['rule'], coverage=mf['coverage'])
                cfg['supervision_rule'] = base['supervision_rule'] + ' | R46 REBUILT: natural top-32 from the round-1 student (same K / null rule), N labels from the observed relation'
            put(cfg, tree, f'R2_{arm}')
write_json(W / 'configs' / 'UNITS_R46.json', units)
with open(W / 'RESOLVED_MATRIX_R46.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('R46 configs', len(units), {b: sum(r['block'] == b for r in rows) for b in 'AB'})
