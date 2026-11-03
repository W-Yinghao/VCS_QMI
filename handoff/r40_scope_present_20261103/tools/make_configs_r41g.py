"""R41-G (amendment to R41; same user authorisation): 9 units on the r41g tree (= r41f + R41-G freeze file):
  G1 Flickr 0% R30-H f00_s1001 N x both_joint x J/BCE/NCE, 8000 updates, training seed 2002 (head init 2002, sampler [2002, 3]) as block B (3);
  G2 ORACLE_MISSING (R39 rule, isolated process) x {frozen, both_joint} x J/BCE/NCE on Flickr60 R30-H s1003 (6), as block D on s1002;
     references: R38 N-frozen s1003 (frozen: init/chain), R38 + R39X N-JOINT s1003 (joint).
Writes configs/R41_G*.json, configs/UNITS_R41G.json, RESOLVED_MATRIX_R41G.csv. Usage: python -I -B tools/make_configs_r41g.py (v9_sgr env)"""
import csv, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'tools'))
import make_configs_r41 as M
from common import sha256, write_json, read_json
AM = dict(name='R41-G', user_quote='你上次的R39X的提议就很好，R40结束的时候你也思考一下，我们还需要补充哪些实验，只要你觉得有用的就提交，不用问我', relayed_by='main session')
units, rows = [], []
def put(cfg, refs):
    cfg.update(code_tree='r41g', runner=cfg['runner'].replace('r41f/', 'r41g/'))
    write_json(W / 'configs' / f"{cfg['unit_id']}.json", cfg); units.append(cfg['unit_id'])
    rows.append(dict(unit_id=cfg['unit_id'], block=cfg['block'], cond=cfg['cond'], supervision=cfg['supervision'], adaptation=cfg['adaptation'], objective=cfg['objective'],
                     steps=cfg['steps'], init_seed=cfg['init_seed'], sampler_seed=str(cfg['sampler_seed']), oracle=cfg['oracle'], runner=cfg['runner'], refs=refs,
                     config_sha256=sha256(W / 'configs' / f"{cfg['unit_id']}.json")))
t39 = {o: read_json(M.R39 / 'configs' / f'R39_Flickr60_N_joint_global_{o}.json') for o in M.OBJ}
for o in M.OBJ:
    jdir = M.R39R / M.PAR['F00_R30_H1001'][1].format(o=o); base = read_json(M.R39 / 'configs' / f'{jdir.name}.json')
    cfg = M.joint_fields(M.finish(base, f'R41_G1_F00_R30_H1001_seed2002_N_joint_{o}', 'G1', 'both_joint'), t39[o], 'both_joint')
    cfg.update(init_seed=2002, sampler_seed=[2002, 3], reference_unit=None, seed_rule='R41-G1: training seed 2002 (head init 2002, sampler [2002, 3]); parent, prep, recipe unchanged',
               head_init='fresh R35 build_heads(G, a0, b0, 2002): FIT-only calibration a0*s+b0 + zero-init residual; null slots init 0',
               amendment=dict(AM, reason='replicate the Flickr 0% NCE-JOINT 551.1 outlier and give a second 0% J/BCE point (training-seed replicate)'))
    put(cfg, f'compare_with={jdir.name}')
    oc = read_json(M.R39 / 'configs' / f'R39_Flickr60_ORACLE_MISSING_frozen_{o}.json'); fid = f'r38_B_f60_R30_s1003_N_shared_{o}_cosine'
    f38 = read_json(M.R38C / f'{fid}.json')
    ofr = M.finish(f38, f'R41_G2_F60_R30_H1003_ORACLE_frozen_{o}', 'G2', 'frozen', oracle=True)
    ofr.update(supervision='ORACLE_MISSING', supervision_label='ORACLE_MISSING', supervision_rule=oc['supervision_rule'] + ' (R41-G2: parent R30-H s1003; bit-identical R39 oracle_truth.py)',
               joint_rule=None, reference_unit=M.r38ref(fid), amendment=dict(AM, reason='second-parent replication of the ORACLE-JOINT gap (with block D on s1002)'))
    put(ofr, f'n_frozen={fid}')
    jdir = M.R39R / M.PAR['F60_R30_H1003'][1].format(o=o); xb = read_json(M.R39 / 'configs' / f'{jdir.name}.json')
    oj = M.joint_fields(M.finish(xb, f'R41_G2_F60_R30_H1003_ORACLE_joint_{o}', 'G2', 'both_joint', oracle=True), t39[o], 'both_joint')
    oj.update(supervision='ORACLE_MISSING', supervision_label='ORACLE_MISSING', supervision_rule=ofr['supervision_rule'], reference_unit=M.r38ref(fid),
              reference_joint=M.ref(jdir, 'R41-G reused N-JOINT reference (R39X s1003): head/null init, student init, batch chain asserted in-unit', True), amendment=ofr['amendment'])
    put(oj, f'n_frozen={fid};n_joint={jdir.name}')
write_json(W / 'configs' / 'UNITS_R41G.json', units)
with open(W / 'RESOLVED_MATRIX_R41G.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('R41-G configs', len(units))
