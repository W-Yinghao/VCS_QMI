"""R44 configs (package VCS_R39X_R42_Review_R44_Plan_20261103, configs/MATRIX.csv, 39 units; main session relaying the user 「查看这个包，准备相关的实验，并提交，
这个我也倾向是normal，优先级还是比R33高」). Unit ids = the package MATRIX unit_id.
  A (9, tree r44f)  Flickr60 R30-H s1001/s1002/s1003 x J/BCE/NCE, N x both_joint, 16000 updates from the original H + fresh head, training seed 2002
     exactly as R41-B (build_heads(..., 2002), sampler [2002, 3]); config = the R41-A 16k config with the R41-B seed fields; readouts
     0/250/500/1000/2000/4000/8000/12000/16000. In-unit reference: s1001 = R41-B seed-2002 8000-update unit as a prefix reference (same init and
     sampler stream; head / null / student init + batch chain at its 7 readouts); s1002 / s1003: none (no seed-2002 unit exists). R41-A seed-1001
     16k units = read-only paired controls in the summary.
  B (24, trees r44f / r44c)  ROUTE_PRESENT / ROUTE_RANDOM x J/BCE/NCE x {F00 R30-H f00_s1001 (n00), F60 R30-H f60_s1001 (n60), F80 IN2R R33 H0 f80_s1001
     (f80in), CC31 R31 H0_default cc_s1001 (cc)}, 8000 updates; config = the ALL (N-JOINT) unit of the condition + routing; in-unit reference =
     that ALL unit (same head init, student init and sampler stream -> identical batch chain; the random mask uses its own stateless stream).
  C (6, tree r44f)  PRESENT_ALL = R40 N_PRESENT_ONLY x both_joint on F00 and F80; config = the ALL unit + the R40 PRESENT supervision fields;
     in-unit reference = the ALL unit (N_PRESENT_ONLY draws the same N lists).
Writes configs/R44_*.json, configs/UNITS_R44.json, RESOLVED_MATRIX_R44.csv. Usage: python -I -B tools/make_configs_r44.py (v9_sgr env)"""
import copy, csv, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'tools'))
import make_configs_r41 as M
from common import sha256, write_json, read_json
PKG = Path('/home/infres/yinwang/CS_QMI/r44_plan/VCS_R39X_R42_Review_R44_Plan_20261103')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
R39C = M.R39 / 'configs'; R39R = M.R39R
QUOTE = '查看这个包，准备相关的实验，并提交，这个我也倾向是normal，优先级还是比R33高'
CKS_A = [0, 250, 500, 1000, 2000, 4000, 8000, 12000, 16000]
ALL = {  # condition key -> (cond, tree, ALL config path, ALL unit dir)
    'F00_R30_H1001': ('n00', 'r44f', R39C / 'R39X_Flickr00_R30H_s1001_N_joint_global_{o}.json', R39R / 'runs_r39x' / 'R39X_Flickr00_R30H_s1001_N_joint_global_{o}'),
    'F60_R30_H1001': ('n60', 'r44f', R39C / 'R39_Flickr60_N_joint_global_{o}.json', R39R / 'runs' / 'R39_Flickr60_N_joint_global_{o}'),
    'F80_IN2R_H1001': ('f80in', 'r44f', W / 'configs' / 'R41_C_F80_IN2R_H1001_N_joint_{o}.json', OUT / 'runs_r41' / 'R41_C_F80_IN2R_H1001_N_joint_{o}'),
    'CC_R31_H1001': ('cc', 'r44c', R39C / 'R39_CC152K_N_joint_global_{o}.json', R39R / 'runs' / 'R39_CC152K_N_joint_global_{o}'),
}
ROUTE_RULE = {
    'ROUTE_PRESENT': 'R44 ROUTE_PRESENT: N lists / labels / Q unchanged (OBS_MISSING_NULL); head + null receive the gradient of the full-N loss L_h = E[ell]; the '
                     'student encoders receive the gradient of L_e = E[m ell] / p_d per direction (m = observed partner present among the 33 slots, D36.valid_rows; '
                     'p_d fixed from the full FIT observed relation + candidate tables, R40 presence_probs); two autograd.grad on one forward graph, one clip per '
                     'peer (head + null + student), one AdamW step',
    'ROUTE_RANDOM': 'R44 ROUTE_RANDOM: as ROUTE_PRESENT but m = independent Bernoulli(p_d) per query row from the stateless generator '
                    'default_rng([init_seed, 44, step]) (never the sampler / init RNG; identical mask sequence for J / BCE / NCE), unrelated to labels / features / truth'}


def put(cfg, tree, rows, units, extra):
    cfg = {k: v for k, v in cfg.items() if k not in ('package_unit', 'package_block', 'job_bundle', 'matrix_source', 'extension', 'teacher_parents', 'r39_template')}
    cfg.update(code_tree=tree, runner=f'{tree}/train_unit.py', line='R44', oracle=False, test_read=False, label='development experiment (no TEST read in this line)',
               package_unit=cfg['unit_id'], authorisation=dict(user_quote=QUOTE, relayed_by='main session', package=str(PKG)))
    write_json(W / 'configs' / f"{cfg['unit_id']}.json", cfg); units.append(cfg['unit_id'])
    rows.append(dict(unit_id=cfg['unit_id'], block=cfg['block'], tree=tree, cond=cfg['cond'], arm=extra['arm'], objective=cfg['objective'], steps=cfg['steps'],
                     training_seed=cfg['init_seed'], sampler_seed=str(cfg['sampler_seed']), supervision=cfg['supervision'], routing=cfg.get('routing'),
                     ckpt_steps=' '.join(map(str, cfg['ckpt_steps'])), in_unit_reference=(cfg.get('reference_joint') or {}).get('unit_id'),
                     reference_mode=(cfg.get('reference_joint') or {}).get('mode'), paired_controls=extra.get('controls', ''), config_sha256=sha256(W / 'configs' / f"{cfg['unit_id']}.json")))


def main():
    mat = {r['unit_id']: r for r in csv.DictReader(open(PKG / 'configs' / 'MATRIX.csv'))}
    units, rows = [], []
    for o in M.OBJ:
        # ---------------- A
        bB = read_json(W / 'configs' / f'R41_B_F60_R30_H1001_seed2002_N_joint_{o}.json')
        for s in ('1001', '1002', '1003'):
            pk = f'F60_R30_H{s}'; a41 = read_json(W / 'configs' / f'R41_A_{pk}_N_joint_16k_{o}.json')
            cfg = copy.deepcopy(a41)
            cfg.update(unit_id=f'R44_A_{pk}_ALL_{o}_train2002', block='A', init_seed=2002, sampler_seed=[2002, 3], seed_rule=bB['seed_rule'].replace('R41 B', 'R44 A (= R41-B definition)'),
                       head_init=bB['head_init'], ckpt_steps=CKS_A, primary_endpoint='fixed_step_16000', reference_unit=None,
                       budget_rule='R41-A 16000-update stretched cosine from the original H + fresh head (never resumed from an 8000-update checkpoint); training seed 2002 as R41-B')
            if s == '1001':
                cfg['reference_joint'] = M.ref(OUT / 'runs_r41' / f'R41_B_F60_R30_H1001_seed2002_N_joint_{o}', 'R44 prefix reference (R41-B seed 2002, 8000 updates, same init / sampler stream): head / null init, student init, batch chain at its 7 readouts', True, 'prefix')
            else:
                cfg['reference_joint'] = None
                cfg['reference_note'] = 'no seed-2002 unit exists for this parent: no in-unit reference (the R41-A seed-1001 16k unit is the paired control in the summary)'
            put(cfg, 'r44f', rows, units, dict(arm='ALL', controls=f'R41_A_{pk}_N_joint_16k_{o} (seed 1001)'))
        # ---------------- B / C
        for ck, (cond, tree, cpath, udir) in ALL.items():
            base = read_json(Path(str(cpath).format(o=o))); adir = Path(str(udir).format(o=o)); assert base['cond'] == cond and base['steps'] == 8000
            refj = M.ref(adir, 'R44 in-unit reference = the ALL (N-JOINT) unit of this condition: head / null init, student init and batch chain asserted', True)
            refj['config_sha256'] = sha256(cpath.parent / f'{adir.name}.json') if (cpath.parent / f'{adir.name}.json').exists() else None
            for arm in ('ROUTE_PRESENT', 'ROUTE_RANDOM', 'PRESENT_ALL'):
                uid = f'R44_{"C" if arm == "PRESENT_ALL" else "B"}_{ck}_{arm}_{o}_train1001'
                if uid not in mat:
                    continue
                cfg = M.joint_fields(M.finish(copy.deepcopy(base), uid, 'C' if arm == 'PRESENT_ALL' else 'B', 'both_joint'), base, 'both_joint')
                cfg.update(reference_unit=None, reference_joint=refj, reference_note='same init / sampler stream as the ALL unit (identical N lists); only the gradient routing / masking differs')
                if arm == 'PRESENT_ALL':
                    p40 = read_json(W / 'configs' / f'R40_B_F60_R30_H1001_PRESENT_both_joint_{o}.json')
                    cfg.update(supervision='N_PRESENT_ONLY', supervision_label='N_PRESENT_ONLY', supervision_rule=p40['supervision_rule'] + ' (R44 C: R40 definition on this parent)')
                else:
                    cfg.update(routing=arm, supervision_rule=base['supervision_rule'] + ' | ' + ROUTE_RULE[arm])
                put(cfg, tree, rows, units, dict(arm=arm, controls=f'ALL={adir.name}'))
    assert sorted(units) == sorted(mat), (set(mat) - set(units), set(units) - set(mat))
    write_json(W / 'configs' / 'UNITS_R44.json', units)
    with open(W / 'RESOLVED_MATRIX_R44.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('R44 configs', len(units), {b: sum(r['block'] == b for r in rows) for b in 'ABC'})


if __name__ == '__main__':
    main()
