"""R37S critic line: 18 unit configs, written to ROOT/configs/ (read by train_unit). Each config starts from the R37C reference config of the same route/objective
(r37c_pilot_cc_h1001_{N,T}_G_{J,BCE,NCE}.json, read-only) -- as R38 make_configs did for its new CC parents -- so head-init seed,
sampler stream, n_queries, steps, readouts, clip and AdamW settings are unchanged; R37S changes the cond/parent (cc_r37s), structure
(G/GD), schedule, and bookkeeping. Units (18): approved 6 = {T, N} shared constant x {J, BCE, NCE}; 12 matching R38 cells added by the
user (「实验要做的全一些」) = N shared-cosine, N directional-constant, N directional-cosine, T shared-cosine x {J, BCE, NCE}.
Usage: python -I -B tools/make_critic_configs.py"""
import csv, json, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r37s_critic'))
from common import sha256, write_json, read_json, utc, CONDS, SERIES

R37C_CFG = Path('/home/infres/yinwang/CS_QMI/handoff/r37c_cc152k_20261102/configs')
SUP = {'N': 'OBS_MISSING_NULL', 'T': 'TRUST_HARD'}
CELLS = [('N', 'shared', 'constant', 'A_APPROVED'), ('T', 'shared', 'constant', 'A_APPROVED'),
         ('N', 'shared', 'cosine', 'B_USER_ADDED'), ('N', 'directional', 'constant', 'B_USER_ADDED'), ('N', 'directional', 'cosine', 'B_USER_ADDED'),
         ('T', 'shared', 'cosine', 'C_USER_ADDED')]
COND = 'cc_r37s'


def uid(route, arch, obj, sched):
    return f'r37s_cc_r30h_s1001_{route}_{arch}_{obj}_{sched}'


def main():
    c = CONDS[COND]
    out, uids = [], []
    (W / 'configs').mkdir(exist_ok=True)
    for route, arch, sched, block in CELLS:
        for obj in ('J', 'BCE', 'NCE'):
            ref_id = f'r37c_pilot_cc_h1001_{route}_G_{obj}'
            base = read_json(R37C_CFG / f'{ref_id}.json')
            assert base['structure'] == 'G' and base['supervision'] == SUP[route] and base['objective'] == obj and base['optimizer']['schedule'] == 'constant'
            assert base['steps'] == 8000 and base['optimizer']['lr_head'] == 1e-4 and not base['oracle']
            u = uid(route, arch, obj, sched)
            cfg = {k: v for k, v in base.items() if k not in ('unit_id', 'package_unit', 'package_block', 'job_bundle', 'block')}
            cfg.update(unit_id=u, package_unit=None, package_block=block, block=block[0], line='R37S-critic',
                       cond=COND, code_tree='r37s_critic', dataset='CC152K (real noise)', parent_key='cc_R37S_s1001', route=route, architecture=arch,
                       structure='GD' if arch == 'directional' else 'G', parent_seed=c['parent_seed'],
                       optimizer=dict(base['optimizer'], schedule=sched, cosine_floor=1.0 if sched == 'constant' else 0.1),
                       schedule_rule=('constant lr 1e-4 (R36/R37 protocol)' if sched == 'constant' else
                                      'common cosine (R38): lr(t) = 1e-4*[0.1+0.45*(1+cos(pi*t/7999))], t = 0..7999 update index; no warmup, optimizer never reset; same for J/BCE/NCE'),
                       architecture_rule=('shared G (R35/R36 GHead): g = a s + b + w^T h(phi) + c' if arch == 'shared' else
                                          'directional G (R38 DirGHead): shared 4097-256-128 trunk, per-direction 128->1 output (zero init) and positive affine (a0/b0); +131 params/peer; i2t rows scored with g_i2t, t2i rows with g_t2i, in training, DEV inference and FIT diagnostics'),
                       evaluation=base['evaluation'] + ('; directional: i2t candidates with g_i2t, t2i candidates with g_t2i, peer mean per direction; GT pairs scored per direction' if arch == 'directional' else ''),
                       parent_H=dict(unit=c['h_unit'], series=SERIES, model_best=str(Path(c['parent_dir']) / 'model_best.pth.tar'), seed=c['parent_seed']),
                       head_init=base['head_init'].replace('parent-specific', 'parent-specific (this parent: R37S)'),
                       reference_unit=None,
                       reference_note=f'no earlier critic unit on this parent; config fields copied from {ref_id} (R37C, R31 parent s1001, read-only), parent/cond replaced; '
                                      'init-state (shared keys) and batch-index chain equality across the 18 units of this parent checked post hoc in the summary',
                       test_read=False, label='development experiment (no TEST read in this line)', runner='r37s_critic/train_unit.py',
                       job_bundle=f'cc_R37S_s1001_{route}_{arch}', matrix_source=('approved R37S critic 6 (shared constant)' if block.startswith('A') else
                                                                                 'user addition 「实验要做的全一些」: matching R38 cells on this parent'))
            write_json(W / 'configs' / f'{u}.json', cfg); uids.append(u)
            out.append(dict(unit_id=u, block=block, route=route, architecture=arch, schedule=sched, objective=obj, structure=cfg['structure'],
                            supervision=cfg['supervision'], steps=cfg['steps'], base_config=ref_id, base_config_sha256=sha256(R37C_CFG / f'{ref_id}.json'),
                            config_sha256=sha256(W / 'configs' / f'{u}.json')))
    assert len(uids) == 18 == len(set(uids))
    write_json(W / 'configs' / 'UNITS.json', uids)
    with open(W / 'CRITIC_MATRIX.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(len(uids), 'configs')


if __name__ == '__main__':
    main()
