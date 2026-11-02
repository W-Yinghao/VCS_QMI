"""CPU: resolve the package MATRIX.csv (42 units) into per-unit JSON configs + MATRIX_RESOLVED.csv. Reads only
the package matrix (passed by path) and writes ROOT/configs and ROOT/MATRIX_RESOLVED.csv."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS, CONDS, CKPT_STEPS_ABC, CKPT_STEPS_D, R30_RUNS, write_json

OPT = dict(name='AdamW', lr_head=1e-4, lr_local=1e-5, weight_decay=1e-4, betas=[0.9, 0.999], eps=1e-8, schedule='constant')
JOBS = {  # bundle -> units are taken in this order inside one GPU job (see SUBMIT plan)
    'A_n00': lambda r: r['block'] == 'A' and r['noise'] == '0',
    'AB_n60': lambda r: (r['block'] == 'A' and r['noise'] == '60') or (r['block'] == 'B' and r['supervision'] == 'TRUST'),
    'B_oracle': lambda r: r['block'] == 'B' and r['supervision'] == 'ORACLE',
    'C_n00': lambda r: r['block'] == 'C' and r['noise'] == '0',
    'C_n60': lambda r: r['block'] == 'C' and r['noise'] == '60',
    'D_n00': lambda r: r['block'] == 'D' and r['noise'] == '0',
    'D_n60': lambda r: r['block'] == 'D' and r['noise'] == '60',
}


def main(matrix_csv):
    rows = list(csv.DictReader(open(matrix_csv)))
    assert len(rows) == 42
    out_rows, uids = [], []
    for r in rows:
        cond = 'n00' if r['noise'] == '0' else 'n60'
        blk = r['block']
        struct = {'ID': 'ID', 'G': 'G', 'L': 'L', 'L_joint_local': 'L', 'L_head_continue': 'L'}[r['structure']]
        cfg = dict(unit_id=r['unit_id'], block=blk, cond=cond, noise_percent=int(r['noise']), seed=int(r['seed']), structure=struct,
                   structure_package=r['structure'], objective=r['objective'], supervision=r['supervision'], oracle=r['oracle'] == 'True',
                   steps=int(r['steps']), ckpt_steps=list(CKPT_STEPS_D if blk == 'D' else CKPT_STEPS_ABC), primary_endpoint=f"step_{r['steps']}",
                   init_seed=int(r['seed']), optimizer=OPT, clip_per_peer=2.0, evaluation=r['evaluation'], test_read=False,
                   parent_H=dict(unit=CONDS[cond]['h_unit'], model_best=str(R30_RUNS / CONDS[cond]['h_unit'] / 'model_best.pth.tar')),
                   precision='float32 training (TF32 off, deterministic algorithms); evaluation affine in float64 on the exact native shard score')
        if blk in ('A', 'B'):
            cfg.update(batch_size=128, sampler_seed=int(r['seed']),
                       sampling='per step B=128 records IID with replacement from the pool (A: all 145000; TRUST: frozen GMM subset; ORACLE: all) via numpy default_rng(1001); image = map[record] (observed; ORACLE truth)',
                       P='record diagonal 1/B', Q='off-diagonal record draws 1/(B(B-1)), same-image retained (R34)', NCE='bidirectional CE, diagonal labels (R34)')
        else:
            cfg.update(n_queries=16, n_slots=32, sampler_seed=[int(r['seed']), 3 if blk == 'C' else 4],
                       sampling='per step 16 i2t + 16 t2i query records IID with replacement over 145000 via numpy default_rng(seed list); 32-slot lists from the frozen-H FIT top-32 tables; slots permuted per row by the same rng',
                       P='P_C: observed-positive slots, uniform per row', Q='Q_C = P_Z x Uniform(32), positive slots included')
        if blk == 'D':
            cfg.update(d_mode='joint' if r['structure'] == 'L_joint_local' else 'head_control', parent_unit=r['parent'],
                       parent_ckpt=str(RUNS / r['parent'] / 'ckpt_step8000.pt'),
                       unfrozen=('image fc + mlp (BN running stats frozen), text BiGRU' if r['structure'] == 'L_joint_local' else 'none (critic heads only)'),
                       frozen='GloVe embedding, BN running stats, GPO/global encoders, native skip, candidate tables')
        job = [k for k, f in JOBS.items() if f(r)]; assert len(job) == 1
        cfg['job_bundle'] = job[0]
        write_json(ROOT / 'configs' / f"{r['unit_id']}.json", cfg)
        uids.append(r['unit_id'])
        out_rows.append({**r, 'cond': cond, 'structure_resolved': struct, 'job_bundle': job[0], 'sampler_seed': json.dumps(cfg['sampler_seed']),
                         'ckpt_steps': ' '.join(map(str, cfg['ckpt_steps'])), 'lr_head': OPT['lr_head'],
                         'lr_local': OPT['lr_local'] if cfg.get('d_mode') == 'joint' else '', 'weight_decay': OPT['weight_decay'],
                         'clip_per_peer': 2.0, 'parent_resolved': cfg.get('parent_ckpt', cfg['parent_H']['model_best']),
                         'runner': 'train_oracle.py' if cfg['oracle'] else 'train_unit.py'})
    write_json(ROOT / 'configs' / 'UNITS.json', uids)
    with open(ROOT / 'MATRIX_RESOLVED.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    print(len(uids), 'configs')


if __name__ == '__main__':
    main(sys.argv[1])
