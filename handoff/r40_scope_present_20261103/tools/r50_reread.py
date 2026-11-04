"""R50 PRIMARY re-read (zero training) of existing N references: the saved checkpoints (heads / nulls / JOINT student, steps 0/250/.../8000) of an R47 G
N-JOINT or R48 N-frozen unit are loaded into the production R50 trainer objects and read out with the production readout (readout_r36: DEV two-stage,
H top-100 reranked, coverage, list-J, calibration) on the R50 FIT_FIXED_CENTER_V1 cond <base>_lfF (R50 cache: FIT = R47 G FIT, DEV centred by the FIT
mean). Also the step-0 native check (ranks == R50 H ranks, near ties excepted) and the R50 in-unit layout checks at steps 0 and 8000. The source checkpoint
lineage is compared with the R50 prep (calibration a0/b0, candidate table sha256) and the source config with the cond-swapped config.
Writes OUT_R50/r50_rereads/<unit>/{eval_step*.json, dev_scores_step{0,8000}.npz, dev_ranks_step8000.npz, R50_LAYOUT_CHECK_*.json, REREAD.json}.
Usage: python -B tools/r50_reread.py <tree r50f|r50c> <unit,unit,...>"""
import json, os, sys, time
from pathlib import Path
import numpy as np
W = Path(__file__).resolve().parents[1]
TREE, UNITS = sys.argv[1], sys.argv[2].split(',')
sys.path.insert(0, str(W / TREE))
import torch
import common as C
C.install_guards(role='trainer')
import train_unit as TU
TU.set_determinism()
import evals as EV, r39ext as R9, r50ext as R50, data36 as D36
from heads import build_heads, NullSlots
from common import PEERS, sha256, write_json, read_json, utc
dev = torch.device('cuda'); OUTR = C.R50_OUT / 'r50_rereads'; ctxs = {}; done = []
for u in UNITS:
    t0 = time.time()
    src_cfg = TU.cfg_of(u); runs = C.OUT / ('runs_r48' if u.startswith('R48_') else 'runs_r47'); src = runs / u
    assert (src / 'DONE.json').exists(), src
    base = src_cfg['cond'][:-len('_lfG')]; assert src_cfg['cond'] == f'{base}_lfG'
    cfg = dict(src_cfg, cond=f'{base}_lfF', code_tree=TREE, unit_id=u, phase=1)
    od = OUTR / u; od.mkdir(parents=True, exist_ok=True)
    if (od / 'REREAD.json').exists():
        print('skip', u); continue
    if cfg['cond'] not in ctxs:
        ctxs = {cfg['cond']: TU.Ctx(cfg['cond'], dev, need_local=False)}
    ctx = ctxs[cfg['cond']]
    joint = cfg.get('encoder') == 'joint_global'
    if joint:
        ctx.ensure_joint()
    d = ctx.d; a0, b0 = d['a0b0']['a0'], d['a0b0']['b0']
    heads = build_heads(cfg['structure'], a0, b0, cfg['init_seed']).to(dev)
    nulls = NullSlots().to(dev) if cfg['supervision'] in D36.NULL_SUPS else None
    stud = R9.build_students(ctx.joint.ck, dev, C.ADAPTATIONS[cfg['adaptation']]) if joint else None
    src_res = read_json(src / 'RESULT.json')
    lineage = None; per_step = {}; ck_sha = {}
    for k in cfg['ckpt_steps']:
        f = src / f'ckpt_step{k}.pt'; ck_sha[k] = sha256(f)
        ck = torch.load(f, map_location=dev, weights_only=False)
        assert ck['step'] == k and ck['unit_id'] == u
        heads.load_state_dict(ck['heads'])
        if nulls is not None:
            nulls.load_state_dict(ck['nulls'])
        if stud is not None:
            stud.load_state_dict(ck['student'])
        lineage = lineage or ck['lineage']
        rec, arrs, rk = TU.readout_r36(heads, nulls, None, ctx, final=(k == cfg['steps']), stud=stud)
        rec['transitions_vs_H'] = EV.transitions(rk, ctx.nat_rk, rk, ctx.nat_rk, d['dev_tab'])
        if k == 0:
            nat = d['native_dev']; TOL = 1e-6; PC = C.PER_CAP
            di = np.nonzero(rk['ranks_i'] != ctx.nat_rk['ranks_i'])[0]; dt = np.nonzero(rk['ranks_t'] != ctx.nat_rk['ranks_t'])[0]
            tied_i = all(any((np.abs(nat[i] - nat[i, PC * i + j]) <= TOL).sum() > 1 for j in range(PC)) for i in di)
            tied_t = all((np.abs(nat[:, c] - nat[c // PC, c]) <= TOL).sum() > 1 for c in dt)
            rec['step0_native_check'] = dict(n_i2t_rank_diff=int(len(di)), n_t2i_rank_diff=int(len(dt)), rsum=rec['dev_retrieval']['rsum'], native_rsum=ctx.nat_metrics['rsum'],
                                             passed=bool((len(di) == 0 and len(dt) == 0) or (tied_i and tied_t)))
        if k in (0, cfg['steps']):
            rec['r50_layout_check'] = R50.layout_checks(ctx, heads, stud, cfg, od, k, TU.g_pairs_lists)
            np.savez(od / f'dev_scores_step{k}.npz', **arrs)
            if k == cfg['steps']:
                np.savez(od / f'dev_ranks_step{k}.npz', **rk)
        rec['step'] = k; rec['protocol_id'] = R50.PROTOCOL_ID; rec['source_ckpt_sha256'] = ck_sha[k]
        write_json(od / f'eval_step{k}.json', rec)
        sec = read_json(src / f'eval_step{k}.json')['dev_retrieval']
        per_step[k] = dict(primary=rec['dev_retrieval'], secondary_transductive=sec)
        print(u, 'step', k, 'PRIMARY', round(rec['dev_retrieval']['rsum'], 3), 'SECONDARY', round(sec['rsum'], 3), flush=True)
    pm = read_json(Path(C.CONDS[cfg['cond']]['prep_dir']) / 'PREP_MANIFEST.json')
    ident = dict(config_equal_except_cond_tree=all(src_cfg.get(k) == cfg.get(k) for k in set(src_cfg) | set(cfg) if k not in ('cond', 'code_tree', 'phase')),
                 a0b0_equal=bool(lineage['a0b0']['a0'] == a0 and lineage['a0b0']['b0'] == b0),
                 candidate_table_sha256_equal=lineage.get('candidate_table_sha256') == d['cand_sha256'],
                 r50_prep_identical_to_r47G=pm.get('identical_to_r47G'),
                 source_cache_manifest_sha256=lineage['cache_manifest_sha256'], r50_cache_fit_side_identical=read_json(Path(C.CONDS[cfg['cond']]['cache_dir']) / 'CACHE_MANIFEST.json')['files'])
    ident['r50_cache_fit_files_are_r47G'] = all(v.get('source_sha256') == v['sha256'] for n_, v in ident.pop('r50_cache_fit_side_identical').items() if n_.startswith('fit_'))
    out = dict(unit=u, protocol_id=R50.PROTOCOL_ID, source_dir=str(src), source_result_sha256=sha256(src / 'RESULT.json'), source_ckpt_sha256=ck_sha, cond=cfg['cond'],
               H_primary=ctx.nat_metrics, H_secondary_transductive=src_res['native_H_dev'], per_step=per_step, identity=ident,
               step0_native_check=read_json(od / 'eval_step0.json')['step0_native_check'], env=dict(gpu=torch.cuda.get_device_name(0), host=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID')),
               code_sha256=TU.code_hashes(), tool_sha256=sha256(Path(__file__)), seconds=time.time() - t0, test_guard=C.guard_record(), utc=utc())
    write_json(od / 'REREAD.json', out); done.append(u)
    print('R50_REREAD_DONE', u, 'step8000 PRIMARY', per_step[cfg['steps']]['primary']['rsum'], 'SECONDARY', per_step[cfg['steps']]['secondary_transductive']['rsum'], flush=True)
print('R50_REREAD_END', done)
