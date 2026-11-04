"""R43 CPU end-to-end (one tree per process): 2-update production run_unit of the listed units on CPU (no readout), as cpu_test_r42.t_e2e
(step-0 cache equivalence for JOINT, R38 first-update graph check, R40 one-step rule) + R43: TRUST units draw their anchor records from the
bound B-TRUST subset (lineage trust_sha256 = prep manifest). Prints R43_E2E <json>. Usage: python -I -B tests/e2e_r43.py <tree> <unit,unit,...>"""
import json, os, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
TREE, UNITS = sys.argv[1], sys.argv[2].split(',')
sys.path.insert(0, str(W / TREE))
import torch
import common as C; C.install_guards('trainer'); import train_unit as TU, data36 as D36
TU.set_determinism()
SCRATCH = C.OUT / 'cpu_scratch_r43'
dev = torch.device('cpu'); ctxs = {}; out = {}
for u in UNITS:
    cfg = TU.cfg_of(u); cfg['_config_sha256'] = C.sha256(W / 'configs' / f'{u}.json')
    assert cfg['code_tree'] == TREE
    if cfg['cond'] not in ctxs:
        ctxs = {cfg['cond']: TU.Ctx(cfg['cond'], dev, need_local=False)}
    ctx = ctxs[cfg['cond']]
    if cfg['encoder'] == 'joint_global':
        ctx.ensure_joint()
    od = SCRATCH / f'{u}_{os.environ.get("SLURM_JOB_ID", "local")}_{int(time.time())}'
    t0 = time.time()
    res = TU.run_unit(dict(cfg, phase=1), ctx, pool=TU.pool_of(cfg, ctx), out_dir=od, stop_at=2, no_readout=True, oracle=None)
    h = res['hist']; r40 = h['r40']
    rec = dict(one_step_r40=r40.get('one_step_check'), step0_cache_equivalence=(h['r39'].get('step0_cache_equivalence') or {}).get('passed'),
               step1_graph_check=h['r38'].get('step1_graph_check', {}).get('passed'), seconds=time.time() - t0, supervision=cfg['supervision'], cond=cfg['cond'])
    ok = bool(rec['one_step_r40'] and rec['one_step_r40']['passed'] and rec['step1_graph_check'] and (cfg['encoder'] != 'joint_global' or rec['step0_cache_equivalence']))
    if cfg['supervision'] in D36.TRUST_SUPS:
        pool = TU.pool_of(cfg, ctx); pm = C.read_json(Path(C.CONDS[cfg['cond']]['prep_dir']) / 'PREP_MANIFEST.json')
        rng = __import__('numpy').random.default_rng(cfg['sampler_seed'])
        ri, rt, _, _ = D36.draw_main(rng, cfg['n_queries'], cfg['n_slots'], pool)
        rec.update(trust_pool_size=int(len(pool)), trust_sha256_bound=ctx.d['trust_sha256'] == pm['files'][C.CONDS[cfg['cond']]['trust_file']]['sha256'],
                   anchors_in_pool=bool(__import__('numpy').isin(ri, pool).all() and __import__('numpy').isin(rt, pool).all()))
        ok &= rec['trust_sha256_bound'] and rec['anchors_in_pool'] and 0 < len(pool) < C.N_TRAIN_CAP
    else:
        ok &= TU.pool_of(cfg, ctx) is None
    rec['passed'] = bool(ok); out[u] = rec
print('R43_E2E ' + json.dumps(out, default=str))
