"""R50 CPU end-to-end for ORACLE_MISSING units (oracle-runner process: basename starts with train_oracle, so the oracle module accepts the import; role oracle):
production run_unit on CPU for 2 updates (readouts off) in ORACLE_RUNS/_cpu_scratch: R40 one-step rule, R38 graph check, R39 step-0 cache equivalence (JOINT),
oracle relabel counters. Prints R50_ORACLE_E2E <json>. Usage: python -I -B tests/train_oracle_e2e_r50.py <unit,unit,...>"""
import json, os, sys, time
from pathlib import Path
os.environ['R39_ORACLE_PROCESS'] = '1'
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r50f'))
import torch
import common as C
C.install_guards(role='oracle')
import oracle_truth
import train_unit as TU
TU.set_determinism()
out = {}; ctxs = {}
for u in sys.argv[1].split(','):
    cfg = TU.cfg_of(u); assert cfg['oracle']; cfg['_config_sha256'] = C.sha256(W / 'configs' / f'{u}.json')
    if cfg['cond'] not in ctxs:
        ctxs = {cfg['cond']: TU.Ctx(cfg['cond'], torch.device('cpu'), need_local=False)}
    ctx = ctxs[cfg['cond']]
    if cfg['encoder'] == 'joint_global':
        ctx.ensure_joint()
    od = C.ORACLE_RUNS / '_cpu_scratch' / f"{u}_{os.environ.get('SLURM_JOB_ID', 'local')}_{int(time.time())}"
    t0 = time.time()
    res = TU.run_unit(dict(cfg, phase=1), ctx, pool=None, out_dir=od, stop_at=2, no_readout=True, oracle=oracle_truth)
    h = res['hist']; r40 = h['r40']
    st = {k: v for k, v in h['stats_sum'].items() if k.startswith('oracle_')}
    rec = dict(one_step_r40=(r40.get('one_step_check') or {}).get('passed'), step1_graph_check=h['r38'].get('step1_graph_check', {}).get('passed'),
               step0_cache_equivalence=h['r39'].get('step0_cache_equivalence', {}).get('passed'), oracle_stats_2_updates=st, guard=C.guard_record(), seconds=time.time() - t0)
    rec['passed'] = bool(rec['one_step_r40'] and rec['step1_graph_check'] and (cfg['encoder'] != 'joint_global' or rec['step0_cache_equivalence']) and st
                         and st.get('oracle_i2t_rows', 0) > 0)
    out[u] = rec
print('R50_ORACLE_E2E ' + json.dumps(out, default=str), flush=True)
