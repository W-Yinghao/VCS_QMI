"""R47 CPU end-to-end (byte copy of tests/e2e_r44.py except this docstring and the output tag; e2e_r46.py passes the R46-only Ctx(cand_file=...) argument, absent in the R47 trees; R47 S units record a skipped step-0 cache equivalence with passed=True) (one tree per process): 2-update production run_unit on CPU (no readout) of the listed units: R40 one-step rule, R38 graph
check, step-0 student-vs-cache equivalence, in-unit references (head/null init, student init, batch chain at 0), and for routed units the update-0
ROUTING_CHECK (all-ones p=1 = ALL encoder grad, head grad = ALL head grad, encoder grad = conditional objective, masked rows: zero encoder /
nonzero head score-gradient) plus the per-update masked-row and single-step counters. Prints R47_E2E <json>.
Usage: python -I -B tests/e2e_r47.py <tree> <unit,unit,...>"""
import json, os, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
TREE, UNITS = sys.argv[1], sys.argv[2].split(',')
sys.path.insert(0, str(W / TREE))
import torch
import common as C; C.install_guards('trainer'); import train_unit as TU
TU.set_determinism()
SCRATCH = C.OUT / 'cpu_scratch_r44'
dev = torch.device('cpu'); ctxs = {}; out = {}
for u in UNITS:
    cfg = TU.cfg_of(u); cfg['_config_sha256'] = C.sha256(W / 'configs' / f'{u}.json'); assert cfg['code_tree'] == TREE
    if cfg['cond'] not in ctxs:
        ctxs = {cfg['cond']: TU.Ctx(cfg['cond'], dev, need_local=False)}
    ctx = ctxs[cfg['cond']]
    if cfg['encoder'] == 'joint_global':
        ctx.ensure_joint()
    od = SCRATCH / f'{u}_{os.environ.get("SLURM_JOB_ID", "local")}_{int(time.time())}'
    t0 = time.time()
    res = TU.run_unit(dict(cfg, phase=1), ctx, pool=TU.pool_of(cfg, ctx), out_dir=od, stop_at=2, no_readout=True, oracle=None)
    h = res['hist']; r40 = h['r40']; r44 = h.get('r44') or {}
    rec = dict(one_step_r40=(r40.get('one_step_check') or {}).get('passed'), step0_cache_equivalence=(h['r39'].get('step0_cache_equivalence') or {}).get('passed'),
               step1_graph_check=h['r38'].get('step1_graph_check', {}).get('passed'), references=r40.get('references'), routing=cfg.get('routing'),
               routing_check=r44.get('routing_check'), r44_counters={k: r44.get(k) for k in ('updates', 'opt_steps', 'masked_rows', 'masked_rows_enc_grad_nonzero', 'masked_rows_head_grad_nonzero', 'mask_rows', 'query_rows', 'mask_hash')},
               present_check=(r40.get('present_check') or {}).get('passed'), seconds=time.time() - t0)
    ok = bool(rec['one_step_r40'] and rec['step1_graph_check'] and rec['step0_cache_equivalence'])
    ok &= all(v['init_equal'] and v['student_init_equal'] in (None, True) and v['chain_equal_at'] == [0] for v in (rec['references'] or {}).values()) and bool(rec['references']) == bool(cfg.get('reference_joint'))
    if cfg.get('routing'):
        ok &= bool(rec['routing_check'] and rec['routing_check']['passed'] and 'update1' in rec['routing_check'] and rec['routing_check']['update1']['all_ones_p1_reproduces_ALL_encoder_grad']['nontrivial']) and r44['updates'] == r44['opt_steps'] == 2 and r44['masked_rows_enc_grad_nonzero'] == 0 and rec['present_check']
    if cfg['supervision'] == 'N_PRESENT_ONLY':
        ok &= bool(rec['present_check']) and r40['masked_grad']['violations'] == 0
    rec['passed'] = bool(ok); out[u] = rec
print('R47_E2E ' + json.dumps(out, default=str))
