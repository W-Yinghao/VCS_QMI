"""R50 CPU end-to-end (H: one real small update) for non-oracle units (one tree per process): production run_unit on CPU for 2 updates (readouts off):
R40 one-step active-path rule (trainable path changes, frozen modality / BN stats / word embedding / parent cache unchanged), R38 graph check, R39 step-0
student-vs-cache equivalence (JOINT), N_PRESENT_ONLY p_d recheck and zero masked-row gradient on every update. Prints R50_E2E <json>.
Usage: python -I -B tests/r50_e2e.py <tree> <unit,unit,...>"""
import json, os, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
TREE, UNITS = sys.argv[1], sys.argv[2].split(',')
sys.path.insert(0, str(W / TREE))
import torch
import common as C; C.install_guards('trainer'); import train_unit as TU
TU.set_determinism()
SCRATCH = C.R50_OUT / 'cpu_scratch_r50'
dev = torch.device('cpu'); ctxs = {}; out = {}
for u in UNITS:
    cfg = TU.cfg_of(u); cfg['_config_sha256'] = C.sha256(W / 'configs' / f'{u}.json'); assert cfg['code_tree'] == TREE and not cfg['oracle']
    if cfg['cond'] not in ctxs:
        ctxs = {cfg['cond']: TU.Ctx(cfg['cond'], dev, need_local=False)}
    ctx = ctxs[cfg['cond']]
    if cfg['encoder'] == 'joint_global':
        ctx.ensure_joint()
    od = SCRATCH / f'{u}_{os.environ.get("SLURM_JOB_ID", "local")}_{int(time.time())}'
    t0 = time.time()
    res = TU.run_unit(dict(cfg, phase=1), ctx, pool=TU.pool_of(cfg, ctx), out_dir=od, stop_at=2, no_readout=True, oracle=None)
    h = res['hist']; r40 = h['r40']
    rec = dict(one_step_r40=(r40.get('one_step_check') or {}).get('passed'), step0_cache_equivalence=(h['r39'].get('step0_cache_equivalence') or {}).get('passed'),
               one_step_r39=(h['r39'].get('one_step_check') or {}).get('passed'), step1_graph_check=h['r38'].get('step1_graph_check', {}).get('passed'),
               present_check=(r40.get('present_check') or {}).get('passed'), masked_grad=r40.get('masked_grad'), seconds=time.time() - t0)
    ok = bool(rec['one_step_r40'] and rec['step1_graph_check'])
    if cfg['encoder'] == 'joint_global':
        ok &= bool(rec['step0_cache_equivalence'] and rec['one_step_r39'])
    if cfg['supervision'] == 'N_PRESENT_ONLY':
        ok &= bool(rec['present_check']) and r40['masked_grad']['violations'] == 0 and r40['masked_grad']['updates_checked'] == 2
    rec['passed'] = bool(ok); out[u] = rec
print('R50_E2E ' + json.dumps(out, default=str))
