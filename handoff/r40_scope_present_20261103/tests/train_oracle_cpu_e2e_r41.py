"""R41 CPU check helper (r41f) (oracle-runner process; basename starts with train_oracle so the R39 oracle module accepts the import):
runs production run_unit for one C unit on CPU for 2 updates (readouts off) in OUT/oracle_runs/_cpu_scratch and prints R41_ORACLE_E2E <json>.
Usage: python -I -B tests/train_oracle_cpu_e2e.py <unit>"""
import json, os, sys, time
from pathlib import Path
os.environ['R39_ORACLE_PROCESS'] = '1'
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / "r41f"))
import torch
import common as C
C.install_guards(role='oracle')
import oracle_truth
import train_unit as TU


def main():
    u = sys.argv[1]
    TU.set_determinism()
    cfg = TU.cfg_of(u); assert cfg['oracle']
    cfg['_config_sha256'] = C.sha256(W / 'configs' / f'{u}.json')
    ctx = TU.Ctx(cfg['cond'], torch.device('cpu'), need_local=False)
    if cfg['encoder'] == 'joint_global':
        ctx.ensure_joint()
    od = C.ORACLE_RUNS / '_cpu_scratch' / f"{u}_{os.environ.get('SLURM_JOB_ID', 'local')}_{int(time.time())}"
    res = TU.run_unit(dict(cfg, phase=1), ctx, pool=None, out_dir=od, stop_at=2, no_readout=True, oracle=oracle_truth)
    h = res['hist']; r40 = h['r40']
    st = {k: v for k, v in h['stats_sum'].items() if k.startswith('oracle_')}
    rec = dict(unit=u, one_step_r40=r40.get('one_step_check'), references=r40.get('references'), r38_chain=h['r38']['reference_chain_equal_at'],
               step0_cache_equivalence=h['r39'].get('step0_cache_equivalence', {}).get('passed'), oracle_stats_2_updates=st, guard=C.guard_record())
    rec['passed'] = bool(rec['one_step_r40']['passed'] and rec['r38_chain'] == ([0] if cfg.get('reference_unit') else []) and (cfg['encoder'] != 'joint_global' or rec['step0_cache_equivalence'])
                         and all(v['init_equal'] and v['chain_equal_at'] == [0] and v['student_init_equal'] in (None, True) for v in (rec['references'] or {}).values()) and st)
    print('R41_ORACLE_E2E ' + json.dumps(rec, default=str), flush=True)


if __name__ == '__main__':
    main()
