"""R50 GPU smoke verdict: every unit DONE in the gate dir, no FAILED file; step-0 native check (ranks == R50 H); R50 layout checks passed at step 0 and at
the last gate step; R38 graph check; R40 one-step rule; JOINT: R39 step-0 student-vs-cache equivalence and one-step check; N_PRESENT_ONLY: p_d recheck and
zero masked-row gradient on every update; ORACLE_MISSING: oracle relabel counters present (oracle process only); cond = *_lfF. Writes <gate>/SMOKE_R50.json
and appends to ROOT/GPU_SMOKE_R50.json. Usage: python -I -B tools/r50_smoke_check.py <gate_dir> <units> <rc>"""
import json, os, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]; G = Path(sys.argv[1]); units = sys.argv[2].split(','); rc = int(sys.argv[3])
rj = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
out = dict(job=os.environ.get('SLURM_JOB_ID'), gate_dir=str(G), rc=rc, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), units={})
for u in units:
    d = G / u; cfg = rj(W / 'configs' / f'{u}.json'); r = rj(d / 'RESULT.json') or {}; th = rj(d / 'TRAIN_HISTORY.json') or {}
    steps = (r.get('phases') or [{}])[-1].get('steps')   # gate steps (RESULT config holds the formal 8000; checker v1 bug read that)
    e0 = rj(d / 'eval_step0.json') or {}; ef = rj(d / f'eval_step{steps}.json') or {} if steps else {}
    lc = {s: (rj(d / f'R50_LAYOUT_CHECK_step{s}.json') or {}).get('passed') for s in (0, steps)}
    r40 = th.get('r40') or {}; st = (r.get('batch_stats') or {}).get('phase1', {})
    rec = dict(done=(d / 'DONE.json').exists(), failed=[f.name for f in d.glob('FAILED_*.json')], cond=(r.get('config') or {}).get('cond'),
               step0=(e0.get('step0_native_check') or {}).get('passed'), r50_layout=lc, one_step_r40=(r40.get('one_step_check') or {}).get('passed'),
               graph_check=((th.get('r38') or {}).get('step1_graph_check') or {}).get('passed'),
               step0_cache_equivalence=((th.get('r39') or {}).get('step0_cache_equivalence') or {}).get('passed'),
               one_step_r39=((th.get('r39') or {}).get('one_step_check') or {}).get('passed'),
               present_check=(r40.get('present_check') or {}).get('passed'), masked_grad=r40.get('masked_grad'),
               oracle_counters={k: v for k, v in st.items() if k.startswith('oracle_')}, rsum_step0=(e0.get('dev_retrieval') or {}).get('rsum'),
               rsum_last=(ef.get('dev_retrieval') or {}).get('rsum'), native_rsum=(r.get('native_H_dev') or {}).get('rsum'), gpu=(r.get('env') or {}).get('gpu'),
               train_seconds=(r.get('cost') or {}).get('train_seconds'), peak_gpu_memory_gb=((r.get('cost') or {}).get('peak_gpu_memory_bytes') or 0) / 2 ** 30)
    ok = rec['done'] and not rec['failed'] and rec['step0'] and all(lc.values()) and rec['one_step_r40'] and rec['graph_check'] and str(rec['cond']).endswith('_lfF')
    if cfg['encoder'] == 'joint_global':
        ok = ok and rec['step0_cache_equivalence'] and rec['one_step_r39']
    if cfg['supervision'] == 'N_PRESENT_ONLY':
        ok = ok and rec['present_check'] and rec['masked_grad'] and rec['masked_grad']['violations'] == 0 and rec['masked_grad']['updates_checked'] > 0
    if cfg['oracle']:
        ok = ok and bool(rec['oracle_counters']) and 'oracle_runs_r50' in str(G)
    rec['passed'] = bool(ok); out['units'][u] = rec
out['ALL_PASS'] = rc == 0 and all(v['passed'] for v in out['units'].values())
(G / 'SMOKE_R50.json').write_text(json.dumps(out, indent=1, default=str))
agg = rj(W / 'GPU_SMOKE_R50.json') or {}; agg[out['job']] = out; (W / 'GPU_SMOKE_R50.json').write_text(json.dumps(agg, indent=1, default=str))
print('R50_SMOKE ALL_PASS', out['ALL_PASS'], json.dumps({u: v['passed'] for u, v in out['units'].items()}))
