"""R49 GPU smoke verdict (copy of r48_smoke_check.py): every unit DONE in the gate dir, no FAILED file, step-0 check passed (A: step0_native_check; B: step0_init_from_check),
R40 one-step check passed; writes <gate>/SMOKE_R49.json and appends to ROOT/GPU_SMOKE_R49.json. Usage: python -I -B tools/r46_smoke_check.py <gate_dir> <units> <rc>"""
import json, os, sys, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]; G = Path(sys.argv[1]); units = sys.argv[2].split(','); rc = int(sys.argv[3])
rj = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
out = dict(job=os.environ.get('SLURM_JOB_ID'), gate_dir=str(G), rc=rc, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), units={})
for u in units:
    d = G / u; e0 = rj(d / 'eval_step0.json') or {}; r = rj(d / 'RESULT.json') or {}; th = rj(d / 'TRAIN_HISTORY.json') or {}
    s0 = e0.get('step0_init_from_check') or e0.get('step0_native_check')
    one = ((th.get('r40') or {}).get('one_step_check') or {}).get('passed')
    rec = dict(done=(d / 'DONE.json').exists(), failed=[f.name for f in d.glob('FAILED_*.json')], step0=s0, one_step_r40=one,
               step0_cache_equivalence=(th.get('r39') or {}).get('step0_cache_equivalence', {}).get('passed') if th.get('r39') else None,
               graph_check=((th.get('r38') or {}).get('step1_graph_check') or {}).get('passed'), rsum_step20=((r.get('final') or {}).get('dev_retrieval') or {}).get('rsum'),
               gpu=(r.get('env') or {}).get('gpu'), train_seconds_20=(r.get('cost') or {}).get('train_seconds'))
    rec['passed'] = bool(rec['done'] and not rec['failed'] and s0 and s0.get('passed') and one and rec['graph_check'])
    out['units'][u] = rec
out['ALL_PASS'] = rc == 0 and all(v['passed'] for v in out['units'].values())
(G / 'SMOKE_R49.json').write_text(json.dumps(out, indent=1, default=str))
agg = rj(W / 'GPU_SMOKE_R49.json') or {}; agg[out['job']] = out; (W / 'GPU_SMOKE_R49.json').write_text(json.dumps(agg, indent=1, default=str))
print('R49_SMOKE ALL_PASS', out['ALL_PASS'], json.dumps({u: v['passed'] for u, v in out['units'].items()}))
