"""CR1 -> NR1 independent DEV evaluator bridge (separate process from training).

Only the outer checkpoint wrapper is converted: the G state-dict (global_residual.*, log_scale, bias)
is passed unchanged to the NR1 DirectScorer(architecture='G'). Scores are computed and hashed by the
NR1 evaluator BEFORE it opens the evaluator-only DEV positives (NR1_BASE_TOP100_ALL_QUERY_V1, float64,
same base Top-100, all queries, unchanged tail). Training never sees this script's inputs.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/code/src')
from cr1.io import read_json, write_json, sha256          # noqa: E402
from cr1.models import GlobalScorer                       # noqa: E402
from nr1.evaluate import evaluate                         # noqa: E402
from nr1.io import FeatureBank                            # noqa: E402
from nr1.models import DirectScorer                       # noqa: E402

MODEL = 'google/siglip2-base-patch16-224'; REV = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'


def bridge(cr1_ckpt, dims, init, dst):
    ck = torch.load(cr1_ckpt, map_location='cpu', weights_only=False)
    if ck['architecture'] != 'G' or ck['dimension'] != dims['global_dim']: raise ValueError('Not a CR1 G checkpoint')
    cfg = dict(**dims, architecture='G', a0=float(init['a']), b0=float(init['b']), seed=1001)
    out = {'step': int(ck['step']), 'model': ck['model'],
           'binding': {'feature_model_id': MODEL, 'feature_model_revision': REV, 'model_config': cfg,
                       'cr1_source_checkpoint': str(cr1_ckpt), 'cr1_source_sha256': sha256(cr1_ckpt),
                       'bridge': 'outer wrapper only; state_dict unchanged'}}
    DirectScorer(**cfg).load_state_dict(ck['model'], strict=True)  # same keys, strict
    torch.save(out, dst)
    return ck


@torch.no_grad()
def equivalence(ck, cfg, bank, device, n=4096, seed=7):
    """Same pair score: CR1 GlobalScorer vs NR1 DirectScorer (fp32), and NR1 fp64 vs CR1 fp32."""
    r = np.random.default_rng(seed); i = r.integers(0, len(bank.x), n); t = r.integers(0, len(bank.y), n)
    x = torch.from_numpy(np.array(bank.x[i])).to(device); y = torch.from_numpy(np.array(bank.y[t])).to(device)
    a = GlobalScorer(cfg['global_dim']).to(device); a.load_state_dict(ck['model']); a.eval()
    b = DirectScorer(**cfg).to(device); b.load_state_dict(ck['model']); b.eval()
    sa, sb = a.pairs(x, y), b.pairs(x, y)
    b64 = DirectScorer(**cfg).to(device); b64.load_state_dict(ck['model']); b64.to(torch.float64); b64.eval()
    s64 = b64.pairs(x.double(), y.double())
    return {'pairs': n, 'cr1_vs_nr1_fp32_max_abs': float((sa-sb).abs().max()),
            'nr1_fp64_vs_cr1_fp32_max_abs': float((s64-sa.double()).abs().max()),
            'residual_abs_mean': float((sa - (a.log_scale.exp()*(torch.nn.functional.normalize(x, dim=-1)*torch.nn.functional.normalize(y, dim=-1)).sum(-1)+a.bias)).abs().mean())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--student', required=True); ap.add_argument('--dev-cache', required=True)
    ap.add_argument('--truth', required=True); ap.add_argument('--prepared', required=True)
    ap.add_argument('--steps', default='0,250,500,1000,2000,4000,8000'); ap.add_argument('--device', default='cuda')
    ap.add_argument('--no-frozen-check', action='store_true')
    a = ap.parse_args()
    if not a.no_frozen_check:
        sys.path.insert(0, str(Path(__file__).resolve().parent)); from cr1_prod import frozen_check; frozen_check()
    st = Path(a.student); done = read_json(st/'DONE.json') if (st/'DONE.json').exists() else None
    init = read_json(Path(a.prepared)/'PREP.json')['student_common_initialization']
    bank = FeatureBank(a.dev_cache, 'dev'); dims = bank.dims
    if bank.manifest['model_id'] != MODEL or bank.manifest['model_revision'] != REV: raise ValueError('DEV cache model')
    (st/'eval_bridge').mkdir(exist_ok=True); summary = {}
    for s in [int(x) for x in a.steps.split(',')]:
        ev = st/f'eval_{s:06d}'
        if (ev/'METRICS.json').exists(): summary[s] = read_json(ev/'METRICS.json')['metrics']; continue
        dst = st/'eval_bridge'/f'step_{s:06d}.nr1.pt'
        ck = bridge(st/f'step_{s:06d}.pt', dims, init, dst)
        eq = equivalence(ck, dict(**dims, architecture='G', a0=float(init['a']), b0=float(init['b']), seed=1001), bank, a.device)
        t0 = time.perf_counter()
        res = evaluate(a.dev_cache, a.truth, ev, str(dst), a.device, 100, 'float64')
        write_json(ev/'BRIDGE.json', {'cr1_checkpoint_sha256': sha256(st/f'step_{s:06d}.pt'), 'bridged_sha256': sha256(dst),
                                      'equivalence': eq, 'eval_seconds': time.perf_counter()-t0,
                                      'unit_done_sha256': sha256(st/'DONE.json') if done else None})
        summary[s] = res['metrics']; print(s, res['metrics']['rSum'], res['delta_rSum'], eq, flush=True)
    write_json(st/'EVAL_SUMMARY.json', {str(k): v for k, v in summary.items()})


if __name__ == '__main__': main()
