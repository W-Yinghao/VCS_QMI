#!/usr/bin/env python3
"""Label-free step-0 order check (fp32 vs fp64) for one condition: --dev-cache --calibration --out."""
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nr1.io import FeatureBank, read_json, write_json
from nr1.models import DirectScorer
from nr1.evaluate import score_and_order
ap = argparse.ArgumentParser(); ap.add_argument('--dev-cache'); ap.add_argument('--calibration'); ap.add_argument('--out'); a = ap.parse_args()
bank = FeatureBank(a.dev_cache, 'dev'); cal = read_json(a.calibration); res = {}
for arch in ['G', 'GL']:
    for dt in [torch.float32, torch.float64]:
        m = DirectScorer(**bank.dims, architecture=arch, a0=cal['a0'], b0=cal['b0'], seed=1001).cuda()
        _, oi, ot, ni, nt, _ = score_and_order(bank, m, 'cuda', 100, dtype=dt)
        res[f'{arch}_{str(dt)[6:]}'] = dict(i2t_queries_differ=int((ni != oi).any(1).sum()), t2i_queries_differ=int((nt != ot).any(1).sum()))
        print(arch, dt, res[f'{arch}_{str(dt)[6:]}'], flush=True)
write_json(a.out, dict(note='label-free; base order vs zero-residual scorer order on all DEV queries', results=res))
