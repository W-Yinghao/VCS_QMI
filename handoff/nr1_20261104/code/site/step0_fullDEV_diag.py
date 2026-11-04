#!/usr/bin/env python3
"""Label-free: on the full DEV cache, does a step-0 (zero-residual) scorer reproduce the base order
for every query, in float32 vs float64 evaluator precision?  Reads NO truth file."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nr1.io import FeatureBank, read_json, write_json
from nr1.models import DirectScorer
from nr1.evaluate import score_and_order
O = Path('/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104')
bank = FeatureBank(O/'cache/dev', 'dev'); res = {}
for c in ['Flickr00', 'Flickr60', 'Flickr80']:
    cal = read_json(O/f'calibration/{c}.json')
    for arch in ['G', 'GL']:
        for dt in [torch.float32, torch.float64]:
            m = DirectScorer(**bank.dims, architecture=arch, a0=cal['a0'], b0=cal['b0'], seed=1001).cuda()
            _, oi, ot, ni, nt, ps = score_and_order(bank, m, 'cuda', 100, dtype=dt)
            di, dtt = (ni != oi).any(1), (nt != ot).any(1)
            res[f'{c}_{arch}_{str(dt)[6:]}'] = dict(i2t_queries_differ=int(di.sum()), t2i_queries_differ=int(dtt.sum()),
                                                  i2t_positions_differ=int((ni != oi).sum()), t2i_positions_differ=int((nt != ot).sum()))
            print(c, arch, dt, res[f'{c}_{arch}_{str(dt)[6:]}'], flush=True)
write_json(O/'STEP0_FULLDEV_DIAG.json', dict(note='label-free; base order vs zero-residual scorer order on all DEV queries, top-100 rerank', results=res))
