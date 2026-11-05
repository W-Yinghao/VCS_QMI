#!/usr/bin/env python3
"""Training-side process for one NR5 unit (FIT only). Prefix = registered NR3/NR4 folded prefix (A) or none (B)."""
import argparse, time
from pathlib import Path
import torch
from nr1.io import read_json, write_json
from nr5 import train


def main():
    ap = argparse.ArgumentParser()
    for k in ('config', 'cache', 'observed', 'calibration', 'tables', 'out'): ap.add_argument('--'+k, required=True)
    ap.add_argument('--prefix'); ap.add_argument('--hidden'); ap.add_argument('--max-steps', type=int)
    a = ap.parse_args(); t0 = time.perf_counter(); cfg = read_json(a.config)
    if (cfg.get('init') == 'NATIVE') != (a.prefix is None): raise SystemExit('prefix must be given iff not NATIVE')
    if a.prefix and a.prefix != cfg['prefix']: raise SystemExit('prefix differs from the receipt-resolved config path')
    train.run(cfg, a.cache, a.observed, a.calibration, a.prefix, a.tables, a.out, a.hidden, 'cuda', a.max_steps)
    out = Path(a.out)
    if (out/'DONE.json').exists():
        d = read_json(out/'DONE.json'); d['wrapper_seconds'] = time.perf_counter() - t0; d['gpu'] = torch.cuda.get_device_name()
        write_json(out/'TRAIN_COST.json', d)


if __name__ == '__main__':
    main()
