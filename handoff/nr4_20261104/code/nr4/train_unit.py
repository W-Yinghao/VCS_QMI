#!/usr/bin/env python3
"""Training-side process for one NR4 unit (FIT only)."""
import argparse, time
from pathlib import Path
import torch
from nr1.io import read_json, write_json
from nr4 import train


def main():
    ap = argparse.ArgumentParser()
    for k in ('config', 'cache', 'observed', 'calibration', 'prefix', 'tables', 'out'): ap.add_argument('--'+k, required=True)
    ap.add_argument('--hidden'); ap.add_argument('--max-steps', type=int)
    a = ap.parse_args(); t0 = time.perf_counter()
    train.run(read_json(a.config), a.cache, a.observed, a.calibration, a.prefix, a.tables, a.out, a.hidden, 'cuda', a.max_steps)
    out = Path(a.out)
    if (out/'DONE.json').exists():
        d = read_json(out/'DONE.json'); d['wrapper_seconds'] = time.perf_counter() - t0; d['gpu'] = torch.cuda.get_device_name()
        write_json(out/'TRAIN_COST.json', d)


if __name__ == '__main__':
    main()
