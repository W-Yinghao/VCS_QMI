#!/usr/bin/env python3
"""Training-side process for one NR3-B unit (FIT cache / observed / NR1 calibration (binding + bin labels) / prefix / tables)."""
import argparse, sys, time
from pathlib import Path
import torch
from nr1.io import read_json, write_json
from nr3 import train_prefix


def main():
    ap = argparse.ArgumentParser()
    for k in ('config', 'cache', 'observed', 'calibration', 'prefix', 'tables', 'out'): ap.add_argument('--'+k, required=True)
    a = ap.parse_args(); t0 = time.perf_counter()
    train_prefix.run(read_json(a.config), a.cache, a.observed, a.calibration, a.prefix, a.tables, a.out, 'cuda')
    out = Path(a.out); d = read_json(out/'DONE.json'); d['wrapper_seconds'] = time.perf_counter() - t0
    d['peak_gpu_mem_bytes'] = int(torch.cuda.max_memory_allocated()); d['gpu'] = torch.cuda.get_device_name()
    write_json(out/'TRAIN_COST.json', d)


if __name__ == '__main__':
    main()
