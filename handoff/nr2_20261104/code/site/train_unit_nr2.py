#!/usr/bin/env python3
"""Training-side process for one NR2 REFERENCE unit (FIT cache / observed / calibration / base tables only)."""
import argparse, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nr1.io import read_json, write_json
from nr2 import train as nr2_train


def main():
    ap = argparse.ArgumentParser()
    for k in ('config', 'cache', 'observed', 'calibration', 'tables', 'out'): ap.add_argument('--'+k, required=True)
    a = ap.parse_args(); t0 = time.perf_counter()
    nr2_train.run(read_json(a.config), a.cache, a.observed, a.calibration, a.tables, a.out, 'cuda')
    out = Path(a.out); d = read_json(out/'DONE.json'); d['wrapper_seconds'] = time.perf_counter() - t0
    d['peak_gpu_mem_bytes'] = int(torch.cuda.max_memory_allocated()); d['gpu'] = torch.cuda.get_device_name()
    write_json(out/'TRAIN_COST.json', d)


if __name__ == '__main__':
    main()
