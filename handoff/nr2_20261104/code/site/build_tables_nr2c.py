#!/usr/bin/env python3
"""NR2C (COCO): same as build_tables.py with an explicit tie-window margin (COCO has up to 159 identical captions; NR2 default 28 tripped the guard). Build the fixed NR2 hard-proposal tables: frozen base FIT cosine Top100 per query, both directions.
Inputs: the NR1 FIT feature cache ONLY (no observed table, no truth). Shared by all conditions (same content cache)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nr1.io import FeatureBank, sha256, write_json
from nr2.reference import build_base_topk


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cache', required=True); ap.add_argument('--out', required=True)
    ap.add_argument("--k", type=int, default=100); ap.add_argument("--margin", type=int, required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out/'base_top100_fit.npz').exists(): raise FileExistsError('tables exist; refuse to overwrite')
    bank = FeatureBank(a.cache, 'fit')
    t0 = time.perf_counter()
    i2t, si = build_base_topk(bank.x, bank.y, bank.text_ids, k=a.k, margin=a.margin)
    t2i, st = build_base_topk(bank.y, bank.x, bank.image_ids, k=a.k, margin=a.margin)
    sec = time.perf_counter() - t0
    np.savez(out/'base_top100_fit.npz', i2t=i2t, t2i=t2i)
    write_json(out/'TABLES_MANIFEST.json', dict(
        protocol='NR2_BASE_FIT_TOP100_V1', k=a.k, cache_manifest_sha256=sha256(bank.root/'manifest.json'),
        tables_sha256=sha256(out/'base_top100_fit.npz'), shape_i2t=list(i2t.shape), shape_t2i=list(t2i.shape),
        tie_window_margin=a.margin, score='float64 cosine of sample-wise L2 FIT globals; ties by sha256(gallery ID)', inputs='FIT cache only; no observed/truth',
        boundary=dict(i2t=si, t2i=st), seconds=sec))
    print('tables', i2t.shape, t2i.shape, si, st, sec)


if __name__ == '__main__':
    main()
