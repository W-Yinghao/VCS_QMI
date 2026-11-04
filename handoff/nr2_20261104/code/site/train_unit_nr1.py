#!/usr/bin/env python3
"""Training-side process for one NR1 unit: light step-0/gradient check, then the package trainer.
Receives FIT cache / observed / calibration only (no DEV cache, no truth path)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from nr1.io import FeatureBank, load_observed, read_json, write_json
from nr1.models import DirectScorer
from nr1.objectives import loss_value
from nr1 import train as nr1_train


def light_check(cfg, cache, observed, calibration, out, device):
    bank = FeatureBank(cache, 'fit', verify_hashes=False)  # trainer re-verifies every hash right after
    im, tx = load_observed(observed, bank); cal = read_json(calibration)
    m = DirectScorer(**bank.dims, architecture=cfg['architecture'], a0=cal['a0'], b0=cal['b0'], seed=int(cfg['seed'])).to(device)
    r = np.random.default_rng(123).integers(0, len(im), size=cfg['batch_size'])
    d = bank.tensors(im[r], tx[r], device, local=cfg['architecture'] == 'GL')
    g = m.matrix(**d, pair_chunk=cfg['pair_chunk'])
    x = torch.nn.functional.normalize(d['x'].float(), dim=-1); y = torch.nn.functional.normalize(d['y'].float(), dim=-1)
    ref = cal['a0'] * (x @ y.T) + cal['b0']
    err = float((g - ref).abs().max())
    loss = loss_value(cfg['objective'], g, torch.as_tensor(im[r], device=device)); loss.backward()
    grads = {n: float(p.grad.abs().max()) if p.grad is not None else None for n, p in m.named_parameters()}
    finite = all(v is not None and np.isfinite(v) for v in grads.values())
    ok = err <= 1e-4 and finite and grads['global_residual.4.weight'] > 0 and np.isfinite(float(loss))
    rec = dict(ok=bool(ok), step0_score_vs_affine_cosine_max_abs=err, loss=float(loss), finite_grads=finite,
               grad_max_abs=grads, note='tolerance 1e-4 on |a0*cos+b0| scale: matrix() vs matmul reference in fp32')
    write_json(Path(out)/'STEP0_GRAD_CHECK.json', rec)
    if not ok: raise RuntimeError(f'STEP0_GRAD_CHECK failed: {rec}')


def main():
    ap = argparse.ArgumentParser()
    for k in ('config', 'cache', 'observed', 'calibration', 'out'): ap.add_argument('--'+k, required=True)
    a = ap.parse_args(); cfg = read_json(a.config); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if not (out/'STEP0_GRAD_CHECK.json').exists():
        light_check(cfg, a.cache, a.observed, a.calibration, out, 'cuda')
    nr1_train.run(cfg, a.cache, a.observed, a.calibration, out, 'cuda')
    d = read_json(out/'DONE.json'); d['wrapper_seconds'] = time.perf_counter() - t0
    d['peak_gpu_mem_bytes'] = int(torch.cuda.max_memory_allocated()); d['gpu'] = torch.cuda.get_device_name()
    write_json(out/'TRAIN_COST.json', d)


if __name__ == '__main__':
    main()
