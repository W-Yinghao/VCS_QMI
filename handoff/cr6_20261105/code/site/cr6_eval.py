"""CR6 DEV evaluator (separate stage from training; DEV positives opened only after predictions are written and hashed).

Primary = NR1_BASE_TOP100_ALL_QUERY_V1 (as CR1 eval_student -> nr1.evaluate, float64): every DEV query, candidates = top100 of
the FROZEN base SigLIP2 cosine from the frozen DEV global cache (ID-hash tie-break), reranked by the current G (state dict loaded
unchanged into the NR1 DirectScorer) on the CURRENT features; ranks beyond top100 keep the base order.
Current features: FROZEN scope = frozen DEV globals; TAIL_LORA8 = raw DEV images/texts (NR4 raw arrays, bound to the same DEV cache
manifest) through SigLIP2 + the checkpoint LoRA, fixed B=32, fp32 (nr4.evaluate path).
Secondary (never used for selection): full-gallery ranking by the current student cosine, and full-gallery ranking by the current G.
Saves six recalls, per-query ranks (RANKS_EVAL_ONLY.npz). No TEST access anywhere.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr6_common as K                                                       # noqa: E402
from nr1.io import FeatureBank                                               # noqa: E402
from nr1.models import DirectScorer                                          # noqa: E402
from nr1.evaluate import order_rows, ranks_from_orders, report_metrics       # noqa: E402
from nr4.evaluate import score_and_order                                     # noqa: E402

_SIGLIP = {}


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def tail_features(cond, ck, bank, device):
    from nr4.tail_encoder import load_siglip, TailEncoder
    rd = K.NR4_RAW/K.DATASET[cond]/'dev'; rm = K.read_json(rd/'RAW_MANIFEST.json')
    if rm['cache_manifest_sha256'] != K.sha256(bank.root/'manifest.json'): raise ValueError('raw DEV arrays not bound to this DEV cache')
    if 'm' not in _SIGLIP: _SIGLIP['m'] = load_siglip(device, lora=True, seed=int(ck['binding']['adapter_seed']))[0]
    m = _SIGLIP['m']
    for n, p in m.named_parameters():
        if p.requires_grad: p.data.copy_(ck['adapter'][n].to(device))
    if not hasattr(tail_features, 'arr'): tail_features.arr = {}
    if cond not in tail_features.arr: tail_features.arr[cond] = (np.load(rd/'pixels_u8.npy', mmap_mode='r'), np.load(rd/'tokens_i32.npy'))
    enc = TailEncoder(m, *tail_features.arr[cond], device)
    sync(); t = time.perf_counter()
    with torch.no_grad():
        u = enc.encode('image', np.arange(len(bank.image_ids)))[0]; v = enc.encode('text', np.arange(len(bank.text_ids)))[0]
    sync()
    return u, v, dict(dev_encode_seconds=time.perf_counter()-t, raw_manifest_sha256=K.sha256(rd/'RAW_MANIFEST.json'))


@torch.no_grad()
def g_full(model, u, v, chunk=65536):
    U, V = u.double(), v.double(); n, m = len(U), len(V); out = torch.empty(n*m, dtype=torch.float64, device=U.device)
    ii = torch.arange(n, device=U.device).repeat_interleave(m); jj = torch.arange(m, device=U.device).repeat(n)
    for k in range(0, n*m, chunk): out[k:k+chunk] = model.pairs(U[ii[k:k+chunk]], V[jj[k:k+chunk]])
    return out.reshape(n, m).cpu().numpy()


def evaluate_ckpt(cond, ckpt, out, device='cuda', topk=100):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'METRICS.json').exists(): return K.read_json(out/'METRICS.json')
    dev, truth_file = K.dev_paths(cond); bank = FeatureBank(dev, 'dev')
    init = K.read_json(K.prep_dir(cond)/'PREP.json')['student_common_initialization']
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    if ck['architecture'] != 'G' or ck['dimension'] != bank.dims['global_dim']: raise ValueError('not a CR G checkpoint')
    cfg = dict(**bank.dims, architecture='G', a0=float(init['a']), b0=float(init['b']), seed=1001)
    model = DirectScorer(**cfg).to(device); model.load_state_dict(ck['model'], strict=True)
    scope = 'TAIL_LORA8' if 'adapter' in ck else 'FROZEN'
    sync(); t0 = time.perf_counter()
    x = torch.from_numpy(np.array(bank.x, copy=True)).to(device); y = torch.from_numpy(np.array(bank.y, copy=True)).to(device)
    if scope == 'FROZEN': u, v, finfo = x, y, {}
    else:
        u, v, finfo = tail_features(cond, ck, bank, device)
        finfo['max_abs_vs_frozen_cache'] = dict(image=float((u-x).abs().max()), text=float((v-y).abs().max()))
    t1 = time.perf_counter()
    base, oi, ot, ni, nt, pairs, z, si, st = score_and_order(bank, u, v, model, topk)
    gm = g_full(model, u, v); gi, gt = order_rows(gm, bank.text_ids), order_rows(gm.T, bank.image_ids)
    sync(); t2 = time.perf_counter()
    np.savez_compressed(out/'PREDICTIONS.npz', i_order=ni, t_order=nt, base_i_order=oi, base_t_order=ot, pairs=pairs, scores=z,
                        student_i_order=si[:, :topk], student_t_order=st[:, :topk], gfull_i_order=gi[:, :topk], gfull_t_order=gt[:, :topk])
    pred_sha = K.sha256(out/'PREDICTIONS.npz')
    di, dt_ = (ni != oi).any(1), (nt != ot).any(1)
    K.write_json(out/'PREDICTIONS_SHA256.json', dict(predictions_sha256=pred_sha, written_before_truth_read=True, score_dtype='float64',
                 order_vs_base=dict(i2t_queries_order_differs=int(di.sum()), t2i_queries_order_differs=int(dt_.sum())), checkpoint_step=int(ck['step']), utc=K.utc()))
    truth = K.read_json(truth_file)
    if set(truth) != {'protocol_id', 'positives'} or truth['protocol_id'] != 'NR1_EVAL_TRUTH_V1': raise ValueError('truth format')
    P = truth['positives']
    bi, bt = ranks_from_orders(oi, ot, bank.image_ids, bank.text_ids, P); ri, rt = ranks_from_orders(ni, nt, bank.image_ids, bank.text_ids, P)
    sri, srt = ranks_from_orders(si, st, bank.image_ids, bank.text_ids, P); gri, grt = ranks_from_orders(gi, gt, bank.image_ids, bank.text_ids, P)
    res = dict(metrics=report_metrics(ri, rt), base=report_metrics(bi, bt), student_cosine_full=report_metrics(sri, srt), g_full_gallery=report_metrics(gri, grt),
               topk=topk, protocol_id='NR1_BASE_TOP100_ALL_QUERY_V1', scope=scope, checkpoint=str(ckpt), checkpoint_sha256=K.sha256(ckpt),
               checkpoint_step=int(ck['step']), n_image_queries=len(ri), n_text_queries=len(rt), test_read=False, score_dtype='float64',
               predictions_sha256=pred_sha, truth_sha256=K.sha256(truth_file), cache_sha256=K.sha256(bank.root/'manifest.json'), features=finfo,
               cost=dict(feature_seconds=t1-t0, score_seconds=t2-t1, pairs_scored=int(len(z)), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'))
    for d, b, r in [('i2t', bi, ri), ('t2i', bt, rt)]:
        res[d] = dict(rescued_top1=int(((b > 1) & (r == 1)).sum()), lost_top1=int(((b == 1) & (r > 1)).sum()), coverage_topk=float(np.mean(b <= topk)))
    res['delta_rSum'] = res['metrics']['rSum'] - res['base']['rSum']
    np.savez_compressed(out/'RANKS_EVAL_ONLY.npz', i_ranks=ri, t_ranks=rt, base_i_ranks=bi, base_t_ranks=bt, student_i_ranks=sri, student_t_ranks=srt, gfull_i_ranks=gri, gfull_t_ranks=grt)
    K.write_json(out/'METRICS.json', res)
    return res


def evaluate_unit(cond, unit_dir, out_dir, steps=K.SAVES, device='cuda'):
    summ = {}
    for s in steps:
        r = evaluate_ckpt(cond, Path(unit_dir)/f'step_{s:06d}.pt', Path(out_dir)/f'eval_{s:06d}', device)
        summ[str(s)] = dict(rSum=r['metrics']['rSum'], metrics=r['metrics'], student_cosine_full_rSum=r['student_cosine_full']['rSum'], g_full_rSum=r['g_full_gallery']['rSum'])
    K.write_json(Path(out_dir)/'EVAL_SUMMARY.json', summ)
    return summ


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--cond', required=True); ap.add_argument('--unit-dir', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); print(evaluate_unit(a.cond, a.unit_dir, a.out))
