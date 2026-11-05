#!/usr/bin/env python3
"""NR5_A1 evaluator = nr5/evaluate.py with ONLY the stage-0 contract replaced by the registered NR5_A1 rule:
stage-0 pair scores within |d| <= 3e-5 + 3e-5|ref| of the reference (A: NR3 eval_008000 scores of the same base-Top100 pairs;
B: a0*cos_base + b0 in float64), and the stage-0 order may differ from the reference order ONLY where the reference scores of the
reordered candidates are equal within that tolerance (each reranked row must be non-increasing in the reference scores up to
2x tolerance). Counts and max ratios are recorded. Everything else unchanged.
Original: NR5 DEV evaluator = NR4 evaluator (byte copy) + head from nr5.heads.make_head (COS/G), B stage-0 contract (order == base order),
and --full-gallery: all-pair current score s at the endpoint (secondary; G in pair chunks, COS by matrix), written before truth.
Original NR4 docstring: NR4 DEV evaluator (separate process from training; reads DEV positives only after scores/orders are written and hashed).
Primary (as nr1.evaluate, protocol NR1_BASE_TOP100_ALL_QUERY_V1): every DEV query, candidates = top100 of the FROZEN base SigLIP2
cosine from the frozen DEV global cache (ID-hash tie-break), reranked by the current s_theta = G(u, v) in float64; ranks beyond
top100 keep the base order.
Current features u, v: FROZEN = cached globals; POST32 = (I+BA) on cached globals; TAIL_LORA8 = raw DEV images/texts (NR4 raw
arrays, hash-bound to the frozen DEV cache manifest) through the SigLIP2 two towers with the checkpoint LoRA, fixed B=32, fp32.
Secondary: full-gallery ranking by the current student cosine cos(u, v) vs the base cosine on the same gallery, and the count of
positives that are outside the frozen base top100 but inside the student-cosine top100 (recoverable space; not used for selection).
Stage-0 contract: the stage-0 order must equal the NR3-B eval_008000 order of the same folded prefix."""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from nr1.io import FeatureBank, read_json, write_json, sha256
from nr1.models import DirectScorer
from nr1.evaluate import order_rows, ranks_from_orders, report_metrics, id_tiebreak
from nr4.train import Post
from nr5.heads import make_head

RAW = Path('/projects/EEG-foundation-model/yinghao/outputs_nr4/NR4_20261104/raw')   # NR4 raw DEV arrays (read-only)
DS = {'Flickr00': 'flickr', 'Flickr60': 'flickr', 'COCO60': 'coco', 'CC152K_RECOVERED': 'cc'}


def current_features(scope, ck, bank, cond, device):
    x = torch.from_numpy(np.array(bank.x, copy=True)).to(device); y = torch.from_numpy(np.array(bank.y, copy=True)).to(device)
    info = dict(scope=scope)
    if scope == 'FROZEN':
        return x, y, info
    if scope == 'POST32':
        post = Post(768, int(ck['binding']['config']['post_rank']), int(ck['binding']['config']['adapter_seed'])).to(device)
        post.load_state_dict(ck['adapter'])
        with torch.no_grad(): return post.img(x), post.txt(y), info
    from nr4.tail_encoder import load_siglip, TailEncoder
    rd = RAW/DS[cond]/'dev'; rm = read_json(rd/'RAW_MANIFEST.json')
    if rm['cache_manifest_sha256'] != sha256(bank.root/'manifest.json'): raise ValueError('raw DEV arrays not bound to this DEV cache')
    m, _ = load_siglip(device, lora=True, seed=int(ck['binding']['config']['adapter_seed']))
    for n, p in m.named_parameters():
        if p.requires_grad: p.data.copy_(ck['adapter'][n].to(device))
    enc = TailEncoder(m, np.load(rd/'pixels_u8.npy', mmap_mode='r'), np.load(rd/'tokens_i32.npy'), device)
    if device.startswith('cuda'): torch.cuda.synchronize()
    t = time.perf_counter()
    u = enc.encode('image', np.arange(len(bank.image_ids)))[0]; v = enc.encode('text', np.arange(len(bank.text_ids)))[0]
    if device.startswith('cuda'): torch.cuda.synchronize()
    info.update(dev_encode_seconds=time.perf_counter() - t, raw_encoder_item_forwards=dict(enc.item_forwards), raw_manifest_sha256=sha256(rd/'RAW_MANIFEST.json'),
                max_abs_vs_frozen_cache=dict(image=float((u - x).abs().max()), text=float((v - y).abs().max())))
    return u, v, info


@torch.no_grad()
def score_and_order(bank, u, v, model, topk=100, pair_chunk=512, dtype=torch.float64):
    x = torch.from_numpy(np.array(bank.x, copy=True)).to(dtype).to(u.device); y = torch.from_numpy(np.array(bank.y, copy=True)).to(dtype).to(u.device)
    base = (F.normalize(x, dim=1) @ F.normalize(y, dim=1).T).cpu().numpy()
    oi, ot = order_rows(base, bank.text_ids), order_rows(base.T, bank.image_ids)
    ki, kt = min(topk, len(y)), min(topk, len(x))
    ia = np.repeat(np.arange(len(x)), ki); ca = oi[:, :ki].reshape(-1)
    ib = ot[:, :kt].reshape(-1); cb = np.repeat(np.arange(len(y)), kt)
    pairs = np.unique(np.concatenate([np.stack([ia, ca], 1), np.stack([ib, cb], 1)]), axis=0)
    codes = pairs[:, 0]*len(y) + pairs[:, 1]
    U, V = u.to(dtype), v.to(dtype); model.eval(); model.to(dtype); out = []
    for s in range(0, len(pairs), pair_chunk):
        b = torch.as_tensor(pairs[s:s+pair_chunk], device=u.device)
        z = model.pairs(U[b[:, 0]], V[b[:, 1]])
        if not bool(torch.isfinite(z).all()): raise FloatingPointError('Nonfinite evaluation score.')
        out.append(z.cpu().numpy())
    z = np.concatenate(out)
    zi = z[np.searchsorted(codes, ia*len(y)+ca)].reshape(len(x), ki); zt = z[np.searchsorted(codes, ib*len(y)+cb)].reshape(len(y), kt)
    newi, newt = oi.copy(), ot.copy(); itk = id_tiebreak(bank.text_ids); itk2 = id_tiebreak(bank.image_ids)
    for i in range(len(x)):
        ids = oi[i, :ki]; newi[i, :ki] = ids[np.lexsort((itk[ids], -zi[i]))]
    for t in range(len(y)):
        ids = ot[t, :kt]; newt[t, :kt] = ids[np.lexsort((itk2[ids], -zt[t]))]
    stu = (F.normalize(U, dim=1) @ F.normalize(V, dim=1).T).cpu().numpy()
    si, st = order_rows(stu, bank.text_ids), order_rows(stu.T, bank.image_ids)
    return base, oi, ot, newi, newt, pairs, z, si, st


def evaluate(cond, checkpoint, cache, truth_file, out, ref_stage0=None, device='cuda', topk=100, stage0_base=False, full_gallery=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'METRICS.json').exists(): raise FileExistsError(f'{out}/METRICS.json exists; refuse to overwrite')
    bank = FeatureBank(cache, 'dev'); ck = torch.load(checkpoint, map_location=device, weights_only=False); scope = ck.get('scope', 'FROZEN')
    if ck['binding'].get('feature_model_id') != bank.manifest.get('model_id') or ck['binding'].get('feature_model_revision') != bank.manifest.get('model_revision'):
        raise ValueError('DEV feature model does not match training model.')
    model = make_head(ck['binding']['model_config']).to(device); model.load_state_dict(ck['model'])
    if device.startswith('cuda'): torch.cuda.synchronize()
    t0 = time.perf_counter(); u, v, finfo = current_features(scope, ck, bank, cond, device)
    if device.startswith('cuda'): torch.cuda.synchronize()
    t1 = time.perf_counter(); base, oi, ot, ni, nt, pairs, z, si, st = score_and_order(bank, u, v, model, topk)
    if device.startswith('cuda'): torch.cuda.synchronize()
    t2 = time.perf_counter()
    full = None
    if full_gallery:
        tf = time.perf_counter()
        with torch.no_grad():
            U, V = u.double(), v.double()
            if getattr(model, 'architecture', None) == 'COS':
                S = model.log_scale.exp() * (F.normalize(U, dim=1) @ F.normalize(V, dim=1).T) + model.bias
            else:
                S = torch.cat([model.matrix(U[i:i+64], V, pair_chunk=65536) for i in range(0, len(U), 64)])
            S = S.cpu().numpy()
        from nr1.evaluate import order_rows as _orw
        fi_, ft_ = _orw(S, bank.text_ids), _orw(S.T, bank.image_ids)
        if device.startswith('cuda'): torch.cuda.synchronize()
        full = dict(i_order=fi_, t_order=ft_, seconds=time.perf_counter() - tf, pairs=int(S.size)); del S
    fu, fv = u.float().cpu().numpy(), v.float().cpu().numpy()
    np.savez_compressed(out/'PREDICTIONS.npz', i_order=ni, t_order=nt, base_i_order=oi, base_t_order=ot, pairs=pairs, scores=z,
                        student_i_order=si[:, :topk], student_t_order=st[:, :topk],
                        **({'fullscore_i_order_top100': full['i_order'][:, :topk], 'fullscore_t_order_top100': full['t_order'][:, :topk]} if full else {}))
    np.savez_compressed(out/'FEATURES.npz', image=fu, text=fv)
    pred_sha = sha256(out/'PREDICTIONS.npz')
    di = (ni != oi).any(1); dt_ = (nt != ot).any(1)
    order_vs_base = dict(i2t_queries_order_differs=int(di.sum()), t2i_queries_order_differs=int(dt_.sum()))
    stage0 = None
    if int(ck['stage_step']) == 0 and ref_stage0:
        r = np.load(ref_stage0)
        stage0 = dict(reference=str(ref_stage0), i_order_equal=bool(np.array_equal(r['i_order'], ni)), t_order_equal=bool(np.array_equal(r['t_order'], nt)),
                      scores_max_abs=float(np.abs(r['scores'] - z).max()) if np.array_equal(r['pairs'], pairs) else None)
    write_json(out/'PREDICTIONS_SHA256.json', dict(predictions_sha256=pred_sha, features_sha256=sha256(out/'FEATURES.npz'), written_before_truth_read=True,
                                                   score_dtype='float64', order_vs_base=order_vs_base, checkpoint_step=int(ck['step']), stage_step=int(ck['stage_step']),
                                                   stage0_vs_prefix=stage0, utc_ns=time.time_ns()))
    if int(ck['stage_step']) == 0 and (stage0_base or ref_stage0):
        cal = ck['binding']['model_config']; codes = pairs[:, 0]*len(bank.text_ids) + pairs[:, 1]
        if stage0_base:
            xb = F.normalize(torch.from_numpy(np.array(bank.x, copy=True)).double(), dim=1).numpy(); yb = F.normalize(torch.from_numpy(np.array(bank.y, copy=True)).double(), dim=1).numpy()
            zref = cal['a0']*(xb[pairs[:, 0]]*yb[pairs[:, 1]]).sum(1) + cal['b0']; roi, rot = oi, ot
        else:
            r = np.load(ref_stage0)
            if not np.array_equal(r['pairs'], pairs): raise RuntimeError('stage-0 reference pair set differs')
            zref, roi, rot = r['scores'], r['i_order'], r['t_order']
        ratio = float((np.abs(z - zref) / (3e-5 + 3e-5*np.abs(zref))).max())
        def mono(order, ref_order, rowq, colmap):
            bad, differ = 0, 0
            for q in range(len(order)):
                if np.array_equal(order[q], ref_order[q]): continue
                differ += 1; c = order[q, :topk]; zz = zref[np.searchsorted(codes, colmap(q, c))]
                bad += int((np.diff(zz) > 2*(3e-5 + 3e-5*np.abs(zz[1:]))).any())
            return differ, bad
        di_n, di_bad = mono(ni, roi, None, lambda q, c: q*len(bank.text_ids) + c)
        dt_n, dt_bad = mono(nt, rot, None, lambda q, c: c*len(bank.text_ids) + q)
        x0c, y0c = torch.from_numpy(np.array(bank.x, copy=True)).to(u.device), torch.from_numpy(np.array(bank.y, copy=True)).to(u.device)
        fr = max(float(((u.float() - x0c).abs() / (3e-5 + 3e-5*x0c.abs())).max()), float(((v.float() - y0c).abs() / (3e-5 + 3e-5*y0c.abs())).max()))
        sabs = float(np.abs(z - zref).max()); sbound = 3e-5*(1 + float(np.abs(zref).max()))
        stage0 = dict(rule='NR5_A1', reference='base order / a0 cos_base + b0' if stage0_base else str(ref_stage0),
                      feature_max_tol_ratio=fr, feature_rule='elementwise |d| <= 3e-5 + 3e-5|ref| vs the frozen DEV cache (NR1 package rule)',
                      scores_max_abs=sabs, scores_bound=sbound, scores_rule='max |d| <= 3e-5 (1 + max|ref|) (scale-relative; elementwise ratio reported)', scores_max_elementwise_tol_ratio=ratio,
                      i2t_queries_order_differs=di_n, t2i_queries_order_differs=dt_n, i2t_rows_violating=di_bad, t2i_rows_violating=dt_bad,
                      order_rule='reordered rows must be non-increasing in the reference scores within 2 x (3e-5 + 3e-5|ref|) elementwise',
                      i_order_equal=di_n == 0, t_order_equal=dt_n == 0, passed=bool(fr <= 1.0 and sabs <= sbound and di_bad == 0 and dt_bad == 0))
        write_json(out/'PREDICTIONS_SHA256.json', dict(predictions_sha256=pred_sha, features_sha256=sha256(out/'FEATURES.npz'), written_before_truth_read=True,
                                                       score_dtype='float64', order_vs_base=order_vs_base, checkpoint_step=int(ck['step']), stage_step=int(ck['stage_step']),
                                                       stage0_vs_prefix=stage0, utc_ns=time.time_ns()))
    if stage0 is not None and not stage0.get('passed', stage0['i_order_equal'] and stage0['t_order_equal']):
        raise RuntimeError(f'STAGE0_ORDER_MISMATCH vs NR3 prefix eval: {stage0}')
    truth = read_json(truth_file)
    if set(truth) != {'protocol_id', 'positives'} or truth['protocol_id'] != 'NR1_EVAL_TRUTH_V1':
        raise ValueError('Expected explicit evaluator-only image/text positive pairs.')
    P = truth['positives']
    bi, bt = ranks_from_orders(oi, ot, bank.image_ids, bank.text_ids, P); ri, rt = ranks_from_orders(ni, nt, bank.image_ids, bank.text_ids, P)
    sri, srt = ranks_from_orders(si, st, bank.image_ids, bank.text_ids, P)
    if full: fri, frt = ranks_from_orders(full['i_order'], full['t_order'], bank.image_ids, bank.text_ids, P)
    res = dict(metrics=report_metrics(ri, rt), base=report_metrics(bi, bt), student_cosine_full=report_metrics(sri, srt), topk=topk,
               protocol_id='NR1_BASE_TOP100_ALL_QUERY_V1', scope=scope, cache_sha256=sha256(bank.root/'manifest.json'), checkpoint_sha256=sha256(checkpoint),
               checkpoint_step=int(ck['step']), stage_step=int(ck['stage_step']), n_image_queries=len(ri), n_text_queries=len(rt), test_read=False,
               score_dtype='float64', predictions_sha256=pred_sha, order_vs_base=order_vs_base, stage0_vs_prefix=stage0, truth_sha256=sha256(truth_file),
               features=finfo, cost=dict(feature_seconds=t1 - t0, score_seconds=t2 - t1, pairs_scored=int(len(z)), n_queries=len(ri)+len(rt),
                                         gpu=torch.cuda.get_device_name() if device.startswith('cuda') else None))
    for d, b, r, s in [('i2t', bi, ri, sri), ('t2i', bt, rt, srt)]:
        res[d] = dict(rescued=int(((b > 1) & (r == 1)).sum()), lost=int(((b == 1) & (r > 1)).sum()), coverage_topk=float(np.mean(b <= topk)),
                      student_cos_coverage_topk=float(np.mean(s <= topk)), recoverable_outside_base_topk=int(((b > topk) & (s <= topk)).sum()))
    res['delta_rSum'] = res['metrics']['rSum'] - res['base']['rSum']
    if full: res['current_score_full_gallery'] = dict(metrics=report_metrics(fri, frt), seconds=full['seconds'], pairs=full['pairs'])
    write_json(out/'METRICS.json', res)
    np.savez_compressed(out/'RANKS_EVAL_ONLY.npz', i_ranks=ri, t_ranks=rt, base_i_ranks=bi, base_t_ranks=bt, student_i_ranks=sri, student_t_ranks=srt,
                        **({'fullscore_i_ranks': fri, 'fullscore_t_ranks': frt} if full else {}))
    return res


def main():
    ap = argparse.ArgumentParser()
    for k in ('cond', 'checkpoint', 'cache', 'truth', 'out'): ap.add_argument('--'+k, required=True)
    ap.add_argument('--ref-stage0'); ap.add_argument('--stage0-base', action='store_true'); ap.add_argument('--full-gallery', action='store_true'); a = ap.parse_args()
    r = evaluate(a.cond, a.checkpoint, a.cache, a.truth, a.out, a.ref_stage0, stage0_base=a.stage0_base, full_gallery=a.full_gallery)
    print(json.dumps(dict(rSum=r['metrics']['rSum'], base=r['base']['rSum'], stage=r['stage_step'])))


if __name__ == '__main__':
    main()
