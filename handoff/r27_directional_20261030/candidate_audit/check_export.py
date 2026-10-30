"""Validation of one exported set (features.npz / labels.npz / metadata.json) before the package tool runs.
Usage: check_export.py SET_DIR [--device cuda]   -> SET_DIR/CHECK.json; exit 1 on failure.
 (1) features.npz has no gt_/true_/clean_ keys;
 (2) A_slot from the export equals production I.search: recomputed dot-product top-5 over the exported eligible bank
     (same features, same slots) vs the production slot indices / IDs saved at export time; mismatching rows must be
     float near-ties (|score gap| <= 1e-5);
 (3) shapes and ID ranges; bank slots all carry both IDs;
 (4) package unique_topk: the order-preserving unique IDs of each A_slot row are a prefix of A_unique (near-ties excepted).
"""
import sys, json, argparse
import numpy as np
import torch

PKG = '/home/infres/yinwang/CS_QMI/r27d_server/VCS_Retrieval_R27D_Server_20261030'
sys.path.insert(0, PKG + '/src')
from candidate_sources import unique_topk  # noqa: E402

TIE = 1e-5


def main():
    p = argparse.ArgumentParser(); p.add_argument('set_dir'); p.add_argument('--device', default='cuda')
    a = p.parse_args(); d = a.device
    f = np.load(a.set_dir + '/features.npz', allow_pickle=False)
    lab = np.load(a.set_dir + '/labels.npz', allow_pickle=False)
    meta = json.load(open(a.set_dir + '/metadata.json'))
    res, ok = {}, True

    def chk(name, cond, **info):
        nonlocal ok
        res[name] = dict(ok=bool(cond), **info); ok &= bool(cond)

    chk('no_truth_keys_in_features', not any(k.startswith(('gt_', 'true_', 'clean_')) for k in f.files), keys=sorted(f.files))
    Bq, dq = f['query_img'].shape; N, db = f['bank_img'].shape
    nC, nI = meta['id_ranges']['n_captions'], meta['id_ranges']['n_images']
    chk('shapes', f['query_txt'].shape == (Bq, dq) and f['bank_txt'].shape == (N, db) and dq == db
        and f['bank_caption_ids'].shape == (N,) and f['bank_observed_img_ids'].shape == (N,)
        and f['query_caption_ids'].shape == (Bq,) and f['query_observed_img_ids'].shape == (Bq,)
        and f['A_slot_i2t'].shape == (Bq, 5) and f['A_slot_t2i'].shape == (Bq, 5)
        and lab['gt_i2t'].shape == (Bq, 5) and lab['gt_t2i'].shape == (Bq, 1) and Bq == meta['n_queries'] and N == meta['bank_n'],
        B=int(Bq), N=int(N), d=int(dq))
    ranges = {k: (int(f[k].min()), int(f[k].max())) for k in ('bank_caption_ids', 'query_caption_ids', 'A_slot_i2t')}
    ranges.update({k: (int(f[k].min()), int(f[k].max())) for k in ('bank_observed_img_ids', 'query_observed_img_ids', 'A_slot_t2i')})
    chk('id_ranges', all(0 <= ranges[k][0] and ranges[k][1] < nC for k in ('bank_caption_ids', 'query_caption_ids', 'A_slot_i2t'))
        and all(0 <= ranges[k][0] and ranges[k][1] < nI for k in ('bank_observed_img_ids', 'query_observed_img_ids', 'A_slot_t2i'))
        and int(lab['gt_i2t'].min()) >= 0 and int(lab['gt_i2t'].max()) < nC and int(lab['gt_t2i'].min()) >= 0 and int(lab['gt_t2i'].max()) < nI,
        ranges=ranges)
    chk('unit_norm', max(float(np.abs(np.linalg.norm(f[k], axis=1) - 1).max()) for k in ('query_img', 'query_txt', 'bank_img', 'bank_txt')) < 1e-4)
    slot = f['bank_slot_index']; pos = {int(s): i for i, s in enumerate(slot)}
    for br, qk, bk, idk in (('i2t', 'query_img', 'bank_img', 'bank_caption_ids'), ('t2i', 'query_txt', 'bank_txt', 'bank_observed_img_ids')):
        q = torch.as_tensor(f[qk], device=d); bank = torch.as_tensor(f[bk], device=d)
        sims = q @ bank.T
        top = torch.topk(sims, 5, dim=1)
        ridx = top.indices.cpu().numpy(); rsc = top.values.cpu().numpy()
        prod_slots = f['A_slot_queue_slot_' + br]
        try:
            ppos = np.vectorize(lambda s: pos[int(s)])(prod_slots)
            in_bank = True
        except KeyError:
            ppos, in_bank = None, False
        if not in_bank:
            chk(f'{br}_production_slots_in_exported_bank', False); continue
        psc = np.take_along_axis(sims.cpu().numpy(), ppos, 1)
        ids_from_prod_slots = f[idk][ppos]
        same_slot_rows = (np.sort(ridx, 1) == np.sort(ppos, 1)).all(1)
        exact_rows = (ridx == ppos).all(1)
        gap = np.abs(np.sort(rsc, 1) - np.sort(psc, 1)).max(1)
        chk(f'{br}_A_slot_ids_match_production_slots', (ids_from_prod_slots == f['A_slot_' + br]).all())
        chk(f'{br}_recomputed_top5_vs_production', bool((same_slot_rows | (gap <= TIE)).all()),
            rows=int(len(ridx)), rows_same_slot_set=int(same_slot_rows.sum()), rows_same_order=int(exact_rows.sum()),
            rows_differ_near_tie=int((~same_slot_rows & (gap <= TIE)).sum()), max_gap=float(gap.max()))
        U, Us, _ = unique_topk(q, bank, f[idk])
        U = U.cpu().numpy(); Us = Us.cpu().numpy()
        bad, tie = 0, 0
        for r in range(len(U)):
            seen = []
            for x in f['A_slot_' + br][r]:
                if int(x) not in seen:
                    seen.append(int(x))
            if list(U[r, :len(seen)]) != seen:
                # allowed only if scores of the involved ranks are near-equal
                if np.isfinite(Us[r, :len(seen)]).all() and np.ptp(Us[r, :min(len(seen) + 1, 5)]) <= TIE * 10:
                    tie += 1
                else:
                    bad += 1
        chk(f'{br}_A_unique_prefix_equals_dedup_A_slot', bad == 0, rows_bad=bad, rows_near_tie=tie)
    res['all_ok'] = ok
    json.dump(res, open(a.set_dir + '/CHECK.json', 'w'), indent=1)
    print(('CHECK_PASS ' if ok else 'CHECK_FAIL ') + a.set_dir, flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
