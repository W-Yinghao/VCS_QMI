"""Aggregate candidate-source audit outputs into numbers-only tables (no interpretation).
Usage: aggregate.py OUT_ROOT [OUT_PREFIX]
  Reads OUT_ROOT/<cell>/<label>/peer_<T>/{metadata.json, CHECK.json, labels.npz, features.npz,
  normal/{SUMMARY.json,candidate_ids.npz}, exclude_same_source/{...}} and writes OUT_PREFIX.json / OUT_PREFIX.md
  (default: this directory / CANDIDATE_AUDIT_RESULTS). Truth (labels.npz) is read only here, after all candidates are frozen.
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATES = [('f40_s1001', 'prefix_reconstructed'), ('f40_s1001', 'a1_iface_last'),
          ('f60_s1001', 'prefix_reconstructed'), ('f60_s1001', 'a1_iface_last'),
          ('f40_s1001', 'bn_cross_last'), ('f40_s1001', 'cn_mix_last'), ('f60_s1001', 'bn_cross_last'), ('f60_s1001', 'cn_mix_last')]  # R27-CS final states (added 2026-10-31)
MODES = ('normal', 'exclude_same_source')
METH = ('A_slot_reference', 'A_unique', 'B_direct', 'C_mix')


def hits(pred, gt):
    out = np.zeros(len(pred), dtype=bool); best = np.zeros(len(pred), dtype=np.int64)
    for r, (pp, tt) in enumerate(zip(pred, gt)):
        t = {int(x) for x in tt if x >= 0}
        pos = [i + 1 for i, x in enumerate(int(y) for y in pp if y >= 0) if x in t]
        out[r] = bool(pos); best[r] = min(pos) if pos else 0
    return out, best


def set_stats(sd):
    meta = json.load(open(os.path.join(sd, 'metadata.json')))
    chk = json.load(open(os.path.join(sd, 'CHECK.json')))
    f = np.load(os.path.join(sd, 'features.npz')); lab = np.load(os.path.join(sd, 'labels.npz'))
    cap2obs = dict(zip(f['bank_caption_ids'].tolist(), f['bank_observed_img_ids'].tolist()))
    noisy = lab['noisy_query']
    rec = dict(meta={k: meta[k] for k in ('checkpoint', 'checkpoint_sha256', 'state_epoch', 'queue_reconstructed', 'train_peer',
                                         'search_peer', 'n_queries', 'n_distinct_query_observed_images', 'n_distinct_query_captions',
                                         'bank_n', 'queue_filled', 'batches', 'gpu', 'job', 'cache_age_epochs')},
               check_all_ok=chk['all_ok'], n_noisy_queries=int(noisy.sum()), modes={})
    for mode in MODES:
        md = os.path.join(sd, mode)
        if not os.path.exists(os.path.join(md, 'SUMMARY.json')):
            rec['modes'][mode] = 'MISSING'; continue
        summ = json.load(open(os.path.join(md, 'SUMMARY.json')))['results']
        cid = np.load(os.path.join(md, 'candidate_ids.npz'))
        m = {}
        for br in ('i2t', 't2i'):
            gt = lab['gt_' + br]; qobs = f['query_observed_img_ids']
            reach = cid[br + '_reachable']
            d = dict(tool=summ[br], extra={})
            H = {}
            for meth in METH:
                key = br + '_' + meth
                if key not in cid.files:
                    continue
                pred = cid[key]
                h, best = hits(pred, gt); H[meth] = h
                valid = pred >= 0
                if br == 'i2t':
                    same = np.array([[cap2obs.get(int(x), -2) == qo for x in row if x >= 0] for row, qo in zip(pred, qobs)], dtype=object)
                else:
                    same = np.array([[int(x) == qo for x in row if x >= 0] for row, qo in zip(pred, qobs)], dtype=object)
                n_same = np.array([sum(s) for s in same])
                d['extra'][meth] = dict(
                    best_true_rank_counts={str(r): int((best == r).sum()) for r in range(0, 6)},
                    hit_n_noisy_queries=int((h & noisy).sum()), hit_n_clean_queries=int((h & ~noisy).sum()),
                    reachable_n_noisy=int((reach & noisy).sum()), reachable_n_clean=int((reach & ~noisy).sum()),
                    mean_valid_candidates=float(valid.sum(1).mean()), mean_unique_ids_per_row=float(np.mean([len(set(int(x) for x in r if x >= 0)) for r in pred])),
                    mean_candidates_with_query_observed_source=float(n_same.mean()),
                    rows_any_candidate_with_query_observed_source=int((n_same > 0).sum()))
            if 'A_unique' in H and 'B_direct' in H:
                a, b = H['A_unique'], H['B_direct']
                d['paired_A_vs_B'] = dict(both=int((a & b).sum()), A_only=int((a & ~b).sum()), B_only=int((~a & b).sum()), neither=int((~a & ~b).sum()))
                c = H['C_mix']
                d['paired_C_vs_A'] = dict(both=int((c & a).sum()), C_only=int((c & ~a).sum()), A_only=int((~c & a).sum()), neither=int((~c & ~a).sum()))
            if 'A_slot_reference' in H:
                s = H['A_slot_reference']; a = H['A_unique']
                d['paired_Aslot_vs_Aunique'] = dict(both=int((s & a).sum()), slot_only=int((s & ~a).sum()), unique_only=int((~s & a).sum()), neither=int((~s & ~a).sum()))
                if not meta['queue_reconstructed']:
                    age = meta['next_epoch'] - f['A_slot_write_epoch_' + br]
                    d['extra']['A_slot_reference']['slot_cache_age_epochs'] = dict(mean=float(age.mean()), min=int(age.min()), max=int(age.max()))
            m[br] = d
        rec['modes'][mode] = m
    return rec, f, lab


def examples(sd, f, lab, n=5):
    """Pre-fixed identity examples: rows 0..n-1 (batch 0 of the training peer A set); IDs only."""
    cid = np.load(os.path.join(sd, 'normal', 'candidate_ids.npz'))
    out = []
    for r in range(n):
        e = dict(row=r, query_caption_id=int(f['query_caption_ids'][r]), query_observed_img_id=int(f['query_observed_img_ids'][r]),
                 noisy_query=bool(lab['noisy_query'][r]))
        for br in ('i2t', 't2i'):
            e[br] = dict(gt=[int(x) for x in lab['gt_' + br][r]], reachable=bool(cid[br + '_reachable'][r]),
                         **{m: [int(x) for x in cid[br + '_' + m][r]] for m in METH if br + '_' + m in cid.files})
        out.append(e)
    return out


def main():
    root = sys.argv[1]
    pre = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'CANDIDATE_AUDIT_RESULTS')
    res, missing, ex = {}, [], {}
    for cell, label in STATES:
        for peer in 'AB':
            sd = os.path.join(root, cell, label, f'peer_{peer}')
            key = f'{cell}/{label}/peer_{peer}'
            if not os.path.exists(os.path.join(sd, 'CHECK.json')):
                missing.append(key); continue
            res[key], f, lab = set_stats(sd)
            if peer == 'A' and os.path.exists(os.path.join(sd, 'normal', 'candidate_ids.npz')):
                ex[f'{cell}/{label}'] = examples(sd, f, lab)
    # pooled over the two peers (counts summed; queries are NOT independent images)
    pooled = {}
    for cell, label in STATES:
        ks = [f'{cell}/{label}/peer_{p}' for p in 'AB']
        if not all(k in res for k in ks):
            continue
        pm = {}
        for mode in MODES:
            if any(res[k]['modes'][mode] == 'MISSING' for k in ks):
                continue
            for br in ('i2t', 't2i'):
                for meth in METH:
                    rows = [res[k]['modes'][mode][br]['tool'].get(meth) for k in ks]
                    if any(r is None for r in rows):
                        continue
                    s = {c: sum(r[c] for r in rows) for c in ('n', 'bank_reachable_n', 'bank_absent_n', 'top5_hit_n', 'fewer_than5_rows')}
                    s['top5_hit_rate'] = s['top5_hit_n'] / s['n']
                    s['top5_given_reachable'] = (sum(r['top5_given_reachable'] * r['bank_reachable_n'] for r in rows) / s['bank_reachable_n']
                                                 if s['bank_reachable_n'] else None)
                    s['mean_gt_fraction'] = sum(r['mean_gt_fraction'] * r['n'] for r in rows) / s['n']
                    s['mrr_at5'] = sum(r['mrr_at5'] * r['n'] for r in rows) / s['n']
                    pm[f'{mode}|{br}|{meth}'] = s
        pooled[f'{cell}/{label}'] = pm
    out = dict(split='FIT', k=5, selection_uses_truth=False, note='numbers only; pooled = sum over the two peers',
               states=[f'{c}/{l}' for c, l in STATES], missing=missing, per_set=res, pooled_over_peers=pooled, identity_examples=ex)
    json.dump(out, open(pre + '.json', 'w'), indent=1, default=float)
    L = ['# R27-D candidate-source audit (FIT, K=5) -- numbers only', '',
         'Per set = (condition/checkpoint, training peer T; searching peer = the other). hit = top-5 identity hit; '
         'reach = true counterpart present in the eligible bank (re-computed under exclusion). '
         'Queries per set are 8 FIT batches (not independent images; distinct observed images listed).', '']
    if missing:
        L += ['Missing sets: ' + ', '.join(missing), '']
    L += ['| state | peer | mode | dir | n | distinct imgs | reach | A_slot hit | A_unique hit | B_direct hit | C_mix hit | A_u hit/reach | B hit/reach | C hit/reach | A&B / A-only / B-only / none |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for key, r in res.items():
        st, pk = key.rsplit('/', 1)
        for mode in MODES:
            mm = r['modes'][mode]
            if mm == 'MISSING':
                L.append(f'| {st} | {pk} | {mode} | - | MISSING |'); continue
            for br in ('i2t', 't2i'):
                t = mm[br]['tool']
                g = lambda m, c='top5_hit_n': (t[m][c] if m in t else '-')
                fr = lambda m: (f"{t[m]['top5_given_reachable']:.4f}" if m in t and t[m]['top5_given_reachable'] is not None else '-')
                pab = mm[br].get('paired_A_vs_B', {})
                L.append(f"| {st} | {pk[5:]} | {mode} | {br} | {t['A_unique']['n']} | {r['meta']['n_distinct_query_observed_images']} | "
                         f"{t['A_unique']['bank_reachable_n']} | {g('A_slot_reference')} | {g('A_unique')} | {g('B_direct')} | {g('C_mix')} | "
                         f"{fr('A_unique')} | {fr('B_direct')} | {fr('C_mix')} | "
                         f"{pab.get('both','-')} / {pab.get('A_only','-')} / {pab.get('B_only','-')} / {pab.get('neither','-')} |")
    L += ['', '## Pooled over the two peers (counts summed)', '',
          '| state | mode | dir | method | n | reach | hit | hit/n | hit/reach | mean GT fraction | MRR@5 | rows <5 |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for st, pm in pooled.items():
        for k, s in pm.items():
            mode, br, meth = k.split('|')
            hr = f"{s['top5_given_reachable']:.4f}" if s['top5_given_reachable'] is not None else '-'
            L.append(f"| {st} | {mode} | {br} | {meth} | {s['n']} | {s['bank_reachable_n']} | {s['top5_hit_n']} | {s['top5_hit_rate']:.4f} | {hr} | "
                     f"{s['mean_gt_fraction']:.4f} | {s['mrr_at5']:.4f} | {s['fewer_than5_rows']} |")
    open(pre + '.md', 'w').write('\n'.join(L) + '\n')
    print('wrote', pre + '.json', pre + '.md', 'missing:', missing)


if __name__ == '__main__':
    main()
