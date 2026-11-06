#!/usr/bin/env python3
"""NR6A_CLOSEOUT_READOUT_SUPPLEMENT_V1 (package VCS_NR6A_Review_Action_20261106), read-only, 0 fits, CPU float64.
Part 1 (36 readouts): 18 TAIL stage-4000 endpoints {Flickr60, COCO60, CC152K_RECOVERED} x J/BCE/NCE x suffix seed {2002, 3003}
  (Flickr60/CC s2002 = NR5-A; COCO60 s2002 and all s3003 = NR6-A) x {E1G0, E0G1}; same definitions as NR5 D0 (nr5/d0.py helpers):
  base Top100 of the frozen DEV cache (ID-hash ties), all DEV queries, float64 pair scores, ranks beyond 100 keep base order.
  E1 = the endpoint evaluator's FEATURES.npz (sha checked vs its PREDICTIONS_SHA256.json); E0 = frozen DEV cache; G0 = the condition's
  folded NR3 NCE prefix head resolved via PREFIX_AND_REUSE_RECEIPT_NR4.json (one hash per condition); G1 = endpoint head.
  E1G1 / E0G0 are reused by identity (stored eval_012000 / eval_008000 predictions) and recomputed here only as a consistency check;
  E1COS on the same Top100 is computed alongside. Scores/orders/hashes are written before the evaluator-only DEV positives are read.
Part 2 (NR6-A 24 units): stage curves, costs, per-query endpoint ranks and paired TAIL-FROZEN R@1 net / grouped bootstrap, from the
  existing artifacts only; anything not saved = NA."""
import csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from nr1.io import FeatureBank, read_json, write_json, sha256
from nr1.models import DirectScorer
from nr1.evaluate import order_rows, ranks_from_orders, report_metrics
from nr5.d0 import rerank, topk_scores, TOPK
from nr5.paths import paths
sys.path.insert(0, '/home/infres/yinwang/CS_QMI/handoff/nr6p_20261105/code')
from nr6p.d0p import groups, boot
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr6a_20261105/readout_p1106'); OUTP = Path('/projects/EEG-foundation-model/yinghao/outputs_nr6/NR6A_20261105/readout_p1106')
O5 = Path('/projects/EEG-foundation-model/yinghao/outputs_nr5/NR5_20261104/runs'); O6 = Path('/projects/EEG-foundation-model/yinghao/outputs_nr6/NR6A_20261105/runs')
REC4 = read_json('/home/infres/yinwang/CS_QMI/handoff/nr4_20261104/PREFIX_AND_REUSE_RECEIPT_NR4.json'); CODE_SHA = sha256(__file__)
OBJS = ('J', 'BCE', 'NCE'); KEYS = ['i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10', 'rSum']


def state_sha(sd):
    h = hashlib.sha256()
    for k in sorted(sd): h.update(k.encode()); h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def src(c, o, s):
    return O5/f'NR5_A_{c}_TAIL_LORA8_{o}_t2002' if (s == 2002 and c != 'COCO60') else O6/f'NR6A_{c}_TAIL_LORA8_{o}_t{s}'


def part1():
    rows, atoms = [], []
    for c in ('Flickr60', 'COCO60', 'CC152K_RECOVERED'):
        P = paths(c); db = FeatureBank(P['dev'], 'dev', verify_hashes=False)
        E0x = torch.from_numpy(np.array(db.x, copy=True)).double(); E0y = torch.from_numpy(np.array(db.y, copy=True)).double()
        base = (F.normalize(E0x, dim=1) @ F.normalize(E0y, dim=1).T).numpy(); boi, bot = order_rows(base, db.text_ids), order_rows(base.T, db.image_ids)
        cand_sha = hashlib.sha256(boi[:, :TOPK].tobytes() + bot[:, :TOPK].tobytes()).hexdigest()
        pre_path = REC4['prefixes'][c]['path']; assert sha256(pre_path) == REC4['prefixes'][c]['sha256']
        pre = torch.load(pre_path, map_location='cpu', weights_only=False); G0 = DirectScorer(**pre['binding']['model_config']); G0.load_state_dict(pre['model']); G0 = G0.double().eval()
        g0sha = state_sha(pre['model']); truth = read_json(P['truth']); Pos = truth['positives']
        bi, bt = ranks_from_orders(boi, bot, db.image_ids, db.text_ids, Pos)
        for s in (2002, 3003):
            for o in OBJS:
                run = src(c, o, s); ck = torch.load(run/'step_012000.pt', map_location='cpu', weights_only=False)
                if not (run/'eval_012000'/'FEATURES.npz').exists():
                    for ro in ('E1G0', 'E0G1'): rows.append(dict(condition=c, seed=s, objective=o, readout=ro, status='BLOCKED_ARTIFACT')); continue
                ps = read_json(run/'eval_012000'/'PREDICTIONS_SHA256.json'); fsha = sha256(run/'eval_012000'/'FEATURES.npz'); assert fsha == ps['features_sha256']
                fz = np.load(run/'eval_012000'/'FEATURES.npz'); E1x = torch.from_numpy(fz['image']).double(); E1y = torch.from_numpy(fz['text']).double()
                G1 = DirectScorer(**ck['binding']['model_config']); G1.load_state_dict(ck['model']); G1 = G1.double().eval()
                cosf = lambda x, y: (F.normalize(x, dim=-1)*F.normalize(y, dim=-1)).sum(-1)
                comb = {}
                for name, fn, U, V, hs, feat in (('E1G0', G0.pairs, E1x, E1y, g0sha, fsha), ('E0G1', G1.pairs, E0x, E0y, state_sha(ck['model']), sha256(db.root/'manifest.json')),
                                                 ('E1G1_recheck', G1.pairs, E1x, E1y, state_sha(ck['model']), fsha), ('E0G0_recheck', G0.pairs, E0x, E0y, g0sha, sha256(db.root/'manifest.json')),
                                                 ('E1COS', cosf, E1x, E1y, None, fsha)):
                    t = time.perf_counter()
                    with torch.no_grad(): si, st = topk_scores(fn, U, V, boi, bot)
                    ni, nt = rerank(boi, bot, db.text_ids, db.image_ids, si, st); comb[name] = (si, st, ni, nt, hs, feat, time.perf_counter() - t)
                tag = f'{c}_{o}_t{s}'
                np.savez_compressed(OUTP/f'SCORES_{tag}.npz', **{f'{k}_{x}': v[i] for k, v in comb.items() for i, x in enumerate(('si', 'st', 'ni', 'nt'))})
                sc_sha = sha256(OUTP/f'SCORES_{tag}.npz')
                r4 = np.load(run/'eval_012000'/'PREDICTIONS.npz'); r0 = np.load(run/'eval_008000'/'PREDICTIONS.npz')
                checks = dict(E1G1_recheck_order_equals_stored=bool(np.array_equal(r4['i_order'], comb['E1G1_recheck'][2]) and np.array_equal(r4['t_order'], comb['E1G1_recheck'][3])),
                              E0G0_recheck_order_equals_stored_stage0=bool(np.array_equal(r0['i_order'], comb['E0G0_recheck'][2]) and np.array_equal(r0['t_order'], comb['E0G0_recheck'][3])))
                # ---- labels only after scores are written and hashed
                stored = read_json(run/'eval_012000'/'METRICS.json'); stored0 = read_json(run/'eval_008000'/'METRICS.json')
                for name, (si, st, ni, nt, hs, feat, sec) in comb.items():
                    ri, rt = ranks_from_orders(ni, nt, db.image_ids, db.text_ids, Pos); m = report_metrics(ri, rt)
                    a = dict(condition=c, seed=s, objective=o, readout=name, source_unit=run.name, status='COMPUTED', checkpoint_sha256=sha256(run/'step_012000.pt'),
                             prefix_sha256=REC4['prefixes'][c]['sha256'], encoder_state_sha256=state_sha(ck['adapter']) if name.startswith('E1') else 'FROZEN_SIGLIP2',
                             head_state_sha256=hs, feature_manifest_sha256=feat, candidate_sha256=cand_sha, evaluator_code_sha256=CODE_SHA, scores_file_sha256=sc_sha, seconds=sec,
                             i2t_rescued_vs_base=int(((bi > 1) & (ri == 1)).sum()), i2t_lost_vs_base=int(((bi == 1) & (ri > 1)).sum()),
                             t2i_rescued_vs_base=int(((bt > 1) & (rt == 1)).sum()), t2i_lost_vs_base=int(((bt == 1) & (rt > 1)).sum()), **m)
                    atoms.append(a)
                    if name in ('E1G0', 'E0G1', 'E1COS'): rows.append(a)
                rows.append(dict(condition=c, seed=s, objective=o, readout='E1G1', status='REUSED_BY_IDENTITY', source_unit=run.name, checkpoint_sha256=stored['checkpoint_sha256'], **stored['metrics']))
                rows.append(dict(condition=c, seed=s, objective=o, readout='E0G0', status='REUSED_BY_IDENTITY', source_unit=run.name, checkpoint_sha256=stored0['checkpoint_sha256'], **stored0['metrics']))
                rows.append(dict(condition=c, seed=s, objective=o, readout='E1COS_full_gallery', status='REUSED_BY_IDENTITY', source_unit=run.name, **stored['student_cosine_full']))
                rows[-1].update(checks=json.dumps(checks)); print(tag, {r['readout']: round(r.get('rSum', float('nan')), 2) for r in rows[-6:]}, checks, flush=True)
    return rows, atoms


def part2():
    curves, costs, rk, paired = [], [], [], []
    for c in ('Flickr60', 'CC152K_RECOVERED', 'COCO60'):
        P = paths(c); db = FeatureBank(P['dev'], 'dev', verify_hashes=False); gi, gt = groups(db, read_json(P['truth'])['positives'])
        for s in ((2002, 3003) if c == 'COCO60' else (3003,)):
            R = {}
            for sc in ('FROZEN', 'TAIL_LORA8'):
                for o in OBJS:
                    u = f'NR6A_{c}_{sc}_{o}_t{s}'; d = O6/u
                    for st in (0, 250, 500, 1000, 2000, 4000):
                        m = d/f'eval_{8000+st:06d}'/'METRICS.json'
                        if not m.exists(): curves.append(dict(unit=u, stage_step=st, status='NA')); continue
                        M = read_json(m); curves.append(dict(unit=u, condition=c, scope=sc, objective=o, seed=s, stage_step=st, **M['metrics'], student_cos_full_rSum=M['student_cosine_full']['rSum'],
                                                             current_score_full_rSum=M.get('current_score_full_gallery', {}).get('metrics', {}).get('rSum', 'NA'),
                                                             i2t_rescued=M['i2t']['rescued'], i2t_lost=M['i2t']['lost'], t2i_rescued=M['t2i']['rescued'], t2i_lost=M['t2i']['lost']))
                    tc = read_json(d/'TRAIN_COST.json') if (d/'TRAIN_COST.json').exists() else {}
                    ev = [read_json(p) for p in sorted(d.glob('eval_*/METRICS.json'))]
                    costs.append(dict(unit=u, gpu=tc.get('gpu', 'NA'), attempts=len((d/'ATTEMPTS.txt').read_text().strip().splitlines()) if (d/'ATTEMPTS.txt').exists() else 'NA',
                                      synchronized_update_loop_seconds=tc.get('synchronized_update_loop_seconds', 'NA'), wrapper_seconds=tc.get('wrapper_seconds', 'NA'),
                                      peak_gpu_mem_bytes=tc.get('peak_gpu_mem_bytes', 'NA'), pair_scores_train=tc.get('pair_forwards_this_attempt', 'NA'),
                                      tail_item_forwards_train=json.dumps(tc.get('raw_encoder_item_forwards')) if tc.get('raw_encoder_item_forwards') else 'NA',
                                      eval_feature_seconds=sum(e['cost']['feature_seconds'] for e in ev) if ev else 'NA', eval_score_seconds=sum(e['cost']['score_seconds'] for e in ev) if ev else 'NA',
                                      hidden_cache_build_seconds='NA (shared NR4 cache; see NR4 RESOURCE_LEDGER)', failed_attempt_cost='NA (none recorded)'))
                    z = np.load(d/'eval_012000'/'RANKS_EVAL_ONLY.npz'); R[(sc, o)] = (z['i_ranks'], z['t_ranks'])
                    for qi, r in enumerate(z['i_ranks']): rk.append(dict(unit=u, direction='i2t', query_index=qi, query_id=db.image_ids[qi], rank=int(r), base_rank=int(z['base_i_ranks'][qi])))
                    for qi, r in enumerate(z['t_ranks']): rk.append(dict(unit=u, direction='t2i', query_index=qi, query_id=db.text_ids[qi], rank=int(r), base_rank=int(z['base_t_ranks'][qi])))
            for o in OBJS:
                b = boot(gi, gt, R[('TAIL_LORA8', o)], R[('FROZEN', o)]); paired.append(dict(condition=c, seed=s, objective=o, pair='TAIL_minus_FROZEN', **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in b.items()}))
            for sc in ('FROZEN', 'TAIL_LORA8'):
                for x, y in (('J', 'BCE'), ('J', 'NCE')):
                    b = boot(gi, gt, R[(sc, x)], R[(sc, y)]); paired.append(dict(condition=c, seed=s, objective=f'{x}-{y}', pair=f'{sc}_{x}_minus_{y}', **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in b.items()}))
    return curves, costs, rk, paired


def wcsv(p, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval='NA'); w.writeheader(); w.writerows(rows)


def main():
    H.mkdir(parents=True, exist_ok=True); OUTP.mkdir(parents=True, exist_ok=True); torch.set_num_threads(8)
    rows, atoms = part1(); wcsv(H/'READOUT_36_AND_REUSED.csv', rows); wcsv(H/'READOUT_ATOMS.csv', atoms)
    curves, costs, rk, paired = part2(); wcsv(H/'NR6A_CURVES.csv', curves); wcsv(H/'NR6A_COSTS.csv', costs); wcsv(H/'NR6A_PAIRED_BOOTSTRAP.csv', paired)
    wcsv(OUTP/'NR6A_PER_QUERY_RANKS_ENDPOINT.csv', rk); write_json(H/'PER_QUERY_RANKS_POINTER.json', dict(path=str(OUTP/'NR6A_PER_QUERY_RANKS_ENDPOINT.csv'), sha256=sha256(OUTP/'NR6A_PER_QUERY_RANKS_ENDPOINT.csv'), rows=len(rk),
              per_stage='per-stage per-query ranks are the evaluator files eval_*/RANKS_EVAL_ONLY.npz in each run directory (not copied)'))
    # results table
    L = ['# RESULTS_NR6A_READOUT (numbers only; NR6A_CLOSEOUT_READOUT_SUPPLEMENT_V1; DEV all queries, frozen base Top100, float64; 0 fits; TEST not read)', '',
         '## Same-candidate readouts at stage 4000 (rSum)', '', '| cond | obj | seed | E0G0 | E1G1 | E1G0 | E0G1 | E1COS (Top100) | E1COS full |', '|---|---|---|---|---|---|---|---|---|']
    idx = {(r['condition'], r['objective'], r['seed'], r['readout']): r for r in rows}
    g = lambda c, o, s, k: idx.get((c, o, s, k), {}).get('rSum')
    f = lambda v: 'NA' if v is None else f'{v:.2f}'
    for c in ('Flickr60', 'COCO60', 'CC152K_RECOVERED'):
        for o in OBJS:
            for s in (2002, 3003):
                L.append(f'| {c} | {o} | {s} | ' + ' | '.join(f(g(c, o, s, k)) for k in ('E0G0', 'E1G1', 'E1G0', 'E0G1', 'E1COS', 'E1COS_full_gallery')) + ' |')
    L += ['', '## J − NCE by readout (rSum)', '', '| cond | seed | E1G1 | E1G0 | E0G1 | E1COS |', '|---|---|---|---|---|---|']
    for c in ('Flickr60', 'COCO60', 'CC152K_RECOVERED'):
        for s in (2002, 3003):
            L.append(f'| {c} | {s} | ' + ' | '.join(('NA' if g(c, 'J', s, k) is None else f"{g(c, 'J', s, k) - g(c, 'NCE', s, k):+.2f}") for k in ('E1G1', 'E1G0', 'E0G1', 'E1COS')) + ' |')
    L += ['', '## NR6-A paired TAIL − FROZEN (grouped bootstrap; groups = DEV image + its positive captions)', '', '| cond | seed | obj | rSum diff | 95% CI | i2t R@1 net | t2i R@1 net |', '|---|---|---|---|---|---|---|']
    for p in paired:
        if p['pair'] == 'TAIL_minus_FROZEN':
            L.append(f"| {p['condition']} | {p['seed']} | {p['objective']} | {p['rSum_diff']:+.2f} | {p['ci95']} | {p['i2t_R1_net']:+d} | {p['t2i_R1_net']:+d} |")
    (H/'RESULTS_NR6A_READOUT.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
