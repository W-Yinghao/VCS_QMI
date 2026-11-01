"""NRCS-v1 results export (SERVER_PROMPT steps 8-10; numbers only, NA where missing; no TEST).
  report.py  ->  <OUT>/summary/{all_seed_results_nrcs.csv, paired_contrasts.csv, learning_curves.csv, completeness.csv,
                                cost.csv, SUMMARY.json} and <W>/RESULT_REPORT_CN.md
The package aggregator (tools/aggregate_results.py) is run separately into <OUT>/summary/package/.
"""
import csv, glob, json, math, os, statistics
from collections import defaultdict

W = '/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101'
O = '/projects/EEG-foundation-model/yinghao/outputs_v9/r32_native_ratio_20261101'
S = f'{O}/summary'
R6 = ('i2t_r1', 'i2t_r5', 'i2t_r10', 't2i_r1', 't2i_r5', 't2i_r10', 'rsum')
NA = 'NA'


def rows():
    with open(f'{W}/package/configs/MATRIX.csv', newline='') as f:
        return list(csv.DictReader(f))


def jl(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return NA
        d = d[k]
    return d


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def msd(v):
    v = [x for x in v if num(x)]
    if not v:
        return NA, NA, 0
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else NA), len(v)


def unit_record(r):
    u, st = r['unit_id'], r['stage']
    rd = f'{O}/runs/{u}'
    rec = {k: r[k] for k in ('unit_id', 'stage', 'arm', 'seed', 'noise_percent', 'balance_weight', 'intra_weight', 'parent_id')}
    if st == 'S0':
        m = jl(f'{rd}/metrics.json')
        rec['status'] = 'COMPLETE' if m and m.get('status') == 'COMPLETE' else ('MISSING' if m is None else m.get('status'))
        if m:
            s = m['selected']
            rec.update(S0_exact_S=s['S'], S0_selected_J=s['J'], S0_posterior_mse=s['posterior_mse'], S0_identity_residual=s['identity_residual'],
                       S0_selected_step=s['step'], S0_last_J=m['last']['J'], S0_last_posterior_mse=m['last']['posterior_mse'],
                       S0_marg_image_l1=g(s, 'marginals', 'image_marginal_l1'), S0_marg_text_l1=g(s, 'marginals', 'text_marginal_l1'),
                       S0_ratio_norm_maxabs=g(s, 'marginals', 'image_conditional_ratio_norm_maxabs'))
        return rec
    if st == 'F1':
        m = jl(f'{rd}/metrics.json')
        rec['status'] = 'COMPLETE' if m and m.get('status') == 'COMPLETE' else 'MISSING'
        if m:
            sel = m['selected']
            hsel = next((h for h in m['history'] if h['step'] == sel['step']), {})
            rec.update(F1_selected_step=sel['step'], F1_select_J=g(hsel, 'select', 'J'),
                       F1_audit_J=sel['audit']['J'], F1_audit_rsum=sel['audit']['retrieval']['rsum'], F1_dev_J=sel['dev']['J'],
                       **{f'F1_dev_{k}': sel['dev']['retrieval'][k] for k in R6},
                       F1_last_dev_rsum=m['last']['dev']['retrieval']['rsum'], F1_last_audit_J=m['last']['audit']['J'],
                       F1_audit_ratio_norm_maxabs=g(sel, 'audit', 'marginals', 'image_conditional_ratio_norm_maxabs'),
                       F1_audit_pool_balance=g(sel, 'audit', 'marginals', 'population_pool_balance'),
                       F1_dev_near_cap_P=g(sel, 'dev', 'ratio_capacity', 'near_cap_fraction_P'), F1_seconds=m.get('elapsed_seconds'))
        return rec
    d = jl(f'{rd}/DONE.json')
    rec['status'] = 'COMPLETE' if d else ('RUNNING_OR_PARTIAL' if os.path.exists(f'{rd}/resume.pt') else 'MISSING')
    if not d:
        return rec
    rec.update(gpu=d['cost']['gpu_model'], seconds=d['cost']['elapsed_seconds_total'], peak_mem_GiB=d['cost']['peak_memory_bytes'] / 2 ** 30,
               attempts=len(d['cost']['attempts']))
    if st == 'W2':
        rec.update(**{f'W2_e5_dev_{k}': d['epoch5']['dev'][k] for k in R6}, W2_e5_select_J=d['epoch5']['select']['J'])
        return rec
    hist = jl(f'{rd}/history.json') or []
    last, sel = d['last'], d['selected']
    rec.update(**{f'E20_dev_{k}': last['dev']['retrieval'][k] for k in R6}, E20_dev_J=last['dev']['J'], E20_audit_J=last['audit']['J'],
               E20_audit_rsum=last['audit']['retrieval']['rsum'], sel_epoch=sel['epoch'], sel_select_J=sel['select_J'],
               sel_dev_rsum=sel['dev']['retrieval']['rsum'], sel_audit_J=sel['audit']['J'], last5_dev_rsum=d['last5_dev_mean']['rsum'],
               E20_audit_ratio_norm_maxabs=g(last, 'audit', 'marginals', 'image_conditional_ratio_norm_maxabs'),
               E20_audit_text_ratio_norm_maxabs=g(last, 'audit', 'marginals', 'text_conditional_ratio_norm_maxabs'),
               E20_audit_pool_balance=g(last, 'audit', 'marginals', 'population_pool_balance'),
               E20_dev_near_cap_P=g(last, 'dev', 'ratio_capacity', 'near_cap_fraction_P'),
               E20_dev_r_over_cap_max=g(last, 'dev', 'ratio_capacity', 'r_over_cap_max'))
    h1 = next((h for h in hist if h['epoch'] == 1), None)
    if h1:
        s0, s1 = h1.get('step0', {}), h1.get('step1', {})
        for tag, s in (('step0', s0), ('step1', s1)):
            dg = s.get('diag', {})
            rec.update({f'{tag}_J': s.get('J', NA), f'{tag}_balance': s.get('balance_raw', NA), f'{tag}_intra': s.get('intra', NA),
                        f'{tag}_grad_main_enc': g(dg, 'main', 'encoder'), f'{tag}_grad_main_head': g(dg, 'main', 'head'),
                        f'{tag}_grad_intra_enc': g(dg, 'intra_weighted', 'encoder'), f'{tag}_grad_bal_head': g(dg, 'balance_weighted', 'head'),
                        f'{tag}_T_sat': dg.get('T_abs_gt_0.99_fraction', NA), f'{tag}_near_cap': dg.get('r_over_K_gt_0.95_fraction', NA),
                        f'{tag}_img_logit_std': g(dg, 'critic', 'image_head_logit_std'), f'{tag}_txt_logit_std': g(dg, 'critic', 'text_head_logit_std'),
                        f'{tag}_img_eff_components': g(dg, 'critic', 'image_mean_per_sample_effective_components'),
                        f'{tag}_img_hard_occupied': g(dg, 'critic', 'image_hard_occupied_components')})
    return rec


CONTRASTS = [  # (name, stage, intra, a_arm/bw, b_arm/bw) on the same noise/seed
    ('ratioJ_minus_cosineJ_intra0', 'E2', [('ratio_J', '1.0', '0.0'), ('cosine_J', '0.0', '0.0')]),
    ('ratioJ_minus_cosineJ_intra0.1', 'E2', [('ratio_J', '1.0', '0.1'), ('cosine_J', '0.0', '0.1')]),
    ('intra_effect_cosineJ', 'E2', [('cosine_J', '0.0', '0.1'), ('cosine_J', '0.0', '0.0')]),
    ('intra_effect_ratioJ', 'E2', [('ratio_J', '1.0', '0.1'), ('ratio_J', '1.0', '0.0')]),
    ('ratioJ_minus_ratioLogistic_intra0.1', 'E2', [('ratio_J', '1.0', '0.1'), ('ratio_logistic', '1.0', '0.1')]),
    ('ratioJ_minus_cosineHinge_intra0.1', 'E2', [('ratio_J', '1.0', '0.1'), ('cosine_hinge', '0.0', '0.1')]),
    ('cosineJ_minus_cosineHinge_intra0.1', 'E2', [('cosine_J', '0.0', '0.1'), ('cosine_hinge', '0.0', '0.1')]),
    ('F1_ratioJ_minus_cosineJ', 'F1', [('ratio_J', '1.0', '0.0'), ('cosine_J', '0.0', '0.0')]),
    ('F1_ratioJ_minus_ratioLogistic', 'F1', [('ratio_J', '1.0', '0.0'), ('ratio_logistic', '1.0', '0.0')]),
    ('F1_ratioJ_balance1_minus_balance0', 'F1', [('ratio_J', '1.0', '0.0'), ('ratio_J', '0.0', '0.0')]),
]
E2_METRICS = ['E20_dev_rsum', 'E20_dev_i2t_r1', 'E20_dev_t2i_r1', 'E20_audit_J', 'E20_dev_J', 'sel_dev_rsum', 'last5_dev_rsum']
F1_METRICS = ['F1_dev_rsum', 'F1_audit_J', 'F1_dev_J', 'F1_select_J']


def main():
    os.makedirs(S, exist_ok=True)
    recs = [unit_record(r) for r in rows()]
    keys = []
    for r in recs:
        keys += [k for k in r if k not in keys]
    with open(f'{S}/all_seed_results_nrcs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys, restval=NA); w.writeheader(); w.writerows(recs)
    idx = {(r['stage'], r['arm'], float(r['balance_weight']), float(r['intra_weight']), r['noise_percent'], r['seed']): r for r in recs}
    out, summ = [], {}
    for name, st, (a, b) in CONTRASTS:
        mets = E2_METRICS if st == 'E2' else F1_METRICS
        for noise in ('0', '60'):
            per = defaultdict(list)
            for seed in ('1001', '1002', '1003'):
                ra = idx.get((st, a[0], float(a[1]), float(a[2]), noise, seed)); rb = idx.get((st, b[0], float(b[1]), float(b[2]), noise, seed))
                row = {'contrast': name, 'stage': st, 'noise_percent': noise, 'seed': seed, 'a': ra and ra['unit_id'], 'b': rb and rb['unit_id']}
                for mtr in mets:
                    va, vb = (ra or {}).get(mtr, NA), (rb or {}).get(mtr, NA)
                    row[mtr] = va - vb if num(va) and num(vb) else NA
                    per[mtr].append(row[mtr])
                out.append(row)
            for mtr in mets:
                m, sd, n = msd(per[mtr])
                summ[f'{name}|n{noise}|{mtr}'] = {'mean': m, 'sample_sd': sd, 'n_seeds': n, 'per_seed': per[mtr]}
    ck = sorted({k for r in out for k in r})
    with open(f'{S}/paired_contrasts.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, ['contrast', 'stage', 'noise_percent', 'seed', 'a', 'b'] + [k for k in ck if k not in ('contrast', 'stage', 'noise_percent', 'seed', 'a', 'b')], restval=NA)
        w.writeheader(); w.writerows(out)
    # learning curves
    lc = []
    for r in rows():
        if r['stage'] not in ('W2', 'E2', 'C2'):
            continue
        for h in jl(f"{O}/runs/{r['unit_id']}/history.json") or []:
            ev = h['eval']
            lc.append({'unit_id': r['unit_id'], 'epoch': h['epoch'], 'select_J': ev['select']['J'], 'audit_J': ev['audit']['J'], 'dev_J': ev['dev']['J'],
                       **{f'dev_{k}': ev['dev']['retrieval'][k] for k in R6}, 'train_mean_J': h.get('train_mean_J', NA),
                       'train_mean_balance': h.get('train_mean_balance_raw', NA), 'train_mean_intra': h.get('train_mean_intra', NA),
                       'epoch_seconds': h.get('epoch_seconds', NA)})
    if lc:
        with open(f'{S}/learning_curves.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, list(lc[0]), restval=NA); w.writeheader(); w.writerows(lc)
    comp = {st: defaultdict(int) for st in ('S0', 'W2', 'F1', 'E2', 'C2')}
    for r in recs:
        comp[r['stage']][r['status']] += 1
    gpu_h = sum(r['seconds'] for r in recs if num(r.get('seconds'))) / 3600
    json.dump({'completeness': {k: dict(v) for k, v in comp.items()}, 'gpu_hours_finished_units_attempts': gpu_h, 'contrasts': summ},
              open(f'{S}/SUMMARY.json', 'w'), indent=1, default=str)
    print(json.dumps({k: dict(v) for k, v in comp.items()}), f'gpu_h={gpu_h:.2f}')


if __name__ == '__main__':
    main()
