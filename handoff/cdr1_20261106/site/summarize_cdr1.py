#!/usr/bin/env python3
"""CDR1 ledgers + RESULTS_CDR1.md. Read-only over outputs; missing items are NA.

Writes to <handoff>/ (or CDR1_TEST_OUT/summary in test mode):
CASE / MODEL / PROJECTION / IDENTIFIABILITY / PREDICTION / DECISION / COST ledgers (CSV+JSON),
D1_TABLE.csv, EXECUTION_STATUS.json, RESULTS_CDR1.md.
"""
import csv, glob, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (H, PKG, OUT, DATA, TEST_MODE, DATASETS, SEEDS, M_SIZES, PIS, TAUS, LAMBDAS, CC_NAME,
                    EPSILONS, dump, load, group_dirs)

DST = (OUT / 'summary') if TEST_MODE else H
CASES = list(csv.DictReader((PKG / 'configs/DATA_CASES.csv').open()))
FITS = list(csv.DictReader((PKG / 'configs/CORE_FITS.csv').open()))
VIEWS = {v['view_id']: v for v in csv.DictReader((PKG / 'configs/PREDICTION_VIEWS.csv').open())}
NA = 'NA'


def jl(p):
    try:
        return load(p)
    except Exception:
        return None


def wcsv(name, rows):
    p = DST / name
    if not rows:
        p.write_text('EMPTY\n'); return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=NA); w.writeheader(); w.writerows(rows)


def fmt(x, n=4):
    if x is None or x == NA or (isinstance(x, float) and not np.isfinite(x)):
        return 'NA'
    return f'{x:.{n}f}' if isinstance(x, (float, np.floating)) else str(x)


def mean_or_na(v):
    v = [x for x in v if x is not None and x != NA]
    return float(np.mean(v)) if v else None


# ------------------------------------------------------------------ CASE
def case_ledger():
    rows = []
    for c in CASES:
        cj = jl(DATA / c['raw_fit_dir'] / 'CASE.json')
        cur = jl(DATA / c['dataset'] / f"d{c['data_seed']}" / 'CURATION_DONE.json')
        gal = jl(DATA / c['dataset'] / f"d{c['data_seed']}" / 'GALLERIES.json') or []
        r = {'case_id': c['case_id'], 'dataset': c['dataset'], 'data_seed': c['data_seed'], 'noise_kind': c['noise_kind'],
             'rho_event': c['rho_event'], 'exact_mixture_assumption': c['exact_mixture_assumption'],
             'status': 'CURATED' if cj else 'MISSING'}
        if cj:
            r.update({'fit_score_sha256': cj['fit_score_sha256'], 'bank_sha256': cj['bank_sha256'],
                      'collision_beta_fit': cj['collision_beta_fit'],
                      'noise_model_valid_by_construction': cj['noise_model_valid_by_construction']})
        if cur:
            for k, v in cur['split_counts'].items():
                r[f'groups_{k}'] = v
        for g in gal:
            if 'id' in g:
                r[f"{g['split']}_M{g['M']}_present/absent"] = f"{g['n_present_groups']}/{g['n_absent_groups']}"
        rows.append(r)
    for s in SEEDS:
        cc = jl(DATA / CC_NAME / f'd{s}' / 'CC_BUILD_DONE.json')
        r = {'case_id': f'{CC_NAME}_transfer_target_d{s}', 'dataset': CC_NAME, 'data_seed': s, 'noise_kind': 'real_unknown',
             'rho_event': 'UNKNOWN (no true rho)', 'status': 'BUILT' if cc else 'MISSING'}
        for g in (cc or {}).get('galleries', []):
            r[f"EVAL_OPEN_M{g['M']}_present/absent"] = f"{g['n_present_groups']}/{g['n_absent_groups']}"
        rows.append(r)
    return rows


# ------------------------------------------------------------------ MODEL
def model_rows():
    rows = []
    def add(path, unit, kind):
        m = jl(path)
        if not m:
            rows.append({'unit_id': unit, 'kind': kind, 'status': 'MISSING'}); return
        d = m['fit_diagnostics']
        rows.append({'unit_id': unit, 'kind': kind, 'objective': m['objective'], 'rho_supplied': m['rho'],
                     'regime': d.get('rho_information_regime'), 'log_a': m['log_a'], 'b': m['b'],
                     'slope_a': float(np.exp(m['log_a'])), 'center_Qfit': m['center'], 'scale_Qfit': m['scale'],
                     'best_fit_objective': d['best_loss'], 'selected_converged': d['selected_converged'],
                     'parameter_at_bound': d['parameter_at_bound'],
                     'starts_converged': sum(t['success'] for t in d['trials']),
                     'max_grad_norm_selected': min(d['trials'], key=lambda t: t['loss'])['grad_norm'],
                     'log_E_Q_fit_r_clean': d.get('log_E_Q_fit_r_clean'), 'wall_seconds': d.get('wall_seconds'),
                     'scores_sha256': d.get('scores_sha256'), 'n_p': d['n_p'], 'n_q': d['n_q'], 'status': 'OK'})
    for f in FITS:
        c = next(x for x in CASES if x['case_id'] == f['case_id'])
        add(group_dirs(c['dataset'], int(c['data_seed']))['models'] / f"{f['unit_id']}.json", f['unit_id'], 'CORE')
    for p in sorted(glob.glob(str(OUT / 'rho' / '*' / '*' / '*' / 'PROF_*.json'))):
        add(p, Path(p).parent.name + '_' + Path(p).stem, 'PROFILE_NEW')
    for p in sorted(glob.glob(str(OUT / 'rho' / '*' / '*' / '*' / 'ANCHOR_EPS0_*.json'))):
        add(p, Path(p).parent.name + '_' + Path(p).stem, 'ANCHOR_EPS0')
    return rows


# ------------------------------------------------------------------ predictions
def parse_metrics(p, scope, target):
    r = jl(p)
    if not r:
        return None
    name = Path(p).name
    view = name.split('__to_')[0]
    v = VIEWS.get(view, {})
    rest = name.split('__to_')[1]
    m = int(rest.split('_M')[1].split('_')[0])
    out = {'scope': scope, 'view_id': view, 'source_case': v.get('case_id', view.rsplit('__', 1)[0]),
           'method': v.get('method', NA), 'extra_clean_information': v.get('extra_clean_information', NA),
           'target': target, 'M': m, 'pi_used': r['pi_used'], 'pi_target': r['pi_target'],
           'n_present': r['n_present'], 'n_absent': r['n_absent'], 'map_action_accuracy': r['map_action_accuracy'],
           'presence_brier': r['presence']['brier'], 'presence_nll': r['presence']['nll'], 'presence_ece15': r['presence']['ece15']}
    for key, pre in (('selected', 'sel'), ('naive_top1_mechanism', 'naivetop1')):
        s = r[key]; cc = s['correctness_calibration']
        out.update({f'{pre}_brier': cc['brier'], f'{pre}_nll': cc['nll'], f'{pre}_ece15': cc['ece15'], f'{pre}_auroc': cc['auroc'],
                    f'{pre}_aurc': s['aurc_tie_grouped_step']})
        for op in s['operating_points']:
            t = int(round(op['threshold'] * 100))
            out.update({f'{pre}_t{t}_coverage': op['coverage'], f'{pre}_t{t}_risk': op['accepted_error_rate'],
                        f'{pre}_t{t}_FAR_absent': op['absent_false_accept_rate'], f'{pre}_t{t}_correct_mass': op['accepted_correct_mass'],
                        f'{pre}_t{t}_cost': op['constant_abstention_cost']})
    out['wrong_pi'] = 'WRONGPI' in name
    return out


def prediction_rows():
    rows = []
    for ds in DATASETS:
        for s in SEEDS:
            for p in glob.glob(str(group_dirs(ds, s)['read'] / '*.metrics.json')):
                r = parse_metrics(p, 'same_domain', ds); r and rows.append(r)
    for s in SEEDS:
        for p in glob.glob(str(OUT / 'transfer' / f'd{s}' / '*_to_*' / '*.metrics.json')):
            tgt = Path(p).parent.name.split('_to_')[1]
            r = parse_metrics(p, 'transfer', tgt); r and rows.append(r)
    return rows


def projection_rows():
    agg = defaultdict(lambda: [[], []])
    for p in glob.glob(str(OUT / 'readouts' / '*' / '*' / '*.npz.json')) + glob.glob(str(OUT / 'transfer' / '*' / '*' / '*.npz.json')):
        r = jl(p); name = Path(p).name; view = name.split('__to_')[0]
        meth = VIEWS.get(view, {}).get('method', NA); case = VIEWS.get(view, {}).get('case_id', NA)
        tgt = name.split('__to_')[1].split('__')[0]
        key = (case, meth, tgt)
        agg[key][0].append(r['post_projection_fraction']); agg[key][1].append(r['background_projection_fraction'])
    rows = [{'source_case': k[0], 'method': k[1], 'target': k[2], 'n_prediction_files': len(v[0]),
             'post_projection_fraction_mean': float(np.mean(v[0])), 'post_projection_fraction_max': float(np.max(v[0])),
             'background_projection_fraction_mean': float(np.mean(v[1])), 'background_projection_fraction_max': float(np.max(v[1]))}
            for k, v in sorted(agg.items())]
    return rows


def d1_rows():
    rows = []
    for p in glob.glob(str(OUT / 'd1' / '*' / '*' / '*.json')):
        r = jl(p); m = r['metrics']
        rows.append({'view_id': r['view_id'], 'case_id': r['case_id'], 'method': r['method'], 'lambda_pair': r['source_prior'],
                     'brier': m['brier'], 'nll': m['nll'], 'ece15': m['ece15'], 'auroc': m['auroc'], 'n': m['n'],
                     'post_projection_fraction': r['post_projection_fraction'],
                     'reference_collision_fraction': r['reference_collision_fraction'],
                     'extra_clean_information': r['extra_clean_information']})
    return rows


def ident_rows():
    rows = []
    for p in sorted(glob.glob(str(OUT / 'rho' / '*' / '*' / '*' / 'RHO_INTERVAL_eps*.json'))):
        r = jl(p); case = Path(p).parent.name; true = float(case.split('_r')[1][:2]) / 100
        iv = r['rho_interval']
        rows.append({'case_id': case, 'epsilon': r['epsilon_assumed'] if r['epsilon_assumed'] is not None else 'none',
                     'status': r['status'], 'rho_lower': iv[0] if iv else NA, 'rho_upper': iv[1] if iv else NA,
                     'rho_point_anchor': r['rho_point'], 'tail_ratio_estimate': r['estimate_kappa_tail'],
                     'true_rho_evaluator_only': true,
                     'interval_contains_true': (iv[0] <= true <= iv[1]) if iv else NA, 'n_events': len(r['events'])})
    for s in SEEDS:
        cr = jl(OUT / 'cc_rho' / f'd{s}' / 'CC_RHO_INTERVALS.json') or {}
        for e, r in cr.items():
            iv = r['rho_interval']
            rows.append({'case_id': f'{CC_NAME}_observedFIT_d{s}', 'epsilon': e, 'status': r['status'],
                         'rho_lower': iv[0] if iv else NA, 'rho_upper': iv[1] if iv else NA,
                         'rho_point_anchor': r['rho_point'], 'tail_ratio_estimate': r['estimate_kappa_tail'],
                         'true_rho_evaluator_only': 'UNKNOWN', 'interval_contains_true': NA, 'n_events': len(r['events'])})
    return rows


def rho_eval_rows():
    d1, env, d2 = [], [], []
    for p in sorted(glob.glob(str(OUT / 'rho' / '*' / '*' / '*' / 'RHO_EVAL.json'))):
        r = jl(p); case = r['case_id']
        for x in r['d1']:
            d1.append({'case_id': case, 'objective': x['objective'], 'rho_grid': x['rho'], **{k: x['metrics'][k] for k in ('brier', 'nll', 'ece15')}})
        for obj, e in r['d1_envelope'].items():
            env.append({'case_id': case, 'objective': obj, 'kind': 'D1_mechanism_prob', 'mean_width': e['mean_width'], 'max_width': e['max_width']})
        for x in r['d2_lower_envelope']:
            for op in x['ops']:
                d2.append({'case_id': case, 'objective': x['objective'], 'M': x['M'], 'pi': x['pi'], 'rule': 'LOWER_ENVELOPE_GRID',
                           'tau': op['tau'], 'coverage': op['coverage'], 'risk': op['accepted_error_rate'],
                           'FAR_absent': op['absent_false_accept_rate'], 'cost': op['constant_abstention_cost'],
                           'mean_envelope_width': x['mean_envelope_width']})
        for x in r['anchor_d2']:
            for op in x['selected']['operating_points']:
                d2.append({'case_id': case, 'objective': x['objective'], 'M': int(x['id'].split('_M')[1].split('_')[0]), 'pi': x['pi_target'],
                           'rule': f"ANCHOR_EPS0_rho{x['rho_point']:.3f}", 'tau': op['threshold'], 'coverage': op['coverage'],
                           'risk': op['accepted_error_rate'], 'FAR_absent': op['absent_false_accept_rate'], 'cost': op['constant_abstention_cost']})
        for x in r['anchor_d1']:
            d1.append({'case_id': case, 'objective': x['objective'], 'rho_grid': f"ANCHOR_{x['rho_point']:.3f}", **{k: x['metrics'][k] for k in ('brier', 'nll', 'ece15')}})
    return d1, env, d2


def ctype(case):
    if case.startswith(CC_NAME): return 'CC'
    ds, kind, r, _ = case.split('_')
    return f'{ds}_{kind}_{r}'


def decision_rows(pred):
    """Mean over data seeds; NA counts kept. Main D2 = selected p_best_correct, same pi used/target."""
    g = defaultdict(list)
    for r in pred:
        if r['wrong_pi']:
            continue
        g[(r['scope'], ctype(r['source_case']), r['target'], r['method'], r['M'], r['pi_target'])].append(r)
    rows = []
    for k, rs in sorted(g.items(), key=lambda kv: tuple(map(str, kv[0]))):
        row = {'scope': k[0], 'source_case_type': k[1], 'target': k[2], 'method': k[3], 'M': k[4], 'pi': k[5], 'n_seeds': len(rs)}
        for t in (90, 95):
            for q in ('coverage', 'risk', 'FAR_absent', 'correct_mass', 'cost'):
                vals = [x.get(f'sel_t{t}_{q}') for x in rs]
                row[f't{t}_{q}_mean'] = mean_or_na(vals)
                if q == 'risk':
                    row[f't{t}_risk_NA_seeds(zero_coverage)'] = sum(v is None for v in vals)
            row[f'naivetop1_t{t}_risk_mean'] = mean_or_na([x.get(f'naivetop1_t{t}_risk') for x in rs])
            row[f'naivetop1_t{t}_coverage_mean'] = mean_or_na([x.get(f'naivetop1_t{t}_coverage') for x in rs])
        for q in ('sel_brier', 'sel_nll', 'sel_ece15', 'sel_aurc', 'presence_brier', 'map_action_accuracy'):
            row[q + '_mean'] = mean_or_na([x[q] for x in rs])
        rows.append(row)
    return rows


def jvb_rows(pred):
    """Matched FORWARD_J − FORWARD_BCE per seed (same case, target, M, pi); also FORWARD−NAIVE, POST−NAIVE."""
    idx = {}
    for r in pred:
        if r['wrong_pi']: continue
        idx[(r['scope'], r['source_case'], r['target'], r['method'], r['M'], r['pi_target'])] = r
    rows = []
    pairs = [('FORWARD_J', 'FORWARD_BCE'), ('NAIVE_J', 'NAIVE_BCE'), ('POST_PROJECTED_J', 'POST_PROJECTED_BCE'),
             ('FORWARD_J', 'NAIVE_J'), ('FORWARD_BCE', 'NAIVE_BCE'), ('FORWARD_J', 'POST_PROJECTED_J'), ('FORWARD_BCE', 'POST_PROJECTED_BCE')]
    for (sc, case, tgt, meth, M, pi), r in idx.items():
        for a, b in pairs:
            if meth != a: continue
            o = idx.get((sc, case, tgt, b, M, pi))
            if not o: continue
            row = {'scope': sc, 'source_case': case, 'case_type': ctype(case), 'target': tgt, 'M': M, 'pi': pi, 'A': a, 'B': b}
            for q in ('sel_brier', 'sel_nll', 'sel_ece15', 'map_action_accuracy', 'sel_t90_cost', 'sel_t95_cost',
                      'sel_t90_coverage', 'sel_t95_coverage', 'sel_t90_risk', 'sel_t95_risk'):
                x, y = r.get(q), o.get(q)
                row[f'd_{q}'] = (x - y) if (x is not None and y is not None) else None
            rows.append(row)
    return rows


def cost_rows():
    rows = []
    for p in glob.glob(str(OUT / 'receipts' / '*' / '*' / 'stage_*.json')):
        r = jl(p); parts = Path(p).parts
        rows.append({'unit': f'{parts[-3]}/{parts[-2]}', 'stage': r['stage'], 'seconds': r['seconds'], 'host': r.get('host')})
    for p in glob.glob(str(OUT / 'curator_only' / 'banks' / '*' / 'bank_meta.json')):
        try:
            r = jl(p); rows.append({'unit': f'bank/{Path(p).parent.name}', 'stage': 'bank_export', 'seconds': r['export_seconds'], 'host': NA})
        except Exception:
            rows.append({'unit': f'bank/{Path(p).parent.name}', 'stage': 'bank_export', 'seconds': NA, 'host': NA})
    return rows


# ------------------------------------------------------------------ RESULTS
def tbl(rows, cols, head):
    if not rows: return 'NA (no rows)\n'
    s = '| ' + ' | '.join(head) + ' |\n|' + '---|' * len(head) + '\n'
    for r in rows:
        s += '| ' + ' | '.join(fmt(r.get(c)) for c in cols) + ' |\n'
    return s


def main():
    DST.mkdir(parents=True, exist_ok=True)
    cases = case_ledger(); models = model_rows(); pred = prediction_rows(); proj = projection_rows()
    d1 = d1_rows(); ident = ident_rows(); rd1, renv, rd2 = rho_eval_rows(); dec = decision_rows(pred); jvb = jvb_rows(pred)
    cost = cost_rows()
    oracle = []
    for ds in DATASETS:
        for s in SEEDS:
            for r in jl(group_dirs(ds, s)['oracle'] / 'ORACLE_METRICS.json') or []:
                r = dict(r); r['dataset'] = ds; r['seed'] = s; oracle.append(r)
    otrans = []
    for s in SEEDS:
        for r in jl(OUT / 'transfer' / f'd{s}' / 'ORACLE_SOURCE_TRANSFER_METRICS.json') or []:
            r = dict(r); r['seed'] = s; otrans.append(r)
    integ = {f'{ds}_d{s}': jl(OUT / 'receipts' / ds / f'd{s}' / 'INTEGRITY.json') for ds in DATASETS for s in SEEDS}
    wcsv('CASE_LEDGER.csv', cases); wcsv('MODEL_LEDGER.csv', models); wcsv('PREDICTION_LEDGER.csv', pred)
    wcsv('PROJECTION_LEDGER.csv', proj); wcsv('D1_TABLE.csv', d1); wcsv('IDENTIFIABILITY_LEDGER.csv', ident)
    wcsv('IDENTIFIABILITY_PROFILE_D1.csv', rd1); wcsv('IDENTIFIABILITY_ENVELOPE.csv', renv); wcsv('IDENTIFIABILITY_D2.csv', rd2)
    wcsv('DECISION_LEDGER.csv', dec); wcsv('J_VS_BCE_AND_CORRECTION_DIFFS.csv', jvb); wcsv('COST_LEDGER.csv', cost)
    flat_or = []
    for r in oracle + otrans:
        x = {k: r.get(k) for k in ('dataset', 'transfer', 'seed', 'gallery', 'M', 'pi', 'method', 'target_FAR')}
        if 'selected' in r:
            for op in r['selected']['operating_points']:
                t = int(round(op['threshold'] * 100))
                x.update({f't{t}_coverage': op['coverage'], f't{t}_risk': op['accepted_error_rate'], f't{t}_FAR': op['absent_false_accept_rate'], f't{t}_cost': op['constant_abstention_cost']})
            x['brier'] = r['selected']['correctness_calibration']['brier']
        else:
            op = r['operating_point']; x.update({'coverage': op['coverage'], 'risk': op['accepted_error_rate'], 'FAR': op['absent_false_accept_rate'], 'cost': op['constant_abstention_cost']})
        flat_or.append(x)
    wcsv('ORACLE_REFERENCE_LEDGER.csv', flat_or)
    n_core = sum(m['status'] == 'OK' and m['kind'] == 'CORE' for m in models)
    status = {'protocol_id': 'CDR1_CONTAMINATION_CALIBRATION_SELECTIVE_RETRIEVAL_V1',
              'cases_curated': sum(c['status'] == 'CURATED' for c in cases if not c['case_id'].startswith(CC_NAME)),
              'core_fits_present': n_core, 'core_fits_registered': len(FITS),
              'profile_new_fits': sum(m['kind'] == 'PROFILE_NEW' and m['status'] == 'OK' for m in models),
              'anchor_fits': sum(m['kind'] == 'ANCHOR_EPS0' and m['status'] == 'OK' for m in models),
              'prediction_metric_rows': len(pred), 'd1_rows': len(d1),
              'groups_done': [k for k in integ if (OUT / 'receipts' / k.split('_d')[0] / ('d' + k.split('_d')[1]) / 'GROUP_DONE.json').exists()],
              'transfers_done': [s for s in SEEDS if (OUT / 'receipts' / 'transfer' / f'd{s}' / 'TRANSFER_DONE.json').exists()],
              'ranking_integrity_all_identical': all(v is not None and all(g['top_index_mismatch_files'] == 0 for g in v.values()) for v in integ.values()) if all(integ.values()) else 'PARTIAL',
              'DEDPUL': 'PENDING (SITE_ADAPTER_REQUIRED; not integrated this round; no comparison claimed)',
              'MA_PaPSP': 'NOT_IMPLEMENTED (literature only)', 'NACP': 'NOT_IMPLEMENTED (literature only)',
              'D3_OT': 'NOT_RUN (out of scope)', 'D4': 'NA (no saved artifacts of the required quantities read this round)',
              'TEST_read': False, 'encoder_updates': 0}
    dump(DST / 'EXECUTION_STATUS.json', status)
    dump(DST / 'LEDGERS_INDEX.json', {'CASE': 'CASE_LEDGER.csv', 'MODEL': 'MODEL_LEDGER.csv', 'PROJECTION': 'PROJECTION_LEDGER.csv',
         'IDENTIFIABILITY': ['IDENTIFIABILITY_LEDGER.csv', 'IDENTIFIABILITY_PROFILE_D1.csv', 'IDENTIFIABILITY_ENVELOPE.csv', 'IDENTIFIABILITY_D2.csv'],
         'PREDICTION': 'PREDICTION_LEDGER.csv', 'DECISION': ['DECISION_LEDGER.csv', 'J_VS_BCE_AND_CORRECTION_DIFFS.csv', 'ORACLE_REFERENCE_LEDGER.csv'],
         'COST': 'COST_LEDGER.csv', 'D1': 'D1_TABLE.csv'})
    write_results(status, cases, models, d1, dec, jvb, ident, rd2, renv, proj, integ, cost, flat_or)
    print(json.dumps(status, indent=1, default=str))


def write_results(st, cases, models, d1, dec, jvb, ident, rd2, renv, proj, integ, cost, orl):
    L = []
    A = L.append
    A('# RESULTS_CDR1\n')
    A('Protocol `CDR1_CONTAMINATION_CALIBRATION_SELECTIVE_RETRIEVAL_V1`; evidence base a55332645aed5b23b7d90f92e662e9242652efd4. '
      'Frozen unadapted SigLIP2 (google/siglip2-base-patch16-224 @75de2d55) per-item-L2 global cosine; two scalars per fit; '
      'ranking never changed. Development assay built from canonical FIT only; TEST not read; old Flickr DEV not used. '
      'All numbers below are generated from the ledgers in this directory. Zero coverage is reported as risk=NA, never 0.\n')
    A('## 0. Execution status\n')
    A('```\n' + json.dumps(st, indent=1, default=str) + '\n```\n')
    A('## 1. Data facts\n')
    A(tbl([c for c in cases if c['dataset'] != CC_NAME and c['data_seed'] in ('1001', 1001)],
          ['case_id', 'status', 'collision_beta_fit', 'noise_model_valid_by_construction', 'groups_FIT_CAL', 'groups_EVAL_OPEN'],
          ['case (seed 1001 shown; all in CASE_LEDGER)', 'status', 'beta_fit', 'mixture valid', 'FIT_CAL groups', 'EVAL groups']))
    A('Every query stays in the denominator; present/absent counts per gallery are in CASE_LEDGER.csv. CC152K_RECOVERED is a secondary transfer target only (693 DEV pairs; no true rho, no clean oracle).\n')
    A('## 2. Model fits (theory condition: two-parameter monotone family; J and BCE fitted on identical inputs/maps)\n')
    core = [m for m in models if m['kind'] == 'CORE']
    A(f"Core fits present {sum(m['status']=='OK' for m in core)}/84; selected converged {sum(m.get('selected_converged') is True for m in core)}; "
      f"at bound {sum(m.get('parameter_at_bound') is True for m in core)}. log E_Q r (should be ~0 if the family fits) per objective/regime:\n")
    g = defaultdict(list)
    for m in core:
        if m['status'] == 'OK': g[(m['objective'], m['regime'])].append(m['log_E_Q_fit_r_clean'])
    A(tbl([{'k': f'{k[0]}/{k[1]}', 'n': len(v), 'mean': float(np.mean(v)), 'min': float(np.min(v)), 'max': float(np.max(v))} for k, v in sorted(g.items())],
          ['k', 'n', 'mean', 'min', 'max'], ['objective/regime', 'n', 'mean logE_Q r', 'min', 'max']))
    A('## 3. D1 mechanism-source probability (EVAL_OPEN clean-P vs independent-Q; not semantic correctness)\n')
    gg = defaultdict(list)
    for r in d1:
        if r['lambda_pair'] == .5: gg[(ctype(r['case_id']), r['method'])].append(r)
    rows = [{'ct': k[0], 'm': k[1], 'n': len(v), 'brier': mean_or_na([x['brier'] for x in v]), 'nll': mean_or_na([x['nll'] for x in v]),
             'ece': mean_or_na([x['ece15'] for x in v]), 'proj': mean_or_na([x['post_projection_fraction'] for x in v])} for k, v in sorted(gg.items())]
    A(tbl(rows, ['ct', 'm', 'n', 'brier', 'nll', 'ece', 'proj'], ['case type', 'method', 'n seeds', 'Brier', 'NLL', 'ECE15', 'post-proj frac']))
    A('lambda_pair=0.1 rows are in D1_TABLE.csv. CLEAN_*_ORACLE rows use extra clean information (rho=0 fit of the same seed).\n')
    A('## 4. D2 selective text->image retrieval, same domain (known-rho regime K; oracle rows = information regime O)\n')
    A('Mean over 3 data seeds at M=128, pi=0.5 (full grid M x pi in DECISION_LEDGER.csv). Selected = model-based p_best_correct; risk = accepted error rate.\n')
    sel = [r for r in dec if r['scope'] == 'same_domain' and r['M'] == 128 and r['pi'] == .5]
    A(tbl(sel, ['source_case_type', 'method', 'n_seeds', 't90_coverage_mean', 't90_risk_mean', 't90_risk_NA_seeds(zero_coverage)', 't90_FAR_absent_mean',
                't95_coverage_mean', 't95_risk_mean', 't95_FAR_absent_mean', 't90_cost_mean', 'sel_brier_mean', 'map_action_accuracy_mean',
                'naivetop1_t90_coverage_mean', 'naivetop1_t90_risk_mean'],
          ['case', 'method', 'n', 'cov@.90', 'risk@.90', 'NA seeds', 'FAR@.90', 'cov@.95', 'risk@.95', 'FAR@.95', 'cost@.90', 'Brier(C)', 'MAP acc', 'naiveTop1 cov@.90', 'naiveTop1 risk@.90']))
    A('## 5. J vs matched forward-BCE (same contamination map) and correction effects\n')
    A('Per-seed differences A−B (same case/target/M/pi), averaged over 3 seeds x 3 M x 3 pi = 27 cells; dRisk averages only cells where both risks are defined (non-zero coverage).\n')
    agg = defaultdict(list)
    for r in jvb:
        agg[(r['scope'], r['case_type'], r['target'], r['A'], r['B'])].append(r)
    rows = []
    for k, v in sorted(agg.items(), key=lambda kv: tuple(map(str, kv[0]))):
        rows.append({'k': ' / '.join(map(str, k)), 'n': len(v), 'dB': mean_or_na([x['d_sel_brier'] for x in v]),
                     'dN': mean_or_na([x['d_sel_nll'] for x in v]), 'dC90': mean_or_na([x['d_sel_t90_cost'] for x in v]),
                     'dCov90': mean_or_na([x['d_sel_t90_coverage'] for x in v]), 'dR90': mean_or_na([x['d_sel_t90_risk'] for x in v]),
                     'dMAP': mean_or_na([x['d_map_action_accuracy'] for x in v])})
    A(tbl(rows, ['k', 'n', 'dB', 'dN', 'dC90', 'dCov90', 'dR90', 'dMAP'],
          ['scope / case / target / A / B', 'cells', 'dBrier(C)', 'dNLL(C)', 'dCost@.90', 'dCov@.90', 'dRisk@.90 (defined cells)', 'dMAP acc']))
    A('## 6. Model mismatch: hard contamination (noise_model_valid_by_construction=false) and transfer\n')
    tr = [r for r in dec if r['scope'] == 'transfer' and r['M'] == 128 and r['pi'] == .5 and r['method'] in ('FORWARD_J', 'FORWARD_BCE', 'NAIVE_J', 'NAIVE_BCE')]
    A(tbl(tr, ['source_case_type', 'target', 'method', 'n_seeds', 't90_coverage_mean', 't90_risk_mean', 't90_risk_NA_seeds(zero_coverage)', 't90_FAR_absent_mean', 't95_coverage_mean', 't95_risk_mean', 'sel_brier_mean'],
          ['source case', 'target', 'method', 'n', 'cov@.90', 'risk@.90', 'NA seeds', 'FAR@.90', 'cov@.95', 'risk@.95', 'Brier(C)']))
    A('Wrong-pi (predictor pi=0.5, target pi=0.25/0.75) rows are in PREDICTION_LEDGER.csv (wrong_pi=True).\n')
    A('## 7. Unknown rho (regime U): partial interval first, then explicit anchor assumptions\n')
    A(tbl(ident, ['case_id', 'epsilon', 'status', 'rho_lower', 'rho_upper', 'rho_point_anchor', 'true_rho_evaluator_only', 'interval_contains_true'],
          ['case', 'epsilon', 'status', 'lower', 'upper', 'anchor point', 'true rho (eval only)', 'contains']))
    A('Profile grid {0,.2,.4,.6,.8,.9} fits are a finite-grid sensitivity, not a confidence band and not used to choose rho. Envelope widths:\n')
    A(tbl(renv, ['case_id', 'objective', 'kind', 'mean_width', 'max_width'], ['case', 'objective', 'kind', 'mean width', 'max width']))
    sub = [r for r in rd2 if r['M'] == 128 and r['pi'] == .5]
    A(tbl(sub, ['case_id', 'objective', 'rule', 'tau', 'coverage', 'risk', 'FAR_absent', 'cost'], ['case', 'objective', 'rule', 'tau', 'coverage', 'risk', 'FAR', 'cost']))
    A('## 8. Extra-information references (oracle, regime O)\n')
    og = defaultdict(list)
    for r in orl:
        if r.get('M') == 128 and r.get('pi') == .5:
            og[(r.get('dataset') or r.get('transfer'), r['method'], r.get('target_FAR'))].append(r)
    rows = [{'k': ' / '.join(map(str, k)), 'n': len(v), 'c90': mean_or_na([x.get('t90_coverage') for x in v]), 'r90': mean_or_na([x.get('t90_risk') for x in v]),
             'cov': mean_or_na([x.get('coverage') for x in v]), 'risk': mean_or_na([x.get('risk') for x in v]), 'far': mean_or_na([x.get('FAR') for x in v])}
            for k, v in sorted(og.items(), key=lambda kv: tuple(map(str, kv[0])))]
    A(tbl(rows, ['k', 'n', 'c90', 'r90', 'cov', 'risk', 'far'], ['domain / method / target FAR', 'n', 'Platt cov@.90', 'Platt risk@.90', 'null cov', 'null risk', 'null FAR']))
    A('## 9. Integrity, projections, cost\n')
    A(f"Ranking integrity (every prediction top_index == base argmax): {st['ranking_integrity_all_identical']}. Base R@K per gallery is in receipts/*/INTEGRITY.json and is identical for every method by construction.\n")
    A(tbl([p for p in proj if p['post_projection_fraction_max'] > 0 or p['background_projection_fraction_max'] > 0][:60],
          ['source_case', 'method', 'target', 'post_projection_fraction_mean', 'post_projection_fraction_max', 'background_projection_fraction_mean', 'background_projection_fraction_max'],
          ['case', 'method', 'target', 'post-proj mean', 'post-proj max', 'bg-proj mean', 'bg-proj max']))
    tot = defaultdict(float)
    for c in cost:
        if isinstance(c['seconds'], (int, float)): tot[c['stage']] += c['seconds']
    A(tbl([{'s': k, 'sec': v} for k, v in sorted(tot.items())], ['s', 'sec'], ['stage', 'CPU wall seconds (sum)']))
    A('No GPU was used. External baselines: DEDPUL PENDING (not integrated, no comparison claimed); MA-PaPSP/NACP not implemented. D3/OT not run; D4 NA.\n')
    (DST / 'RESULTS_CDR1.md').write_text('\n'.join(L))


if __name__ == '__main__':
    main()
