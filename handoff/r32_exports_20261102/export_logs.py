#!/usr/bin/env python3
"""R32 (NRCS-v1) read-only log export for the 2026-11-02 review package (docs/SERVER_PLAN_CN.md §1, R32 part).
No training, no TEST, nothing written under the R32 handoff or output dirs.
 1) runs the existing nrcs/report.py unchanged (imported; only its output dir constant S is redirected to out/report/)
 2) exports from existing logs (history.json / steps.jsonl / metrics.json):
      out/curves_epoch.csv   W2/E2/C2 per-epoch evaluation (SELECT/AUDIT/DEV J + J terms, six recalls, marginals,
                             ratio capacity) and train means
      out/curves_steps.csv   W2/E2/C2 logged steps (every 50 steps; gradient terms at steps 0, 1 and every 200)
      out/curves_f1.csv      F1 per-step evaluation history;  out/curves_s0.csv  S0 history
      out/UNITS.json         per-unit endpoint records (status; unfinished units NA)
Usage: python -B export_logs.py
"""
import csv, glob, importlib.util, json, math, os, sys

W = '/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101'
O = '/projects/EEG-foundation-model/yinghao/outputs_v9/r32_native_ratio_20261101'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f'{HERE}/out'
NA = 'NA'


def jl(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def flat(d, pre='', out=None, skip=('scope', 'pair_law', 'tie_rule', 'upper_bound_rule', 'quantile_sampling')):
    out = {} if out is None else out
    if isinstance(d, dict):
        for k, v in d.items():
            if k in skip:
                continue
            flat(v, f'{pre}{k}.' if pre or k else '', out, skip)
    elif isinstance(d, list):
        out[pre.rstrip('.')] = json.dumps(d)
    else:
        out[pre.rstrip('.')] = d
    return out


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, keys, restval=NA)
        w.writeheader(); w.writerows(rows)
    return len(rows), len(keys)


def entropy_from_eff(x):
    return math.log(x) if isinstance(x, (int, float)) and x > 0 else NA


def main():
    os.makedirs(f'{OUT}/report', exist_ok=True)
    spec = importlib.util.spec_from_file_location('nrcs_report', f'{W}/nrcs/report.py')
    rep = importlib.util.module_from_spec(spec); spec.loader.exec_module(rep)
    rep.S = f'{OUT}/report'            # redirect output only; code unchanged
    rep.main()
    matrix = rep.rows()
    ep_rows, st_rows, f1_rows, s0_rows, units = [], [], [], [], {}
    for r in matrix:
        u, st = r['unit_id'], r['stage']
        rd = f'{O}/runs/{u}'
        base = {k: r[k] for k in ('unit_id', 'stage', 'arm', 'seed', 'noise_percent', 'balance_weight', 'intra_weight', 'parent_id')}
        if st in ('W2', 'E2', 'C2'):
            done = jl(f'{rd}/DONE.json')
            status = 'COMPLETE' if done else ('RUNNING_OR_PARTIAL' if os.path.exists(f'{rd}/resume.pt') else 'MISSING')
            units[u] = dict(base, status=status)
            if not done:
                continue                       # unfinished units: NA (curves not exported)
            for h in jl(f'{rd}/history.json') or []:
                row = dict(base, epoch=h['epoch'], global_step=h['global_step'])
                for role in ('select', 'audit', 'dev'):
                    row.update({f'{role}.{k}': v for k, v in flat(h['eval'][role]).items()})
                for k, v in h.items():
                    if k.startswith('train_') or k in ('epoch_seconds', 'records_drawn', 'peak_mem_bytes'):
                        row[k] = v
                ep_rows.append(row)
            seen = set()
            for line in open(f'{rd}/steps.jsonl'):
                s = json.loads(line)
                row = dict(base, **{k: v for k, v in s.items() if k != 'diag'})
                row['duplicate_of_earlier_attempt'] = s['step'] in seen
                seen.add(s['step'])
                dg = s.get('diag')
                if dg:
                    for k, v in flat(dg).items():
                        row['diag.' + k] = v
                    c = dg.get('critic', {})
                    for side in ('image', 'text'):
                        row[f'diag.critic.{side}_mean_posterior_code_entropy_nats(derived=ln eff)'] = entropy_from_eff(c.get(f'{side}_mean_posterior_effective_components'))
                        row[f'diag.critic.{side}_mean_per_sample_code_entropy_nats(derived=ln eff, Jensen lower bound)'] = entropy_from_eff(c.get(f'{side}_mean_per_sample_effective_components'))
                st_rows.append(row)
        elif st == 'F1':
            m = jl(f'{rd}/metrics.json')
            units[u] = dict(base, status='COMPLETE' if m and m.get('status') == 'COMPLETE' else 'MISSING')
            if not m:
                continue
            for h in m['history']:
                row = dict(base, step=h['step'])
                for role in ('select', 'audit', 'dev'):
                    if role in h:
                        row.update({f'{role}.{k}': v for k, v in flat(h[role]).items()})
                for k, v in h.items():
                    if k not in ('select', 'audit', 'dev', 'step') and not isinstance(v, dict):
                        row[k] = v
                row['is_selected_step'] = h['step'] == m['selected']['step']
                f1_rows.append(row)
        elif st == 'S0':
            m = jl(f'{rd}/metrics.json')
            units[u] = dict(base, status='COMPLETE' if m and m.get('status') == 'COMPLETE' else 'MISSING')
            if not m:
                continue
            for h in m.get('history', []):
                s0_rows.append(dict(base, **flat(h)))
    info = {}
    for name, rows in (('curves_epoch.csv', ep_rows), ('curves_steps.csv', st_rows), ('curves_f1.csv', f1_rows), ('curves_s0.csv', s0_rows)):
        info[name] = write_csv(f'{OUT}/{name}', rows)
    rec = {x['unit_id']: x for x in (rep.unit_record(r) for r in matrix)}
    for u in units:
        units[u]['report_record'] = rec.get(u)
    json.dump({'source': 'nrcs/report.py unit_record (unchanged) + log exports; no TEST; numbers only', 'files': info, 'units': units},
              open(f'{OUT}/UNITS.json', 'w'), indent=1, default=str)
    st = {}
    for x in units.values():
        st.setdefault(x['stage'], {}).setdefault(x['status'], 0)
        st[x['stage']][x['status']] += 1
    print(json.dumps(st), json.dumps(info))


if __name__ == '__main__':
    main()
