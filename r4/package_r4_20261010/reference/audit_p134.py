"""Read-only P134 posterior audit.

Example:
    python reference/audit_p134.py --input /server/reports/P134 --output /new/R4_P134

Requires NumPy and SciPy. Only registered selected VCS/JS rows are read. The
script does not fit, reinterpret contaminated MSE as clean MSE, or overwrite raw
files. Validate against actual server schema before using report outputs.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
from collections import defaultdict
from typing import Any
import numpy as np
from scipy.stats import t as student_t
from vcs_r4_reference import read_common_fields


def extract_selected(path: Path) -> list[dict[str, Any]]:
    data=json.loads(path.read_text(encoding='utf-8'))
    c=data.get('cell',{}); con=c.get('contamination',{})
    if 'seed' not in c.get('base',{}) or 'type' not in con or 'eps' not in con:
        raise ValueError(f'{path.name}: unknown cell/base/contamination schema')
    seed=int(c['base']['seed']); eps=float(con['eps']); typ=str(con['type'])
    result=[]; seen=set()
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    for row in data.get('rows',[]):
        kind=row.get('kind')
        if kind not in ('vcs','js') or row.get('selected') is not True:
            continue
        pairing=row.get('negative_construction')
        key=(kind,pairing)
        if pairing is None or key in seen:
            raise ValueError(f'{path.name}: duplicate selected family or missing construction {key}')
        seen.add(key)
        shared=read_common_fields(row)
        if shared['posterior_mse'] is None:
            raise ValueError(f'{path.name}: missing posterior.posterior_mse for {key}')
        result.append(dict(source_file=path.name,sha256=sha,seed=seed,contamination=typ,
                           epsilon=eps,method=kind,negative_construction=pairing,
                           posterior_mse=float(shared['posterior_mse']),
                           common_J=shared['common_J'],common_J_minus_S=shared['common_J_minus_S'],
                           native_estimand=row.get('estimand'),native_value=row.get('native',{}).get('value'),
                           oracle_S=data.get('truth',{}).get('S'),
                           truth_note=data.get('truth',{}).get('truth_note'),
                           selected_lr=row.get('lr'),selected_update=row.get('select',{}).get('selected_update'),
                           error_target='contaminated_posterior_check_truth_note'))
    if not result: raise ValueError(f'{path.name}: no selected VCS/JS rows found')
    return result


def paired_changes(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    index={};clean={}
    for r in rows:
        k=(r['contamination'],r['epsilon'],r['negative_construction'],r['seed'],r['method'])
        if k in index: raise ValueError(f'Duplicate source row {k}')
        index[k]=r
        if r['epsilon']==0:
            ck=(r['negative_construction'],r['seed'],r['method'])
            if ck in clean: raise ValueError(f'Duplicate clean baseline {ck}')
            clean[ck]=r
    out=[]
    for key,rv in index.items():
        typ,ep,pair,seed,method=key
        if method!='vcs' or ep==0:continue
        rj=index.get((typ,ep,pair,seed,'js'))
        cv=clean.get((pair,seed,'vcs'));cj=clean.get((pair,seed,'js'))
        if any(r is None for r in (rj,cv,cj)):
            raise ValueError(f'Missing paired JS or clean baseline: {key}')
        dv=rv['posterior_mse']-cv['posterior_mse'];dl=rj['posterior_mse']-cj['posterior_mse']
        out.append(dict(contamination=typ,epsilon=ep,negative_construction=pair,seed=seed,
                        vcs_mse=rv['posterior_mse'],js_mse=rj['posterior_mse'],
                        absolute_difference=rv['posterior_mse']-rj['posterior_mse'],
                        vcs_degradation=dv,js_degradation=dl,degradation_difference=dv-dl,
                        descriptive_ratio=dv/dl if dl>1e-8 and dv>=0 else None,
                        ratio_note='descriptive_only_no_ratio_CI; unstable denominator requires omission',
                        target='contaminated_truth; epsilon also changes distribution'))
    return out


def summaries(pairs):
    grouped=defaultdict(list)
    for r in pairs:grouped[(r['contamination'],r['epsilon'],r['negative_construction'])].append(r)
    out=[]
    for key,rs in sorted(grouped.items()):
        o=dict(contamination=key[0],epsilon=key[1],negative_construction=key[2],n_seeds=len(rs))
        for field in ('absolute_difference','degradation_difference'):
            x=np.array([r[field] for r in rs]);mean=float(x.mean());sd=float(x.std(ddof=1)) if len(x)>1 else None
            rad=float(student_t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x))) if sd is not None else None
            o.update({field+'_mean':mean,field+'_sd':sd,
                      field+'_lower':None if rad is None else mean-rad,
                      field+'_upper':None if rad is None else mean+rad})
        o['interpretation']='paired-seed t intervals, descriptive; no multiple-cell inference'
        out.append(o)
    return out


def write_csv(path,rows):
    if not rows:raise ValueError('No results to write.')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if not a.input.is_dir():raise FileNotFoundError(a.input)
    if a.output.exists():raise FileExistsError('Use a new output directory; existing results are not overwritten.')
    paths=sorted(a.input.glob('P134_*.json'))
    if not paths:raise FileNotFoundError('No P134_*.json files.')
    rows=[r for path in paths for r in extract_selected(path)];pairs=paired_changes(rows);summary=summaries(pairs)
    a.output.mkdir(parents=True)
    write_csv(a.output/'selected_rows.csv',rows);write_csv(a.output/'paired_changes.csv',pairs);write_csv(a.output/'summary.csv',summary)
    (a.output/'audit_metadata.json').write_text(json.dumps({'files':len(paths),'rows':len(rows),'paired_rows':len(pairs),'target':'posterior fields against contaminated truth; inspect truth_note','source':'read_only_existing_cells'},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'files':len(paths),'selected_rows':len(rows),'output':str(a.output)}))
if __name__=='__main__':main()
