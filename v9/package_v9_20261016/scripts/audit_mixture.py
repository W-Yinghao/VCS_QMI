#!/usr/bin/env python3
"""Read-only bounded-score audit. No data downloading or job submission.

Input roles must be independent cluster means with explicit image/donor provenance.
This tool does not infer correctness of the mixture or fixed-score assumptions.
"""
import sys,json,argparse
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference.core import vector,role_disjointness,rho_from_means,rho_confidence_set

def main():
    ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('--out',required=True);args=ap.parse_args()
    data=json.loads(Path(args.input).read_text());rows=data['rows']
    role_disjointness(rows)
    ar={k:vector([r['score'] for r in rows if r['role']==k],k) for k in ['P','Q','R']}
    rr=rho_from_means(*(float(ar[k].mean()) for k in ['P','Q','R']))
    assumptions=data.get('assumptions',{})
    ok=all(assumptions.get(k) is True for k in ['fixed_score_outside_eval','independent_clusters','specified_mixture'])
    out={'point':asdict(rr),'counts':{k:len(v) for k,v in ar.items()},'finite_sample_ci':None,
         'assumption_status':'asserted_by_input_not_inferred' if ok else 'not_verified',
         'warning':'A nonempty moment interval does not establish absence of covariate shift.'}
    if ok:out['finite_sample_ci']=asdict(rho_confidence_set(ar['P'],ar['Q'],ar['R'],data.get('alpha',.05),assumptions_verified=True))
    dst=Path(args.out);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
