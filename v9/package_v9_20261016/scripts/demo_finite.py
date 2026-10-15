#!/usr/bin/env python3
"""Small exact-distribution demonstration; no natural-data claim."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from dataclasses import asdict
from reference.core import finite_target,finite_j,mixture_mass,clean_posterior,rho_from_means

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    p=np.array([.60,.25,.10,.05]);q=np.array([.10,.15,.30,.45]);rho=.3
    d=finite_target(p,q);eta=d['eta'];r=mixture_mass(p,q,rho)
    a,b,c=map(float,(p@eta,q@eta,r@eta));rr=rho_from_means(a,b,c)
    ww=clean_posterior(eta,rho);ell=np.array([.2,.5,.8,.3])
    out={'scope':'exact finite support, not natural training','S':d['S'],'J_eta':finite_j(p,q,eta),
         'rho_design':rho,'rho_estimate':asdict(rr),'E_P_eta':a,'E_Q_eta':b,
         'weighted_risk':float(r@(ww*ell)),'clean_mass_times_risk':float((1-rho)*(p@ell)),
         'identity_error':float(abs(r@(ww*ell)-(1-rho)*(p@ell)))}
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
