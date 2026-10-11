from __future__ import annotations
import argparse, json
from pathlib import Path
from r6_reference import make_task,evidence_summary,oracle_policy_risk,finite_model_utility,shift_demo

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    res={'status':'controlled_reference_only','natural_image_training':False,'server_execution':False,'tasks':{}}
    for name in ('redundant','irrelevant','xor','context'):
        t=make_task(name);e=evidence_summary(t)
        e['conditional_gain']={str(k):v for k,v in e['conditional_gain'].items()}
        e['oracle_cost_0.3']=oracle_policy_risk(t,.3)
        res['tasks'][name]=e
    t=make_task('redundant')
    a={-1:-1.,1:1.};b={(-1,-1):0.,(1,1):0.}
    res['imperfect_predictor_counterexample']=finite_model_utility(t,a,b)
    res['registration_shift_check']=shift_demo()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':res['status'],'recovered_shift':res['registration_shift_check']['recovered_shift'],'saved':str(args.out)}))

if __name__=='__main__':main()
