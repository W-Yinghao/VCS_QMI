"""Generate labelled synthetic fixtures only; these are not natural-data results."""
from pathlib import Path
import json, argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists(): raise SystemExit('Output already exists')
    rows=[]
    targets=[[1,3,5],[2,4,3],[3,3,3],[5,2,1]]
    scores={'demo-aligned':[[1,3,5],[2,4,3],[3,3,3],[5,2,1]],'demo-shifted':[[2,4,5],[3,5,4],[4,4,4],[5,3,2]],'demo-reversed':[[5,3,1],[4,2,3],[3,3,3],[1,4,5]]}
    for method,pools in scores.items():
        for p,(ys,ss) in enumerate(zip(targets,pools)):
            for i,(y,s) in enumerate(zip(ys,ss)):
                q=[0.]*5;q[int(s)-1]=1.
                rows.append({'dataset':'SYNTHETIC_ONLY','role':'DEV','method':method,'seed':0,'checkpoint_id':'demo-not-a-model',
                             'prompt_id':f'p{p}','pool_id':f'p{p}','cluster_id':f'p{p}','image_id':f'p{p}-i{i}','candidate_index':i,
                             'score':float(s),'score_kind':'rating_mean','q':q,'votes':[y,y,y]})
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    print(f'Synthetic fixture only: {len(rows)} rows')
if __name__=='__main__': main()
