"""Read a selected V8 prediction file and the known public TRAIN role manifest.

No archive traversal, connector access, fitting or external-label loading.
Supply the V7 public-TRAIN manifest documented by v8_labels.py, NOT a test file.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from core import validate_rows

def read_jsonl(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines() if s.strip()]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pred',type=Path,required=True)
    ap.add_argument('--public-train-manifest',type=Path,required=True)
    ap.add_argument('--role',choices=['CAL_SELECT','CAL_PROB','DEV'],required=True)
    ap.add_argument('--method',required=True); ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--checkpoint-id',required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists(): raise SystemExit('Output already exists')
    labels={}
    for r in read_jsonl(a.public_train_manifest):
        if r.get('role')==a.role and r.get('overall_votes'):
            if r['image_id'] in labels: raise SystemExit('Duplicate manifest image ID')
            labels[r['image_id']]=r
    pred={}
    for r in read_jsonl(a.pred):
        if r['image_id'] in pred: raise SystemExit('Duplicate prediction image ID')
        pred[r['image_id']]=r
    if set(labels)!=set(pred):
        raise SystemExit(f'ID mismatch: missing={len(set(labels)-set(pred))}, extra={len(set(pred)-set(labels))}')
    per={}
    for i,r in labels.items(): per.setdefault(str(r['prompt_id']),[]).append(i)
    order={i:k for ids in per.values() for k,i in enumerate(sorted(ids,key=str))}
    out=[]
    for i,r in sorted(labels.items(),key=lambda kv:str(kv[0])):
        pr=pred[i]
        if str(pr.get('prompt_id',r['prompt_id'])) != str(r['prompt_id']): raise SystemExit('Prompt mismatch')
        row={'dataset':'EvalMuse-public-train','role':a.role,'method':a.method,'seed':a.seed,'checkpoint_id':a.checkpoint_id,
             'prompt_id':str(r['prompt_id']),'pool_id':str(r['prompt_id']),'image_id':str(i), 'candidate_index':order[i],
             'cluster_id':str(r.get('component_id',r['prompt_id'])),'score':float(pr['mu']),'score_kind':'rating_mean','votes':r['overall_votes']}
        for k in ('q','u'):
            if k in pr: row[k]=pr[k]
        out.append(row)
    out=validate_rows(out,labelled=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in out),encoding='utf-8')
    meta={'source_pred':str(a.pred),'source_pred_sha256':hashlib.sha256(a.pred.read_bytes()).hexdigest(),
          'manifest_sha256':hashlib.sha256(a.public_train_manifest.read_bytes()).hexdigest(),
          'n_rows':len(out),'candidate_order':'lexical image ID (development only, not official GenAI-Rank order)',
          'cluster_fallback':'component_id where present; prompt otherwise'}
    a.out.with_suffix('.meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False))
if __name__=='__main__': main()
