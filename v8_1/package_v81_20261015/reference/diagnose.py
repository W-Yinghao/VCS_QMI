"""Run the V8.1 CPU reference diagnostics on explicit normalized JSONL inputs."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from core import summarize

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--label-mode',choices=['development','none'],required=True)
    a=p.parse_args()
    if a.out.exists(): raise SystemExit(f'Refusing to overwrite: {a.out}')
    raw=a.rows.read_bytes()
    rows=[json.loads(s) for s in raw.decode('utf-8').splitlines() if s.strip()]
    try: result=summarize(rows,labelled=a.label_mode=='development')
    except ValueError as e: raise SystemExit(f'Input/metric validation failed: {e}')
    result['input']={'path':str(a.rows),'sha256':hashlib.sha256(raw).hexdigest()}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(a.out),'n_rows':len(rows),'models':len(result['model_levels']), 'status':result['status']},ensure_ascii=False))
if __name__=='__main__': main()
