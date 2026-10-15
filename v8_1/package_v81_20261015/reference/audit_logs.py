"""Read V8 log.json files only. Does not deserialize checkpoints or modify them."""
from __future__ import annotations
import argparse, ast, hashlib, json
from pathlib import Path

def inspect_log(log: dict) -> dict:
    epochs=log.get('epochs_log',[]); selected=log.get('selected_epoch')
    matches=[e for e in epochs if e.get('epoch')==selected]
    e=matches[0] if len(matches)==1 else None
    steps=e.get('steps') if e else None; seen=e.get('images_seen') if e else None
    has_updates=isinstance(steps,(int,float)) and steps>0
    status='after_logged_epoch_if_snapshot_matches' if e and has_updates else 'insufficient_log_evidence'
    return {'arm':log.get('arm'), 'seed':log.get('seed'),'selected_epoch_index':selected,
            'logged_epoch_count':len(epochs),'selected_epoch_steps_attempted':steps,'selected_epoch_images_seen':seen,
            'completed_epoch_count_if_zero_based':selected+1 if isinstance(selected,int) and e else None,
            'successful_optimizer_updates':'unknown_unless_recorded_in_runtime_trace',
            'resume_history':log.get('resumed_at',[]), 'status':status,
            'not_equivalent_to_preupdate_init':bool(e and has_updates), 'epochs_log':epochs}

def inspect_source(path: Path) -> dict:
    raw=path.read_bytes(); src=raw.decode('utf-8'); tree=ast.parse(src)
    f=next((n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=='cmd_train'),None)
    if f is None: raise ValueError('No cmd_train in source')
    calls=[]
    for n in ast.walk(f):
        if isinstance(n,ast.Call):
            name=ast.unparse(n.func)
            if name in {'scaler.step','cal_metrics','predict','torch.save','model.load_state_dict'}:
                calls.append({'function':name,'lineno':n.lineno})
    return {'source_path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'calls':sorted(calls,key=lambda r:r['lineno']),
            'scope':'static line-order evidence only; inspect control flow and actual job snapshot before concluding'}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--logs',nargs='+',type=Path,required=True)
    ap.add_argument('--source',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists(): raise SystemExit('Output already exists')
    res={'units':[],'checkpoint_bytes_read':False}
    for p in a.logs:
        raw=p.read_bytes();d=inspect_log(json.loads(raw));d.update({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest()});res['units'].append(d)
    if a.source: res['source']=inspect_source(a.source)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(res,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'Wrote {len(res["units"])} log audits to {a.out}; no checkpoint loaded')
if __name__=='__main__': main()
