import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def verify():
    assert os.environ.get('SLURM_JOB_ID'),'Numerical work must run inside Slurm'
    for p,h in read(HERE/'FREEZE.json')['files'].items():assert sha(BASE/p)==h,p
    return sha(HERE/'FREEZE.json')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def geometry():
    return load('mechanism_long',BASE/'handoff/r22_geometry_long_v2_20261024/run.py')
def check_pass(name):
    r=read(HERE/'checks'/f'{name}.json');assert r['status']=='PASS' and r['freeze_sha256']==sha(HERE/'FREEZE.json')
def output(name):
    p=HERE/'runs'/name;p.mkdir(parents=True,exist_ok=True)
    assert not (p/'COMPLETE.json').exists(),'Do not duplicate successful work'
    assert not list(p.glob('ATTEMPT_*.json')),'Review failed attempt; do not overwrite or retry silently'
    write(p/f'ATTEMPT_{os.environ["SLURM_JOB_ID"]}.json',dict(status='STARTED',unix=time.time()))
    return p
def done(out,started,**kwargs):
    write(out/'COMPLETE.json',dict(status='COMPLETE',freeze_sha256=sha(HERE/'FREEZE.json'),seconds=time.time()-started,job=os.environ['SLURM_JOB_ID'],**kwargs))
