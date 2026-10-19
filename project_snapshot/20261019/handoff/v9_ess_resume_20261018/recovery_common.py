"""Narrow diagnostic-metadata recovery; no numerical training imports."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

BUNDLE=Path(__file__).resolve().parent
PROJECT=BUNDLE.parents[1]
SPLIT=PROJECT/'handoff/v9_unit_split_20261018'
FULL=PROJECT/'handoff/v9_full_20261017'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def utc():return time.strftime('%FT%TZ',time.gmtime())
def require(ok,message):
    if not ok:raise RuntimeError(message)
def write(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name('.'+p.name+'.'+str(os.getpid())+'.tmp')
    q.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)

def compare_manifest(original,recomputed):
    require(original.keys()==recomputed.keys(),'Manifest fields changed')
    require({k:v for k,v in original.items() if k!='ess_U'}==
            {k:v for k,v in recomputed.items() if k!='ess_U'},'Training/config/normalizer field changed')
    old,new=original['ess_U'],recomputed['ess_U']
    require(type(old) in (int,float) and type(new) in (int,float) and
            math.isfinite(old) and math.isfinite(new) and old>0 and new>0,'Invalid diagnostic ESS')
    require(abs(old-new)<=4*math.ulp(old),'ESS difference exceeds four fp64 ulps')
    return {'original_ess':old,'recomputed_ess':new,'difference':new-old,
            'difference_ulps':(new-old)/math.ulp(old),'other_fields_exactly_equal':True}

def verify_bundle():
    for name,value in read(BUNDLE/'FREEZE.json')['files'].items():
        require(sha(BUNDLE/name)==value,'Recovery source changed: '+name)
    cfg=read(BUNDLE/'CONFIG.json')
    require(sha(FULL/'FROZEN.json')==cfg['original_freeze'],'Science freeze changed')
    require(sha(SPLIT/'CONTROL_FREEZE.json')==cfg['split_freeze'],'Split freeze changed')
    for name,value in cfg['source_sha256'].items():require(sha(name)==value,'Original source changed')
    return cfg

def preflight(cfg):
    unit=Path(cfg['unit']);archive=Path(cfg['archive'])
    require(sha(archive/'run_manifest.json')==cfg['manifest_sha256'],'Original manifest changed')
    result=compare_manifest(read(archive/'run_manifest.json'),read(unit/'run_manifest.json'))
    for name,value in cfg['checkpoint_hashes'].items():
        require(sha(unit/name)==value and sha(archive/name)==value,'Checkpoint/identity changed: '+name)
    require(not (unit/'done.json').exists() and not (unit/'full_result.json').exists(),'Recovery unit already complete')
    failure=read(SPLIT/'units/903_1/0/FAILED_1046652.json')
    require(failure['traceback'].rstrip().endswith('ValueError: Resume recipe differs from the saved checkpoint manifest'),
            'Failure is not the reviewed manifest guard')
    require(read(SPLIT/'units/903_1/0/EXIT_1046652.json')['exit_code']==1,'Unexpected prior exit')
    return result
