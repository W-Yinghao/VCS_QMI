"""Identity, completion and accounting helpers; no scientific imports."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/v10_bf1_20261017')
REPO = Path(__file__).resolve().parents[2]
FAMILIES = ('MF_VCS_perm1','MF_VCS_M0','MF_VCS_M1','MLP_VCS_perm1')
METHODS = dict(zip(FAMILIES,('MF_perm1','MF_M0','MF_M1','MLP_perm1')))


def utc():
    return time.strftime('%FT%TZ',time.gmtime())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


@contextlib.contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def append(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())


def lines(path):
    p=Path(path)
    if not p.exists():return []
    # A broken append is an integrity error, not a silently dropped record.
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def freeze():
    p=ROOT/'FROZEN.json'
    if os.environ.get('V10_BF1_EXPECT_FREEZE') and sha(p)!=os.environ['V10_BF1_EXPECT_FREEZE']:
        raise RuntimeError('Frozen BF1 identity changed after submission')
    f=read(p)
    for name,value in f['source_files'].items():
        if sha(REPO/name)!=value:raise RuntimeError('Frozen source mismatch: '+name)
    for name,value in f['run_inputs'].items():
        if sha(ROOT/name)!=value:raise RuntimeError('Frozen input mismatch: '+name)
    return f


def candidates(family):
    assert family in FAMILIES
    out=[]
    for lr in (1e-4,3e-4,1e-3):
        for cap in ((10,30,100) if family=='MF_VCS_M1' else (None,)):
            out.append({'family':family,'lr':lr,'cap':cap,'order':len(out)})
    return out


def identity(family,seed,lr,cap,phase,dependency=None):
    return {'schema':'v10.bf1.unit.v1','freeze_sha256':sha(ROOT/'FROZEN.json'),
            'family':family,'method':METHODS[family],'seed':seed,'lr':lr,'cap':cap,
            'phase':phase,'steps':3000,'checkpoint_interval':250,'K':16,
            'dependency':dependency,'data_lock_sha256':sha(ROOT/'DATA_LOCK.json')}


def metrics(rows):
    errs=[float(r['err_full']) for r in rows]
    if not errs or not all(math.isfinite(x) and x>=0 for x in errs):
        raise ValueError('Empty or non-finite registration results')
    return {'n':len(errs),'success42':sum(x<42 for x in errs)/len(errs),
            'success9':sum(x<9 for x in errs)/len(errs),'median_err':statistics.median(errs)}


def selection_key(rec):
    m=rec['metrics']
    return (-m['success42'],-m['success9'],m['median_err'],rec['step'],rec.get('order',0))


def verify_complete(directory,expected):
    directory=Path(directory);p=directory/'COMPLETE.json'
    if not p.exists():return None
    r=read(p)
    if r.get('status')!='COMPLETE' or r.get('identity')!=expected:
        raise RuntimeError('Stale COMPLETE identity: '+str(directory))
    for name,value in r['artifacts'].items():
        if sha(directory/name)!=value:raise RuntimeError('Stale COMPLETE artifact: '+name)
    if r['steps']!=3000 or len(r['checkpoints'])!=12:
        raise RuntimeError('Incomplete full training marked COMPLETE')
    return r


def require_identity(directory,expected):
    p=Path(directory)/'IDENTITY.json'
    if p.exists() and read(p)!=expected:raise RuntimeError('Refusing mismatched resume: '+str(p))
    write(p,expected)


def spent_seconds(family):
    by_job={}
    for r in lines(ROOT/'COST_LEDGER.jsonl'):
        if r.get('family')==family and r.get('phase')=='CAL':
            j=str(r['job_id']);by_job[j]=max(by_job.get(j,0),r.get('allocated_seconds',0))
    return sum(by_job.values())


class BudgetExceeded(RuntimeError):
    pass


class Allocation:
    def __init__(self,family,phase,start=None):
        self.family,self.phase=family,phase
        self.job=os.environ.get('SLURM_JOB_ID','local-validation')
        self.start=float(os.environ.get('V10_BF1_JOB_START',time.time())) if start is None else start
        self.previous=spent_seconds(family) if phase=='CAL' else 0
        self.last=0
        self.record('START')

    def elapsed(self):return max(0,time.time()-self.start)

    def record(self,event,**extra):
        append(ROOT/'COST_LEDGER.jsonl',{'utc':utc(),'event':event,'job_id':self.job,
            'family':self.family,'phase':self.phase,'allocated_seconds':self.elapsed(),
            'start_unix':self.start,'gpu_count':1,'accounting':'batch-start wall; reconciled with sacct after exit',**extra})
        self.last=time.monotonic()

    def check(self):
        if time.monotonic()-self.last>30:self.record('HEARTBEAT')
        # A reserve for record flush and orderly exit, never a budget extension.
        if self.phase=='CAL' and self.previous+self.elapsed()>=8*3600-30:
            raise BudgetExceeded('Family allocated-time cap reached')
