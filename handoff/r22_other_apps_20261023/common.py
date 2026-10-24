import hashlib
import json
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def verify():
    assert os.environ.get('SLURM_JOB_ID'),'Run numerical work inside Slurm'
    for rel,digest in read(HERE/'FREEZE.json')['files'].items():assert sha(BASE/rel)==digest,rel
    return sha(HERE/'FREEZE.json')

