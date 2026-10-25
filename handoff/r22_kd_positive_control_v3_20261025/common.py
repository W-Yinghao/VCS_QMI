import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
OLD=BASE/'handoff/r22_other_apps_20261023'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def verify():
    for rel,h in read(HERE/'FREEZE.json')['files'].items():assert sha(BASE/rel)==h,rel
    return sha(HERE/'FREEZE.json')

