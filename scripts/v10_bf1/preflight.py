"""Slurm-only data identity/QC for the already activated BF FIT/CAL/DEV roles."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/v10_bf1_20261017')
CACHE = Path('/home/infres/yinwang/CS_QMI/outputs/R6_REG')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Data QC must run under Slurm'
    assert sha(__file__) == os.environ['V10_PREFLIGHT_SHA'], 'preflight changed after submission'
    manifest = REPO/'v10/T00/V10_DATA_MANIFEST.csv'
    assert sha(manifest) == 'cc309cece34b574b2a518b5775a1e30987ebd9015693cca1543d5610f13564f6'
    rows = [r for r in csv.DictReader(manifest.open()) if r['dataset']=='BF_SHG' and r['inclusion']=='include']
    roles = json.loads((CACHE/'manifest.json').read_text())['roles']
    assert {k:sum(r['role']==k for r in rows) for k in ('FIT','CAL','DEV')} == {'FIT':32,'CAL':8,'DEV':32}
    files=[]
    for r in rows:
        assert r['role'] in ('FIT','CAL','DEV') and roles[r['record_id']]==r['role']
        assert not any(bad in r['fixed_path']+r['moving_path'] for bad in ('/closed/','outputs_v9/reg_test','test.zip','/test/'))
        p=CACHE/'cache'/f"{r['record_id']}.npz"
        shapes={}
        with np.load(p,allow_pickle=False) as z:
            for key,shape in (('bf_rgb_n',(512,512,3)),('shg_log_n',(512,512)),('bf',(512,512)),('shg_log',(512,512))):
                a=z[key]
                assert a.shape==shape and np.isfinite(a).all(), (r['record_id'],key,a.shape)
                shapes[key]={'shape':list(a.shape),'dtype':str(a.dtype)}
        files.append({'core':r['record_id'],'role':r['role'],'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'arrays':shapes})
    out={'schema':'v10.bf1.data_lock.v1','status':'PASS','utc':datetime.now(timezone.utc).isoformat(),
         'job_id':os.environ['SLURM_JOB_ID'],'preflight_sha256':sha(__file__),'manifest_sha256':sha(manifest),
         'r6_manifest':{'path':str(CACHE/'manifest.json'),'sha256':sha(CACHE/'manifest.json')},
         'counts':{'FIT':32,'CAL':8,'DEV':32},'held_test_read':False,'files':files}
    ROOT.mkdir(parents=True,exist_ok=True)
    p=ROOT/'DATA_LOCK.json'
    if p.exists():
        old=json.loads(p.read_text())
        assert old['files']==files and old['manifest_sha256']==out['manifest_sha256'], 'Data identity changed; preserve original lock'
    else:
        p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','files':len(files),'data_lock_sha256':sha(p),'job_id':out['job_id']}),flush=True)


if __name__=='__main__':
    main()
