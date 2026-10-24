"""Admit this finite batch to the existing pool only after both real GPU gates pass."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
POOL=BASE/'handoff/r22_task_diagnostics_20261023'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())

def main():
    assert os.environ.get('SLURM_JOB_ID')
    with (HERE/'admit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.time()
        while time.time()-start<94*3600 and not (HERE/'STOP_ADMIT').exists():
            freeze=sha(HERE/'FREEZE.json');cpu=read(HERE/'CPU_CHECK.json')
            assert cpu['status']=='PASS' and cpu['freeze_sha256']==freeze
            for rel,digest in read(HERE/'FREEZE.json')['files'].items():assert sha(BASE/rel)==digest,rel
            paths=[HERE/'gates'/ds/'COMPLETE.json' for ds in ('bf','epfl')]
            if all(p.exists() for p in paths):
                for p in paths:
                    gate=read(p);assert gate['status']=='COMPLETE' and gate['freeze_sha256']==freeze and len(gate['rows'])==8
                    assert {(r['n_pos'],r['arm']) for r in gate['rows']}=={(n,a) for n in (512,2048) for a in ('vcs_exact','vcs_perm','bce','nce')}
                    for row in gate['rows']:
                        assert sha(row['path'])==row['sha256'];result=read(row['path'])
                        assert result['status']=='COMPLETE' and result['freeze_sha256']==freeze
                manifest=HERE/'JOBS.json';jobs=read(manifest);assert len(jobs)==54
                assert all(r['freeze_sha256']==freeze for r in jobs)
                source=POOL/'POOL_SOURCES.json';sources=read(source)
                if str(manifest) not in sources:
                    sources.append(str(manifest));write(source,sources)
                write(HERE/'ADMISSION.json',dict(status='ADMITTED',utc=utc(),job=os.environ['SLURM_JOB_ID'],pool_job='1061067',
                      training_units=48,probe_units=6,freeze_sha256=freeze,manifest_sha256=sha(manifest),
                      gates=[dict(path=str(p),sha256=sha(p),job=read(p)['job']) for p in paths]))
                print('ADMITTED 48 training +6 FIT probes',flush=True);return
            write(HERE/'ADMISSION_STATUS.json',dict(status='WAITING_FOR_GPU_GATES',utc=utc(),job=os.environ['SLURM_JOB_ID'],
                  completed=[p.parent.name for p in paths if p.exists()],full_jobs_admitted=False))
            print(utc(),'WAITING_FOR_GPU_GATES',flush=True);time.sleep(60)
        write(HERE/'ADMISSION_STATUS.json',dict(status='STOPPED_OR_EXPIRED',utc=utc(),job=os.environ['SLURM_JOB_ID']))

if __name__=='__main__':main()
