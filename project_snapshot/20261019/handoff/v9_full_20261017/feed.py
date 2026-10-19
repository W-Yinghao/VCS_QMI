"""Bounded normal-QoS full-experiment feeder; runs inside a CPU Slurm allocation."""
from __future__ import annotations
import fcntl
import getpass
import os
import subprocess
import time
from pathlib import Path
from common import ROOT, frozen, read, sha, write
from summarize import summarize


def queue():
    p=subprocess.run(['squeue','-u',getpass.getuser(),'-h','-o','%i|%j|%T|%q|%b'],capture_output=True,text=True,timeout=30,check=True)
    return [dict(zip(('id','name','state','qos','tres'),line.split('|'))) for line in p.stdout.splitlines() if line.strip()]


def gpu_count(job):
    total=0
    for token in job['tres'].split(','):
        if 'gpu:' in token:
            try:total+=int(token.rsplit(':',1)[1])
            except ValueError:raise ValueError('Unrecognised GPU TRES '+token)
    return total


def main():
    lock=(ROOT/'FEED.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen()
    from validate import validate
    validate()
    lp=ROOT/'LAUNCH.json'; ledger=read(lp) if lp.exists() else {}
    groups=read(ROOT/'GROUPS.json'); started=time.monotonic(); misses={}
    while time.monotonic()-started < 3*86400:
        try:q=queue()
        except (subprocess.SubprocessError,OSError) as e:
            write(ROOT/'FEED_STATUS.json',{'status':'QUEUE_UNAVAILABLE_NO_SUBMISSION','error':str(e)})
            time.sleep(45);continue
        ids={j['id'] for j in q}; own=[j for j in q if j['name'].startswith('v9f_') and j['name']!='v9f_feed']
        # Reconcile by unique names before submission, including a previous uncertain sbatch outcome.
        for g in groups:
            matches=[j for j in q if j['name']=='v9f_'+g['id']]
            if len(matches)>1:raise RuntimeError('Duplicate live full group '+g['id'])
            if matches and g['id'] not in ledger:
                ledger[g['id']]={'job':matches[0]['id'],'attempt':1,'recovered':True};write(lp,ledger)
        candidates=[];failed=[]
        for g in groups:
            gd=ROOT/'results'/g['id'];rec=ledger.get(g['id'])
            if (gd/'DONE.json').exists():continue
            if rec is None:candidates.append(g);continue
            if rec['job'] in ids:misses.pop(g['id'],None);continue
            misses[g['id']]=misses.get(g['id'],0)+1
            if misses[g['id']]<3:continue
            ep=gd/'SCHED_EXIT.json';exit_rec=read(ep) if ep.exists() else {}
            # Only a recorded termination/resource exit permits one unchanged resumption.
            if exit_rec.get('job_id')==rec['job'] and exit_rec.get('exit_code') in (137,143) and rec.get('attempt',1)<2:
                candidates.append(g)
            else:failed.append(g['id'])
        report=summarize()
        normal=[j for j in q if j['qos']=='normal']
        status={'utc':time.strftime('%FT%TZ',time.gmtime()),'queue_total':len(q),'normal_queue_total':len(normal),
                'normal_gpu_running':sum(gpu_count(j) for j in normal if j['state']=='RUNNING'),
                'own_live':len(own),'own_pending':sum(j['state']=='PENDING' for j in own),'groups_submitted':len(ledger),
                'full_complete':sum(r['status']=='COMPLETE' for r in report['units']),'failed_groups':failed,'candidates':len(candidates),
                'feeder_job':os.environ.get('SLURM_JOB_ID')}
        write(ROOT/'FEED_STATUS.json',status)
        if all((ROOT/'results'/g['id']/'DONE.json').exists() or g['id'] in failed for g in groups):
            write(ROOT/'FEED_FINISHED.json',status);return 1 if failed else 0
        if candidates and len(normal)<27 and len(own)<8 and status['own_pending']<2:
            g=candidates[0];prev=ledger.get(g['id'],{})
            intent={'group':g['id'],'attempt':prev.get('attempt',0)+1,'utc':status['utc'],'freeze_sha256':sha(ROOT/'FROZEN.json')}
            write(ROOT/'SUBMIT_INTENT.json',intent)
            cmd=['sbatch','--parsable','--qos=normal','--job-name=v9f_'+g['id'],str(ROOT/'full.sbatch'),g['id']]
            try:p=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
            except (subprocess.SubprocessError,OSError) as e:
                write(ROOT/'SUBMIT_UNCERTAIN.json',{**intent,'error':str(e)});raise
            if p.returncode:
                write(ROOT/'LAST_SUBMIT_REJECTED.json',{**intent,'stdout':p.stdout,'stderr':p.stderr})
                # Explicit scheduler rejection is safe to retry; an ambiguous success is not.
                time.sleep(45);continue
            job=p.stdout.strip().split(';')[0]
            if not job.isdigit():
                write(ROOT/'SUBMIT_UNCERTAIN.json',{**intent,'stdout':p.stdout,'stderr':p.stderr});raise RuntimeError('Ambiguous sbatch response')
            ledger[g['id']]={**intent,'job':job};write(lp,ledger)
            print('SUBMITTED',g['id'],job,flush=True)
            continue
        time.sleep(45)
    raise RuntimeError('Feeder exceeded 3-day budget; inspect remaining full units')


if __name__=='__main__':raise SystemExit(main())
