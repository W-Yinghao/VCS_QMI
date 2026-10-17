"""CPU Slurm watcher: reserve the next two slots for BF1, account, summarize.

Never stops a running GPU job. Only pending v9f_* jobs temporarily held by this
watcher are released by it. The unchanged V9 frozen feeder remains in service.
"""
from __future__ import annotations
import argparse
import getpass
import os
import signal
import subprocess
import time
from common import ROOT,FAMILIES,append,freeze,lines,lock,read,sha,utc,write


def command(args):
    return subprocess.run(args,capture_output=True,text=True,timeout=30,check=True).stdout.strip()


def queue():
    s=command(['squeue','-r','-u',getpass.getuser(),'-h','-o','%i|%j|%T|%q|%b|%R'])
    return [dict(zip(('id','name','state','qos','tres','reason'),r.split('|',5))) for r in s.splitlines() if r]


def summary():
    cal=[];final=[];fail=[]
    for family in FAMILIES:
        fd=ROOT/'results'/'CAL'/family
        for p in sorted(fd.glob('candidate_*/COMPLETE.json')):
            r=read(p);cal.append({'family':family,'unit':p.parent.name,'steps':r['steps'],'selected':r['selected']})
        for seed in (1001,1002,1003):
            p=ROOT/'results'/'FINAL_DEV'/family/f'seed_{seed}'/'COMPLETE.json'
            if p.exists():
                r=read(p);final.append({'family':family,'seed':seed,'DEV32':r['DEV32'],'DEV24':r['DEV24']})
    for p in sorted((ROOT/'failures').glob('*.json')):fail.append(read(p))
    out={'utc':utc(),'cal_complete':len(cal),'cal_planned':18,'final_complete':len(final),'final_planned':12,
         'cal':cal,'final':final,'failures':fail,
         'note':'Progress summary only; final scientific report must verify identities/artifacts and include grouped uncertainty.'}
    write(ROOT/'READOUT.json',out);return out


def reconcile_jobs(launch,live_ids):
    accounted={r['scheduler_id'] for r in lines(ROOT/'SCHEDULER_ACCOUNTING.jsonl')}
    ids=[launch['gate_job']]+[f"{launch['array_job']}_{i}" for i in range(4)]
    for jid in ids:
        if jid in live_ids or jid in accounted:continue
        text=command(['sacct','-X','-j',jid,'-n','-P','--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES'])
        match=[x.split('|') for x in text.splitlines() if x.split('|')[0]==jid]
        if not match:continue
        a=match[0]
        if a[1] in ('RUNNING','PENDING','COMPLETING'):continue
        append(ROOT/'SCHEDULER_ACCOUNTING.jsonl',{'scheduler_id':jid,'utc':utc(),'state':a[1],
               'exit_code':a[2],'allocated_seconds':int(a[3]),'allocated_tres':a[4],
               'note':'Entire allocation, including CAL and final phases; do not add its total to CAL phase cost.'})


def main():
    freeze();launch=read(ROOT/'LAUNCH.json')
    held_path=ROOT/'V9_PENDING_HELD.json';held=read(held_path).get('jobs',[]) if held_path.exists() else []
    begin=time.monotonic()
    def terminate(signum,frame):raise SystemExit(f'Watcher received signal {signum}; releasing its pending holds')
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    def release(jid):
        command(['scontrol','release',jid]);append(ROOT/'SCHEDULING_EVENTS.jsonl',{'utc':utc(),'action':'release','job':jid})
    try:
        while time.monotonic()-begin<3*86400:
            try:
                q=queue();live={j['id'] for j in q}
                own=[j for j in q if j['name'].startswith('v10bf_') and 'gpu:1' in j['tres']]
                running=sum(j['state']=='RUNNING' for j in own)
                # A failed gate cannot leave downstream jobs indefinitely held.
                if (ROOT/'GPU_GATE_FAILURE.json').exists() and not any(j['id']==launch['gate_job'] for j in q):
                    command(['scancel',launch['array_job']])
                    write(ROOT/'DISPATCH_BLOCKED.json',{'utc':utc(),'reason':'GPU execution gate failed; full array cancelled before training'})
                    break
                pending_ready=any(j['state']=='PENDING' and j['reason'] not in ('(DependencyNeverSatisfied)',) for j in own)
                reserve=pending_ready and running<2
                if reserve:
                    for j in q:
                        if (j['name'].startswith('v9f_') and j['name']!='v9f_feed' and j['state']=='PENDING'
                            and j['id'] not in held and j['reason']!='(JobHeldUser)'):
                            command(['scontrol','hold',j['id']]);held.append(j['id'])
                            write(held_path,{'utc':utc(),'jobs':held})
                            append(ROOT/'SCHEDULING_EVENTS.jsonl',{'utc':utc(),'action':'hold_pending_for_V10','job':j['id']})
                else:
                    for jid in held[:]:
                        if jid in live:release(jid)
                        held.remove(jid);write(held_path,{'utc':utc(),'jobs':held})
                result=summary();reconcile_jobs(launch,live)
                state={'utc':utc(),'watch_job':os.environ['SLURM_JOB_ID'],'v10_running':running,
                       'v10_live':own,'v9_pending_temporarily_held':held,'cal_complete':result['cal_complete'],
                       'final_complete':result['final_complete'],'source_freeze':sha(ROOT/'FROZEN.json')}
                write(ROOT/'WATCH_STATUS.json',state)
                if not own and all(j not in live for j in [launch['gate_job']]+[f"{launch['array_job']}_{i}" for i in range(4)]):
                    write(ROOT/'WATCH_FINISHED.json',state);break
            except (subprocess.SubprocessError,OSError) as e:
                write(ROOT/'WATCH_LAST_ERROR.json',{'utc':utc(),'error':repr(e)})
            time.sleep(30)
    finally:
        for jid in held[:]:
            try:release(jid);held.remove(jid)
            except subprocess.SubprocessError:pass
        write(held_path,{'utc':utc(),'jobs':held})
    return 0


if __name__=='__main__':
    with lock(ROOT/'WATCH.lock'):raise SystemExit(main())
