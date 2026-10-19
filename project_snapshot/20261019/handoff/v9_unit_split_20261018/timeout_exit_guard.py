"""Fill a missing trap record only from a proven TIMEOUT and real 9/15 signal.

Fixed target1044282; does not submit jobs or authorize any scientific retry.
The sole V9 feeder still owns the original one infrastructure resume.
"""
import os
import re
import subprocess
import time
from split_common import BUNDLE, ROOT, read, write, utc, require, queue, sha
JOB='1044282'
GROUP='902_2'


def normalize(text):
    fields=dict(re.findall(r'(\w+)=(\S+)',text))
    require(fields.get('JobId')==JOB and fields.get('JobName')=='v9f_'+GROUP,'Wrong controller job')
    require(fields.get('JobState')=='TIMEOUT','Only controller-confirmed timeout is eligible')
    code,signal=map(int,fields['ExitCode'].split(':'))
    canonical=128+signal if signal in (9,15) else code
    require(canonical in (137,143),'No eligible recorded exit signal/code')
    require(fields.get('EndTime') not in (None,'Unknown','N/A'),'Missing scheduler end time')
    return {'exit_code':canonical,'job_id':JOB,'utc':fields['EndTime']+'Z',
            'source':'SLURM_CONTROLLER_TIMEOUT_NOT_BATCH_TRAP',
            'slurm_state':fields['JobState'],'slurm_exit_code':fields['ExitCode'],
            'code_derivation':'128 + recorded signal9/15, or explicit recorded status137/143',
            'collector_job':os.environ.get('SLURM_JOB_ID')}


def main():
    require(bool(os.environ.get('SLURM_JOB_ID')),'CPU Slurm only')
    manifest=read(BUNDLE/'TIMEOUT_GUARD_FREEZE.json')
    require(sha(__file__)==manifest['script_sha256'],'Guard script changed')
    target=ROOT/'results'/GROUP/'SCHED_EXIT.json'
    begin=time.monotonic(); terminal=None; absent_since=None
    while time.monotonic()-begin<2*3600:
        p=subprocess.run(['scontrol','show','job','-o',JOB],text=True,capture_output=True,timeout=45,env={**os.environ,'TZ':'UTC'})
        if p.returncode==0:
            write(BUNDLE/'TIMEOUT_GUARD_CONTROLLER.json',{'utc':utc(),'query_timezone':'UTC','stdout':p.stdout})
            if 'JobState=TIMEOUT ' in p.stdout:
                terminal=normalize(p.stdout)
            elif any('JobState='+s+' ' in p.stdout for s in ('CANCELLED','COMPLETED','FAILED','NODE_FAIL')):
                write(BUNDLE/'TIMEOUT_GUARD_RESULT.json',{'status':'NO_ACTION_OTHER_TERMINAL_STATE','utc':utc(),'stdout':p.stdout}); return
        if target.exists() and read(target).get('job_id')==JOB:
            write(BUNDLE/'TIMEOUT_GUARD_RESULT.json',{'status':'ORIGINAL_TRAP_PRESENT_NO_ACTION','utc':utc(),'exit':read(target)}); return
        live={r['id'] for r in queue()}
        if terminal and JOB not in live:
            absent_since=absent_since or time.monotonic()
            if time.monotonic()-absent_since>=30:
                launch=read(ROOT/'LAUNCH.json')[GROUP]
                require(launch['job']==JOB and launch.get('attempt',1)==1,'Feeder already moved or retry budget changed')
                require(not (ROOT/'results'/GROUP/'FAILED.json').exists(),'Scientific failure cannot be repaired as timeout')
                if target.exists():
                    write(BUNDLE/'TIMEOUT_GUARD_PREVIOUS_EXIT.json',read(target))
                terminal['controller_evidence']=str(BUNDLE/'TIMEOUT_GUARD_CONTROLLER.json')
                terminal['controller_evidence_sha256']=sha(BUNDLE/'TIMEOUT_GUARD_CONTROLLER.json')
                terminal['observed_utc']=utc()
                write(target,terminal)
                write(BUNDLE/'TIMEOUT_GUARD_RESULT.json',{'status':'MISSING_TRAP_FILLED_FROM_CONTROLLER','utc':utc(),'exit':terminal})
                return
        time.sleep(15)
    raise RuntimeError('Timeout guard exceeded observation window; no synthetic exit written')


if __name__=='__main__': main()
