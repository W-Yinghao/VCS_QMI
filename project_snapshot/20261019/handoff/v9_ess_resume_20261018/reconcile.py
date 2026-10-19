"""Recover one failed unit; let the existing feeder finish naturally; reconcile."""
import fcntl
import os
import sys
import time
import traceback
from recovery_common import *

def main():
    cfg=verify_bundle()
    sys.path.insert(0,str(SPLIT))
    import split_common as S
    from summarize import summarize
    with (BUNDLE/'LOOP.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (BUNDLE/'LAUNCH.json').exists(),'Recovery already submitted; inspect instead of duplicating')
        live=S.queue()
        require(not any(r['name']=='v9f_903_1_ess' for r in live),'Recovery writer already live')
        require(cfg['failed_job'] not in {r['id'] for r in live},'Prior writer still live')
        write(BUNDLE/'PREFLIGHT_CPU.json',{'utc':utc(),'status':'PASS',**preflight(cfg)})
        args=['sbatch','--parsable','--job-name=v9f_903_1_ess',str(BUNDLE/'unit.sbatch')]
        write(BUNDLE/'SUBMIT_INTENT.json',{'utc':utc(),'args':args})
        job=S.command(args).strip().split(';')[0]
        require(job.isdigit(),'Ambiguous submission: inspect queue, do not repeat')
        rec={'job':job,'name':'v9f_903_1_ess','attempt':2,'status':'SUBMITTED',
             'recovery':'diagnostic ESS serialization only','control_freeze_sha256':sha(BUNDLE/'FREEZE.json'),
             'previous_job':cfg['failed_job'],'utc':utc()}
        write(BUNDLE/'LAUNCH.json',rec)
        started=time.monotonic();missing=None;proof=None
        while time.monotonic()-started<70*3600:
            try:live=S.queue()
            except (RuntimeError,OSError) as error:
                write(BUNDLE/'STATUS.json',{'status':'QUERY_RETRY_NO_RECONCILIATION','utc':utc(),'error':repr(error)})
                time.sleep(30);continue
            ids={r['id'] for r in live}
            if job not in ids and proof is None:
                missing=missing or time.monotonic()
                if time.monotonic()-missing>=120:
                    proof=S.verify_unit_completion(cfg['group'],cfg['index'],rec,ids)
                    require(proof is not None,'Recovery completion unavailable')
                    write(BUNDLE/'RECOVERED_UNIT.json',proof)
            remaining=[r for r in live if r['name'].startswith('v9') and r['id']!=os.environ['SLURM_JOB_ID']]
            write(BUNDLE/'STATUS.json',{'utc':utc(),'status':'WAIT_V9_NATURAL_CLOSEOUT' if proof else 'RECOVERY_SUBMITTED',
                  'controller':os.environ['SLURM_JOB_ID'],'recovery_job':job,'recovery_verified':proof is not None,
                  'remaining_v9':remaining,'running_jobs_cancelled':0})
            if proof and not remaining:
                require(cfg['feeder_job'] not in ids,'Original feeder still active')
                with (FULL/'FEED.lock').open('a') as feed_lock:
                    fcntl.flock(feed_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    ledger=read(SPLIT/'UNIT_LEDGER.json');launch=read(FULL/'LAUNCH.json')
                    old=ledger[cfg['group']][str(cfg['index'])]
                    require(old['job']==cfg['failed_job'],'Unexpected intervening unit resubmission')
                    write(BUNDLE/'UNIT_LEDGER_BEFORE_RECONCILIATION.json',ledger)
                    rec.update(status='COMPLETE_VERIFIED',completion=proof,verified_utc=utc(),history=[old])
                    ledger[cfg['group']][str(cfg['index'])]=rec
                    for index,row in ledger[cfg['group']].items():
                        row['completion']=S.verify_unit_completion(cfg['group'],int(index),row,ids)
                        row['status']='COMPLETE_VERIFIED'
                    S.finalize_group(cfg['group'],ledger[cfg['group']],launch)
                    S.write(SPLIT/'UNIT_LEDGER.json',ledger)
                    previous=read(FULL/'FEED_FINISHED.json')
                    write(BUNDLE/'FEED_FINISHED_BEFORE_RECONCILIATION.json',previous)
                    require(set(previous.get('failed_groups',[]))<={'903_1'},'Other unresolved V9 failures')
                    groups=read(FULL/'GROUPS.json')
                    require(all((FULL/'results'/g['id']/'DONE.json').exists() for g in groups),'V9 groups missing')
                    report=summarize()
                    require(len(report['units'])==33 and all(u['status']=='COMPLETE' for u in report['units']),'Full33 not complete')
                    status={**previous,'utc':utc(),'feeder_job':os.environ['SLURM_JOB_ID'],'full_complete':33,
                            'groups_complete':18,'failed_groups':[],'own_live':0,'own_running':0,'own_pending':0,
                            'reconciled_after_natural_feeder_exit':cfg['feeder_job'],'recovery':str(BUNDLE),
                            'independent_closeout_still_required':True}
                    write(FULL/'FEED_STATUS.json',status);write(FULL/'FEED_FINISHED.json',status)
                    write(BUNDLE/'STATUS.json',{'status':'PASS_RECOVERY_AND_RECONCILIATION','utc':utc(),
                          'recovery_job':job,'full_complete':33,'independent_audit':'1046165 still required',
                          'running_jobs_cancelled':0})
                    return
            time.sleep(30)
        raise RuntimeError('Recovery monitor exceeded70hours')

if __name__=='__main__':
    try:main()
    except BaseException:
        write(BUNDLE/'ERROR.json',{'utc':utc(),'traceback':traceback.format_exc()});raise
