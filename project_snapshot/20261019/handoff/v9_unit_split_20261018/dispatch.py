"""One V9 owner: split units plus unchanged recovery for all other groups."""
from __future__ import annotations
import fcntl
import os
import time
import traceback
from split_common import *
from summarize import summarize


def main():
    cfg=verify_control()
    require(read(BUNDLE/'ACTIVATION.json')['status']=='SPLIT_ACTIVE','Incomplete transition')
    with (ROOT/'FEED.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        groups=read(ROOT/'GROUPS.json'); launch=read(ROOT/'LAUNCH.json')
        ledger=read(BUNDLE/'UNIT_LEDGER.json'); missing={}; started=time.monotonic()
        require(len(groups)==18 and sum(len(g['units']) for g in groups)==33,'Matrix changed')
        while time.monotonic()-started<cfg['max_hours']*3600:
            try: rows=queue()
            except (RuntimeError,OSError,subprocess.SubprocessError) as exc:
                write(ROOT/'FEED_STATUS.json',{'status':'QUEUE_UNAVAILABLE_NO_SUBMISSION','utc':utc(),
                      'feeder_job':os.environ['SLURM_JOB_ID'],'error':repr(exc)})
                time.sleep(30); continue
            ids={r['id'] for r in rows}; candidates=[]; failed=[]
            # Reconciliation by exact name forbids duplicate groups/units or silent job adoption.
            expected={f'v9f_{g["id"]}':launch[g['id']]['job'] for g in groups if g['id'] not in ledger}
            expected.update({rec['name']:rec['job'] for units in ledger.values() for rec in units.values()})
            for name,job in expected.items():
                found=[r for r in rows if r['name']==name]
                require(len(found)<=1 and (not found or found[0]['id']==job),'Duplicate/mismatched writer: '+name)
            for g,units in ledger.items():
                require(cfg['split_groups'][g]['old_job'] not in ids,'Retired serial writer reappeared')
                for index,rec in units.items():
                    key=g+'/'+index; i=int(index)
                    if rec.get('status')=='COMPLETE_VERIFIED': continue
                    if rec['job'] in ids:
                        missing.pop(key,None); continue
                    first=missing.setdefault(key,time.monotonic())
                    if time.monotonic()-first<cfg['exit_record_grace_seconds']: continue
                    ex=unit_exit(g,i,rec['job'])
                    failure=BUNDLE/'units'/g/index/('FAILED_'+rec['job']+'.json')
                    if ex.get('job_id')==rec['job'] and ex.get('exit_code')==0 and not failure.exists():
                        try:
                            proof=verify_unit_completion(g,i,rec,ids)
                            require(proof is not None,'Completion raced with live job')
                            rec.update(status='COMPLETE_VERIFIED',completion=proof,verified_utc=utc())
                        except Exception as exc:
                            rec.update(status='FAILED_REVIEW_REQUIRED',error=repr(exc)); failed.append(g)
                        write(BUNDLE/'UNIT_LEDGER.json',ledger)
                    elif (ex.get('job_id')==rec['job'] and ex.get('exit_code') in (137,143)
                          and rec['attempt']<2 and not failure.exists()):
                        candidates.append({'group':g,'index':i,'previous':rec})
                    else:
                        rec.update(status='FAILED_REVIEW_REQUIRED',exit=ex)
                        failed.append(g); write(BUNDLE/'UNIT_LEDGER.json',ledger)
                if all(r.get('status')=='COMPLETE_VERIFIED' for r in units.values()):
                    if not (ROOT/'results'/g/'DONE.json').exists():
                        finalize_group(g,units,launch)
            for g in groups:
                gid=g['id']; gd=ROOT/'results'/gid
                if gid in ledger or (gd/'DONE.json').exists(): continue
                rec=launch[gid]
                if rec['job'] in ids:
                    missing.pop(gid,None); continue
                first=missing.setdefault(gid,time.monotonic())
                if time.monotonic()-first<cfg['exit_record_grace_seconds']: continue
                ep=gd/'SCHED_EXIT.json'; ex=read(ep) if ep.exists() else {}
                if (ex.get('job_id')==rec['job'] and ex.get('exit_code') in (137,143)
                        and rec.get('attempt',1)<2 and not (gd/'FAILED.json').exists()):
                    # The original one infrastructure resume remains available to 902_2.
                    candidates.insert(0,{'group':gid,'index':None,'previous':rec})
                else: failed.append(gid)
            report=summarize(); normal=[r for r in rows if r['qos']=='normal']
            own=[r for r in normal if r['name'].startswith('v9f_') and gpu_count(r)>0]
            status={'utc':utc(),'feeder_job':os.environ['SLURM_JOB_ID'],'queue_total':len(rows),
                    'normal_queue_total':len(normal),'normal_gpu_running':sum(gpu_count(r) for r in normal if r['state']=='RUNNING'),
                    'own_live':len(own),'own_running':sum(r['state']=='RUNNING' for r in own),
                    'own_pending':sum(r['state']=='PENDING' for r in own),'groups_submitted':len(launch),
                    'groups_not_submitted':0,'full_complete':sum(r['status']=='COMPLETE' for r in report['units']),
                    'groups_complete':sum((ROOT/'results'/g['id']/'DONE.json').exists() for g in groups),
                    'failed_groups':sorted(set(failed)),'candidates':len(candidates),'target_pending':cfg['target_pending'],
                    'max_normal_jobs':cfg['max_normal_jobs'],'max_own_live':cfg['max_own_live'],'poll_seconds':30,
                    'array_tasks_counted_separately':True,'dispatch_record':str(BUNDLE/'ACTIVATION.json'),
                    'split_unit_ledger':str(BUNDLE/'UNIT_LEDGER.json'),'submission_capacity_available':capacity(rows,cfg)}
            write(ROOT/'FEED_STATUS.json',status); write(BUNDLE/'STATUS.json',status)
            if all((ROOT/'results'/g['id']/'DONE.json').exists() or g['id'] in failed for g in groups):
                write(ROOT/'FEED_FINISHED.json',status)
                return 1 if failed else 0
            if candidates and capacity(rows,cfg):
                c=candidates[0]; g=c['group']; i=c['index']; previous=c['previous']
                intent={'group':g,'unit_index':i,'attempt':previous.get('attempt',1)+1,'utc':utc(),
                        'freeze_sha256':cfg['experiment_freeze_sha256'],'previous_job':previous['job'],
                        'control_freeze_sha256':sha(BUNDLE/'CONTROL_FREEZE.json')}
                name='v9f_'+g if i is None else f'v9f_{g}_u{i}'
                script=ROOT/'full.sbatch' if i is None else BUNDLE/'unit.sbatch'
                args=['sbatch','--parsable','--qos=normal','--job-name='+name,str(script),g]
                if i is not None: args.append(str(i))
                job=submit(args,intent)
                if job:
                    rec={**intent,'job':job,'name':name,'status':'SUBMITTED',
                         'history':previous.get('history',[])+[previous]}
                    if i is None:
                        launch[g]=rec; write(ROOT/'LAUNCH.json',launch)
                    else:
                        ledger[g][str(i)]=rec; write(BUNDLE/'UNIT_LEDGER.json',ledger)
                    missing.pop(g if i is None else g+'/'+str(i),None)
                    print('SUBMITTED',g,i,job,flush=True); continue
            time.sleep(30)
        raise RuntimeError('V9 dispatch exceeded 72 hours')


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        write(BUNDLE/'LOOP_ERROR.json',{'utc':utc(),'error':repr(exc),'traceback':traceback.format_exc()}); raise
