"""Preflight, queue replacements held, then migrate only at committed checkpoints."""
from __future__ import annotations
import fcntl
import os
import shutil
import time
import traceback
from split_common import *


def checkpoint(group_id, index):
    # CPU Slurm allocation only. Never deserialize checkpoints on the login host.
    require(bool(os.environ.get('SLURM_JOB_ID')), 'Slurm required')
    import torch
    torch.set_num_threads(1)
    ud = unit_path(group_id,index)
    verify_identity(ud,group_id,group_record(group_id)['units'][index])
    with (ud/'last.pt').open('rb') as fh:
        stat = os.fstat(fh.fileno())
        ck = torch.load(fh, map_location='cpu', weights_only=False)
    require(set(('model','opt','epoch','best','log','rng')) <= set(ck), 'Incomplete checkpoint')
    n = ck['epoch']+1
    require(0<n<=40 and [e['epoch'] for e in ck['log']['epochs']]==list(range(n)), 'Invalid checkpoint epoch sequence')
    require(bool(ck['model']) and bool(ck['opt']['state']) and bool(ck['opt']['param_groups']), 'Missing model/optimizer state')
    require(set(('python','numpy','torch','cuda')) <= set(ck['rng']), 'Missing checkpoint RNG')
    for value in ck['model'].values():
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            require(bool(torch.isfinite(value).all()), 'Nonfinite checkpoint tensor')
    best = max(ck['log']['epochs'],key=lambda e:e['dev']['rsum'])
    require(ck['best']['epoch']==best['epoch'] and abs(ck['best']['rsum']-best['dev']['rsum'])<1e-8,'Checkpoint best mismatch')
    manifest = read(ud/'run_manifest.json')
    require(manifest['seed']==903 and manifest['lr_mult']==1.0 and manifest['cfg']['num_epochs']==40
            and manifest['cfg']['warmup_epochs']==5 and manifest['n_pool']==145000,'Checkpoint recipe mismatch')
    return {'group':group_id,'unit_index':index,'completed_epochs':n,'remaining_epochs':40-n,
            'best':ck['best'],'checkpoint_mtime_ns':stat.st_mtime_ns,'checkpoint_size':stat.st_size,
            'identity_sha256':sha(ud/'identity.json'),'manifest_sha256':sha(ud/'run_manifest.json'),
            'model_optimizer_rng_present':True,'utc':utc()}


def wait_absent(job):
    start=time.monotonic()
    while job in {r['id'] for r in queue()}:
        require(time.monotonic()-start<180,'Cancellation did not finish: '+job)
        time.sleep(3)


def main():
    cfg=verify_control()
    require(not (BUNDLE/'ACTIVATION.json').exists(),'Transition already activated; do not repeat')
    require(not (BUNDLE/'UNIT_LEDGER.json').exists(),'Existing unit ledger requires reconciliation; do not duplicate')
    require(read(BUNDLE/'CONTROL_TESTS.json')['status']=='PASS','Control tests missing')
    rows=queue(); ids={r['id']:r for r in rows}
    require(ids.get(cfg['old_feeder'],{}).get('name')=='v9f_feed','Original feeder not as expected')
    launch=read(ROOT/'LAUNCH.json'); before={}
    for g, item in cfg['split_groups'].items():
        old=item['old_job']; require(launch[g]['job']==old,'Original launch changed')
        require(ids.get(old,{}).get('name')=='v9f_'+g and ids[old]['state']=='RUNNING','Original worker not running')
        require(not (unit_path(g,1)/'identity.json').exists(),'Second unit already initialized; revise transition')
        require(not (ROOT/'results'/g/'FAILED.json').exists(),'Unresolved scientific failure')
        before[g]=checkpoint(g,0)
    write(BUNDLE/'PREFLIGHT.json',{'status':'PASS','utc':utc(),'checkpoints':before,'queue':rows})
    write(BUNDLE/'LAUNCH_BEFORE.json',launch)
    ledger={g:{} for g in cfg['split_groups']}
    write(BUNDLE/'UNIT_LEDGER.json',ledger)
    # Held jobs are inert; no original worker is stopped until all four replacements exist.
    for i in (0,1):
        for g in cfg['split_groups']:
            while True:
                rows=queue()
                if capacity(rows,cfg):
                    name=f'v9f_{g}_u{i}'
                    require(not any(r['name']==name for r in rows),'Duplicate split job name')
                    job=submit(['sbatch','--parsable','--hold','--job-name='+name,str(BUNDLE/'unit.sbatch'),g,str(i)],
                               {'group':g,'unit_index':i,'attempt':1,'phase':'PREPARE_HELD'})
                    if job:
                        ledger[g][str(i)]={'job':job,'attempt':1,'name':name,'status':'HELD_FOR_CHECKPOINT_MIGRATION','utc':utc()}
                        write(BUNDLE/'UNIT_LEDGER.json',ledger); break
                write(BUNDLE/'TRANSITION_STATUS.json',{'status':'WAIT_SUBMIT_CAPACITY','utc':utc(),'group':g,'index':i})
                time.sleep(15)
    command(['scancel',cfg['old_feeder']]); wait_absent(cfg['old_feeder'])
    with (ROOT/'FEED.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(BUNDLE/'TRANSITION_STATUS.json',{'status':'WAIT_NEXT_COMMITTED_CHECKPOINT','utc':utc(),'old_feeder_stopped':cfg['old_feeder']})
        migrated={}; started=time.monotonic()
        while len(migrated)<len(cfg['split_groups']):
            require(time.monotonic()-started<2400,'No new checkpoint within 40 minutes; inspect before proceeding')
            rows=queue(); live={r['id'] for r in rows}
            for g,item in cfg['split_groups'].items():
                if g in migrated: continue
                old=item['old_job']; ud=unit_path(g,0)
                require(old in live,'Original worker disappeared before controlled migration')
                if (ud/'last.pt').stat().st_mtime_ns<=before[g]['checkpoint_mtime_ns']:
                    continue
                # last.pt is atomically replaced only after the complete epoch + DEV.
                ready=checkpoint(g,0)
                require(ready['completed_epochs']>before[g]['completed_epochs'],'New checkpoint did not advance')
                require(not (unit_path(g,1)/'identity.json').exists(),'Second writer already started')
                write(BUNDLE/'transitions'/g/'CANCEL_INTENT.json',{'old_job':old,'utc':utc(),'checkpoint':ready})
                command(['scancel',old]); wait_absent(old)
                after=checkpoint(g,0)
                require(after['completed_epochs']>=ready['completed_epochs'],'Checkpoint regressed')
                require(not (unit_path(g,1)/'identity.json').exists(),'Original touched second unit; inspect before release')
                archive=BUNDLE/'checkpoints_before_split'/g; archive.mkdir(parents=True,exist_ok=True)
                artifacts={}
                for name in ('last.pt','best.pt','run_manifest.json','identity.json','progress.jsonl'):
                    shutil.copy2(ud/name,archive/name); artifacts[name]=sha(archive/name)
                for name in ('hardware.json','SCHED_EXIT.json'):
                    p=ROOT/'results'/g/name
                    if p.exists(): shutil.copy2(p,archive/name)
                # Verify best.pt corresponds to the checkpoint, not an interrupted future epoch.
                import torch
                best=torch.load(archive/'best.pt',map_location='cpu',weights_only=False)
                require(best['epoch']==after['best']['epoch'] and best['best']==after['best'],'Best checkpoint mismatch')
                migrated[g]={'old_job':old,'checkpoint':after,'archive':str(archive),'artifact_sha256':artifacts,
                             'new_jobs':{i:r['job'] for i,r in ledger[g].items()},'utc':utc(),
                             'hardware_before':read(archive/'hardware.json'),'physical_gpu_pairing_preserved':False}
                write(BUNDLE/'transitions'/g/'MIGRATED.json',migrated[g])
                launch[g]={**launch[g],'split':True,'original_serial_job':old,'unit_jobs':migrated[g]['new_jobs'],
                           'unit_ledger':str(BUNDLE/'UNIT_LEDGER.json'),'amendment':str(BUNDLE/'SCHEDULING_AMENDMENT.json')}
                write(ROOT/'LAUNCH.json',launch)
                for i,rec in ledger[g].items():
                    command(['scontrol','release',rec['job']])
                    rec.update(status='SUBMITTED',released_utc=utc())
                    write(BUNDLE/'UNIT_LEDGER.json',ledger)
                write(BUNDLE/'TRANSITION_STATUS.json',{'status':'MIGRATING','utc':utc(),'migrated':migrated})
            if len(migrated)<len(cfg['split_groups']): time.sleep(3)
        write(BUNDLE/'ACTIVATION.json',{'status':'SPLIT_ACTIVE','utc':utc(),'feeder_job':os.environ['SLURM_JOB_ID'],
              'retired_feeder':cfg['old_feeder'],'groups':migrated,'science_freeze_sha256':cfg['experiment_freeze_sha256'],
              'control_freeze_sha256':sha(BUNDLE/'CONTROL_FREEZE.json'),'v10_dispatcher_unchanged':'1046189',
              'independent_closeout_unchanged':'1046165'})
    print('SPLIT_ACTIVE',utc(),flush=True)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        write(BUNDLE/'TRANSITION_ERROR.json',{'utc':utc(),'error':repr(exc),'traceback':traceback.format_exc()})
        raise
