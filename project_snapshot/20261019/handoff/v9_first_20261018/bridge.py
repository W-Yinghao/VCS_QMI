"""Preserve running jobs; hold pending V10 until audited V9 completion."""
from __future__ import annotations
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

BUNDLE=Path(__file__).resolve().parent
PROJECT=BUNDLE.parents[1]
V9=PROJECT/'handoff/v9_full_20261017'
V10=PROJECT/'handoff/v10_bf1_20261017'
DISPATCH=PROJECT/'handoff/v10_parallel8_20261018'
AUDIT=PROJECT/'handoff/v9_overlap_v10_20261018'
POLICY='v9_first_20261018_2143'
ACTIVE=PROJECT/'handoff/ACTIVE_SCHEDULING_POLICY.json'
PAUSE=DISPATCH/'PAUSE_SUBMISSIONS'


def read(p): return json.loads(Path(p).read_text())
def utc(): return time.strftime('%FT%TZ',time.gmtime())
def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()
def write(p,j):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(f'.{p.name}.{os.getpid()}.tmp')
    t.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); t.replace(p)
def command(args):
    p=subprocess.run(args,text=True,capture_output=True,timeout=45)
    require(p.returncode==0,f'{args}: {p.stderr}')
    return p.stdout

def queue():
    out=command(['squeue','-u','yinwang','-r','-h','-o','%i|%j|%T|%q|%b|%R'])
    rows=[]
    for line in out.splitlines():
        fields=line.split('|',5); require(len(fields)==6,'Malformed queue row')
        rows.append(dict(zip(('id','name','state','qos','gres','reason'),fields)))
    return rows

def v10_pending(rows):
    return [r for r in rows if r['name'].startswith('v10p8_') and r['state']=='PENDING']

def verify_freeze():
    for name,digest in read(BUNDLE/'FREEZE.json')['files'].items():
        require(sha(BUNDLE/name)==digest,'Priority control changed: '+name)
    cfg=read(BUNDLE/'CONFIG.json')
    for name,digest in cfg['external_freezes'].items():
        require(sha(PROJECT/name)==digest,'Referenced freeze changed: '+name)
    return cfg

def policy_current():
    return ACTIVE.exists() and read(ACTIVE).get('policy_id')==POLICY

def load_auditor():
    cfg=read(BUNDLE/'CONFIG.json')
    require(sha(AUDIT/'bridge.py')==cfg['auditor_sha256'],'Auditor changed')
    spec=importlib.util.spec_from_file_location('frozen_v9_auditor',AUDIT/'bridge.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.verify_bundle(); return module

def closeout_ready(rows, feed, status, audit):
    reasons=[]
    if any(r['name'].startswith('v9') or r['id']=='1046165' for r in rows): reasons.append('V9_JOBS_STILL_LIVE')
    if feed.get('full_complete')!=33 or feed.get('failed_groups'): reasons.append('V9_FEED_NOT_SUCCESSFUL')
    if status.get('state')!='PASS_V9_CLOSEOUT_INDEPENDENT': reasons.append('V9_AUDIT_NOT_FINISHED')
    if audit.get('status')!='PASS' or len(audit.get('units',[]))!=33: reasons.append('V9_AUDIT_NOT_PASS33')
    return not reasons,reasons

def validate_closeout(audit,module):
    module.verify_science()
    cfg=read(BUNDLE/'CONFIG.json')
    require(audit['v9_freeze_sha256']==cfg['v9_freeze_sha256'],'Audited V9 identity mismatch')
    require(audit['v10_freeze_sha256']==cfg['v10_freeze_sha256'],'Audited V10 identity mismatch')
    require(sha(AUDIT/'V9_CLOSEOUT_CN.md')==audit['report_sha256'],'Closeout report changed')
    groups=read(V9/'GROUPS.json')
    expected={(g['id'],u['method'],u['scale']) for g in groups for u in g['units']}
    require(expected=={(u['group'],u['method'],u['scale']) for u in audit['units']},'Audited matrix differs')
    for g in groups:
        d=read(V9/'results'/g['id']/'DONE.json')
        require(d['freeze_sha256']==cfg['v9_freeze_sha256'],'Group identity changed')
        require(not (V9/'results'/g['id']/'FAILED.json').exists(),'Unresolved V9 failure')
    for u in audit['units']:
        p=V9/'results'/u['group']/f"{u['method']}_{u['scale']}"/'full_result.json'
        require(sha(p)==u['full_result_sha256'],'Result changed after audit')
    require(audit['held_test_read'] is False,'Unexpected held-data read')

def hold_pending(rows,ledger):
    allowed={u.get('job') for u in read(DISPATCH/'LEDGER.json')['units'] if u.get('job')}
    for row in v10_pending(rows):
        job=row['id']; require(job in allowed,'Unknown V10 pending job; inspect ownership')
        if row['reason'] in ('(JobHeldUser)','(JobHeldAdmin)'):
            if job not in ledger['held_by_this_policy']:
                ledger['preexisting_holds'][job]=row
            continue
        # Check immediately before mutation. A start racing with hold is preserved.
        latest=next((r for r in queue() if r['id']==job),None)
        if not latest or latest['state']!='PENDING': continue
        write(BUNDLE/'HOLD_INTENT.json',{'utc':utc(),'job':job,'observed':latest})
        p=subprocess.run(['scontrol','hold',job],capture_output=True,text=True,timeout=45)
        after=next((r for r in queue() if r['id']==job),None)
        if after and after['state']=='RUNNING':
            ledger['raced_running_preserved'][job]=after
        else:
            require(p.returncode==0,'Hold failed: '+p.stderr)
            require(after is not None and after['state']=='PENDING' and after['reason']=='(JobHeldUser)','Hold not confirmed')
            ledger['held_by_this_policy'][job]={'utc':utc(),'before':latest,'after':after}
        write(BUNDLE/'LEDGER.json',ledger)

def release_after_closeout(ledger,audit,module):
    verify_freeze(); validate_closeout(audit,module)
    require(policy_current(),'Policy superseded before release')
    require(PAUSE.exists() and read(PAUSE).get('policy_id')==POLICY,'Pause belongs to a different decision')
    rows=queue()
    ready,_=closeout_ready(rows,read(V9/'FEED_FINISHED.json'),read(AUDIT/'STATUS.json'),audit)
    require(ready,'V9 readiness changed during validation')
    for job in ledger['held_by_this_policy']:
        require(policy_current(),'Policy superseded during release')
        row=next((r for r in queue() if r['id']==job),None)
        if row and row['state']=='PENDING' and row['reason']=='(JobHeldUser)':
            command(['scontrol','release',job])
            ledger.setdefault('released',{})[job]={'utc':utc(),'reason':'V9_CLOSEOUT_PASS'}
            write(BUNDLE/'LEDGER.json',ledger)
        elif row and row['reason']=='(JobHeldAdmin)':
            raise RuntimeError('Administrative hold must not be removed')
    require(policy_current() and read(PAUSE).get('policy_id')==POLICY,'Pause ownership changed')
    PAUSE.replace(BUNDLE/'PAUSE_SUBMISSIONS_RELEASED.json')
    result={'status':'V9_COMPLETE_V10_RESUMED','utc':utc(),'policy_id':POLICY,'job':os.environ['SLURM_JOB_ID'],
            'v9_closeout':str(AUDIT/'CLOSEOUT.json'),'v9_closeout_sha256':sha(AUDIT/'CLOSEOUT.json'),
            'v10_dispatcher':'1046189','released_jobs':ledger.get('released',{}),
            'implementation_work_order':str(BUNDLE/'V10_REMAINING_WORK.json'),
            'new_baselines_implemented':False,'new_baselines_require_coding_validation_and_freeze':True}
    write(BUNDLE/'RESUMED.json',result)
    write(PROJECT/'handoff/V10_IMPLEMENTATION_READY.json',result)
    write(BUNDLE/'STATUS.json',result)

def main():
    verify_freeze(); module=load_auditor()
    with (BUNDLE/'LOOP.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (BUNDLE/'LEDGER.json').exists(),'Existing controller ledger; do not start duplicate loop')
        rows=queue()
        ledger={'utc':utc(),'job':os.environ['SLURM_JOB_ID'],'policy_id':POLICY,
                'running_preserved_at_start':[r for r in rows if r['state']=='RUNNING'],
                'held_by_this_policy':{},'preexisting_holds':{},'raced_running_preserved':{}}
        write(BUNDLE/'LEDGER.json',ledger); begin=time.monotonic()
        while time.monotonic()-begin<94*3600:
            if not policy_current():
                write(BUNDLE/'STATUS.json',{'status':'SUPERSEDED_NO_RELEASE','utc':utc()}); return
            require(PAUSE.exists() and read(PAUSE).get('policy_id')==POLICY,'V10 pause removed or replaced; inspect policy')
            try:
                rows=queue(); hold_pending(rows,ledger)
            except (RuntimeError,OSError,subprocess.SubprocessError) as exc:
                write(BUNDLE/'STATUS.json',{'status':'QUERY_OR_HOLD_RETRY_NO_RELEASE','utc':utc(),'error':repr(exc)})
                time.sleep(30); continue
            rows=queue()
            feed=read(V9/'FEED_FINISHED.json') if (V9/'FEED_FINISHED.json').exists() else {}
            status=read(AUDIT/'STATUS.json') if (AUDIT/'STATUS.json').exists() else {}
            audit=read(AUDIT/'CLOSEOUT.json') if (AUDIT/'CLOSEOUT.json').exists() else {}
            ready,reasons=closeout_ready(rows,feed,status,audit)
            write(BUNDLE/'STATUS.json',{'status':'V9_PRIORITY_WAIT_CLOSEOUT','utc':utc(),'job':os.environ['SLURM_JOB_ID'],
                  'policy_id':POLICY,'held_jobs':list(ledger['held_by_this_policy']),
                  'v10_running_preserved':[r for r in rows if r['name'].startswith('v10p8_') and r['state']=='RUNNING'],
                  'v9_jobs':[r for r in rows if r['name'].startswith('v9')],
                  'release_ready':ready,'release_wait_reasons':reasons,'v10_new_submissions_paused':True})
            if ready:
                release_after_closeout(ledger,audit,module); return
            time.sleep(30)
        raise RuntimeError('Priority monitor exceeded94hours')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        write(BUNDLE/'ERROR.json',{'utc':utc(),'error':repr(exc),'traceback':traceback.format_exc()}); raise
