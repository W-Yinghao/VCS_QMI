"""Scheduling-only helpers; scientific imports occur only in Slurm jobs."""
from __future__ import annotations
import getpass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parent / 'v9_full_20261017'
sys.path.insert(0, str(ROOT))
from common import read, sha, frozen
from feed import gpu_count


def write(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(f'.{p.name}.{os.getpid()}.tmp')
    t.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    t.replace(p)


def utc():
    return time.strftime('%FT%TZ', time.gmtime())


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def command(args, timeout=45):
    p = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    require(p.returncode == 0, f'{args}: rc={p.returncode}: {p.stderr}')
    return p.stdout


def queue():
    output = command(['squeue', '-r', '-u', getpass.getuser(), '-h', '-o', '%i|%j|%T|%q|%b|%R'])
    rows = []
    for line in output.splitlines():
        fields = line.split('|', 5)
        require(len(fields) == 6, 'Malformed scheduler row')
        rows.append(dict(zip(('id', 'name', 'state', 'qos', 'tres', 'reason'), fields)))
    return rows


def check_old_absent(job):
    require(job not in {r['id'] for r in queue()}, 'Original serial writer still live: '+job)


def verify_control():
    manifest = read(BUNDLE/'CONTROL_FREEZE.json')
    for name, digest in manifest['files'].items():
        require(sha(BUNDLE/name) == digest, 'Control file changed: '+name)
    cfg = read(BUNDLE/'CONFIG.json')
    require(sha(ROOT/'FROZEN.json') == cfg['experiment_freeze_sha256'], 'Science freeze changed')
    frozen()
    return cfg


def group_record(group_id):
    return next(g for g in read(ROOT/'GROUPS.json') if g['id'] == group_id)


def unit_path(group_id, index):
    u = group_record(group_id)['units'][index]
    return ROOT/'results'/group_id/f"{u['method']}_{u['scale']}"


def expected_identity(group_id, unit):
    m = read(ROOT/'FROZEN.json')
    return {'group':group_id,'seed':group_record(group_id)['seed'],**unit,'epochs':40,
            'freeze_sha256':sha(ROOT/'FROZEN.json'),'protocol_sha256':m['code_sha256']['PROTOCOL_CN.md'],
            'oracle':unit['method'].startswith('Oracle'),'selection':'best full DEV rSum; earliest tie',
            'dataset':'f30k full train','design_rho':0.4,'learning_rate_multiplier':1.0}


def verify_identity(path, group_id, unit):
    expected = expected_identity(group_id, unit)
    require(read(path/'identity.json') == expected, 'Identity mismatch: '+str(path))
    if (path/'full_result.json').exists():
        require(read(path/'full_result.json')['identity'] == expected, 'Full-result identity mismatch')


def verify_full(group_id, index):
    spec = importlib.util.spec_from_file_location('v9_closeout_verifier', BUNDLE.parent/'v9_overlap_v10_20261018/bridge.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # This verifier is independently frozen by the already-running closeout job.
    require(sha(BUNDLE.parent/'v9_overlap_v10_20261018/bridge.py') == read(BUNDLE/'CONFIG.json')['closeout_verifier_sha256'], 'Closeout verifier changed')
    g = group_record(group_id); u = g['units'][index]; p = unit_path(group_id,index)
    verify_identity(p,group_id,u)
    return module.verify_unit(p,g,u,binary=True)


def capacity(rows, cfg):
    normal = [r for r in rows if r['qos']=='normal']
    own = [r for r in normal if r['name'].startswith('v9f_') and gpu_count(r)>0]
    return (len(normal)<cfg['max_normal_jobs'] and len(own)<cfg['max_own_live']
            and sum(r['state']=='PENDING' for r in own)<cfg['target_pending'])


def submit(args, intent):
    require(not (BUNDLE/'SUBMIT_UNCERTAIN.json').exists(), 'Unresolved ambiguous submission; reconcile first')
    write(BUNDLE/'SUBMIT_INTENT.json', {**intent,'command':args,'utc':utc()})
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=45)
    except (subprocess.SubprocessError,OSError) as exc:
        write(BUNDLE/'SUBMIT_UNCERTAIN.json',{**intent,'error':repr(exc)}); raise
    if p.returncode:
        write(BUNDLE/'LAST_SUBMIT_REJECTED.json',{**intent,'stdout':p.stdout,'stderr':p.stderr,'utc':utc()})
        return None
    job = p.stdout.strip().split(';')[0]
    if not job.isdigit():
        write(BUNDLE/'SUBMIT_UNCERTAIN.json',{**intent,'stdout':p.stdout,'stderr':p.stderr})
        raise RuntimeError('Ambiguous submission; no automatic retry')
    return job


def unit_exit(group_id, index, job):
    p = BUNDLE/'units'/group_id/str(index)/('EXIT_'+job+'.json')
    return read(p) if p.exists() else {}


def verify_unit_completion(group_id,index,rec,live_ids):
    if rec['job'] in live_ids:
        return None
    directory = BUNDLE/'units'/group_id/str(index)
    ex = unit_exit(group_id,index,rec['job'])
    require(ex.get('job_id') == rec['job'] and ex.get('exit_code') == 0, 'Unit exit not successful')
    require(not (directory/('FAILED_'+rec['job']+'.json')).exists(), 'Unit scientific failure marker')
    done = read(directory/'DONE.json')
    require(done['job_id'] == rec['job'] and done['unit_index'] == index and done['group'] == group_id, 'Unit DONE mismatch')
    require(done['freeze_sha256']==sha(ROOT/'FROZEN.json') and done['units']==[str(unit_path(group_id,index)/'full_result.json')], 'Unit DONE provenance mismatch')
    result = verify_full(group_id,index)
    return {'exit':ex,'result':result,'done_sha256':sha(directory/'DONE.json')}


def finalize_group(group_id, group_units, launch):
    require(all(r.get('status')=='COMPLETE_VERIFIED' for r in group_units.values()), 'Cannot finalize partial group')
    proofs = {i:r['completion'] for i,r in group_units.items()}
    final = max((p['exit'] for p in proofs.values()), key=lambda e:e['utc'])
    # job_id/utc are the actual last worker's exit trap, never the CPU finalizer's clock.
    ex = {**final, 'aggregate':True,'unit_exits':{i:p['exit'] for i,p in proofs.items()},
          'amendment':str(BUNDLE/'SCHEDULING_AMENDMENT.json')}
    gd = ROOT/'results'/group_id
    launch[group_id] = {**launch[group_id], 'job':final['job_id'], 'split':True,
                        'unit_jobs':{i:r['job'] for i,r in group_units.items()},
                        'unit_ledger':str(BUNDLE/'UNIT_LEDGER.json')}
    write(ROOT/'LAUNCH.json',launch)
    write(gd/'SCHED_EXIT.json',ex)
    write(gd/'DONE.json',{'group':group_id,'freeze_sha256':sha(ROOT/'FROZEN.json'),
          'units':[str(unit_path(group_id,i)/'full_result.json') for i in (0,1)],
          'split_completion_proofs':proofs,'amendment':str(BUNDLE/'SCHEDULING_AMENDMENT.json')})
