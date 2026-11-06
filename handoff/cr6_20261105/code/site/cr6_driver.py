"""CR6 CPU driver (no --dependency). usage: cr6_driver.py PART:JOB [PART:JOB ...]   e.g. L40S:1090959 RTX6000PRO:1090989 V100-32GB:1090990
Per-GPU-type acceptance release (addendum registered before any formal training):
1. as acceptance results appear, record them in ACCEPTANCE_REGISTRY.json (sha-bound; the reported GPU must match the partition);
   at the first ALL_PASS -> freeze_cr6.py (with every acceptance result available at that time); if all acceptances fail -> stop
2. a group is submitted only to accepted GPU types that validate its kind (A1/A2: L40S,RTX6000PRO,H100,V100-32GB; B: L40S,RTX6000PRO,H100);
   a resubmitted group is pinned to the GPU type recorded in its GROUP_ENV (one GPU type per matched group)
3. order: A1, B_Flickr60, B_CC -> (A1 done) GRAPHS (CPU) -> (graphs) A2 t1002, t1003; a job leaving the queue without its completion marker is
   resubmitted (exact resume) at most twice, then FAILED (kept on record)
4. all nodes finished -> summarize_cr6.py -> DRIVER_DONE.  Submit-limit errors are retried every 2 min.  STOP: touch STOP_DRIVER.
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import cr6_common as K                                                       # noqa: E402

SB = K.TREE/'sbatch'; STATE = K.TREE/'DRIVER_STATE.json'; TSV = K.TREE/'SUBMITTED_CR6.tsv'; REG = K.TREE/'ACCEPTANCE_REGISTRY.json'
ALLOWED = {'A1': ['RTX6000PRO', 'L40S'], 'A2': ['RTX6000PRO', 'L40S', 'V100-32GB'], 'B': ['RTX6000PRO', 'L40S'], 'CPU': []}   # addendum revision_2: gated on the single RTX acceptance
NODES = {
    'A1': dict(kind='A1', cmd=['group.sbatch', 'A1'], done=K.OUT/'A1'/'UNIT_COMPLETE.json', env=K.OUT/'A1'/'GROUP_ENV.json', deps=[], sb=['--mem=48G', '--time=06:00:00'], name='cr6_A1_src2002'),
    'B_Flickr60': dict(kind='B', cmd=['group.sbatch', 'B', '--condition', 'Flickr60'], done=K.OUT/'groups'/'B_Flickr60_t1002'/'UNIT_COMPLETE.json',
                       env=K.OUT/'groups'/'B_Flickr60_t1002'/'GROUP_ENV.json', deps=[], sb=['--mem=110G', '--time=1-00:00:00'], name='cr6_B_Flickr60_t1002'),
    'B_CC': dict(kind='B', cmd=['group.sbatch', 'B', '--condition', 'CC152K_RECOVERED_20261104'], done=K.OUT/'groups'/'B_CC152K_RECOVERED_20261104_t1002'/'UNIT_COMPLETE.json',
                 env=K.OUT/'groups'/'B_CC152K_RECOVERED_20261104_t1002'/'GROUP_ENV.json', deps=[], sb=['--mem=150G', '--time=1-00:00:00'], name='cr6_B_CC152K_t1002'),
    'GRAPHS': dict(kind='CPU', cmd=['cpu_py.sbatch', 'cr6_graphs.py'], done=K.OUT/'A1'/'GRAPH_REGISTRY_SRC2002.json', env=None, deps=['A1'], sb=[], name='cr6_graphs_src2002'),
    'A2_t1002': dict(kind='A2', cmd=['group.sbatch', 'A2', '--seed', '1002'], done=K.OUT/'groups'/'A2_Flickr60_t1002'/'UNIT_COMPLETE.json',
                     env=K.OUT/'groups'/'A2_Flickr60_t1002'/'GROUP_ENV.json', deps=['GRAPHS'], sb=['--mem=48G', '--time=1-00:00:00'], name='cr6_A2_Flickr60_t1002'),
    'A2_t1003': dict(kind='A2', cmd=['group.sbatch', 'A2', '--seed', '1003'], done=K.OUT/'groups'/'A2_Flickr60_t1003'/'UNIT_COMPLETE.json',
                     env=K.OUT/'groups'/'A2_Flickr60_t1003'/'GROUP_ENV.json', deps=['GRAPHS'], sb=['--mem=48G', '--time=1-00:00:00'], name='cr6_A2_Flickr60_t1003'),
}
MAX_ATTEMPTS = 3


def part_of(gpu):
    g = gpu or ''
    return 'L40S' if 'L40S' in g else 'RTX6000PRO' if 'RTX PRO 6000' in g else 'H100' if 'H100' in g else 'V100-32GB' if 'V100' in g else None


def log(*a): print(K.utc(), *a, flush=True)


def queue():
    r = subprocess.run(['squeue', '-u', 'yinwang', '-h', '-o', '%i'], capture_output=True, text=True, timeout=120)
    if r.returncode: raise RuntimeError(r.stderr)
    return set(r.stdout.split())


def submit(k, n, parts):
    cmd = ['sbatch', '--parsable', f"--job-name={n['name']}", *([f'--partition={",".join(parts)}'] if parts else []), *n['sb'], str(SB/n['cmd'][0]), *n['cmd'][1:]]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=SB)
    if r.returncode: log('submit failed', k, r.stderr.strip()); return None
    j = r.stdout.strip().split(';')[0]
    with open(TSV, 'a') as f: f.write(f"{j}\t{n['name']}\t{k}\t{','.join(parts) or 'CPU'} {' '.join(n['sb'])}\t{K.utc()}\tdriver\n")
    log('submitted', k, j, parts); return j


def update_acceptances(accs, q):
    reg = K.read_json(REG) if REG.exists() else {}
    for part, job in accs:
        if job in reg: continue
        f = K.TREE/'acceptance'/f'ACCEPTANCE_CR6_{job}.json'
        if f.exists():
            A = K.read_json(f); ok = A['status'] == 'ALL_PASS' and part_of(A['gpu_name']) == part
            reg[job] = dict(partition=part, file=str(f), sha256=K.sha256(f), status=A['status'], gpu=A['gpu_name'], mode=A.get('mode'),
                            validates=A.get('validates_groups', []) if ok else [], recorded_utc=K.utc(), note=None if ok else 'not usable (failed or GPU mismatch)')
            log('ACCEPTANCE', job, part, A['status'], reg[job]['validates'])
        elif job not in q:
            time.sleep(20)
            if not f.exists(): reg[job] = dict(partition=part, status='LEFT_QUEUE_WITHOUT_RESULT', validates=[], recorded_utc=K.utc()); log('ACCEPTANCE_MISSING', job, part)
    REG.write_text(json.dumps(reg, indent=1))
    return reg


def main():
    accs = [tuple(x.split(':')) for x in sys.argv[1:]]
    st = K.read_json(STATE) if STATE.exists() else {}
    while not (K.TREE/'STOP_DRIVER').exists():
        try: q = queue()
        except Exception as e: log('squeue failed', e); time.sleep(60); continue
        reg = {j: v for j, v in update_acceptances(accs, q).items() if j in {a[1] for a in accs}}   # only the acceptances named on the command line (superseded ones stay on record only)
        passed = [v for v in reg.values() if v['validates']]
        if not K.FROZEN.exists():
            if passed:
                files = [v['file'] for v in reg.values() if 'file' in v]
                r = subprocess.run([K.PY, '-I', '-B', str(HERE/'freeze_cr6.py'), *files], capture_output=True, text=True); log(r.stdout.strip(), r.stderr[-2000:])
                if r.returncode: print('DRIVER_STOPPED_FREEZE_FAILED'); return
            elif len(reg) == len(accs): log('ALL_ACCEPTANCES_FAILED', reg); print('DRIVER_STOPPED_ACCEPTANCE_FAIL'); return
            else: time.sleep(60); continue
        if not st.get('_frozen_checked'): K.frozen_check(); st['_frozen_checked'] = dict(status='DONE', jobs=[], sha=K.sha256(K.FROZEN)); log('FROZEN_OK', K.sha256(K.FROZEN))
        for k, n in NODES.items():
            s = st.setdefault(k, dict(status='WAITING', jobs=[]))
            if s['status'] in ('DONE', 'FAILED'): continue
            if n['done'].exists(): s['status'] = 'DONE'; log('DONE', k); continue
            if s['status'] == 'SUBMITTED' and s['jobs'][-1] not in q:
                time.sleep(5)
                if n['done'].exists(): s['status'] = 'DONE'; log('DONE', k); continue
                s['status'] = 'FAILED' if len(s['jobs']) >= MAX_ATTEMPTS else 'RETRY'; log('LEFT_QUEUE_INCOMPLETE', k, s['jobs'][-1], s['status'])
            if s['status'] in ('WAITING', 'RETRY') and all(st.get(d, {}).get('status') == 'DONE' for d in n['deps']):
                if any(st.get(d, {}).get('status') == 'FAILED' for d in n['deps']): s['status'] = 'FAILED'; s['note'] = 'prerequisite failed'; continue
                if n['kind'] == 'CPU': parts = []
                else:
                    parts = list(ALLOWED[n['kind']]) if any(n['kind'] in v['validates'] for v in reg.values()) else []
                    if n['env'] is not None and Path(n['env']).exists():
                        pinned = part_of(K.read_json(n['env'])['gpu_name']); parts = [p for p in parts if p == pinned]
                    if not parts: continue
                j = submit(k, n, parts)
                if j: s['jobs'].append(j); s['status'] = 'SUBMITTED'; s['partitions'] = parts
            elif s['status'] == 'WAITING' and any(st.get(d, {}).get('status') == 'FAILED' for d in n['deps']): s['status'] = 'FAILED'; s['note'] = 'prerequisite failed'
        STATE.write_text(json.dumps(st, indent=1))
        if all(st.get(k, {}).get('status') in ('DONE', 'FAILED') for k in NODES): break
        if len(reg) == len(accs):   # every acceptance finished: nodes whose kind has no accepted GPU type can never start
            for k, n in NODES.items():
                if st[k]['status'] in ('WAITING', 'RETRY') and n['kind'] != 'CPU' and not any(n['kind'] in v['validates'] for v in reg.values()):
                    st[k]['status'] = 'FAILED'; st[k]['note'] = 'no accepted GPU type for this group kind'; log('NO_ACCEPTED_GPU', k)
        time.sleep(120)
    STATE.write_text(json.dumps(st, indent=1))
    log('GROUPS_FINISHED', {k: st[k]['status'] for k in NODES if k in st})
    r = subprocess.run([K.PY, '-I', '-B', str(HERE/'summarize_cr6.py')], capture_output=True, text=True); log(r.stdout[-3000:], r.stderr[-3000:])
    print(K.utc(), 'DRIVER_DONE', flush=True)


if __name__ == '__main__': main()
