import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
OLD=BASE/'handoff/r22_mechanism_review_20261024'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

assert not (HERE/'FREEZE.json').exists()
(HERE/'mechanism_common.py').write_text((OLD/'mechanism_common.py').read_text())
(HERE/'SOURCE_LOCK.json').write_bytes((OLD/'SOURCE_LOCK.json').read_bytes())
src=(OLD/'kd_terms.py').read_text()
new=src.replace('def run(seed,anchor):','def run(seed,anchor,device_name="cuda",max_batches=8,technical=""):',1)
new=new.replace("out=output(f'kd_terms_{anchor}_{seed}')", "out=output(f'{technical}kd_terms_{anchor}_{seed}')",1)
new=new.replace("device=torch.device('cuda')",'device=torch.device(device_name)',1)
new=new.replace("model=K.create_model(cfg['student'],manifest['num_classes'],seed).to(device);model.load_state_dict(ckpt['student'])", "model=K.create_model(cfg['student'],manifest['num_classes'],seed)\n        student_dim=K.feature_dim(model)  # CPU probe before moving model to the target device\n        model=model.to(device);model.load_state_dict(ckpt['student'])",1)
new=new.replace('DistillationLoss(K.feature_dim(model),tdim,lc)','DistillationLoss(student_dim,tdim,lc)',1)
new=new.replace('if batch==8:break','if batch==max_batches:break',1)
new=new.replace('done(out,started,seed=seed,anchor=anchor,results=results,','done(out,started,seed=seed,anchor=anchor,technical=technical,max_batches=max_batches,execution_device=device_name,results=results,',1)
new=new[:new.index("if __name__=='__main__':")]+'''if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','cpu_path','gate','run']);p.add_argument('--seed',type=int);p.add_argument('--anchor',choices=['ce','dkd']);a=p.parse_args()
    if a.mode=='check':check()
    elif a.mode=='cpu_path':
        for anchor in ('ce','dkd'):
            for seed in (31415,16180,57721):run(seed,anchor,'cpu',1,'cpu_path_')
    else:
        start=time.time()
        try:run(a.seed,a.anchor,'cuda',1 if a.mode=='gate' else 8,'gate_' if a.mode=='gate' else '')
        except Exception as exc:
            import os,traceback
            write(HERE/('FAILURE_'+os.environ['SLURM_JOB_ID']+'.json'),dict(status='FAILED',seconds=time.time()-start,error=repr(exc),traceback=traceback.format_exc(),mode=a.mode,anchor=a.anchor,seed=a.seed))
            raise
'''
ast.parse(new)
(HERE/'kd_terms.py').write_text(new)
(HERE/'CHANGE.diff').write_text(''.join(difflib.unified_diff(src.splitlines(True),new.splitlines(True),fromfile='original/kd_terms.py',tofile='repair/kd_terms.py')))
write(HERE/'PROTOCOL.json',dict(schema='r22.kd_mechanism.technical_repair.v2',parent_freeze=sha(OLD/'FREEZE.json'),
 scientific_protocol=read(OLD/'PROTOCOL.json')['kd'],cause='CPU-only feature_dim was called after moving model to CUDA; all6 fail before diagnostic batches',
 repair='Compute student feature dimension on CPU before .to(device), then restore exactly the same checkpoint. Loss/gradient/update/data/seed/source/budget unchanged.',
 acceptance='Original term gradcheck plus all6 real CPU execution paths, both source checkpoints, one full FIT minibatch and all four disposable SGD updates each. Then one real GPU minibatch for each CE/DKD anchor; both gates required before six full8batch diagnostics.',
 cost='Original six failed attempts retained; scheduler records no longer available, duration unknown, never recorded as zero. CPU checks and two GPU gates separately counted. Six full jobs hard30min each, gates15min each.',
 no='No TEST, deletion, public push, subagents, new controller or modification of original frozen files. No automatic retry.',
 admission='Existing unique pool only; after CPU checks retire only failed six original KD entries from active manifest, preserving original file, logs, attempts and retirement record. Registration entries retained.'))
spec={'gates':[],'full':[]}
for mode in ('gate','run'):
 for anchor in ('ce','dkd'):
  for seed in ((31415,) if mode=='gate' else (31415,16180,57721)):
   name=f'r22_mech_kd2_{mode}_{anchor}_{seed}'
   spec['gates' if mode=='gate' else 'full'].append(dict(name=name,args=['sbatch','--partition=L40S','--time='+('00:15:00' if mode=='gate' else '00:30:00'),'--job-name='+name,str(HERE/'kd.sbatch'),mode,'--anchor',anchor,'--seed',str(seed)],completion=str(HERE/'runs'/f'{"gate_" if mode=="gate" else ""}kd_terms_{anchor}_{seed}'/'COMPLETE.json'),group='other_mechanism_kd_gate' if mode=='gate' else 'other_mechanism_kd',freeze=str(HERE/'FREEZE.json')))
write(HERE/'JOBS_SPEC.json',spec)
(HERE/'kd.sbatch').write_text((OLD/'kd.sbatch').read_text().replace(str(OLD),str(HERE))+f'\n/usr/bin/python3 {HERE}/admit.py\n')
(HERE/'check.sbatch').write_text((OLD/'check.sbatch').read_text().split('set -euo')[0].replace('r22_mechanism_cpu','r22_kd2_cpu')+f'''set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
{BASE}/env_kd/bin/python -B -s {HERE}/kd_terms.py check
{BASE}/env_kd/bin/python -B -s {HERE}/kd_terms.py cpu_path
/usr/bin/python3 {HERE}/admit.py
''')
files=read(OLD/'FREEZE.json')['files'].copy()
for name in ['mechanism_common.py','kd_terms.py','SOURCE_LOCK.json','PROTOCOL.json','CHANGE.diff','JOBS_SPEC.json','kd.sbatch','check.sbatch','admit.py']:
 p=HERE/name;files[str(p.relative_to(BASE))]=sha(p)
write(HERE/'FREEZE.json',dict(parent_freeze=sha(OLD/'FREEZE.json'),files=files))
print(sha(HERE/'FREEZE.json'))
