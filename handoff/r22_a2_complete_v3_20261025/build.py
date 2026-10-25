"""One-time stdlib-only freeze. Never run after activation."""
import datetime
import subprocess
from common import BASE,HERE,REPO,SRC,OLD,OUT,ARMS,SEEDS,read,write,sha

assert not (HERE/'FREEZE.json').exists() and not OUT.exists()
design=BASE/'handoff/r22_next_round_20261025/A2_COMPLETION_DESIGN_20261025.json'
protocol=read(design)
protocol.update(status='FROZEN_EXECUTION_PENDING_TECHNICAL_GATES',execution_package=str(HERE),output_root=str(OUT),
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    implementation='Original R22 A2 r22_student.run, make_student, WeightedStudent.fit and losses unchanged; wrapper preflights immutable inputs/init/manifest/environment. Separate new output root. No old run overwrite, no silent resume.',
    execution_clarifications={'storage':'All new GPU outputs on projects, six GiB reserved above 300GB floor; home code/CPU checks only. Home still below1450GiB. Chosen before any numerical execution.',
        'GPU_gate':'One RTX job, seeds902/903 x three arms x40 sparse epochs. Each cell original first8 batches at epoch0 then firstbatch of epochs1-39, 47updates/cell282total. Original SGR and Adam, two LR groups, real16DEVimages/80captions. Sparse scores are only a finite/serialization check, never task results.',
        'full':'Six RTX jobs, same GPUmodel/driver/software/precision as A2 seed901; full iterator never patched. Each seed init and all40 epoch input orders must match original A metadata before any update. All40epochs45320updates, no early-stop, no automatic retry.',
        'historical_controls':'Reuse seed901 A2 and original A three-seed VCS/VCSW/BCE/NCE; historical mixed GPU explicitly descriptive. N2M ConstantU/weighted hinge and N2F unweighted/weighted hinge metadata frozen separately. Checkpoint declared hashes retained; original A2 selected checkpoints rehashed here. Other historical best checkpoints must be rehashed at final analysis before any checkpoint-dependent claim.'})
write(HERE/'PROTOCOL.json',protocol)
files={}
def add(p):
    p=p.resolve();files[str(p)]=sha(p)
units={}
for seed in (901,902,903):
    for arm in ('VCS','VCSW','BCE','NCE') + (ARMS if seed==901 else ()):
        d=OLD/'runs'/f's{seed}__{arm}';r=read(d/'unit_result.json')
        assert r['status']=='COMPLETE_VERIFIED' and r['log_checks']['updates']==45320
        names=('unit_result.json','run_manifest.json','train_log.json','done.json','attempts.jsonl')
        for n in names:add(d/n)
        for n in ('run_manifest.json','train_log.json','done.json'):assert sha(d/n)==r['artifacts_sha256'][n]
        if arm in ARMS:
            assert sha(d/'best.pt')==r['artifacts_sha256']['best.pt']
            assert sha(SRC/'r22_student.py')==r['source']['adapter_sha256']
            assert sha(SRC/'losses_sharp.py')==r['losses_sharp_sha256']
            assert sha(REPO/'src/v9/n2/train.py')==r['source']['train_py_sha256']
        units[f'{seed}_{arm}']=dict(directory=str(d),files={n:sha(d/n) for n in names},
            checkpoint_declared_sha256={n:r['artifacts_sha256'][n] for n in ('best.pt','last.pt')},environment=r['environment'])
for seed in SEEDS:
    for arm in ARMS:assert not (OLD/'runs'/f's{seed}__{arm}').exists()
n2f=BASE/'handoff/v9_full_20261017';nf=read(n2f/'FROZEN.json');groups=read(n2f/'GROUPS.json');hinges={}
for seed in (901,902,903):
    for method in ('Unweighted_A','VCS_Weight'):
        gid=next(g['id'] for g in groups if g['seed']==seed and {'method':method,'scale':'sum'} in g['units'])
        d=n2f/'results'/gid/f'{method}_sum'
        paths=[p for p in (d/'train_log.json',d/'run_manifest.json',d/'done.json') if p.is_file()]
        assert len(paths)==3
        for p in paths:add(p)
        hinges[f'N2F_{method}_{seed}']={str(p):sha(p) for p in paths}
    for method in ('ConstantU','VCS'):
        d=__import__('pathlib').Path('/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1/runs')/f's{seed}__{method}'
        paths=[d/n for n in ('train_log.json','run_manifest.json','done.json')]
        for p in paths:add(p)
        hinges[f'N2M_{method}_{seed}']={str(p):sha(p) for p in paths}
write(HERE/'CONTROL_LOCK.json',dict(units=units,hinge_metadata=hinges))
input_hash={}
for seed in (901,902,903):
    d=__import__('pathlib').Path('/projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k/rho0.4')/f's{seed}'
    ps=[d/'pairs.observed_pairs.json']
    for method in ('Unweighted_A','VCS_Weight'):
        ps.extend([d/'weights'/f'{method}.csv',d/'weights'/f'{method}.csv.lock.json'])
    for p in ps:
        add(p);input_hash[str(p)]=sha(p)
        if str(p) in nf['input_sha256']:assert sha(p)==nf['input_sha256'][str(p)]
for p,rec in nf['data_files'].items():
    assert __import__('pathlib').Path(p).name.startswith(('train_','dev_'))
    st=__import__('pathlib').Path(p).stat();assert st.st_size==rec['size'] and st.st_mtime_ns==rec['mtime_ns']
write(HERE/'INPUT_LOCK.json',dict(input_sha256=input_hash,large_data_stat=nf['data_files'],
    data_identity='Original N2F large train/DEV file size+mtime identity retained; not claimed to be a new full byte hash. Token/vocab/observed/weights hash locked.'))
# Freeze source code, host lock, role protocol and existing train/DEV token cache only.
for root in (REPO/'src',REPO/'v9_n2p/c_server',SRC):
    for p in root.rglob('*.py'):add(p)
for p in (REPO/'reports/V9/HOST_LOCK.yaml',REPO/'v9_n2p/VCS_V9_N2P_V10_Review_Server_Package_v1_20261020/config/ROUND_PROTOCOL.json',
    n2f/'FROZEN.json',n2f/'GROUPS.json',n2f/'source/src/v9/n2/train.py'):
    add(p)
for p in (n2f/'source/token_cache').glob('*.json'):
    if '_train_' in p.name or '_dev_' in p.name:add(p)
add(__import__('pathlib').Path('/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/vocab/f30k_precomp_vocab.json'))
git_refs={}
for rel in ('src/v9/n2/train.py','v9_r22/vcsloss/losses.py'):
    blob=subprocess.check_output(['git','-C',str(REPO),'show','df8a0707:'+rel]);git_refs[rel]=__import__('hashlib').sha256(blob).hexdigest()
write(HERE/'CPU_GIT_REFERENCE.json',dict(commit='df8a0707',sha256=git_refs))
env='''set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH
'''
(HERE/'cpu.sbatch').write_text('''#!/usr/bin/env bash
#SBATCH --job-name=r22_a2complete3_cpu
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
'''+env+f'{BASE}/env/bin/python -B -s {HERE}/run.py check\n/usr/bin/python3 {HERE}/admit.py\n')
(HERE/'gpu.sbatch').write_text('''#!/usr/bin/env bash
#SBATCH --partition=RTX6000PRO
#SBATCH --qos=normal
#SBATCH --exclude=node51,node53,node60,node61
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nice=60
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
'''+env+f'{BASE}/env/bin/python -B -s {HERE}/run.py "$@"\nif [[ "$1" == "gate" ]]; then /usr/bin/python3 {HERE}/admit.py; fi\n')
spec={'gates':[],'full':[]}
def job(group,name,args,completion,limit):
    spec[group].append(dict(name=name,args=['sbatch',f'--job-name={name}',f'--time={limit}',str(HERE/'gpu.sbatch'),*args],
        completion=str(completion),group='other_a2_complete_'+group,freeze=str(HERE/'FREEZE.json')))
job('gates','r22_a2complete3_gate',['gate'],OUT/'gates/COMPLETE.json','01:00:00')
for seed in SEEDS:
    for arm in ARMS:job('full',f'r22_a2complete3_{arm.lower()}_s{seed}',['run','--seed',str(seed),'--arm',arm],OUT/'runs'/f's{seed}__{arm}'/'COMPLETE.json','23:50:00')
write(HERE/'JOBS_SPEC.json',spec)
for p in [*HERE.glob('*.py'),*HERE.glob('*.json'),*HERE.glob('*.sbatch'),design]:add(p)
write(HERE/'FREEZE.json',dict(schema='r22.a2.completion.freeze.v1',files=dict(sorted(files.items()))))
print('FROZEN',sha(HERE/'FREEZE.json'),len(files))
