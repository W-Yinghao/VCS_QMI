"""Freeze the gradient direction/norm mechanism protocol before execution."""
import ast
from common import BASE,HERE,read,write,sha

def main():
    assert not (HERE/'FREEZE.json').exists()
    parent=BASE/'handoff/r22_reg_trajectory_v2_20261025';files=dict(read(parent/'FREEZE.json')['files'])
    for rel,h in files.items():assert sha(BASE/rel)==h,rel
    def pin(p):files[str(p.relative_to(BASE))]=sha(p)
    pin(parent/'FREEZE.json')
    for name in ['SOURCE_LOCK.json','INPUT_LOCK.json']:(HERE/name).write_bytes((parent/name).read_bytes())
    baseline={}
    for ds in ('bf','epfl'):
        for arm in ('vcs_exact','bce','nce'):
            for seed in (1001,1002,1003):
                key=f'{ds}_{arm}_{seed}';p=parent/'runs'/key;r=read(p/'COMPLETE.json')
                assert r['status']=='COMPLETE' and r['freeze_sha256']==sha(parent/'FREEZE.json')
                assert sha(p/'TRAJECTORIES.jsonl')==r['trajectory_sha256']
                baseline[key]=dict(completion=str(p/'COMPLETE.json'),completion_sha256=sha(p/'COMPLETE.json'),
                    trajectories=str(p/'TRAJECTORIES.jsonl'),trajectories_sha256=sha(p/'TRAJECTORIES.jsonl'),host=r['environment']['host'])
                pin(p/'COMPLETE.json');pin(p/'TRAJECTORIES.jsonl')
    write(HERE/'BASELINE_LOCK.json',baseline)
    evidence=BASE/'handoff/r22_watch10m_20261024/ANALYSIS_REG_TRAJECTORY_ALL18_20261025.json';pin(evidence)
    protocol=dict(schema='r22.reg_gradient_factor.paired.v2',evidence=str(evidence),evidence_sha256=sha(evidence),
        hypothesis='EPFL POS_QUAD precision loss comes from the evolving gradient direction, scalar norm history, or their interaction; a2x2 crossed intervention separates these mechanisms without retraining.',
        theory='Original VCS theory unchanged. These synthetic gradient paths are mechanism controls, not completeVCS or a new production objective.',
        units='Same18 fixed30k encoders (BF/EPFL x exact/BCE/NCE x3seeds1001-3), all original CAL cores/windows x2perturbations x3frozen COSstarts. No coarse search, no new training.',
        factors=dict(direction=['P1','POS_QUAD'],norm=['P1','POS_QUAD'],
            formula='At each arm current pose, compute both gradients. Applied ascent gradient = grad(direction) * norm(grad(norm_source))/norm(grad(direction)), detached before assigning negative gradient to Adam.',
            zero='eps1e-12; both norms <=eps ->zero; undefined direction with nonzero target norm fails closed; no clipping or tuning.'),
        reuse='P1_NATIVE and POS_NATIVE reuse all324 original per-problem trajectories; only crossed arms new. Two dataset gates reexecute6 natural trajectories each against original at all7 recorded steps. Full runs verify initial scalar/gradient/position/truth fingerprints on every start.',
        replay_tolerance='Same original node/GPU type/software/precision/driver. Gate natural replay: pose maxabs<=1e-5, nativeerror abs<=1e-4 at every recorded step. Initial terms atol1e-6/rtol1e-5; gradients atol1e-5/rtol1e-5. Fail closed, do not relax after results. If reuse fails, freeze separate samejob4arm design before new runs.',
        optimizer='Original100steps Adam lr.5 betas(.9,.999) eps1e-8, fresh state per path, original pose projection; state50 copy roundtrip.',
        primary='Scene-equal mean3start native endpoint error, pertrainingobjective/seed/dataset. Direction contrast = .5[(POS_NATIVE-P1_DIR_POS_NORM)+(POS_DIR_P1_NORM-P1_NATIVE)]. Norm(POS relativeP1) contrast=.5[(POS_NATIVE-POS_DIR_P1_NORM)+(P1_DIR_POS_NORM-P1_NATIVE)]. Interaction=(POS_NATIVE-P1_DIR_POS_NORM)-(POS_DIR_P1_NORM-P1_NATIVE). Curves0/1/5/10/25/50/100.',
        secondary='Rank all four arms by common terminalP1 energy (available in old traces), strict/loose/native error and per-scene changes. Never rank synthetic paths by a nonexistent own scalar energy; do not use truth to select starts or candidates.',
        interpretation='>=2/3seed same contrast sign plus leave-one-scene-out aggregate sign stable is descriptive directional evidence; completeall18 irrespective of effect. Comparisons are interventions at different evolving current poses, not an additive causal decomposition of encoder training/generalization.',
        gates='Static undefined globals; inherited CPUchecks all18 sources plus crossgradient formula/zero/native100step bitwise checks; two GPUdatasetgates each four arms x3starts x100steps with naturalreplay. Both gates PASS beforefull18.',
        budget=dict(estimated_gpu_hours=[.25,.75],full_jobs=18,full_job_minutes=15,full_hard_gpu_hours=4.5,gate_jobs=2,gate_minutes=10,gate_hard_gpu_hours=1/3,cpu_minutes=20,max_output_gib=2),
        limits='CAL repeated development; starts/perturbations correlated; natural baseline replay gates cover2problems not324 and full initial fingerprints do not prove every future floating-point operation identical. Report this reuse limitation and driver/node pin. No pooled dataset mean.',
        controls='Originalunique pool/submissionlock/storagegates; noTEST/push/deletion/subagent/newwatcher; no implicitretry or successful duplicate jobs. Baseline gate replays are technical reproducibility checks.')
    previous=BASE/'handoff/r22_reg_gradfactor_20261025'
    for name in ['FREEZE.json','PROTOCOL.json','FAILED_GATE_REVIEW_20261025.json']:
        pin(previous/name)
    protocol['supersedes']=dict(package=str(previous),freeze_sha256=sha(previous/'FREEZE.json'),failure=str(previous/'FAILED_GATE_REVIEW_20261025.json'),old_full_never_admitted=True)
    protocol['reuse']='Reuse source models and all324 fixed CAL problem inputs/start poses only. Reexecute allfour arms in each same job; native P1/POS trajectories are new paired references for both crossed arms. Historical trajectory differences are recorded descriptively, never used as primary comparator.'
    protocol['replay_tolerance']='The old cross-run trajectory reuse gate FAILED at its original fixed tolerances; its thresholds and failures remain untouched. This separately frozen fallback does not reuse those endpoints. Retain samehost/GPU/software/precision/driver and initial value atol1e-6 rtol1e-5, gradient atol1e-5 rtol1e-5, exact pose/truth/position checks, including allfour samejob initial fingerprints. No old trajectory agreement criterion is claimed by this different design.'
    protocol['gates']='Static undefined globals; inherited CPU all18source/term/gradient checks and both native100step exact CPU equivalence; two GPUdataset gates each 4arms x3same starts x100steps, allfinite/termgradient identity, same initial scalar/gradient fingerprints, Adam state50 roundtrip, CAL only, source unchanged. Both PASS before full18.'
    protocol['budget'].update(estimated_gpu_hours=[.5,1.5],full_job_minutes=20,full_hard_gpu_hours=6.)
    protocol['limits']='CAL repeated development; starts/perturbations correlated; samejob paired references do not eliminate nondeterministic CUDA numerical effects. Originalnode pin retained. Historical failure demonstrates cross-run drift, not a cause isolated to CUDA or manualgradient. Full data for allfour arms, no original endpoint reuse; no pooled dataset mean.'
    protocol['controls']='Originalunique pool/submissionlock/storagegates; noTEST/push/deletion/subagent/newwatcher. New native references are scientifically necessary paired controls after failed reuse, not occupancy duplicates. All18 complete regardless effects; no implicit retry.'
    write(HERE/'PROTOCOL.json',protocol)
    (HERE/'gpu.sbatch').write_text(f'''#!/usr/bin/env bash
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --nice=60
#SBATCH --no-requeue
#SBATCH --output={BASE}/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH
{BASE}/env/bin/python -B -s {HERE}/gradfactor.py "$@"
if [[ "$1" == "gate" ]]; then /usr/bin/python3 {HERE}/admit.py; fi
''')
    (HERE/'check.sbatch').write_text(f'''#!/usr/bin/env bash
#SBATCH --job-name=r22_gradpair_cpu
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output={BASE}/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/usr/bin/python3 -B {HERE}/static_check.py
{BASE}/env/bin/python -B -s {HERE}/gradfactor.py check
/usr/bin/python3 {HERE}/admit.py
''')
    jobs={'gates':[],'full':[]}
    for ds in ('bf','epfl'):
        for gate,arm,seed in [(True,'vcs_exact',1001)]+[(False,a,s) for a in ('vcs_exact','bce','nce') for s in (1001,1002,1003)]:
            key=f'{ds}_{arm}_{seed}';name=f'r22_gradpair_gate_{ds}' if gate else f'r22_gradpair_{ds}_{arm}_s{seed}'
            host=baseline[key]['host'];assert host in ('node39','node52','node57','node50')
            args=['sbatch','--job-name='+name,'--nodelist='+host]
            if gate:args+=['--time=00:10:00']
            args+=[str(HERE/'gpu.sbatch'),'gate' if gate else 'run','--ds',ds,'--arm',arm,'--seed',str(seed)]
            jobs['gates' if gate else 'full'].append(dict(name=name,args=args,
                completion=str(HERE/'runs'/(f'gate_{ds}' if gate else key)/'COMPLETE.json'),
                group='other_reg_gradpair_gate' if gate else f'other_reg_gradpair_{ds}',freeze=str(HERE/'FREEZE.json')))
    write(HERE/'JOBS_SPEC.json',jobs)
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    for p in sorted(HERE.iterdir()):
        if p.is_file():pin(p)
    write(HERE/'FREEZE.json',dict(parent_freeze_sha256=sha(parent/'FREEZE.json'),files=files))
    print('FROZEN',sha(HERE/'FREEZE.json'),len(files))
if __name__=='__main__':main()
