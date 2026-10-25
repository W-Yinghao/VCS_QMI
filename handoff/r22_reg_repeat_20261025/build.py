"""Freeze a bounded fixed-code repeat/order diagnostic before any new evaluation."""
import ast
from common import BASE,HERE,read,write,sha


def main():
    assert not (HERE/'FREEZE.json').exists()
    parent=BASE/'handoff/r22_reg_gradfactor_paired_20261025'
    files=dict(read(parent/'FREEZE.json')['files'])
    for rel,h in files.items():assert sha(BASE/rel)==h,rel
    def pin(p):files[str(p.relative_to(BASE))]=sha(p)
    pin(parent/'FREEZE.json')
    for name in ('SOURCE_LOCK.json','INPUT_LOCK.json','BASELINE_LOCK.json'):
        (HERE/name).write_bytes((parent/name).read_bytes())
    baseline=read(HERE/'BASELINE_LOCK.json');paired={}
    for key in baseline:
        p=parent/'runs'/key;r=read(p/'COMPLETE.json')
        assert r['status']=='COMPLETE' and r['freeze_sha256']==sha(parent/'FREEZE.json')
        assert sha(p/'TRAJECTORIES.jsonl')==r['trajectory_sha256']
        paired[key]=dict(completion=str(p/'COMPLETE.json'),completion_sha256=sha(p/'COMPLETE.json'),
            trajectories=str(p/'TRAJECTORIES.jsonl'),trajectories_sha256=sha(p/'TRAJECTORIES.jsonl'))
        pin(p/'COMPLETE.json');pin(p/'TRAJECTORIES.jsonl')
    write(HERE/'PAIRED_EVIDENCE_LOCK.json',paired)
    evidence=[]
    for ds in ('BF','EPFL'):
        p=BASE/f'handoff/r22_watch10m_20261024/ANALYSIS_GRADPAIR_{ds}_20261025.json'
        assert read(p)['status']=='COMPLETE_REVIEW'
        pin(p);evidence.append(dict(path=str(p),sha256=sha(p)))
    protocol=dict(schema='r22.reg_repeat.v1',evidence=evidence,
        hypothesis='Small fixed-CAL direction/norm contrasts may be comparable to same-code repeated-evaluation variability. Quantify reproducibility and balance execution position before attributing a mechanism or changing the production objective.',
        motivation='BF historical native endpoint mean absolute drift .12-.23px and EPFL .015-.027px. Historical comparisons change execution path/run and do not isolate a numerical cause. This study reuses exactly the paired four-arm path function, with no retraining or search expansion.',
        theory='Original VCS theory unchanged. Crossed gradients and positive-only paths are mechanism controls, not complete VCS-QMI.',
        units='18 frozen30k encoders: BF/EPFL x exact/BCE/NCE x seeds1001,1002,1003. All original CAL:8BF cores or10EPFLwindows,2perturbations,3frozen COSstartposes,4096positions. No DEV/TEST reads, new coarse search or encoder training.',
        implementation=dict(path=str(parent/'gradfactor.py'),path_sha256=sha(parent/'gradfactor.py'),function='path',
            methods=['P1_NATIVE','POS_NATIVE','P1_DIR_POS_NORM','POS_DIR_P1_NORM'],
            repeats=4,order='For each problem i and repeat r, rotate the listed method order left by (i+r)%4. Every method occupies each rank exactly once per problem; starts always0,1,2.',
            optimizer='Fresh pose and Adam state for every path; original100steps lr.5, projection and frozen precision unchanged; state50 copy check inherited.',
            rng='No reseeding between repeats; assert torch CPU and current CUDA RNG states unchanged over each repeat. No stochastic encoder/dropout updates.',
            measurements='Keep all7 path checkpoints, actual applied ascent gradient and first-pass norms/scale, separately recorded second autograd gradients, chosen endpoint by common terminalP1, all four repetitions including failures.'),
        primary='For each dataset/trainingobjective/seed/repeat compute scene-equal mean3start endpoint native error. Compute direction,norm,interaction and nativePOS-P1 contrasts as in paired protocol. Report per-seed four-repeat mean/SD/range, all4 aggregate replicate contrasts, seed variability and leave-one-scene-out signs separately.',
        secondary='Common terminalP1 candidate selection error and strict/loose success; all7point curves; per-problem/start within-method repeat SD/range and execution-position summaries. Repetitions and starts are not independent seeds. Rank effects are descriptive, not isolated CUDA root causes.',
        contrast_definitions=dict(direction='.5*((POS_NATIVE-P1_DIR_POS_NORM)+(POS_DIR_P1_NORM-P1_NATIVE))',
            norm='.5*((POS_NATIVE-POS_DIR_P1_NORM)+(P1_DIR_POS_NORM-P1_NATIVE))',
            interaction='POS_NATIVE-P1_DIR_POS_NORM-POS_DIR_P1_NORM+P1_NATIVE',native='POS_NATIVE-P1_NATIVE'),
        success='Technical completeness is independent of outcomes. A precision-qualified directional signal requires all3 seed mean contrasts with the same nonzero sign, all4 seed-aggregate replicate contrasts with that sign, stable aggregate leave-one-scene-out sign, and min(abs(seed means))>=2*max(within-seed repeat SD). This is a descriptive preregistered screen, not a significance test; report every contrast even if it fails.',
        stopping='Complete all18 once both gates pass, regardless effect. Any freeze/source/identity/finite/role/budget violation stops that unit; no automatic retry or tolerance relaxation. If signal unresolved, do not reinterpret numerical differences as mechanism advantage; any subsequent implementation study needs a separate concrete hypothesis/freeze, not an indefinite repetition loop.',
        gates='Static undefined globals; inherited all18source/term/gradient/nativeCPUequivalence checks; exact4CPUrepeats under balanced order. One GPU gate per dataset seed1001 exact firstCALproblem:4repeats x4arms x3starts x100updates=4800updates, allfinite/initialfingerprints/RNGunchanged/CALonly/sourceunchanged. Both pass before18formaljobs.',
        budget=dict(estimated_gpu_hours=[2,4],full_jobs=18,full_job_minutes=30,full_hard_gpu_hours=9,
            paths=15552,adam_updates=1555200,gate_jobs=2,gate_minutes=10,gate_hard_gpu_hours=1/3,cpu_minutes=20,max_output_gib=2),
        limits='Small repeatedly used CAL;3seeds, correlated scenes/starts/perturbations. Samejob repetition does not measure every between-job source of variation or confirm generalization. Preserve hardware/driver/node pin from original source; do not pool datasets or choose checkpoints/starts by truth. No fullJ arm in this diagnostic.',
        controls='Original unique pool and shared submissionlock/storage gates. Successful prior paths are repeated only to answer a new registered reproducibility question, never to fill GPUs. No new controller/subagent/TEST/push/deletion.')
    write(HERE/'PROTOCOL.json',protocol)
    (HERE/'gpu.sbatch').write_text(f'''#!/usr/bin/env bash
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --nice=60
#SBATCH --no-requeue
#SBATCH --output={BASE}/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH
{BASE}/env/bin/python -B -s {HERE}/repeat.py "$@"
if [[ "$1" == "gate" ]]; then /usr/bin/python3 {HERE}/admit.py; fi
''')
    (HERE/'check.sbatch').write_text(f'''#!/usr/bin/env bash
#SBATCH --job-name=r22_repeval_cpu
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output={BASE}/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/usr/bin/python3 -B {HERE}/static_check.py
{BASE}/env/bin/python -B -s {HERE}/repeat.py check
/usr/bin/python3 {HERE}/admit.py
''')
    jobs={'gates':[],'full':[]}
    for ds in ('bf','epfl'):
        for gate,arm,seed in [(True,'vcs_exact',1001)]+[(False,a,s) for a in ('vcs_exact','bce','nce') for s in (1001,1002,1003)]:
            key=f'{ds}_{arm}_{seed}';name=f'r22_repeval_gate_{ds}' if gate else f'r22_repeval_{ds}_{arm}_s{seed}'
            host=baseline[key]['host'];assert host in ('node39','node52','node57','node50')
            args=['sbatch','--job-name='+name,'--nodelist='+host]
            if gate:args+=['--time=00:10:00']
            args+=[str(HERE/'gpu.sbatch'),'gate' if gate else 'run','--ds',ds,'--arm',arm,'--seed',str(seed)]
            jobs['gates' if gate else 'full'].append(dict(name=name,args=args,
                completion=str(HERE/'runs'/(f'gate_{ds}' if gate else key)/'COMPLETE.json'),
                group='other_reg_repeat_gate' if gate else f'other_reg_repeat_{ds}',freeze=str(HERE/'FREEZE.json')))
    write(HERE/'JOBS_SPEC.json',jobs)
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    for p in sorted(HERE.iterdir()):
        if p.is_file():pin(p)
    write(HERE/'FREEZE.json',dict(parent_freeze_sha256=sha(parent/'FREEZE.json'),files=files))
    print('FROZEN',sha(HERE/'FREEZE.json'),len(files))


if __name__=='__main__':main()
