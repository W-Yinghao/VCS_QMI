"""Freeze once, before numerical checks or CAL image access in this package."""
import csv
from common import BASE, HERE, read, write, sha

assert not (HERE/'FREEZE.json').exists(), 'Never revise an activated freeze'
sources, inputs, candidates = {}, {}, {}
parent = BASE/'handoff/r22_mechanism_review_20261024'
files = dict(read(parent/'FREEZE.json')['files'])
for path, digest in files.items(): assert sha(BASE/path) == digest, path
files[str((parent/'FREEZE.json').relative_to(BASE))] = sha(parent/'FREEZE.json')
def pin(p):
    from pathlib import Path
    p = Path(p)
    try: key = str(p.relative_to(BASE))
    except ValueError: key = str(p)
    files[key] = sha(p)

manifest = BASE/'ssl_pilot_v10_r22c/v10/T00/V10_DATA_MANIFEST.csv'
pin(manifest)
with manifest.open() as f: records = list(csv.DictReader(f))
for ds in ('bf', 'epfl'):
    u = read(BASE/'handoff/r22_c3_codex_20261023/units'/f'r22_pmi3_{ds}_MLPp1_s1001.json')
    lockpath = u['data_lock']['path']; assert sha(lockpath) == u['data_lock']['sha256']; pin(lockpath)
    lock = read(lockpath)
    if ds == 'bf':
        rr = [r for r in lock['files'] if r['role'] == 'CAL']
        cores = sorted(r['core'] for r in rr); hashes = {r['core']: r['sha256'] for r in rr}; scenes = {c:c for c in cores}
        assert len(cores) == 8
    else:
        rr = [r for r in lock['windows'] if r['role'] == 'CAL']
        cores = sorted(r['window'] for r in rr); hashes = {r['window']:r['sha256'] for r in rr}
        scenes = {r['record_id'].replace('/', '_'):r['group_id'] for r in records if r['dataset']=='EPFL_RGB_NIR' and r['role']=='CAL' and r['inclusion']=='include'}
        assert len(cores) == 10
        for key in ('record', 'group_record'): pin(u['guard'][key])
    inputs[ds] = dict(cores=cores, cache_hashes=hashes, scenes=scenes, perturbation_indices=[0,1],
                     perturbation_rule='Frozen host Q.perturbations_p2(core), first2; no outcome selection')
    for arm in ('vcs_exact', 'bce', 'nce'):
        old = parent/'runs'/f'reg_terms_{ds}_{arm}_1001'
        rec = read(old/'COMPLETE.json')
        assert rec['status'] == 'COMPLETE' and rec['freeze_sha256'] == sha(parent/'FREEZE.json')
        assert rec['rerank_sha256'] == sha(old/'TOPK_RERANK.jsonl')
        pin(old/'COMPLETE.json'); pin(old/'TOPK_RERANK.jsonl')
        prior_rows = [__import__('json').loads(x) for x in (old/'TOPK_RERANK.jsonl').read_text().splitlines()]
        candidates[f'{ds}_{arm}_1001'] = {f"{r['core']}|{r['perturbation']}":r['readouts']['P1']['candidates'] for r in prior_rows}
        assert len(prior_rows) == 8
        for seed in (1001,1002,1003):
            p = BASE/'handoff/r22_geometry_long_v2_20261024/runs'/f'{ds}_{arm}_{seed}'
            r = read(p/'COMPLETE.json'); assert r['status'] == 'COMPLETE'
            ckpt = p/'checkpoints/step_30000.pt'; pin(ckpt); pin(p/'COMPLETE.json')
            sources[f'{ds}_{arm}_{seed}'] = dict(checkpoint=str(ckpt), sha256=sha(ckpt), completion=str(p/'COMPLETE.json'), completion_sha256=sha(p/'COMPLETE.json'))
            if seed == 1001: assert rec['source']['sha256'] == sha(ckpt)
write(HERE/'SOURCE_LOCK.json', sources); write(HERE/'INPUT_LOCK.json', inputs); write(HERE/'CACHED_STARTS.json', candidates)
protocol = dict(schema='r22.registration_trajectory.v1', parent_design_sha256=sha(BASE/'handoff/r22_next_round_20261025/FREEZE.json'),
    hypothesis='Positive-square and Q constraints can redirect an evolving 100-step pose optimizer differently; distinguish optimization path from final energy ranking.',
    units='2datasets x3training-objectives(exact,BCE,NCE) x3seeds(1001,1002,1003) =18 existing clean30000 checkpoints; no retraining.',
    inputs='All8 BF CAL cores / all10 EPFL CAL windows, perturbation0/1, original preprocessing/positions. CAL cache hashes only; FIT identifiers metadata only. No DEV image/cache or TEST access.',
    starts='Each checkpoint/problem shares3 original COS coarse starts across all objectives. Reuse48 existing seed1001 CAL P1 top3 records; otherwise one new CAL COS coarse pass. No exact reranking or truth-based choice of starts.',
    objectives=['P1','POS_QUAD=P1-P2/2','J=P1-P2/2-Q1-Q2/2'],
    optimizer='Original fresh Adam lr.5, betas(.9,.999), eps1e-8, no weight decay,100updates, scaled rotation u=theta*RHO and original rotation/translation projection after every update. Record actual evolving moments, no unit-normalized gradient surrogate.',
    trace='At0,1,5,10,25,50,100 record pose/error, all6 scalar terms/pose gradients, Adam moments. Each trajectory starts from identical pose/fresh optimizer; parameters/BN frozen and restored by host evaluation context.',
    primary='Scene-equal mean error at step100 across same three starts; comparisons POS_QUAD-P1 and J-POS_QUAD. Keep separate dataset/training-objective/seed; full trajectories at registered steps.',
    secondary='Original largest-own-energy final selection among3 terminal poses, ties choose first coarse rank; selected strict/loose, median/tail, win/loss scenes. BF fullpx9/42, EPFLnativepx2.25/10.5. No pooled dataset mean, no truth selection.',
    criterion='Directional mechanism if >=2/3seed same primary sign and leave-one-scene-out sign robust, report magnitude/allcases. Descriptive development signal only, not a permanent early stopping rule; completeall18.',
    gates='CPU finite-difference exact terms; literal-host100-step CPU optimizer equivalence/actualstate50 roundtrip; restoreall18 models and CAL identity. Then eachdataset firstCALwindow perturb0 seed1001exact,3starts x3objectives x100actualGPUupdates, finite/state/role checks. BothPASS beforefull18.',
    budget=dict(estimated_gpu_hours=[2,6], full_hard_gpu_hours=18, gpu_gates_hard_hours=1, full_jobs=18, gate_jobs=2, output_limit_gib=2),
    failure='Fail closed on identity/nonfinite/data/state errors; immutable attempt/failure; no implicitretry, freeze revision separately. No outcome-based stopping.',
    limits='CAL repeatedly used for development; 2perturbations/window are correlated. Search nondeterminism and drivers recorded. Mean-three-start error separates path effect; selected error additionally includes ranking. Removing terms gives mechanism controls, not full VCS-QMI.',
    operations='Existing unique pool only, original storage gates home1450GiB/projects300GB. No TEST/publicpush/deletion/subagents/newwatcher.')
write(HERE/'PROTOCOL.json', protocol)
header = '''#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --partition=L40S
#SBATCH --exclude=node51,node53,node60,node61
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nice=60
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH
'''
(HERE/'gpu.sbatch').write_text(header+f'{BASE}/env/bin/python -B -s {HERE}/trajectory.py "$@"\nif [[ "$1" == "gate" ]]; then /usr/bin/python3 {HERE}/admit.py; fi\n')
(HERE/'check.sbatch').write_text(f'''#!/usr/bin/env bash
#SBATCH --job-name=r22_regtraj_cpu
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output={BASE}/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
{BASE}/env/bin/python -B -s {HERE}/trajectory.py check
/usr/bin/python3 {HERE}/admit.py
''')
spec = dict(gates=[], full=[])
for ds in ('bf','epfl'):
    name = f'r22_regtraj_gate_{ds}'
    spec['gates'].append(dict(name=name,args=['sbatch','--time=00:30:00',f'--job-name={name}',str(HERE/'gpu.sbatch'),'gate','--ds',ds],completion=str(HERE/'runs'/f'gate_{ds}'/'COMPLETE.json'),group='other_reg_trajectory_gate',freeze=str(HERE/'FREEZE.json')))
    for arm in ('vcs_exact','bce','nce'):
        for seed in (1001,1002,1003):
            name=f'r22_regtraj_{ds}_{arm}_s{seed}'
            spec['full'].append(dict(name=name,args=['sbatch',f'--job-name={name}',str(HERE/'gpu.sbatch'),'run','--ds',ds,'--arm',arm,'--seed',str(seed)],completion=str(HERE/'runs'/f'{ds}_{arm}_{seed}'/'COMPLETE.json'),group='other_reg_trajectory_'+ds,freeze=str(HERE/'FREEZE.json')))
write(HERE/'JOBS_SPEC.json',spec)
for p in HERE.iterdir():
    if p.is_file(): pin(p)
write(HERE/'FREEZE.json',dict(parent_design_sha256=protocol['parent_design_sha256'],files=files))
print('FROZEN',sha(HERE/'FREEZE.json'),len(files),flush=True)
