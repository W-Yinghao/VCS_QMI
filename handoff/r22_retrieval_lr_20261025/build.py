"""One-time immutable execution preparation. No numerical training here."""
from datetime import datetime, timezone
from lr_common import BASE,HERE,OLD,read,write,sha

assert not (HERE/'FREEZE.json').exists(), 'Never rewrite an activated freeze'
analysis=BASE/'handoff/r22_watch10m_20261024/ANALYSIS_SHUFFLED_RETRIEVAL_ALL3_20261025.json'
a=read(analysis);assert a['status']=='COMPLETE_REVIEW' and a['criterion']['allocation_signal']
protocol=dict(schema='r22.retrieval.lr-factorial.v1',utc=datetime.now(timezone.utc).isoformat(),
 authorization='User asked to analyze and design next round now; gated submissions in original pool already authorized.',
 hypothesis='Real reliability allocation improves late stability, but best DEV remains below constant. A matched host-schedule intervention tests whether optimization preserves allocation benefit while reducing the selected-model deficit.',
 evidence=dict(analysis=str(analysis),sha256=sha(analysis),real_minus_shuffle_last5=20.67850098619337,real_minus_constant_best=-3.2938856015779),
 design='2 weight arms (real protected VCS,constant .5) x 2 periodic LR schedules (lr_update30,lr_update10) x seeds901/902/903. Reuse six completed period30 controls; add six period10 runs from matched random initialization.',
 implementation='Original frozen train_one/WeightedStudent.fit/Objective/loaders unchanged. Process-local wrapper replaces only dataclass cfg.lr_update after original loader validation. Same forty epochs, 45320 updates, SGR, NCEtau.05, sum loss, normalizer.55, A=1, U files unchanged.',
 schedule=dict(formula='initial_lr * 0.1 ** (zero_based_epoch // lr_update)',initial_lr=2e-4,
   original_period=30,new_period=10,new_decay_epochs_one_based=[11,21,31],original_decay_epochs_one_based=[31],
   vectors={str(p):[2e-4*.1**(e//p) for e in range(40)] for p in (30,10)},
   clarification='Prior design called decay at10/30 but explicitly specified lr_update. This executable schedule is periodic, not a single shifted milestone. It jointly changes decay timing/count and integrated LR; no attribution to timing alone.'),
 primary=dict(endpoint='best full DEV rSum of forty candidates, earliest epoch on tie',
   contrasts=['VCS_period10 minus VCS_period30','constant_period10 minus constant_period30',
     '(VCS10-VCS30)-(constant10-constant30)','VCS10 minus constant10'],
   criterion='A useful schedule gain requires mean>=3rSum and >=2/3positive seeds. Weight-by-schedule interaction separately uses same rule. Report all contrasts regardless pass. A positive interaction alone does not establish VCS10>constant10 or superiority to BCE.'),
 secondary=['epoch40','last5 mean','last10 mean','all40 DEV/FIT curves','best epoch and post-peak decline','all attempts and costs'],
 matching='Same per-seed init, original full pool/order, first8 batch hashes, original manifest except lr_update, weight file, precision, GPU model and all software plus driver required equal to reused control. Host recorded and differences disclosed. Full entry preflights each seed before first training update.',
 gates=dict(CPU='Six source/weight/config checks, exact production fit on a tiny synthetic host for all40 epochs at both periods; observe actual Adam.step LR, checkpoint/log paths and finite updates.',
   GPU='One RTX job runs both weight arms x both schedules with original SGR fit for forty sparse epochs: original first8 FIT batches at epoch0, first FIT batch each later epoch, 47updates percell/188total. Same frozen input/init/order/precision/software/driver, all actual Adam.step LRs, finite losses, first16DEV80captions, changed model/checkpoint optimizer LR/read trace. Sparse results are technical only, never performance.',
   full='Only after CPU and four GPU cells PASS; no relaxed tolerance/retry. Original full iterator is restored and full uses original train_one without fit interception.'),
 budget=dict(new_jobs=6,reused_jobs=6,full_epochs=40,updates_per_job=45320,
   estimated_gpu_hours=[36,45],full_hard_gpu_hours=143,gate_hard_gpu_hours=1,cpu_hard_minutes=25,retained_output_gib_max=6,
   basis='Six control receipts total37.251048GPUh; gate estimate5-20min, hard1h; source/CPU/prior failed attempts separately retained.'),
 stopping='All six complete fixed forty epochs regardless interim DEV. Nonfinite/input/init/environment/hash failure stops affected unit, preserves attempt; no automatic retry or outcome-based LR/weight change. No posthoc schedule grid.',
 interpretation='Development only: schedule chosen after reused DEV peaks8-14 and shuffled study; three seeds including selected pilot901. Not independent confirmation, no TEST. Real VCS weights share scalar ordering with BCE: this study tests host weighting interaction, not unique VCS superiority.',
 operations='Original unique pool and submission lock only; no daemon/dependency/timer/subagent, no public push/deletion/TEST. Home<1450GiB and projects>=300GB guards retained. Six justified jobs need not fill8+12 target.',
 other_routes='Registration/KD mechanism batches complete; retain COS/P1 readout. No automatic longer training/repeat/AIG bank or MOS grid. Original paper theory unchanged.')
write(HERE/'PROTOCOL.json',protocol)
locks={}
for seed in (901,902,903):
 for method in ('vcs','constant'):
    p=OLD/'runs'/f'retrieval_{method}_s{seed}';r=read(p/'COMPLETE.json');assert r['status']=='COMPLETE'
    names=['COMPLETE.json','run_manifest.json','train_log.json','done.json','best.pt',f"OPEN_TRACE_{r['job']}.json"]
    locks[f'{method}_{seed}']=dict(directory=str(p),files={n:sha(p/n) for n in names})
write(HERE/'SOURCE_LOCK.json',dict(controls=locks))
env='''set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH
'''
(HERE/'gpu.sbatch').write_text('''#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO
#SBATCH --exclude=node51,node53,node60,node61
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nice=60
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
'''+env+f'{BASE}/env/bin/python -B -s {HERE}/run.py "$@"\n'+f'if [[ "$1" == "gate" ]]; then /usr/bin/python3 {HERE}/admit.py; fi\n')
(HERE/'cpu.sbatch').write_text('''#!/usr/bin/env bash
#SBATCH --job-name=r22_lrs10_cpu
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
'''+env+f'{BASE}/env/bin/python -B -s {HERE}/run.py check\n/usr/bin/python3 {HERE}/admit.py\n')
spec={'gates':[],'full':[]}
def job(group,name,args,completion,limit):
 spec[group].append(dict(name=name,args=['sbatch','--partition=RTX6000PRO',f'--time={limit}',f'--job-name={name}',str(HERE/'gpu.sbatch'),*args],
  completion=str(HERE/completion),group='other_retrieval_lr_'+('gate' if group=='gates' else 'full'),freeze=str(HERE/'FREEZE.json')))
job('gates','r22_lrs10_gate',['gate'],'gates/COMPLETE.json','01:00:00')
for seed in (901,902,903):
 for method in ('vcs','constant'):
  job('full',f'r22_lrs10_{method}_s{seed}',['run','--seed',str(seed),'--method',method],f'runs/retrieval_{method}_s{seed}/COMPLETE.json','23:50:00')
write(HERE/'JOBS_SPEC.json',spec)
files=dict(read(OLD/'FREEZE.json')['files']);files[str((OLD/'FREEZE.json').relative_to(BASE))]=sha(OLD/'FREEZE.json')
for rel,h in files.items():assert sha(BASE/rel)==h,rel
for p in [*HERE.glob('*.py'),*HERE.glob('*.sbatch'),HERE/'PROTOCOL.json',HERE/'SOURCE_LOCK.json',HERE/'JOBS_SPEC.json',analysis]:
 files[str(p.relative_to(BASE))]=sha(p)
write(HERE/'FREEZE.json',dict(schema='r22.retrieval.lr.freeze.v1',files=dict(sorted(files.items()))))
print('FROZEN',sha(HERE/'FREEZE.json'),len(files))
