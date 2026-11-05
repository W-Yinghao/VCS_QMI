#!/usr/bin/env python3
"""Write NR5_A1.json (amendment; PROTOCOL_FROZEN_NR5.json is not edited). Group -> GPU type decided at submit time by availability,
recorded here; per-unit actual GPU/driver/torch are recorded by each unit (ATTEMPTS.txt, STEP0_GRAD_CHECK.json, TRAIN_COST.json)."""
import glob, hashlib, json, sys, time
from pathlib import Path
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr5_20261104'); OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_nr5/NR5_20261104')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
groups = json.loads(sys.argv[1]); accept = sys.argv[2]
files = sorted(glob.glob(str(H/'code/nr5a1/*.py'))) + [str(H/'sbatch/unit_a1.sbatch'), str(H/'sbatch/gpu_l40s.sbatch')]
units = sorted(Path(f).stem for f in glob.glob(str(H/'configs_production/NR5_*.json')) if not ('_A_' in f and '_FROZEN_' in f))
A = dict(amendment='NR5_A1', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), frozen_protocol_unchanged_sha256=sha(H/'PROTOCOL_FROZEN_NR5.json'),
         user_decision_verbatim='不行不行，现在这个只能用一种GPU还是太慢了，你来组织一下，至少我们要能在L40S上跑',
         changes=['allowed GPU types RTX6000PRO, L40S, H100 (exclude node51,node53,node60); one GPU type per matched group, chosen at submit time by availability (never by results)',
                  'groups: A = per condition FROZEN and TAIL x J/BCE/NCE; B = per condition COS and G_ZERO x J/BCE/NCE',
                  'step-0 contracts (train) and stage-0 contract (eval): bitwise / absolute-1e-5 replaced by the NR1 package elementwise rule |d| <= 3e-5 + 3e-5|ref|, the same rule on every GPU type; the stage-0 order may differ from the reference only where the reference scores of the reordered candidates are equal within 2x that tolerance; observed max diffs/ratios are recorded per unit',
                  'NR4 hidden cache (built on RTX) reused as is; not rebuilt or duplicated',
                  'code: nr5a1/{train,evaluate,train_unit}.py = byte copies of nr5/* with only the contract tolerances changed (see IMPLEMENTATION_DIFF_NR5_A1.patch); unit_a1.sbatch = unit.sbatch calling them + this gate'],
         kept_units_under_original_code=sorted(Path(f).stem for f in glob.glob(str(H/'configs_production/NR5_A_*_FROZEN_*.json'))),
         kept_reason='A FROZEN units completed (L40S) before the amendment; FROZEN uses only cached globals, so the stage-0 contract was exact (0) under the original code',
         cancelled_pending_jobs=[1088432, 1088433, 1088434, 1088438, 1088439, 1088440, 1088441, 1088442, 1088443, 1088463, 1088468, 1088469, 1088470],
         group_gpu_type=groups, units=units, l40s_acceptance=accept, l40s_acceptance_sha256=sha(accept), file_sha256={f: sha(f) for f in files})
(H/'NR5_A1.json').write_text(json.dumps(A, indent=1)); (H/'NR5_A1.json').chmod(0o444); print('NR5_A1 written', len(units), 'units')
