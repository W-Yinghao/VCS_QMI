#!/usr/bin/env python3
import glob, hashlib, json, sys, time
from pathlib import Path
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr6a_20261105'); fz = H/'PROTOCOL_FROZEN_NR6A.json'
if fz.exists(): sys.exit('already frozen')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
files = sorted(glob.glob(str(H/'code/nr6a/*.py')) + glob.glob(str(H/'sbatch/*'))) + sorted(glob.glob(str(H/'configs_production/*.json')))
imp = sorted(glob.glob('/home/infres/yinwang/CS_QMI/handoff/nr5_20261104/code/nr5*/*.py') + glob.glob('/home/infres/yinwang/CS_QMI/handoff/nr4_20261104/code/nr4/*.py')
             + glob.glob('/home/infres/yinwang/CS_QMI/handoff/nr3_20261104/code/nr3/*.py') + glob.glob('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/code/src/nr*/*.py'))
units = sorted(Path(f).stem for f in glob.glob(str(H/'configs_production/NR6A_*.json')))
assert len(units) == 24
P = dict(protocol_id='NR6A_STRUCTURE_CONFIRM_V1', frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         authorisation_verbatim='下来这两个你都自己来分析分析结果，设计新一轮的方向，不要让GPU空着',
         reason='TAIL_LORA8 - FROZEN positive in all 6 cells for both seeds (NR4 s1001: F60 +2.23/+2.09/+2.70, CC +1.88/+2.45/+3.61; NR5-A s2002: F60 +2.62/+2.05/+2.94, CC +1.30/+1.73/+3.32); TAIL J-NCE differs by seed (F60 -0.18/+0.06, CC -1.73/-2.16). Third seed on Flickr60/CC and second+third seed on COCO60 (NR4 s1001 only: +2.54/+2.76/+3.90).',
         design='NR5-A confirmation unchanged except seeds/condition: {Flickr60, CC152K_RECOVERED} x J/BCE/NCE x {FROZEN, TAIL_LORA8} x seed 3003 (training 3003, sampler 103003, adapter 3003); COCO60 x same x seeds 2002 and 3003; NR4 stage-0 folded prefix (receipt-resolved), 4000 steps; primary stage 4000',
         code='NR5_A1 trainer/evaluator (nr5a1.*) imported read-only; NR4 hidden cache reused (COCO60: images in RAM, texts memmap); no new large cache',
         gpu_rule='NR5_A1: one GPU type per matched group (condition x seed: FROZEN and TAIL x 3 objectives), chosen at submit by availability (prefer RTX6000PRO for a whole group when free/runfill-preemptible); non-RTX uses the registered NR5_A1 tolerance rule and replay on <64 GiB',
         units=units, file_sha256={f: sha(f) for f in files}, imported_code_sha256={f: sha(f) for f in imp},
         nr5_a1_sha256=sha('/home/infres/yinwang/CS_QMI/handoff/nr5_20261104/NR5_A1.json'), read_TEST=False, truth_in_training=False)
fz.write_text(json.dumps(P, indent=1)); fz.chmod(0o444); print('frozen', len(files), 'files', len(units), 'units')
