#!/usr/bin/env python3
"""Freeze PROTOCOL_FROZEN_NR5.json before any formal unit."""
import glob, hashlib, json, sys, time
from pathlib import Path
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr5_20261104'); OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_nr5/NR5_20261104')
fz = H/'PROTOCOL_FROZEN_NR5.json'
if fz.exists(): sys.exit('already frozen')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
files = sorted(glob.glob(str(H/'code/nr5/*.py')) + glob.glob(str(H/'sbatch/*.sbatch')) + [str(H/'sbatch/common_env.sh')] + glob.glob(str(H/'configs_production/NR5_*.json')))
units = sorted(Path(f).stem for f in glob.glob(str(H/'configs_production/NR5_*.json')))
assert len(units) == 30
nr4 = sorted(glob.glob('/home/infres/yinwang/CS_QMI/handoff/nr4_20261104/code/nr4/*.py')) + sorted(glob.glob('/home/infres/yinwang/CS_QMI/handoff/nr3_20261104/code/nr3/*.py')) + sorted(glob.glob('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/code/src/nr*/*.py'))
P = dict(protocol_ids=['NR5A_NR4_SUFFIX_CONFIRM_V1', 'NR5B_PRODUCT_Q_NATIVE_ENCODER_V1'], frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         evidence_snapshot='e2e1509b38c73bc3c29ab475f2afb6c19ae7326f', package_zip_sha256=sha('/home/infres/yinwang/CS_QMI/VCS_NR4_Review_NR5_Server_20261104.zip'),
         units=units, file_sha256={f: sha(f) for f in files}, imported_nr2_nr3_nr4_code_sha256={f: sha(f) for f in nr4},
         nr4_hidden_manifests={str(f): sha(f) for f in sorted(Path('/projects/EEG-foundation-model/yinghao/outputs_nr4/NR4_20261104').glob('raw/*/*/*_MANIFEST.json'))},
         acceptance_sha256=sha(OUT/'acceptance/ACCEPTANCE_NR5.json'), replay_sha256=sha(OUT/'acceptance/REPLAY_EQUIVALENCE_NR5.json'), dedup=json.loads((H/'DEDUP_NR5.json').read_text()),
         recipe=dict(A='NR4 trainer path from the receipt-resolved NR3/NR4 folded prefix; only training_seed 2002 / sampler 102002 / LoRA 2002 change; 4000 steps; HREF; eval stages 0/250/500/1000/2000/4000; primary 4000',
                     B='pretrained SigLIP2 + zero LoRA, head COS or G_ZERO at NR1 a0,b0 (head seed 1001); U reference alpha 1 (log w 0), sampler 20261104501, LoRA 1001; head lr 1e-4->1e-5, LoRA lr 1e-5->1e-6 cosine 8000; wd 1e-4; one global clip 2; eval stages 0/250/500/1000/2000/4000/8000; primary 8000',
                     tail='LoRA r8 a8 q/v blocks 10-11 both towers; blocks 0-9 from the verified NR4 hidden cache; blocks 10-11 + final LN + pooler every step at fixed B=32; direct backward (replay verified)',
                     precision='fp32, TF32 off, eager, MHA fastpath off, deterministic algorithms; eval float64', hardware='TAIL units on RTX6000PRO (hidden cache GPU type); FROZEN on RTX6000PRO,L40S,H100',
                     eval='frozen base Top100, all DEV queries; secondary student cosine full gallery; endpoint full-gallery current score'),
         read_TEST=False, truth_in_training=False)
fz.write_text(json.dumps(P, indent=1)); fz.chmod(0o444); print('frozen', len(files), 'files', len(units), 'units')
