#!/usr/bin/env python3
"""Freeze PROTOCOL_FROZEN_NR4.json (before any formal unit): code/config/package hashes, data bindings, acceptance + receipts."""
import glob, hashlib, json, sys, time
from pathlib import Path
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr4_20261104'); OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_nr4/NR4_20261104')
fz = H/'PROTOCOL_FROZEN_NR4.json'
if fz.exists(): sys.exit('already frozen')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
files = sorted(glob.glob(str(H/'code/nr4/*.py')) + glob.glob(str(H/'sbatch/*.sbatch')) + [str(H/'sbatch/common_env.sh')] + glob.glob(str(H/'configs_production/NR4_*.json')))
units = sorted(Path(f).stem for f in glob.glob(str(H/'configs_production/NR4_*_s1001.json')) if 'FROZENCHK' not in f)
acc = json.loads((OUT/'acceptance/Flickr60_J/ACCEPTANCE_NR4.json').read_text())
rec = {Path(f).parent.name: json.loads(Path(f).read_text())['decision'] for f in glob.glob(str(OUT/'reuse_check/*/RECEIPT.json'))}
assert len(units) == 24 and len(rec) == 12, (len(units), len(rec))
P = dict(protocol_id='NR4_MATCHED_SCOPE_V1', frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), evidence_snapshot='11b5af74fc6628069c945369ec9b545ecd6de793',
         package_zip_sha256=sha('/home/infres/yinwang/CS_QMI/VCS_NR3_Review_NR4_EncoderAdapt_20261104.zip'), units=units, frozen_controls_reuse=rec,
         file_sha256={f: sha(f) for f in files},
         nr2_nr3_code_sha256={f: sha(f) for f in sorted(glob.glob('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/code/src/nr*/*.py') + glob.glob('/home/infres/yinwang/CS_QMI/handoff/nr3_20261104/code/nr3/*.py'))},
         raw_manifests={str(f): sha(f) for f in sorted(OUT.glob('raw/*/*/RAW_MANIFEST.json'))}, hidden_manifests={str(f): sha(f) for f in sorted(OUT.glob('raw/*/fit/HIDDEN_MANIFEST.json'))},
         acceptance_sha256=sha(OUT/'acceptance/Flickr60_J/ACCEPTANCE_NR4.json'), replay_equivalence_sha256=sha(OUT/'acceptance/Flickr60_J/REPLAY_EQUIVALENCE.json'),
         recipe=dict(steps=4000, stage_checkpoints=[0, 250, 500, 1000, 2000, 4000], primary='stage 4000 (total 12000), all DEV queries, frozen base top100, float64 scores',
                     optimizer='fresh AdamW, two groups (head, adapter) same lr 1e-5->1e-6 cosine, wd 1e-4, global clip 2 over head+adapter, one clip+step per update',
                     sampler='HREF alpha .5, Top100 frozen base FIT, 32 q/dir x (1+63), seed 101001', adapter_seed=1001, precision='fp32, TF32 off, eager attention, MHA fastpath off, deterministic algorithms, eval fp64',
                     tail='LoRA r8 alpha8 dropout0 on {vision,text}.encoder.layers.{10,11}.self_attn.{q,v}_proj; blocks 0-9 from verified frozen hidden cache; blocks 10-11+final LN+pooler every step, fixed B=32; direct backward',
                     post='u=(I+B_x A_x)x, v=(I+B_y A_y)y on cached globals, rank 32, no bias, A seeded (1001 img / 1002 txt), B=0',
                     hardware='TAIL on RTX6000PRO (frozen caches + hidden cache extracted there; bitwise t0); POST on RTX6000PRO,L40S,H100'),
         read_TEST=False, truth_in_training=False)
fz.write_text(json.dumps(P, indent=1)); fz.chmod(0o444); print('frozen', len(files), 'files', len(units), 'units')
