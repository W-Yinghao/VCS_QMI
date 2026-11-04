#!/usr/bin/env python3
"""Freeze NR3 before formal units: code (NR3 + the NR2/NR1 modules it runs), sbatch, package spec, production configs, all per-
condition inputs (caches/observed/calibration/tables/truth/baseline), NR3 calibration files, pools, prefixes, parent NCE
checkpoints, acceptance results; also writes CALIBRATION_RECEIPTS.json and PREFIX_LINEAGE.json."""
import hashlib, json, os, sys, time
from pathlib import Path
N3 = Path('/home/infres/yinwang/CS_QMI/handoff/nr3_20261104'); N2C = Path('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/code')
sys.path.insert(0, str(N3/'code'))
from nr3.paths import OUT, CONDS, cond_paths
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = N3/'PROTOCOL_FROZEN_NR3.json'
if out.exists(): sys.exit('already frozen')
fails = [c for c in CONDS if not json.loads((N3/f'checks/ACCEPTANCE_{c}.json').read_text())['all_pass']]
if fails: sys.exit(f'acceptance not passed: {fails}')
files = [*sorted((N3/'code/nr3').glob('*.py')), N3/'sbatch/unit.sbatch', N3/'sbatch/common_env.sh',
         *sorted((N2C/'src/nr1').glob('*.py')), N2C/'src/nr2/reference.py', N2C/'src/nr2/proposal_reference.py', N2C/'src/nr2/train.py',
         N2C/'site/train_unit_nr2.py', N2C/'site/fit_readout.py', N3/'package/MANIFEST.json', N3/'package/MATRIX.csv', N3/'package/reference/ref_init.py',
         OUT/'inputs_flickr/COPY_RECEIPT.json', *[N3/f'checks/ACCEPTANCE_{c}.json' for c in CONDS]]
cal, lin = {}, {}
for c in CONDS:
    P = cond_paths(c); od = OUT/'prep'/c; R = json.loads((od/'PREP_RECEIPT.json').read_text())
    files += [Path(P['fit'])/'manifest.json', Path(P['dev'])/'manifest.json', P['observed'], P['calibration'], P['tables'], Path(P['tables']).parent/'TABLES_MANIFEST.json',
              P['truth'], P['baseline'], od/'calibration_HCAL.json', od/'POOL_CAL_A.npz', od/'POOL_READ_B.npz', od/'PREP_RECEIPT.json',
              od/'prefix/PREFIX_step_008000.pt', od/'PREFIX_ORDER_CHECK.json', P['parent']('NCE')/'step_008000.pt']
    cal[c] = dict(A=R['A'], B=dict(fit=R['B']['fit'], start=R['B']['start'], bounds=R['B']['bounds'], CAL_A_J=R['B']['CAL_A_J'], READ_B_J=R['B']['READ_B_J']),
                  pools=R['pools'], inputs=R['inputs'])
    lin[c] = dict(parent_unit=R['B']['parent_unit'], parent_checkpoint=R['B']['parent_checkpoint'], parent_checkpoint_sha256=R['B']['parent_checkpoint_sha256'],
                  parent_binding_sha256=R['B']['parent_binding_sha256'], a=R['B']['fit']['a'], b=R['B']['fit']['b'], fold=R['B']['fold'],
                  prefix_checkpoint=R['B']['prefix_checkpoint'], prefix_checkpoint_sha256=R['B']['prefix_checkpoint_sha256'],
                  prefix_binding_sha256=R['B']['prefix_binding_sha256'], prefix_dev_order_check=json.loads((od/'PREFIX_ORDER_CHECK.json').read_text()),
                  branches=[f'NR3_B_NCE2{o}_{c}_G_s1001' for o in ('J', 'BCE', 'NCE')], sampler_seed=101001, new_updates=4000, total_task_updates=12000)
(N3/'CALIBRATION_RECEIPTS.json').write_text(json.dumps(cal, indent=1)); (N3/'PREFIX_LINEAGE.json').write_text(json.dumps(lin, indent=1))
files += [N3/'CALIBRATION_RECEIPTS.json', N3/'PREFIX_LINEAGE.json', N3/'CONFIG_EQUIVALENCE.json']
units = {p.stem: dict(production=sha(p), package=sha(N3/'package/configs'/p.name)) for p in sorted((N3/'configs_production').glob('NR3_*.json'))}
assert len(units) == 30
fz = dict(protocol_id='NR3_REF_INIT_AND_TASK_PREFIX_V1', site_tag='NR3_20261104', frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          authorisation=dict(user_quote='查看这个包，准备并提交相关实验', package='VCS_NR2_Review_NR3_Server_20261104.zip (sha256 86e7d8a7…, MANIFEST 48/48)',
                             evidence_snapshot='a05c6281626203db6cd9de73d45be08ef0a8bba9'),
          scope='30 units per package MATRIX.csv: A 15 (HREF-aware calibrated init, 8000 steps) + B 15 (shared folded NR2 HREF-NCE s1001 step-8000 prefix, NCE->J/BCE/NCE 4000 steps)',
          constraints=['retrieval only; SigLIP2 75de2d55 frozen; G only; original J', 'no TEST; no DEV/truth in calibration or training',
                       'deployment ranking uses s; training g = s - log w on P and reference terms', 'complete matched groups; no early arm cancellation; no extra budget',
                       'parent checkpoints read only; B branches start from one identical folded prefix with fresh AdamW and sampler seed 101001'],
          storage='all NR3 outputs under /projects/EEG-foundation-model/yinghao/outputs_nr3/NR3_20261104; Flickr inputs and NR2 Flickr HREF s1001 parents copied from home '
                  'outputs_nr1/outputs_nr2 with sha256 verification (inputs_flickr/COPY_RECEIPT.json, 11 files cross-checked against PROTOCOL_FROZEN_NR2.json); COCO/CC read in place',
          production_deviations=dict(
              A='production config = registered NR2-HREF s1001 config (protocol_id NR2_QUERY_REFERENCE_V1, NR2 trainer unchanged) + nr3 metadata; only the calibration file differs (calibration_HCAL.json); recipe fields asserted equal to the package config (CONFIG_EQUIVALENCE.json)',
              B='nr3/train_prefix.py = nr2/train.py + minimal changes (TRAIN_PREFIX_DIFF_vs_NR2_train.patch): prefix load, fresh AdamW, seed from train_sampler_seed, lr_start/lr_end, checkpoints stamped 8000+stage_step, step-0 contract s==a*s_parent+b; package config + 2 key aliases',
              calibration='nr3/ref_init.py is a byte copy of the package reference; pools defined in nr3/pools.py (draw order recorded)'),
          hashes={str(p): sha(p) for p in files}, units=units)
out.write_text(json.dumps(fz, indent=2, ensure_ascii=False) + '\n'); os.chmod(out, 0o444); print('frozen', len(fz['hashes']), 'files', len(units), 'units')
