#!/usr/bin/env python3
"""Production configs for the 30 NR3 units from the package configs (the spec).
A: the registered NR2-HREF seed-1001 config of the same condition/objective with only unit_id changed, plus nr3 metadata; trained
   by the unchanged NR2 trainer with calibration_HCAL.json. B: the package config plus two key aliases the NR2 sampler reads
   (queries_per_direction, reference_draws_per_query). Every recipe field is asserted equal to the package config."""
import hashlib, json
from pathlib import Path
N3 = Path('/home/infres/yinwang/CS_QMI/handoff/nr3_20261104'); N2 = Path('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104')
import sys; sys.path.insert(0, str(N3/'code'))
from nr3.paths import cond_paths
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = N3/'configs_production'; out.mkdir(exist_ok=True); rows = []
for f in sorted((N3/'package/configs').glob('NR3_*.json')):
    pk = json.loads(f.read_text()); c = pk['condition']; P = cond_paths(c)
    common = dict(backbone=pk['backbone'], model_revision=pk['model_revision'], normalization=pk['normalization'], architecture=pk['architecture'])
    if pk['block'] == 'A_HREF_CALIBRATED_INIT':
        nr2 = json.loads((N2/P['nr2_config'](pk['objective'])).read_text())
        chk = dict(lr=(nr2['lr'], pk['lr_start']), lr_min=(nr2['lr_min'], pk['lr_end']), wd=(nr2['weight_decay'], pk['weight_decay']), clip=(nr2['grad_clip'], pk['grad_clip']),
                   steps=(nr2['steps'], pk['new_updates']), seed=(nr2['seed'], pk['seed']), sampler_seed=(nr2['seed'], pk['train_sampler_seed']), alpha=(nr2['alpha'], pk['alpha']),
                   hard_k=(nr2['hard_k'], pk['hard_k']), nq=(nr2['queries_per_direction'], pk['n_query_per_direction']), nr=(nr2['reference_draws_per_query'], pk['n_reference_per_query']),
                   ckpts=(nr2['checkpoints'], pk['score_checks']), objective=(nr2['objective'], pk['objective']), reference=(nr2['reference'], pk['reference']),
                   **{k: (nr2[k], v) for k, v in common.items()})
        bad = {k: v for k, v in chk.items() if v[0] != v[1]}; assert not bad, (f.name, bad)
        cfg = dict(nr2, unit_id=pk['unit_id'], nr3_protocol_id=pk['protocol_id'], nr3_block=pk['block'], nr3_condition=c,
                   nr3_package_config_sha256=sha(f), nr3_init='calibration_HCAL.json (prep/<cond>)', nr3_source_nr2_config=P['nr2_config'](pk['objective']))
        diff = sorted(k for k in set(cfg) | set(nr2) if cfg.get(k) != nr2.get(k))
    else:
        cfg = dict(pk, queries_per_direction=pk['n_query_per_direction'], reference_draws_per_query=pk['n_reference_per_query'], nr3_package_config_sha256=sha(f))
        diff = sorted(k for k in set(cfg) | set(pk) if cfg.get(k) != pk.get(k))
    (out/f.name).write_text(json.dumps(cfg, indent=2) + '\n'); rows.append(dict(unit=pk['unit_id'], block=pk['block'], diff_vs_source=diff))
(N3/'CONFIG_EQUIVALENCE.json').write_text(json.dumps(rows, indent=1)); print(len(rows), sorted({tuple(r['diff_vs_source']) for r in rows}))
