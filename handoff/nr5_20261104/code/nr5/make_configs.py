#!/usr/bin/env python3
"""NR5 production configs (30) + 2 equivalence-receipt configs.
A: the matched NR4 production config (TAIL: NR4_TAIL_LORA8_*; FROZEN: NR4_FROZENCHK_* = NR3-B) with ONLY the registered randomness
   changed (training_seed 2002, train_sampler_seed 102002, adapter_seed 2002) + NR5 identity fields; prefix resolved via the NR4
   receipt PREFIX_AND_REUSE_RECEIPT_NR4.json.
B: NR4 TAIL skeleton (same sampler sizes / precision / LoRA) with init NATIVE, reference U (alpha 1), readout COS/G_ZERO, head lr
   1e-4->1e-5, adapter lr 1e-5->1e-6, 8000 steps, sampler seed 20261104501, adapter seed 1001, head seed 1001, NR1 calibration.
Dedup: protocol + full config sha256 against every NR2/NR3/NR4 production config; written to DEDUP_NR5.json."""
import csv, glob, hashlib, json
from pathlib import Path
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr5_20261104'); N4 = Path('/home/infres/yinwang/CS_QMI/handoff/nr4_20261104')
out = H/'configs_production'; out.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows = list(csv.DictReader(open(H/'package/MATRIX_NEW_30.csv', encoding='utf-8-sig')))
rec4 = json.loads((N4/'PREFIX_AND_REUSE_RECEIPT_NR4.json').read_text())
made = []
for r in rows:
    u, c, o, sc, rd = r['unit_id'], r['condition'], r['objective'], r['scope'], r['readout']; pk = json.loads((H/f'package/configs/{u}.json').read_text())
    if r['group'] == 'A_CONFIRM':
        src = N4/'configs_production'/(f'NR4_TAIL_LORA8_{c}_G_{o}_s1001.json' if sc == 'TAIL_LORA8' else f'NR4_FROZENCHK_{c}_G_{o}_s1001.json')
        cfg = json.loads(src.read_text())
        cfg.update(unit_id=u, protocol_id='NR5A_NR4_SUFFIX_CONFIRM_V1', group='A_CONFIRM', init='NR4_PREFIX', readout='G',
                   training_seed=int(pk['training_seed']), train_sampler_seed=int(pk['sampler_seed']), adapter_seed=int(pk['adapter_seed']),
                   nr4_source_config=str(src), nr4_source_config_sha256=sha(src), prefix=rec4['prefixes'][c]['path'], prefix_sha256=rec4['prefixes'][c]['sha256'],
                   prefix_resolved_by=str(N4/'PREFIX_AND_REUSE_RECEIPT_NR4.json'), nr4_reference_unit=f"NR4_{'TAIL_LORA8' if sc == 'TAIL_LORA8' else 'FROZEN(NR3-B)'}_{c}_{o}_s1001")
        assert cfg['new_updates'] == pk['steps'] == 4000 and cfg['lr_start'] == pk['head_lr_start'] and cfg['lr_end'] == pk['head_lr_end']
    else:
        cfg = json.loads((N4/f'configs_production/NR4_TAIL_LORA8_{c if c != "Flickr00" else "Flickr00"}_G_{o}_s1001.json').read_text()) if c != 'Flickr00' else json.loads((N4/f'configs_production/NR4_TAIL_LORA8_Flickr00_G_{o}_s1001.json').read_text())
        for k in ('parent_unit', 'parent_step', 'initial_score', 'calibration', 'base_prefix', 'reference_unit', 'nr3_reference_config_sha256', 'nr4_package_config_sha256',
                  'optimizer_state', 'calibration_pool_seed', 'witness_pool_seed', 'calibration_p_per_direction', 'calibration_q_per_direction', 'calibration_bounds', 'maxiter', 'total_task_updates', 'post_rank'):
            cfg.pop(k, None)
        cfg.update(unit_id=u, protocol_id='NR5B_PRODUCT_Q_NATIVE_ENCODER_V1', group='B_NATIVE', init='NATIVE', readout=rd, reference='U', alpha=1.0,
                   training_seed=1001, train_sampler_seed=int(pk['sampler_seed']), adapter_seed=int(pk['adapter_seed']), head_seed=1001,
                   lr_start=pk['head_lr_start'], lr_end=pk['head_lr_end'], adapter_lr_start=pk['adapter_lr_start'], adapter_lr_end=pk['adapter_lr_end'],
                   new_updates=int(pk['steps']), score_checks=[0, 250, 500, 1000, 2000, 4000, 8000], schedule='cosine', prefix=None,
                   calibration_source='condition NR1 product-Q base-cosine affine a0,b0 (cond_paths calibration), FIT-only, same for all 6 units',
                   encoder='LoRA q/v blocks 10-11 both towers, from locked pretrained SigLIP2, zero LoRA, no task prefix')
        assert pk['proposal_alpha'] == 1.0 and pk['log_w_must_be_zero'] and not pk['use_task_prefix'] and pk['queries_per_direction'] == cfg['queries_per_direction'] == 32
    assert cfg['condition'] == c and cfg['objective'] == o and cfg['scope'] == sc and cfg['weight_decay'] == pk['weight_decay'] and cfg['grad_clip'] == pk['clip_norm']
    cfg['nr5_package_config_sha256'] = sha(H/f'package/configs/{u}.json'); cfg['read_TEST'] = False
    (out/f'{u}.json').write_text(json.dumps(cfg, indent=1)); made.append(u)
# equivalence receipts: NR5 trainer with the unchanged NR4 configs must reproduce NR4 (written as copies)
for u in ('NR4_TAIL_LORA8_Flickr60_G_J_s1001', 'NR4_FROZENCHK_Flickr60_G_J_s1001'):
    (out/f'EQUIV_{u}.json').write_text((N4/f'configs_production/{u}.json').read_text())
old = {}
for f in glob.glob('/home/infres/yinwang/CS_QMI/handoff/nr[234]_20261104/configs_production/*.json'):
    d = json.loads(Path(f).read_text()); d.pop('unit_id', None); old[hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()] = f
dup = {}
for u in made:
    d = json.loads((out/f'{u}.json').read_text()); d.pop('unit_id', None); h = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest(); dup[u] = old.get(h)
(H/'DEDUP_NR5.json').write_text(json.dumps(dict(compared_against=len(old), rule='protocol + full config (minus unit_id) sha256', duplicates={k: v for k, v in dup.items() if v}, n_new=len(made)), indent=1))
print(len(made), 'configs; duplicates:', sum(1 for v in dup.values() if v), 'compared', len(old))
