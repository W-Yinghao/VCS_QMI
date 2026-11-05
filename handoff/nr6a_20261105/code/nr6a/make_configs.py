#!/usr/bin/env python3
"""NR6-A configs (24): NR5-A confirmation design with only seeds / condition changed.
Flickr60/CC: copy of the NR5-A production config of the same condition/scope/objective, seeds -> 3003/103003/3003.
COCO60 (s2002, s3003): built by the identical NR5 make_configs A rule from the COCO60 NR4 production config (receipt-resolved prefix),
then asserted equal to the NR5-A Flickr60 config of the same scope/objective on every condition-independent key."""
import hashlib, json
from pathlib import Path
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr6a_20261105'); N5 = Path('/home/infres/yinwang/CS_QMI/handoff/nr5_20261104'); N4 = Path('/home/infres/yinwang/CS_QMI/handoff/nr4_20261104')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); out = H/'configs_production'
rec4 = json.loads((N4/'PREFIX_AND_REUSE_RECEIPT_NR4.json').read_text())
COND_KEYS = {'unit_id', 'condition', 'prefix', 'prefix_sha256', 'parent_unit', 'reference_unit', 'nr3_reference_config_sha256', 'nr4_package_config_sha256', 'nr4_source_config',
             'nr4_source_config_sha256', 'nr4_reference_unit', 'nr5_package_config_sha256', 'training_seed', 'train_sampler_seed', 'adapter_seed', 'source_commit'}
units = []
def write(u, c, o, sc, seed, base):
    c2 = dict(base); c2.update(unit_id=u, training_seed=seed, train_sampler_seed=100000 + seed, adapter_seed=seed, protocol_id='NR5A_NR4_SUFFIX_CONFIRM_V1',
                               nr6a=dict(design='NR6A_STRUCTURE_CONFIRM_V1', seed=seed), nr5a_template=f'NR5_A_Flickr60_{sc}_{o}_t2002')
    ref = json.loads((N5/f'configs_production/NR5_A_Flickr60_{sc}_{o}_t2002.json').read_text())
    diff = sorted(k for k in set(ref) | set(c2) if k not in COND_KEYS | {'nr6a', 'nr5a_template', 'nr5_package_config_sha256'} and ref.get(k) != c2.get(k))
    assert not diff, (u, diff)
    assert c2['condition'] == c and c2['objective'] == o and c2['scope'] == sc and c2['new_updates'] == 4000
    (out/f'{u}.json').write_text(json.dumps(c2, indent=1)); units.append(u)
for c in ('Flickr60', 'CC152K_RECOVERED'):
    for sc in ('FROZEN', 'TAIL_LORA8'):
        for o in ('J', 'BCE', 'NCE'):
            write(f'NR6A_{c}_{sc}_{o}_t3003', c, o, sc, 3003, json.loads((N5/f'configs_production/NR5_A_{c}_{sc}_{o}_t2002.json').read_text()))
for seed in (2002, 3003):
    for sc in ('FROZEN', 'TAIL_LORA8'):
        for o in ('J', 'BCE', 'NCE'):
            src = N4/'configs_production'/(f'NR4_TAIL_LORA8_COCO60_G_{o}_s1001.json' if sc == 'TAIL_LORA8' else f'NR4_FROZENCHK_COCO60_G_{o}_s1001.json')
            b = json.loads(src.read_text())
            b.update(group='A_CONFIRM', init='NR4_PREFIX', readout='G', nr4_source_config=str(src), nr4_source_config_sha256=sha(src), prefix=rec4['prefixes']['COCO60']['path'],
                     prefix_sha256=rec4['prefixes']['COCO60']['sha256'], prefix_resolved_by=str(N4/'PREFIX_AND_REUSE_RECEIPT_NR4.json'),
                     nr4_reference_unit=f"NR4_{'TAIL_LORA8' if sc == 'TAIL_LORA8' else 'FROZEN(NR3-B)'}_COCO60_{o}_s1001", read_TEST=False)
            write(f'NR6A_COCO60_{sc}_{o}_t{seed}', 'COCO60', o, sc, seed, b)
print(len(units), 'configs, all equal to the NR5-A template on condition-independent keys')
