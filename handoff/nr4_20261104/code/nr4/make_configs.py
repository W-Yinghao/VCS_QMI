#!/usr/bin/env python3
"""NR4 production configs = the NR3-B reference unit's production config (identical sampler/HREF/optimizer/schedule/checks fields)
+ NR4 scope fields from the package config. FROZEN configs (reuse receipt) are the same with scope FROZEN."""
import csv, hashlib, json
from pathlib import Path
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr4_20261104'); N3 = Path('/home/infres/yinwang/CS_QMI/handoff/nr3_20261104/configs_production')
out = H/'configs_production'; out.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows = list(csv.DictReader(open(H/'package/MATRIX_NEW_24.csv', encoding='utf-8-sig')))
def build(unit, cond, obj, scope, ref, pkg=None):
    c = json.load(open(N3/f'{ref}.json')); c.pop('block'); c.pop('nr3_package_config_sha256', None)
    c.update(unit_id=unit, protocol_id='NR4_MATCHED_SCOPE_V1', scope=scope, reference_unit=ref, nr3_reference_config_sha256=sha(N3/f'{ref}.json'),
             base_prefix='NR3_FOLDED_NCE_PREFIX_BEFORE_BRANCH', adapter_seed=1001, start_from_NR3_final=False, read_TEST=False,
             encoder={'FROZEN': 'frozen', 'POST32': 'frozen (output low-rank residual on cached globals)', 'TAIL_LORA8': 'LoRA q/v blocks 10-11 both towers'}[scope],
             post_rank=32 if scope == 'POST32' else None,
             tail_lora=dict(last_layers=2, projections=['q_proj', 'v_proj'], rank=8, alpha=8, dropout=0) if scope == 'TAIL_LORA8' else None,
             tail_backward='direct' if scope == 'TAIL_LORA8' else None, adapter_param_group='same lr/wd as head (second AdamW group)')
    if pkg: c['nr4_package_config_sha256'] = sha(pkg)
    assert c['condition'] == cond and c['objective'] == obj and c['train_sampler_seed'] == 101001 and c['new_updates'] == 4000
    (out/f'{unit}.json').write_text(json.dumps(c, indent=1)); return unit
n = [build(r['unit_id'], r['condition'], r['objective'], r['scope'], r['reference_unit'], H/f"package/configs/{r['unit_id']}.json") for r in rows]
f = [build(f"NR4_FROZENCHK_{r['condition']}_G_{r['objective']}_s1001", r['condition'], r['objective'], 'FROZEN', r['reference_unit']) for r in rows if r['scope'] == 'POST32']
print(len(n), len(f))
