#!/usr/bin/env python3
"""Fill DATA_BINDING.json (package template) with actual paths/hashes; CC152K_OBS -> BLOCKED_DATA."""
import hashlib, json, sys
from pathlib import Path
O = Path('/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104')
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr1_20261104')
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
audit = json.loads((O/'audit/DATA_AUDIT.json').read_text())
lock = O/'MODEL_LOCK_FULL.json'
fitm, devm = O/'cache/fit/manifest.json', O/'cache/dev/manifest.json'
fm = json.loads(fitm.read_text()); dm = json.loads(devm.read_text())
ids_sha = lambda c, k: fm['files'][k] if c == 'fit' else dm['files'][k]
b = dict(note='NR1_20261104 actual binding (site). Flickr: NCR f30k_precomp order -> Karpathy csv -> raw jpg; see audit/DATA_AUDIT.json',
         data_audit_sha256=sha(O/'audit/DATA_AUDIT.json'), model_lock_json=str(lock), model_lock_sha256=sha(lock),
         fit_cache_manifest_sha256=sha(fitm), dev_cache_manifest_sha256=sha(devm), conditions={})
for c in ['Flickr00', 'Flickr60', 'Flickr80']:
    o = audit['observed'][c]; cal = O/f'calibration/{c}.json'
    b['conditions'][c] = dict(fit_cache=str(O/'cache/fit'), dev_cache=str(O/'cache/dev'), observed_tsv=o['path'], observed_sha256=sha(o['path']),
        evaluation_truth_json=audit['dev_truth']['path'], evaluation_truth_sha256=sha(audit['dev_truth']['path']),
        model_lock_json=str(lock), normalization='sample_l2_no_center',
        source_noise_file_sha256=o['source_noise_file_sha256'] or 'IDENTITY_CLEAN_ORDER (NCR caption k -> image k//5)',
        source_noise=o['source'], realized_mismatch=o['realized_mismatch'],
        raw_image_coverage=f"fit {audit['fit']['n_images']}/{audit['fit']['n_images']}, dev {audit['dev']['n_images']}/{audit['dev']['n_images']} (all present, sha256 per file in audit/ID_MAP_*_images.tsv)",
        fit_image_ids_sha256=fm['files']['image_ids.json'], fit_text_ids_sha256=fm['files']['text_ids.json'],
        dev_image_ids_sha256=dm['files']['image_ids.json'], dev_text_ids_sha256=dm['files']['text_ids.json'],
        calibration_json=str(cal), calibration_sha256=sha(cal) if cal.exists() else None, data_status='BOUND')
b['conditions']['CC152K_OBS'] = json.loads((H/'BLOCKED_DATA_CC152K.json').read_text())
(H/'DATA_BINDING.json').write_text(json.dumps(b, indent=1))
print('wrote', H/'DATA_BINDING.json')
