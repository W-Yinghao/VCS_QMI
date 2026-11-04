"""R50 NORMALIZATION_MANIFEST.json (package SERVER_PLAN section 9): per parent / peer / modality the FIT-mean source file + sha256, weighting, recomputation
result, full-scale check A, DEV-side numbers; one entry per R50 build (cache/<base>/NORMALIZATION_<base>.json, sha256 recorded). Usage: python3 -I -B tools/r50_norm_manifest.py"""
import hashlib, json, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]; OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ent = {}
for f in sorted((OUT / 'r50_parents' / 'cache').glob('*/NORMALIZATION_*.json')):
    x = json.loads(f.read_text()); cm = f.parent / 'CACHE_MANIFEST.json'
    ent[x['base_cond']] = dict(file=str(f), sha256=sh(f), cache_manifest_sha256=sh(cm), h_only=x['h_only'], fit_side_identical_to_r47G=x['fit_side_identical_to_r47G'],
                               parent=x['parent'], per_peer_modality={k: {kk: v[kk] for kk in ('fit_mean_source', 'fit_mean_sha256', 'weighting', 'accumulation', 'n_fit_rows',
                               'fit_mean_recomputed_equal_bitwise', 'fit_mean_recomputed_max_abs', 'dev_mean_minus_fit_mean_l2') if kk in v} for k, v in x['per_peer_modality'].items()},
                               check_A_passed=x['check_A_full_scale']['passed'], env=x['env'])
out = dict(protocol_id='FIT_FIXED_CENTER_V1', rule='one raw-encoder-output FIT mean per parent peer / modality, shape [1, d], fp64 accumulation -> fp32; applied to FIT and DEV; no DEV statistic',
           utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), parents=ent)
(W / 'NORMALIZATION_MANIFEST.json').write_text(json.dumps(out, indent=1, default=str)); print('NORMALIZATION_MANIFEST', list(ent))
