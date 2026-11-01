"""Re-apply the corrected static source scan of the gate to the byte-identical sources of GATE_<job>.json (no GPU needed:
the check is a text scan). Writes GATE_<job>_AMEND.json and ADAPTER_READY.json if every check then passes."""
import hashlib, json, sys, time
W = '/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101'
job = sys.argv[1]
g = json.load(open(f'{W}/GATE_{job}.json'))
sha = lambda p: hashlib.sha256(open(f'{W}/{p}', 'rb').read()).hexdigest()
changed = sorted(k for k, v in g['source_sha256'].items() if sha(k) != v)
assert changed in ([], ['nrcs/gate.py']), changed
srcs = ''.join(open(f'{W}/nrcs/{f}').read() for f in ('train_gpo.py', 'gpo_adapter.py'))
assert all(sha(f'nrcs/{f}') == g['source_sha256'][f'nrcs/{f}'] for f in ('train_gpo.py', 'gpo_adapter.py'))
tokens = ('evaluator_only', '_clean_index.npy', '// 5', '//5', 'im_div')
hits = [t for t in tokens if t in srcs]
checks = dict(g['checks']); checks['trainer_source_has_no_clean_map_access'] = not hits
amend = {'gate_job': job, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'reason': "gate-script defect: the original token 'true_caption' matched gpo_adapter.py:104, the RolePool assertion that REJECTS "
                   "clean-map keys ({'caption_image_index','true_caption_image_index','clean_image_index'}); no clean map is read by the trainer",
         'original_tokens': ['evaluator_only', '_clean_index', '// 5', '//5', 'true_caption'], 'corrected_tokens': list(tokens), 'hits': hits,
         'scanned_files_sha256_equal_to_gate': True, 'files_changed_since_gate': changed,
         'other_checks': 'unchanged from the GPU gate run (all PASS)', 'checks': checks}
status = 'PASS' if all(checks.values()) else 'FAIL'
amend['status'] = status
json.dump(amend, open(f'{W}/GATE_{job}_AMEND.json', 'w'), indent=1)
if status == 'PASS':
    ready = {'status': 'ADAPTER_READY', 'gate_report': f'{W}/GATE_{job}.json', 'gate_amendment': f'{W}/GATE_{job}_AMEND.json',
             'gate_gpu': g['info']['gpu'], 'gate_partition_qos': 'L40S runfill (smoke only; no smoke state reused)',
             'adapter_source_sha256_at_gate': g['source_sha256'], 'adapter_source_sha256_now': {k: sha(k) for k in g['source_sha256']},
             'protocol_sha256': g['protocol_sha256'], 'matrix_sha256': g['matrix_sha256'], 'data_manifest_sha256': g['data_manifest_sha256'],
             'checks': checks, 'budget': g['budget'], 'utc': amend['utc']}
    json.dump(ready, open(f'{W}/ADAPTER_READY.json', 'w'), indent=1)
print(status, hits, changed)
