"""Fill RUN_BINDING.json (copy of the package template) after the GPU gate: status BOUND only with ADAPTER_READY present.
Per-parent receipts / frozen caches are written into per-condition bindings by export_f1.py (no shared-file races)."""
import hashlib, json, os, subprocess, sys
W = '/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101'
O = '/projects/EEG-foundation-model/yinghao/outputs_v9/r32_native_ratio_20261101'


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


b = json.load(open(f'{W}/package/templates/RUN_BINDING.template.json'))
ready = json.load(open(f'{W}/ADAPTER_READY.json'))
assert ready['status'] == 'ADAPTER_READY'
man = json.load(open(f'{O}/data/DATA_MANIFEST.json'))
commit = subprocess.check_output(['git', '-C', W, 'rev-parse', 'HEAD'], text=True).strip()
b.update(status='BOUND', server_adapter_commit=commit, server_adapter_repo=f'{W} (local git, branch nrcs/native-ratio-v1-20261101)',
         runs_root=f'{O}/runs', python_executable='/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python (read-only reuse, python -B)',
         r30_jobs_left_unchanged=True,
         r30_receipt='NRCS jobs only read Flickr data/vocab/GloVe and the IN2R 909e5329 snapshot; no R30 job, output, parent or snapshot is read, '
                     'written, held, cancelled or modified; NRCS GPU jobs nice 10 (R30 nice 0), <=2 concurrent while any r30*/r30v11*/r30a1* job exists',
         data_manifest_sha256=sha_file(f'{O}/data/DATA_MANIFEST.json'), data_manifest_path=f'{O}/data/DATA_MANIFEST.json',
         split_manifest_sha256=man['split_manifest_sha256'],
         noise_map_sha256_by_condition={f'{r}_{n}': man['roles'][r]['noise'][n]['observed_map_sha256'] for r in ('fit', 'select') for n in ('n00', 'n60')},
         realized_mismatch_by_condition={f'{r}_{n}': man['roles'][r]['noise'][n]['realized_mismatch_rate'] for r in ('fit', 'select') for n in ('n00', 'n60')},
         production_gate_receipt=f'{W}/ADAPTER_READY.json', production_gate_report=ready['gate_report'],
         environment_lock={'path': f'{W}/ENV_LOCK.txt', 'sha256': sha_file(f'{W}/ENV_LOCK.txt')},
         budget_estimate_from_smoke=ready['budget'],
         upstream_snapshot={'path': '/projects/EEG-foundation-model/yinghao/models/r25_src/IN2R', 'git_head': '909e53293ec3c633823597b8215ce23a71796d69',
                            'copied_files': {f: sha_file(f'{W}/upstream/{f}') for f in sorted(os.listdir(f'{W}/upstream')) if f.endswith(('.py', '.patch'))}},
         base_repo_check={'commit': 'cbfdbb4817d2d8fed3d1dfd8db5fef32750a8c12', 'found_in': 'handoff/r22_publication_20261101l/transport.git (public v9/r22-vcsloss-20261022 before push l)',
                          'later_public_commit': 'e0249e1fd3f94b5035cbcc063ead6d9f5e9c8ee3 (adds one-shot TEST evaluation files only; not merged into this protocol)'},
         W2_parent_receipts='per-condition: outputs bindings/BINDING_<n>_<s>.json (written by nrcs/export_f1.py)',
         frozen_cache_by_noise_seed='per-condition: outputs bindings/BINDING_<n>_<s>.json')
json.dump(b, open(f'{W}/RUN_BINDING.json', 'w'), indent=1)
print('RUN_BINDING BOUND', commit)
