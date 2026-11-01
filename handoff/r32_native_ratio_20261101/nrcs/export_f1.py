"""Export the four-role fixed-feature NPZ caches for F1 from one NRCS W2 parent (eval mode, dropout off, frozen BN,
no batch centering; features = pointwise L2-normalized 1024-d encoder outputs h, F1 standardization is done by the
package runner on FIT only), plus sidecar meta and a per-condition RUN_BINDING (status BOUND) for tools/run_registered.py.
  export_f1.py --parent <W2 unit_id> --data <OUT>/data --ims <dir> --runs <OUT>/runs --cache <OUT>/f1_cache --bindings <dir>
"""
import argparse, json, os, time
import numpy as np
import torch
import gpo_adapter as ga
import train_gpo as tg


def main():
    ap = argparse.ArgumentParser()
    for k in ('parent', 'data', 'ims', 'runs', 'cache', 'bindings', 'binding_template'):
        ap.add_argument('--' + k.replace('_', '-'), required=True)
    a = ap.parse_args()
    tg.set_precision()
    rows = tg.matrix_rows(); row = rows[a.parent]
    assert row['stage'] == 'W2'
    pdir = f'{a.runs}/{a.parent}'
    done = json.load(open(f'{pdir}/DONE.json'))
    psha = tg.sha_file(f'{pdir}/parent.pt')
    assert psha == done['parent_checkpoint_sha256']
    noise, seed = int(row['noise_percent']), int(row['seed'])
    key = f'n{noise:02d}_s{seed}'
    out = f'{a.cache}/{key}'
    os.makedirs(out, exist_ok=True)
    pools = tg.load_pools(a.data, a.ims, noise, 'cuda')
    w2i = ga.load_vocab()
    model = ga.GPOTwoTower('cosine_hinge', w2i)
    model.load_state_dict(torch.load(f'{pdir}/parent.pt', map_location='cpu', weights_only=False)['model'])
    model.to('cuda').eval()
    man = json.load(open(f'{a.data}/DATA_MANIFEST.json'))
    paths = {}
    t0 = time.time()
    for role, pool_name in (('train', 'fit'), ('select', 'select'), ('audit', 'audit'), ('dev', 'dev')):
        pool = pools[pool_name]
        hi, ht = ga.encode_pool(model, pool, len(w2i), w2i['<mask>'])
        mk = 'observed_image_index' if role in ('train', 'select') else 'caption_image_index'
        assert pool.mapping_kind == mk
        p = f'{out}/{role}.npz'
        np.savez(p, image_features=hi.cpu().numpy().astype(np.float32), text_features=ht.cpu().numpy().astype(np.float32),
                 image_ids=pool.image_ids, caption_ids=pool.caption_ids, **{mk: pool.mapping})
        src = f'{pool_name}_n{noise:02d}' if pool_name in ('fit', 'select') else pool_name
        meta = {'role': role, 'encoder_parent_id': a.parent, 'encoder_checkpoint_sha256': psha, 'cache_sha256': tg.sha_file(p),
                'schema_version': 'NRCS-NPZ-v1', 'mapping_semantics': 'observed_for_train_select_clean_for_audit_dev',
                'split_manifest_sha256': man['split_manifest_sha256'],
                'noise_mapping_sha256': (man['roles'][pool_name]['noise'][f'n{noise:02d}']['observed_map_sha256'] if pool_name in ('fit', 'select') else None),
                'source_role_file': man['files'][src], 'export_mode': 'eval_dropout_off_frozen_BN_no_batch_centering',
                'feature_preprocessing': 'encoder_output_only_F1_standardization_fit_in_runner',
                'features': 'pointwise L2-normalized 1024-d GPO outputs h (before the sqrt(1024) head scale)',
                'shapes': {'image_features': list(hi.shape), 'text_features': list(ht.shape)},
                'pointwise_gate_receipt': '/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101/ADAPTER_READY.json'}
        json.dump(meta, open(p + '.meta.json', 'w'), indent=1)
        paths[role] = p
    b = json.load(open(a.binding_template))
    assert b['status'] == 'BOUND'
    b['W2_parent_receipts'] = {a.parent: {'checkpoint_path': f'{pdir}/parent.pt', 'checkpoint_sha256': psha, 'gpu_model': done['cost']['gpu_model']}}
    b['frozen_cache_by_noise_seed'] = {key: paths}
    b['device'] = 'cpu'
    os.makedirs(a.bindings, exist_ok=True)
    json.dump(b, open(f'{a.bindings}/BINDING_{key}.json', 'w'), indent=1)
    print('NRCS_F1_EXPORT_DONE', key, f'{time.time()-t0:.1f}s', json.dumps(paths))


if __name__ == '__main__':
    main()
