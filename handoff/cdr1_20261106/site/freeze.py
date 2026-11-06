#!/usr/bin/env python3
import json, sys, time, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import H, PKG, OUT, BANKS, sha, dump, load, M_SIZES, PIS, TAUS, LAMBDAS, RHO_GRID, EPSILONS, NULL_FAR
assert not (H / 'PROTOCOL_FROZEN_CDR1.json').exists(), 'already frozen'
site = ['site/common.py', 'site/readout_lib.py', 'site/run_group.py', 'site/run_transfer.py', 'site/export_bank.py',
        'sbatch/group.sbatch', 'sbatch/transfer.sbatch']
pkgf = sorted(str(p.relative_to(PKG)) for d in ('src/cdr1', 'tools', 'configs') for p in (PKG / d).iterdir() if p.is_file())
banks = {ds: {'bank_sha256': load(BANKS / ds / 'bank_meta.json')['bank_sha256'], 'meta_sha256': sha(BANKS / ds / 'bank_meta.json'),
              'n_image_groups': load(BANKS / ds / 'bank_meta.json')['n_images_groups'], 'n_texts': load(BANKS / ds / 'bank_meta.json')['n_texts'],
              'n_merged_duplicate_images': load(BANKS / ds / 'bank_meta.json')['n_merged_images'],
              'content_manifest_sha256': load(BANKS / ds / 'bank_meta.json')['content_manifest_sha256']} for ds in ('Flickr30K', 'COCO')}
fz = {'protocol_id': 'CDR1_CONTAMINATION_CALIBRATION_SELECTIVE_RETRIEVAL_V1',
 'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
 'user_quote': '查看，准备，并提交这个', 'user_quote_efficiency': '效率为先',
 'evidence_base': 'a55332645aed5b23b7d90f92e662e9242652efd4',
 'package_zip_sha256': '8103e594c65d55d44cb2fa34d3aa80606e559109c31813ca1b9b0f24041d35db', 'package_manifest_verified': '65/65',
 'frozen_before_any_real_fit': True,
 'encoder': {'model': 'google/siglip2-base-patch16-224', 'revision': '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2',
             'model_lock_full_sha256': 'd4ff4992a1662ae237d1c86291ba8a7dd124892da8942b15a5329217b269c7dd',
             'features': 'existing NR1 (Flickr) / NR2C (COCO) / NR1-CC152K_RECOVERED global caches, per-item L2, no centering; no NR/CR checkpoints, no G/LoRA/teacher; no GPU extraction needed'},
 'banks': banks,
 'S0': {'job': 1091819, 'pytest': '52 passed', 'demo/cli_smoke/registry_smoke/stress': 'ran; shipped results reproduced to max rel diff 2.0e-7; regenerated configs byte-identical'},
 'bridge': {'job': 1091833, 'duplicate_rule': 'raw image sha256 identical -> one group; Flickr 1 merge, COCO 13 merges (feature diff 0.0)'},
 'synthetic_interface_test': {'job': 1091832, 'note': 'synthetic 6000-image banks through every site stage incl. transfer/CC; interface only, not evidence'},
 'matrix': {'cases': 24, 'core_unique_fits': 84, 'prediction_views': 192, 'data_seeds': [1001, 2002, 3003],
            'gallery_M': list(M_SIZES), 'presence_pi': list(PIS), 'tau_fixed': list(TAUS), 'lambda_pair': list(LAMBDAS),
            'n_scores_per_role': 20000, 'D2_direction': 't2i, at most one annotated partner',
            'D2_actions': 'fixed tau on p_best_correct (base Top1) + M+1 MAP; also naive Top1 pair mechanism probability ablation'},
 'unknown_rho': {'cases': 12, 'epsilons': list(EPSILONS), 'anchor_quantiles': [.01, .05, .10], 'delta': .05,
                 'profile_grid': list(RHO_GRID), 'profile_new_fits_max': 96, 'profile_role': 'finite-grid sensitivity; never selects rho; D2 lower-envelope rule reported as sensitivity',
                 'anchor_point_fits': 'eps=0 only, only when interval non-empty and point exists; J+BCE; <=24 fits; reported beside interval, labelled ANCHOR_CONDITIONAL',
                 'DEDPUL': 'PENDING (SITE_ADAPTER_REQUIRED); not integrated this round, no comparison claimed'},
 'oracle_refs': {'CLEAN_TOP1_PLATT': 'THRESH_CAL clean Top1-correctness labels, pi-weighted per pi, per M; target-domain and source-domain-transfer',
                 'MATCHED_NULL_MAX': 'THRESH_CAL absent-query Top1 scores; target FAR 0.05/0.10 (cost paired with tau .95/.90)'},
 'transfer': {'pairs': 'Flickr->COCO, COCO->Flickr (package run_readouts), Flickr/COCO->CC152K_RECOVERED_20261104 DEV galleries',
              'frozen': 'parameters, standardization, beta(source), thresholds unchanged', 'wrong_pi': 'predictor pi=0.5 evaluated at target pi 0.25/0.75',
              'CC': 'DEV 693 one-to-one pairs, hash galleries M=32/128/512 (M512 absent=181); no CC fit; no clean oracle; CC local rho intervals on observed FIT = model-conditional sensitivity only'},
 'not_run': {'D3_OT': 'out of scope', 'D4': 'NA (read-only; no saved required quantities)', 'MA_PaPSP/NACP': 'literature only', 'encoder_training': 0, 'TEST': 'not read'},
 'truth_isolation': 'evaluator_only/ chmod 000 during fit/predict/rho_fit/rho_predict stages; package whitelist loaders; separate processes per stage',
 'execution': {'check_group': 'Flickr30K d1001 full group (contains the Flickr60 matched group) first; then remaining 5 groups + 3 transfers by a retrying CPU driver (no --dependency)',
               'partitions': 'CPU,cpu-high normal QOS', 'interface_failure_rule': 'fix interface only; record amendment; matrix unchanged'},
 'site_code_sha256': {f: sha(H / f) for f in site},
 'reporting_code_sha256_at_freeze': {'site/summarize_cdr1.py': sha(H / 'site/summarize_cdr1.py')},
 'package_code_sha256': {f: sha(PKG / f) for f in pkgf}}
dump(H / 'PROTOCOL_FROZEN_CDR1.json', fz)
(H / 'PROTOCOL_FROZEN_CDR1.json').chmod(0o444)
print(sha(H / 'PROTOCOL_FROZEN_CDR1.json'))
