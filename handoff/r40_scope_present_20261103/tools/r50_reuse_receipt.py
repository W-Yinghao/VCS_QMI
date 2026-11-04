"""R50 baseline reuse receipt (package SERVER_PLAN section 1/3: R47 G N-JOINT and R48 N-frozen weights reused, only the DEV side re-read). Collects, per
reference unit (F60 / CC / F80 x frozen / G-joint x J / BCE / NCE = 18): the training-side identity evidence (FIT cache files = R47 G files, FIT mean
recomputed bitwise, prep candidates + a0/b0 identical, source checkpoint lineage == R50 prep, config identical except cond / tree), the replay result
(REPLAY_R50.json: chain + state at the replay step vs the reference checkpoint), and the PRIMARY re-read (r50_rereads/<unit>/REREAD.json) next to the
SECONDARY_TRANSDUCTIVE original readout. Writes ROOT/REUSE_RECEIPT_R50.json. Usage: python3 -I -B tools/r50_reuse_receipt.py"""
import hashlib, json, time
from pathlib import Path
W = Path(__file__).resolve().parents[1]
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103'); R50OUT = Path('/home/infres/yinwang/CS_QMI/outputs_v9_r50')
rj = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else None
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
REFS = [f'R48_{pk}_N_frozen_lfG_{o}' for pk in ('F60_R30_H1001', 'CC31_H1001', 'F80_IN2R_H1001') for o in ('J', 'BCE', 'NCE')] + \
       [f'R47_G_{pk}_N_joint_{o}' for pk in ('F60_R30_H1001', 'CC31_H1001', 'F80_IN2R_H1001') for o in ('J', 'BCE', 'NCE')]
rep = {}
for j, v in (rj(W / 'REPLAY_R50.json') or {}).items():
    for u, x in v['units'].items():
        if u.startswith('R50_REPLAY_'):
            rep.setdefault(u.replace('R50_REPLAY_', ''), []).append(dict(job=j, **x))
units = {}
for u in REFS:
    base = rj(W / 'configs' / f'{u}.json')['cond'][:-4]
    nm = rj(OUT / 'r50_parents/cache' / base / f'NORMALIZATION_{base}.json'); pm = rj(OUT / 'r50_parents/prep' / base / 'PREP_MANIFEST.json')
    rr = rj(R50OUT / 'r50_rereads' / u / 'REREAD.json')
    fit = dict(fit_mean_recomputed_equal_bitwise=all(x['fit_mean_recomputed_equal_bitwise'] for x in nm['per_peer_modality'].values()),
               fit_cache_recomputed_max_abs=max(x['fit_cache_vs_r47G']['max_abs'] for x in nm['per_peer_modality'].values()),
               fit_cache_files_hardlinked_or_copied_from_r47G=nm['fit_side_identical_to_r47G'] is not None, prep_identical_to_r47G=pm['identical_to_r47G'],
               weighting={k: x['weighting'] for k, x in nm['per_peer_modality'].items()})
    rec = dict(reference=u, base=base, fit_side=fit, replay=rep.get(u), reread=None)
    if rr:
        st = max(int(k) for k in rr['per_step'])
        rec['reread'] = dict(identity=rr['identity'], step0_native_check=rr['step0_native_check'], H_primary_rsum=rr['H_primary']['rsum'], H_secondary_rsum=rr['H_secondary_transductive']['rsum'],
                             step8000_primary=rr['per_step'][str(st)]['primary'], step8000_secondary=rr['per_step'][str(st)]['secondary_transductive'], source_ckpt_sha256=rr['source_ckpt_sha256'],
                             gpu=rr['env']['gpu'])
    ok_fit = fit['fit_mean_recomputed_equal_bitwise'] and fit['fit_cache_recomputed_max_abs'] == 0.0 and fit['prep_identical_to_r47G']
    ok_id = bool(rr) and all(rr['identity'][k] for k in ('config_equal_except_cond_tree', 'a0b0_equal', 'candidate_table_sha256_equal', 'r50_prep_identical_to_r47G', 'r50_cache_fit_files_are_r47G'))
    # replay rule: at least one replay of the reference config on the R50 cond reproduces the reference chain AND state bitwise (same GPU type; the JOINT
    # references additionally needed their original node: R47 G F60 / F80 joint ran on node50 (driver 580.178.04); on node52 both the R50 replay and the
    # unchanged R47 code on the R47 cond (control 1086281) give one identical state that differs from the node50 checkpoint -> environment, recorded below)
    reps = rec['replay'] or []
    ok_rep = bool(reps) and any(x.get('chain_equal') and x.get('state_bitwise') for x in reps)
    rec['replay_rule'] = 'any replay with chain_equal and state_bitwise'
    rec['reuse_valid'] = bool(ok_fit and ok_id and ok_rep); rec['checks'] = dict(fit_side=ok_fit, lineage_and_config=ok_id, replay=ok_rep)
    units[u] = rec
ctl = {}
for j, v in (rj(W / 'REPLAY_R50.json') or {}).items():
    for u, x in v['units'].items():
        if u.startswith('R50_REPLAYCTL_'):
            ctl[u] = dict(job=j, **x)
out_ctl = dict(r47_code_control=ctl, note='R50_REPLAYCTL_* = the R47 reference config run by the unchanged R47 tree on its R47 cond (gate mode); compared with the R50 replay of the same config '
               'on the same node in REPLAY_R50_CONTROL_COMPARE.json')
out = dict(controls=out_ctl, line='R50 reuse receipt: R48 N-frozen / R47 G N-JOINT weights reused (no retraining), DEV re-read under FIT_FIXED_CENTER_V1', protocol_id='FIT_FIXED_CENTER_V1',
           utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), units=units, all_reuse_valid=all(v['reuse_valid'] for v in units.values()),
           rule='reuse iff FIT cache / FIT mean / candidates / calibration identical to R47 G, source lineage and config identical (cond / tree aside), and the replay of the '
                'reference config on the R50 cond reproduces the reference chain and state; otherwise a matched control is needed (registered, not silently compared)')
(W / 'REUSE_RECEIPT_R50.json').write_text(json.dumps(out, indent=1, default=str))
print('R50 reuse all valid', out['all_reuse_valid'], {u: v['checks'] for u, v in units.items() if not v['reuse_valid']})
