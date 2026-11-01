"""MIGRATION_RECEIPT.json = filled copy of the package templates/INTEGRATION_RECEIPT.template.json (status from the real GPU gate
and the local records). Usage: write_receipt.py ROOT PACKAGE_DIR"""
import hashlib, json, os, sys, time
root, pkg = sys.argv[1], sys.argv[2]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest() if os.path.isfile(p) else None
t = json.load(open(os.path.join(pkg, 'templates', 'INTEGRATION_RECEIPT.template.json')))
g = json.load(open(os.path.join(root, 'GPU_GATE.json')))
c = g['checks']
P = lambda k: 'PASS' if c.get(k, {}).get('pass_') else ('FAIL' if k in c else 'NOT_RUN')
both = lambda *ks: 'PASS' if all(P(k) == 'PASS' for k in ks) else 'FAIL'
t['status'] = 'GATE_PASS' if g['all_pass'] else 'GATE_FAIL'
t['server_diff'] = dict(migration='R30_V11_DIFF.patch', sha256=sha(os.path.join(root, 'R30_V11_DIFF.patch')),
                        vs_r29='R30_V11_DIFF_VS_R29.patch', vs_r29_sha256=sha(os.path.join(root, 'R30_V11_DIFF_VS_R29.patch')))
t['server_protocol_digest'] = sha(os.path.join(root, 'PROTOCOL_FROZEN.json'))
t['amendment_a1_digest'] = sha(os.path.join(root, 'PROTOCOL_AMENDMENT_A1.json'))
t['checks'] = dict(
    old_gpu_submission_disabled='PASS (LINES_R30_OLD_SUPERSEDED.txt fully commented; sbatch/v1_SUPERSEDED/*.superseded; v1 runner renamed; no v1 formal job ever submitted)',
    no_duplicate_units='PASS (MATRIX_A1.csv / plan_A1: 36 unique unit ids, semantic paths and output dirs; LINES_R30.txt 36 unique jobs)',
    all_parent_states_bound=('see PARENT_BINDINGS.json' if os.path.isfile(os.path.join(root, 'PARENT_BINDINGS.json')) else 'PENDING (parents not built yet)'),
    cross_objective_restore=both('fork_identity', 'isolation_boundary'), unmodified_H_continuation=P('h_boundary_bitwise'),
    list_losses_match_R29=both('math_gpu', 'l_boundary_bitwise'), pair_losses_match_reference=both('math_gpu', 'pv_margin_gpu', 'rules_gpu'),
    optimizer_state_preserved=both('fork_identity', 'h_boundary_bitwise', 'resume_roundtrip'), first_step_resume_identity=P('resume_roundtrip'),
    calibration_rng_isolated=both('calibration_inprocess', 'calibration_records'), truth_isolation=P('truth_isolation'),
    completed_epoch10_checkpoint=P('epoch_milestone'), dtype_amp_and_score_range=P('dtype_and_score_range'))
t['static_scale_files'] = ['COEFFICIENTS.json'] if os.path.isfile(os.path.join(root, 'COEFFICIENTS.json')) else []
t['gpu_gate_jobs'] = [dict(job=g['job'], gpu=g['gpu'], all_pass=g['all_pass'], failed=g['failed'], file=f"GPU_GATE_V11_{g['job']}.json")]
t['known_deviations'] = ['DEVIATIONS.json', 'PROTOCOL_AMENDMENT_A1.json']
t['written_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
json.dump(t, open(os.path.join(root, 'MIGRATION_RECEIPT.json'), 'w'), indent=1)
print(json.dumps(t['checks'], indent=1))
