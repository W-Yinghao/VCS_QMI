"""Read only frozen FIT diagnostic receipts; stdlib aggregation, no model/data load."""
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
PKG = BASE / 'handoff/r22_mechanism_kd_repair_20261024'
SEEDS = (31415, 16180, 57721)
TERMS = ('cosine', 'positive_quadratic', 'full_vcs', 'reference')
UPDATES = ('anchor_only', 'cosine', 'positive_quadratic', 'full_vcs')
SUBSETS = ('all', 'teacher_correct_student_wrong', 'teacher_correct_student_margin_le1')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def finite(value):
    if isinstance(value, dict):
        for x in value.values():
            finite(x)
    elif isinstance(value, list):
        for x in value:
            finite(x)
    elif isinstance(value, float):
        assert math.isfinite(value)


def contrast(a, b):
    v = [x-y for x, y in zip(a, b)]
    return dict(per_seed=v, mean=mean(v), positive_seeds=sum(x > 0 for x in v))


def main():
    freeze = read(PKG / 'FREEZE.json')
    digest = sha(PKG / 'FREEZE.json')
    assert digest == 'a18ea75d1eb36c2369830ebccbc4399239160407cd26870c77d74b145f519361'
    # These are frozen source/configuration files, not held-out data.
    for path, expected in freeze['files'].items():
        assert sha(BASE / path) == expected, path
    lock = read(PKG / 'SOURCE_LOCK.json')['kd']
    check = read(PKG / 'checks/kd.json')
    assert check['status'] == 'PASS' and check['freeze_sha256'] == digest
    jobs = read(PKG / 'JOBS_full.json')
    assert len(jobs) == len({j['name'] for j in jobs}) == 6
    units, receipts, batch_rows, inputs = {}, [], [], {}
    for job in jobs:
        path = Path(job['completion'])
        d = read(path)
        finite(d)
        assert d['status'] == 'COMPLETE' and d['freeze_sha256'] == digest
        assert d['max_batches'] == 8 and d['execution_device'] == 'cuda'
        assert d['technical'] == '' and d['test_read'] is False
        assert not d['guard'].get('forbidden_hits', [])
        assert d['guard']['allowed_opens_in_this_process'] == 6
        attempts = list(path.parent.glob('ATTEMPT_*.json'))
        assert len(attempts) == 1 and attempts[0].stem == 'ATTEMPT_' + d['job']
        # ATTEMPT is an immutable start record, not a terminal-state record.
        assert read(attempts[0])['status'] == 'STARTED'
        anchor, seed = d['anchor'], d['seed']
        assert anchor in ('ce', 'dkd') and seed in SEEDS
        receipts.append(dict(anchor=anchor, seed=seed, job=d['job'], seconds=d['seconds'],
                             path=str(path.relative_to(BASE)), sha256=sha(path),
                             attempt_sha256=sha(attempts[0]), environment=d['environment']))
        source_initial, source_teacher, source_inputs = [], [], []
        for source in ('cosine', 'vcs'):
            r = d['results'][source]
            src = lock[f'{anchor}_{source}_{seed}']
            assert r['source'] == src
            root = BASE / src['directory']
            assert sha(root / 'last.pt') == src['sha256']
            cfg, rec = read(root / 'resolved_config.json'), read(root / 'run.json')
            cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
            assert sha(root / 'resolved_config.json') == src['config_sha256']
            assert rec['config_sha256'] == cfg_sha
            assert rec['status'] == 'complete' and rec['epochs_completed'] == 240
            assert not rec['test_loaded_during_train'] and rec['teacher_frozen_verified']
            assert cfg['seed'] == seed and cfg['training']['batch_size'] == 64
            assert cfg['loss']['decision'] == dict(method=source, weight=10)
            assert r['fixed_epoch'] == 240 and r['aux_weight'] == 10
            assert math.isclose(r['optimizer_lr'], 5e-5, rel_tol=1e-12)
            assert rec['environment'] == d['environment']
            source_initial.append(rec['student_initial_state_sha256'])
            source_teacher.append(rec['teacher_initial_state_sha256'])
            rows = r['batches']
            assert len(rows) == 8 and [x['batch'] for x in rows] == list(range(8))
            hashes = [x['batch_sha256'] for x in rows]
            source_inputs.append(hashes)
            if seed in inputs:
                assert inputs[seed] == hashes  # Also paired across CE and DKD anchors.
            inputs[seed] = hashes
            per = dict(anchor=anchor, source=source, seed=seed,
                       mean_positive_cosine=mean(x['positive_cosine'] for x in rows),
                       positive_over_cosine_gradient_norm=mean(x['terms']['positive_quadratic']['gradient_norm'] / x['terms']['cosine']['gradient_norm'] for x in rows),
                       subsets={})
            for x in rows:
                assert set(x['terms']) == set(TERMS) and set(x['disposable_update']) == set(UPDATES)
                assert math.isclose(x['terms']['full_vcs']['value'], x['terms']['positive_quadratic']['value'] + x['terms']['reference']['value'], abs_tol=1e-7)
                for subset in SUBSETS:
                    n = x['terms']['cosine']['subsets'][subset]['n']
                    assert n == 64 if subset == 'all' else 0 <= n <= 64
                    for term in TERMS:
                        assert x['terms'][term]['subsets'][subset]['n'] == n
                    for update in UPDATES:
                        u = x['disposable_update'][update][subset]
                        assert u['n'] == n and 0 <= u['improved_count'] <= n
                        assert (u['mean_margin_delta'] is None) == (n == 0)
                    if n:
                        t = x['terms']
                        assert math.isclose(t['full_vcs']['subsets'][subset]['logit_margin_direction'], t['positive_quadratic']['subsets'][subset]['logit_margin_direction'] + t['reference']['subsets'][subset]['logit_margin_direction'], abs_tol=1e-7)
                    flat = dict(anchor=anchor, source=source, seed=seed, batch=x['batch'],
                                batch_sha256=x['batch_sha256'], subset=subset, n=n, positive_cosine=x['positive_cosine'])
                    for term in TERMS:
                        t = x['terms'][term]
                        for key in ('value', 'gradient_norm', 'weighted_over_anchor'):
                            flat[term+'_'+key] = t[key]
                        for key in ('logit_margin_direction', 'anchor_cosine'):
                            flat[term+'_'+key] = t['subsets'][subset][key]
                    for update in UPDATES:
                        for key in ('mean_margin_delta', 'improved_count'):
                            flat[update+'_'+key] = x['disposable_update'][update][subset][key]
                    batch_rows.append(flat)
            for subset in SUBSETS:
                n = sum(x['disposable_update']['cosine'][subset]['n'] for x in rows)
                assert n > 0
                updates = {u: sum(x['disposable_update'][u][subset]['n'] * (x['disposable_update'][u][subset]['mean_margin_delta'] or 0) for x in rows) / n for u in UPDATES}
                dirs = {t: sum(x['terms'][t]['subsets'][subset]['logit_margin_direction'] or 0 for x in rows) / n for t in TERMS}
                cosines = {t: mean(x['terms'][t]['subsets'][subset]['anchor_cosine'] for x in rows if x['terms'][t]['subsets'][subset]['n']) for t in TERMS}
                per['subsets'][subset] = dict(n=n, actual_update_mean_margin_delta=updates,
                                             logit_margin_direction_per_selected_example=dirs,
                                             mean_nonempty_batch_anchor_gradient_cosine=cosines,
                                             positive_minus_cosine=updates['positive_quadratic']-updates['cosine'],
                                             full_minus_positive=updates['full_vcs']-updates['positive_quadratic'],
                                             full_minus_anchor=updates['full_vcs']-updates['anchor_only'])
            units[f'{anchor}_{source}_{seed}'] = per
        assert source_inputs[0] == source_inputs[1]
        assert len(set(source_initial)) == len(set(source_teacher)) == 1
    assert len(units) == 12 and len(batch_rows) == 288
    for unit in units.values():
        assert 0 < unit['positive_over_cosine_gradient_norm'] < 1
        for subset in SUBSETS:
            assert unit['subsets'][subset]['positive_minus_cosine'] < 0
        for subset in SUBSETS[1:]:
            assert unit['subsets'][subset]['logit_margin_direction_per_selected_example']['reference'] < 0
            assert unit['subsets'][subset]['mean_nonempty_batch_anchor_gradient_cosine']['reference'] < 0
    summary = {}
    for anchor in ('ce', 'dkd'):
        for source in ('cosine', 'vcs'):
            us = [units[f'{anchor}_{source}_{s}'] for s in SEEDS]
            group = dict(positive_cosine=mean(x['mean_positive_cosine'] for x in us),
                         positive_over_cosine_gradient_norm=mean(x['positive_over_cosine_gradient_norm'] for x in us), subsets={})
            for subset in SUBSETS:
                v = {u: [x['subsets'][subset]['actual_update_mean_margin_delta'][u] for x in us] for u in UPDATES}
                group['subsets'][subset] = dict(n_per_seed=[x['subsets'][subset]['n'] for x in us],
                    actual_update_means={u: mean(v[u]) for u in UPDATES},
                    positive_minus_cosine=contrast(v['positive_quadratic'], v['cosine']),
                    full_minus_positive=contrast(v['full_vcs'], v['positive_quadratic']),
                    full_minus_anchor=contrast(v['full_vcs'], v['anchor_only']),
                    cosine_minus_anchor=contrast(v['cosine'], v['anchor_only']),
                    reference_logit_margin_direction=[x['subsets'][subset]['logit_margin_direction_per_selected_example']['reference'] for x in us],
                    reference_anchor_gradient_cosine=[x['subsets'][subset]['mean_nonempty_batch_anchor_gradient_cosine']['reference'] for x in us])
            summary[anchor+'_'+source] = group
    auxiliary = []
    for path in sorted((PKG / 'runs').glob('*/COMPLETE.json')):
        d = read(path)
        if d['technical']:
            assert d['status'] == 'COMPLETE' and d['freeze_sha256'] == digest and d['max_batches'] == 1
            auxiliary.append(dict(name=path.parent.name, job=d['job'], seconds=d['seconds'], device=d['execution_device'], sha256=sha(path)))
    assert sum(x['device'] == 'cpu' for x in auxiliary) == 6
    assert sum(x['device'] == 'cuda' for x in auxiliary) == 2
    seconds = sum(x['seconds'] for x in receipts)
    gates = sum(x['seconds'] for x in auxiliary if x['device'] == 'cuda')
    report = dict(status='COMPLETE_REVIEWED', utc=datetime.now(timezone.utc).isoformat(),
        series='r22_mechanism_kd_repair_20261024:other_mechanism_kd', freeze_sha256=digest,
        verification=dict(frozen_files=len(freeze['files']), jobs=6, final_checkpoints=12,
                          batches_per_checkpoint=8, examples_per_checkpoint=512,
                          inputs_identical_across_sources_and_anchors_by_seed=True,
                          source_checkpoint_and_config_hashes=True, epoch=240, lambda_=10, lr=5e-5,
                          paired_source_initialization_teacher_and_environment=True,
                          no_forbidden_guard_hits=True, no_test_read=True,
                          one_attempt_each=True, scalar_and_margin_direction_term_sums=True,
                          gradient_identity_gate=check['checks']),
        analyzer_sha256=sha(__file__), receipts=receipts, technical_receipts=auxiliary, input_batch_hashes=inputs,
        units=units, summary=summary,
        cost=dict(full6_receipt_seconds=seconds, full6_receipt_gpu_hours=seconds/3600,
                  shared_gpu_gate_seconds=gates, full_plus_gate_receipt_gpu_hours=(seconds+gates)/3600,
                  cpu_scheduler_seconds=read(PKG/'CPU_SUBMISSION.json')['scheduler_seconds'],
                  prior_failed_six=read(BASE/'handoff/r22_mechanism_review_20261024/FAILURE_KD_20261024.json'),
                  excludes='Module/CUDA import, freeze verification, scheduler setup/teardown, original training and prior failed6 duration; do not call receipt sum total scheduler cost.'),
        interpretation=[
            'All12 source-checkpoint/seed units have smaller positive-quadratic logit-gradient norm and lower actual one-step mean margin gain than cosine, in all three registered subsets.',
            'Q/reference has negative direct logit margin direction and negative mean task-gradient cosine on both difficult teacher-correct subsets in all12 units, but its actual parameter-space SGD incremental margin effects are mixed across seeds. It is not uniformly harmful.',
            'This distinguishes a reproducible local attenuation signal from a less consistent Q-update signal; it does not explain final CAL accuracy causally or replace complete matched training.',
            'Continue the already frozen six positive-only full240 controls, unchanged lambda and budgets, only after both hardware gates pass. No extra lambda/critic/negative-class grid.'
        ],
        limitations=[
            'FIT first8 epoch0 augmented minibatches only; 512 examples/seed, not independent test samples. No CAL/TEST predictions in this diagnostic.',
            'Subsets are defined with source training-mode logits; actual before/after margins are evaluated with restored source BN buffers. Results are conditional on that frozen definition.',
            'Each disposable update reloads source model and historical SGD momentum; batches do not form a training curve. Anchor-only and within-checkpoint paired differences control shared momentum, not training-path interactions.',
            'Source cosine and VCS checkpoints differ in representation and hard-subset membership; compare objectives within each checkpoint and report source strata separately.',
            'Logit directions use batch-mean auxiliary gradients without multiplying by lambda, LR, model Jacobian or momentum; they are not actual parameter-update predictions.',
            'All diagnostics requested L40S, while some historical source training used RTX. Receipts record software precision but not diagnostic GPU name/driver; no claim of exact hardware matching across source training and diagnostic.',
            'Original six pre-batch initialization failures remain preserved; expired scheduler cost unknown, not zero.'
        ])
    stem = HERE / 'ANALYSIS_KD_MECHANISM_ALL6_20261024'
    stem.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (HERE/'KD_MECHANISM_ALL6_BATCHES_20261024.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(batch_rows[0])); writer.writeheader(); writer.writerows(batch_rows)
    print(json.dumps(dict(verification=report['verification'], summary=summary, cost={k:v for k,v in report['cost'].items() if k!='prior_failed_six'}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
