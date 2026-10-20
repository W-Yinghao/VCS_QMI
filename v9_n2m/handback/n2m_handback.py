#!/usr/bin/env python3
"""V9 N2M results hand-back (V9_N2M_SERVER_HANDOFF_20261019_CN.md section 6 and 7): build the report from the
server records and check it.

Read-only. No job submission, no training, no TEST read, no model loading. Every number in the report is computed
here from the N2M output records; every quote is copied verbatim from the cited file and line. Before writing, the
record chains are verified (experiment receipt, artifact hashes, attempt ledgers, control and diagnostic bindings,
package manifest, git state) and the headline values are recomputed from the raw training logs; any mismatch refuses
(exit 2) and nothing is written.

    python3 n2m_handback.py             verify, then (re)write the report, the file list and the hand-off copy
    python3 n2m_handback.py --check     verify, regenerate in memory, compare byte-for-byte with the written files
    python3 n2m_handback.py --binaries  also re-hash the 18 N2M checkpoints and the 3 N2F reference checkpoints
                                        against their recorded artifact SHA (about 4.6 GB read; server only)

Outputs: v9_n2m/V9_N2M_RESULTS_HANDBACK_20261020_CN.md, v9_n2m/handback/V9_N2M_HANDBACK_FILES_20261020.json and a
copy of the report in /home/infres/yinwang/CS_QMI/handoff/v9_n2m_20261019/. They depend only on the records and on
this file, so a re-run with unchanged records is byte-identical.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
from pathlib import Path

CS = Path('/home/infres/yinwang/CS_QMI')
OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1')
REPO = Path(__file__).resolve().parents[2]
N2M = REPO / 'v9_n2m'
PKG = N2M / 'VCS_V9_N2M_Server_Package_v1_20261019'
HANDOFF_DOC = N2M / 'V9_N2M_SERVER_HANDOFF_20261019_CN.md'
HANDOFF_DOC_ROOT = CS / 'V9_N2M_SERVER_HANDOFF_20261019_CN.md'
ZIP = CS / 'VCS_V9_N2M_Server_Package_v1_20261019.zip'
N2F_CLOSEOUT = CS / 'handoff/v9_overlap_v10_20261018/CLOSEOUT.json'
N2F_CLOSEOUT_SHA = '0f5ba97134826d14e2bad05d047f48afeb2f4684d316c6da32319af29072c978'
N2F_RESULTS = CS / 'handoff/v9_full_20261017/results'
ADDENDUM = N2M / 'n2f_closeout'
SLURM_LOGS = CS / 'slurm_logs'
REPORT = N2M / 'V9_N2M_RESULTS_HANDBACK_20261020_CN.md'
FILES_JSON = N2M / 'handback' / 'V9_N2M_HANDBACK_FILES_20261020.json'
COPY_DIR = CS / 'handoff/v9_n2m_20261019'
GENERATOR_REL = 'v9_n2m/handback/n2m_handback.py'
COMMITS = {'package': '8fe6f8a92f8685a2f4fee8f0b8acfb74f373089f',
           'server': '4ea8b69ea2fe4ac1cb1cf31dcd3e298d4b12bfab',
           'n2f_addendum': 'c13b249d59e02e52cd08c717aad8fd9750802225'}
SEEDS = (901, 902, 903)
ARMS = ('VCS', 'ConstantU', 'ShuffleVCS')
CONTROLS = ('ConstantU', 'ShuffleVCS')
ARM_CN = {'VCS': 'VCS', 'ConstantU': 'Constant-U', 'ShuffleVCS': 'Shuffle-VCS'}
EPOCHS, STEPS = 40, 1133
ENV_IGNORED = ('host', 'job_id', 'observed_utc')
GPU_JOBS_VERIFY = '1048422'
GPU_SHORT = {'NVIDIA L40S': 'L40S', 'NVIDIA H100 NVL': 'H100 NVL', 'NVIDIA RTX PRO 6000 Blackwell Server Edition': 'RTX PRO 6000'}


class Refusal(Exception):
    pass


def require(condition, message):
    if not condition:
        raise Refusal(message)


_SHA = {}


def sha(path):
    path = Path(path)
    if path not in _SHA:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1 << 20), b''):
                h.update(block)
        _SHA[path] = h.hexdigest()
    return _SHA[path]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def git(*args):
    p = subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, text=True)
    return p.returncode, p.stdout.strip()


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def f2(x):
    return f'{x:.2f}'


def s2(x):
    return f'{x:+.2f}'


def f3(x):
    return f'{x:.3f}'


def f4(x):
    return f'{x:.4f}'


def _line_of(text, index):
    return text.count('\n', 0, index) + 1


def quote(path, start, end=None):
    """Verbatim span from `start` to the end of `end` (same line), or the sentence/line containing `start`."""
    text = Path(path).read_text(encoding='utf-8')
    require(text.count(start) == 1, f'Quote anchor not unique in {path}: {start[:40]}')
    i = text.index(start)
    line_end = text.find('\n', i)
    line_end = len(text) if line_end < 0 else line_end
    if end is None:
        j = text.find('。', i)
        j = line_end if j < 0 or j > line_end else j + 1
    else:
        j = text.find(end, i)
        require(0 <= j and j + len(end) <= line_end, f'Quote end not on the same line in {path}: {end[:40]}')
        j += len(end)
    return {'text': text[i:j], 'path': str(path), 'line': _line_of(text, i)}


def json_quote(path, value, field):
    return {'text': value, 'path': str(path), 'field': field}


def blockquote(q):
    where = f"`{q['path']}` 字段 `{q['field']}`" if 'field' in q else f"`{q['path']}:{q['line']}`"
    return ['', f"> {q['text']}", '>', f'> —— {where}', '']


def md_table(header, rows, align=None):
    align = align or ['---'] * len(header)
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(align) + '|']
    return out + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]


# ------------------------------------------------------------------------------------------------ verification

def verify_package():
    manifest = read_json(PKG / 'PACKAGE_MANIFEST.json')
    files = manifest['files_sha256']
    bad = [rel for rel, digest in files.items() if sha(PKG / rel) != digest]
    require(not bad, f'Package files differ from PACKAGE_MANIFEST.json: {bad[:5]}')
    present = {str(p.relative_to(PKG)) for p in PKG.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    require(present == set(files) | {'PACKAGE_MANIFEST.json'}, 'Package directory has extra or missing files')
    zip_record = (N2M / 'PACKAGE_ZIP.sha256').read_text().split()[0]
    require(sha(ZIP) == zip_record, 'Package zip differs from PACKAGE_ZIP.sha256')
    require(sha(HANDOFF_DOC) == sha(HANDOFF_DOC_ROOT) == files['01_SERVER_AGENT_PROMPT_CN.md'], 'Hand-off document copies differ')
    git_state = {}
    for key, rel in (('package', PKG), ('server', N2M / 'server'), ('n2f_addendum', ADDENDUM)):
        commit = COMMITS[key]
        rc, _ = git('merge-base', '--is-ancestor', commit, 'HEAD')
        require(rc == 0, f'Commit {commit} is not in the history of HEAD')
        rc, _ = git('diff', '--quiet', commit, 'HEAD', '--', str(rel.relative_to(REPO)))
        require(rc == 0, f'{rel} changed after {commit}')
        rc, out = git('status', '--porcelain', '--', str(rel.relative_to(REPO)))
        require(rc == 0 and out == '', f'Uncommitted changes under {rel}')
        rc, last = git('log', '-1', '--format=%H', '--', str(rel.relative_to(REPO)))
        require(rc == 0 and last == commit, f'Last commit touching {rel} is {last}, not {commit}')
        git_state[key] = commit
    rc, branch = git('rev-parse', '--abbrev-ref', 'HEAD')
    require(rc == 0, 'git unavailable')
    return manifest, zip_record, git_state


def verify_experiment():
    exp_sha = sha(OUT / 'experiment.json')
    receipt = (OUT / 'EXPERIMENT.sha256').read_text(encoding='ascii').split()
    require(receipt == [exp_sha, 'experiment.json'], 'EXPERIMENT.sha256 receipt does not match experiment.json')
    exp = read_json(OUT / 'experiment.json')
    protocol_sha = sha(PKG / 'config/protocol.json')
    require(exp['protocol_sha256'] == protocol_sha and exp['test_read'] is False, 'Experiment protocol/data role')
    require(exp['server_verification_sha256'] == sha(OUT / 'SERVER_VERIFICATION.json'), 'Server verification hash')
    sv = read_json(OUT / 'SERVER_VERIFICATION.json')
    require(sv['status'] == 'PASS' and sv['verification_report_sha256'] == sha(OUT / 'TORCH_VERIFICATION_cuda.json'), 'CUDA report')
    torch_report = read_json(OUT / 'TORCH_VERIFICATION_cuda.json')
    require(torch_report['status'] == 'PASS' and torch_report['test_read'] is False, 'CUDA verification status')
    require(sv['runtime_receipt_sha256'] == exp['runtime_receipt_sha256'] == sha(OUT / 'runtime/RUNTIME.json'), 'Runtime receipt')
    require(sha(OUT / 'runtime/src/v9/n2/train.py') == torch_report['patched_train_sha256'], 'Patched trainer differs from the verified one')
    require(exp['N2F_closeout_sha256'] == sha(OUT / 'N2F_CLOSEOUT.json'), 'N2F census hash')
    for seed in SEEDS:
        require(exp['control_manifests'][str(seed)]['sha256'] == sha(OUT / f'controls/s{seed}/manifest.json'), 'Control manifest hash')
    plan = read_csv(OUT / 'UNIT_PLAN.csv')
    require(len(plan) == 9 and len(exp['units']) == 9, 'Unit plan size')
    exp_by = {u['unit_id']: u for u in exp['units']}
    require(sorted(exp_by) == sorted(r['unit_id'] for r in plan), 'UNIT_PLAN.csv unit set differs from experiment.json')
    for row in plan:
        unit = exp_by[row['unit_id']]
        require(row['unit_id'] == unit['unit_id'] and int(row['seed']) == unit['seed'] and row['arm'] == unit['arm']
                and row['action'] == unit['action'] and int(row['order']) == unit['order'] and row['output_dir'] == unit['output_dir'],
                f"UNIT_PLAN.csv disagrees with experiment.json: {row['unit_id']}")
    require({(u['seed'], u['arm']) for u in exp['units']} == {(s, a) for s in SEEDS for a in ARMS}, 'Unit matrix')
    return exp, exp_sha, protocol_sha, sv, torch_report, plan


def verify_units(exp, exp_sha, protocol_sha, summary, plan, binaries):
    target = exp['target_environment']
    by_unit = {u['unit_id']: u for u in summary['units']}
    plan_by = {r['unit_id']: r for r in plan}
    units, ckpt_checked = {}, 0
    for unit in exp['units']:
        uid, seed, arm = unit['unit_id'], unit['seed'], unit['arm']
        d = Path(unit['output_dir'])
        require(d == OUT / 'runs' / uid, f'Unexpected unit directory: {d}')
        fr = read_json(d / 'full_result.json')
        ident = read_json(d / 'identity.json')
        require(fr['status'] == 'COMPLETE' and fr['identity'] == ident and ident['unit_id'] == uid and ident['seed'] == seed
                and ident['arm'] == arm and ident['experiment_sha256'] == exp_sha and ident['protocol_sha256'] == protocol_sha
                and ident['test_read'] is False and ident['reuse'] is False, f'Identity: {uid}')
        require(ident['weights_sha256'] == plan_by[uid]['weights_sha256'], f'Weights identity: {uid}')
        for name in ('train_log.json', 'run_manifest.json', 'identity.json'):
            require(sha(d / name) == fr['artifact_sha256'][name], f'Artifact hash: {uid}/{name}')
        if binaries:
            for name in ('best.pt', 'last.pt'):
                require(sha(d / name) == fr['artifact_sha256'][name], f'Checkpoint hash: {uid}/{name}')
                ckpt_checked += 1
        require(read_json(d / 'done.json') == fr['done'] and fr['done']['manifest_sha256'] == fr['artifact_sha256']['run_manifest.json'],
                f'done.json: {uid}')
        log = read_json(d / 'train_log.json')
        epochs = log['epochs']
        require([e['epoch'] for e in epochs] == list(range(EPOCHS)) and all(e['steps_in_epoch'] == STEPS for e in epochs), f'Epochs: {uid}')
        require(sum(e['steps_in_epoch'] for e in epochs) == fr['done']['global_step'] == EPOCHS * STEPS, f'Updates: {uid}')
        rs = [float(e['dev']['rsum']) for e in epochs]
        be = max(range(EPOCHS), key=lambda i: rs[i])
        require(fr['done']['best'] == {'rsum': rs[be], 'epoch': be}, f'Best selection: {uid}')
        dev = epochs[be]['dev']
        recalls = [dev[k] for k in ('i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10')]
        require(close(sum(recalls), rs[be]) and close(dev['mR'], rs[be] / 6), f'Recall/rSum consistency: {uid}')
        hard = [(rs[e], e) for e in range(5, EPOCHS)]
        worst = min(hard)
        last5 = sum(rs[35:40]) / 5
        s = by_unit[uid]
        require(s['status'] == 'COMPLETE_VERIFIED' and s['complete_for_primary'] is True and s['errors'] == [] and s['missing'] == []
                and s['provenance'] == 'new_N2M_run' and s['action'] == unit['action'], f'Summary status: {uid}')
        require(s['primary_rsum'] == rs[be] and s['best_epoch'] == be and s['primary_mR'] == dev['mR']
                and all(s[k] == dev[k] for k in ('i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10')), f'Summary best: {uid}')
        require(s['epoch4_rsum'] == rs[4] and s['epoch5_rsum'] == rs[5] and s['minimum_hard_rsum'] == worst[0]
                and s['last_rsum'] == rs[39] and close(s['last5_mean_rsum'], last5), f'Summary trajectory values: {uid}')
        require(s['actual_updates_from_epochs'] == EPOCHS * STEPS and s['epochs_observed'] == EPOCHS and s['resumed_at'] == [], f'Summary updates: {uid}')
        require(all(v == 'VERIFIED' for v in s['artifact_verification'].values()) and len(s['artifact_verification']) == 5, f'Summary artifacts: {uid}')
        require(s['hardware'] == fr['hardware'], f'Summary hardware: {uid}')
        require(all(fr['hardware'][k] == v for k, v in target.items() if k not in ENV_IGNORED)
                and set(fr['hardware']) == set(target), f'Environment differs from the frozen target: {uid}')
        coll = s['collapse']
        require(coll['status'] == 'NO' and coll['severe_collapse'] is False and coll['full_trajectory'] is True, f'Collapse: {uid}')
        require(all(rs[e] >= 0.5 * rs[4] for e in range(5, EPOCHS)), f'A hard epoch fell below half of epoch 4: {uid}')
        prog = [json.loads(x) for x in (d / 'progress.jsonl').read_text().splitlines() if x.strip()]
        require([p['epoch'] for p in prog] == list(range(EPOCHS)) and all(p['dev_rsum'] == rs[i] for i, p in enumerate(prog)), f'progress.jsonl: {uid}')
        # attempts ledger
        entries = fr['attempts_summary']['entries']
        files = sorted((d / 'attempts').glob('*.json'))
        require(len(entries) == len(files) == 1 and fr['attempts_summary']['total_duration_complete'] is True, f'Attempts: {uid}')
        att = read_json(files[0])
        require(str(files[0]) == entries[0]['file'] and sha(files[0]) == entries[0]['sha256'] and att['status'] == 'SUCCEEDED'
                and att['completed_epochs'] == EPOCHS and att['completed_updates'] == EPOCHS * STEPS and att['job_id'] == fr['hardware']['job_id']
                and att['elapsed_s'] == entries[0]['elapsed_s'] == fr['elapsed_this_attempt_s'], f'Attempt record: {uid}')
        require(s['cost']['known_attempt_seconds'] == att['elapsed_s'] and s['cost']['duration_complete'] is True
                and s['cost']['unknown_or_unclosed_attempts'] == 0, f'Summary cost: {uid}')
        # gradient window, recomputed from grad_norms.jsonl (last record per update wins)
        recs = {}
        for line in (d / 'grad_norms.jsonl').read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                recs[(r['epoch'], r['step_in_epoch'])] = r
        gw = s['gradient_window']
        clip = sum(bool(r['clipped']) for r in recs.values()) / len(recs)
        gmean = sum(r['grad_norm_pre_clip'] for r in recs.values()) / len(recs)
        gmax = max(r['grad_norm_pre_clip'] for r in recs.values())
        require(gw['available'] is True and gw['n_unique_samples'] == len(recs) and close(gw['clip_fraction'], clip)
                and close(gw['mean_grad_norm_pre_clip'], gmean) and close(gw['max_grad_norm_pre_clip'], gmax), f'Gradient window: {uid}')
        windows = sorted({k[0] for k in recs})
        units[uid] = {'unit_id': uid, 'seed': seed, 'arm': arm, 'dir': d, 'fr': fr, 'log': log, 'rs': rs, 'best_epoch': be,
                      'recalls': recalls, 'mR': dev['mR'], 'worst': worst, 'last5': last5, 'attempt': att, 'grad': gw,
                      'grad_epochs': windows, 'summary': s, 'epoch0_loss': epochs[0]['mean_loss']}
    for seed in SEEDS:
        jobs = {units[f's{seed}__{a}']['fr']['hardware']['job_id'] for a in ARMS}
        hosts = {units[f's{seed}__{a}']['fr']['hardware']['host'] for a in ARMS}
        require(len(jobs) == 1 and len(hosts) == 1, f'Seed {seed} arms did not share one job/GPU')
        ordered = sorted((int(plan_by[f's{seed}__{a}']['order']), units[f's{seed}__{a}']['attempt']) for a in ARMS)
        require([o for o, _ in ordered] == [0, 1, 2] and all(x['end_utc'] <= y['start_utc'] for (_, x), (_, y) in zip(ordered, ordered[1:])),
                f'Seed {seed} arms did not run sequentially in the frozen order')
    return units, ckpt_checked


def verify_contrasts(exp, summary, units):
    rule = exp['analysis_parameters']['investment_rule']
    protocol = read_json(PKG / 'config/protocol.json')
    require(rule == summary['analysis_parameters']['investment_rule'] == protocol['analysis']['investment_rule']
            == summary['investment_rule']['frozen_parameters'], 'Investment-rule parameters drifted')
    paired = read_csv(OUT / 'analysis/full/paired_contrasts.csv')
    out = []
    for control, c in zip(CONTROLS, summary['contrasts']):
        require(c['contrast'] == f'VCS-{control}', 'Contrast order')
        diffs = []
        for seed, p in zip(SEEDS, c['pairs']):
            a, b = units[f's{seed}__VCS']['rs'][units[f's{seed}__VCS']['best_epoch']], units[f's{seed}__{control}']['rs'][units[f's{seed}__{control}']['best_epoch']]
            require(p['seed'] == seed and p['eligible'] is True and p['VCS_rsum'] == a and p['control_rsum'] == b and p['difference'] == a - b,
                    f'Pair {control} {seed}')
            row = [r for r in paired if r['contrast'] == c['contrast'] and int(r['seed']) == seed][0]
            require(float(row['difference']) == a - b and row['eligible'] == 'True', f'paired_contrasts.csv {control} {seed}')
            diffs.append(a - b)
        mean = sum(diffs) / 3
        vcs_sev = sum(units[f's{s}__VCS']['summary']['collapse']['severe_collapse'] for s in SEEDS)
        ctl_sev = sum(units[f's{s}__{control}']['summary']['collapse']['severe_collapse'] for s in SEEDS)
        passes = mean >= rule['mean_rsum_difference_each_at_least'] and sum(x > 0 for x in diffs) >= rule['positive_seeds_each_at_least'] \
            and len(diffs) == rule['complete_pairs_required'] and (vcs_sev <= ctl_sev or not rule['no_more_severe_collapse'])
        require(c['mean_difference'] == mean and c['minimum_difference'] == min(diffs) and c['maximum_difference'] == max(diffs)
                and c['n_positive'] == sum(x > 0 for x in diffs) and c['n_zero'] == sum(x == 0 for x in diffs)
                and c['investment_rule_pass'] == passes and c['p_value'] is None and c['confidence_interval'] is None
                and c['complete_three_seed_contrast'] is True and c['n_eligible_pairs'] == 3, f'Summary contrast {control}')
        out.append({'contrast': c['contrast'], 'control': control, 'diffs': diffs, 'mean': mean, 'min': min(diffs), 'max': max(diffs),
                    'positive': sum(x > 0 for x in diffs), 'zero': sum(x == 0 for x in diffs), 'pass': passes,
                    'vcs_severe': vcs_sev, 'control_severe': ctl_sev})
    combined = all(c['pass'] for c in out)
    require(summary['investment_rule']['pass'] == combined and summary['investment_rule']['all_conditions_known'] is True, 'Combined rule')
    rows = read_csv(OUT / 'analysis/full/unit_results.csv')
    require(len(rows) == 9 and all(float(r['primary_rsum']) == units[r['unit_id']]['rs'][units[r['unit_id']]['best_epoch']]
                                   and r['status'] == 'COMPLETE_VERIFIED' for r in rows), 'unit_results.csv')
    traj = read_csv(OUT / 'analysis/full/epoch_trajectories.csv')
    require(len(traj) == 9 * EPOCHS and all(float(r['rsum']) == units[r['unit_id']]['rs'][int(r['epoch'])] for r in traj), 'epoch_trajectories.csv')
    return out, combined, rule


def verify_offline(exp):
    frozen_path = CS / 'handoff/v9_full_20261017/FROZEN.json'
    require(sha(frozen_path) == exp['base_freeze_sha256'], 'N2F FROZEN.json differs from the experiment base freeze')
    frozen_inputs = read_json(frozen_path)['input_sha256']
    res = {}
    for seed in SEEDS:
        m = read_json(OUT / f'controls/s{seed}/manifest.json')
        for key in ('vcs_csv', 'vcs_lock', 'observed'):
            src = m['sources'][key]
            require(frozen_inputs.get(src['path']) == src['sha256'] == sha(Path(src['path'])), f'Control source differs from the N2F frozen input: s{seed} {key}')
        require(m['arms']['VCS']['sha256'] == m['sources']['vcs_csv']['sha256'], f'VCS arm is not the frozen production CSV: s{seed}')
        require(sha(Path(m['permutation']['file'])) == m['permutation']['sha256'], f'Permutation hash s{seed}')
        for arm in ARMS:
            a = m['arms'][arm]
            require(sha(Path(a['csv'])) == a['sha256'] and sha(Path(a['lock'])) == a['lock_sha256'], f'Control file hash s{seed} {arm}')
        require(all(v is True for k, v in m['invariants'].items() if k not in ('roundoff_relative_tolerance', 'roundoff_absolute_tolerance'))
                and m['label_fields_read'] is False and m['random_permutation_selected_using_labels'] is False, f'Control invariants s{seed}')
        o = read_json(OUT / f'analysis/offline/s{seed}/offline_metrics.json')
        require(o['source']['controls_manifest_sha256'] == sha(OUT / f'controls/s{seed}/manifest.json')
                and o['training_label_access'] is False and o['test_read'] is False, f'Offline identity s{seed}')
        for arm in ARMS:
            require(o['source']['arm_hashes'][arm] == {'csv_sha256': m['arms'][arm]['sha256'], 'lock_sha256': m['arms'][arm]['lock_sha256']},
                    f'Offline arm hashes s{seed} {arm}')
            require(o['arms'][arm]['total_mass'] == m['arms'][arm]['total_mass_U'] or close(o['arms'][arm]['total_mass'], m['arms'][arm]['total_mass_U'], 1e-12),
                    f'Offline mass s{seed} {arm}')
        rows = read_csv(OUT / f'analysis/offline/s{seed}/offline_metrics.csv')
        for r in rows:
            src = o['arms'][r['arm']] if r['arm'] in ARMS else o['same_T_oracle_rho']['metrics']
            for k in ('mean_weight', 'ess', 'brier', 'logloss', 'auroc', 'auprc_average_precision', 'residual_noise_mass_fraction'):
                require(float(r[k]) == src[k], f'offline_metrics.csv s{seed} {r["arm"]} {k}')
        require(o['production_reconstruction']['consistent'] is True and o['production_reconstruction']['n_outside_tolerance'] == 0, f'Reconstruction s{seed}')
        f = read_json(OUT / f'analysis/fixed_loss/s{seed}/fixed_loss_metrics.json')
        meta_path = OUT / f'diagnostics/s{seed}/fixed_losses.csv.meta.json'
        meta = read_json(meta_path)
        require(f['loss_csv_sha256'] == sha(OUT / f'diagnostics/s{seed}/fixed_losses.csv') == meta['loss_csv_sha256']
                and f['metadata_sha256'] == sha(meta_path) and f['source_metadata'] == meta
                and f['controls_manifest_sha256'] == sha(OUT / f'controls/s{seed}/manifest.json')
                and meta['experiment_sha256'] == sha(OUT / 'experiment.json') and f['scientific_training_complete'] is False
                and f['test_read'] is False and meta['evaluator_labels_read'] is False and meta['same_negative_pool'] is True, f'Fixed-loss identity s{seed}')
        frows = read_csv(OUT / f'analysis/fixed_loss/s{seed}/fixed_loss_metrics.csv')
        require(len(frows) == len(f['batches']) and all(float(r['normalized_mean_objective']) == b['normalized_mean_objective'] and r['arm'] == b['arm']
                                                       for r, b in zip(frows, f['batches'])), f'fixed_loss_metrics.csv s{seed}')
        res[seed] = {'manifest': m, 'offline': o, 'fixed': f, 'meta': meta}
    return res


def verify_n2f(offline, binaries):
    raw = N2F_CLOSEOUT.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == N2F_CLOSEOUT_SHA, 'N2F closeout changed')
    co = json.loads(raw)
    require(co['status'] == 'PASS', 'N2F closeout status')
    n2f = {(u['seed'], u['method'], u['scale']): u for u in co['units']}
    census = read_json(OUT / 'N2F_CLOSEOUT.json')
    require(census['complete'] == census['expected'] == 33 and census['checkpoints_checked'] is True, 'N2M copy of the N2F census')
    for u in census['units']:
        c = n2f[(u['seed'], u['method'], u['scale'])]
        require(u['status'] == 'COMPLETE' and u['best_dev_rsum'] == c['best_dev_rsum'] and u['best_epoch'] == c['best_epoch'], 'N2F census vs closeout')
    refs, checked = {}, 0
    for seed in SEEDS:
        meta = offline[seed]['meta']
        c = n2f[(seed, 'VCS_Weight', 'sum')]
        p = N2F_RESULTS / c['group'] / 'VCS_Weight_sum'
        require(sha(p / 'full_result.json') == c['full_result_sha256'], f'N2F full_result s{seed}')
        fr = read_json(p / 'full_result.json')
        require(Path(meta['checkpoint']) == p / 'best.pt' and meta['checkpoint_sha256'] == fr['artifact_sha256']['best.pt']
                and meta['checkpoint_epoch'] == c['best_epoch'], f'Fixed-loss reference checkpoint s{seed}')
        if binaries:
            require(sha(p / 'best.pt') == fr['artifact_sha256']['best.pt'], f'N2F reference checkpoint hash s{seed}')
            checked += 1
        refs[seed] = {'path': p / 'best.pt', 'sha256': fr['artifact_sha256']['best.pt'], 'size': (p / 'best.pt').stat().st_size,
                      'group': c['group'], 'epoch': c['best_epoch']}
    for f in ('N2F_FINAL_33_UNITS.csv', 'N2F_FINAL_33_UNITS.json', 'V9_CLOSEOUT_ADDENDUM_CN.md'):
        require((ADDENDUM / f).exists(), f'Missing addendum output {f}')
    return co, n2f, refs, checked


def parse_sacct(text):
    rows = {}
    for line in text.splitlines():
        if line.strip():
            job, state, elapsed, tres, code, start, end = line.split('|')
            rows[job] = {'state': state, 'elapsed': int(elapsed), 'tres': tres, 'exit': code, 'start': start, 'end': end}
    return rows


def runtime_seconds(text):
    m = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d+):(\d+)', text)
    require(m is not None, f'Unexpected RunTime {text}')
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + s


def verify_cost(units):
    acc = read_json(OUT / 'analysis/cost/allocation_accounting.json')
    sacct_text = (OUT / 'analysis/cost/sacct_from_scontrol.txt').read_text()
    require(sacct_text == (OUT / 'analysis/cost/sacct_raw.txt').read_text(), 'sacct_raw.txt is not the reconstructed text')
    sacct = parse_sacct(sacct_text)
    scontrol = {}
    for p in sorted((OUT / 'slurm_records').glob('scontrol_*.txt')):
        fields = dict(re.findall(r'(?:^|\s)([A-Za-z]+)=(\S*)', p.read_text()))
        scontrol[fields['JobId']] = fields
    for job, row in sacct.items():
        f = scontrol[job]
        require(f['JobState'] == row['state'] and runtime_seconds(f['RunTime']) == row['elapsed'] and f['ExitCode'] == row['exit']
                and f['StartTime'] == row['start'] and f['EndTime'] == row['end'], f'Reconstructed accounting line differs from scontrol: {job}')
    total = 0.0
    for j in acc['jobs']:
        row = sacct[j['job_id']]
        gpus = int(re.search(r'gres/gpu=(\d+)', row['tres']).group(1)) if 'gres/gpu=' in row['tres'] else 0
        require(j['elapsed_s'] == row['elapsed'] and j['gpu_count'] == gpus and j['gpu_allocation_hours'] == row['elapsed'] * gpus / 3600
                and j['state'] == row['state'] and j['terminal'] is True, f'Accounting job {j["job_id"]}')
        total += j['gpu_allocation_hours']
    require(total == acc['gpu_allocation_hours_known'] and acc['complete_terminal_accounting'] is False, 'Accounting total')
    require(acc['missing_jobs'] == [GPU_JOBS_VERIFY] and GPU_JOBS_VERIFY not in sacct and not (OUT / f'slurm_records/scontrol_{GPU_JOBS_VERIFY}.txt').exists(),
            'Missing-job record')
    receipts = {(r['path'], r['sha256'], r['job_id']) for r in acc['attempt_receipts']}
    require(receipts == {(str(sorted((u['dir'] / 'attempts').glob('*.json'))[0]), sha(sorted((u['dir'] / 'attempts').glob('*.json'))[0]),
                          u['attempt']['job_id']) for u in units.values()}, 'Accounting attempt receipts')
    return acc, sacct, scontrol


# ------------------------------------------------------------------------------------------------ file list

def file_entries(units, refs, offline_res):
    groups = []

    def add(group, path, status='server', recorded=None, note=''):
        path = Path(path)
        require(path.exists(), f'Missing deliverable {path}')
        digest = recorded if recorded is not None else sha(path)
        entries = groups[-1][1] if groups and groups[-1][0] == group else None
        if entries is None:
            groups.append((group, []))
            entries = groups[-1][1]
        entries.append({'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size, 'status': status, 'note': note})

    g = '1 新包实际代码commit与package SHA'
    add(g, ZIP, 'server', note=f"zip；sha256 与 v9_n2m/PACKAGE_ZIP.sha256 一致；包原样提交于 {COMMITS['package']}")
    add(g, N2M / 'PACKAGE_ZIP.sha256', 'repo')
    add(g, PKG / 'PACKAGE_MANIFEST.json', 'repo', note='包内全部文件与此清单逐一核对一致')
    add(g, PKG / 'config/protocol.json', 'repo', note='= experiment.json protocol_sha256')
    add(g, HANDOFF_DOC, 'repo', note='= 包内 01_SERVER_AGENT_PROMPT_CN.md')
    for p in sorted((N2M / 'server').iterdir()):
        if p.is_file():
            add(g, p, 'repo', note=f"服务器侧脚本，最后提交 {COMMITS['server']}")
    g = '2 原N2F最终33项表'
    add(g, N2F_CLOSEOUT, 'server', note='N2F收尾（CPU作业1046165，PASS）')
    add(g, OUT / 'N2F_CLOSEOUT.json', 'server', note='prepare时由包重建的33/33 census（含checkpoint核对）')
    add(g, OUT / 'inspection/N2F_CENSUS.json', 'server', note='prepare前的census（32/33）')
    for name in ('N2F_FINAL_33_UNITS.csv', 'N2F_FINAL_33_UNITS.json', 'V9_CLOSEOUT_ADDENDUM_CN.md'):
        add(g, ADDENDUM / name, 'repo', note=f"N2F收尾附录（含分段硬件），提交 {COMMITS['n2f_addendum']}")
    g = '3 experiment.json / receipt / UNIT_PLAN.csv'
    for name in ('experiment.json', 'EXPERIMENT.sha256', 'UNIT_PLAN.csv', 'runtime/RUNTIME.json', 'runtime/HOST_LOCK.yaml',
                 'runtime/src/v9/n2/train.py', 'inspection/INSPECTION.json'):
        add(g, OUT / name, 'server')
    g = '4 CUDA实际报告（及CPU核对）'
    for name in ('SERVER_VERIFICATION.json', 'TORCH_VERIFICATION_cuda.json', 'package_checks/unittest_output.txt',
                 'package_checks/local_validation/LOCAL_VALIDATION.json', 'package_checks/local_validation/unittest_output.txt'):
        add(g, OUT / name, 'server')
    g = '5 controls manifest与置换SHA'
    for seed in SEEDS:
        m = offline_res[seed]['manifest']
        add(g, OUT / f'controls/s{seed}/manifest.json', 'server')
        add(g, Path(m['permutation']['file']), 'server', note='= manifest permutation.sha256')
        for arm in ARMS:
            add(g, Path(m['arms'][arm]['csv']), 'server', note='逐pair权重，留server')
            add(g, Path(m['arms'][arm]['lock']), 'server')
    g = '6 每臂identity / run_manifest / full_result'
    for u in units.values():
        for name in ('identity.json', 'run_manifest.json', 'full_result.json', 'done.json'):
            add(g, u['dir'] / name, 'server')
    g = '7 完整train_log与progress'
    for u in units.values():
        for name in ('train_log.json', 'progress.jsonl'):
            add(g, u['dir'] / name, 'server')
    g = '8 grad窗口'
    for u in units.values():
        add(g, u['dir'] / 'grad_norms.jsonl', 'server', note='epoch4末200与epoch5初200次更新')
    g = '9 全部attempt与sacct记录'
    for u in units.values():
        for p in sorted((u['dir'] / 'attempts').glob('*.json')):
            add(g, p, 'server')
    for name in ('SOURCE_NOTE.md', 'sacct_from_scontrol.txt', 'sacct_raw.txt', 'allocation_accounting.json', 'allocation_accounting.csv'):
        add(g, OUT / 'analysis/cost' / name, 'server', note='sacct不可用；由scontrol快照重建' if name.startswith('sacct') else '')
    for p in sorted((OUT / 'slurm_records').iterdir()):
        add(g, p, 'server')
    add(g, OUT / 'submissions/SUBMISSIONS.jsonl', 'server')
    for p in sorted(SLURM_LOGS.iterdir()):
        if re.match(r'^(n2m_|v9n2m_).*\.out$', p.name):
            add(g, p, 'server', note='作业stdout')
    g = '10 三个离线后验表'
    for seed in SEEDS:
        for name in ('offline_metrics.json', 'offline_metrics.csv', 'calibration_10bins.csv', 'OFFLINE_READOUT_CN.md'):
            add(g, OUT / f'analysis/offline/s{seed}' / name, 'server')
    g = '11 三个固定风险诊断'
    for seed in SEEDS:
        for name in ('fixed_loss_metrics.json', 'fixed_loss_metrics.csv'):
            add(g, OUT / f'analysis/fixed_loss/s{seed}' / name, 'server')
        add(g, OUT / f'diagnostics/s{seed}/fixed_losses.csv', 'server', note='逐anchor ell，留server')
        add(g, OUT / f'diagnostics/s{seed}/fixed_losses.csv.meta.json', 'server')
        r = refs[seed]
        add(g, r['path'], 'server_binary', recorded=r['sha256'],
            note=f"N2F {r['group']} VCS SUM best（epoch {r['epoch']}）；模型二进制留server，checkpoints_not_materialized；sha256为N2F artifact SHA")
    g = '12 最终summary与两个配对差表'
    for name in ('summary.json', 'paired_contrasts.csv', 'unit_results.csv', 'epoch_trajectories.csv', 'RESULTS_CN.md'):
        add(g, OUT / 'analysis/full' / name, 'server',
            note='两项配对差表：VCS-ConstantU 与 VCS-ShuffleVCS 各3行' if name == 'paired_contrasts.csv' else
                 ('含两项配对差、工程规则与逐单元记录' if name == 'summary.json' else ''))
    g = '13 模型二进制（N2M每臂best/last）'
    for u in units.values():
        for name in ('best.pt', 'last.pt'):
            add(g, u['dir'] / name, 'server_binary', recorded=u['fr']['artifact_sha256'][name],
                note='留server，checkpoints_not_materialized；sha256为full_result.json artifact SHA（summary.json记为VERIFIED）')
    return groups


# ------------------------------------------------------------------------------------------------ report

def build(binaries=False):
    manifest, zip_record, git_state = verify_package()
    exp, exp_sha, protocol_sha, sv, torch_report, plan = verify_experiment()
    summary = read_json(OUT / 'analysis/full/summary.json')
    require(summary['experiment_sha256'] == exp_sha and summary['protocol_sha256'] == protocol_sha and summary['test_read'] is False
            and summary['analysis_identity']['experiment_receipt_sha256'] == sha(OUT / 'EXPERIMENT.sha256')
            and all(summary['analysis_identity'][k] is True for k in ('experiment_verified', 'protocol_verified', 'analysis_parameters_verified', 'package_code_verified'))
            and summary['n_expected_units'] == 9 and summary['n_complete_verified_units'] == 9 and summary['counts'] == {'COMPLETE_VERIFIED': 9},
            'Summary identity')
    units, ckpt_checked = verify_units(exp, exp_sha, protocol_sha, summary, plan, binaries)
    contrasts, rule_pass, rule = verify_contrasts(exp, summary, units)
    offline = verify_offline(exp)
    co, n2f, refs, ref_checked = verify_n2f(offline, binaries)
    acc, sacct, scontrol = verify_cost(units)
    groups = file_entries(units, refs, offline)
    gen_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    files_payload = {'schema': 'cs_qmi.v9_n2m_handback_files.v1', 'generator': {'path': GENERATOR_REL, 'sha256': gen_sha},
                     'experiment_sha256': exp_sha, 'commits': COMMITS,
                     'checkpoints': 'checkpoints_not_materialized: model binaries stay on the server; listed sha256 = recorded artifact SHA',
                     'groups': [{'group': g, 'files': e} for g, e in groups]}
    files_text = json.dumps(files_payload, ensure_ascii=False, indent=1) + '\n'
    md = markdown(exp, exp_sha, protocol_sha, sv, torch_report, summary, units, contrasts, rule_pass, rule, offline, co, n2f, refs,
                  acc, sacct, scontrol, groups, gen_sha, zip_record, manifest, hashlib.sha256(files_text.encode()).hexdigest())
    return {'report': md, 'files': files_text}, {'checkpoints_hashed': ckpt_checked + ref_checked}


def markdown(exp, exp_sha, protocol_sha, sv, torch_report, summary, units, contrasts, rule_pass, rule, offline, co, n2f, refs,
             acc, sacct, scontrol, groups, gen_sha, zip_record, manifest, files_sha):
    L = []
    add = L.append
    H = HANDOFF_DOC
    P = PKG / 'PROTOCOL_CN.md'
    U = lambda s, a: units[f's{s}__{a}']
    best = lambda u: u['rs'][u['best_epoch']]
    c_by = {c['control']: c for c in contrasts}

    add('# V9 N2M 结果交回（2026-10-20）')
    add('')
    add(f"按 `{H}`（与 `{HANDOFF_DOC_ROOT}` 逐字节相同）§6、§7 交回。本文件由 `{GENERATOR_REL}`（sha256 `{gen_sha}`）只读server记录生成，所有数字由该脚本从下列文件计算，"
        '引文逐字取自所注文件与行号；生成与核对不提交作业、不训练、不读TEST、不加载模型。'
        f"冻结实验 `{OUT / 'experiment.json'}` sha256 `{exp_sha}`（与 receipt `EXPERIMENT.sha256` 一致），协议 `config/protocol.json` sha256 `{protocol_sha}`。")
    add('')
    cc, cs = c_by['ConstantU'], c_by['ShuffleVCS']
    add('## 0 摘要')
    add('')
    add(f"- 完整性：{summary['n_complete_verified_units']}/9 个单元 COMPLETE_VERIFIED（3 seed × VCS/Constant-U/Shuffle-VCS，均为本轮新跑，每个 40 epoch、{EPOCHS * STEPS} 次更新，每个 seed 的三臂同一作业同一 GPU）。")
    add('- best full DEV rSum（VCS / Constant-U / Shuffle-VCS）：' + '；'.join(
        f"s{s} {f2(best(U(s, 'VCS')))} / {f2(best(U(s, 'ConstantU')))} / {f2(best(U(s, 'ShuffleVCS')))}" for s in SEEDS) + '。')
    add(f"- VCS−Constant-U 均差 {s2(cc['mean'])}（{', '.join(s2(x) for x in cc['diffs'])}，{cc['positive']}/3 为正）；VCS−Shuffle-VCS 均差 {s2(cs['mean'])}"
        f"（{', '.join(s2(x) for x in cs['diffs'])}，{cs['positive']}/3 为正）；严重塌陷 0/9。")
    add(f"- 工程投入规则：{rule_pass}。这只是资源决定（继续内容关联可靠度扩展），不是显著性、等价或确认；n=3、DEV 复用，仅描述性。")
    auc = {a: [offline[s]['offline']['arms'][a]['auroc'] for s in SEEDS] for a in ARMS}
    require(cc['positive'] == cs['positive'] == 3 and min(auc['VCS']) > max(auc['ConstantU'] + auc['ShuffleVCS']), 'Summary mechanism bullet no longer matches')
    add(f"- 机制：保持总质量的 Constant-U、保持总质量与权重分布/ESS 的 Shuffle-VCS 在 3/3 个 seed 都低于 VCS（差 {f2(cc['min'])}–{f2(cc['max'])} 与 {f2(cs['min'])}–{f2(cs['max'])}）；"
        f"后验评分中只有 VCS 的分配区分注入错配（AUROC {f3(min(auc['VCS']))}–{f3(max(auc['VCS']))}，两种控制 ≤ {f3(max(auc['ConstantU'] + auc['ShuffleVCS']))}），固定风险方向一致。"
        '错误分配是否比恒定权重更差、再平衡本身相对不加权的作用、相对其他估计器的独有优势、显著性与泛化均未被支持或未检验（§3.5）。')
    add(f"- 成本：已知 GPU allocation {acc['gpu_allocation_hours_known']:.2f} GPU-h（scontrol 快照重建，sacct 不可用）；CUDA 核对作业 {', '.join(acc['missing_jobs'])} 的 allocation 未知，不计为 0。")
    add('')
    add('## 1 主结果：三seed三臂 best full DEV rSum 与两项配对差')
    add('')
    L += blockquote(quote(H, '先展示三seed三臂best full DEV rSum和两项配对差', '不替换best主终点。'))
    add('')
    rows = []
    for s in SEEDS:
        rows.append([s] + [f"{f2(best(U(s, a)))} (e{U(s, a)['best_epoch']})" for a in ARMS] + [s2(c_by[c]['diffs'][SEEDS.index(s)]) for c in CONTROLS])
    L += md_table(['seed', 'VCS', 'Constant-U', 'Shuffle-VCS', 'VCS−Constant', 'VCS−Shuffle'], rows,
                  ['---:', '---:', '---:', '---:', '---:', '---:'])
    add('')
    add('括号内为 best epoch；差值为同 seed 的 VCS 减控制。')
    add('')
    rows = [[c['contrast'], f"{len(c['diffs'])}/3", s2(c['mean']), s2(c['min']), s2(c['max']), f"{c['positive']}/3", c['zero'],
             'null（未计算）'] for c in contrasts]
    L += md_table(['比较', '完整配对', '三seed均差', '最小差', '最大差', '正方向seed', '零差', 'p值 / 置信区间'], rows,
                  ['---', '---:', '---:', '---:', '---:', '---:', '---:', '---'])
    add('')
    add(f"三对均完整（{summary['n_complete_verified_units']}/9 个单元 COMPLETE_VERIFIED），因此报告主均差。summary.json 中两项比较的 `p_value` 与 `confidence_interval` 均为 null；"
        'n=3 只作描述性配对比较，本文件不给 pair-IID 置信区间、显著性或等价结论，也不把它当作独立确认（DEV 已在本路线中多次使用）。')
    add('')
    v1, v2 = U(901, 'VCS'), U(902, 'VCS')
    grid = [round(best(u) * 5070 / 100) for u in (v1, v2)]
    require(best(v1) == best(v2) and grid[0] == grid[1] and all(close(g * 100 / 5070, best(u)) for g, u in zip(grid, (v1, v2))),
            'The VCS 901/902 tie statement no longer matches the records')
    require(v1['best_epoch'] != v2['best_epoch'] and v1['recalls'] != v2['recalls'] and v1['epoch0_loss'] != v2['epoch0_loss']
            and v1['fr']['identity']['weights_sha256'] != v2['fr']['identity']['weights_sha256'], 'VCS 901/902 runs are not distinct')
    rec = lambda u: '/'.join(f2(x) for x in u['recalls'])
    add(f"**VCS s901 与 s902 的 best rSum 相同（都是 {best(v1)!r}），这是巧合。** DEV 有 1014 张图像、5070 条 caption："
        'i2t 每项 Recall 为 100·k/1014 = 100·5k/5070，t2i 为 100·m/5070，所以 rSum 只能取 100/5070 的整数倍；两个值都等于 '
        f"{grid[0]}×100/5070。两次运行在其他所有字段上都不同：seed、噪声映射与权重文件不同（权重 sha256 `{v1['fr']['identity']['weights_sha256'][:12]}…` 与 "
        f"`{v2['fr']['identity']['weights_sha256'][:12]}…`），epoch 0 平均损失 {v1['epoch0_loss']:.4f} 与 {v2['epoch0_loss']:.4f}，best epoch {v1['best_epoch']} 与 {v2['best_epoch']}，"
        f"六项 Recall（i2t R1/R5/R10、t2i R1/R5/R10）{rec(v1)} 与 {rec(v2)}。")
    add('')
    add('同 checkpoint 的六项 Recall 与 mR（best epoch）：')
    add('')
    rows = [[f"s{u['seed']}", ARM_CN[u['arm']], u['best_epoch']] + [f2(x) for x in u['recalls']] + [f2(u['mR'])] for u in units.values()]
    L += md_table(['seed', '臂', 'best epoch', 'i2t R1', 'i2t R5', 'i2t R10', 't2i R1', 't2i R5', 't2i R10', 'mR'], rows,
                  ['---', '---', '---:'] + ['---:'] * 7)
    add('')

    # 2 completeness
    add('## 2 完整性')
    add('')
    dec = exp['reference_decisions']
    diff_keys = sorted({k for s in SEEDS for k in dec[str(s)]['environment_differences']})
    require(all(dec[str(s)]['VCS_action'] == 'RUN' for s in SEEDS), 'Reference decisions changed')
    gpu_diff = dec['901']['environment_differences']['gpu']
    for s_ in SEEDS:
        ed = dec[str(s_)]['environment_differences']
        require(sorted(ed) == diff_keys and ed['gpu']['reason'] == 'different_or_missing' and ed['gpu']['actual'] == gpu_diff['actual']
                and all(v['reason'] == 'historical_or_target_value_not_recorded' and v['expected'] is None for k, v in ed.items() if k != 'gpu'),
                'Reference-decision reasons changed')
    exp_gpus = '、'.join(sorted({dec[str(s_)]['environment_differences']['gpu']['expected'] for s_ in SEEDS}))
    add(f"- 矩阵：{len(exp['units'])} 个单元 = 3 seed × 3 臂，`requested_mode={exp['requested_mode']}`，新完整训练 {exp['new_full_runs']} 次。"
        f"三个 VCS 参照都判为新跑（`VCS_action=RUN`）：历史 N2F VCS 的环境与本轮目标不一致或未记录（GPU 期望 {exp_gpus}、实际 {gpu_diff['actual']}；"
        f"另有 {len(diff_keys) - 1} 项历史环境字段未记录：{', '.join(k for k in diff_keys if k != 'gpu')}），因此全部为同环境的新运行，没有历史复用。")
    add(f"- 状态：{'；'.join(f'{k} {v}/9' for k, v in summary['counts'].items())}；每个单元 40 个完整 epoch、每 epoch {STEPS} 次更新、共 {EPOCHS * STEPS} 次（与 checkpoint 的 global_step 一致），"
        '恰好一次 attempt（SUCCEEDED），没有续跑，`errors` 与 `missing` 均为空。')
    add('- 产物：summary.json 对每个单元记录 train_log.json、run_manifest.json、identity.json、best.pt、last.pt 均为 VERIFIED。本脚本重新核对了 JSON 产物的 sha256、'
        'identity 与 experiment/协议/权重身份、best 的选择（最大值、并列取更早）、六项 Recall 与 rSum/mR 的一致性、progress.jsonl 与训练日志逐 epoch 一致，'
        '以及 summary.json、unit_results.csv、paired_contrasts.csv、epoch_trajectories.csv 与训练日志重算值一致。模型二进制见 §6（checkpoints_not_materialized）。')
    hw_rows = []
    for s in SEEDS:
        h = U(s, 'VCS')['fr']['hardware']
        hw_rows.append(f"s{s}：job {h['job_id']}，{h['host']}")
    tgt = exp['target_environment']
    add(f"- 配对条件：每个 seed 的三臂在同一个作业、同一张 GPU 上按冻结顺序串行（{'；'.join(hw_rows)}）。9 个单元的环境记录除 host/job/时间外与冻结目标环境逐字段相同："
        f"{tgt['gpu']}、driver {tgt['driver']}、Torch {tgt['torch']}、CUDA {tgt['cuda']}、Python {tgt['python']}、NumPy {tgt['numpy']}、cuDNN {tgt['cudnn_version']}、"
        f"deterministic_algorithms={str(tgt['deterministic_algorithms']).lower()}、TF32 关闭、cuDNN benchmark 关闭。")
    add(f"- CUDA 核对：`SERVER_VERIFICATION.json` status={sv['status']}（job {sv['environment']['job_id']}，{sv['environment']['host']}），"
        f"`TORCH_VERIFICATION_cuda.json` status={torch_report['status']}，scope：{torch_report['scope']}。")
    require(all(offline[s]['offline']['training_label_access'] is False and offline[s]['offline']['evaluator_label_access'] is True
                and offline[s]['fixed']['labels_used_only_by_clean_oracle'] is True for s in SEEDS), 'Label-access records changed')
    add('- 生产权重未重拟合：三个 seed 的 VCS 臂 CSV、其 lock 与 observed 表都与 N2F `FROZEN.json` 登记的冻结输入 sha256 相同（VCS 臂逐字节复制原 CSV）。')
    add('- 数据角色：experiment、summary 与 9 个 identity 的 test_read 均为 false；三个 controls manifest 的 `label_fields_read` 与 '
        '`random_permutation_selected_using_labels` 均为 false；离线评分的 `training_label_access` 为 false、`evaluator_label_access` 为 true，'
        '即机制标签只在 EVAL_ONLY 评分中读取；固定风险诊断的 `labels_used_only_by_clean_oracle` 为 true。')
    L += blockquote(quote(H, '固定pair机制标签不进入critic、校准选择或控制生成', '本轮最多9次完整训练，不增加MEAN/LR/rho/anchor/COCO/REG/Unweighted新网格。'))
    add('')

    # 3 diagnostics
    add('## 3 机制诊断与可支持的结论')
    add('')
    add('### 3.1 总质量、权重分布与 ESS（控制的构造）')
    add('')
    rows = []
    for s in SEEDS:
        m = offline[s]['manifest']
        o = offline[s]['offline']['arms']
        rows.append([f"s{s}", f"{m['mu_float64']:.6f}", f2(m['arms']['VCS']['total_mass_U']), f"{m['mean_v_full_pool']:.6f}",
                     f"{o['VCS']['anchor_fraction_of_retained_full_mass']:.4f}", f"{m['arms']['VCS']['ess_U']:.0f} / {m['arms']['ConstantU']['ess_U']:.0f} / {m['arms']['ShuffleVCS']['ess_U']:.0f}",
                     m['permutation']['n_fixed_points'], f"{m['permutation']['weight_pearson_r']:.4f}"])
        for a in ARMS:
            require(close(m['arms'][a]['total_mass_U'], m['arms']['VCS']['total_mass_U'], 1e-12) and close(o[a]['anchor_fraction_of_retained_full_mass'],
                    o['VCS']['anchor_fraction_of_retained_full_mass'], 1e-12), 'Mass not shared across arms')
    L += md_table(['seed', 'μ = mean(w_U)', 'U总质量（三臂相同）', '全池分母 v̄', 'A占保留质量', 'ESS_U：VCS / Constant / Shuffle', 'Shuffle固定点', '置换前后权重 r'], rows,
                  ['---', '---:', '---:', '---:', '---:', '---:', '---:', '---:'])
    add('')
    add('三臂的 U 总质量、全池分母与 A 的质量占比在 float64 舍入内相同（manifest 不变量全部为 true）；Shuffle-VCS 的权重多重集、总质量与 ESS 与 VCS 相同，只改变权重与图像内容的对应。')
    add('')
    add('### 3.2 后验评分（EVAL_ONLY 机制标签）')
    add('')
    L += blockquote(quote(OUT / 'analysis/offline/s901/OFFLINE_READOUT_CN.md', '标签含义：clean = 未被注入机制实际错配'))
    add('')
    rows = []
    for s in SEEDS:
        o = offline[s]['offline']
        for a in ARMS:
            x = o['arms'][a]
            rows.append([f"s{s}", ARM_CN[a], f4(x['brier']), f4(x['logloss']), f4(x['auroc']), f4(x['auprc_average_precision']),
                         f4(x['calibration_ece_10_equal_width']), f4(x['clean_mean_retention']), f4(x['mismatched_mean_retention']),
                         f4(x['residual_noise_mass_fraction'])])
    L += md_table(['seed', '臂', 'Brier', 'logloss', 'AUROC (clean=1)', 'AP', 'ECE (10箱)', 'clean平均保留', 'mismatch平均保留', '残余噪声质量占比'], rows,
                  ['---', '---'] + ['---:'] * 8)
    add('')
    rho = '；'.join(f"s{s} {f4(offline[s]['offline']['rho']['raw'])}/{f4(offline[s]['offline']['rho']['realized'])}" for s in SEEDS)
    orc = '；'.join(f"s{s} Brier {f4(offline[s]['offline']['same_T_oracle_rho']['metrics']['brier'])}、残余噪声 {f4(offline[s]['offline']['same_T_oracle_rho']['metrics']['residual_noise_mass_fraction'])}"
                   for s in SEEDS)
    add(f"估计 rho（原始）/实际机制比例：{rho}。同一 T 换成实际比例的 oracle-rho 诊断（不可部署）：{orc}。"
        f"生产权重由锁定 CSV 重建，三个 seed 的 `production_reconstruction.consistent` 均为 true。")
    add('')
    add('### 3.3 固定风险（同一学生、同一 batch、同一负候选）')
    add('')
    rows, counts = [], {}
    for s in SEEDS:
        f = offline[s]['fixed']
        by = {}
        for b in f['batches']:
            by.setdefault(b['batch_id'], {})[b['arm']] = b['normalized_mean_objective']
        arms = ('VCS', 'ConstantU', 'ShuffleVCS', 'CleanOracle_EVAL_ONLY')
        means = {a: sum(v[a] for v in by.values()) / len(by) for a in arms}
        cnt = {'VCS<Constant': sum(v['VCS'] < v['ConstantU'] for v in by.values()), 'VCS<Shuffle': sum(v['VCS'] < v['ShuffleVCS'] for v in by.values()),
               'Clean<VCS': sum(v['CleanOracle_EVAL_ONLY'] < v['VCS'] for v in by.values()), 'Shuffle<Constant': sum(v['ShuffleVCS'] < v['ConstantU'] for v in by.values())}
        counts[s] = (cnt, len(by))
        r = refs[s]
        rows.append([f"s{s}", len(by)] + [f"{means[a]:.4f}" for a in arms] + [f"{cnt[k]}/{len(by)}" for k in cnt] + [f"{r['group']} e{r['epoch']}"])
    L += md_table(['seed', 'batch数', 'VCS', 'Constant-U', 'Shuffle-VCS', 'Clean oracle', 'VCS<Constant', 'VCS<Shuffle', 'Clean<VCS', 'Shuffle<Constant', '固定学生'], rows,
                  ['---', '---:'] + ['---:'] * 8 + ['---'])
    add('')
    add('表中为 `sum(v·ell)/v̄` 除以 batch 大小的均值（越低表示加权后的 hinge 目标越小）。固定学生是原 N2F VCS SUM 的 best checkpoint，其 sha256 与 N2F 结果的 artifact SHA 一致（已核对）。')
    L += blockquote(json_quote(OUT / 'analysis/fixed_loss/s901/fixed_loss_metrics.json', offline[901]['fixed']['interpretation'], 'interpretation'))
    add('')
    L += blockquote(json_quote(OUT / 'analysis/fixed_loss/s901/fixed_loss_metrics.json', offline[901]['fixed']['warnings'][0], 'warnings[0]'))
    add('')
    add('### 3.4 轨迹、梯度窗口与严重塌陷')
    add('')
    L += blockquote(quote(P, '同时报告六Recall、mR、best epoch、epoch4/5', '末值不能替换已经固定的主终点。'))
    add('')
    rows = []
    for u in units.values():
        g = u['grad']
        rows.append([f"s{u['seed']}", ARM_CN[u['arm']], f"{f2(best(u))} (e{u['best_epoch']})", f2(u['rs'][4]), f2(u['rs'][5]),
                     f"{f2(u['worst'][0])} (e{u['worst'][1]})", f2(u['rs'][39]), f2(u['last5']), f"{g['clip_fraction']:.2f}",
                     f"{g['mean_grad_norm_pre_clip']:.1f} / {g['max_grad_norm_pre_clip']:.1f}", g['n_unique_samples'],
                     '否' if not u['summary']['collapse']['severe_collapse'] else '是'])
    L += md_table(['seed', '臂', 'best (epoch)', 'epoch4', 'epoch5', '最差hard (epoch)', 'last (e39)', '末5轮均值', 'clip比例', '裁剪前梯度范数 均值/最大', '窗口样本', '严重塌陷'], rows,
                  ['---', '---'] + ['---:'] * 9 + ['---'])
    add('')
    proto = read_json(PKG / 'config/protocol.json')
    require(proto['student']['weights_from_update'] == 0 and proto['student']['warmup_epochs'] == 5
            and proto['telemetry']['grad_log_window'] == [200, 200], 'Protocol design facts changed')
    clip_norm = proto['student']['grad_clip']
    gap4 = {c: [U(s, 'VCS')['rs'][4] - U(s, c)['rs'][4] for s in SEEDS] for c in CONTROLS}
    min_ratio = min(u['worst'][0] / u['rs'][4] for u in units.values())
    worst_u = min(units.values(), key=lambda u: u['worst'][0] / u['rs'][4])
    diag = {}
    for c in CONTROLS:
        diag[c] = {'last': [U(s, 'VCS')['rs'][39] - U(s, c)['rs'][39] for s in SEEDS], 'last5': [U(s, 'VCS')['last5'] - U(s, c)['last5'] for s in SEEDS]}
    add(f"- 严重塌陷 {sum(u['summary']['collapse']['severe_collapse'] for u in units.values())}/9。更强的事实：没有任何 hard epoch 的 DEV 低于本臂 epoch4 的50%"
        f"（hard 阶段最差值与 epoch4 之比最低为 {f3(min_ratio)}，{ARM_CN[worst_u['arm']]} s{worst_u['seed']} epoch {worst_u['worst'][1]}），故严重塌陷定义的第一个条件从未满足。")
    add(f"- VCS 在 warm-up 结束时（epoch 4）已领先：相对 Constant-U {', '.join(s2(x) for x in gap4['ConstantU'])}，相对 Shuffle-VCS {', '.join(s2(x) for x in gap4['ShuffleVCS'])}"
        f"（seed 901/902/903）。协议规定权重从第 {proto['student']['weights_from_update']} 次更新起应用、前 {proto['student']['warmup_epochs']} 个 epoch 为 warm-up，"
        '所以这一差距在 hard-negative 阶段开始前就已出现。')
    grad_epochs = sorted({e for u in units.values() for e in u['grad_epochs']})
    require(all(u['grad']['clip_fraction'] == 1.0 for u in units.values()), 'The clip sentence assumes every logged step was clipped')
    win = proto['telemetry']['grad_log_window']
    add(f"- 梯度窗口只覆盖 epoch {grad_epochs[0]} 末 {win[0]} 次与 epoch {grad_epochs[-1]} 初 {win[1]} 次更新（每臂 {units['s901__VCS']['grad']['n_unique_samples']} 个样本），"
        f"9 个单元的 clip 比例均为 {f2(min(u['grad']['clip_fraction'] for u in units.values()))}：窗口内每一步都被裁剪到范数 {clip_norm:g}，因此该窗口内的更新步长由 clip 决定；它不是全程梯度估计。")
    add(f"- 仅作诊断（不替换 best 主终点）：按 last（e39），VCS−Constant 为 {', '.join(s2(x) for x in diag['ConstantU']['last'])}，VCS−Shuffle 为 {', '.join(s2(x) for x in diag['ShuffleVCS']['last'])}；"
        f"按末5轮均值，VCS−Constant 为 {', '.join(s2(x) for x in diag['ConstantU']['last5'])}，VCS−Shuffle 为 {', '.join(s2(x) for x in diag['ShuffleVCS']['last5'])}。")
    add('')
    add('### 3.5 可支持的机制结论与不支持的方向')
    add('')
    o_auc = {a: [offline[s]['offline']['arms'][a]['auroc'] for s in SEEDS] for a in ARMS}
    o_res = {a: [offline[s]['offline']['arms'][a]['residual_noise_mass_fraction'] for s in SEEDS] for a in ARMS}
    rng = lambda xs, fmt=f3: fmt(min(xs)) if fmt(min(xs)) == fmt(max(xs)) else f"{fmt(min(xs))}–{fmt(max(xs))}"
    fx = {k: sum(counts[s][0][k] for s in SEEDS) for k in counts[901][0]}
    nb = sum(counts[s][1] for s in SEEDS)
    require(all(c['positive'] == 3 for c in contrasts), 'The supported-conclusion text below assumes 3/3 positive contrasts')
    require(fx['VCS<Constant'] == fx['VCS<Shuffle'] == fx['Clean<VCS'] == nb, 'The fixed-risk text below assumes uniform batch ordering')
    add('**可支持（描述性，n=3，限于本设计与 DEV）：**')
    add('')
    m0 = offline[901]['manifest']
    a_share = [offline[s]['offline']['arms']['VCS']['anchor_fraction_of_retained_full_mass'] for s in SEEDS]
    add(f"1. **不是 A/U 总质量再平衡。** Constant-U 与 VCS 的 U 总质量、全池分母和 A 质量占比相同，但 best rSum 在 3/3 个 seed 低 {f2(c_by['ConstantU']['min'])}–{f2(c_by['ConstantU']['max'])}。"
        f"因此 VCS 相对本组控制的终点差不能由“A 的质量占比从 {m0['n_A'] / m0['n_pool']:.0%} 提高到 {min(a_share):.1%}–{max(a_share):.1%}”这一再平衡解释。")
    add(f"2. **不是权重分布或 ESS 本身。** Shuffle-VCS 保留 VCS 的权重多重集、总质量与 ESS，只打断权重与图像内容的对应，best rSum 在 3/3 个 seed 低 {f2(c_by['ShuffleVCS']['min'])}–{f2(c_by['ShuffleVCS']['max'])}。")
    add(f"3. **终点差与权重中关于注入错配的逐 pair 信息同时出现。** 只有 VCS 的分配区分了 clean 与 mismatch：AUROC {rng(o_auc['VCS'])}（Constant-U {rng(o_auc['ConstantU'])}，Shuffle-VCS {rng(o_auc['ShuffleVCS'])}），"
        f"残余噪声质量占比 {rng(o_res['VCS'])}（Constant-U {rng(o_res['ConstantU'])}，Shuffle-VCS {rng(o_res['ShuffleVCS'])}）。这符合协议预定的解释分支：")
    L += blockquote(quote(P, '| VCS 同时优于两个控制 |', '再看校准或更有信息的输入 |'))
    add('')
    L += blockquote(quote(P, '这些是预定的解释分支，不是对未执行结果的预测。'))
    add('')
    add(f"4. **固定风险方向一致。** 在同一个固定学生和同一批负候选上，VCS 重加权后的目标在 {fx['VCS<Constant']}/{nb} 个 batch 低于 Constant-U、{fx['VCS<Shuffle']}/{nb} 个低于 Shuffle-VCS，"
        f"clean oracle 在 {fx['Clean<VCS']}/{nb} 个 batch 更低：VCS 把质量从高损失 anchor 移开，方向与 clean oracle 相同。它只说明同一模型上的重加权风险。")
    require(all(x > 0 for c in CONTROLS for k in ('last', 'last5') for x in diag[c][k]) and all(x > 0 for c in CONTROLS for x in gap4[c]),
            'The trajectory sentence assumes positive epoch-4, last and last-5 differences')
    add(f"5. **训练过程。** 9 个单元都没有严重塌陷，hard 阶段也从未跌到 epoch4 的一半以下；VCS 的领先在 epoch 4 已经出现，并保持到 best、last 与末5轮均值（以上均为诊断）。")
    add('')
    add('**不支持或本轮未检验（如实交回）：**')
    add('')
    sc = [U(s, 'ShuffleVCS')['rs'][U(s, 'ShuffleVCS')['best_epoch']] - U(s, 'ConstantU')['rs'][U(s, 'ConstantU')['best_epoch']] for s in SEEDS]
    fixed_sc = '、'.join(f"{counts[s][0]['Shuffle<Constant']}/{counts[s][1]}" for s in SEEDS)
    add(f"1. **“A/U 再平衡是 N2F 带权收益的主要来源”：不支持**（见上第1点）。本轮没有 Unweighted 臂，再平衡本身相对不加权是否有益不在预定比较内；"
        '原 N2F 的 Unweighted 只能作上下文（下表），不能与本轮单元拼成新的预定比较。')
    require(0 < sum(x > 0 for x in sc) < 3, 'The Shuffle-vs-Constant sentence assumes mixed signs')
    add(f"2. **“错误分配比恒定权重更差”：不成立为结论。** Shuffle-VCS − Constant-U 的 best 差为 {', '.join(s2(x) for x in sc)}（seed 901/902/903），"
        f"只有 {sum(x > 0 for x in sc)}/3 个 seed 为正，符号不一致；这也不是预定比较。固定风险中 Shuffle 的目标低于 Constant 的 batch 数为 {fixed_sc}（seed 901/902/903），"
        '与完整训练终点的排序并不对应。')
    add('3. **统计显著性、置信区间、等价性或独立确认：不提供。** n=3 个配对 seed，DEV 已多次使用；工程规则通过不是检验结果。')
    add('4. **VCS 相对 BCE/RankMass/RuLSIF 等其他估计器的独有优势：本轮未检验。** N2M 只比较锁定 VCS 权重的分配与两种控制；N2F 收尾中 VCS 专有门槛未通过的结论不变。')
    ece = {a: [offline[s]['offline']['arms'][a]['calibration_ece_10_equal_width'] for s in SEEDS] for a in ARMS}
    brier = [offline[s]['offline']['arms']['VCS']['brier'] for s in SEEDS]
    require(max(ece['ConstantU']) < min(ece['VCS']), 'The ECE sentence below no longer matches the records')
    add(f"5. **校准与终点的剂量关系：无法估计。** 三个 seed 的 VCS Brier 只在 {rng(brier, f4)} 之间，本轮只有“有信息 vs 无信息”的对比；"
        f"Constant-U 的 ECE（{rng(ece['ConstantU'], f4)}）低于 VCS（{rng(ece['VCS'], f4)}）而终点最低，所以单看 ECE 不能解释本轮的终点差。")
    add('6. **固定风险不是重训终点，也不是 clean 负样本池风险**；评分器未加载模型，只做哈希绑定（见上方引文）。')
    add('7. **泛化：未检验。** 没有读 TEST，也没有 COCO、其他 rho、anchor 预算、MEAN 尺度或其他学习率；机制标签只表示注入映射，不是语义真值。')
    vcs_n2f = [U(s, 'VCS')['rs'][U(s, 'VCS')['best_epoch']] - n2f[(s, 'VCS_Weight', 'sum')]['best_dev_rsum'] for s in SEEDS]
    add(f"8. **跨硬件等价：不主张。** 本轮 VCS 在 RTX PRO 6000 上重跑，与 N2F 的 VCS SUM 相差 {', '.join(s2(x) for x in vcs_n2f)}（N2F 硬件见下表），"
        '只说明同配方在不同硬件/作业上的描述性差异。')
    add('')
    add('原 N2F 上下文（只作上下文，来自 N2F 收尾 CLOSEOUT.json；不是本轮预定比较）：')
    add('')
    rows = []
    for s in SEEDS:
        a = n2f[(s, 'VCS_Weight', 'sum')]
        b = n2f[(s, 'Unweighted_A', 'sum')]
        rows.append([f"s{s}", f"{f2(a['best_dev_rsum'])}（{GPU_SHORT[a['gpu']]} {a['host']}）", f"{f2(b['best_dev_rsum'])}（{GPU_SHORT[b['gpu']]} {b['host']}）",
                     f2(best(U(s, 'VCS'))), f2(best(U(s, 'ConstantU'))), f2(best(U(s, 'ShuffleVCS')))])
    L += md_table(['seed', 'N2F VCS/SUM', 'N2F Unweighted_A/SUM', 'N2M VCS', 'N2M Constant-U', 'N2M Shuffle-VCS'], rows, ['---', '---:', '---:', '---:', '---:', '---:'])
    add('')
    L += blockquote(quote(P, '原 Unweighted、RankMass、BCE、RuLSIF、OracleMask/Drop 只在原 N2F 完整表中作为上下文。'))
    add('')

    # 4 engineering rule
    add('## 4 工程投入规则：结果与含义')
    add('')
    L += blockquote(quote(H, '工程投入规则已经固定', '仍需考虑分布/ESS/梯度效应。'))
    add('')
    L += blockquote(quote(P, '**工程继续规则**', '未通过也不证明等价。'))
    add('')
    rows = [[c['contrast'], s2(c['mean']), f"≥ +{rule['mean_rsum_difference_each_at_least']:g}", f"{c['positive']}/3", f"≥ {rule['positive_seeds_each_at_least']}/3",
             f"{c['vcs_severe']} vs {c['control_severe']}", '是' if c['pass'] else '否'] for c in contrasts]
    L += md_table(['比较', '三seed均差', '要求', '正方向', '要求', '严重塌陷 VCS vs 控制', '满足'], rows, ['---', '---:', '---', '---:', '---', '---', '---'])
    add('')
    require(rule_pass is True, 'The rule-interpretation text below assumes the rule passed')
    add(f"规则结果：{rule_pass}（summary.json `investment_rule.pass`，参数取自冻结协议，本脚本复算一致）。按协议，它的含义只有：")
    add('')
    add('- **是资源决定**：满足“继续投入内容关联可靠度扩展”的预设条件；按预定解释分支，下一步看校准或更有信息的输入。')
    add('- **不是检验**：不表示显著性、置信区间或独立确认，也不排除任何 run；n=3 与 DEV 复用的限制不变。')
    add('- **规则未过的两个分支不适用**：没有出现“未过→转向解释 A/U 再平衡”的情形；VCS 同时优于 Constant 与 Shuffle，所以“只优于 Constant 时仍需考虑分布/ESS/梯度效应”的分支也不适用。')
    require(co['readout']['decision']['vcs_specific_confirmation_worth_testing'] is False, 'N2F VCS-specific gate record changed')
    add('- **它不改变 N2F 的已有结论**：N2F 收尾记录的 VCS 专有门槛（相对 BCE、RankMass、RuLSIF）为 false；本轮问题不同，不据此改写。')
    add('')
    for w in summary['warnings']:
        L += blockquote(json_quote(OUT / 'analysis/full/summary.json', w, 'warnings[]'))
        add('')

    # 5 cost
    add('## 5 成本')
    add('')
    L += blockquote(quote(OUT / 'analysis/cost/SOURCE_NOTE.md', 'sacct / slurmdbd were unreachable for the whole N2M round'))
    add('')
    rows = []
    for j in acc['jobs']:
        sc_ = scontrol[j['job_id']]
        arms = [u for u in units.values() if u['fr']['hardware']['job_id'] == j['job_id']]
        what = f"seed {arms[0]['seed']} 三臂训练（{sc_['NodeList']}）" if arms else f"固定损失导出（{sc_['NodeList']}）"
        proc = f"{sum(u['attempt']['elapsed_s'] for u in arms) / 3600:.2f}" if arms else '—'
        rows.append([j['job_id'], sc_['JobName'], what, j['state'], j['exit_code'], j['gpu_count'], f"{j['gpu_allocation_hours']:.3f}", proc])
    L += md_table(['job', '名称', '内容', '状态', '退出码', 'GPU', 'allocation GPU-h（重建）', '各臂进程时长合计 h'], rows,
                  ['---', '---', '---', '---', '---', '---:', '---:', '---:'])
    add('')
    seeds_h = sum(j['gpu_allocation_hours'] for j in acc['jobs'] if any(u['fr']['hardware']['job_id'] == j['job_id'] for u in units.values()))
    t0, t1 = torch_report['started_utc'], torch_report['finished_utc']
    add(f"- 已知 GPU allocation 合计 **{acc['gpu_allocation_hours_known']:.2f} GPU-h**（其中三个 seed 训练作业 {seeds_h:.2f}）；`complete_terminal_accounting=false`。")
    add(f"- **CUDA 核对作业 {', '.join(acc['missing_jobs'])} 的 allocation 时长未知，不计为 0**：该作业没有 scontrol 快照，sacct 也不可用。"
        f"其核对进程自记录 {t0} 至 {t1}，这只是进程时间，不是 allocation。")
    attempts_h = sum(u['attempt']['elapsed_s'] for u in units.values()) / 3600
    add(f"- 9 次训练的 attempt 进程时长合计 {attempts_h:.2f} h（每臂 {min(u['attempt']['elapsed_s'] for u in units.values()) / 3600:.2f}–{max(u['attempt']['elapsed_s'] for u in units.values()) / 3600:.2f} h）；"
        '同一 seed 的三臂共享一次 allocation，allocation 只计一次。没有历史复用成本。')
    cpu = []
    for job in sorted(scontrol):
        f = scontrol[job]
        if 'gres/gpu' not in f.get('AllocTRES', '') and job not in {j['job_id'] for j in acc['jobs']}:
            never = f['JobState'] == 'CANCELLED' and runtime_seconds(f['RunTime']) == 0
            cpu.append(f"{job} {f['JobName']} {f['JobState']} RunTime {f['RunTime']}" + (f"（Reason={f['Reason']}，未运行）" if never else ''))
    logs = {m.group(2): m.group(1) for p in SLURM_LOGS.iterdir() for m in [re.match(r'^((?:n2m_|v9n2m_)\w*?)_(\d+)\.out$', p.name)] if m}
    no_snap = sorted(j for j in logs if j not in scontrol and j != GPU_JOBS_VERIFY)
    fix_jobs = [j for j in no_snap if logs[j] == 'n2m_fixscore']
    require(len(fix_jobs) == 1, 'Expected exactly one n2m_fixscore run without a snapshot')
    fix_log = (SLURM_LOGS / f'n2m_fixscore_{fix_jobs[0]}.out').read_text()
    require(all(f'fixscore_s{s_}_rc=0' in fix_log for s_ in SEEDS), 'The fixed-loss scoring job did not record rc=0 for every seed')
    fix_end = re.findall(r'^end=(\S+)$', fix_log, re.M)
    require(len(fix_end) == 1, 'Fixed-loss scoring end time not recorded')
    add(f"- CPU 作业（不计 GPU）：有 scontrol 快照的为 {'；'.join(cpu)}；只有 stdout 日志、没有快照（时长未知）的为 "
        f"{'、'.join(f'{j} {logs[j]}' for j in no_snap)}。固定风险评分由 {fix_jobs[0]} 完成（其日志记录三个 seed 均 rc=0，end={fix_end[0]}）。")
    L += blockquote(quote(H, '成本按当前观测粗估：9次RTX完整训练约57 allocation内小时'))
    add('')

    # 6 file list
    add('## 6 §7 交回文件清单')
    add('')
    L += blockquote(quote(H, '回传新包实际代码commit及package SHA'))
    add('')
    L += blockquote(quote(H, '大图像特征、caption原文和模型二进制可以留server。', '报告保留其artifact SHA。'))
    add('')
    n_files = sum(len(e) for _, e in groups)
    n_bin = sum(e['status'] == 'server_binary' for _, es in groups for e in es)
    bin_gb = sum(e['bytes'] for _, es in groups for e in es if e['status'] == 'server_binary') / 1e9
    n_n2m_bin = sum(e['status'] == 'server_binary' and e['path'].startswith(str(OUT)) for _, es in groups for e in es)
    n_verified = sum(v == 'VERIFIED' for u in units.values() for v in u['summary']['artifact_verification'].values())
    add(f"代码与包：包原样提交于 `{COMMITS['package']}`（之后未改动，本脚本以 git 核对）；服务器侧脚本最后提交于 `{COMMITS['server']}`；包 zip 的 sha256 `{zip_record}`；"
        f"包内 {len(manifest['files_sha256'])} 个文件与 `PACKAGE_MANIFEST.json` 逐一一致；server 核对记录的 package_code_digest `{sv['package_code_digest']}`，"
        f"effective runtime digest `{sv['effective_runtime_digest']}`。")
    add('')
    add(f"**checkpoints_not_materialized**：本交回只含元数据，{n_bin} 个模型二进制（{n_n2m_bin} 个 N2M best/last 与 {n_bin - n_n2m_bin} 个 N2F 参照 checkpoint，共 {bin_gb:.2f} GB）留在 server。"
        f"下表中它们的 sha256 是记录的 artifact SHA；主比较的二进制核验已在仍持有二进制的 server 上由 summarize 完成（summary.json 中 {n_verified} 项产物为 VERIFIED，含每个单元的 best.pt 与 last.pt），"
        f"需要时在 server 上运行 `python3 {GENERATOR_REL} --binaries` 重算。")
    checks_bin = (n_bin, bin_gb)
    add('')
    add(f"共 {n_files} 个文件；机器可读清单 `v9_n2m/handback/V9_N2M_HANDBACK_FILES_20261020.json`（sha256 `{files_sha}`）。位置：repo = 工作树 `{REPO}`（分支 v9/n2m-20261019），"
        f"server = 本服务器路径（OUT = `{OUT}`）。")
    add('')
    for gname, entries in groups:
        add(f"### {gname}")
        add('')
        rows = []
        for e in entries:
            p = e['path']
            short = p.replace(str(OUT) + '/', 'OUT/').replace(str(REPO) + '/', '')
            where = {'repo': 'repo', 'server': 'server', 'server_binary': 'server（二进制，checkpoints_not_materialized）'}[e['status']]
            rows.append([f"`{short}`", f"`{e['sha256']}`", e['bytes'], where, e['note']])
        L += md_table(['文件', 'sha256', '字节', '位置', '说明'], rows, ['---', '---', '---:', '---', '---'])
        add('')

    # 7 checks
    add('## 7 核验与复现')
    add('')
    add(f"`python3 {GENERATOR_REL} --check` 重新核验全部记录并与已写出的文件逐字节比较；`--binaries` 另外重算 {checks_bin[0]} 个 checkpoint 的 sha256（{checks_bin[1]:.2f} GB，只在 server 上运行）。"
        '核验包括：experiment receipt、协议、CUDA/运行时 receipt 与 N2F census 的哈希链；包文件与 PACKAGE_MANIFEST、zip 与 PACKAGE_ZIP.sha256、相关 git 提交未变；'
        '9 个单元的 identity、JSON 产物哈希、40×1133 次更新、best 选择、Recall/rSum/mR、progress、attempt 记录、梯度窗口（由 grad_norms.jsonl 重算）、同 job 配对与环境一致性；'
        'summary/unit_results/paired_contrasts/epoch_trajectories 与训练日志重算一致；两项配对差与工程规则复算；controls 的置换与三臂文件哈希、离线评分与固定风险的身份绑定和 CSV/JSON 一致；'
        '固定学生 checkpoint 与 N2F artifact SHA 一致；成本行与 scontrol 快照一致、缺失作业确实没有记录。任一不符时脚本拒绝运行，不写文件。')
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(L)).rstrip('\n') + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description='V9 N2M hand-back report generator and checker')
    ap.add_argument('--check', action='store_true', help='regenerate in memory and compare with the written files')
    ap.add_argument('--binaries', action='store_true', help='also re-hash the checkpoints against their recorded artifact SHA')
    args = ap.parse_args(argv)
    try:
        outputs, info = build(binaries=args.binaries)
    except Refusal as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        return 2
    targets = {REPORT: outputs['report'], FILES_JSON: outputs['files'], COPY_DIR / REPORT.name: outputs['report']}
    if args.binaries:
        print(f"checkpoints re-hashed and matched: {info['checkpoints_hashed']}")
    if args.check:
        bad = 0
        for path, text in targets.items():
            same = path.exists() and path.read_bytes() == text.encode('utf-8')
            print(f"{'OK  ' if same else 'DIFF'} {path}")
            bad += not same
        return 1 if bad else 0
    for path, text in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'.{path.name}.tmp')
        tmp.write_bytes(text.encode('utf-8'))
        tmp.replace(path)
        print(f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}  {path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
