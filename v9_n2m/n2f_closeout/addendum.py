#!/usr/bin/env python3
"""Deterministic, records-only addendum to the V9 N2F closeout (CPU audit job 1046165, 2026-10-19T14:14:57Z).

Reads only frozen V9 N2F records: the closeout bundle, the 33 unit result directories, the frozen protocol,
the split / ESS-recovery records, the wave-1 history and the A/B diagnostic results.  No training, no TEST read,
no model inference, no checkpoint loading; the frozen closeout bundle and the V9 code are never written.

Every unit file it uses is verified against CLOSEOUT.json (full_result.json sha256) or the hash chain that starts
there (full_result.json artifact_sha256 -> train_log.json / run_manifest.json / identity.json); every other record is
verified against a sha256 pinned below.  Any mismatch or inconsistency refuses (exit 2) before anything is written.

    python addendum.py              verify inputs and (re)write the four outputs next to this script
    python addendum.py --check      verify inputs, regenerate in memory, compare byte-for-byte with the files on disk
    python addendum.py --out DIR    write the outputs to DIR instead (e.g. to diff a re-run)
    python addendum.py --plot PNG   additionally draw the 33 DEV trajectories (optional, needs matplotlib, not checked)

Outputs: N2F_FINAL_33_UNITS.csv, N2F_FINAL_33_UNITS.json, N2F_TRAJECTORIES.csv, V9_CLOSEOUT_ADDENDUM_CN.md.
They depend only on the pinned inputs and on this file, so a re-run is byte-identical.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # importing the frozen summarizer must not create __pycache__ in the frozen tree

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

PROJECT = Path('/home/infres/yinwang/CS_QMI')
V9 = PROJECT / 'handoff/v9_full_20261017'
BUNDLE = PROJECT / 'handoff/v9_overlap_v10_20261018'
SPLIT = PROJECT / 'handoff/v9_unit_split_20261018'
ESS = PROJECT / 'handoff/v9_ess_resume_20261018'
REPORTS_V9 = PROJECT / 'ssl_pilot_v9/reports/V9'
REVIEW = PROJECT / 'handoff/review_claude_20261019/REVIEW_CN.md'  # cited as the origin of the findings; not read
HERE = Path(__file__).resolve().parent
GENERATOR_REL = 'v9_n2m/n2f_closeout/addendum.py'

V9_FREEZE = 'ae8d1e55d7ec5ae9439950d776a2c5efa122de45a58490aadcde0f6046fe1907'
CLOSEOUT_SHA = '0f5ba97134826d14e2bad05d047f48afeb2f4684d316c6da32319af29072c978'
OUTPUTS = ('N2F_FINAL_33_UNITS.csv', 'N2F_FINAL_33_UNITS.json', 'N2F_TRAJECTORIES.csv', 'V9_CLOSEOUT_ADDENDUM_CN.md')
EPOCHS = 40
LATE = range(30, 40)  # the ten epochs after the epoch-30 learning-rate decay
TOLERANCE_S = 300  # node/controller clock skew allowance when placing epoch timestamps inside job windows

# Records outside the closeout hash chain, pinned at first generation (2026-10-19).
_P = PROJECT
_W1 = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k/rho0.4')
PINNED = {
    BUNDLE / 'JOB_OBSERVATIONS.json': '9d58c8810f571bad732569f5a0565be7b69e6ce3c5c713124b2257540483dfe2',
    SPLIT / 'UNIT_LEDGER.json': '9712336f661aaf320539d279beec964bc1a5d601977e9cbc592bd7a205f4b6bb',
    SPLIT / 'SCHEDULING_AMENDMENT.json': '6cfb050d1b9432b997e7a527d44e94e1e670a23d0728ae4c4aadb2a798a0b3ea',
    SPLIT / 'README_CN.md': '35bbd69add1600b38530bdc9935d60dba5b390c8f575a270e5faf082f626e78a',
    SPLIT / 'transitions/903_1/MIGRATED.json': '13667deeaab413efd25a0284d11463e3711b5785efd558f540d27a4c31f2f1e6',
    SPLIT / 'transitions/903_2/MIGRATED.json': 'c021c5f3b974a6343884e78f72009faa818554fd0db168a46dfcdc504f32eca1',
    SPLIT / 'transitions/903_1/CONTROLLER_ORIGINAL.json': 'c018bff335b137b9148e767f6016fe46695324f6074bbbd5052b6809cda5d196',
    SPLIT / 'transitions/903_2/CONTROLLER_ORIGINAL.json': 'fb3a7df670a31554ba0149a9839e4eaf62cd425a9ab4961ec3461f6a1a420e69',
    SPLIT / 'units/903_1/0/hardware_1046652.json': '5cf4c2df4b28e5f731affcc0ea27b5b8e8889217f638fa826f7b6e729456a8f5',
    SPLIT / 'units/903_1/0/hardware_1047108.json': 'ba4d1432a50b2eee2a30aea60e2bf8e69b00f24f9a1dbb92774019441dc46c7b',
    SPLIT / 'units/903_1/1/hardware_1046654.json': '81e7b9ead07e4238cd670fa380d8fbc3ea38c39c8d323002d715420bd8b7630b',
    SPLIT / 'units/903_2/0/hardware_1046653.json': 'ab22d9f9a3416bb01af8b1ac68a807dc1b5928e0698387e949ab5571b97b6dda',
    SPLIT / 'units/903_2/1/hardware_1046655.json': '33c0547ad77281ac851049109f2a0aaf154c42ced2234d29c3f71bd61e63e37e',
    SPLIT / 'units/903_1/0/EXIT_1046652.json': '971901abe3d53e9736f25ff79b3e82f1c9c9e111e9b4776703a7e9a4c9b8dd1e',
    SPLIT / 'units/903_1/0/FAILED_1046652.json': '6ec0c7f6c13554679ce5352456fdaef157fd4552ed17e46dd5603c6a30ba998d',
    SPLIT / 'checkpoints_before_split/903_1/hardware.json': '3349e5737821776a420d6d784016cb1a172c0e43b26eaac11aed0265a8e14fd3',
    SPLIT / 'checkpoints_before_split/903_2/hardware.json': '40ef776803a9c78decbeaeb71a2fdab31bedf749ed559f3bb1a9a9eba01edd45',
    SPLIT / 'TIMEOUT_GUARD_CONTROLLER.json': '5043a7cf283240124f7bcee03635e526a133ae5cd4ae7b6d2dfca4f6045dd80e',
    SPLIT / 'TIMEOUT_GUARD_RESULT.json': '52108f0923d8f5e59c3a41dbac894e6445891cc571aaeb9b4121453bd9c022e8',
    ESS / 'DIAGNOSIS.json': '16f42425a5dd41e9707700e02522d4e722654ecef9be5a3bef95ea2fcab8c94b',
    ESS / 'RECOVERED_UNIT.json': 'af0b07052ad66e40248731b65da8af2dc7bdff01407d51114e3eaf791f58dfe7',
    REPORTS_V9 / 'V9_N2_units/wave1_paused.json': '94395161976a9fab35256a2c20288aab01ac45dec155d4a5ec100c893fb70c64',
    REPORTS_V9 / 'V9_N2_units/formal_units.json': '946db0584719f33b548cd7aa849321267acae19ace2e22316a5215f23eb33b2e',
    REPORTS_V9 / 'V9_N2_units/tuning_units.json': '6a02ffa992272d3f3661a9784acca4c1d9fdbfbebe49c3890a823bf6710f8aef',
    REPORTS_V9 / 'V9_N2_PROTOCOL_FROZEN_20261015.yaml': 'ce4af32b6091a4d1fd8eba55890ab25a5d093a78644be390ddf4d7813b46eaab',
    REPORTS_V9 / 'V9_N2_diag/spec_job1.json': '48e56051ddcf95aa4c518be818005102c1c880cbddfd899871e5be0a1bc59821',
    REPORTS_V9 / 'V9_N2_diag/spec_job2.json': '263b3ee56e84151945b5b9bd7fa672a8480b6c5248d780c8fb41f91031ed29a6',
    REPORTS_V9 / 'V9_N2_diag/spec_job1.results.json': '97a26152586071d60f357016dba1bd6eeabb309cb52dfbcd14e81ffd4c03464e',
    REPORTS_V9 / 'V9_N2_diag/spec_job2.results.json': '0007a350bada14326860fccccfadbb5eea3c716c6b48ae6ffe6e38732fc95001',
    _W1 / 's901/unit_BCE_Weight_lr1/train_log.json': '0e7d89f1ae263884f28355f27408cbced0957b1885d134f41d7009585aebc8d8',
    _W1 / 's901/unit_BCE_Weight_lr1/done.json': '5bc441fedeb5694509196ae5e0ac5beebe4507d9730c44635baf314b485cf27b',
    _W1 / 's901/unit_BCE_Weight_lr1/run_manifest.json': '8435de9f0fb377b573ce439f761eb3a7953016c9bb1c0175e6e891bfd598deba',
    _W1 / 's902/unit_Unweighted_A_lr1/train_log.json': '2db8fdeca5c89e468cb10c273ef383668c543a95c4fe93a1d5337c2c068345ac',
    _W1 / 's902/unit_Unweighted_A_lr1/done.json': '4fb0373eaa23be4b9c688e8298b51d0c3d27c18c9889c1ca52385fa13ac66cec',
    _W1 / 's902/unit_Unweighted_A_lr1/run_manifest.json': '5aea6d08e6a494baf362270471a92a053897f7e1bdb7f64431613cbedcab9eaa',
    _W1 / 's917/unit_Unweighted_A_lr0.5/train_log.json': '4fc8b16796141af2df4f187f1bd20826ff9af492531e579759a273ef230bcb1d',
    _W1 / 's917/unit_Unweighted_A_lr0.5/done.json': 'f8633e5cd6ba9f53d03f1e5065162a682bcca9fcbffbf314c8919b51484b4012',
    _W1 / 's917/unit_Unweighted_A_lr0.5/run_manifest.json': 'f36df9e03af5fee4759807634cba670488c1a4941c59187ae9f54bcee66baff9',
    _W1 / 's917/unit_VCS_Weight_lr1/train_log.json': 'aeff38cf3052ffd3d642b6ea1e0183afec4524301636c0d43b500097bcf3836b',
    _W1 / 's917/unit_VCS_Weight_lr1/done.json': 'ec773e037c71b28b8919f6efee68363ee6072262eddb5907b3965db653201d12',
    _W1 / 's917/unit_VCS_Weight_lr1/run_manifest.json': '78865a7d2a16c0f666bffbc00267918e09e2a60f6f3666f8442c214adf2c5283',
    _W1 / 's917/unit_Unweighted_A_lr1/train_log.json': 'b87d986544b68743e2a4dce05d75d20b3e25f17c8134403f64424a19f7a0b1e8',
    _W1 / 's917/unit_Unweighted_A_lr1/done.json': 'dc709175576c2eefa5ddb92425dc94ce009fdc7223a7d355b8b1f90dc613482b',
    _W1 / 's917/unit_Unweighted_A_lr1/run_manifest.json': '92bda4050fa13cb4cce62a0c16d95bc49444a18dd9af86cb46bada8c3d23fbe7',
}
PROGRESS_SHA = {  # results/<group>/<method>_<scale>/progress.jsonl (not part of the closeout hash chain)
    '901_1/VCS_Weight_mean': '2eb95864b478339cba38679a54fcb8b3456f70a9428b77cf56a6fc20718a35ad',
    '901_1/VCS_Weight_sum': '91c08b25248058f3951d0a398144d567a93023b80f848bf77cfdd2f1649816c5',
    '901_2/BCE_Weight_mean': '8c188aa03ffbfd8fe9cc13ac1ce8c7c6dc3198ba0c9a4879eec0e122f1999994',
    '901_2/BCE_Weight_sum': 'e6606a47a5a8761e686b516f1a615c73cc06892c182b47910be4176b72ab2a1e',
    '901_3/RankMass_mean': '233c362819735b13341a694e2664f5889cc0e9fa8d0ab4d1c448810bc8b8d211',
    '901_3/RankMass_sum': '4f0c1029cdf3793ab7a035a32ffb8e44e00589b09ececdc2386f1ef5b688c173',
    '901_4/Unweighted_A_mean': '24001140fbbb94033d72ff5d387955de4afc35f464cbfc94624e5b76dda1f9ba',
    '901_4/Unweighted_A_sum': '0dc06a9241cab932f59fd10e04f55c9402c6be30c2754c396fe7c773aef9eaf6',
    '901_or/OracleDrop_sum': '0efca8e920874e3addd3a1ebe21b4d531c49a91c4b3f5b71fd32220bb7b0e2b6',
    '901_or/OracleMask_sum': 'c32307b844cb956ab7a2056519e0146e72a5960d737b5df88101d0c5d8c72f3e',
    '901_ru/RuLSIF_Weight_sum': '72e9260d5da31af5c6edacf9f5ee8fc063812b6c41b2422d313d34acbcc0d00e',
    '902_1/VCS_Weight_mean': '3ee47249e77c4644e249a832299b373f20dc7b6248bbb5c21ee56f96e06836ef',
    '902_1/VCS_Weight_sum': '4546eeda2a8524cf3ffe3b834c883f62098c9e5aec7feba0cdef1cda01ce559e',
    '902_2/BCE_Weight_mean': 'c55a02478ccf46d300f6d66e2a3538891c72afc5f225e7ddd1399bd0bbdc3c17',
    '902_2/BCE_Weight_sum': '7357a757a4930e9762ea64ff4c3b575b8c378dd6da11a169b88df73c9ef8d79e',
    '902_3/RankMass_mean': 'b1ff654f2c0aa9c02f0c8e32c2d8bf2d88d00c68f78f76e9d533bf84a3bd5aee',
    '902_3/RankMass_sum': '61d4d1189e684be91cbc16c98f49509829f160492f2c5b8b70299b5bc9ce542c',
    '902_4/Unweighted_A_mean': 'c388ef5e94a57ae290a7afc5e75ca07af73533afc30364cf2bfbedc9b417bc13',
    '902_4/Unweighted_A_sum': 'fc5e9e292261a015e31587885d9e27f83847b61078dc4f41a233422ddfa3cc2f',
    '902_or/OracleDrop_sum': '834abfd8a5d32269ee2dca71a025e55b9c5e7a53201947b8f6e027c6c56246e8',
    '902_or/OracleMask_sum': 'fa4176ef52a7590af3412685ef0cfdaea0e1b1450d831d5ac8b7600f3fb5a84d',
    '902_ru/RuLSIF_Weight_sum': '0c23ea6ea13b28ca98de545c80c4d48770e7063ccfb166cf67d898224d1a47b7',
    '903_1/VCS_Weight_mean': 'bf77c325af47fa1777b1d22cf926b23ce7c092964b78f949e9da958e221df1e7',
    '903_1/VCS_Weight_sum': '02085933b7cb62741e909a129e97d4f187dcb2dbdf0eb55b1764d34d40c0c7e1',
    '903_2/BCE_Weight_mean': '4967780b049fb307369bba7a7ef49cd48d4580452abf4e4e1354c8ca452640fa',
    '903_2/BCE_Weight_sum': '71701526353008ff4a8ee4e41b0b57ee4930d4462005d1a45678c7bb1fb48588',
    '903_3/RankMass_mean': 'a4dbc795d598056663539b16a610aab92c68187ff852294922cc7a0b891066ea',
    '903_3/RankMass_sum': '2d3a1c8885862c7d06faf24aab5d8f71da522650dc690c78f54d6fa4d4d7418e',
    '903_4/Unweighted_A_mean': 'a74b13750a2cba4f3fab7014531bc722c571b0b6cc1f3f3e96abb50ed2daecfe',
    '903_4/Unweighted_A_sum': 'ce49f02218dd5ded1dce7c49236ed6d006daf1b524dd79245ae10a1c33e1e0bc',
    '903_or/OracleDrop_sum': 'f7437a0ab18a7e26758104246fd4faa4037c242ab36db86fa90cafaf2afb41b2',
    '903_or/OracleMask_sum': '027b92d2227ed3fb5d7efa6aa8a99bc21ca101459aba7a71456b6226641b70ad',
    '903_ru/RuLSIF_Weight_sum': '5610f1abbe31d44bf16cd5bf3f64dab132b2a8a125c91151772d19662365ac71',
}

GPU_SHORT = {'NVIDIA L40S': 'L40S', 'NVIDIA H100 NVL': 'H100 NVL',
             'NVIDIA RTX PRO 6000 Blackwell Server Edition': 'RTX PRO 6000'}
PARTITION_GPU = {'L40S': 'NVIDIA L40S', 'H100': 'NVIDIA H100 NVL', 'RTX6000PRO': 'NVIDIA RTX PRO 6000 Blackwell Server Edition'}
PROTOCOL_COST_LABEL = {'NVIDIA RTX PRO 6000 Blackwell Server Edition': 'RTX', 'NVIDIA L40S': 'L40S', 'NVIDIA H100 NVL': 'H100 NVL'}
MAIN_METHODS = ('VCS_Weight', 'BCE_Weight', 'RankMass', 'Unweighted_A', 'RuLSIF_Weight')
TABLE_ORDER = [(m, 'sum') for m in ('VCS_Weight', 'BCE_Weight', 'RankMass', 'Unweighted_A', 'RuLSIF_Weight', 'OracleMask', 'OracleDrop')] + \
              [(m, 'mean') for m in ('VCS_Weight', 'BCE_Weight', 'RankMass', 'Unweighted_A')]
SHOW_EPOCHS = (0, 4, 5, 6, 7, 10, 20, 29, 30, 39)
WAVE1_CENSUS = (  # the units the frozen census names, in its order
    (901, 'BCE_Weight', 1.0, 'formal', 'collapsed'), (902, 'Unweighted_A', 1.0, 'formal', 'collapsed'),
    (917, 'Unweighted_A', 0.5, 'tuning', 'collapsed'), (917, 'VCS_Weight', 1.0, 'tuning', 'collapsed'),
    (917, 'Unweighted_A', 1.0, 'tuning', 'dipped and recovered'))


class Refusal(Exception):
    pass


def require(condition, message):
    if not condition:
        raise Refusal(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


class Sources:
    """Every input read goes through here; each must be verified by a pin or by the closeout hash chain."""

    def __init__(self):
        self.read_log = {}

    def raw(self, path, expected=None, route=None):
        path = Path(path)
        data = path.read_bytes()
        digest = sha256_bytes(data)
        if path in PINNED:
            require(digest == PINNED[path], f'Pinned record changed: {path}')
            require(expected is None or expected == digest, f'Pinned record disagrees with its chain: {path}')
            route = route or 'sha256 pinned in addendum.py'
        else:
            require(expected is not None and route, f'Unverified input: {path}')
            require(digest == expected, f'Hash mismatch ({route}): {path}')
        previous = self.read_log.get(str(path))
        require(previous is None or previous['sha256'] == digest, f'Input changed during the run: {path}')
        self.read_log[str(path)] = {'path': str(path), 'sha256': digest, 'verified_by': route}
        return data

    def json(self, path, expected=None, route=None):
        return json.loads(self.raw(path, expected, route).decode('utf-8'))

    def text(self, path, expected=None, route=None):
        return self.raw(path, expected, route).decode('utf-8')


# ---------------------------------------------------------------- verbatim quotes

def _line_of(text, index):
    return text.count('\n', 0, index) + 1


def quote_span(src_text, path, start, end=None):
    """Verbatim substring from `start` through `end` (same line), or the sentence containing `start`."""
    require(src_text.count(start) == 1, f'Quote anchor not unique in {path}: {start[:40]}')
    i = src_text.index(start)
    line_end = src_text.find('\n', i)
    line_end = len(src_text) if line_end < 0 else line_end
    if end is None:
        j = src_text.find('。', i)
        require(0 <= j < line_end, f'Sentence end not found in {path}: {start[:40]}')
        j += 1
    else:
        j = src_text.find(end, i)
        require(0 <= j and j + len(end) <= line_end, f'Quote end not found on the same line in {path}: {end[:40]}')
        j += len(end)
    return {'text': src_text[i:j], 'path': str(path), 'line': _line_of(src_text, i)}


def yaml_value(src_text, path, key, prefix):
    """The double-quoted scalar of the unique `key:` line whose value starts with `prefix` (no YAML parser needed)."""
    pattern = re.compile(r'^(\s*)' + re.escape(key) + r': "(.*)"$', re.M)
    hits = [m for m in pattern.finditer(src_text) if m.group(2).startswith(prefix)]
    require(len(hits) == 1, f'YAML key not unique in {path}: {key}')
    return {'text': hits[0].group(2), 'path': str(path), 'line': _line_of(src_text, hits[0].start(2))}


# ---------------------------------------------------------------- helpers

def parse_utc(value):
    return datetime.strptime(value.rstrip('Z'), '%Y-%m-%dT%H:%M:%S')


def parse_scontrol(stdout):
    fields = dict(re.findall(r'(?:^|\s)([A-Za-z]+)=(\S*)', stdout))
    require('JobId' in fields and 'NodeList' in fields, 'Malformed scontrol record')
    return fields


def ulps(a, b):
    return abs(a - b) / math.ulp(max(abs(a), abs(b)))


def f2(x):
    return f'{x:.2f}'


def s2(x):
    return f'{x:+.2f}'


def short_gpu(name):
    require(name in GPU_SHORT, f'Unknown GPU model: {name}')
    return GPU_SHORT[name]


def seg_label(seg, with_job=True):
    core = f"e{seg['first_epoch']}–{seg['last_epoch']} {short_gpu(seg['gpu'])} {seg['host']}"
    return core + (f" ({seg['job_id']})" if with_job else '')


def unit_name(u):
    return f"{u['method']}/{u['scale'].upper()} s{u['seed']}"


def load_frozen_collapse(src, frozen):
    for rel in ('summarize.py', 'common.py'):
        src.raw(V9 / rel, frozen['code_sha256'][rel], 'FROZEN.json code_sha256')
    sys.path.insert(0, str(V9))
    try:
        spec = importlib.util.spec_from_file_location('frozen_v9_summarize', V9 / 'summarize.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # top level only defines functions; summarize() (which writes) is never called
    finally:
        sys.path.remove(str(V9))
    return module.collapse


# ---------------------------------------------------------------- records

def load_records(src):
    co = src.json(BUNDLE / 'CLOSEOUT.json', CLOSEOUT_SHA, 'sha256 pinned in addendum.py (closeout)')
    require(co['status'] == 'PASS' and co['job_id'] == '1046165' and co['utc'] == '2026-10-19T14:14:57Z',
            'Closeout identity changed')
    require(co['v9_freeze_sha256'] == V9_FREEZE and co['held_test_read'] is False, 'Closeout freeze/data role mismatch')
    report_text = src.text(BUNDLE / 'V9_CLOSEOUT_CN.md', co['report_sha256'], 'CLOSEOUT.json report_sha256')
    frozen = src.json(V9 / 'FROZEN.json', V9_FREEZE, 'CLOSEOUT.json v9_freeze_sha256')
    groups = src.json(V9 / 'GROUPS.json', frozen['code_sha256']['GROUPS.json'], 'FROZEN.json code_sha256')
    protocol = src.text(V9 / 'PROTOCOL_CN.md', frozen['code_sha256']['PROTOCOL_CN.md'], 'FROZEN.json code_sha256')
    collapse = load_frozen_collapse(src, frozen)
    sbatch = src.text(V9 / 'full.sbatch', frozen['code_sha256']['full.sbatch'], 'FROZEN.json code_sha256')
    require(sbatch.count('\n#SBATCH --gres=gpu:1\n') == 1, 'Frozen group job no longer requests exactly one GPU')
    return co, report_text, frozen, groups, protocol, collapse, sbatch


def verify_unit(src, co_unit, ro_unit, group, index, unit, collapse, protocol_sha):
    p = V9 / 'results' / group['id'] / f"{unit['method']}_{unit['scale']}"
    route = 'CLOSEOUT.json units[].full_result_sha256'
    fr = src.json(p / 'full_result.json', co_unit['full_result_sha256'], route)
    chain = 'full_result.json artifact_sha256 (closeout chain)'
    log = src.json(p / 'train_log.json', fr['artifact_sha256']['train_log.json'], chain)
    manifest = src.json(p / 'run_manifest.json', fr['artifact_sha256']['run_manifest.json'], chain)
    ident = src.json(p / 'identity.json', fr['artifact_sha256']['identity.json'], chain)
    key = f"{group['id']}/{unit['method']}_{unit['scale']}"
    progress_text = src.text(p / 'progress.jsonl', PROGRESS_SHA[key], 'sha256 pinned in addendum.py (progress.jsonl)')

    expected = {'group': group['id'], 'seed': group['seed'], 'method': unit['method'], 'scale': unit['scale'],
                'epochs': EPOCHS, 'freeze_sha256': V9_FREEZE, 'protocol_sha256': protocol_sha}
    require(fr['identity'] == ident and all(ident.get(k) == v for k, v in expected.items()), f'Identity mismatch: {key}')
    require(fr['test_read'] is False and fr['full_epochs'] == EPOCHS and fr['done']['epochs_run'] == EPOCHS, f'Role/length: {key}')
    require(fr['done']['manifest_sha256'] == fr['artifact_sha256']['run_manifest.json'], f'Done manifest mismatch: {key}')
    require(manifest['seed'] == group['seed'] and manifest['n_pool'] == fr['n_pool'], f'Manifest mismatch: {key}')
    epochs = log['epochs']
    require([e['epoch'] for e in epochs] == list(range(EPOCHS)), f'Not 40 epochs: {key}')
    rsum = [float(e['dev']['rsum']) for e in epochs]
    require(all(math.isfinite(x) for x in rsum), f'Nonfinite DEV: {key}')
    require([e['phase'] for e in epochs] == ['warmup_mean_hinge'] * 5 + ['max_violation'] * 35, f'Phase schedule: {key}')
    require(all(e['lr'] == epochs[0]['lr'] for e in epochs[:30]) and all(e['lr'] == epochs[0]['lr'] / 10 for e in epochs[30:]),
            f'Learning-rate decay is not at epoch 30: {key}')
    steps = {e['steps_in_epoch'] for e in epochs}
    require(len(steps) == 1, f'Varying steps per epoch: {key}')
    best_epoch = max(range(EPOCHS), key=lambda i: rsum[i])  # earliest tie, as the frozen selection
    require(fr['done']['best'] == {'rsum': rsum[best_epoch], 'epoch': best_epoch}, f'Best selection mismatch: {key}')
    resumed = log.get('resumed_at', [])
    boundaries = [0] + resumed + [EPOCHS]
    require(boundaries == sorted(boundaries) and len(set(boundaries)) == len(boundaries), f'Resume boundaries: {key}')
    measured = sum(epochs[end - 1]['seconds'] for start, end in zip(boundaries, boundaries[1:-1]) if end > start) + fr['done']['seconds']
    c = collapse(log)
    closeout_view = {'best_dev_rsum': rsum[best_epoch], 'best_epoch': best_epoch, 'last_dev_rsum': rsum[-1],
                     'gpu': fr['hardware']['gpu'], 'host': fr['hardware']['host'], 'measured_training_seconds': measured,
                     'resumed_at': resumed, 'full_result_sha256': co_unit['full_result_sha256']}
    require(all(co_unit[k] == v for k, v in closeout_view.items()), f'Recomputed values differ from CLOSEOUT.json units: {key}')
    require(ro_unit['status'] == 'COMPLETE' and ro_unit['best_dev_rsum'] == rsum[best_epoch] and ro_unit['best_epoch'] == best_epoch
            and ro_unit['n_pool'] == fr['n_pool'] and ro_unit['gpu'] == fr['hardware']['gpu']
            and ro_unit['severe_collapse'] == c['severe_collapse'], f'Recomputed values differ from CLOSEOUT.json readout: {key}')

    progress = [json.loads(line) for line in progress_text.splitlines() if line.strip()]
    require([x['epoch'] for x in progress] == list(range(EPOCHS)), f'progress.jsonl epochs: {key}')
    require(all(x['dev_rsum'] == rsum[i] and x['phase'] == epochs[i]['phase'] and x['steps_in_epoch'] == epochs[i]['steps_in_epoch']
                for i, x in enumerate(progress)), f'progress.jsonl disagrees with train_log.json: {key}')
    windows = {w['epoch']: w for w in c['windows']}
    require(sorted(windows) == list(range(5, EPOCHS)), f'Collapse windows: {key}')
    by_epoch = {}
    for st in log['steps']:
        by_epoch.setdefault(st['epoch'], []).append(st['step'])
    intervals = {b - a for v in by_epoch.values() for a, b in zip(v, v[1:])}
    require(sorted(by_epoch) == list(range(EPOCHS)) and all(v == sorted(v) for v in by_epoch.values()), f'Step log: {key}')
    return {'group': group['id'], 'seed': group['seed'], 'method': unit['method'], 'scale': unit['scale'], 'unit_index': index,
            'path': p, 'fr': fr, 'log': log, 'manifest': manifest, 'rsum': rsum, 'best_epoch': best_epoch, 'resumed_at': resumed,
            'boundaries': boundaries, 'measured_seconds': measured, 'collapse': c, 'windows': windows,
            'utc': [parse_utc(x['utc']) for x in progress], 'steps_in_epoch': steps.pop(), 'progress_sha256': PROGRESS_SHA[key],
            'progress_lines': progress_text.splitlines(), 'log_intervals': intervals}


def job_windows(src, co):
    """Known start / end of each GPU job, from controller records and the closeout's exit records."""
    win, exits = {}, {}

    def put_exit(job, code, utc, source):
        rec = {'exit_code': code, 'utc': utc, 'source': source}
        require(exits.get(job, rec)['exit_code'] == code, f'Conflicting exit codes for job {job}')
        exits.setdefault(job, rec)

    def put(job, side, value, source):
        entry = win.setdefault(job, {})
        if side in entry:
            require(entry[side][0] == value, f'Conflicting {side} records for job {job}')
        else:
            entry[side] = (value, source)

    for job, obs in src.json(BUNDLE / 'JOB_OBSERVATIONS.json').items():
        require(obs['id'] == job, 'Observation key mismatch')
        if obs['start'] not in ('N/A', 'Unknown'):
            put(job, 'start', parse_utc(obs['start']), 'JOB_OBSERVATIONS.json start (squeue %S)')
    for group, rec in co['final_scheduler_exits'].items():
        if rec.get('aggregate'):
            for exit_rec in rec['unit_exits'].values():
                put(exit_rec['job_id'], 'end', parse_utc(exit_rec['utc']), 'CLOSEOUT.json final_scheduler_exits unit_exits (job trap)')
                put_exit(exit_rec['job_id'], exit_rec['exit_code'], exit_rec['utc'], 'CLOSEOUT.json final_scheduler_exits unit_exits')
        else:
            put(rec['job_id'], 'end', parse_utc(rec['utc']), 'CLOSEOUT.json final_scheduler_exits (job trap)')
            put_exit(rec['job_id'], rec['exit_code'], rec['utc'], 'CLOSEOUT.json final_scheduler_exits')
    scontrol = {}
    for path in (SPLIT / 'transitions/903_1/CONTROLLER_ORIGINAL.json', SPLIT / 'transitions/903_2/CONTROLLER_ORIGINAL.json',
                 SPLIT / 'TIMEOUT_GUARD_CONTROLLER.json'):
        rec = src.json(path)
        require(rec['query_timezone'] == 'UTC', f'Unexpected scontrol timezone: {path}')
        fields = parse_scontrol(rec['stdout'])
        scontrol[fields['JobId']] = dict(fields, record=str(path))
        put(fields['JobId'], 'start', parse_utc(fields['StartTime']), f'{path.name} StartTime (scontrol, UTC)')
        put(fields['JobId'], 'end', parse_utc(fields['EndTime']), f'{path.name} EndTime (scontrol, UTC)')
    guard = src.json(SPLIT / 'TIMEOUT_GUARD_RESULT.json')['exit']
    require(guard['source'] == 'SLURM_CONTROLLER_TIMEOUT_NOT_BATCH_TRAP' and guard['slurm_state'] == 'TIMEOUT', 'Timeout guard record')
    put(guard['job_id'], 'end', parse_utc(guard['utc']), 'TIMEOUT_GUARD_RESULT.json exit (controller TIMEOUT)')
    put_exit(guard['job_id'], guard['exit_code'], guard['utc'], 'TIMEOUT_GUARD_RESULT.json')
    return win, scontrol, guard, exits


def build_segments(src, units, co, scontrol, guard, exits):
    by_key = {(u['group'], u['method'], u['scale']): u for u in units}
    ledger = src.json(SPLIT / 'UNIT_LEDGER.json')
    migrated = {g: src.json(SPLIT / f'transitions/{g}/MIGRATED.json') for g in ('903_1', '903_2')}
    for u in units:
        launch = co['launch'][u['group']]
        hw = u['fr']['hardware']
        segs = [{'first_epoch': a, 'last_epoch': b - 1, 'n_epochs': b - a} for a, b in zip(u['boundaries'], u['boundaries'][1:])]
        epochs = u['log']['epochs']
        for k, seg in enumerate(segs):
            seg['seconds'] = u['fr']['done']['seconds'] if k == len(segs) - 1 else epochs[seg['last_epoch']]['seconds']
        segs[-1].update(job_id=hw['job_id'], host=hw['host'], gpu=hw['gpu'], hardware_source='full_result.json hardware')
        u['attempts'] = []
        if launch.get('split'):
            require(launch['unit_jobs'][str(u['unit_index'])] == hw['job_id'], f"Split unit job mismatch: {u['group']}")
            entry = ledger[u['group']][str(u['unit_index'])]
            require(entry['job'] == hw['job_id'] and entry['status'] == 'COMPLETE_VERIFIED'
                    and entry['completion']['result']['full_result_sha256'] == u['fr_sha'],
                    f"UNIT_LEDGER disagrees: {u['group']}")
            hw_file = src.json(SPLIT / f"units/{u['group']}/{u['unit_index']}/hardware_{hw['job_id']}.json")
            require(hw_file == hw, f"Split hardware record differs from full_result: {u['group']}")
            segs[-1]['hardware_source'] = f"full_result.json hardware = units/{u['group']}/{u['unit_index']}/hardware_{hw['job_id']}.json"
            for h in entry.get('history', []):
                failed = src.json(SPLIT / f"units/{u['group']}/{u['unit_index']}/FAILED_{h['job']}.json")
                exit_rec = src.json(SPLIT / f"units/{u['group']}/{u['unit_index']}/EXIT_{h['job']}.json")
                fhw = src.json(SPLIT / f"units/{u['group']}/{u['unit_index']}/hardware_{h['job']}.json")
                diagnosis = src.json(ESS / 'DIAGNOSIS.json')
                require(h['exit'] == exit_rec and exit_rec['exit_code'] != 0 and failed['job_id'] == h['job'], 'Failed attempt record')
                require(diagnosis['status'] == 'METADATA_ONLY_FAILURE_BEFORE_TRAINING'
                        and diagnosis['completed_epochs'] == u['resumed_at'][0], 'Failed attempt trained epochs')
                u['attempts'].append({'job_id': h['job'], 'host': fhw['host'], 'gpu': fhw['gpu'], 'epochs_trained': 0,
                                      'exit_code': exit_rec['exit_code'], 'kind': 'failed_before_training',
                                      'diagnosis_status': diagnosis['status'], 'changed_fields': diagnosis['changed_fields'],
                                      'difference_ulps': diagnosis['difference_ulps'],
                                      'checkpoints_unchanged': diagnosis['last_and_best_checkpoints_unchanged'],
                                      'note': f"{diagnosis['status']}: changed {diagnosis['changed_fields']} by "
                                              f"{diagnosis['difference_ulps']:g} fp64 ULP; checkpoints unchanged"})
            is_migrated = u['group'] in migrated and migrated[u['group']]['checkpoint']['unit_index'] == u['unit_index']
            require((len(segs) == 2) == is_migrated, f"Split unit resume pattern: {u['group']}")
            if is_migrated:
                m = migrated[u['group']]
                require(len(segs) == 2 and m['checkpoint']['completed_epochs'] == u['resumed_at'][0], f"Migration boundary: {u['group']}")
                require(m['old_job'] == launch['original_serial_job'] and m['hardware_before']['job_id'] == m['old_job'], 'Old job')
                require(src.json(SPLIT / f"checkpoints_before_split/{u['group']}/hardware.json") == m['hardware_before'], 'Archive hw')
                require(m['physical_gpu_pairing_preserved'] is False, 'Migration pairing flag')
                for name in ('identity.json', 'run_manifest.json'):
                    require(m['artifact_sha256'][name] == u['fr']['artifact_sha256'][name], f'Archived {name} differs from the final unit')
                archived = src.text(SPLIT / f"checkpoints_before_split/{u['group']}/progress.jsonl", m['artifact_sha256']['progress.jsonl'],
                                    'MIGRATED.json artifact_sha256')
                lines = archived.splitlines()
                final_lines = u['progress_lines']
                require(len(lines) == u['resumed_at'][0] and final_lines[:len(lines)] == lines, 'Archived progress is not a prefix')
                ctl = scontrol[m['old_job']]
                require(ctl['NodeList'] == m['hardware_before']['host'] and PARTITION_GPU[ctl['Partition']] == m['hardware_before']['gpu'],
                        'Controller record disagrees with the archived hardware')
                old = m['hardware_before']
                segs[0].update(job_id=old['job_id'], host=old['host'], gpu=old['gpu'],
                               hardware_source=f"transitions/{u['group']}/MIGRATED.json hardware_before = checkpoints_before_split/{u['group']}/hardware.json")
                u['migration'] = {'old_job': m['old_job'], 'completed_epochs': m['checkpoint']['completed_epochs'],
                                  'archived_best': m['checkpoint']['best'], 'old_job_state': ctl['JobState'], 'old_job_exit': ctl['ExitCode'],
                                  'archived_progress_lines': len(lines), 'physical_gpu_pairing_preserved': m['physical_gpu_pairing_preserved']}
                u['attempts'].insert(0, {'job_id': m['old_job'], 'host': old['host'], 'gpu': old['gpu'], 'epochs_trained': segs[0]['n_epochs'],
                                         'exit_code': None, 'kind': 'migration_stop', 'controller_state': ctl['JobState'],
                                         'controller_exit': ctl['ExitCode'],
                                         'note': f"stopped for the split after a committed epoch (scontrol {ctl['JobState']}, ExitCode {ctl['ExitCode']})"})
        elif len(segs) == 2:
            prev = launch.get('previous_job')
            require(prev and launch['attempt'] == 2 and launch['history'][-1]['job'] == prev, f"No prior attempt record: {u['group']}")
            sibling = [v for v in units if v['group'] == u['group'] and v is not u and v['fr']['hardware']['job_id'] == prev]
            require(len(sibling) == 1, f"Prior-attempt hardware not recorded: {u['group']}")
            phw = sibling[0]['fr']['hardware']
            ctl = scontrol[prev]
            require(ctl['NodeList'] == phw['host'] and PARTITION_GPU[ctl['Partition']] == phw['gpu'] and guard['job_id'] == prev,
                    'Controller record disagrees with the prior-attempt hardware')
            segs[0].update(job_id=prev, host=phw['host'], gpu=phw['gpu'],
                           hardware_source=f"{u['group']}/{sibling[0]['method']}_{sibling[0]['scale']}/full_result.json hardware（同一job {prev}）; "
                                           f"TIMEOUT_GUARD_CONTROLLER.json NodeList/Partition")
            u['attempts'].append({'job_id': prev, 'host': phw['host'], 'gpu': phw['gpu'], 'epochs_trained': segs[0]['n_epochs'],
                                  'exit_code': guard['exit_code'], 'kind': 'timeout', 'controller_state': ctl['JobState'],
                                  'controller_exit': ctl['ExitCode'], 'time_limit': ctl['TimeLimit'],
                                  'note': f"{ctl['JobState']} at TimeLimit {ctl['TimeLimit']}; resumed once unchanged as attempt 2"})
        else:
            require(len(segs) == 1, f"Unexpected resume pattern: {u['group']}")
            jobs = {launch['job']} | {h['job'] for h in launch.get('history', [])}
            require(hw['job_id'] in jobs, f"Unit job is not a recorded attempt of its group: {u['group']}")
        require(hw['job_id'] in exits, f"No exit record for job {hw['job_id']}")
        job_exit = exits[hw['job_id']]['exit_code']
        u['attempts'].append({'job_id': hw['job_id'], 'host': hw['host'], 'gpu': hw['gpu'], 'epochs_trained': segs[-1]['n_epochs'],
                              'exit_code': job_exit, 'kind': 'completed',
                              'note': 'completed the unit' if job_exit == 0 else
                                      'completed the unit; the job later ended while training the next unit of its group'})
        if u['group'] == '903_1' and u['unit_index'] == 0:
            rec = src.json(ESS / 'RECOVERED_UNIT.json')
            require(rec['result']['full_result_sha256'] == u['fr_sha']
                    and rec['exit']['job_id'] == hw['job_id'] and rec['exit']['exit_code'] == 0, 'ESS recovery record')
        for seg in segs:
            short_gpu(seg['gpu'])
            seg['best_rsum_in_segment'] = max(u['rsum'][seg['first_epoch']:seg['last_epoch'] + 1])
            seg['best_epoch_in_segment'] = seg['first_epoch'] + u['rsum'][seg['first_epoch']:seg['last_epoch'] + 1].index(seg['best_rsum_in_segment'])
        u['segments'] = segs
        require(abs(sum(s['seconds'] for s in segs) - u['measured_seconds']) < 1e-6, 'Segment seconds do not add up')
        best_seg = [s for s in segs if s['first_epoch'] <= u['best_epoch'] <= s['last_epoch']][0]
        u['best_segment'] = best_seg
    for u in units:  # a unit may finish inside a job that later fails only if that job went on to train the next unit and ended there
        done = u['attempts'][-1]
        if done['exit_code'] != 0:
            later = [v for v in units if v['group'] == u['group'] and v['unit_index'] > u['unit_index']
                     and len(v['segments']) > 1 and v['segments'][0]['job_id'] == done['job_id']]
            require(len(later) == 1, f"Unit completed in a job without a successful exit: {u['group']}")
    return by_key, ledger, migrated


def check_time_windows(units, windows):
    checked, margins = 0, {'start': [], 'end': []}
    for u in units:
        prev_last = None
        for seg in u['segments']:
            times = u['utc'][seg['first_epoch']:seg['last_epoch'] + 1]
            require(times == sorted(times), f"Epoch timestamps not increasing: {unit_name(u)}")
            require(prev_last is None or prev_last < times[0], f"Segments overlap in time: {unit_name(u)}")
            prev_last = times[-1]
            w = windows.get(seg['job_id'], {})
            seg['job_start_utc'] = w['start'][0].strftime('%Y-%m-%dT%H:%M:%SZ') if 'start' in w else None
            seg['job_end_utc'] = w['end'][0].strftime('%Y-%m-%dT%H:%M:%SZ') if 'end' in w else None
            seg['job_start_source'] = w['start'][1] if 'start' in w else 'not recorded'
            seg['job_end_source'] = w['end'][1] if 'end' in w else 'not recorded'
            seg['start_margin_seconds'] = seg['end_margin_seconds'] = None
            if 'start' in w:
                margin = (times[0] - w['start'][0]).total_seconds()
                require(margin >= -TOLERANCE_S, f"Epoch before job start: {unit_name(u)}")
                seg['start_margin_seconds'] = margin
                margins['start'].append((margin, unit_name(u), seg['job_id']))
                checked += 1
            if 'end' in w:
                margin = (w['end'][0] - times[-1]).total_seconds()
                require(margin >= -TOLERANCE_S, f"Epoch after job end: {unit_name(u)}")
                seg['end_margin_seconds'] = margin
                margins['end'].append((margin, unit_name(u), seg['job_id']))
                checked += 1
            seg['first_epoch_utc'] = times[0].strftime('%Y-%m-%dT%H:%M:%SZ')
            seg['last_epoch_utc'] = times[-1].strftime('%Y-%m-%dT%H:%M:%SZ')
    summary = {side: {'n': len(v), 'min_seconds': min(v)[0], 'unit': min(v)[1], 'job_id': min(v)[2]} for side, v in margins.items()}
    return checked, summary


# ---------------------------------------------------------------- analyses

def contrast_spec(key):
    left, right = key.split(':', 1)
    if right == 'SUM-MEAN':
        return (left, 'sum'), (left, 'mean')
    a, b = right.split('-', 1)
    full = {'VCS': 'VCS_Weight'}
    return (full.get(a, a), left), (full.get(b, b), left)


def endpoint_values(u):
    late = [u['rsum'][e] for e in LATE]
    return {'best': u['rsum'][u['best_epoch']], 'last': u['rsum'][-1], 'late_mean': statistics.mean(late)}


def contrasts_all(co, by_seed):
    out = []
    for key, closeout_c in co['readout']['contrasts'].items():
        a, b = contrast_spec(key)
        row = {'contrast': key, 'a': list(a), 'b': list(b)}
        for ep in ('best', 'last', 'late_mean'):
            pairs = [{'seed': s, 'delta': by_seed[(s, *a)]['endpoints'][ep] - by_seed[(s, *b)]['endpoints'][ep]} for s in (901, 902, 903)]
            ds = [p['delta'] for p in pairs]
            row[ep] = {'pairs': pairs, 'mean': statistics.mean(ds), 'positive': sum(d > 0 for d in ds), 'range': [min(ds), max(ds)]}
        require(row['best']['pairs'] == closeout_c['pairs'] and row['best']['mean'] == closeout_c['mean']
                and row['best']['positive'] == closeout_c['positive'], f'Best-endpoint contrast differs from CLOSEOUT.json: {key}')
        for ep in ('last', 'late_mean'):
            row[ep]['mean_sign_flip_vs_best'] = (row[ep]['mean'] > 0) != (row['best']['mean'] > 0)
        out.append(row)
    return out


def gates(co, contrasts, by_seed):
    """Re-expresses the frozen summarize.py decision on the closeout's own numbers; must reproduce it exactly."""
    c = {x['contrast']: x['best'] for x in contrasts}

    def collapse_count(method, scale):
        return sum(by_seed[(s, method, scale)]['collapse']['severe_collapse'] for s in (901, 902, 903))

    def passes(key, margin, a, b, scale):
        return c[key]['mean'] >= margin and c[key]['positive'] >= 2 and collapse_count(a, scale) <= collapse_count(b, scale)

    vcs_rows = []
    for other in ('BCE_Weight', 'RankMass', 'RuLSIF_Weight'):
        key = f'sum:VCS-{other}'
        d = {p['seed']: p['delta'] for p in c[key]['pairs']}
        other903 = by_seed[(903, other, 'sum')]['endpoints']['best']
        required = 3 * 3 - d[901] - d[902] + other903  # VCS 903 SUM making the 3-seed mean exactly +3 (others fixed)
        vcs_rows.append({'contrast': key, 'mean': c[key]['mean'], 'positive': c[key]['positive'], 'pairs': c[key]['pairs'],
                         'passes': passes(key, 3, 'VCS_Weight', other, 'sum'), 'other_903_sum_best': other903,
                         'vcs_903_sum_needed_for_mean_3': required,
                         'positive_rule_met_at_needed_value': (d[901] > 0) + (d[902] > 0) + (required - other903 > 0) >= 2})
    vcs = all(r['passes'] for r in vcs_rows)
    generic_rows = []
    for scale, methods in (('sum', ('VCS_Weight', 'BCE_Weight', 'RankMass', 'RuLSIF_Weight')), ('mean', ('VCS_Weight', 'BCE_Weight', 'RankMass'))):
        for m in methods:
            key = f'{scale}:VCS-Unweighted_A' if m == 'VCS_Weight' else f'{scale}:{m}-Unweighted_A'
            generic_rows.append({'scale': scale, 'contrast': key, 'mean': c[key]['mean'], 'positive': c[key]['positive'],
                                 'passes': passes(key, 5, m, 'Unweighted_A', scale)})
    generic = {s: any(r['passes'] for r in generic_rows if r['scale'] == s) for s in ('mean', 'sum')}
    dec = co['readout']['decision']
    require(dec['vcs_specific_confirmation_worth_testing'] == vcs and dec['generic_weighting_gate'] == generic
            and dec['stop_current_route'] == (not any(generic.values())), 'Gate re-expression does not reproduce CLOSEOUT.json')
    binding = max(vcs_rows, key=lambda r: r['vcs_903_sum_needed_for_mean_3'])
    return {'vcs_specific': vcs_rows, 'vcs_specific_passes': vcs, 'binding': binding, 'vcs_903_sum_final': by_seed[(903, 'VCS_Weight', 'sum')]['endpoints']['best'],
            'generic': generic_rows, 'generic_passes': generic, 'closeout_decision': dec}


def pairing(units, by_key, groups):
    rows = []
    for g in groups:
        if len(g['units']) != 2:
            continue
        a, b = (by_key[(g['id'], x['method'], x['scale'])] for x in g['units'])
        if a['method'] == b['method']:
            s_u, m_u = (a, b) if a['scale'] == 'sum' else (b, a)
            kind = 'SUM-MEAN'
        else:
            s_u, m_u = (a, b) if a['method'] == 'OracleDrop' else (b, a)
            kind = 'OracleDrop-OracleMask'
        jobs_s = {s['job_id'] for s in s_u['segments']}
        jobs_m = {s['job_id'] for s in m_u['segments']}
        shared = jobs_s & jobs_m
        models = {s['gpu'] for s in s_u['segments']} | {s['gpu'] for s in m_u['segments']}
        hosts = {s['host'] for s in s_u['segments']} | {s['host'] for s in m_u['segments']}
        strict = len(s_u['segments']) == 1 and len(m_u['segments']) == 1 and jobs_s == jobs_m
        if strict:
            status, label = 'same_job', '同一job、同一分配GPU（顺序运行）'
        elif len(models) > 1:
            status, label = 'different_gpu_models', 'GPU型号不同'
        elif shared:
            status, label = 'partly_same_job', '部分epoch同job；其余在另一分配（型号、节点相同，物理GPU编号未记录）'
        elif len(hosts) > 1:
            status, label = 'different_jobs_and_nodes', '不同job与节点，型号相同'
        else:
            status, label = 'different_jobs_same_node', '不同job、同节点同型号（物理GPU编号未记录）'
        if strict:
            order = [g['units'].index({'method': x['method'], 'scale': x['scale']}) for x in (a, b)]
            first, second = (a, b) if order[0] < order[1] else (b, a)
            require(first['utc'][-1] < second['utc'][0], f"Same-job pair is not sequential: {g['id']}")
        shared_epochs = {('first' if u is s_u else 'second'): sum(s['n_epochs'] for s in u['segments'] if s['job_id'] in shared) for u in (s_u, m_u)}
        rows.append({'group': g['id'], 'seed': g['seed'], 'kind': kind, 'method': s_u['method'] if kind == 'SUM-MEAN' else 'Oracle',
                     'first': s_u, 'second': m_u, 'status': status, 'label': label, 'strict_same_job': strict,
                     'epochs_in_shared_job': shared_epochs, 'gpu_models': sorted(models), 'hosts': sorted(hosts),
                     'delta_best': s_u['endpoints']['best'] - m_u['endpoints']['best']})
    return rows


def collapse_disclosure(units):
    rows = []
    for u in units:
        base = u['rsum'][4]
        hard = [(e, u['rsum'][e] / base) for e in range(5, EPOCHS)]
        min_epoch, min_ratio = min(hard, key=lambda t: (t[1], t[0]))
        w = u['windows']
        cond_a = [e for e in range(5, EPOCHS) if w[e]['rsum'] < 0.5 * base]
        cond_b = [e for e in range(5, EPOCHS) if w[e]['locked_share'] is not None and w[e]['locked_share'] >= 0.9]
        below = [e for e in range(5, EPOCHS) if u['rsum'][e] < base]
        u['disclosure'] = {'epoch4_rsum': base, 'min_hard_ratio': min_ratio, 'min_hard_epoch': min_epoch,
                           'epochs_below_epoch4': below, 'epochs_rsum_below_half_epoch4': cond_a, 'epochs_locked_share_ge_0p9': cond_b,
                           'flag_epochs': u['collapse']['flag_epochs'], 'severe_collapse': u['collapse']['severe_collapse'],
                           'max_hard_locked_share': max(w[e]['locked_share'] for e in range(5, EPOCHS) if w[e]['locked_share'] is not None),
                           'n_logged_steps_per_window': sorted({w[e]['n_logged_steps'] for e in range(5, EPOCHS)})}
        if below or cond_a or cond_b:
            rows.append(u)
    return rows


def wave1(src, frozen, collapse, by_seed):
    formal = src.json(REPORTS_V9 / 'V9_N2_units/formal_units.json')
    tuning = src.json(REPORTS_V9 / 'V9_N2_units/tuning_units.json')
    root = Path(frozen['old_output']).parent  # FROZEN.json old_output = <out>/f30k; unit_dir is relative to <out>
    specs = [src.json(REPORTS_V9 / f'V9_N2_diag/spec_job{i}.json') for i in (1, 2)]
    results = [src.json(REPORTS_V9 / f'V9_N2_diag/spec_job{i}.results.json') for i in (1, 2)]
    require(specs[0]['rule'] == specs[1]['rule'], 'A/B specs disagree on the readout rule')
    ab = []
    for spec, res in zip(specs, results):
        require([c['id'] for c in spec['configs']] == [r['id'] for r in res], 'A/B results do not match their spec')
        for cfg, r in zip(spec['configs'], res):
            require(all(r[k] == cfg[k] for k in ('rho', 'seed', 'method', 'lr_mult', 'loss_scale')), 'A/B config mismatch')
            ab.append(r)
    rows = []
    for seed, method, lr, kind, census in WAVE1_CENSUS:
        pool = formal['runs'] if kind == 'formal' else tuning['runs']
        hit = [r for r in pool if r['rho'] == 0.4 and r['seed'] == seed and r['method'] == method and r['lr_mult'] == lr]
        require(len(hit) == 1, f'Wave-1 unit not in its unit list: {seed} {method} x{lr}')
        d = root / hit[0]['unit_dir']
        log = src.json(d / 'train_log.json')
        manifest = src.json(d / 'run_manifest.json')
        done = src.json(d / 'done.json')
        require(done['manifest_sha256'] == PINNED[d / 'run_manifest.json'], 'Wave-1 done/manifest chain')
        rsum = [float(e['dev']['rsum']) for e in log['epochs']]
        require([e['epoch'] for e in log['epochs']] == list(range(EPOCHS)) and done['epochs_run'] == EPOCHS, 'Wave-1 length')
        best_epoch = max(range(EPOCHS), key=lambda i: rsum[i])
        require(done['best'] == {'rsum': rsum[best_epoch], 'epoch': best_epoch}, 'Wave-1 best')
        c = collapse(log)
        hard = [(e, rsum[e] / rsum[4]) for e in range(5, EPOCHS)]
        min_epoch, min_ratio = min(hard, key=lambda t: (t[1], t[0]))
        row = {'seed': seed, 'method': method, 'lr_mult': lr, 'kind': kind, 'census': census, 'unit_dir': hit[0]['unit_dir'],
               'rsum': rsum, 'best_epoch': best_epoch, 'best': rsum[best_epoch], 'last': rsum[-1], 'epoch4': rsum[4],
               'min_hard_ratio': min_ratio, 'min_hard_epoch': min_epoch, 'severe_collapse_frozen_rule': c['severe_collapse'],
               'n_flag_epochs': len(c['flag_epochs']), 'first_flag_epoch': c['flag_epochs'][0] if c['flag_epochs'] else None,
               'locked_share_e5_e6': [c['windows'][0]['locked_share'], c['windows'][1]['locked_share']]}
        require((census == 'collapsed') == c['severe_collapse'], 'Frozen rule disagrees with the wave-1 census')
        n2f = by_seed.get((seed, method, 'mean')) if lr == 1.0 else None
        if n2f is not None:
            diff = {}
            for k in sorted(set(manifest) | set(n2f['manifest'])):
                x, y = manifest.get(k), n2f['manifest'].get(k)
                if x != y:
                    diff[k] = {'wave1': x, 'n2f': y, 'fp64_ulps': ulps(x, y) if isinstance(x, float) and isinstance(y, float) else None}
            require(set(diff) <= {'formal', 'mean_v_full_pool'}, f'Unexpected wave-1/N2F manifest difference: {sorted(diff)}')
            row['n2f_rerun'] = {'group': n2f['group'], 'manifest_differences': diff, 'hardware': n2f['fr']['hardware']}
        cfg = [r for r in ab if r['seed'] == seed and r['method'] == method and r['lr_mult'] == lr]
        row['ab'] = cfg
        rows.append(row)
    aa = [r for r in ab if r['id'].startswith('AA_')]
    sums = [r for r in ab if r['loss_scale'] == 'sum']
    require(len(aa) == 1 and len(sums) == 4, 'A/B design')
    verdict = {'aa_valid': aa[0]['collapse_present'], 'sum_configs_with_collapse': sum(r['collapse_present'] for r in sums),
               'n_sum_configs': len(sums)}
    require(all(r['steps_after_switch'] == specs[0]['rule']['stop_steps'] for r in ab), 'A/B runs stopped at a different step')
    return rows, ab, verdict, specs[0]['rule']


def cost(units, protocol_text):
    q = quote_span(protocol_text, V9 / 'PROTOCOL_CN.md', '按实测单次成本 RTX', '确定性运行开销另计）')
    m = re.search(r'RTX (\d+\.\d+)h / L40S (\d+\.\d+)h / H100 NVL (\d+\.\d+)h，本矩阵约 (\d+)–(\d+) GPU-h（OracleDrop 按约(\d+)%训练池估计', q['text'])
    require(m is not None, 'Protocol budget sentence changed')
    per_unit = {'RTX': float(m.group(1)), 'L40S': float(m.group(2)), 'H100 NVL': float(m.group(3))}
    lo, hi, drop = int(m.group(4)), int(m.group(5)), int(m.group(6)) / 100
    n_full = sum(u['method'] != 'OracleDrop' for u in units)
    n_drop = sum(u['method'] == 'OracleDrop' for u in units)
    equivalents = n_full + n_drop * drop
    rec_lo, rec_hi = equivalents * min(per_unit.values()), equivalents * max(per_unit.values())
    require(round(rec_lo) == lo and round(rec_hi) == hi, 'Cannot reconstruct the protocol range')
    by_model, durations, planned_mix = {}, {}, 0.0
    for u in units:
        factor = drop if u['method'] == 'OracleDrop' else 1.0
        for seg in u['segments']:
            label = PROTOCOL_COST_LABEL[seg['gpu']]
            by_model[label] = by_model.get(label, 0.0) + seg['seconds']
            planned_mix += seg['n_epochs'] / EPOCHS * per_unit[label] * factor
            prev = 0.0
            for e in range(seg['first_epoch'], seg['last_epoch'] + 1):
                cum = u['log']['epochs'][e]['seconds']
                key = (label, seg['host'], u['method'] == 'OracleDrop')
                durations.setdefault(key, []).append(cum - prev)
                prev = cum
    measured_total = sum(u['measured_seconds'] for u in units)
    rows = []
    for (label, host, is_drop), vals in sorted(durations.items()):
        med = statistics.median(vals)
        planned_epoch = per_unit[label] * 3600 / EPOCHS * (drop if is_drop else 1.0)
        rows.append({'model': label, 'host': host, 'oracle_drop': is_drop, 'epochs': len(vals), 'median_epoch_seconds': med,
                     'forty_epoch_hours': med * EPOCHS / 3600, 'planned_unit_hours': planned_epoch * EPOCHS / 3600,
                     'ratio_to_planned': med / planned_epoch})
    return {'quote': q, 'per_unit_hours': per_unit, 'range': [lo, hi], 'oracle_drop_factor': drop, 'n_full_pool_units': n_full,
            'n_oracle_drop_units': n_drop, 'unit_equivalents': equivalents, 'reconstructed_range': [rec_lo, rec_hi],
            'measured_hours_by_model': {k: v / 3600 for k, v in sorted(by_model.items())}, 'measured_hours_total': measured_total / 3600,
            'planned_hours_at_actual_mix': planned_mix, 'epoch_durations': rows}


# ---------------------------------------------------------------- outputs

def csv_text(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(header)

    def cell(v):
        if v is None:
            return ''
        if isinstance(v, bool):
            return 'true' if v else 'false'
        if isinstance(v, float):
            return repr(v)
        if isinstance(v, (list, tuple)):
            return '|'.join(cell(x) for x in v)
        return str(v)
    for r in rows:
        w.writerow([cell(r[h]) for h in header])
    return buf.getvalue()


def unit_row(u):
    best = u['log']['epochs'][u['best_epoch']]['dev']
    d = u['disclosure']
    late = [u['rsum'][e] for e in LATE]
    return {
        'group': u['group'], 'seed': u['seed'], 'method': u['method'], 'scale': u['scale'], 'unit_index': u['unit_index'],
        'best_dev_rsum': u['endpoints']['best'], 'best_epoch': u['best_epoch'], 'last_dev_rsum': u['endpoints']['last'],
        'mean_dev_rsum_e30_39': u['endpoints']['late_mean'], 'max_minus_median_e30_39': max(late) - statistics.median(late),
        'best_minus_mean_e30_39': u['endpoints']['best'] - u['endpoints']['late_mean'],
        'best_i2t_R1': best['i2t_R1'], 'best_i2t_R5': best['i2t_R5'], 'best_i2t_R10': best['i2t_R10'],
        'best_t2i_R1': best['t2i_R1'], 'best_t2i_R5': best['t2i_R5'], 'best_t2i_R10': best['t2i_R10'], 'best_mR': best['mR'],
        'epoch4_dev_rsum': d['epoch4_rsum'], 'min_hard_ratio_to_epoch4': d['min_hard_ratio'], 'min_hard_epoch': d['min_hard_epoch'],
        'max_hard_locked_share': d['max_hard_locked_share'], 'collapse_flag_epochs': d['flag_epochs'], 'severe_collapse': d['severe_collapse'],
        'n_pool': u['fr']['n_pool'], 'steps_in_epoch': u['steps_in_epoch'], 'n_segments': len(u['segments']),
        'segments': [f"e{s['first_epoch']}-{s['last_epoch']}:{s['job_id']}:{s['host']}:{s['gpu']}" for s in u['segments']],
        'gpu_models': sorted({s['gpu'] for s in u['segments']}), 'job_ids': [a['job_id'] for a in u['attempts']],
        'best_epoch_job': u['best_segment']['job_id'], 'best_epoch_host': u['best_segment']['host'], 'best_epoch_gpu': u['best_segment']['gpu'],
        'closeout_gpu': u['fr']['hardware']['gpu'], 'closeout_host': u['fr']['hardware']['host'], 'resumed_at': u['resumed_at'],
        'measured_training_seconds': u['measured_seconds'], 'segment_seconds': [s['seconds'] for s in u['segments']],
        'full_result_sha256': u['fr_sha'], 'train_log_sha256': u['fr']['artifact_sha256']['train_log.json'],
    }


UNIT_COLUMNS = ['group', 'seed', 'method', 'scale', 'unit_index', 'best_dev_rsum', 'best_epoch', 'last_dev_rsum', 'mean_dev_rsum_e30_39',
                'max_minus_median_e30_39', 'best_minus_mean_e30_39', 'best_i2t_R1', 'best_i2t_R5', 'best_i2t_R10', 'best_t2i_R1',
                'best_t2i_R5', 'best_t2i_R10', 'best_mR', 'epoch4_dev_rsum', 'min_hard_ratio_to_epoch4', 'min_hard_epoch',
                'max_hard_locked_share', 'collapse_flag_epochs', 'severe_collapse', 'n_pool', 'steps_in_epoch', 'n_segments', 'segments',
                'gpu_models', 'job_ids', 'best_epoch_job', 'best_epoch_host', 'best_epoch_gpu', 'closeout_gpu', 'closeout_host', 'resumed_at',
                'measured_training_seconds', 'segment_seconds', 'full_result_sha256', 'train_log_sha256']
TRAJ_COLUMNS = ['group', 'seed', 'method', 'scale', 'epoch', 'phase', 'lr', 'dev_rsum', 'dev_mR', 'ratio_to_epoch4', 'is_best_epoch',
                'mean_u_risk_unnormalised', 'locked_share_last200', 'n_logged_steps_last200', 'frozen_collapse_window_flag',
                'segment', 'job_id', 'host', 'gpu', 'epoch_seconds', 'epoch_utc']


def trajectory_rows(units):
    rows = []
    for u in units:
        base = u['rsum'][4]
        for k, seg in enumerate(u['segments']):
            prev = 0.0
            for e in range(seg['first_epoch'], seg['last_epoch'] + 1):
                ep = u['log']['epochs'][e]
                w = u['windows'].get(e)
                rows.append({'group': u['group'], 'seed': u['seed'], 'method': u['method'], 'scale': u['scale'], 'epoch': e,
                             'phase': ep['phase'], 'lr': ep['lr'], 'dev_rsum': u['rsum'][e], 'dev_mR': ep['dev']['mR'],
                             'ratio_to_epoch4': u['rsum'][e] / base, 'is_best_epoch': e == u['best_epoch'],
                             'mean_u_risk_unnormalised': ep['mean_u_risk_unnormalised'],
                             'locked_share_last200': w['locked_share'] if w else None, 'n_logged_steps_last200': w['n_logged_steps'] if w else None,
                             'frozen_collapse_window_flag': w['flag'] if w else None, 'segment': k, 'job_id': seg['job_id'],
                             'host': seg['host'], 'gpu': seg['gpu'], 'epoch_seconds': ep['seconds'] - prev,
                             'epoch_utc': u['utc'][e].strftime('%Y-%m-%dT%H:%M:%SZ')})
                prev = ep['seconds']
    return rows


def md_table(header, rows, align=None):
    align = align or ['---'] * len(header)
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(align) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return out


def blockquote(q):
    where = f"`{q['path']}` 字段 `{q['field']}`" if 'field' in q else f"`{q['path']}:{q['line']}`"
    return [f"> {q['text']}", '>', f"> —— {where}"]


def build(src):
    co, report_text, frozen, groups, protocol, collapse, sbatch = load_records(src)
    co_units = {(u['group'], u['method'], u['scale']): u for u in co['units']}
    ro_units = {(u['group'], u['method'], u['scale']): u for u in co['readout']['units']}
    require(len(co_units) == 33 and len(ro_units) == 33 and frozen['full_units'] == 33, 'Closeout must hold 33 distinct units')
    units = []
    for g in groups:
        for i, unit in enumerate(g['units']):
            key = (g['id'], unit['method'], unit['scale'])
            units.append(verify_unit(src, co_units[key], ro_units[key], g, i, unit, collapse, frozen['code_sha256']['PROTOCOL_CN.md']))
    require(len(units) == 33, 'Expected 33 units')
    for u in units:
        u['fr_sha'] = co_units[(u['group'], u['method'], u['scale'])]['full_result_sha256']
        u['endpoints'] = endpoint_values(u)
    windows, scontrol, guard, exits = job_windows(src, co)
    by_key, ledger, migrated = build_segments(src, units, co, scontrol, guard, exits)
    n_window_checks, window_margins = check_time_windows(units, windows)
    require(abs(sum(u['measured_seconds'] for u in units) / 3600 - co['cost']['measured_training_gpu_hours']) < 1e-9, 'Cost total')
    by_seed = {(u['seed'], u['method'], u['scale']): u for u in units}
    contrasts = contrasts_all(co, by_seed)
    gate = gates(co, contrasts, by_seed)
    pairs = pairing(units, by_key, groups)
    flagged = collapse_disclosure(units)
    w1_rows, ab_rows, ab_verdict, ab_rule = wave1(src, frozen, collapse, by_seed)
    cost_info = cost(units, protocol)

    # verbatim quotes (each verified to occur exactly once in its source)
    pp = V9 / 'PROTOCOL_CN.md'
    cp = BUNDLE / 'V9_CLOSEOUT_CN.md'
    split_readme = src.text(SPLIT / 'README_CN.md')
    yaml_path = REPORTS_V9 / 'V9_N2_PROTOCOL_FROZEN_20261015.yaml'
    yaml_text = src.text(yaml_path)
    paused = src.json(REPORTS_V9 / 'V9_N2_units/wave1_paused.json')
    amendment = src.json(SPLIT / 'SCHEDULING_AMENDMENT.json')
    Q = {
        'same_job': quote_span(protocol, pp, '每个 method/seed 的两个尺度在同一个 Slurm job、同一 GPU 上顺序运行'),
        'one_gpu': quote_span(sbatch, V9 / 'full.sbatch', '#SBATCH --gres=gpu:1', '#SBATCH --gres=gpu:1'),
        'determinism': quote_span(protocol, pp, '新成对实验启用 deterministic algorithms'),
        'trajectory': quote_span(protocol, pp, '主比较在每 seed 内报告'),
        'endpoint': quote_span(protocol, pp, '每 epoch 使用完整原 DEV 文件'),
        'no_significance': quote_span(protocol, pp, 'DEV 的多次选择与 n=3 不支持宣称统计显著'),
        'collapse_def': quote_span(protocol, pp, '预先定义“严重训练崩塌”', '报告实际抽样点数。'),
        'scale_rule': quote_span(protocol, pp, '投入决策是工程阈值而非统计显著性', '不取更有利者作为事先主分析'),
        'gate_vcs': quote_span(protocol, pp, 'VCS 专有路线只有在完整 3 seeds 中', '扩展。'),
        'gate_generic': quote_span(protocol, pp, '通用可靠性加权路线只有在', '不可把这里选择的优胜者当预先选定方法。'),
        'ab_history': quote_span(protocol, pp, '原 N2 在 rho=0.4 有多例 warmup→hard-negative 后崩塌'),
        'cause': quote_span(protocol, pp, '因此不能把完整训练成功归为某个已证明的无偏结论'),
        'c_pairing': quote_span(report_text, cp, 'SUM为预先固定主尺度', '方法之间也可能使用不同型号GPU。'),
        'c_descriptive': quote_span(report_text, cp, '以上三seed差值为描述性结果'),
        'c_vcs': quote_span(report_text, cp, 'VCS未同时达到相对BCE、RankMass、RuLSIF的原预设资源阈值', '不追加本路线网格。'),
        'c_generic': quote_span(report_text, cp, '通用加权资源阈值：', '崩塌不更多。'),
        'c_keep': quote_span(report_text, cp, '至少一个尺度支持通用加权的实际收益'),
        'c_collapse': quote_span(report_text, cp, '严重崩塌0/33；未解决失败0。', '未解决失败0。'),
        'c_cost': quote_span(report_text, cp, '实际训练计时累计', '不能冒充完整allocated GPU时长。'),
        'c_sacct': quote_span(report_text, cp, 'sacct会计服务当前不可用'),
        'split_readme': quote_span(split_readme, SPLIT / 'README_CN.md', 'SUM—MEAN仍可报告为固定seed敏感性结果'),
        'w1_census': yaml_value(yaml_text, yaml_path, 'observation', 'collapse census at the pause'),
        'w1_scales': yaml_value(yaml_text, yaml_path, 'scales', 'sum: batch loss'),
        'w1_aa_valid': yaml_value(yaml_text, yaml_path, 'aa_valid', 'the mean-scale rerun of C4'),
        'w1_verdict': yaml_value(yaml_text, yaml_path, 'verdict', 'scale_fix_supported if'),
        'w1_readout': yaml_value(yaml_text, yaml_path, 'readout_per_run', 'locked = U risk'),
    }
    paused_anchor = 'pause wave 1 after the collapse report (4 of 24 rho-0.4 SGR-student runs collapsed right after the epoch-5 switch)'
    require(paused['decision'].count(paused_anchor) == 1, 'wave1_paused.json decision changed')
    Q['w1_paused'] = {'text': paused_anchor, 'path': str(REPORTS_V9 / 'V9_N2_units/wave1_paused.json'), 'field': 'decision'}
    require(amendment['hardware'].startswith('Original allowed partitions'), 'Split amendment changed')
    Q['split_hw'] = {'text': amendment['hardware'], 'path': str(SPLIT / 'SCHEDULING_AMENDMENT.json'), 'field': 'hardware'}

    gen_bytes = Path(__file__).read_bytes()
    gen_sha = sha256_bytes(gen_bytes)
    unit_rows = [unit_row(u) for u in units]
    traj = trajectory_rows(units)
    csv_units = csv_text(UNIT_COLUMNS, unit_rows)
    csv_traj = csv_text(TRAJ_COLUMNS, traj)
    require(len(traj) == 33 * EPOCHS, 'Trajectory rows')

    payload = {
        'schema': 'cs_qmi.v9_n2f_closeout_addendum.v1',
        'description': 'Final N2F 33-unit table with per-segment hardware, rebuilt from records and verified against the closeout.',
        'generator': {'path': GENERATOR_REL, 'sha256': gen_sha},
        'closeout': {'path': str(BUNDLE / 'CLOSEOUT.json'), 'sha256': CLOSEOUT_SHA, 'status': co['status'], 'utc': co['utc'],
                     'job_id': co['job_id'], 'v9_freeze_sha256': co['v9_freeze_sha256'], 'report_sha256': co['report_sha256']},
        'scope': {'held_test_read': False, 'training_or_inference': False, 'checkpoints_loaded_or_rehashed': False,
                  'closeout_bundle_or_v9_code_modified': False, 'new_conclusions': False,
                  'endpoint': 'pre-registered best full-DEV rSum (earliest tie); last and mean(e30-39) are descriptive only'},
        'checks': {'units_verified_against_closeout': len(units), 'progress_epochs_matching_train_log': len(units) * EPOCHS,
                   'job_window_bounds_checked': n_window_checks, 'clock_tolerance_seconds': TOLERANCE_S,
                   'job_window_min_margins': window_margins, 'step_log_intervals': sorted({i for u in units for i in u['log_intervals']}),
                   'deterministic_and_tf32_off_units': sum(u['fr']['hardware']['deterministic_algorithms'] is True
                                                           and u['fr']['hardware']['tf32'] is False for u in units),
                   'best_endpoint_contrasts_equal_closeout': len(contrasts), 'gate_reexpression_equals_closeout': True},
        'columns': UNIT_COLUMNS,
        'units': [dict(r, segments_detail=[{k: v for k, v in s.items()} for s in u['segments']], attempts=u['attempts'],
                       dev_rsum_by_epoch=u['rsum']) for r, u in zip(unit_rows, units)],
        'derived': {
            'pairing': [{k: v for k, v in p.items() if k not in ('first', 'second')} | {'first_unit': unit_name(p['first']), 'second_unit': unit_name(p['second'])}
                        for p in pairs],
            'endpoint_sensitivity': contrasts,
            'gates_reexpressed': {k: v for k, v in gate.items() if k != 'closeout_decision'},
            'collapse_disclosure': [{'unit': unit_name(u), 'group': u['group'], **u['disclosure']} for u in flagged],
            'wave1_comparison': [{k: v for k, v in r.items() if k != 'rsum'} | {'dev_rsum_by_epoch': r['rsum']} for r in w1_rows],
            'ab_diagnostic': {'rule': ab_rule, 'results': ab_rows, 'verdict_inputs': ab_verdict},
            'cost': {k: v for k, v in cost_info.items() if k != 'quote'},
        },
    }
    payload['inputs'] = sorted(src.read_log.values(), key=lambda x: x['path'])
    json_text = json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False) + '\n'
    md = markdown(co, units, by_seed, pairs, contrasts, gate, flagged, w1_rows, ab_rows, ab_verdict, cost_info, Q, gen_sha,
                  {'N2F_FINAL_33_UNITS.csv': sha256_bytes(csv_units.encode()), 'N2F_FINAL_33_UNITS.json': sha256_bytes(json_text.encode()),
                   'N2F_TRAJECTORIES.csv': sha256_bytes(csv_traj.encode())}, len(src.read_log), n_window_checks, payload['checks'], ab_rule)
    return {'N2F_FINAL_33_UNITS.csv': csv_units, 'N2F_FINAL_33_UNITS.json': json_text, 'N2F_TRAJECTORIES.csv': csv_traj,
            'V9_CLOSEOUT_ADDENDUM_CN.md': md}, units


CENSUS_CN = {'collapsed': '崩塌', 'dipped and recovered': '下探后恢复'}
KIND_CN = {'formal': '正式', 'tuning': '调参'}


def markdown(co, units, by_seed, pairs, contrasts, gate, flagged, w1_rows, ab_rows, ab_verdict, cost_info, Q, gen_sha, out_sha,
             n_inputs, n_window_checks, checks, ab_rule):
    L = []
    add = L.append
    resumed = [u for u in units if len(u['segments']) > 1]
    split_fresh = [u for u in units if co['launch'][u['group']].get('split') and len(u['segments']) == 1]
    sm_pairs = [p for p in pairs if p['kind'] == 'SUM-MEAN']
    strict = [p for p in sm_pairs if p['strict_same_job']]
    nonstrict = [p for p in sm_pairs if not p['strict_same_job']]
    main = [u for u in units if not u['method'].startswith('Oracle')]
    oracle = [u for u in units if u['method'].startswith('Oracle')]
    n_severe = sum(u['collapse']['severe_collapse'] for u in units)
    wm = checks['job_window_min_margins']

    def rng(xs):
        return f"{f2(min(xs))}–{f2(max(xs))}"

    add('# V9 N2F 收尾附录：分段硬件、完整轨迹、崩塌披露、后期敏感性与成本')
    add('')
    add(f"本附录由 `{GENERATOR_REL}`（sha256 `{gen_sha}`）只读记录生成；文中数字均由该脚本计算，引文逐字取自所注文件与行号。"
        f"它补充收尾记录 `{BUNDLE / 'CLOSEOUT.json'}`（status {co['status']}，{co['utc']}，CPU作业 {co['job_id']}），"
        '不修改冻结收尾包、V9代码或任何实验产物；不训练、不读TEST、不加载或重算模型。收尾的预注册门槛与结论保持不变，本附录不增加新结论。')
    add('')
    add(f"补充内容对应独立审查 `{REVIEW}` §2.2 P1-3 与 §2.3 “V9”所列缺口：拆分/续跑单元只记录了终末段硬件、协议要求的完整40 epoch轨迹未输出、崩塌图景不完整、"
        '最佳epoch与后期波动、计划与实测成本。')
    add('')
    add('## 0 来源与核验')
    add('')
    add(f"- 收尾 `CLOSEOUT.json` sha256 `{CLOSEOUT_SHA}`；`V9_CLOSEOUT_CN.md` 的sha256等于其 `report_sha256`。")
    add(f"- {len(units)}/33 个单元：`full_result.json` 的sha256等于CLOSEOUT记录；`train_log.json`、`run_manifest.json`、`identity.json` 等于 `full_result.json` 的 "
        "`artifact_sha256`；身份、`test_read=false`、40 epoch、最佳DEV（最早并列）、末轮、续跑边界、训练计时、终末硬件、`n_pool` 与严重崩塌标记全部重算并与CLOSEOUT逐值相等。"
        "崩塌判定直接调用冻结的 `summarize.py` 中 `collapse()`（sha256与FROZEN.json一致，不调用会写文件的 `summarize()`）。best.pt/last.pt 已由收尾核验，本附录不使用模型二进制。")
    add(f"- `progress.jsonl` 不在收尾哈希链内：sha256固定于生成器，{checks['progress_epochs_matching_train_log']} 个epoch的DEV rSum与 `train_log.json` 逐值相等。")
    add(f"- 拆分前归档的 `progress.jsonl` 等于 `MIGRATED.json` 记录的哈希，并且逐行是终末 `progress.jsonl` 的前缀；分段边界等于 `train_log.json` 的 `resumed_at`。"
        f"{n_window_checks} 个作业起止边界（控制器StartTime/EndTime、squeue观察的起点、作业trap或控制器记录的结束）与各段epoch时间戳相容（允许 {TOLERANCE_S} 秒时钟偏差）；"
        f"最小余量：段内首个epoch晚于作业起点 {wm['start']['min_seconds']:.0f} 秒（{wm['start']['unit']}，job {wm['start']['job_id']}），"
        f"段内末个epoch早于作业结束 {wm['end']['min_seconds']:.0f} 秒（{wm['end']['unit']}，job {wm['end']['job_id']}）。")
    add(f"- 本附录共读取 {n_inputs} 个记录文件；路径、sha256与核验路线列在 `N2F_FINAL_33_UNITS.json` 的 `inputs`。")
    add('')

    # 1 final table
    add('## 1 最终33单元表')
    add('')
    add('完整列（含六个Recall、mR、各段硬件与job、n_pool、计时与哈希）见 `N2F_FINAL_33_UNITS.csv` / `.json`；下表为摘要。'
        'best为预注册终点；末轮=epoch 39；后期均值=epoch 30–39（epoch 30起学习率降为1/10，生成器已逐单元核验）。')
    add('')
    rows = []
    for method, scale in TABLE_ORDER:
        for s in (901, 902, 903):
            u = by_seed[(s, method, scale)]
            late = [u['rsum'][e] for e in LATE]
            rows.append([f"{method}/{scale.upper()}", s, f"{f2(u['endpoints']['best'])} ({u['best_epoch']})", f2(u['endpoints']['last']),
                         f2(u['endpoints']['late_mean']), f2(max(late) - statistics.median(late)),
                         f2(u['log']['epochs'][u['best_epoch']]['dev']['mR']), u['fr']['n_pool'],
                         '；'.join(seg_label(x) for x in u['segments'])])
    L += md_table(['方法/尺度', 'seed', 'best (epoch)', '末轮', '30–39均值', '30–39 max−中位', 'mR@best', 'n_pool', '各段硬件 (job)'], rows,
                  ['---', '---:', '---:', '---:', '---:', '---:', '---:', '---:', '---'])
    add('')

    # 2 hardware
    add('## 2 分段硬件与“同GPU”尺度配对')
    add('')
    L += blockquote(Q['same_job'])
    add('')
    L += blockquote(Q['c_pairing'])
    add('')
    L += blockquote(Q['split_readme'])
    add('')
    L += blockquote(Q['split_hw'])
    add('')
    changed = [u for u in units if len({(s['gpu'], s['host']) for s in u['segments']}) > 1]
    add(f"CLOSEOUT.json 每个单元的 `gpu`/`host` 取自 `full_result.json`，即终末段硬件。{len(resumed)} 个单元由续跑拼接（`resumed_at` 非空），"
        f"其中 {len(changed)} 个的早期段与终末段节点或型号不同；拆分后另有 {len(split_fresh)} 个单元整段重新训练。逐段记录如下（“段内最佳”为该段内DEV rSum最大的epoch）：")
    add('')
    rows = []
    for u in resumed + split_fresh:
        for k, seg in enumerate(u['segments']):
            rows.append([unit_name(u), k, f"{seg['first_epoch']}–{seg['last_epoch']}", seg['n_epochs'], seg['job_id'], seg['host'],
                         short_gpu(seg['gpu']), f"{f2(seg['best_rsum_in_segment'])} ({seg['best_epoch_in_segment']})",
                         '是' if seg is u['best_segment'] else '', seg['hardware_source']])
    L += md_table(['单元', '段', 'epoch', '轮数', 'job', '节点', 'GPU', '段内最佳 (epoch)', '含全程best', '硬件来源'], rows,
                  ['---', '---:', '---', '---:', '---', '---', '---', '---:', '---', '---'])
    add('')
    for u in units:
        for a in u['attempts']:
            if a['kind'] == 'failed_before_training':
                add(f"- {unit_name(u)} 另有一次未训练任何epoch的尝试：job {a['job_id']}（{a['host']} {short_gpu(a['gpu'])}），退出码 {a['exit_code']}；"
                    f"`DIAGNOSIS.json` 记为 {a['diagnosis_status']}，字段 {'、'.join(a['changed_fields'])} 相差 {a['difference_ulps']:g} 个fp64 ULP，"
                    f"`last_and_best_checkpoints_unchanged={str(a['checkpoints_unchanged']).lower()}`。")
            elif a['kind'] == 'timeout':
                add(f"- {unit_name(u)}：第一段所在job {a['job_id']} 在训练本单元第 {a['epochs_trained']} 轮后到达时限（控制器 {a['controller_state']}，"
                    f"TimeLimit {a['time_limit']}，记录退出码 {a['exit_code']}），随后按原配置续跑一次。")
            elif a['kind'] == 'completed' and a['exit_code'] != 0:
                add(f"- {unit_name(u)} 在job {a['job_id']} 内完成全部 {a['epochs_trained']} 轮；该job随后在训练同组下一单元时结束（退出码 {a['exit_code']}）。")
    for u in resumed:
        if 'migration' in u:
            m = u['migration']
            add(f"- {unit_name(u)}：拆分时原串行作业 {m['old_job']} 已完成 {m['completed_epochs']} 轮（归档时段内最佳 {f2(m['archived_best']['rsum'])}，epoch {m['archived_best']['epoch']}），"
                f"控制器记录 {m['old_job_state']}/ExitCode {m['old_job_exit']}；`MIGRATED.json` 记 `physical_gpu_pairing_preserved={str(m['physical_gpu_pairing_preserved']).lower()}`。")
    add('')
    add(f"按协议，{len(sm_pairs)} 个SUM–MEAN配对都应在同一job、同一GPU上顺序运行；冻结的组作业脚本每个job只申请一张GPU（`{Q['one_gpu']['text']}`，"
        f"`{Q['one_gpu']['path']}:{Q['one_gpu']['line']}`），所以同job即同一分配GPU。实际 {len(strict)} 个满足（两单元均单段且同job，时间上顺序执行，已核验）；"
        f"{len(nonstrict)} 个不满足：")
    add('')
    rows = []
    for p in sm_pairs:
        s_u, m_u = p['first'], p['second']
        rows.append([p['group'], f"{p['method']} s{p['seed']}", '；'.join(seg_label(x) for x in s_u['segments']),
                     '；'.join(seg_label(x) for x in m_u['segments']), p['label'], s2(p['delta_best']),
                     f"{short_gpu(s_u['best_segment']['gpu'])} {s_u['best_segment']['host']} / {short_gpu(m_u['best_segment']['gpu'])} {m_u['best_segment']['host']}"])
    L += md_table(['组', '方法 seed', 'SUM各段', 'MEAN各段', '配对状态', 'best差 SUM−MEAN', 'best所在硬件 SUM / MEAN'], rows,
                  ['---', '---', '---', '---', '---', '---:', '---'])
    add('')
    oracle_pairs = [p for p in pairs if p['kind'] != 'SUM-MEAN']
    add(f"OracleMask/OracleDrop 共 {len(oracle_pairs)} 组，其中 {sum(p['strict_same_job'] for p in oracle_pairs)} 组为同job顺序运行（" +
        '，'.join(f"{p['group']} {'是' if p['strict_same_job'] else '否'}" for p in oracle_pairs) + '）。')
    add('')
    add('只用严格同job配对时，SUM−MEAN的best差（描述性，不改变任何门槛）：')
    add('')
    rows = []
    for method in ('VCS_Weight', 'BCE_Weight', 'RankMass', 'Unweighted_A'):
        ps = [p for p in sm_pairs if p['method'] == method]
        st = [p for p in ps if p['strict_same_job']]
        rows.append([method, '，'.join(f"s{p['seed']} {s2(p['delta_best'])}" for p in st) or '无',
                     f"{s2(statistics.mean([p['delta_best'] for p in st]))} (n={len(st)})" if st else '—',
                     '，'.join(f"s{p['seed']} {s2(p['delta_best'])}" for p in ps if not p['strict_same_job']) or '无',
                     f"{s2(statistics.mean([p['delta_best'] for p in ps]))} ({sum(p['delta_best'] > 0 for p in ps)}/3正向)"])
    L += md_table(['方法', '严格同job配对', '严格子集均值', '非严格配对', '收尾全部3 seed'], rows, ['---', '---', '---:', '---', '---:'])
    add('')
    add('方法之间的比较本来就跨job，GPU型号也可能不同；各单元型号与节点（含分段，→表示续跑换段）：')
    add('')
    rows = []
    for method, scale in TABLE_ORDER:
        rows.append([f"{method}/{scale.upper()}"] + ['→'.join(f"{short_gpu(x['gpu'])} {x['host']}" for x in by_seed[(s, method, scale)]['segments'])
                                                     for s in (901, 902, 903)])
    L += md_table(['方法/尺度', 's901', 's902', 's903'], rows)
    add('')

    # 3 trajectories
    add('## 3 完整40 epoch DEV轨迹')
    add('')
    L += blockquote(Q['trajectory'])
    add('')
    add(f"全部 {len(units)}×{EPOCHS} 个epoch的DEV rSum、mR、相对epoch4比值、epoch内平均U-risk、hard阶段最后200步抽样的锁定比例与冻结判定标记、所在段/job/节点/GPU、"
        "epoch用时与时间戳见 `N2F_TRAJECTORIES.csv`。节选（DEV rSum）：")
    add('')
    rows = []
    for method, scale in TABLE_ORDER:
        for s in (901, 902, 903):
            u = by_seed[(s, method, scale)]
            rows.append([f"{method}/{scale.upper()}", s] + [f2(u['rsum'][e]) for e in SHOW_EPOCHS] +
                        [f"{f2(u['endpoints']['best'])} ({u['best_epoch']})", f2(u['endpoints']['late_mean'])])
    L += md_table(['方法/尺度', 'seed'] + [f'e{e}' for e in SHOW_EPOCHS] + ['best (epoch)', '30–39均值'], rows,
                  ['---', '---:'] + ['---:'] * (len(SHOW_EPOCHS) + 2))
    add('')
    early = [u for u in units if u['best_epoch'] < 30]
    if early:
        add(f"{len(units) - len(early)}/33 个单元的best落在epoch 30–39；其余：" + '，'.join(f"{unit_name(u)} epoch {u['best_epoch']}" for u in early) + '。')
    else:
        add(f"全部 {len(units)}/33 个单元的best都落在epoch 30–39（学习率衰减之后），最早为epoch {min(u['best_epoch'] for u in units)}。")
    add('')

    # 4 collapse
    add('## 4 崩塌与近崩塌的完整披露')
    add('')
    L += blockquote(Q['collapse_def'])
    add('')
    L += blockquote(Q['c_collapse'])
    add('')
    nlog = sorted({n for u in units for n in u['disclosure']['n_logged_steps_per_window']})
    intervals = checks['step_log_intervals']
    add(f"按冻结判定重算：严重崩塌 {n_severe}/33，与收尾一致；每个hard-epoch窗口的实际抽样点数为 {'、'.join(map(str, nlog))}"
        f"（训练日志每 {'、'.join(map(str, intervals))} 步记录一次）。下表列出满足以下任一条件的全部单元：hard阶段（epoch 5–39）任一epoch的DEV rSum低于本次epoch4；"
        "DEV rSum < 50%·epoch4；锁定比例（U-risk≥0.39的抽样点占比）≥90%。")
    add('')
    rows = []
    for u in flagged:
        d = u['disclosure']
        rows.append([unit_name(u), f2(d['epoch4_rsum']), f"{d['min_hard_ratio']:.3f} (e{d['min_hard_epoch']})",
                     ','.join(map(str, d['epochs_below_epoch4'])) or '—', ','.join(map(str, d['epochs_rsum_below_half_epoch4'])) or '—',
                     ','.join(map(str, d['epochs_locked_share_ge_0p9'])) or '—', ','.join(map(str, d['flag_epochs'])) or '—',
                     '是' if d['severe_collapse'] else '否'])
    L += md_table(['单元', 'epoch4', '最低比值 (epoch)', '低于epoch4的epoch', 'rSum<50%的epoch', '锁定≥90%的epoch', '两条件同时满足', '严重崩塌'], rows,
                  ['---', '---:', '---:', '---', '---', '---', '---', '---'])
    add('')
    for u in flagged:
        d = u['disclosure']
        if d['epochs_below_epoch4']:
            e0 = 4
            last_below = max(d['epochs_below_epoch4'])
            e1 = min(last_below + 2, EPOCHS - 1)
            seq = ' → '.join(f"{f2(u['rsum'][e])}" for e in range(e0, e1 + 1))
            locks = '，'.join(f"e{e} {u['windows'][e]['locked_share']:.1f}" for e in range(5, e1 + 1))
            recovered = all(u['rsum'][e] >= d['epoch4_rsum'] for e in range(last_below + 1, EPOCHS))
            after = (f"epoch {last_below + 1} 起DEV rSum始终不低于epoch4" if recovered else f"epoch {last_below} 之后仍有低于epoch4的epoch")
            flags = ','.join(map(str, d['flag_epochs'])) if d['flag_epochs'] else '无'
            add(f"- {unit_name(u)}：epoch {e0}–{e1} DEV rSum {seq}（最低 {d['min_hard_ratio']:.3f}×epoch4，epoch {d['min_hard_epoch']}）；"
                f"锁定比例 {locks}；{after}，best {f2(u['endpoints']['best'])}（epoch {u['best_epoch']}）。"
                f"两条件同时满足的epoch：{flags}；冻结定义要求连续两个这样的epoch，因而{'是' if d['severe_collapse'] else '不是'}“严重崩塌”。")
    locked_only = [u for u in flagged if u['disclosure']['epochs_locked_share_ge_0p9'] and not u['disclosure']['epochs_below_epoch4']]
    if locked_only:
        add(f"- 另有 {len(locked_only)} 个单元只有锁定比例达到阈值，DEV rSum在hard阶段始终高于epoch4：" + '，'.join(unit_name(u) for u in locked_only) +
            '；单独的锁定比例不满足冻结判定。')
    dips = [u for u in units if u['disclosure']['epochs_below_epoch4']]
    dip_cfg = sorted({(u['method'], u['scale']) for u in dips})
    add(f"- hard阶段DEV rSum低于自身epoch4的单元共 {len(dips)} 个：" + '，'.join(f"{unit_name(u)}（最低 {u['disclosure']['min_hard_ratio']:.3f}×）" for u in dips) +
        f"，分属 {'、'.join(f'{m}/{s.upper()}' for m, s in dip_cfg)}。这是描述，不是尺度因果证据（见下文）。")
    add('')
    add('### 与wave-1崩塌的比较')
    add('')
    L += blockquote(Q['w1_paused'])
    add('')
    L += blockquote(Q['w1_census'])
    add('')
    add(f"对普查点名的 {len(w1_rows)} 个wave-1单元（unit目录取自 `V9_N2_units/formal_units.json` 与 `tuning_units.json`，根目录为N2F FROZEN.json的 `old_output` 上级）"
        f"按N2F同一冻结判定重算，并与N2F中同seed、同方法、同学习率、同MEAN尺度的重跑及A/B诊断（{ab_rule['stop_steps']}步）对照：")
    add('')
    rows = []
    for r in w1_rows:
        n2f = r.get('n2f_rerun')
        if n2f:
            u = by_seed[(r['seed'], r['method'], 'mean')]
            n2f_cell = (f"{n2f['group']}：best {f2(u['endpoints']['best'])} (e{u['best_epoch']})，最低 {u['disclosure']['min_hard_ratio']:.3f}×，"
                        f"严重崩塌{'是' if u['collapse']['severe_collapse'] else '否'}")
        else:
            n2f_cell = 'N2F不含此seed/学习率'
        ab = '；'.join(f"{x['id']}（{x['loss_scale'].upper()}）：epoch4 {x['dev_rsum_epoch4']} → 切换后{x['steps_after_switch']}步 {x['dev_rsum_after_switch']}，"
                      f"锁定 {x['locked_fraction']}，collapse_present={str(x['collapse_present']).lower()}" for x in r['ab']) or '—'
        rows.append([f"{r['method']} s{r['seed']} x{r['lr_mult']:g}（{KIND_CN[r['kind']]}）", CENSUS_CN[r['census']], f2(r['epoch4']),
                     f"{r['min_hard_ratio']:.3f} (e{r['min_hard_epoch']})", f"{f2(r['best'])} (e{r['best_epoch']})", f2(r['last']),
                     f"{'是' if r['severe_collapse_frozen_rule'] else '否'}（{r['n_flag_epochs']}个两条件epoch）", n2f_cell, ab])
    L += md_table(['wave-1单元', '普查', 'epoch4', '最低比值', 'best', 'e39', '冻结判定：严重崩塌', 'N2F MEAN重跑', 'A/B诊断'], rows,
                  ['---', '---', '---:', '---:', '---:', '---:', '---', '---', '---'])
    add('')
    reruns = [r for r in w1_rows if r.get('n2f_rerun')]
    add('两个正式配置在切换前后的DEV rSum（epoch 4–8），wave-1与N2F MEAN重跑并列：')
    add('')
    rows = []
    for r in reruns:
        u = by_seed[(r['seed'], r['method'], 'mean')]
        rows.append([f"{r['method']} s{r['seed']}", 'wave-1'] + [f2(r['rsum'][e]) for e in range(4, 9)])
        rows.append([f"{r['method']} s{r['seed']}", f"N2F {u['group']}"] + [f2(u['rsum'][e]) for e in range(4, 9)])
    L += md_table(['配置', '运行', 'e4', 'e5', 'e6', 'e7', 'e8'], rows, ['---', '---', '---:', '---:', '---:', '---:', '---:'])
    add('')
    for r in reruns:
        n2f = r['n2f_rerun']
        parts = []
        for k, v in n2f['manifest_differences'].items():
            if v['fp64_ulps'] is not None:
                parts.append(f"`{k}` {v['wave1']!r} vs {v['n2f']!r}（{v['fp64_ulps']:g} 个fp64 ULP）")
            else:
                parts.append(f"`{k}` {json.dumps(v['wave1'])} vs {json.dumps(v['n2f'])}")
        hw = n2f['hardware']
        add(f"- {r['method']} s{r['seed']}：wave-1与N2F的 `run_manifest.json` 只在 " + '、'.join(parts) +
            f" 上不同，权重锁、训练配置与其余字段逐字段相同；N2F在 {short_gpu(hw['gpu'])} {hw['host']} 上运行，"
            f"deterministic_algorithms={str(hw['deterministic_algorithms']).lower()}、tf32={str(hw['tf32']).lower()}。")
    add('')
    L += blockquote(Q['determinism'])
    add('')
    add(f"A/B诊断（`{REPORTS_V9 / 'V9_N2_diag'}`）的预注册尺度定义、读法与判词：")
    add('')
    for key in ('w1_scales', 'w1_readout', 'w1_aa_valid', 'w1_verdict'):
        L += blockquote(Q[key])
        add('')
    add(f"记录结果：A/A（MEAN尺度从头重跑C4）collapse_present={str(ab_verdict['aa_valid']).lower()}，即A/A{'成立' if ab_verdict['aa_valid'] else '不成立'}；"
        f"{ab_verdict['n_sum_configs']} 个SUM尺度配置中 {ab_verdict['sum_configs_with_collapse']} 个出现崩塌。"
        + ('按预注册判词，A/A不成立时只报告逐配置结果，不给出尺度结论。' if not ab_verdict['aa_valid'] else ''))
    add('')
    L += blockquote(Q['ab_history'])
    add('')
    L += blockquote(Q['cause'])
    add('')
    rerun_severe = sum(by_seed[(r['seed'], r['method'], 'mean')]['collapse']['severe_collapse'] for r in reruns)
    require(not ab_verdict['aa_valid'] and rerun_severe == 0 and all(r['severe_collapse_frozen_rule'] for r in reruns),
            'The scale-cause statement below no longer matches the records; review the addendum text')
    add(f"**明确说明：mean与sum尺度之间的崩塌因果没有被确立。** wave-1中按冻结判定严重崩塌的 {len(reruns)} 个正式MEAN配置，在N2F以相同配置重跑时"
        f"严重崩塌 {rerun_severe}/{len(reruns)}；A/A未复现原崩塌，故A/B不给尺度结论；N2F全部33次训练严重崩塌 {n_severe}/33。"
        '上面的近崩塌与低于epoch4的波动只作披露。')
    add('')

    # 5 endpoint sensitivity
    add('## 5 最佳epoch与后期波动（描述性敏感性）')
    add('')
    L += blockquote(Q['endpoint'])
    add('')
    L += blockquote(Q['scale_rule'])
    add('')
    L += blockquote(Q['no_significance'])
    add('')
    mm_main = [max(u['rsum'][e] for e in LATE) - statistics.median([u['rsum'][e] for e in LATE]) for u in main]
    mm_or = [max(u['rsum'][e] for e in LATE) - statistics.median([u['rsum'][e] for e in LATE]) for u in oracle]
    gap_main = [u['endpoints']['best'] - u['endpoints']['late_mean'] for u in main]
    gap_or = [u['endpoints']['best'] - u['endpoints']['late_mean'] for u in oracle]
    add(f"预注册终点是40轮中DEV rSum最大的epoch。epoch 30–39 的“最大值−中位数”：{len(main)} 个非Oracle单元为 {rng(mm_main)}，{len(oracle)} 个Oracle单元为 {rng(mm_or)}；"
        f"“best−后期均值”：非Oracle {rng(gap_main)}，Oracle {rng(gap_or)}。下表把收尾的 {len(contrasts)} 项配对差分别用best、末轮（epoch 39）与epoch 30–39均值重算"
        "（各格为三seed均值，括号内为正方向seed数；†表示均值符号与best相反）。best列与CLOSEOUT逐值相等；末轮与30–39均值是看到轨迹之后的描述口径，不是预注册终点。")
    add('')
    rows = []
    for c in contrasts:
        cells = []
        for ep in ('best', 'last', 'late_mean'):
            mark = '†' if ep != 'best' and c[ep]['mean_sign_flip_vs_best'] else ''
            cells.append(f"{s2(c[ep]['mean'])} ({c[ep]['positive']}/3){mark}")
        rows.append([c['contrast']] + cells + ['，'.join(f"{s2(p['delta'])}" for p in c['late_mean']['pairs'])])
    L += md_table(['比较', 'best（预注册）', '末轮 e39', '30–39均值', '30–39均值的逐seed差（901，902，903）'], rows, ['---', '---:', '---:', '---:', '---'])
    add('')
    flips = [(c['contrast'], ep) for c in contrasts for ep in ('last', 'late_mean') if c[ep]['mean_sign_flip_vs_best']]
    add(f"均值符号翻转共 {len(flips)} 处：" + ('，'.join(f"{k}（{'末轮' if ep == 'last' else '30–39均值'}）" for k, ep in flips) if flips else '无') + '。')
    generic_keys = [c for c in contrasts if c['contrast'].endswith('-Unweighted_A')]
    for ep, name in (('best', 'best'), ('last', '末轮'), ('late_mean', '30–39均值')):
        means = [c[ep]['mean'] for c in generic_keys]
        add(f"- 通用加权相对Unweighted_A的 {len(generic_keys)} 项比较，{name}：均值 {s2(min(means))}…{s2(max(means))}，"
            f"每项正方向seed数至少 {min(c[ep]['positive'] for c in generic_keys)}/3。")
    vcs_keys = [c for c in contrasts if c['contrast'] in ('sum:VCS-BCE_Weight', 'sum:VCS-RankMass', 'sum:VCS-RuLSIF_Weight')]
    add('- 收尾VCS专有门槛涉及的三项SUM比较：' + '；'.join(
        f"{c['contrast']} best {s2(c['best']['mean'])} ({c['best']['positive']}/3)、末轮 {s2(c['last']['mean'])} ({c['last']['positive']}/3)、"
        f"30–39均值 {s2(c['late_mean']['mean'])} ({c['late_mean']['positive']}/3)" for c in vcs_keys) +
        f"。在两种事后口径下三项均值{'都' if all(c[ep]['mean'] >= 3 for c in vcs_keys for ep in ('last', 'late_mean')) else '并非都'}不低于+3；"
        '这两种口径是看到轨迹之后的描述，协议的checkpoint选择与主比较都按最佳DEV rSum（见上方引文），VCS专有门槛仍按best判定为未通过。')
    add('门槛不因此改变；收尾按预注册best终点得出的结论保持不变。')
    add('')

    # 6 cost
    add('## 6 计划与实测成本')
    add('')
    L += blockquote(cost_info['quote'])
    add('')
    ci = cost_info
    add(f"协议区间可由单次成本复原：{ci['n_full_pool_units']} 个完整训练池单元 + {ci['n_oracle_drop_units']} 个OracleDrop×{ci['oracle_drop_factor']:g} = "
        f"{ci['unit_equivalents']:.2f} 个单元当量；×{min(ci['per_unit_hours'].values()):.2f} h = {ci['reconstructed_range'][0]:.1f} GPU-h，"
        f"×{max(ci['per_unit_hours'].values()):.2f} h = {ci['reconstructed_range'][1]:.1f} GPU-h（协议写作 {ci['range'][0]}–{ci['range'][1]}）。")
    add('')
    add(f"实测训练计时（收尾口径，含续跑前已完成段）合计 {ci['measured_hours_total']:.2f} GPU-h；按各段所在GPU："
        + '，'.join(f"{k} {v:.2f}" for k, v in ci['measured_hours_by_model'].items())
        + f"。用协议单次成本、按实际分段硬件组合计算的计划值为 {ci['planned_hours_at_actual_mix']:.2f} GPU-h，实测为其 "
        f"{ci['measured_hours_total'] / ci['planned_hours_at_actual_mix']:.2f} 倍。协议注明确定性运行开销另计；"
        f"{checks['deterministic_and_tf32_off_units']}/33 个单元的硬件记录为 deterministic_algorithms=true、tf32=false。逐型号/节点的epoch用时：")
    add('')
    rows = [[r['model'], r['host'], '是' if r['oracle_drop'] else '否', r['epochs'], f"{r['median_epoch_seconds']:.1f}",
             f"{r['forty_epoch_hours']:.2f}", f"{r['planned_unit_hours']:.2f}", f"{r['ratio_to_planned']:.2f}"] for r in ci['epoch_durations']]
    L += md_table(['GPU（协议标签）', '节点', 'OracleDrop', 'epoch数', 'epoch用时中位数 (s)', '折合40轮 (h)', '协议单次 (h)', '倍数'], rows,
                  ['---', '---', '---', '---:', '---:', '---:', '---:', '---:'])
    add('')
    L += blockquote(Q['c_cost'])
    add('')
    L += blockquote(Q['c_sacct'])
    add('')

    # 7 gates
    add('## 7 收尾门槛（按收尾原样复述，不新增结论）')
    add('')
    L += blockquote(Q['gate_vcs'])
    add('')
    L += blockquote(Q['gate_generic'])
    add('')
    rows = [[r['contrast'], s2(r['mean']), f"{r['positive']}/3", '是' if r['passes'] else '否', f2(r['vcs_903_sum_needed_for_mean_3'])] for r in gate['vcs_specific']]
    L += md_table(['VCS专有比较（SUM）', '三seed均值', '正方向', '达到门槛', '使均值恰为+3所需的VCS s903 SUM'], rows, ['---', '---:', '---:', '---', '---:'])
    add('')
    b = gate['binding']
    require(not gate['vcs_specific_passes'] and gate['vcs_903_sum_final'] < b['vcs_903_sum_needed_for_mean_3'] and b['positive_rule_met_at_needed_value'],
            'The VCS-gate sentence below no longer matches the records; review the addendum text')
    add(f"其余单元结果固定时，三项同时达到+3需要VCS s903 SUM ≥ {f2(b['vcs_903_sum_needed_for_mean_3'])}（由 {b['contrast']} 决定；在该值下正方向规则也满足）；"
        f"实际为 {f2(gate['vcs_903_sum_final'])}，所以VCS专有门槛未通过，与收尾一致。这只是收尾门槛算术的复述，不是新判据。")
    add('')
    rows = [[r['scale'].upper(), r['contrast'], s2(r['mean']), f"{r['positive']}/3", '是' if r['passes'] else '否'] for r in gate['generic']]
    L += md_table(['尺度', '通用加权比较', '三seed均值', '正方向', '达到门槛'], rows, ['---', '---', '---:', '---:', '---'])
    add('')
    add(f"通用加权门槛：SUM={gate['generic_passes']['sum']}，MEAN={gate['generic_passes']['mean']}，与收尾一致。")
    add('')
    for key in ('c_vcs', 'c_generic', 'c_keep', 'c_descriptive'):
        L += blockquote(Q[key])
        add('')

    # 8 files
    add('## 8 文件与复现')
    add('')
    for name, digest in out_sha.items():
        add(f"- `{name}`：sha256 `{digest}`")
    add('- `V9_CLOSEOUT_ADDENDUM_CN.md`：本文件。')
    add('')
    add(f"复现：`python3 {GENERATOR_REL}` 重写四个文件（输入不变时逐字节相同）；`python3 {GENERATOR_REL} --check` 只重算并与仓库中的文件逐字节比较。"
        '任一输入哈希或重算值与记录不符时，生成器拒绝运行且不写任何文件。可选 `--plot <png>` 另绘33条轨迹（不入库，不参与 --check）。')
    return '\n'.join(L) + '\n'


def plot(units, path):
    """Small multiples of the 33 DEV trajectories (one panel per method/scale, one line per seed). Optional, not checked."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    surface, ink, ink2, muted, grid, axis, wash = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#f0efec'
    seeds = {901: ('#2a78d6', 'o'), 902: ('#eb6834', 's'), 903: ('#1baf7a', '^')}  # validated 3-slot set; shapes = 2nd channel
    px = 72 / 110  # points per pixel at dpi 110
    by_seed = {(u['seed'], u['method'], u['scale']): u for u in units}
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5), sharex=True, sharey=True, facecolor=surface)
    for ax, item in zip(axes.flat, TABLE_ORDER + [None]):
        ax.set_facecolor(surface)
        if item is None:
            ax.axis('off')
            handles = [Line2D([], [], color=c, lw=2 * px, marker=m, ms=9 * px, mec=surface, mew=2 * px, label=f'seed {s}')
                       for s, (c, m) in seeds.items()]
            handles.append(Line2D([], [], ls='none', marker='D', ms=8 * px, mfc=surface, mec=ink2, mew=2 * px, label='first epoch after a resume'))
            leg = ax.legend(handles=handles, loc='center', frameon=False, fontsize=10, labelcolor=ink)
            for t in leg.get_texts():
                t.set_color(ink)
            continue
        method, scale = item
        ax.axvspan(29.5, 39.5, color=wash, lw=0, zorder=0)
        ax.axvline(4.5, color=axis, lw=1 * px, zorder=1)
        for s, (color, marker) in seeds.items():
            u = by_seed[(s, method, scale)]
            ax.plot(range(EPOCHS), u['rsum'], color=color, lw=2 * px, solid_joinstyle='round', solid_capstyle='round', zorder=3)
            ax.plot([u['best_epoch']], [u['endpoints']['best']], ls='none', marker=marker, ms=9 * px, color=color,
                    mec=surface, mew=2 * px, zorder=4)
            for seg in u['segments'][1:]:
                e = seg['first_epoch']
                ax.plot([e], [u['rsum'][e]], ls='none', marker='D', ms=8 * px, mfc=surface, mec=color, mew=2 * px, zorder=5)
        ax.set_title(f'{method} / {scale.upper()}', fontsize=10, color=ink, loc='left')
        ax.set_ylim(100, 500)
        ax.set_xlim(-0.5, 39.5)
        ax.grid(axis='y', color=grid, lw=1 * px, ls='-')
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(axis)
            ax.spines[side].set_linewidth(1 * px)
        ax.tick_params(colors=muted, labelcolor=ink2, labelsize=8, width=1 * px)
    n_cols = axes.shape[1]
    off = len(TABLE_ORDER)  # the legend cell; the panel above it is the bottom panel of its column
    bottoms = [axes[-1][c] for c in range(n_cols) if (axes.shape[0] - 1) * n_cols + c < off] + \
              [axes[off // n_cols - 1][off % n_cols]]
    for ax in bottoms:
        ax.set_xlabel('epoch', color=ink2, fontsize=9)
        ax.xaxis.set_tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel('DEV rSum', color=ink2, fontsize=9)
    fig.suptitle('V9 N2F: full 40-epoch DEV rSum, 33 units (marker = best epoch; grey band = epochs 30-39 after the lr decay; '
                 'rule at 4.5 = end of warm-up)', fontsize=11, color=ink, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110, facecolor=surface, metadata={'Software': None})
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true', help='regenerate in memory and compare with the files on disk')
    ap.add_argument('--out', type=Path, default=HERE, help='output directory (default: next to this script)')
    ap.add_argument('--plot', type=Path, help='optional PNG of the 33 trajectories (not part of --check)')
    args = ap.parse_args(argv)
    try:
        outputs, units = build(Sources())
    except Refusal as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        return 2
    if args.check:
        bad = []
        for name in OUTPUTS:
            target = args.out / name
            same = target.exists() and target.read_bytes() == outputs[name].encode('utf-8')
            print(f"{'OK  ' if same else 'DIFF'} {target}")
            bad += [] if same else [name]
        return 1 if bad else 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        target = args.out / name
        tmp = target.with_name(f'.{name}.tmp')
        tmp.write_bytes(outputs[name].encode('utf-8'))
        tmp.replace(target)
        print(f"{sha256_bytes(outputs[name].encode('utf-8'))}  {target}")
    if args.plot:
        plot(units, args.plot)
        print(f'plot {args.plot}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
