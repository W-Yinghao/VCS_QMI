"""Freeze the R37S critic protocol (PROTOCOL_FROZEN_CRITIC.json) after CRITIC_CPU_TEST ALL_PASS: 18 units, code tree r37s_critic,
configs, parent binding, cache/prep manifests. No standalone GPU gate (waived by the user, as R37/R38): every unit asserts the step-0
two-stage == native-rank check and the first/second-update graph checks (fail-fast). train_unit.check_frozen refuses formal units
otherwise. Usage: python -I -B tools/freeze_critic.py"""
import csv, glob, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r37s_critic'))
from common import sha256, write_json, read_json, utc, CONDS, SERIES

QUOTE = '「实验要做的全一些」'


def main():
    cpu = read_json(W / 'CRITIC_CPU_TEST.json'); assert cpu['ALL_PASS'], 'critic CPU test must pass'
    units = read_json(W / 'configs' / 'UNITS.json'); assert len(units) == 18
    rows = list(csv.DictReader(open(W / 'CRITIC_MATRIX.csv'))); assert [r['unit_id'] for r in rows] == units
    c = CONDS['cc_r37s']
    mans = [str(Path(c['cache_dir']) / 'CACHE_MANIFEST.json'), str(Path(c['prep_dir']) / 'PREP_MANIFEST.json')]
    write_json(W / 'PROTOCOL_FROZEN_CRITIC.json', dict(
        line='R37S critic-only fits on the R37S CC152K R30-style H parent (18 units, 8000 steps); development experiment, no TEST read',
        frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS', series=SERIES,
        units=units, config_sha256={u: sha256(W / 'configs' / f'{u}.json') for u in units}, matrix_sha256=sha256(W / 'CRITIC_MATRIX.csv'),
        matrix=dict(approved_6='{T, N} shared head, constant lr x {J, BCE, NCE} (R37C definitions on this parent)',
                    user_added_12=dict(quote=QUOTE, relayed_by='supervising session (main cs-qmi session)',
                                       cells='N shared-cosine, N directional-constant, N directional-cosine, T shared-cosine x {J, BCE, NCE} (the R38 cells, same definitions)',
                                       reason='completeness requested by the user, before any R37S critic result existed (not outcome-driven)'),
                    not_included='T directional (undefined in the R38 package), oracle/M/V units'),
        definitions=dict(steps=8000, lr_head=1e-4, cosine='lr(t) = 1e-4*[0.1+0.45*(1+cos(pi*t/7999))], t = update index 0..7999 (R38)',
                         directional='R38 DirGHead: shared trunk, per-direction output (zero init) + positive affine; +131 params/peer',
                         N='OBS_MISSING_NULL (frozen-H top-32 + null slot, Q uniform 33)', T='TRUST_HARD (this parent\'s native-GMM trusted subset; observed positive + 31 frozen-H top-32 others, Q uniform 32)'),
        parent_binding_sha256=sha256(W / 'PARENT_BINDING.json'),
        code_sha256_by_tree={'r37s_critic': {Path(f).name: sha256(f) for f in sorted(glob.glob(str(W / 'r37s_critic' / '*.py')))}},
        code_not_bound=['build_cache.py', 'prep.py'], code_diff_vs_r38c=dict(file='R37S_CRITIC_DIFF_VS_R38C.patch', sha256=sha256(W / 'R37S_CRITIC_DIFF_VS_R38C.patch')),
        parent_cache_manifest_sha256={m: sha256(m) for m in mans},
        cpu_test=dict(job=cpu['job'], sha256=sha256(W / 'CRITIC_CPU_TEST.json')),
        gpu_gate=dict(waived_by_user=True, utc=utc(), basis='R37 waiver (user 2026-11-02 「好，取消gate，直接提交正式实验」), carried to R38 and to this line by the supervising session: no standalone GPU gate; every unit asserts step-0 two-stage ranks == native ranks and the first/second-update graph checks (fail-fast); no earlier unit on this parent -> no reference init/chain assertion (cross-unit equality checked post hoc)'),
        hardware='--qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=0; formal units never on runfill',
        primary_endpoint='fixed step 8000 (curves 0/250/500/1000/2000/4000/8000; no per-method stop/extend/best-step)', test_read=False))
    print('PROTOCOL_FROZEN_CRITIC written', len(units))


if __name__ == '__main__':
    main()
