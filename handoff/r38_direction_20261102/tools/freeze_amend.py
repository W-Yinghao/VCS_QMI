"""R38 user amendment: re-freeze PROTOCOL_FROZEN.json for 84 units (original 30 frozen at PROTOCOL_FROZEN_30.json, kept byte-for-byte;
its 30 configs unchanged) after AMENDMENT_CPU_CHECK ALL_PASS. r38f code changes only by the R37 parent conditions n60s1002/n60s1003 in
common.py (copied from R37). Usage: python -I -B tools/freeze_amend.py"""
import csv, glob, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'r38c'))
from common import sha256, write_json, read_json, utc

QUOTE = '「检查一下R38，实验要做的全一些」'


def main():
    old = read_json(W / 'PROTOCOL_FROZEN_30.json'); am = read_json(W / 'AMENDMENT_CPU_CHECK.json'); assert am['ALL_PASS']
    units = read_json(W / 'configs' / 'UNITS.json'); assert len(units) == 84 and set(old['units']) <= set(units)
    for u, h in old['config_sha256'].items():
        assert sha256(W / 'configs' / f'{u}.json') == h, u
    added = [r['unit_id'] for r in csv.DictReader(open(W / 'MATRIX_AMENDMENT.csv'))]; assert set(added) == set(units) - set(old['units'])
    new = dict(old)
    code = {t: {Path(f).name: sha256(f) for f in sorted(glob.glob(str(W / t / '*.py')))} for t in ('r38f', 'r38c')}
    mans = dict(old['parent_cache_manifest_sha256'])
    for c in ('n60s1002', 'n60s1003'):
        m = f'/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/cache/{c}/CACHE_MANIFEST.json'; mans[m] = sha256(m)
    m = '/projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/prep/PREP_MANIFEST.json'; mans[m] = sha256(m)
    new.update(line=old['line'].replace('30 critic-only fits', '84 critic-only fits (30 package + 54 user amendment)'), frozen_utc=utc(), status='FROZEN_BEFORE_FORMAL_UNITS (amended)',
               units=units, config_sha256={u: sha256(W / 'configs' / f'{u}.json') for u in units}, resolved_matrix_sha256=sha256(W / 'RESOLVED_MATRIX.csv'),
               parent_binding_sha256=sha256(W / 'PARENT_BINDING.json'), reuse_references_sha256=sha256(W / 'REUSE_REFERENCES.json'),
               code_sha256_by_tree=code, parent_cache_manifest_sha256=mans,
               amendment_matrix_sha256=sha256(W / 'MATRIX_AMENDMENT.csv'), amendment_cpu_check=dict(job=am['job'], sha256=sha256(W / 'AMENDMENT_CPU_CHECK.json')),
               user_amendment_pre_freeze=dict(
                   quote=QUOTE, relayed_by='supervising session (main cs-qmi session)', received_utc='2026-11-02T18:10Z (approx.)',
                   timing='received right after the 30-unit freeze (PROTOCOL_FROZEN_30.json, 2026-11-02T18:09:56Z) and before any R38 unit had finished; the 30 package units were already submitted (1082570-1082575) and are kept unchanged (configs byte-identical, same frozen code for r38c; r38f gains only the two R37 parent conditions)',
                   reason='completeness requested by the user (apply the package B/C cells to every available parent); not outcome-driven (no R38 result existed)',
                   design_change='none: same N/T lists, null, trusted subsets, heads, schedules, 8000 steps, in-unit checks; T-directional not added (undefined in the package)',
                   added_parents=dict(f60_R30_s1002='R37 parent r30v11_f60_s1002_HH (R37 cache/prep/trust, read-only)', f60_R30_s1003='R37 parent r30v11_f60_s1003_HH (R37 cache/prep/trust, read-only)',
                                      cc_R31_s1002='R31 H0_default cc_s1002 (R38 cache/prep)', cc_R31_s1003='R31 H0_default cc_s1003 (R38 cache/prep)'),
                   reused_read_only='Flickr s1002/s1003 N and T shared-constant = R37 confirm units; CC s1002/s1003 N shared-constant = R38 A block',
                   n_units_added=len(added), units_added=added),
               freeze_history=[dict(file='PROTOCOL_FROZEN_30.json', sha256=sha256(W / 'PROTOCOL_FROZEN_30.json'), frozen_utc=old['frozen_utc'], units=len(old['units']))])
    write_json(W / 'PROTOCOL_FROZEN.json', new)
    print('PROTOCOL_FROZEN (amended) written', len(units))


if __name__ == '__main__':
    main()
