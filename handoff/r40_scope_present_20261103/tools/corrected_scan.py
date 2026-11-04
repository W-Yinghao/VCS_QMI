"""Corrected-readout unit scan (no training): every completed G / GD critic unit (frozen-encoder and JOINT) of R37-R46 with its final checkpoint,
grouped by (evaluation tree, cond, training GPU type). Writes corrected/UNITS_SCAN.json and corrected/groups/<group>.json.
Evaluation trees: Flickr conds -> r44f; COCO conds -> r46o; CC152K 'cc' -> r44c; other CC parents (cc_s1002 / cc_s1003 / cc_r37s) -> NOT_SUPPORTED
(first scan: listed, not evaluated; now -> CC audit tree r47cca). Usage: python3 -I -B tools/corrected_scan.py"""
import glob, json
from pathlib import Path
W = Path(__file__).resolve().parents[1]
H = Path('/home/infres/yinwang/CS_QMI/handoff'); O = Path('/projects/EEG-foundation-model/yinghao/outputs_v9')
CFG_DIRS = {'R37': H / 'r37_confirm_20261102/configs', 'R37C': H / 'r37c_cc152k_20261102/configs', 'R37S': H / 'r37s_cc152k_r30h_20261102/configs',
            'R38': H / 'r38_direction_20261102/configs', 'R39': H / 'r39_complete_joint_20261102/configs', 'R40+': W / 'configs'}
RUN_ROOTS = [O / 'r37_confirm_20261102/runs', O / 'r37c_cc152k_20261102/runs', O / 'r37s_cc152k_r30h_20261102/critic/runs', O / 'r38_direction_20261102/runs',
             O / 'r39_complete_joint_20261102/runs', O / 'r39_complete_joint_20261102/runs_r39x', O / 'r39_complete_joint_20261102/oracle_runs'] + \
            [Path(p) for p in sorted(glob.glob(str(O / 'r40_scope_present_20261103/runs*')) + glob.glob(str(O / 'r40_scope_present_20261103/oracle_runs*')))]
FLICKR = ('n60', 'n60s1002', 'n60s1003', 'n00', 'f40in', 'f80in', 'f20in_s1001', 'f20in_s1002', 'f20in_s1003', 'f40in_s1002', 'f40in_s1003', 'f80in_s1002', 'f80in_s1003')


def tree_of(cond):
    if cond in FLICKR:
        return 'r44f'
    if cond.startswith('c') and 'in_s' in cond:
        return 'r46o'
    if cond == 'cc':
        return 'r44c'
    if cond in ('cc_s1002', 'cc_s1003', 'cc_r37s'):
        return 'r47cca'   # CC audit tree (r47c + the R39X / R37S CC parents)
    return None


def main():
    seen, units, groups = set(), [], {}
    for line, cd in CFG_DIRS.items():
        for f in sorted(cd.glob('*.json')):
            try:
                c = json.loads(f.read_text())
            except Exception:
                continue
            if not isinstance(c, dict) or 'unit_id' not in c or c['unit_id'] in seen or c.get('structure') not in ('G', 'GD'):
                continue
            u = c['unit_id']; seen.add(u)
            rd = next((r / u for r in RUN_ROOTS if (r / u / 'DONE.json').exists()), None)
            if rd is None:
                continue
            steps = int(c['steps']); ck = rd / f'ckpt_step{steps}.pt'
            r = json.loads((rd / 'RESULT.json').read_text()) if (rd / 'RESULT.json').exists() else None
            ev = rd / f'eval_step{steps}.json'
            e = dict(unit=u, line=line, cond=c['cond'], structure=c['structure'], encoder='joint' if c.get('encoder') == 'joint_global' else 'frozen', supervision=c.get('supervision'),
                     objective=c.get('objective'), steps=steps, oracle=bool(c.get('oracle')), run_dir=str(rd), ckpt=str(ck), ckpt_exists=ck.exists(),
                     logged_rsum=json.loads(ev.read_text())['dev_retrieval']['rsum'] if ev.exists() else None, gpu=(r or {}).get('env', {}).get('gpu'), init_seed=c.get('init_seed'),
                     config=str(f), tree=tree_of(c['cond']))
            e['status'] = 'TO_EVALUATE' if (e['tree'] and e['ckpt_exists'] and e['logged_rsum'] is not None) else ('NOT_SUPPORTED_TREE' if not e['tree'] else 'MISSING_CKPT_OR_EVAL')
            units.append(e)
            if e['status'] == 'TO_EVALUATE':
                g = f"{e['tree']}__{e['cond']}__{(e['gpu'] or 'unknown').split()[-1] if 'RTX' not in (e['gpu'] or '') else 'RTX6000PRO'}"
                groups.setdefault(g, []).append(e)
    od = W / 'corrected'; (od / 'groups').mkdir(parents=True, exist_ok=True)
    (od / 'UNITS_SCAN.json').write_text(json.dumps(units, indent=1))
    for g, v in groups.items():
        (od / 'groups' / f'{g}.json').write_text(json.dumps(v, indent=1))
    from collections import Counter
    print('units', len(units), Counter(e['status'] for e in units), 'groups', len(groups))
    for g, v in sorted(groups.items()):
        print(g, len(v), Counter(e['encoder'] for e in v))


if __name__ == '__main__':
    main()
