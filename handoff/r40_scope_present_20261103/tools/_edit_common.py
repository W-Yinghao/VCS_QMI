import sys
from pathlib import Path
for t in ('r40f', 'r40c'):
    p = Path(__file__).resolve().parents[1] / t / 'common.py'
    s = p.read_text()
    s = s.replace('"""R39 ', '"""R40 tree %s = byte copy of R39 %s (R40 edits: ROOT/OUT/RUNS/ORACLE_RUNS to the R40 dirs, R39_PREP pinned to the R39 output tree (read-only), trainer guard also refuses the R39 oracle/audit outputs, R40 constants at the end). R39 ' % (t, 'r39f' if t == 'r40f' else 'r39c'), 1)
    old_root = "ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102')"
    assert old_root in s
    s = s.replace(old_root, "ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103')\nR39_ROOT = Path('/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102')   # R39 (read-only)")
    old_out = "OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102')"
    assert old_out in s
    s = s.replace(old_out, "OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103')\nR39_OUT = Path('/projects/EEG-foundation-model/yinghao/outputs_v9/r39_complete_joint_20261102')   # R39 outputs (read-only)")
    old_prep = "R39_PREP = OUT / 'prep'"
    assert s.count(old_prep) == 1
    s = s.replace(old_prep, "R39_PREP = R39_OUT / 'prep'   # R40: the R39 prep is reused read-only")
    old_g = "            _FORBIDDEN_PREFIXES[:] = [str(ORACLE_RUNS), str(OUT / 'audit')]"
    assert old_g in s
    s = s.replace(old_g, "            _FORBIDDEN_PREFIXES[:] = [str(ORACLE_RUNS), str(OUT / 'audit'), str(R39_OUT / 'oracle_runs'), str(R39_OUT / 'audit'), str(OUT / 'diag7' / 'oracle')]   # R40: + R39 oracle/audit outputs")
    old_o = "            _FORBIDDEN_PREFIXES[:] = [str(OUT / 'audit')]"
    assert old_o in s
    s = s.replace(old_o, "            _FORBIDDEN_PREFIXES[:] = [str(OUT / 'audit'), str(R39_OUT / 'audit')]")
    s += '''

# ----------------------------------------------------------------------------- R40 constants (update scope x incomplete-correspondence supervision)
R40_PACKAGE = Path('/home/infres/yinwang/CS_QMI/r40_plan/VCS_R39_Review_R40_Plan_20261103')   # R40 design package (read-only)
FROZEN_FILE = 'PROTOCOL_FROZEN.json'      # R40 protocol (ROOT/PROTOCOL_FROZEN.json); the R39/R39X freeze files are never read for binding
ADAPTATIONS = {'frozen': (), 'image_only': ('img',), 'text_only': ('txt',), 'both_joint': ('img', 'txt')}
'''
    p.write_text(s)
    print(t, 'ok')
