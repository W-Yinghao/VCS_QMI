"""Write the registered R44 diffs with difflib (the representation cpu_test_r44.t_code compares). Usage: python3 tools/write_r44_diffs.py"""
import difflib
from pathlib import Path
W = Path(__file__).resolve().parents[1]
for new, old in (('r44f', 'r42f'), ('r44c', 'r41c')):
    t = ''.join(''.join(difflib.unified_diff((W / old / f).read_text().splitlines(True), (W / new / f).read_text().splitlines(True), f'{old}/{f}', f'{new}/{f}')) for f in ('common.py', 'train_unit.py'))
    (W / f'R44_DIFF_{new.upper()}.patch').write_text(t); print(new, len(t.splitlines()))
