"""Write the registered R50 code diffs R50_DIFF_R50F.patch / R50_DIFF_R50C.patch (unified diff of every changed / new file vs the R48 tree).
Usage: python3 -I -B tools/r50_write_diffs.py"""
import difflib, hashlib
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for new, old in (('r50f', 'r48f'), ('r50c', 'r48c')):
    a = sorted(p.name for p in (W / new).glob('*.py'))
    cur = ''.join(''.join(difflib.unified_diff((W / old / f).read_text().splitlines(True) if (W / old / f).exists() else [], (W / new / f).read_text().splitlines(True),
                                               f'{old}/{f}', f'{new}/{f}')) for f in a if not ((W / old / f).exists() and sh(W / old / f) == sh(W / new / f)))
    (W / f'R50_DIFF_{new.upper()}.patch').write_text(cur); print(new, len(cur.splitlines()), 'lines')
