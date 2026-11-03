"""R41 de-duplication against every R37/R37C/R37S/R38/R39/R39X/R40 config (cell key as tools/dedupe.py + steps/seed). Writes DEDUPE_R41.json."""
import glob, json, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'tools'))
from dedupe import SOURCES, key
SOURCES = dict(SOURCES, R40=W / 'configs')
ex = {}
for name, d in SOURCES.items():
    for f in sorted(glob.glob(str(d / '*.json'))):
        c = json.load(open(f))
        if isinstance(c, dict) and 'unit_id' in c and not c['unit_id'].startswith('R41_'):
            ex.setdefault(key(c), []).append(dict(line=name, unit=c['unit_id']))
units = json.load(open(W / 'configs/UNITS_R41.json')) + json.load(open(W / 'configs/UNITS_R41C.json'))
rows = []
for u in units:
    c = json.load(open(W / 'configs' / f'{u}.json')); k = key(c)
    rows.append(dict(unit=u, key=list(k), duplicate_of=ex.get(k, [])))
dups = [r['unit'] for r in rows if r['duplicate_of']]
json.dump(dict(n=len(units), n_existing=sum(len(v) for v in ex.values()), duplicates=dups, ALL_UNIQUE=not dups, units=rows), open(W / 'DEDUPE_R41.json', 'w'), indent=1)
print('R41 DEDUPE ALL_UNIQUE', not dups, len(units), dups)
