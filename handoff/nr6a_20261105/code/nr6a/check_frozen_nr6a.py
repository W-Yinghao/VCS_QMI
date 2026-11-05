#!/usr/bin/env python3
"""NR6-A unit gate: PROTOCOL_FROZEN_NR6A.json exists, the unit is listed, and every frozen file (NR6-A files + imported NR5/NR5_A1/NR4/NR3/NR2 code) is unchanged."""
import hashlib, json, sys
from pathlib import Path
fz = Path('/home/infres/yinwang/CS_QMI/handoff/nr6a_20261105/PROTOCOL_FROZEN_NR6A.json')
if not fz.exists(): sys.exit('PROTOCOL_FROZEN_NR6A.json missing: formal units not allowed')
P = json.loads(fz.read_text()); u = sys.argv[1]
if u not in P['units']: sys.exit(f'{u} not in frozen unit list')
bad = [f for f, h in {**P['file_sha256'], **P['imported_code_sha256']}.items() if hashlib.sha256(Path(f).read_bytes()).hexdigest() != h]
if bad: sys.exit(f'frozen files changed: {bad}')
print('NR6A frozen OK', u)
