#!/usr/bin/env python3
"""Refuse to run a formal NR5 unit unless PROTOCOL_FROZEN_NR5.json exists and every frozen file still has its frozen sha256."""
import hashlib, json, sys
from pathlib import Path
H = Path('/home/infres/yinwang/CS_QMI/handoff/nr5_20261104'); fz = H/'PROTOCOL_FROZEN_NR5.json'
if not fz.exists(): sys.exit('PROTOCOL_FROZEN_NR5.json missing: formal units not allowed')
P = json.loads(fz.read_text()); u = sys.argv[1]
if u not in P['units']: sys.exit(f'{u} not in frozen unit list')
bad = [f for f, h in P['file_sha256'].items() if hashlib.sha256(Path(f).read_bytes()).hexdigest() != h]
if bad: sys.exit(f'frozen files changed: {bad}')
print('frozen OK', u)
