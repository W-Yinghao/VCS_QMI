#!/usr/bin/env python3
"""Refuse to run unless every hash in PROTOCOL_FROZEN_NR3.json and the unit's production + package config hashes match."""
import hashlib, json, sys
from pathlib import Path
N3 = Path('/home/infres/yinwang/CS_QMI/handoff/nr3_20261104')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
u = sys.argv[1]; fz = json.loads((N3/'PROTOCOL_FROZEN_NR3.json').read_text())
bad = [p for p, h in fz['hashes'].items() if sha(p) != h]
if u not in fz['units'] or sha(N3/f'configs_production/{u}.json') != fz['units'][u]['production'] or sha(N3/f'package/configs/{u}.json') != fz['units'][u]['package']:
    bad.append('config:' + u)
if bad: sys.exit(f'FROZEN_MISMATCH {bad}')
print('frozen ok', u)
