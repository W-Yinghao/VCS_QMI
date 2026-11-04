#!/usr/bin/env python3
"""NR2E: refuse to run unless every hash in PROTOCOL_FROZEN_NR2E.json and the unit config hash match."""
import hashlib, json, sys
from pathlib import Path
N = Path('/home/infres/yinwang/CS_QMI/handoff/nr2_20261104')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
u = sys.argv[1]; fz = json.loads((N/'PROTOCOL_FROZEN_NR2E.json').read_text())
bad = [p for p, h in fz['hashes'].items() if sha(p) != h]
if u not in fz['units'] or sha(N/f'locked_configs_nr2e/{u}.json') != fz['units'][u]: bad.append('config:' + u)
if bad: sys.exit(f'FROZEN_MISMATCH {bad}')
print('frozen ok', u)
