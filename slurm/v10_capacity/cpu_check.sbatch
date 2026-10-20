#!/usr/bin/env bash
#SBATCH --job-name=v10k_cpucheck
#SBATCH --partition=cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# BF1_CAPACITY_EXT_K8_K32 CPU gate: control checks (no Torch) + CPU smoke of K8/K16/K32 on FIT data.  Writes the bundle CHECKS.json
# that every GPU unit job requires (status PASS for the same FREEZE.json sha).
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH
: "${V10_CAPACITY_FREEZE:?sha256 of the bundle FREEZE.json is required (--export=ALL,V10_CAPACITY_FREEZE=...)}"
export V10_EXPECT_CODE=e08ee2c176475e2b V10_SUBMIT_COMMIT=b8f59ea825dc7a2064457d47a628bef09ca4e185
WT=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_capacity
BUNDLE=/home/infres/yinwang/CS_QMI/handoff/v10_capacity_ext_20261019
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
OUTDIR="$BUNDLE/validation/$SLURM_JOB_ID"
mkdir -p "$OUTDIR"
cd "$WT/scripts/v10_capacity"
set +e
"$PYTHON" -u check_control.py > "$OUTDIR/check_control.log" 2>&1
control_rc=$?
cat "$OUTDIR/check_control.log"
"$PYTHON" -u smoke_k.py --out "$OUTDIR"
smoke_rc=$?
set -e
rm -rf "$OUTDIR"/v10k_smoke_*
CONTROL_RC=$control_rc SMOKE_RC=$smoke_rc OUTDIR="$OUTDIR" "$PYTHON" - <<'PY'
import os
import capacity_common as C
out = C.Path(os.environ['OUTDIR'])
log = (out / 'check_control.log').read_text()
smoke = C.read(out / 'SMOKE.json') if (out / 'SMOKE.json').exists() else {}
ok = os.environ['CONTROL_RC'] == '0' and os.environ['SMOKE_RC'] == '0' and smoke.get('status') == 'PASS'
summary = [l for l in log.splitlines() if l.startswith(('Ran ', 'OK', 'FAILED'))]
if C.sha(C.BUNDLE / 'FREEZE.json') != os.environ['V10_CAPACITY_FREEZE']:
    ok = False
    summary.append('FREEZE.json changed during the checks')
C.write(C.BUNDLE / 'CHECKS.json', {'status': 'PASS' if ok else 'FAIL', 'utc': C.utc(), 'job': os.environ['SLURM_JOB_ID'],
        'freeze_sha256': os.environ['V10_CAPACITY_FREEZE'], 'control': {'returncode': int(os.environ['CONTROL_RC']), 'summary': summary,
        'log': str(out / 'check_control.log'), 'log_sha256': C.sha(out / 'check_control.log')},
        'smoke': {'returncode': int(os.environ['SMOKE_RC']), 'status': smoke.get('status'), 'n_checks': smoke.get('n_checks'),
                  'n_failed': smoke.get('n_failed'), 'seconds': smoke.get('seconds'), 'path': str(out / 'SMOKE.json'),
                  'sha256': C.sha(out / 'SMOKE.json') if (out / 'SMOKE.json').exists() else None},
        'scope': 'execution checks only (CPU); not scientific evidence; GPU units require this PASS for the same freeze'})
print('CHECKS', 'PASS' if ok else 'FAIL')
raise SystemExit(0 if ok else 1)
PY
