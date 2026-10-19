#!/usr/bin/env bash
#SBATCH --job-name=v9_queue_limits
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
cd /home/infres/yinwang/CS_QMI/handoff/v9_dispatch_pending8_20261017
python3 - <<'PY'
import datetime, json, os, subprocess
from pathlib import Path
commands = [
    ['sacctmgr', '-n', '-P', 'show', 'qos', 'normal', 'format=Name,MaxTRESPU,MaxSubmitPU,MaxJobsPU'],
    ['sacctmgr', '-n', '-P', 'show', 'assoc', 'where', 'user=yinwang', 'format=User,Account,Partition,QOS,MaxSubmitJobs,MaxJobs'],
    ['sacct', '-X', '-j', '1042382', '-n', '-P', '--format=JobID,State,ExitCode,ElapsedRaw'],
]
out = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'job': os.environ['SLURM_JOB_ID'], 'queries': []}
for cmd in commands:
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
    out['queries'].append({'command': cmd, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr})
Path('LIMITS.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
PY
