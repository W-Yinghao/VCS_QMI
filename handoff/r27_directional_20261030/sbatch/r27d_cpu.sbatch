#!/bin/bash
#SBATCH --job-name=r27d_cpu
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_directional_20261030/logs/%x_%j.out
# R27-D package CPU tests (pytest, synthetic tensors) + import of the installed snapshot with all D arms registered.
set -euo pipefail
PKG=/home/infres/yinwang/CS_QMI/r27d_server/VCS_Retrieval_R27D_Server_20261030
D=/home/infres/yinwang/CS_QMI/handoff/r27_directional_20261030
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
cd $PKG
$PY -B -m pytest -q -p no:cacheprovider tests --junitxml=$D/CPU_PYTEST_${SLURM_JOB_ID}.xml
cd $D/code_frozen
$PY -B -c "
import r27_co_train as R, r27d_policy as P
assert all(R.ARMS[k]=='S' for k in P.D_ARMS)
for a,pol in P.D_ARMS.items():
    for b in ('i2t','t2i'): assert R.direction_kind(a,b,R.S_ARMS)==pol[b]
for a in ('s4_single_cap_vcs','s6_single_cap_snce','s0_identity'):
    assert R.direction_kind(a,'i2t',R.S_ARMS)==R.S_ARMS[a]
print('SNAPSHOT_IMPORT_OK')"
echo CPU_DONE
