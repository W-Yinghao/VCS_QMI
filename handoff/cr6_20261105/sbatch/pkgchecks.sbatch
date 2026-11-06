#!/usr/bin/env bash
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --nice=50
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr6_20261105/logs/%x_%j.out
# Package run_checks.sh in a scratch copy with an env that has torch+pandas (vcs_aigiqa_ir: torch 2.9.0, pandas 3.0.6).
# Frozen envs (env_nr1) are not modified; package files are not modified (copy is used).
set -euo pipefail
S=/tmp/cr6_pkgchecks_$SLURM_JOB_ID; rm -rf $S; cp -a /home/infres/yinwang/CS_QMI/handoff/cr6_20261105/package $S; chmod -R u+w $S
export PATH=/home/infres/yinwang/anaconda3/envs/vcs_aigiqa_ir/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
echo "JOB $SLURM_JOB_ID $(hostname) $(python -c 'import torch,pandas,sys;print(sys.executable,torch.__version__,pandas.__version__)') $(date -u +%FT%TZ)"
cd $S && bash run_checks.sh 2>&1
echo "EXIT_OK $(date -u +%FT%TZ)"
