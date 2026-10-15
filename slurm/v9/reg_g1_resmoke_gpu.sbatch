#!/usr/bin/env bash
#SBATCH --job-name=v9_reg_g1_resmoke
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-G1 re-smoke of the CLI path edited after the gate (smoke-root redirect + mixing guards): one VCS smoke unit end-to-end under <root>/smoke,
# then the same call again (must skip on the existing smoke result), on the formal root.  Code check only.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9; cd "$R" || exit 2; export PYTHONPATH="$R/src:$R"; export V9_REG_ROOT=/projects/EEG-foundation-model/yinghao/outputs_v9/reg
echo "host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git -C "$R" rev-parse --short HEAD)"
"$PYTHON" scripts/v9/reg_g1.py run --unit V9-G1-vcs-frac1-eps0-s903 --smoke --n-pert 2; rc1=$?
"$PYTHON" scripts/v9/reg_g1.py run --unit V9-G1-vcs-frac1-eps0-s903 --smoke --n-pert 2; rc2=$?
echo "smoke_dir_in_formal_root=$(ls -d $V9_REG_ROOT/units/V9-G1-vcs-frac1-eps0-s903 2>/dev/null | wc -l) smoke_dir_under_smoke=$(ls -d $V9_REG_ROOT/smoke/units/V9-G1-vcs-frac1-eps0-s903 2>/dev/null | wc -l)"
echo "rc1=$rc1 rc2=$rc2 exit=$(( rc1 | rc2 )) end=$(date -u +%FT%TZ)"; exit $(( rc1 | rc2 ))
