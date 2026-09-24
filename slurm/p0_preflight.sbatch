#!/usr/bin/env bash
#SBATCH --job-name=vcs_p0_preflight
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p0_preflight_%j.out
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P0 preflight (+ manifest creation)"
"$PYTHON" -m vcs_ssl.prepare_data --root "$DATA_ROOT" --report reports/P0_data_verification.json || exit 2
"$PYTHON" -m vcs_ssl.preflight --config configs/cifar10_pilot_vcs.yaml --create-manifest
rc=$?
for c in simclr vicreg; do "$PYTHON" -c "from vcs_ssl.config import load_config; c=load_config('configs/cifar10_pilot_$c.yaml'); print('$c policy PASS', c['_meta']['config_hash'][:16])" || rc=2; done
cp "$MANIFEST_ROOT"/cifar10_dev45k_val5k.sha256 reports/ 2>/dev/null
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
