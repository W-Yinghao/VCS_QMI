#!/usr/bin/env bash
#SBATCH --job-name=diag_cpu_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/diag_cpu_gate_%j.out
# D line CPU gate (P93 D1 / D2 + P83 addendum 1): unit tests, then a CPU smoke of each script.  No GPU, no full data pass.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=8
S=/home/infres/yinwang/CS_QMI/outputs/P93_D_diag/smoke; mkdir -p "$S"
banner "D line CPU gate -> $S"
"$PYTHON" -m pytest -q tests/test_diag_v2.py -p no:cacheprovider 2>&1 | tail -30; rc1=${PIPESTATUS[0]}
echo "=== D1 smoke"; "$PYTHON" scripts/diag_p72_adapters.py --smoke --cpu --smoke-n 1024 --smoke-epochs 2 --shift-dirs animal=/home/infres/yinwang/CS_QMI/outputs/P49_precheck_A \
  --out "$S/d1_smoke" --ckpt-dir "$S/p72_adapters" 2>&1 | tail -20; rc2=${PIPESTATUS[0]}
echo "=== D2 smoke"; "$PYTHON" scripts/diag_same_class_negatives.py --smoke --cpu --ckpts P35_vcs_a5_views4_800ep_seed0:epoch_800,P41_simclr_views4_800ep_seed0:epoch_800 --out "$S/d2_smoke" 2>&1 | tail -20; rc3=${PIPESTATUS[0]}
echo "=== P83 addendum 1 smoke"; "$PYTHON" scripts/diag_p84_supplement.py --smoke --cpu --workers 4 --ckpts P35_vcs_a5_views4_800ep_seed0:epoch_800 --out-dir "$S/p84_supplement" --report "$S/p84_supplement.md" 2>&1 | tail -30; rc4=${PIPESTATUS[0]}
echo "tests_rc=$rc1 d1_rc=$rc2 d2_rc=$rc3 addendum_rc=$rc4 end=$(date -u +%FT%TZ)"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ] && [ $rc4 -eq 0 ]
