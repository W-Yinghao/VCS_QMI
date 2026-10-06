#!/usr/bin/env bash
#SBATCH --job-name=mv6_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# MV6 smokes before the P142–P144 freeze (outputs under outputs/MV6_smoke, never reports/):
#  1 CIFAR-10 regression of the patched P109 extractor vs the HEAD version (CPU device, smoke sizes): tensors must be identical
#  2 CIFAR-100 extraction + audit (power, level) + effects smoke   3 fp32-map extraction + P144 smoke (map gate printed)
#  4 P142 block A smoke: 2 repeats of colour_s0.2:2000 seed 612 with the P116 reproduction gate (CPU device, as P116)
#  5 P142 block B extraction smoke (P41 SimCLR seed 1, --limit 3000)
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
banner "MV6 smoke"; S=$OUTPUT_ROOT/MV6_smoke; mkdir -p $S; T=$(mktemp -d); rc_all=0
step() { echo "[step] $(date -u +%FT%TZ) $*"; }
chk() { local rc=$?; echo "[step-exit] rc=$rc"; [ $rc -ne 0 ] && rc_all=$rc; }
# 1
git show HEAD:scripts/p109_i1_features.py | sed "s|REPO = Path(__file__).resolve().parents\[1\]|REPO = Path('$REPO_ROOT')|" > $T/p109_orig.py
step "1 regression (CPU)"
CUDA_VISIBLE_DEVICES= $PYTHON $T/p109_orig.py --runs P107_AP3_views4_800ep_seed1 --out $S/reg_orig --smoke --workers 4; chk
CUDA_VISIBLE_DEVICES= $PYTHON scripts/p109_i1_features.py --runs P107_AP3_views4_800ep_seed1 --out $S/reg_new --smoke --workers 4; chk
$PYTHON - <<PY
import torch
a = torch.load("$S/reg_orig/P107_AP3_views4_800ep_seed1/features.pt", weights_only=False); b = torch.load("$S/reg_new/P107_AP3_views4_800ep_seed1/features.pt", weights_only=False)
same = all(torch.equal(a["features"][v][k], b["features"][v][k]) for v in a["features"] for k in a["features"][v]) and all(torch.equal(a[k], b[k]) for k in ("ids", "part", "y"))
same = same and all(torch.equal(a["classifier"][k], b["classifier"][k]) for k in a["classifier"])
print("REGRESSION_C10_IDENTICAL", same); raise SystemExit(0 if same else 3)
PY
chk
# 2
step "2 CIFAR-100 extraction + audit"
$PYTHON scripts/p109_i1_features.py --runs P107_AP3_c100_views4_800ep_seed1 --out $S/c100 --smoke; chk
$PYTHON scripts/p109_i1_audit.py --features $S/c100 --run P107_AP3_c100_views4_800ep_seed1 --family colour --cells "planted:0.1:400" --repeats 2 --perms 20 --seed 1 --out $S/c100_power --smoke; chk
$PYTHON scripts/p109_i1_audit.py --features $S/c100 --run P107_AP3_c100_views4_800ep_seed1 --family colour --cells "null_label_only:0:400;null_all_planted:0:400" --repeats 2 --perms 20 --seed 2 --out $S/c100_level --smoke; chk
$PYTHON scripts/p109_i1_audit.py --features $S/c100 --run P107_AP3_c100_views4_800ep_seed1 --family colour --seed 3 --out $S/c100_effects --effects --smoke; chk
# 3
step "3 fp32 maps + P144"
$PYTHON scripts/p109_i1_features.py --runs P107_AP3_views4_800ep_seed1 --out $S/maps --smoke --fp32-maps; chk
$PYTHON scripts/p144_compress.py --runs P107_AP3_views4_800ep_seed1 --features $S/maps --out $S/p144 --smoke; chk
# 4
step "4 P142 block A gate (CPU)"
CUDA_VISIBLE_DEVICES= $PYTHON scripts/p142_table3.py --features $OUTPUT_ROOT/P45_precheck_D_simclr --family colour --cells "colour_s0.2:2000" --seed 612 \
  --check reports/P116/full_simclr_colour.json --out $S/p142_A_colour --smoke; chk
# 5
step "5 block B extraction (limit 3000)"
$PYTHON scripts/precheck_d_features.py --run P41_simclr_views4_800ep_seed1 --checkpoint epoch_800.pt --out $S/d45_simclr_s1_colour --strengths 0,0.1,0.2 --limit 3000; chk
rm -rf $T; echo "exit=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
