#!/usr/bin/env bash
#SBATCH --job-name=v9_ncr_eval
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --nice=400
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-03 native A-table reproduction: the released NCR checkpoints evaluated with NCR's own evaluation.py (evalrank; two-network average) on the
# official test splits (Flickr 1K; COCO 5K and 5 x 1K).  Evaluation only — nothing is trained or selected.  Output under outputs_v9/ncr_eval.
E=/projects/EEG-foundation-model/yinghao/envs/v9_sgr; C=/projects/EEG-foundation-model/yinghao/code_v9/2021-NeurIPS-NCR/NCR
DATA=${DATA:?directory containing f30k_precomp/ and coco_precomp/}; VOCAB=${VOCAB:?directory containing f30k_precomp_vocab.json and coco_precomp_vocab.json}
M=/projects/EEG-foundation-model/yinghao/models/ncr/zips; O=/projects/EEG-foundation-model/yinghao/outputs_v9/ncr_eval; mkdir -p $O
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) ncr=$(git -C $C/.. rev-parse --short HEAD) data=$DATA"
export PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1; cd $C; rc=0   # torch >= 2.6 loads with weights_only=True by default; the released checkpoints pickle argparse.Namespace
for spec in "f30k_0 f30k_precomp test 0" "f30k_0.2 f30k_precomp test 0" "f30k_0.5 f30k_precomp test 0" "coco_0 coco_precomp testall 0" "coco_0 coco_precomp testall 1" "coco_0.2 coco_precomp testall 0" "coco_0.2 coco_precomp testall 1" "coco_0.5 coco_precomp testall 0" "coco_0.5 coco_precomp testall 1"; do
  set -- $spec; ck=$M/ncr_$1_model_best.pth.tar; out=$O/ncr_$1_$3_fold5$4.txt; [ -s $out ] && { echo "[skip] $out"; continue; }
  echo "[eval] $(date -u +%FT%TZ) $1 $2 split=$3 fold5=$4"
  $E/bin/python - "$ck" "$DATA" "$VOCAB" "$2" "$3" "$4" > $out.tmp 2>&1 <<'PY'
import sys; sys.argv, (ck, data, vocab, name, split, fold5) = sys.argv[:1], sys.argv[1:7]
import evaluation as EV  # NCR/evaluation.py (unchanged)
EV.evalrank(ck, data_path=data, vocab_path=vocab, split=split, fold5=(fold5 == "1"))  # prints rsum / R@K as the official script does
PY
  r=$?; mv $out.tmp $out; echo "[done] rc=$r $(grep -i -E 'rsum|Average|r1|R@' $out | tail -4 | tr '\n' ' ')"; [ $r -eq 0 ] || rc=1
done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
