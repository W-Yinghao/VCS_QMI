#!/usr/bin/env bash
#SBATCH --job-name=v9_ncr_eval
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --nice=400
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-03 native A-table reproduction: the released NCR checkpoints evaluated with NCR's own evaluation.py (evalrank; two-network average) on the
# official test splits (Flickr 1K; COCO 5K and 5 x 1K).  Evaluation only — nothing is trained or selected.  Output under outputs_v9/ncr_eval.
# Memory: job 1042193 (32G) was OOM-killed on every COCO testall eval; NCR's encode_data allocates float64 arrays (25,000 x 36 x 1024 image and
# 25,000 x max_n_word x 1024 caption embeddings) for BOTH networks + the 7.4 GB raw testall features ≈ 48 GB; 96G gives a 2x margin; each eval's
# peak RSS is recorded by /usr/bin/time -v.  An output counts as done only if it contains "rsum" (failed outputs are archived, not skipped).
E=/projects/EEG-foundation-model/yinghao/envs/v9_sgr; C=/projects/EEG-foundation-model/yinghao/code_v9/2021-NeurIPS-NCR/NCR
DATA=${DATA:?directory containing f30k_precomp/ and coco_precomp/}; VOCAB=${VOCAB:?directory containing f30k_precomp_vocab.json and coco_precomp_vocab.json}
M=/projects/EEG-foundation-model/yinghao/models/ncr/zips; O=/projects/EEG-foundation-model/yinghao/outputs_v9/ncr_eval; mkdir -p $O
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) ncr=$(git -C $C/.. rev-parse --short HEAD) data=$DATA mem_request=${SLURM_MEM_PER_NODE:-?}MB"
export PYTHONNOUSERSITE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1; cd $C; rc=0   # torch >= 2.6 loads with weights_only=True by default; the released checkpoints pickle argparse.Namespace
for spec in "f30k_0 f30k_precomp test 0" "f30k_0.2 f30k_precomp test 0" "f30k_0.5 f30k_precomp test 0" "coco_0 coco_precomp testall 0" "coco_0 coco_precomp testall 1" "coco_0.2 coco_precomp testall 0" "coco_0.2 coco_precomp testall 1" "coco_0.5 coco_precomp testall 0" "coco_0.5 coco_precomp testall 1"; do
  set -- $spec; ck=$M/ncr_$1_model_best.pth.tar; out=$O/ncr_$1_$3_fold5$4.txt; grep -q "rsum" $out 2>/dev/null && { echo "[skip] $out (complete)"; continue; }
  echo "[eval] $(date -u +%FT%TZ) $1 $2 split=$3 fold5=$4"; s=$(date +%s)
  /usr/bin/time -v $E/bin/python - "$ck" "$DATA" "$VOCAB" "$2" "$3" "$4" > $out.tmp 2>&1 <<'PY'
import sys; sys.argv, (ck, data, vocab, name, split, fold5) = sys.argv[:1], sys.argv[1:7]
import evaluation as EV  # NCR/evaluation.py (unchanged)
EV.evalrank(ck, data_path=data, vocab_path=vocab, split=split, fold5=(fold5 == "1"))  # prints rsum / R@K as the official script does
PY
  r=$?; mv $out.tmp $out; echo "[done] rc=$r seconds=$(( $(date +%s) - s )) peak_rss_kb=$(grep -o 'Maximum resident set size (kbytes): [0-9]*' $out | grep -o '[0-9]*$') $(grep -E '^rsum' $out | tail -1)"; [ $r -eq 0 ] || rc=1
done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
