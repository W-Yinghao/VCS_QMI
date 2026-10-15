#!/usr/bin/env bash
#SBATCH --job-name=v8_tifa160
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V8-05 §B (TIFA160 secondary; owner-approved read, labels never touched here): predictions of the frozen V8 units and of the local external routes on
# the 800 TIFA160 images.  Positional modes, several per job (run in sequence):
#   v8-export        all 24 formal units -> <unit>/pred/tifa160.jsonl                       (env v7_fga)
#   b0-native        released FGA-BLIP2 through its native eval harness (as bench-native)    (env v7_fga)
#   b1               VQAScore CLIP-FlanT5-XXL (v7_vqascore.py rank)                          (env v7_vqascore)
#   tifa-qg          TIFA question generation + filter for the 160 prompts                   (env v7_tifa)
#   tifa-score       TIFA scores on the 800 images                                           (env v7_tifa)
#   dsg-gen          DSG generation with Qwen2.5-32B-Instruct (EXPOSED: in-context tifa160)  (env v7x_llm; >= 80 GB GPU)
#   dsg-score        DSG scores with mPLUG-large                                             (env v7_tifa)
# Code (scripts/v8, v7, v7x of THIS worktree) is snapshotted at job start.  Outputs: /projects/EEG-foundation-model/yinghao/outputs_v8/confirm_v1/tifa160/
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v8; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
SNAP=$(mktemp -d /tmp/v8_tifa160_${SLURM_JOB_ID}_XXXX); trap "rm -rf $SNAP" EXIT; mkdir -p $SNAP/scripts; cp -r scripts/v8 scripts/v7 scripts/v7x $SNAP/scripts/
E=/projects/EEG-foundation-model/yinghao/envs/v7_fga; PY="$E/bin/python -I"; BIG=/projects/EEG-foundation-model/yinghao/outputs_v7; B=/projects/EEG-foundation-model/yinghao/outputs_v8
X=$B/confirm_v1/tifa160; mkdir -p $X; IN=$X/tifa160_inputs.jsonl; IMG=/projects/EEG-foundation-model/yinghao/data/tifa160/images; PR=$X/tifa160_prompts.jsonl
T=/projects/EEG-foundation-model/yinghao/envs/v7_tifa/bin/python; L=/projects/EEG-foundation-model/yinghao/envs/v7x_llm/bin/python
export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf_cache_v7x MODELSCOPE_CACHE=$BIG/x30_bench/modelscope_cache V7X_LOCAL=/tmp/v7x_${SLURM_JOB_ID}; mkdir -p $V7X_LOCAL
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) code=$(cat $SNAP/scripts/v8/v8_export_tifa160.py $SNAP/scripts/v8/v8_confirm.py | sha256sum | cut -c1-16) modes=$*"
[ -f "$PR" ] || $PY - "$IN" "$PR" <<'EOF'
import json, sys
seen = {}; [seen.setdefault(json.loads(l)["prompt_id"], json.loads(l)["prompt"]) for l in open(sys.argv[1])]
open(sys.argv[2], "w").write("".join(json.dumps({"prompt_id": k, "prompt": v}) + "\n" for k, v in sorted(seen.items()))); print("prompts", len(seen))
EOF
rc_all=0
for what in "$@"; do t0=$(date +%s); echo "=== $(date -u +%FT%TZ) $what"
  case "$what" in
    v8-export) $PY $SNAP/scripts/v8/v8_export_tifa160.py --units $(ls -d $B/units/V8-FIT-*) --inputs $IN --images $IMG; rc=$?;;
    b0-native) ck=/projects/EEG-foundation-model/yinghao/models/fga_blip2_released/fga_blip2.pth; U=$X/native_B0_released; mkdir -p $U; R=$BIG/n0/run_eval_tifa160_B0_released; mkdir -p $R
       ln -sfn /projects/EEG-foundation-model/yinghao/models/bert-base-uncased $R/bert-base-uncased
       (cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7 && $PY scripts/v7/v7_native.py evalset-rank --inputs $IN --out $U/evalset_tifa160.json) && \
       (cd $R && $E/bin/python -I /home/infres/yinwang/CS_QMI/ssl_pilot_v7/scripts/v7/v7_native_launch.py eval.py --data_file $U/evalset_tifa160.json --save_path $U/native_eval_tifa160.json --model_path "$ck" --dataset_dir $IMG/); rc=$?;;
    b1) B1W=$BIG/b1_work; mkdir -p $B1W; ln -sfn /projects/EEG-foundation-model/yinghao/models/hf_cache_v7/hub $B1W/hf_cache
       (cd $B1W && PATH=/projects/EEG-foundation-model/yinghao/envs/v7_vqascore/bin:$PATH HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf_cache_v7 /projects/EEG-foundation-model/yinghao/envs/v7_vqascore/bin/python -I \
          /home/infres/yinwang/CS_QMI/ssl_pilot_v7/scripts/v7/v7_vqascore.py rank --inputs $IN --images $IMG --out $X/b1_vqascore_clipflant5xxl.jsonl --batch 16); rc=$?;;
    tifa-qg) (cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7 && $T -I scripts/v7x/x30_tifa.py qg --prompts $PR --out $X/tifa_qa.jsonl); rc=$?;;
    tifa-score) (cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7 && $T -I scripts/v7x/x30_tifa.py score --qa $X/tifa_qa.jsonl --inputs $IN --image-root $IMG --out $X/tifa_scores_0of1.jsonl --shard 0/1); rc=$?;;
    dsg-gen) (cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7 && $L -I scripts/v7x/x30_dsg.py gen --prompts $PR --out $X/dsg_gen_0of1.jsonl --shard 0/1); rc=$?;;
    dsg-score) (cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7 && $T -I scripts/v7x/x30_dsg.py score --dsg $X/dsg_gen_*.jsonl --inputs $IN --image-root $IMG --out $X/dsg_scores_0of1.jsonl --shard 0/1); rc=$?;;
    *) echo "unknown mode $what"; rc=2;;
  esac
  echo "=== rc=$rc $what $(( $(date +%s) - t0 )) s"; [ $rc -ne 0 ] && rc_all=$rc
done
rm -rf $V7X_LOCAL; echo "exit=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
