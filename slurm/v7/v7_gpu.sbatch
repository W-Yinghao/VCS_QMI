#!/usr/bin/env bash
#SBATCH --job-name=v7_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V7 GPU units (reports/V7/): env envs/v7_fga (torch 2.9 cu128 -> RTX6000PRO / H100 / L40S).  Positional arguments only.
#   smoke-m <arm>               M runner: 40 steps + 64 CAL_SELECT images (code check + step time; no result is used)
#   smoke-n0                    native FGA trainer on 70 FIT records for 1 epoch + native eval.py on 20 CAL_SELECT items (code check + step time)
#   m <arm> <lr> <seed> <unit>  M unit: train (5 epochs, CAL_SELECT selection) then export CAL_SELECT / CAL_PROB / DEV predictions
#   n0 <seed> <unit>            N0 unit: stage FIT + eval images node-local, native train.py (5 epochs, last epoch), native eval.py on the eval roles
#   parse <cache> <prompts...>  prompt-only element parser cache for external prompts (before any image score exists)
#   vqascore-rank <out>         baseline B1 (t2v-metrics 3.0 CLIP-FlanT5-XXL) on the GenAI-Rank candidates
#   native-rank <ckpt> <tag> | native-evalmuse <ckpt> <tag> <roles...>   FGA's eval.py (B0 released weights or an N0 checkpoint)
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
E=/projects/EEG-foundation-model/yinghao/envs/v7_fga; PY="$E/bin/python -I"; BIG=/projects/EEG-foundation-model/yinghao/outputs_v7; D=/home/infres/yinwang/CS_QMI/outputs/V7/data
export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) branch=$(git branch --show-current) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts/v7 tests/v7 slurm/v7 | wc -l)"
echo "code: $(cat scripts/v7/*.py tests/v7/*.py slurm/v7/*.sbatch | sha256sum | cut -c1-16) (sha256 of the V7 scripts, tests and wrappers)"
echo "local disk: $(df -h --output=avail /tmp | tail -1) /tmp, $(df -h --output=avail /dev/shm | tail -1) /dev/shm"
local_dir() {   # node-local scratch for the N0 image copy: /tmp if >= 60 GB free, else /dev/shm if >= 60 GB free, else stop (never silently on NFS)
  local need=60; for base in /tmp /dev/shm; do avail=$(( $(df -B1G --output=avail $base | tail -1) )); [ "$avail" -ge "$need" ] && { echo "$base/v7_${SLURM_JOB_ID}"; return 0; }; done; return 1; }
what="$1"; shift
case "$what" in
  smoke-m) arm="$1"; $PY scripts/v7/v7_train.py train --arm "$arm" --lr 1e-5 --seed 17 --unit-dir $BIG/smoke/m_$arm --max-steps 40 --eval-n 64 --workers 8; rc=$?;;
  m) arm="$1"; lr="$2"; seed="$3"; unit="$4"
     $PY scripts/v7/v7_train.py train --arm "$arm" --lr "$lr" --seed "$seed" --unit-dir $BIG/units/$unit --workers 8; rc=$?
     [ $rc -eq 0 ] && { $PY scripts/v7/v7_train.py export --unit-dir $BIG/units/$unit; rc=$?; };;
  smoke-n0|n0)
     if [ "$what" = smoke-n0 ]; then seed=17; unit=smoke_n0; ep=1; W=$BIG/smoke/n0_work; else seed="$1"; unit="$2"; ep=0; W=$BIG/n0/work; fi
     [ -f $W/dataset/train_mask.json ] || { echo "missing $W/dataset/train_mask.json (run the CPU n0-prep first)"; exit 3; }
     L=$(local_dir) || { echo "no node-local space >= 60 GB for the N0 images"; exit 4; }; mkdir -p $L; trap "rm -rf $L" EXIT
     roles="FIT CAL_SELECT CAL_PROB DEV"; [ "$what" = smoke-n0 ] && roles="FIT CAL_SELECT"
     $PY scripts/v7/v7_native.py stage --dest $L/images --roles $roles || exit 5
     U=$BIG/units/$unit; mkdir -p $U; $PY scripts/v7/v7_native.py yaml --work $W --images $L/images --seed $seed --out-dir $U --max-epoch $ep || exit 6
     R=$BIG/n0/run_$unit; mkdir -p $R; ln -sfn /projects/EEG-foundation-model/yinghao/models/bert-base-uncased $R/bert-base-uncased; cd $R
     PORT=$(( 20000 + SLURM_JOB_ID % 20000 )); $E/bin/python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=$PORT \
        /home/infres/yinwang/CS_QMI/ssl_pilot_v7/scripts/v7/v7_native_launch.py train.py --cfg-path $U/fga_blip2_unit.yaml; rc=$?
     cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7
     if [ $rc -eq 0 ]; then
       CK=$(ls -t $U/*/checkpoint_*.pth | head -1); echo "native checkpoint (last epoch): $CK"
       for role in $([ "$what" = smoke-n0 ] && echo CAL_SELECT || echo CAL_SELECT CAL_PROB DEV); do
         $PY scripts/v7/v7_native.py evalset --roles $role --out $U/evalset_$role.json || exit 7
         [ "$what" = smoke-n0 ] && $PY -c "import json,sys; d=json.load(open(sys.argv[1])); json.dump(d[:20], open(sys.argv[1],'w'))" $U/evalset_$role.json
         (cd $R && $E/bin/python -I /home/infres/yinwang/CS_QMI/ssl_pilot_v7/scripts/v7/v7_native_launch.py eval.py --data_file $U/evalset_$role.json \
            --save_path $U/native_eval_$role.json --model_path "$CK" --dataset_dir $L/images/) || { rc=8; break; }
       done
     fi;;
  vqascore-rank) out="$1"                   # B1: t2v-metrics 3.0 CLIP-FlanT5-XXL on every GenAI-Rank candidate (L40S / H100: torch 2.5.1 has no Blackwell kernels)
     L=$(local_dir) || { echo "no node-local space >= 60 GB"; exit 4; }; mkdir -p $L; trap "rm -rf $L" EXIT
     $PY scripts/v7/v7_native.py stage-rank --dest $L/rank || exit 5
     W=$BIG/b1_work; mkdir -p $W; ln -sfn /projects/EEG-foundation-model/yinghao/models/hf_cache_v7/hub $W/hf_cache
     (cd $W && PATH=/projects/EEG-foundation-model/yinghao/envs/v7_vqascore/bin:$PATH HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf_cache_v7 /projects/EEG-foundation-model/yinghao/envs/v7_vqascore/bin/python -I \
        /home/infres/yinwang/CS_QMI/ssl_pilot_v7/scripts/v7/v7_vqascore.py rank --inputs $D/genai_rank_inputs.jsonl --images $L/rank --out "$out" --batch 16); rc=$?;;
  native-rank|native-evalmuse) ck="$1"; tag="$2"; shift 2      # FGA's own eval.py with checkpoint ck (B0 = released weights; or an N0 unit) on GenAI-Rank / EvalMuse roles
     L=$(local_dir) || { echo "no node-local space >= 60 GB"; exit 4; }; mkdir -p $L; trap "rm -rf $L" EXIT; U=$BIG/native_eval/$tag; mkdir -p $U
     R=$BIG/n0/run_eval_$tag; mkdir -p $R; ln -sfn /projects/EEG-foundation-model/yinghao/models/bert-base-uncased $R/bert-base-uncased; rc=0
     if [ "$what" = native-rank ]; then
       $PY scripts/v7/v7_native.py stage-rank --dest $L/images || exit 5; $PY scripts/v7/v7_native.py evalset-rank --out $U/evalset_rank.json || exit 7; sets="rank"
     else
       $PY scripts/v7/v7_native.py stage --dest $L/images --roles "$@" || exit 5; sets="$*"
       for role in "$@"; do $PY scripts/v7/v7_native.py evalset --roles $role --out $U/evalset_$role.json || exit 7; done
     fi
     for set in $sets; do
       (cd $R && $E/bin/python -I /home/infres/yinwang/CS_QMI/ssl_pilot_v7/scripts/v7/v7_native_launch.py eval.py --data_file $U/evalset_$set.json \
          --save_path $U/native_eval_$set.json --model_path "$ck" --dataset_dir $L/images/) || { rc=8; break; }
     done;;
  parse) out="$1"; shift                   # prompt-only parser cache (Qwen2.5-VL-7B text-only, vl_baselines env); args: <cache.jsonl> <prompts.jsonl ...>
     VLPY="/projects/EEG-foundation-model/yinghao/envs/vl_baselines/bin/python -I"; $VLPY scripts/v7/v7_parser.py run --prompts "$@" --out "$out" --batch 16; rc=$?;;
  *) echo "usage: smoke-m <arm> | smoke-n0 | m <arm> <lr> <seed> <unit> | n0 <seed> <unit> | parse <cache> <prompts...>"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
