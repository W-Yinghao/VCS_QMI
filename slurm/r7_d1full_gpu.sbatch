#!/usr/bin/env bash
#SBATCH --job-name=r7_d1full_gpu
#SBATCH --partition=H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-30 D1-FULL LLaVA-env GPU units (SparseVLM env: torch 2.1.2 + flash-attn 2.3.3, no Blackwell kernels -> H100 / L40S only, never RTX6000PRO).
# One host process per job; every unit resumes where it stopped.  Positional arguments only (no --export).
#   bundle <id>                 teacher bundle (frozen D1-10 teacher, strict image source, probe gate)        -> teacher/bundle_<id>.ok.json
#   cache [--pca-basis B]       feature cache, frozen D1-20 format (B = fitfull | frozen; coordinator)       -> cache/cache_d1_full.json
#   eval <bench> <i/n>          frozen D1-30 runner: D1-FULL selector rows, D1-30 sets, own root d1_30/        -> d1_30/pred/<bench>/done_shard<i>of<n>.json
#   devrisk                     frozen D1-30 DEV-risk secondary for the D1-FULL units (reads the sealed D1-10 DEV records; after the D1-30 freeze)
#   timing <bench>              frozen D1-30 timing of the D1-FULL dynamic rows (reference GPU nodesumo01 only)  -> d1_30/timing/<bench>/timing_<bench>.json
#   smoke                       GPU smoke, code checks only (scripts/r7_d1full_smoke.py)
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; DF=/home/infres/yinwang/CS_QMI/outputs/R7_D1_FULL
P=reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf PYTHONNOUSERSITE=1
cd /home/infres/yinwang/CS_QMI/ssl_pilot; what="$1"; shift
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ)"
case "$what" in
  bundle) $E/bin/python scripts/r7_d1full_teacher.py bundle "$1" --manifest-dir "$DF/data" --out "$DF/teacher" --probe-report "$DF/smoke/probe/probe_report.json"; rc=$?;;
  cache) $E/bin/python scripts/r7_d1full_cache.py --manifest-dir "$DF/data" --out "$DF/cache" "$@"; rc=$?;;
  eval) $E/bin/python scripts/r7_d1_30_run.py --sets-dir "$D1/d1_30/sets" --bench "$1" --configs-file "$DF/d1_30/configs/configs_$1.txt" --shard "$2" \
          --out "$DF/d1_30/pred" --selectors-dir "$DF/selectors" --pca "$DF/cache/pca.pt"; rc=$?;;
  devrisk) grep -q 'd1_full_staged_freeze' "$P" || { echo "D1-FULL not frozen"; exit 3; }
           $E/bin/python scripts/r7_d1_30_devrisk.py --out "$DF/d1_30/devrisk" --protocol "$P" --lr-choice "$DF/selectors/lr_choice.json" \
             --selectors-dir "$DF/selectors" --pca "$DF/cache/pca.pt"; rc=$?;;
  timing) $E/bin/python scripts/r7_d1_30_timing.py --sets-dir "$D1/d1_30/sets" --bench "$1" --configs-file "$DF/d1_30/configs/timing_$1.txt" \
            --out "$DF/d1_30/timing/$1" --selectors-dir "$DF/selectors" --pca "$DF/cache/pca.pt"; rc=$?;;
  timing_all)   # owner rule 2026-10-13: the nodesumo01 timing units run back-to-back in ONE job; resumable (finished benchmarks are skipped)
          rc=0; for b in gqa_dev textvqa_val chartqa_val; do
            [ -f "$DF/d1_30/timing/$b/timing_$b.json" ] && { echo "timing $b already present"; continue; }
            $E/bin/python scripts/r7_d1_30_timing.py --sets-dir "$D1/d1_30/sets" --bench "$b" --configs-file "$DF/d1_30/configs/timing_$b.txt" \
              --out "$DF/d1_30/timing/$b" --selectors-dir "$DF/selectors" --pca "$DF/cache/pca.pt" || { rc=1; echo "timing $b failed"; }
          done
          [ $rc -eq 0 ] && echo "{\"benches\": [\"gqa_dev\", \"textvqa_val\", \"chartqa_val\"], \"done\": \"$(date -u +%FT%TZ)\"}" > "$DF/d1_30/timing/timing_bundle.done.json";;
  smoke) $E/bin/python scripts/r7_d1full_smoke.py --out "$DF/smoke" "$@"; rc=$?;;
  *) echo "usage: bundle <id> | cache [--pca-basis fitfull|frozen] | eval <bench> <i/n> | devrisk | timing <bench> | timing_all | smoke"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
