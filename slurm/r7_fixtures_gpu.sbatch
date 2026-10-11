#!/usr/bin/env bash
#SBATCH --job-name=r7_fix_gpu
#SBATCH --partition=H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r7 R7-20 GPU fixtures: LLaVA-1.5-7B host + SparseVLM official code (torch 2.1.2 + flash-attn 2.3.3 -> H100 / L40S only; no Blackwell kernels).
# First arg: d1all (full, SparseVLM 192, SparseVLM 64, then score) | d2pope | d1qwen (pass --partition=RTX6000PRO,H100,L40S).  Env /projects/EEG-foundation-model/yinghao/envs/sparsevlm, offline.
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; OUT=/home/infres/yinwang/CS_QMI/outputs/R7_fixtures; FX=/projects/EEG-foundation-model/yinghao/data/r7_fixtures
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf PYTHONNOUSERSITE=1   # keep ~/.local out of the pinned env
mkdir -p "$OUT" "$FX"; [ -e "$OUT/fixture" ] || ln -s "$FX" "$OUT/fixture"
cd /home/infres/yinwang/CS_QMI/ssl_pilot; what="$1"; shift
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ)"
case "$what" in
  d1all)
    rc=0
    $E/bin/python scripts/r7_fixtures_d1.py build --out "$OUT" || rc=1
    [ $rc -eq 0 ] && { $E/bin/python scripts/r7_fixtures_d1.py run --out "$OUT" --config full || rc=2; }
    [ $rc -eq 0 ] && { $E/bin/python scripts/r7_fixtures_d1.py run --out "$OUT" --config sparsevlm --retain 192 || rc=3; }
    [ $rc -eq 0 ] && { $E/bin/python scripts/r7_fixtures_d1.py run --out "$OUT" --config sparsevlm --retain 64 || rc=4; }
    [ $rc -eq 0 ] && { $E/bin/python scripts/r7_fixtures_d1.py score --out "$OUT" || rc=5; };;
  d2pope) $E/bin/python scripts/r7_fixtures_d2.py pope --out "$OUT" "$@"; rc=$?;;
  d1rest)   # SparseVLM 192 / 64, then the Qwen2.5-VL host, then score (H100 / L40S because of the SparseVLM env)
    Q=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; rc=0
    $E/bin/python scripts/r7_fixtures_d1.py run --out "$OUT" --config sparsevlm --retain 192 || rc=3
    [ $rc -eq 0 ] && { $E/bin/python scripts/r7_fixtures_d1.py run --out "$OUT" --config sparsevlm --retain 64 || rc=4; }
    $Q/bin/python scripts/r7_fixtures_d1.py run --out "$OUT" --config qwen25vl || rc=$((rc + 10))
    $E/bin/python scripts/r7_fixtures_d1.py score --out "$OUT" || rc=$((rc + 100));;
  d1qwen)   # Qwen2.5-VL-7B-Instruct host (D1 plan's named host) in the vl_baselines env (torch 2.9 cu128: any of the three partitions), then rescore
    Q=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; rc=0
    $Q/bin/python scripts/r7_fixtures_d1.py run --out "$OUT" --config qwen25vl || rc=6
    [ $rc -eq 0 ] && { $E/bin/python scripts/r7_fixtures_d1.py score --out "$OUT" || rc=5; };;
  d1fulltime)   # full LLaVA host again under a separate tag (same-GPU timing reference for the SparseVLM runs), then score
    rc=0; $E/bin/python scripts/r7_fixtures_d1.py run --out "$OUT" --config full --tag "$1" || rc=2
    $E/bin/python scripts/r7_fixtures_d1.py score --out "$OUT" || rc=$((rc + 100));;
  *) echo "usage: d1all | d2pope | d1qwen | d1rest | d1fulltime TAG"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
