#!/usr/bin/env bash
#SBATCH --job-name=v8_formal
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V8 formal units (V8-03 FROZEN; configs/FIT_UNITS.csv).  MODE=fit ARM=<arm> [SEEDS="0+1+2"]: the arm's lr comes from the frozen rule
# (v8_select_lr.py, which refuses unless all 24 tuning units are done), then train + export (EvalMuse roles, GenAI-Rank and GenAI-Bench candidates;
# predictions only, no label).  MODE=smoke_export SRC=<finished unit dir>: copies its log.json / best.pt to a scratch unit and runs this wrapper's
# export (CAL_SELECT + GenAI-Rank + GenAI-Bench) from the snapshot: the gate for the GenAI-Bench path before any formal unit.
# Code (scripts/v8, v7, v7x) is snapshotted at job start, so later edits never change a running job.  Units are resumable and skip when done.
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v8; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
: "${MODE:?}"
SNAP=$(mktemp -d /tmp/v8_formal_${SLURM_JOB_ID}_XXXX); trap "rm -rf $SNAP" EXIT; mkdir -p $SNAP/scripts; cp -r scripts/v8 scripts/v7 scripts/v7x $SNAP/scripts/
PY="/projects/EEG-foundation-model/yinghao/envs/v7_fga/bin/python -I"; B=/projects/EEG-foundation-model/yinghao/outputs_v8; O=$B/units; mkdir -p $O
export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RANK=/home/infres/yinwang/CS_QMI/outputs/V7/data/genai_rank_inputs.jsonl; BENCH=/home/infres/yinwang/CS_QMI/outputs/V7/data/genai_bench_inputs.jsonl
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) code=$(cat $SNAP/scripts/v8/*.py | sha256sum | cut -c1-16) mode=$MODE"
rc=0
case "$MODE" in
  smoke_export) : "${SRC:?}"; S=$B/smoke/export_bench_${SLURM_JOB_ID}; mkdir -p $S; cp $SRC/log.json $SRC/best.pt $S/
      $PY $SNAP/scripts/v8/v8_train.py export --unit-dir $S --roles CAL_SELECT --rank-inputs $RANK --bench-inputs $BENCH; rc=$?
      [ $rc -eq 0 ] && $PY - "$SRC" "$S" <<'EOF'; rc=$((rc | $?))
import json, sys
src, s = sys.argv[1], sys.argv[2]
a = {json.loads(l)["image_id"]: json.loads(l)["score"] for l in open(f"{src}/pred/genai_rank.jsonl")}
b = {json.loads(l)["image_id"]: json.loads(l)["score"] for l in open(f"{s}/pred/genai_rank.jsonl")}
bench = [json.loads(l) for l in open(f"{s}/pred/genai_bench.jsonl")]
n_bench = sum(1 for _ in open("/home/infres/yinwang/CS_QMI/outputs/V7/data/genai_bench_inputs.jsonl"))
ok_rank = a.keys() == b.keys() and max(abs(a[k] - b[k]) for k in a) < 0.02   # same checkpoint; GPU type may differ from the tuning export
ok_bench = len(bench) == n_bench == len({r["image_id"] for r in bench}) and all(1.0 <= r["score"] <= 5.0 for r in bench)
print(f"[smoke] rank re-export max |diff| {max(abs(a[k] - b[k]) for k in a):.2e} (ok {ok_rank}); bench rows {len(bench)} / {n_bench}, scores in [1, 5] (ok {ok_bench})")
sys.exit(0 if ok_rank and ok_bench else 1)
EOF
      echo "smoke_rc=$rc" ;;
  fit) : "${ARM:?}"; SEEDS=${SEEDS:-0+1+2}
      LR=$($PY $SNAP/scripts/v8/v8_select_lr.py --out $B/lr_choice/lr_choice_${SLURM_JOB_ID}.json --arm $ARM) || { echo "lr rule refused"; echo "exit=4 end=$(date -u +%FT%TZ)"; exit 4; }
      echo "[formal] $ARM lr $LR (frozen rule; $B/lr_choice/lr_choice_${SLURM_JOB_ID}.json)"
      for seed in ${SEEDS//+/ }; do u=$O/V8-FIT-$ARM-s$seed
        echo "[formal] FIT $ARM lr $LR seed $seed -> $(basename $u) start $(date -u +%FT%TZ)"
        $PY $SNAP/scripts/v8/v8_train.py train --arm $ARM --lr $LR --seed $seed --unit-dir $u --workers 8 && \
          $PY $SNAP/scripts/v8/v8_train.py export --unit-dir $u --rank-inputs $RANK --bench-inputs $BENCH; r=$?
        echo "[formal] $(basename $u) done (rc $r) $(date -u +%FT%TZ)"; [ $r -eq 0 ] || rc=1
      done ;;
  *) echo "unknown MODE $MODE"; rc=2 ;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
