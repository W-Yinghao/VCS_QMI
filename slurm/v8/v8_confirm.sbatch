#!/usr/bin/env bash
#SBATCH --job-name=v8_confirm
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V8-05 confirmation (reports/V8_UTILITY/V8_05_CONFIRMATION_PROTOCOL_*.yaml).  Env envs/v7_fga (python -I).  Positional:
#   gate      tests/v8 (all) + `v8_confirm.py dry-run` (inventory; no rating column is ever opened; parquet bytes hashed only with --hash)
#   dry-run OUT [--hash]   the inventory only
#   run OUT   append the GenAI-Bench ledger row, then `v8_confirm.py run --allow-scoring --out OUT` (OUT must be new; the CONFIRMED request file must exist)
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v8; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
E=/projects/EEG-foundation-model/yinghao/envs/v7_fga; PY="$E/bin/python -I"; REP=reports/V8_UTILITY
echo "host=$(hostname) start=$(date -u +%FT%TZ) branch=$(git branch --show-current) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts/v8 tests/v8 slurm/v8 | wc -l)"
echo "code: $(cat scripts/v8/v8_confirm.py | sha256sum | cut -c1-16) (v8_confirm)"
what="$1"; shift; rc=0
case "$what" in
  gate) $PY -m pytest -q -p no:cacheprovider tests/v8 2>&1 | tail -5; rc=${PIPESTATUS[0]}
        OUT=${1:-/projects/EEG-foundation-model/yinghao/outputs_v8/confirm_v1/gate_$SLURM_JOB_ID}
        $PY scripts/v8/v8_confirm.py dry-run --out "$OUT" --hash; r=$?; echo "dry-run rc=$r"; [ $r -eq 0 ] || rc=$r;;
  dry-run) OUT="$1"; shift; $PY scripts/v8/v8_confirm.py dry-run --out "$OUT" "$@"; rc=$?;;
  run) OUT="$1"; shift; [ -n "$OUT" ] || { echo "run needs OUT"; exit 2; }
       grep -q "CONFIRMED" $REP/V8_05_CONFIRMATION_REQUEST.md 2>/dev/null || { echo "refusing: $REP/V8_05_CONFIRMATION_REQUEST.md missing or not CONFIRMED"; exit 3; }
       [ -e "$OUT/confirm.json" ] && { echo "refusing: $OUT/confirm.json exists"; exit 4; }
       PH=$($PY scripts/v8/v8_confirm.py prompt-list | $PY -c "import json,sys; print(json.loads(sys.stdin.read())['sha256_ids'])")
       LH=$(cat /projects/EEG-foundation-model/yinghao/data/genai_bench/data/*.parquet | sha256sum | cut -d' ' -f1)
       [ -f $REP/LABEL_ACCESS_LEDGER.csv ] || echo "dataset,prompt_set_hash,label_file_hash,first_access_utc,purpose,user_or_process,authorized_by,affected_model_versions,already_exposed,notes" > $REP/LABEL_ACCESS_LEDGER.csv
       echo "GenAI-Bench human ratings (HumanRatings column of the 12 parquet shards; 850 non-Rank prompts x 6 generators),$PH,$LH,$(date -u +%FT%TZ) (job $SLURM_JOB_ID; first read inside it),V8-05 confirmation (primary_spearman 4 / primary_pairwise 4 / attribution 2 / same-data descriptive) on the registered untouched source,v8_confirm.py run (job $SLURM_JOB_ID),owner decision 2026-10-15 (see V8_05_CONFIRMATION_REQUEST.md),V8 formal units 8 arms x seeds 0-2 (frozen; no selection after this read),no (750 Rank-overlap prompts excluded; V7 never scored on these prompts),one read; V7 methods are not scored here" >> $REP/LABEL_ACCESS_LEDGER.csv
       $PY scripts/v8/v8_confirm.py run --allow-scoring --out "$OUT"; rc=$?;;
  tifa160-gate) $PY -m pytest -q -p no:cacheprovider tests/v8 2>&1 | tail -5; rc=${PIPESTATUS[0]}
        OUT=${1:-/projects/EEG-foundation-model/yinghao/outputs_v8/confirm_v1/tifa160/gate_$SLURM_JOB_ID}
        $PY scripts/v8/v8_confirm.py tifa160-dry-run --out "$OUT" --hash; r=$?; echo "tifa160-dry-run rc=$r (1 = not all predictions present yet)";;
  tifa160-dry-run) OUT="$1"; shift; $PY scripts/v8/v8_confirm.py tifa160-dry-run --out "$OUT" "$@"; rc=$?;;
  tifa160-run) OUT="$1"; shift; [ -n "$OUT" ] || { echo "tifa160-run needs OUT"; exit 2; }
       grep -q "TIFA160 CONFIRMED" $REP/V8_05_CONFIRMATION_REQUEST.md 2>/dev/null || { echo "refusing: $REP/V8_05_CONFIRMATION_REQUEST.md without 'TIFA160 CONFIRMED'"; exit 3; }
       [ -e "$OUT/tifa160_confirm.json" ] && { echo "refusing: $OUT/tifa160_confirm.json exists"; exit 4; }
       CL=/projects/EEG-foundation-model/yinghao/third_party/closed; DH=$(sha256sum $CL/DSG@7330119/tifa160-likert-anns.csv | cut -d' ' -f1); TH=$(sha256sum $CL/tifa@8e37050/human_annotations/human_annotations.json | cut -d' ' -f1)
       IH=$(sha256sum /projects/EEG-foundation-model/yinghao/outputs_v8/confirm_v1/tifa160/tifa160_inputs.jsonl | cut -d' ' -f1)
       [ -f $REP/LABEL_ACCESS_LEDGER.csv ] || echo "dataset,prompt_set_hash,label_file_hash,first_access_utc,purpose,user_or_process,authorized_by,affected_model_versions,already_exposed,notes" > $REP/LABEL_ACCESS_LEDGER.csv
       echo "TIFA160 human annotations (DSG 5-rater tifa160-likert-anns.csv sha256 $DH; TIFA 2-rater human_annotations.json sha256 $TH; 800 images = 160 text inputs x 5 generators),$IH,$DH;$TH,$(date -u +%FT%TZ) (job $SLURM_JOB_ID; first read inside it),V8-05 section B secondary external evaluation (tifa_primary 4 contrasts Holm; tifa_attribution 2; same-data descriptive; three correlations with intervals; no claim),v8_confirm.py tifa160-run (frozen V8 units + local external routes),owner 2026-10-15 (TIFA160 read approved as a secondary external evaluation of the frozen V8 models; reports/OWNER_DECISIONS_20261015.md),V8 formal units 24 (frozen) + B0 released FGA + B1 VQAScore + B4T TIFA local (+ B4D DSG local if present: exposed in-context),B4D DSG route exposed to DSG tifa160 in-context examples; V8 / FGA / VQAScore / TIFA route unexposed,secondary evaluation added 2026-10-15 — not the registered confirmation source" >> $REP/LABEL_ACCESS_LEDGER.csv
       $PY scripts/v8/v8_confirm.py tifa160-run --allow-scoring --out "$OUT"; rc=$?;;
  *) echo "unknown mode $what"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
