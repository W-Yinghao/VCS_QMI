#!/usr/bin/env bash
#SBATCH --job-name=v8_eval
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V8 evaluation step (reports/V8_UTILITY/V8_03_PROTOCOL_FROZEN.yaml `evaluation:`).  Env envs/v7_fga (python -I).  Positional:
#   gate      tests/v8 (all) + `v8_eval.py dry-run` on the units present (missing units are listed; nothing is read from any label file)
#   dry-run   the inventory / external / scorer checks only (no label read)
#   run OUT   append the GenAI-Rank ledger row, then `v8_eval.py run --allow-scoring --out OUT` (OUT must be new; the request file must exist)
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v8; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
E=/projects/EEG-foundation-model/yinghao/envs/v7_fga; PY="$E/bin/python -I"; REP=reports/V8_UTILITY
echo "host=$(hostname) start=$(date -u +%FT%TZ) branch=$(git branch --show-current) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts/v8 tests/v8 slurm/v8 | wc -l)"
echo "code: $(cat scripts/v8/v8_eval.py scripts/v7/v7_score.py scripts/v7/v7_common.py | sha256sum | cut -c1-16) (v8_eval + frozen scorer + v7_common)"
what="$1"; shift; rc=0
case "$what" in
  gate) $PY -m pytest -q -p no:cacheprovider tests/v8 2>&1 | tail -5; rc=${PIPESTATUS[0]}
        $PY scripts/v8/v8_eval.py dry-run "$@"; r=$?; echo "dry-run rc=$r"; [ $r -eq 0 ] || rc=$r;;
  dry-run) $PY scripts/v8/v8_eval.py dry-run "$@"; rc=$?;;
  run) OUT="$1"; shift; [ -n "$OUT" ] || { echo "run needs OUT"; exit 2; }
       [ -f $REP/V8_EVAL_SCORING_REQUEST.md ] || { echo "refusing: $REP/V8_EVAL_SCORING_REQUEST.md missing"; exit 3; }
       [ -e "$OUT" ] && { echo "refusing: $OUT exists"; exit 4; }
       SEAL=/home/infres/yinwang/CS_QMI/outputs/V7/data/genai_rank_seal_report.json
       PH=$($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['prompts_sha256'])" $SEAL); LH=$($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['sealed_sha256'])" $SEAL)
       [ -f $REP/LABEL_ACCESS_LEDGER.csv ] || echo "dataset,prompt_set_hash,label_file_hash,first_access_utc,purpose,user_or_process,authorized_by,affected_model_versions,already_exposed,notes" > $REP/LABEL_ACCESS_LEDGER.csv
       echo "GenAI-Rank human ratings (sealed),$PH,$LH,$(date -u +%FT%TZ) (job $SLURM_JOB_ID; first read inside it),V8-03 evaluation: Tables A / B on GenAI-Rank (primary 8 / same-conditions 8 / attribution 2 / full rows 4 / subset sensitivities) with the frozen V7 scorer,slurm job $SLURM_JOB_ID (slurm/v8/v8_eval.sbatch run) commit $(git rev-parse --short HEAD),V8-03 frozen protocol evaluation.primary_external (pre-specified follow-up on the same benchmark; frozen before V7's read); owner's idle-GPU / package-completion rules; request file $REP/V8_EVAL_SCORING_REQUEST.md,24 formal units V8-FIT-{U-CS;U-BCE;U-RPS;U-CE;U-MSE;U-L1;U-RANK;U-LIST}-s{0;1;2} at the lr-rule lrs,yes: V7C-04 read the same file on 2026-10-15 (V7 families); the file was never used for any V8 selection,the external predictions are the V7X-50 v3 frozen files; GenAI-Bench and TIFA160 remain sealed" >> $REP/LABEL_ACCESS_LEDGER.csv
       $PY scripts/v8/v8_eval.py run --allow-scoring --out "$OUT" --python $E/bin/python; rc=$?;;
  *) echo "unknown mode $what"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
