#!/bin/bash
# Incremental TEST step for newly finished retrieval units (run after units finish; idempotent).
# Submits one runfill GPU job (sbatch/increment.sbatch) unless one is already queued/running. That job freezes a new
# TEST_MANIFEST_INC_<utc>.json (units with DONE.json + model_best.pth.tar whose checkpoint sha256 is in no earlier frozen
# manifest; run roots from ROOTS.json), logs it in TEST_ACCESS_LOG.jsonl before any TEST read, evaluates every pending unit
# (DEV-recompute gate + TEST) and rebuilds TEST_RESULTS.json / TEST_TABLES.md.
set -euo pipefail
T=/home/infres/yinwang/CS_QMI/handoff/test_eval_20261101
cd $T
if squeue -u $USER -h -o "%j" | grep -qx te_increment; then echo "te_increment already queued/running; nothing submitted"; exit 0; fi
j=$(sbatch --parsable sbatch/increment.sbatch)
printf "%s\t%s\tte_increment\tsbatch sbatch/increment.sbatch\ttools/test_increment.sh\n" "$(date -u +%FT%TZ)" "$j" >> SUBMITTED.tsv
echo "submitted te_increment $j"
