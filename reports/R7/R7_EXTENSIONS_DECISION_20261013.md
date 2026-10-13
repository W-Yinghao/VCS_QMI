# r7 extensions decided 2026-10-13 11:39:56 UTC (coordinator; commit dbf628d) — before any R7-40c result existed (result files written 11:40:44 UTC)

Context: the owner asked "没有别的要提交的实验了吗" (are there no other experiments to submit?).  R7-40c generation completed at 11:37 UTC;
its score job (1037387) was running and no R7-40c result file had been opened when this note was committed.  The pre-registered queue
continues first (R7-50 phase Q, D1-FULL evaluation and timing, R7-50 phase F).  Two extensions of the same D1 line are added after it,
each with its own pre-registration (protocol draft → freeze → smoke → units), on validation data only (DEV / val; no official-test read —
an official-test column would need a new owner approval).

## E1 — external Qwen2.5-VL baselines (fills R7-40c's disclosed gap, R7_40C draft external_baseline_OPEN_R740C_4)
- Methods: MMTok (question-conditioned, official Qwen2.5-VL code, Apache-2.0) and HiPrune (question-agnostic, official code, MIT), each
  run as shipped in its own pinned environment (batch 1), at the fixed fractions 0.125 / 0.25 / 0.5 / 0.75 of the merged visual tokens,
  on R7-40c's GQA DEV / TextVQA val / ChartQA val items and prompts.
- Reported as separate host columns (owner I1: never a same-condition head-to-head), outside every primary family; quality at fixed
  fractions with the frozen evaluators, batch-1 latency secondary.  IF-Prune stays excluded (no licence file).
- Chosen from the admissible list recorded at the R7-40c freeze (2026-10-13 02:00 UTC), not from any result.

## E2 — Qwen full-data replication (the Qwen analogue of D1-FULL; owner directive "多做全量实验")
- The frozen R7-40 pipeline (R7-40a teacher code, R7-40b recipe and CAL rule, R7-40c evaluation) on D1-FULL's FIT_FULL (70,256 images),
  so that the cross-host statement rests on full-data selectors as the LLaVA one does (D1-FULL, claim rule DF-8).
- First step: feasibility (teacher cost on the 240,827 new questions; selector memory at 7× the data — the 10k Qwen selectors peaked at
  about 49 GB and D1-FULL's LLaVA selectors at 77.5 GiB reserved), then its own protocol and freeze before any unit runs.
