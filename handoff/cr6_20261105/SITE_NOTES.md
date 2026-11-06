Publication copy (sample names aliased; numbers and conclusions unchanged).

# CR6 site implementation notes (production diff vs package and vs older trees)

Protocol `CR6_SOURCE_REPLICATE_AND_STUDENT_SCOPE_V1` (name not used before on this server). Package `package/` is a read-only copy of
`cr6_route/VCS_CR5_Review_CR6_Server_20261105` (zip sha c4ce17b5…, MANIFEST 68/68). The package's run_checks.sh passed (12/12 tests +
summarize_cr5) in a scratch copy with env `vcs_aigiqa_ir` (torch 2.9.0, pandas 3.0.6), job 1090928; frozen envs were not touched.

No file of CR1–CR5P, NR4 or the package is modified. All new code is in `code/site/`:

| file | role | derived from |
|---|---|---|
| cr6_source.py | A1 estimator fit | `cr1_prod.fit_estimator`, identical except R1 (GlobalScorer seed) and R2 (numpy batch sampler) — RNG role list in its docstring |
| cr6_graphs.py | A1 graphs (CPU) | calls CR1 `cr1_prod.py match --mode CAP`, CR2 `build_relation_pkg.py` (KEEP_TOP1, λ=.5) and the CR3 `mass_observation` lines unchanged |
| cr6_group.py | matched-group runner | CR5 group pattern; score exports via CR1 `cr1_prod.py score --models-root <CR6 experts>` |
| cr6_tail.py | B TAIL_LORA8 PC student | CR5 COMPLETE_PC step (same draws/build/packed_loss PC/head init/optimizer) + NR4 tail encoder/hidden cache/LoRA injector + NR4 two-pass replay |
| cr6_eval.py | DEV evaluator | NR1 protocol through `nr4.evaluate.score_and_order`; adds a full-gallery current-G ranking (secondary) |
| acceptance_cr6.py | real acceptance | — |
| freeze_cr6.py / cr6_driver.py / summarize_cr6.py | freeze, DAG driver, tables | — |

Site decisions (registered before any CR6 training):
- New source seeds per expert index e: R1 = 2002000+e, R2 = `default_rng(20261105200200+e)` (package MATRIX_NEW values), identical for J and BCE.
  CR1 used R1 = 1001, R2 = `default_rng([1001, e, 51])`. CAL pools (seed 20261104051+e), folds and candidates are unchanged.
- TAIL head lr schedule = CR1 `CosineAnnealingLR(T_max=8000, eta_min=1e-5)` on the head AdamW; LoRA uses its own AdamW +
  `CosineAnnealingLR(eta_min=1e-6)` (per-parameter identical to a second param group); one global clip (G + LoRA, 2.0), one step each.
- TAIL always uses the exact two-pass replay backward (no direct/replay switching by batch size); direct-vs-replay equality is an acceptance check.
- Hidden cache index = CR1 bank index = NR1 FIT cache index (checked: same image/text id order, all images kept, globals byte-equal).
- B group jobs load the hidden cache into RAM (Flickr 46 GB, CC 78 GB) once per job; read-only.
- References: reuse the CR5 run only after a same-job 250-step replay is bitwise equal and binding/endpoint identity holds, else retrain
  the reference in the same group job (both decisions recorded in `groups/<group>/refs/<id>/REUSE_RECEIPT.json`).
- Plan §6 FIT truth stratification of graphs (Flickr true/false edges) is not produced by the CPU summarizer (it would need the isolated
  FIT clean mapping); the label-free source-vs-source diagnostics are produced.
