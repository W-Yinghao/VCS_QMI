# Evidence ledger — author-side record, excluded from the anonymous manuscript

Snapshot: W-Yinghao/VCS_QMI@bc28bf89c94d5331de811a83aa8bcbfb77d7d155.
Commit time: 2026-10-06 05:58:03 UTC.
Existing manuscript base: VCS_QMI_Manuscript_v5.pdf and its mounted LaTeX source.

## Main-text mappings

| Location | Source at snapshot | Treatment |
|---|---|---|
| Abstract: mathematical construction | Collaborator Research Plan, sections 2–8 | Retained definitions; no Shannon-MI equivalence |
| Theorem 1 | Original plan, exact CS construction, optimum, gap | Combined statement; proof completed explicitly |
| Proposition 2 | Derived from original gap and independent-pool variance | New presentation/derivation; independently checked; not attributed verbatim to original plan |
| Proposition 3 | Original gap + conditional expectation; 2026-09-28 supplement §2.6 | New complete proof; general observations, not SSL-specific theorem |
| Figures 1–2 | reports/P86_ESTIMATOR_BENCHMARK_REPORT_20260929.md | Rounded reported means; no invented error bars; same-target posterior error |
| Table 1 (compression) | reports/P118_V5_NEST_REPORT_20261003.md | Same moderate condition, same sampled evaluation; VCS and logistic rows paired with nested implementation |
| Main Table 2, CIFAR-10 VCS/SimCLR | reports/P107_AP3_LAYER2_REPORT_20261003.md | Three-seed entries; five-seed confirmation separately described |
| Main Table 2, CIFAR-10 selected logistic | reports/P129_ADDENDUM1_REPORT_20261005.md | Selected (2,.25); mean/SD recomputed from three reported linear values |
| Main Table 2, CIFAR-100 VCS/SimCLR | reports/P107_AP3_LAYER2_REPORT_20261003.md | Three-seed, standard augmentation, original frozen-h probe |
| Main Table 2, CIFAR-100 selected logistic | reports/P129_ADDENDUM3_REPORT_20261006.md | Newly completed result included; higher selected-logistic accuracy not omitted |
| Main Table 3, external SimCLR audit | reports/P116_results.md; sections full_simclr_blur and full_simclr_colour | Fixed external-model fixture, exact shared split; A1 vs short A3@50; no unsupported 30x claim |
| Paired intervention exemplar | reports/P109_V4_X_I_REPORT_20261002.md; reports/P118_nest_results.md | Separate paired-image fixture; no pooling with P116 as if same experiment |
| Layerwise observations | reports/P141_V7_C100_LAYER_READOUT_REPORT_20261006.md | Task readability only; contradictory 'all methods lose fine information' not repeated |
| Strong augmentation | reports/P107_AP3_LAYER2_REPORT_20261003.md; P115 factor report | Completed negative boundary retained |
| Supplement native estimators | P86 same path, corrected-SMILE addendum dated 2026-09-30 | Uses corrected implementation, retains good in-batch baseline behavior |

## Naming

- 'VCS (ours)' in the current SSL main table = the fixed-scale all-view full-gradient configuration internally called A-P3.
- Old VCS recipe and G2 are internal configurations. They are not separate proposed methods.
- Balanced logistic is an objective/measurement control, not a second newly proposed method.
- VCS measurement critic is fitted on a frozen distribution; it need not equal the fixed SSL scorer.

## Result boundaries

All current main SSL results are development-validation results on 45k/5k splits.
The source histories include earlier official-test use on an older configuration; supplementary protocol discloses that and does not merge those scores.
No ImageNet training score, spatial heatmap, new concept benchmark, or unexecuted method comparison appears as an observed result.
No deployment selection of one encoder is based on its interpretability score in this manuscript.

## Scope changes from the interpretability redesign

The three-state oracle design is not reported as a completed visual benchmark. The available known-truth nested study is used for estimator validation.
The concept audit is retained as the direct secondary visual application.
Region-mask saliency, response-noise scaling, RISE/RELAX benchmarking, and VOC expansion are deferred to a distinct possible project; no performance is invented.

## Corrections and safeguards

- P141 probe scores do not measure mutual information. SimCLR h->z fine accuracy is 56.6->56.7 in its pooled standardized readout, so 'every method loses information' is unsupported.
- Current selected JS beats the rule-selected VCS on CIFAR-100 linear by about .49 points; shared-default five-seed VCS superiority is a separate comparison.
- P116 ridge+tanh is not an exact bounded nonlinear solution. Short JS fitting is about .05-.07 seconds in the displayed cells, not the intentionally longer 800-closure budget.
- P109 non-rejection is described as failure to detect at the given sample and critic budget, not removal of information.
- Separate audit fixtures are expressly identified rather than reconciling their different rejection rates by assumption.
- The numerical checks in verification_results.json are finite algebra/table arithmetic only, not new neural evidence.

## Public reference handling

The source references.bib preserves canonical URLs of cited papers. references_print.bib has the same records without long URL fields, for readable typesetting.
RPC, relative least squares, generic f-divergences, and proper composite losses are explicitly attributed. No 'first quadratic loss' claim is made.
