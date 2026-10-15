# Review recommendations received 2026-10-15 (forwarded by the owner) — status of each item

Source: the owner forwarded an external review of V8 (`1815a60`) and V9 (`ac6373d`) with per-item recommendations.  This record lists what the server
applied at once (non-irreversible), what is being prepared, and what still needs the owner's explicit word (label reads, budgets).  Nothing below reads a
sealed label or changes a frozen result.

| item | recommendation | status on the server |
|---|---|---|
| GenAI-Bench 850 non-Rank prompts | approve as V8's one confirmation evaluation; V7 does not use it | **awaiting the owner's explicit go for the read**; confirmation protocol (sample list, metrics for the 6-generator structure, families, Holm scopes, readings fixed in advance) being drafted (`V8_05_CONFIRMATION_PROTOCOL_DRAFT.yaml`, v8 branch); dry-run only until the go |
| TIFA160 | approve as a supplementary secondary external evaluation of the frozen V8 models; labelled "added 2026-10-15" | **awaiting the owner's explicit go for the read**; V8 predictions on the 160 images and the external pipelines are being prepared (no API routes) |
| RichHF | do not download / open | applied: not downloaded |
| V8 lr-grid edge / U-RANK | disclose; no grid extension, no RANK repair search | applied: Corrections section in V8-04 (`9059c71`); wording limited to effectively trained arms |
| V8 "level with FGA" | report the SDXL_Base +0.039 (Holm 0.046) as a detected small improvement | applied: V8-04 Corrections |
| V9-N1 | close H1; correct "all fixed scores coincide" to the affine-invariance statement; "not_verified" is a design choice, not shift detection | applied: Corrections section in V9-N1 report (`961caf7`); no further moment-estimator grids |
| V9-N2 scope | continue; first block Flickr30K only (48 formal + ≤ 48 tuning); COCO a separate decision; budget by measured per-method cost (SGR + NCR dual net + L2RM matching + teacher) | applied to the plan: fork A measuring; the N2 protocol will carry the Flickr-only block and a **pre-registered pre-check table** (teacher score distributions, per-method weight distributions / ESS / pairwise differences, per-sample loss coefficients with anchors and the fixed denominator, anchor roles vs evaluation isolation) before any formal unit; **GPU-hour total awaits the owner after timing** |
| NCR+A / L2RM+A | minimal patches on the official code (ids, noise map, seeds, anchor flag + trusted supervision, paths, export); keep co-teaching / rematching intact; both native-reproduction and anchor-assisted tables; degradation check (anchors off → native path) | applied to the plan (fork D's patches exist; the degradation check is added as a required test before N2's freeze) |
| NCR feature usage basis | data route acceptable for non-commercial research reproduction; do not write "COCO images CC-BY"; code licence ≠ data licence; no redistribution of images / captions / full features; register original terms, feature provenance and code licence separately | being applied in V9_01 / DATA_AUDIT (fork A instructed) |
| Flickr30K official form | check whether an attestation is still required; owner completes it if so, otherwise archive the acquisition record | fork A reports the current requirement; owner action if needed |
| EPFL RGB-NIR | no V9-G2 until the usage basis is settled; one bounded verification task (README / terms, contact the provider if needed) | applied: G2 not prepared; verification task = read the download package terms + draft the enquiry (owner sends); existing EPFL results kept |
| D1 direct hard top-k seeds | do not run; finish REV10, then freeze D1 | applied: no new D1 unit; REV10 evals running → score → closeout report |
| BF-SHG closed test for G1 | agree in principle; trigger = all G1 methods/seeds done, CAL selection done, test samples/perturbations/metrics/families/prediction files fixed; no "only if VCS wins on DEV" condition | applied: decision note in the frozen G1 protocol; **the read itself needs a separate explicit go** |
| NUISANCE / COMPRESS / AMORTIZE, r7 D2 / D3 | archive candidates, off the active queue | applied: removed from the active list; restart needs a concrete scientific reason, not free GPUs |

Execution priority adopted: V8 confirmation (after the go) → finish G1 / REV10 → Flickr-N2 budget filled and one complete block → everything else not
expanded.  Research expectation corrected as the review states: the identities hold, but neither the proportion step (N1) nor P157's posterior result
guarantees an application advantage; N2 can only be judged on complete training results.

## Owner approvals (2026-10-15, verbatim)
1. 「批准读取 GenAI-Bench 850 条非 Rank 提示的人工标签，作为 V8 的一次确认评价（V7 不使用）。」
2. 「批准读取 TIFA160 标签，作为冻结 V8 模型的次级外部评价。」
Scope as approved: one read each; frozen V8 models and selection rule unchanged (all seeds); V7 does not use either; the reads happen only after the
confirmation protocol (`V8_05`) is frozen and its dry-run passes; results reported in full whatever their sign; no tuning feedback from them.
