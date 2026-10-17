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

## Owner decision (2026-10-15, later, verbatim)
「所有的数据，只要是现在有了的，我们就有权限。你不需要再纠结，直接开始相关实验」 — every dataset already on the server is cleared for use; no further
usage-basis deliberation; start the related experiments.  Consequences applied: EPFL RGB-NIR cleared → V9-G2 may be prepared and run; NCR
precomputed features / Flickr30K / COCO cleared for the N2 block (no attestation step); TIFA160 images (being fetched) cleared for the approved
secondary read; BF-SHG closed test stays unread until the G1 confirmation trigger.  The data register keeps recording sources and terms as facts,
without blocking work.

## Owner pause and approvals (2026-10-15 evening, verbatim)
- ~20:40 UTC: 「先暂停一下费token的agent吧，先维持监控」 — all token-using agents stopped; SLURM jobs, the N2 login-node release script and the
  job monitor keep running.
- ~20:55 UTC, in reply to the coordinator's list of GPU work that needs the owner's approval (V8.1 C1: 9 units, about 6–8 GPU-h estimated from
  the V8 unit timings; G1 BF-SHG sealed test: one confirmation read, test-side files fixed and committed before the read): 「这俩都可以提交」.
  Consequences: (1) V8.1 C1 is approved — code, smoke on the real stack and the measured budget still come first, then a frozen C1 protocol and
  the 9 units (package docs/06 C1: U-CS / U-RPS / U-MSE × seeds 0–2, last 2 Q-former blocks + ln_vision + query tokens + head trainable, each
  arm's originally selected lr, CAL_SELECT MSE selection; DEV readout; no V8-05 / GenAI / TIFA160 label).  (2) The G1 closed-test trigger of
  decision note 1 (`reports/V9/V9_G1_PROTOCOL_FROZEN_20261015.yaml`, v9 branch) has its explicit go: one read, after the test samples,
  perturbations, metrics, comparison families and prediction files are fixed and committed.  Everything else stays paused.

## 2026-10-17 (verbatim; recorded by the main session)
- Between the main session's queue check at 05:03:54 UTC and the G2 aggregate start at 05:07:47 UTC (job 1043798), in reply to the question
  whether the paused work should resume: 「解除暂停」.  Consequences: the pause of 2026-10-15 ~20:40 UTC on token-using agents is lifted.  Resumed:
  V9-G2 (aggregate 1043798 on cpu-high, results 9fb315f, report 8803bfa + display fix b70db7b, v9 branch), V9-N2 wave 2 (smoke fixed in ccd7548,
  rerun pending) and fork A's NCR evaluation (OOM fixed by --mem 96G, rerun 1043799), and the D1 closeout report (this branch, 478fbe7 + 9a9560a).
- Earlier the same morning the owner accepted the V10 phase-1 report and decided the phase-2 protocol; the decision text and the owner's JSON
  contract are stored verbatim on branch v10/mf-vcs under v10/decisions/ (249baac, 9cb1ae3).
- Main-session decisions recorded for the owner's review: V9-N2 SGR recipe kept unchanged for the current block although the warm-up's original
  rationale was withdrawn (HOST_LOCK correction_20261017, v9 15a5eec); after 4/24 SGR-student collapses at rho 0.4, wave 1 is paused (14 pending
  jobs cancelled, resubmittable unchanged) and a pre-registered A/B diagnostic of the loss-reduction scale runs first (~2.8 GPU-h); any matched
  rerun of the SGR-student units (~230-410 GPU-h) waits for the owner's decision on the diagnostic result.
