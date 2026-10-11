# R5-COND-SEM — frozen semantic reference law: design only, not authorized, not recommended now — 2026-10-11

The package (docs/05 §4) allows drafting only if three conditions all hold.  Status:

| condition | status after r5 |
|---|---|
| the detector-score reference gives too little mass to truly dangerous, language-related distractors | partly: it doubles the reference mass on each model's own top wrong box (0.057 → 0.117), yet the wrong-box gradient share barely moves (0.48 → 0.52) |
| reproducible in shared-candidate diagnostics | the shared (zero-shot) order shows the same pattern (reference mass 0.057 → 0.066, gradient share unchanged) |
| the distribution-softmax control does not explain the remaining gap | yes: D ≈ E while VCS / JS trail D by 2.1–3.2 points |

**Design (if ever run):**
- ν_λ(r | w, G) = (1 − λ)/m + λ softmax_r(s_teacher / τ), with a frozen teacher and λ < 1.
- τ and λ set on FIT / CAL only.
- Rank by log ν + 2f.
- VCS and JS both trained with it; set softmax kept.
- A teacher-score fusion control for every arm.
- Related work (NCE, hard-negative sampling, reference-distribution design) stated before any claim.

**Not recommended now.**  A semantic ν moves the same lever as the detector-score ν: it re-weights Q.  r5 shows that re-weighting Q did not
focus the pairwise gradient on the dangerous candidate.  The deficit grows with within-query competition, which a reference law does not
introduce.  Any run needs its own protocol and budget; it is not an implicit extra task of r5.
