# Second application — the four property pre-checks: summary and hand-back (2026-09-27)

Brief: `CS_QMI/VCS_QMI_Second_Application_Server_Brief_v1.md`.  Plan and status: `SECOND_APP_PRECHECKS_PLAN_20260927.md`.  Per the brief §4
this document selects **no task**: it lists the surviving families with the engineering each would need and its data risk, for discussion.
All four pre-checks were pre-registered before compute (frozen files + dated addenda), use identity splits with recorded hashes, touch no
official test set, give competitors the same tuning grid (budgets tabled in each report), keep failed configurations, and stay inside the
movable set: no hard-negative selection, no score-weighted pairs, no auxiliary losses, tanh kept on every critic, reference measure unchanged.
Nothing from the "discuss first" list was touched.  Evidence category for all four: **completed** (single split each; seeds as stated).

## 1. Verdicts

| pre-check | property | verdict | decisive evidence | report |
|---|---|---|---|---|
| A calibration & threshold transfer | pairwise T has absolute meaning | **holds conditionally** | Native (1+T)/2 is as calibrated as post-hoc Platt only in the mid-dependence regime (setting 2, J 0.57 / 0.16: ECE 0.023 / 0.088 vs 0.022 / 0.089 source / target); at the degenerate edge (setting 1, J 0.92) it is worse (0.046 / 0.089 vs 0.004 / 0.036).  Threshold transfer: no advantage, structurally — T = tanh(a·cos + b) is monotone in the cosine, so T ≥ 0 is a cosine threshold. | `P50_PRECHECK_A_REPORT_20260927.md` |
| B1 closed-form critic | quadratic objective ⇒ closed-form critic in a fixed feature class | **holds** | Closed-form linear-class critic (+ tanh scalar) within 0.012 of the trained neural critic on every VCS checkpoint; ≥ the neural critic on the cosine family. | `P48_PRECHECK_B1_REPORT_20260927.md` |
| B2 registration energy | closed-form J* as an alignment energy | **does not hold** | On 60 constructed modality pairs the J* surface has the truth as global maximum and never diverges, but is rougher than histogram MI / NMI: local maxima 10.2 vs 6.5, basin 20 px / 15° vs 24 / 21, Nelder–Mead success from R = 20 / 30: 0.77 / 0.54 vs 0.88 / 0.70. | `P52_PRECHECK_B2_REPORT_20260927.md` |
| C batch decoupling | positives and negatives averaged separately | **does not hold** (frozen-tower adapters) | VCS, InfoNCE and pairwise-logistic adapters are all flat within 1 % from B = 512 to 16 on both axes (fixed updates / fixed exposure); the only degradation is logistic −4 % at B = 16 fixed updates.  No batch effect to be immune to. | `P54_PRECHECK_C_REPORT_20260927.md` |
| D leakage-detection power | bounded objective, distribution-free bound, exact conditional test | **does not hold — by the frozen false-alarm clause** (framing flagged for the owner, §3) | Every power / exactness / monotonicity clause held: permutation-calibrated VCS test on par with HSIC (within 0.03 in 32 / 36 small-sample cells, never ≥ 0.10 below), far above C2ST; within-class shuffle exact; Ĵ monotone where non-zero; guaranteed Hoeffding test conservative (0 / 1400 nulls) but 20–50× more sample-hungry.  Tripped: at R = 100 the permutation test's null rejection rate is 0.10–0.12 in 3 of 16 large-n cells (pooled 0.067 / 0.057 per encoder vs 0.05; HSIC trips on the same draws in one cell) — above the frozen 0.09 figure. | `P46_PRECHECK_D_REPORT_20260927.md` |

## 2. Families (brief appendix A) after the pre-checks

| row | opened / conditional | closed |
|---|---|---|
| A | mismatch detection, open-set rejection & verification, pointwise PMI maps — *conditional*: only where the pair problem is genuinely uncertain (S ≪ 1) and "no calibration data" is the value proposition | data re-weighting by T (no advantage over a calibrated cosine; threshold transfer gives nothing a cosine threshold does not) |
| B | unlabelled embedding probes / dependence measurement between frozen representations (B1: closed form = neural critic, cheaper and deterministic) | registration, sensor extrinsic calibration, cross-spectral stereo cost (B2: not a competitive energy) |
| C | — | micro-batch & large-voxel, memory bank & streaming, k-way, asymmetric sample sizes (no batch effect on frozen-tower adapters for any method; a from-scratch encoder would be the only remaining place to look, and that is SSL compute under the stop order) |
| D | — by the letter.  If the owner reads the false-alarm clause by its intent (it was written to catch an invalid test, not a level of 0.06): leakage & shortcut detection, dependence profiles, domain-shift magnitude as a permutation-calibrated O(n) test with an exact conditional version for discrete Y and a monotone magnitude; the certified bound only for strong dependence or n ~ 10⁴ | leakage & shortcut detection, dependence profiles, domain-shift magnitude, invariance penalties (all of row D, by the letter of the frozen clause) |

## 3. Surviving families — engineering and data risk (hand-back, no pick)

1. **Leakage / shortcut detection and dependence profiles (D — closed by the letter of the frozen clause; listed because the owner's reading of that clause decides it, see §1 and the D report §3.1).**
   *What survives:* one statistic that is a permutation test, a magnitude and an exact conditional test, O(n), equal to HSIC in power at
   small n.  *Engineering:* small–medium — the test code exists (`scripts/precheck_d_tests.py`: three critics with VAL selection,
   permutation nulls, within-class shuffling); missing are a nuisance-planting protocol beyond global colour (texture, subgroup, continuous
   and multi-class N via the k-way / coordinate-wise shuffle the brief §5 allows), a dependence-vs-layer profile, and reporting.
   *Data risk:* low for planted nuisances (any labelled set); medium for realistic shortcuts (spurious-attribute benchmarks need licence
   checks; none downloaded).  *Honest positioning:* the claim is "same power as HSIC, linear cost, plus bound and conditional version", not
   "more powerful".  *Open:* a second encoder seed; if the family is kept, the permutation test's level at n ≥ 5000 should be pinned down with R ≥ 1000 null draws (≈ 1 GPU-hour per encoder) before any claim of exactness is written.
2. **Closed-form critic as a measurement on frozen features (B1, holds).**
   *What survives:* dependence between two frozen embeddings (or an embedding and a discrete/continuous variable) computed in closed form
   in a fixed feature class, matching the trained critic.  *Engineering:* small — a probe library (feature class, ridge and tanh scalar
   chosen on a validation part, as in B1 and inside D).  *Data risk:* low.  *Caveat:* no gain over the neural critic except cost and
   determinism; B2 shows it is not a good *optimisation* energy, so this family is measurement only.
3. **Mismatch detection / open-set rejection / pointwise PMI maps without a calibration set (A, conditional).**
   *What survives:* in the mid-dependence regime the native (1+T)/2 is as calibrated as a Platt fit.  *Engineering:* medium — a real
   mismatch or open-set protocol with a downstream consumer of calibrated scores, and a fair comparison against Platt-on-cosine, which
   needs only a few hundred labelled pairs (so the "no calibration set" advantage is small in practice).  *Data risk:* medium — needs a
   genuine paired distribution shift; the COCO topic shift is a proxy, an external target (Flickr8k/30k) needs a licence check.
   *Caveat:* nothing to claim on threshold transfer.

Closed: registration / calibration / stereo energies (B2), the whole batch-decoupling row (C, on frozen-tower evidence), and — by the letter — row D.

## 4. Reading of the whole (brief §4, last paragraph)
No family opened unconditionally except closed-form *measurement* (B1).  The conditional survivor A and the letter-closed, intent-conditional D are both uses of VCS as an
**estimator / test** on frozen representations, not as a training loss with special batch behaviour (C) or as an energy (B2).  Read
with the brief's own rule, this says the tool's practical second-application value is diagnostic (measure, test, calibrate-without-data in
a regime), and that the paper's weight stays on the SSL application plus controlled estimator validation.  Whether one of the three
survivors is worth a task-level pre-check — and whether D's false-alarm clause is read by its letter or its intent — is the discussion this document hands back.

## 5. Provenance
Preregs: `P45_PRECHECK_D_PREREG_FROZEN_20260927.md` (+4 addenda), `P47_PRECHECK_B1_PREREG_FROZEN_20260927.md`, `P49_PRECHECK_A_PREREG_FROZEN_20260927.md`
(+5 addenda), `P51_PRECHECK_B2_PREREG_FROZEN_20260927.md`, `P53_PRECHECK_C_PREREG_FROZEN_20260927.md`.  Third-party assets: open CLIP ViT-B/32
laion2b_s34b_b79k (owner-approved download; provenance file with sha256 under `models/open_clip/`); COCO 2017 local copy (train2017 for
A/C, val2017 for B2; identity = image id; split hashes in `outputs/P49_precheck_A/splits.json`); CIFAR-10 frozen selection split (D, B1).
Job ids: `reports/job_ids.json`.  Result-only commits precede every interpretation commit.
