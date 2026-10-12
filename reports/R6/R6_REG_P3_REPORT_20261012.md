# r6 REG-P3 — the REG energies on a second modality pair (EPFL RGB–NIR, rigid ±30° / 51 px) — report — 2026-10-12

Protocol `reports/R6/R6_REG_P3_PROTOCOL_FROZEN_20261012.yaml` (frozen before any REG-P3 DEV registration; decision-timing disclosure inside).
Results-only commit `2baf6c5`: aggregate (job 1035613) and the descriptive secondary readouts (job 1035618).  Run as low-priority filler
(owner decision 2026-10-12).
- **Data:** EPFL RGB–NIR Scene (nirscene1).  RGB is the reference, NIR moves.  33 of 477 pairs were excluded by the pre-registered
  base-alignment check (peak ≥ 2 px from zero shift; listed).  DEV: 72 units, 74 windows of 512 px; 10 rigid perturbations each (θ* uniform on
  ±30°, |t*| ≤ 51 px), so 740 registrations per run.
- **Models:** REG-P1's recipes retrained on EPFL FIT without retuning (VCS, logistic and InfoNCE critics, seeds 0–2; CoMIR seeds 0–2).
  Critic checkpoints were selected on REG-P3's CAL by the pre-registered lexicographic rule; MI, NMI and CoMIR need no selection.
- **Optimiser and metric:** REG-P2's shared rigid optimiser for every energy (3° × 4 px start grid, best 3 starts refined by Adam).  Success is
  CoMIR's mean corner error below 10.5 native px (primary) or 2.25 px (secondary): REG-P2's 42 / 9 px at the same relative scale.
  Unit-macro averages.

## 1. Results (DEV unit-macro success, %; mean ± sd over seeds)
| method | < 10.5 px | < 2.25 px | median corner error (px) | p90 (px) | median rotation error | failures per run (energy) | CAL < 10.5 px |
|---|---|---|---|---|---|---|---|
| **VCS critic** | **99.3 ± 0.2** | **87.6 ± 1.7** | 0.79 | 2.4 | 0.14° | 3 / 6 / 6 (all energy) | 100.0 |
| logistic critic | 99.2 ± 0.3 | 87.3 ± 1.4 | 0.67 | 2.4 | 0.11° | 8 / 3 / 6 (16 of 17 energy) | 100.0 |
| InfoNCE critic | 99.4 ± 0.7 | 90.0 ± 1.1 | 0.58 | 2.3 | 0.09° | 0 / 3 / 10 (8 of 13 energy) | 100.0 |
| CoMIR (encode-once) | 96.2 ± 1.2 | 85.3 ± 1.0 | 0.54 | 2.6 | 0.07° | 20 / 27 / 39 (78 of 86 energy) | 100.0 |
| MI | 93.8 | 82.9 | 0.47 | 3.0 | 0.06° | 45 (22 energy) | 70.0 |
| NMI | 94.0 | 82.9 | 0.47 | 3.0 | 0.06° | 44 (21 energy) | 72.2 |
| RFF kernel critic | 87.8 | 76.0 | 0.65 | 41.5 | 0.09° | 93 (82 energy) | 88.9 |

Contrasts VCS − X (percentage points; 95 % t over the 3 seeds; unit bootstrap, 2 000 resamples):
| X | < 10.5 px (primary metric) | < 2.25 px (secondary; truth caveat) |
|---|---|---|
| **MI (primary)** | **+5.5 [+4.9, +6.1]; unit [+1.5, +10.8]** | +4.7 [+0.4, +9.0]; unit [−1.2, +10.8] |
| **CoMIR (primary)** | **+3.1 [+0.7, +5.5]; unit [+0.5, +6.3]** | +2.4 [−4.2, +8.9]; unit [−1.6, +6.4] |
| NMI | +5.3 [+4.7, +5.9]; unit [+1.2, +10.6] | +4.7 [+0.4, +9.0]; unit [−1.0, +10.7] |
| kernel | +11.5 [+10.9, +12.1]; unit [+5.3, +18.8] | +11.7 [+7.3, +16.0]; unit [+5.8, +18.4] |
| logistic | +0.1 [−1.3, +1.5]; unit [+0.0, +0.3] | +0.3 [−2.3, +3.0]; unit [−1.6, +2.4] |
| InfoNCE | −0.1 [−1.5, +1.3]; unit [−1.6, +1.0] | −2.4 [−9.3, +4.5]; unit [−6.4, +1.3] |

Reading rule (pre-registered, as in REG-P1 / REG-P2): a difference counts as detected only when both intervals exclude 0.  For the
deterministic rows (MI, NMI, kernel) the seed interval reflects only the critic's seed variation.

## 2. Pre-fixed readings (frozen wording)
1. **Ceiling readout not triggered.**  CAL success < 10.5 px is 70.0 % for MI, 72.2 % for NMI and 88.9 % for the kernel, so < 10.5 px stays
   the primary metric.
2. **VCS − MI (primary): "VCS registers 5.5 percentage points more DEV registrations (< 10.5 px)"** (seed [+4.9, +6.1], unit [+1.5, +10.8]).
3. **VCS − CoMIR (primary): "VCS registers 3.1 percentage points more DEV registrations (< 10.5 px)"** (seed [+0.7, +5.5], unit
   [+0.5, +6.3]).  On BF/SHG (REG-P2) this contrast's seed interval contained 0.
4. **"VCS ≈ logistic ≈ InfoNCE → a learned-critic class effect; no VCS-specific claim":** both contrasts contain 0 at both thresholds.
5. **"REG-P1's recipes retrained on EPFL FIT without retuning; REG-P3 measures transfer of the method family to a second modality pair, not a
   retuned method."**

At < 2.25 px (secondary) only VCS − kernel is detected.  VCS − MI and VCS − CoMIR have unit intervals that contain 0.

## 3. Secondary observations (descriptive)
- **Smaller margins than on BF/SHG.**  Every energy is far more reliable on EPFL than on BF/SHG: MI reaches 93.8 % here against 74.1 % in
  REG-P2.  The learned critics' lead over MI is correspondingly smaller (+5.5 vs +20.1 points), and MI's failures are not catastrophic here
  (p90 3.0 px, against 269 px on BF/SHG).
- **Failure attribution: half of MI's failures are search failures.**  An energy failure means the energy at the estimate is ≥ the energy at the
  truth; otherwise the shared search missed the truth's basin.
  - MI: 22 energy + 23 search failures; NMI: 21 + 23.
  - VCS: all 15 failures (3 seeds) are energy failures.  The learned critics and CoMIR have at most a few search failures each.
  - So part of the MI gap belongs to the energy under this optimiser, not to the energy alone.  Counting energy failures only, MI still has 22
    per run against about 5 for VCS.  That count is a description, not a pre-registered contrast.
- **Energy profiles (the first DEV window of 5 categories; θ in 1° steps up to ±10°, clipped at the ±30° search range; x ±40 px in 2-px steps).**
  - Every profile except the kernel's peaks within one step of the truth.  All off-truth peaks fall on one window (field_0001), where every
    energy peaks 2 px from the recorded truth, which is consistent with the 2-px resolution of the truth.  The kernel peaks 24 px off there.
  - MI and NMI peaks are narrow (median half-maximum width 2° and 4 px).  VCS's are broader (6° and 8 px), logistic's in between.  InfoNCE's
    are as narrow as MI's.
  - This is consistent with MI's lower median error when it succeeds (0.47 vs 0.79 px) and with its search failures: a 3° × 4 px start grid
    can miss a narrow basin.  It is a description over five windows, not a tested mechanism.
- **Strata.**
  - Success barely depends on |θ*| within ±30°: VCS 99.7 / 99.3 / 99.0 % and MI 94.2 / 93.0 / 94.5 % over 0–10 / 10–20 / 20–30°.
    CoMIR falls from 97.1 to 95.0 %.
  - Success does not depend on the translation stratum.
- **Window-macro success (REG-P2's summary)** gives the same ordering, within 0.4 points of the unit-macro values (`R6_REG_P3_secondary.md`).
- **Selection.**  Every critic reached 100 % CAL success < 10.5 px, so the lexicographic rule chose on the < 2.25 px tie-break.  Chosen
  checkpoints: VCS 500 / 2 000 / 2 750; logistic 250 / 250 / 2 500; InfoNCE 2 500 / 250 / 500 steps.
- **Cost** (s per registration): VCS 0.98, logistic 1.23, InfoNCE 1.01, CoMIR 0.19, MI 0.51, NMI 0.50, kernel 2.17.  Jobs ran on L40S
  and RTX PRO 6000, so these are indicative only.

## 4. Caveats
- One dataset, 72 DEV units, one shared optimiser.  About half of MI's failures are search failures, so a different search (finer start grid,
  multi-resolution, more starts) could narrow the MI gap.  The comparison is of registration energies under this optimiser.
- CoMIR is evaluated as "CoMIR features + our optimiser" (encode-once), not as its full native pipeline.  The warp-then-encode row was not run
  in REG-P3 (coordinator rule (e)).
- The alignment check resolves 2 native px, so the < 2.25 px readout carries truth noise (protocol `truth_caveat`).  The 33 excluded pairs are
  listed with the rule.
- **Licence:** EPFL RGB–NIR has no explicit licence text.  This run is internal research under P60's acceptance.  Publication needs the usage
  basis first (owner decision D5; the licence register is being compiled).
- Decision timing: the coordinator knew the REG-P1 / REG-P2 results and P60's EPFL MI / NMI numbers when freezing (protocol disclosure).

## 5. What this adds to the REG application (owner decision 2026-10-12: rewritten attribution)
- REG-P3 is the second modality pair the r6 synthesis said a stronger claim would need.  The candidate statement can now read:
  "In the tested BF–SHG and EPFL RGB–NIR rigid-registration settings, learned local cross-modal relations achieve higher success at the primary
  threshold than the compared per-image MI / NMI registration energies under one shared optimiser; VCS is one effective training criterion
  among them."
- Two qualifiers go with it.  On EPFL the margin is 5.5 points and about half of MI's failures are search failures.  At the stricter 2.25 px
  threshold no difference from MI or CoMIR is detected on EPFL.
- There is still no VCS-specific advantage: VCS ≈ logistic ≈ InfoNCE on both pairs.

## 6. Next
- r6 synthesis: add REG-P3 (addendum).
- D4 cleanup: REG-P3's smoke and check directories can now join `scripts/cleanup_records.sh` (checkpoints / arrays only; records stay).
