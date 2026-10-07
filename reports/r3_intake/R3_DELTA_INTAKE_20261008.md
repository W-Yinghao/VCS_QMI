# r3 delta intake — CVPR-MV6-THEORY-20261007-r3 — 2026-10-08

Package checks: `verify_package.py` PASS (9 tasks), `check_reference_math.py` 28 / 28.  Snapshot a6fb342 = this checkout's HEAD at intake; the r2 intake
(`manuscript_v6_handoff/`) and the P142–P148 unit map are reused, not redone.  Reconciled against raw run files, the job queue and the disk.

## 1. State at intake
| item | value |
|---|---|
| HEAD | a6fb342 (+ this intake); evidence reports unchanged since the V6 snapshot bc28bf8 except the new unit reports listed in `STATUS_20261007.md` |
| queue | 30 / 30 normal-QOS jobs (cap reached): 8 GPU jobs running, P130 add. 1 chains, P138 add. 1 ×6, P145 add. 1 ×2, P139 ×2 (nice 3000) pending; a feeder submits the remaining P130 chain links and the r3 CPU gates as slots free |
| disk | home ≈ 1.57 TB after cleanup v4 + v5 (quota not readable; estimated limit ≈ 1.67–1.69 TB); `outputs/` 157 GB; 50 MB write probe OK |
| caches | **P122 two-view cache gone** (cleanup R6 v4); **P143 feature store gone** (v5); P101 never cached; P85 / P86 / P108 / P134 JSON records intact (nothing deleted) |
| checkpoints | all four r3 fixture encoders have `epoch_800.pt` (P107_AP3 C10 / C100 seed 1, P41_simclr C10 seed 1, P91_c100_simclr seed 1) |
| permissions | standing programme rules; r3 authorises no new pretraining and no new GPU budget; the measurement units below run on CPU |

## 2. Field matrix (R3-E0 first deliverable; `reports/P151_E0/field_matrix.json`)
| field | P85 / P86 (+ SMILE fix) | P108 | consequence |
|---|---|---|---|
| oracle S / MI / J_oracle | yes | yes | precision panels re-aggregated (P151) |
| 200-draw bootstrap of the EVAL value | yes | no (analytic SE) | resolution / eval-variance panels re-aggregated |
| SELECT-risk curve every 100 updates | yes | budgets 250 / 1000 / 4000 | training-stability proxy only |
| selected critic weights | **never written** | never | E1 writes them |
| per-sample T on EVAL | **never written** (quantiles only) | never | E1 writes them (paired bootstraps need them) |
| critic-parameter gradient norms per update | **never written** | never | E1 records them — the §12.2 "gradient-norm variance" field |
| critic-output histograms | saturation fractions / quantiles only | — | E1 records 201-bin f and 101-bin T histograms |
| repeated independent EVAL batches for a fixed fit | one EVAL block bootstrapped | — | E1: 64 independent blocks of 512 |
| cost (fit / eval seconds, memory) | yes | yes | reused |

## 3. Task map
| r3 id | decision | P | what |
|---|---|---|---|
| R3-00 | DONE | — | this document |
| R3-E0 | DONE (re-aggregation) | **P151** | `scripts/p151_e0_aggregate.py` → `reports/P151_E0/` (precision / stability / resolution / N-batch axes / P108 J-vs-plug-in / field matrix) |
| R3-E1 | FROZEN after gate, ≤ 54 units | **P152** | recorded re-fit of the 3-level Gaussian ladder (I 2 / 6 / 10 × 6 in-batch methods × 3 seeds) at the P85-selected lr; adds the four missing fields; CPU |
| R3-V0 | fixture regenerated | **P149** | 4 encoders × 20 k base images × 2 standard-law views (law hash 22f3b58bfde64c92), h + z fp32, independent EVAL pools |
| R3-V1 | FROZEN after gate | **P149** | 4 fixtures × 4 settings × 3 estimators × 3 init seeds = 144 selected cells (≤ 432 candidate fits); CPU |
| R3-C0 | FROZEN after gate | **P150** | one crossed diagnostic (C10 VCS seed 1, colour 0.1, h → O + null; P143 and P144 classes × VCS / JS objectives, one exact-Q common readout, paired-by-image bootstrap) |
| R3-A0 | REUSE | P142 / P143 | kept as detection / model-observation tables (see `manuscript_v6_handoff/phase_report.md`) |
| R3-S0 | CONTINUE | P130 / P138 / P145 add. 1, P147, P148; P139 parked | priorities in §5 |
| R3-W0 | owner | — | claim ledger below is the input; no manuscript edit |

## 4. Theory-to-evidence ledger (what each §12 metric rests on now, from the raw records)
| O1 metric | existing evidence (numbers from `reports/P151_E0/`) | gap → unit |
|---|---|---|
| estimator mean / sd over seeds vs dependence (§12.2, §17.5) | Gaussian ladder, same target S: VCS in-batch posterior MSE .065 → .027 (I 2 → 10), matched JS .061 → .018, S-KDE .44 → .70, RFF .17 → .13, Nyström .40 → .56, rLS-tanh .10 → .08; seed sd ≤ .005 for both neural methods.  Native targets: in-batch InfoNCE 1.76 / 3.54 / 4.99 / 5.97 / 6.53 for MI 2…10 (bound log 1024 = 6.93, not log 256 as P86 §5 states), NWJ in-batch 1.76 … 7.72, DV in-batch 1.77 … 8.33, corrected SMILE in-batch 1.67 … 13.4 (overshoots above I 6); product-sampled NWJ / DV diverge at I ≥ 6 (values −61, −1.5e8, seed sd 109) | none for the curve itself; the honest reading is that **matched JS is at least as precise as VCS on S at every level**, and that MI methods are compared only on their own targets |
| gradient-norm variance (§12.2) | **not recorded anywhere** (only the generator channel derivative) | P152 |
| batch / lr sensitivity | P85 batch axes (B 64 / 256 / 1024 at equal updates and equal exposure) and the 3-lr grid, `axes_N_batch.json` | none |
| critic-output histograms | saturation fractions and residual quantiles only | P152 |
| NaN / failure rate in high dependence | non-finite steps: 0 for every in-batch method on the ladder; product DV 1981 non-finite steps (P86 §8); product NWJ / DV finite but divergent values | P152 confirms at I 10 with per-step records |
| wall-clock per update | fit seconds per method (in-batch neural ≈ 9–10 s / 2000 updates on GPU; kernels 0.1–4 s) | none |
| exact gap / J vs plug-in (§4, §10) | P108: plug-in smaller |bias| in 70 / 72 and smaller RMSE in 69 / 72 cells; at N 16 384 C1 J −0.041 vs S_plug −0.015 (VCS); at N 256 both ≈ −0.55 (fit-limited) | none — this caveat must stay in the paper |
| resolution of adjacent levels | ordering probability 1.00 for every in-batch method on the Gaussian ladder (|Δ| / pooled sd 6–94); xor 8 → 10: VCS in-batch 0.89 (1.0 sd), kernels / MI methods ≥ 2.6 sd | real-image resolution is what P149 tests |
| invertible invariance / DPI (§8) | P108 E2 synthetic rotation; P101 noise on frozen VCS: no resolution gain with kNN kept (negative) | P149 settings 2–4 (real two-view relation) |
| application where dependence measurement is central (§12.4) | P122 two-view (saturated at 0.97–0.99 under the standard law); P142 / P143 detection tables | P149 (two-view relation under a controlled channel) |

Writing items added to the intake list: **W6** P86 §5 / paper text: in-batch InfoNCE EVAL bound is log 1024 (EVAL blocks of 1024), not log 256 (training batch); **W7** any sentence implying VCS has lower finite-sample variance than the MI estimators must cite the same-target posterior-MSE panel and the native-target panel separately (the ladder shows JS ≥ VCS on S and finite in-batch MI methods ordering the ladder perfectly).

## 5. Priority of what is queued or not yet submitted (my judgement)
1. **P145 add. 1, SimCLR C100 seeds 1–2** (2 GPU jobs, 4.4 h each): closes a triggered rule; cheapest. Nice 0 → first in my pending list.
2. **P138 add. 1** (6 GPU jobs, 4.5–9 h): closes the paired non-inferiority claim for K = 16. Nice 300.
3. **r3 CPU units P149 → P150 → P152** (no GPU; ≈ 2 h + 1 h + ≈ 9 h CPU): the new central measurement evidence; fed as cap slots free; gates first.
4. **P130 add. 1 ResNet-50 seeds 1–2** (6 chains, ≈ 170 GPU-h): closes the largest effect we have (A-P3 +6.1 over SimCLR at R50, seed 0); worth its cost, but it should not displace the short closers. Nice 500.
5. **P147 / P148 remaining cells** (running): close the batch and weight-decay axes; no new cells after them.
6. **P139 DDP scaling checks** (parked, nice 3000): only if ImageNet gets a budget.
7. Not submitted and not recommended now: STRESS ε = 0.20 (needs a budget decision), any further single-axis SSL exploration (the recipe is flat, §A1), ImageNet full training (owner budget).
