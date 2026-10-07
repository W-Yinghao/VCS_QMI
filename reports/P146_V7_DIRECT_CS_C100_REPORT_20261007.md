# P146 — v7 §7.1 direct-CS route-transfer controls on CIFAR-100 (S-Kernel, CS-K-native), seed 0 — report — 2026-10-07

Pre-registration `P146_V7_DIRECT_CS_C100_PREREG_FROZEN_*.md`; results-only commit `d4d7ac5` (`P146_results.json`).  P87 / P88 designs on the CIFAR-100
recipe base; bandwidth multiples transferred from the CIFAR-10 dev grid (not re-screened; disclosed); FIT-side median calibration at step 0.  The
S-Kernel run was interrupted at epoch 3 by the 2026-10-07 home-quota stop and resumed from last.pt.

| route | linear / kNN | h effective rank @800 | comparator: recipe VCS (P91, seed 0) | Δ linear / kNN |
|---|---|---|---|---|
| S-Kernel (RFF critic, same J, m 4 096, bw 0.5) | **4.28 / 5.42** | 1.4 (collapse flag from epoch 200 on) | 60.00 / 55.42 | −55.7 / −50.0 |
| CS-K-native (classical kernel CS-QMI, bw 0.5) | 41.02 / 31.50 | 7.2 (no flag; dimensional collapse) | 60.00 / 55.42 | −19.0 / −23.9 |
| A-P3 (current complete method, listed separately) | 60.20 / 55.92 | 155.5 | | |

## Reading (frozen)
- **S-Kernel: numerical collapse** by the pre-stated rule (collapse flag at consecutive kNN epochs 200, 400, 600, 800; h rank → 1.0, top
  eigenvalue fraction 0.99–1.0).  The training J stayed at 0.46–0.63 while the held-out J fell to −0.55 / −0.68 / −0.11 (epochs 400 / 600 / 800):
  the 4 096-feature RFF critic separated the training pairs of a collapsed representation without generalising.  Recorded as the result; not re-run.
- **CS-K-native** trained without numerical failure (no underflow, final D_CS 1.55) but much weaker than the neural recipe (−19.0 linear), with the
  same dimensional collapse of h seen on CIFAR-10 (P88: rank 10–39).
- Both kernel routes are therefore far behind the neural cosine critic on CIFAR-100, more so than on CIFAR-10 (P88: −2.1 / −8.5 linear).  Because
  the bandwidth multiples were transferred rather than re-selected on CIFAR-100, this is a statement about these transferred settings, not about
  kernel CS estimators in general (plan §7.1).
