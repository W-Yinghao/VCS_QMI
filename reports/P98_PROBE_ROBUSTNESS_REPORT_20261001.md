# P98 — probe-robustness report for the P91 CIFAR-100 readings — 2026-10-01

Pre-registration `P98_PROBE_ROBUSTNESS_PREREG_FROZEN_20260930.md`; results-only commit (`P98_probe_robustness.{md,json}`, job 1015879, exit 0).  Three probes on
the same frozen h, identical for all 27 checkpoints (18 P91 CIFAR-100 + the 9 CIFAR-10 8× reference cells): **frozen** (the recipe probe), **l2** (row-normalised
h, recipe probe), **tuned** (300 epochs, lr from {0.1, 0.3, 1.0} on a 10 % FIT hold-out — lr 0.1 chosen everywhere).  Selection split only.  The frozen probe
reproduces every stored value exactly.

## 1. P91 cells under the three probes (mean ± sd, 3 seeds)
| probe | VCS 2× | VCS 8× | SimCLR 2× | SimCLR 8× | VICReg 2× | VICReg 8× |
|---|---|---|---|---|---|---|
| frozen | 57.96 ± 0.21 | 59.91 ± 0.20 | 59.42 ± 0.28 | 58.25 ± 0.35 | 56.43 ± 0.41 | 56.01 ± 0.25 |
| l2 | 47.13 ± 0.06 | 53.06 ± 0.13 | 54.68 ± 0.34 | 56.37 ± 0.79 | 49.41 ± 0.29 | 49.65 ± 0.72 |
| tuned | 55.95 ± 0.44 | 58.06 ± 0.56 | 57.57 ± 0.15 | 57.88 ± 0.59 | 55.90 ± 0.12 | 55.12 ± 0.18 |

## 2. Pre-stated reading ("robust" = every pairwise difference ≥ 1.0 keeps its sign under all three probes)
| quantity | frozen | l2 | tuned | robust |
|---|---|---|---|---|
| SimCLR − VCS, 2× | +1.46 | +7.55 | +1.61 | yes |
| VCS − VICReg, 2× | +1.53 | −2.28 | +0.05 | **no** |
| SimCLR − VCS, 8× | −1.65 | +3.31 | −0.18 | **no** |
| VCS − VICReg, 8× | +3.89 | +3.41 | +2.94 | yes |
| SimCLR − VICReg (2× / 8×) | +2.99 / +2.24 | +5.27 / +6.72 | +1.67 / +2.76 | yes |
| gain_VCS − gain_SimCLR (2× → 8×) | +3.11 | +4.24 | +1.79 | yes |
| gain_VCS − gain_VICReg | +2.36 | +5.69 | +2.89 | yes |

**Verdict: not all P92 readings are robust.**
- **Robust:** VCS's 2× → 8× gain exceeds both controls' gains under every probe (scaling replicates); VCS > VICReg at 8×; SimCLR > VCS at 2×.
- **Not robust:** "VCS > SimCLR at CIFAR-100 8×" (frozen −1.65 for VCS ahead, tuned −0.18 ≈ tie, l2 +3.31 SimCLR ahead) and "VCS > VICReg at 2×".

## 3. Correction to P92 §3
P92 attributed SimCLR's lower 8× linear value to probe under-fitting (train CE 1.12).  The tuned probe lowers SimCLR 8×'s train CE to 0.95 but does **not**
raise its selection accuracy (58.25 → 57.88); the longer probe slightly over-fits every method instead (all tuned values ≤ frozen).  So the drop is not simply a
probe-optimisation artefact.  What does change the picture is the h scale: with row-normalised h, SimCLR improves from 2× to 8× (54.68 → 56.37) and leads VCS
at 8× by 3.3, while VCS's un-normalised h carries class information in its norm that l2 removes (VCS 59.91 → 53.06).  The 8× VCS-vs-SimCLR comparison on
CIFAR-100 is probe-dependent and must be reported as such; the scaling statement is robust.

## Delivery
```yaml
experiment_family: evaluation_only (probe robustness)
protocol_id: P98_probe_robustness
source_commit: e466fd5 (frozen); results this commit's parent
evaluation_readout: selection-split linear top-1 and probe train CE, three probes, 27 checkpoints
status: complete
```
