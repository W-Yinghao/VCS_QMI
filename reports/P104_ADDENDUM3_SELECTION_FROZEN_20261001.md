# P104 addendum 3 — first-batch selection (spec §10.1) and confirmation launch — FROZEN 2026-10-01T14:16:39Z

All eight seed-0 runs are complete.  Final-h linear-val (kNN):

| rank | cell | linear | kNN |
|---|---|---|---|
| 1 | **G2** fixed (a, b) = (2, −1), K8 cross-view, right detach | **88.66** | 86.96 |
| 2 | **U2** fixed (1, 0), all-view-token pairs, right detach | **87.96** | 86.10 |
| 3 | G3 fixed (1, 0), full negative routing | 87.64 | 85.38 |
| 4 | U1 learned (5, 0), all-view-token pairs | 87.56 | 85.58 |
| 5 | N1 learned (5, 0), noise τ 0.3, R = 4 | 87.30 | 85.32 |
| 6 | G1 fixed (1, 0) | 87.04 | 85.32 |
| 7 | N2 matched JS, noise τ 0.3, R = 1 (control) | 86.84 | 84.84 |
| 8 | G4 learned (5, 0), full negative routing | 81.70 | 78.04 |

Reference: recipe P35 seeds 0/1/2 86.42 / 87.16 / 87.44 (87.01 ± 0.53); tuned SimCLR P41 88.32 ± 0.30.

**Selection (§10.1, by final linear-val; ties < 0.1 → cheaper / simpler):** **G2** and **U2**.  The rank-2/3 gap (U2 − G3 = 0.32) exceeds 0.1;
no tie rule applies.  N1 is not selected, so N2 needs no further seeds (§10.1).  Hyper-parameters are frozen from here; seeds 1–2 change nothing.

**Required controls (§4.1, §10.1): both candidates are fixed-scale cells.**
- G2 → **G2F**: a, b learned from (2, −1), everything else as G2 (seeds 0, 1 submitted under addendum 1; **seed 2 now**).
- U2 → **U2F** (new): a, b learned from (1, 0), all-view-token pairs (chunk 256) and right detach as U2; **seeds 0, 1, 2 now**.

**Confirmation runs submitted now:** U2 seeds 1, 2; U2F seeds 0, 1, 2; G2F seed 2 (G2 seeds 1, 2 and G2F seeds 0, 1 running since addendum 1;
G2 stays in the selection, so those runs count as confirmation runs).  Comparison layer 1 (§10.2): each candidate's seeds 0–2 vs P41 SimCLR seeds 0–2
(standard augmentation), per-seed accuracies, paired differences, mean, sd and interval; plus candidate vs its same-init control (seeds 0–2), which
decides between "better starting point" and "scale held fixed throughout".  Layer 2 (seeds 3–4 and strong augmentation, both sides) only for a
candidate that is stably positive in layer 1.

**Configs.**  `make_p104_configs.py --uids U2F` (new variant); the generator reproduces the frozen U2 seed-0 YAML byte for byte (6b8c9a6a…).
New hashes: U2 s1 feded306…, s2 c7bf8b94…; U2F s0 57ababbf…, s1 bfe063f2…, s2 4204d01a…; G2F s2 d56c8195….
