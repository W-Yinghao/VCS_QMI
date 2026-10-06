# Pre-registration — P143 (MV6-I2): fixed VCS measurement, vary encoder — VICReg and selected logistic on CIFAR-10, four families on CIFAR-100 — DRAFT 2026-10-06

Source: CVPR-MV6-HANDOFF-20261006-r2 (docs/04 §6, docs/05 §A–F); intake `reports/manuscript_v6_handoff/intake_report.md`.  Axis
`fixed_measurement_vary_encoder`.  Existing checkpoints only (no pretraining).  DRAFT until the owner's go (D1, D3).

## 1. Protocol (unchanged from the frozen P109 I1 / P118 audit)
`scripts/p109_i1_features.py` → `scripts/p109_i1_audit.py`: 20 000 base images from the 45k FIT partition, FIT 10 000 / VAL 2 000 / EVAL
8 000 by base-image ID (fixed seed, identical for all encoders); versions clean, colour .05/.1/.2, blur .25/.5/1 (P45 constructor); layer family
layer3 (pooled), h, z (L2-normalised projector output — the same site definition for every encoder, including VICReg whose training loss uses
un-normalised z), logits of a multinomial logistic head on standardised clean h fitted on FIT.  Cells: planted colour 0.1 / n 2000 and blur
0.25 / n 2000, R = 100; nulls `null_label_only` and `null_all_planted`, R = 200; B = 200 shared within-class permutations, exact Q, α = .05,
max-T over the layer family; statistics VCS `closed_form_critic` (primary, the Table 4 column) and JS `exact_js_critic` (separate table);
paired prediction effects on the 8 000 EVAL pairs (Δacc in points, Δp_true, flip %, margin), paired bootstrap by base image.  Critic weights
are refitted per encoder under this procedure; the procedure, strengths and n do not change with the encoder.

## 2. Stage 0 — reproduction gate (the deleted cache)
Re-extract `P107_AP3_views4_800ep_seed1` and rerun its colour cell (power + level) with the recorded seed.  Pass: every layer × statistic power
within ± 0.05 of `reports/P118_i1_P107_AP3_views4_800ep_seed1_colour_{power,level}.json` and the colour-0.1 effects inside the recorded 95 %
intervals.  Feature extraction is not bit-deterministic on GPU, so exact equality is not required.  Failure stops stages 1–2 (technical).

## 3. Stage 1 — CIFAR-10, pre-selected encoders (chosen by provenance, before any EVAL)
| encoder (display) | runs | provenance |
|---|---|---|
| VICReg | `P41_vicreg_views4_800ep_seed1`, `_seed2` | P41 addendum 2 prereg/report; configs `cifar10_hpN_vicreg_views4_800ep_seed*.yaml` (projector 512→128, small; disclosed) |
| selected balanced logistic | `P129_JS_c10_a2_k0.25_seed1`, `_seed2` | P129 addendum 1 (selected C10 cell) |
Families colour and blur → 8 audit units.  Rows join the existing four VCS / SimCLR rows as a separately versioned block.

## 4. Stage 2 — CIFAR-100 (decision D3)
Runner change (reviewed, unit-tested): `--dataset {cifar10,cifar100}` via `load_train_partition`, head with `n_classes` outputs,
per-class effect rows over `n_classes`; parity rule for P(N | Y) unchanged (50 even / 50 odd classes); a CIFAR-10 smoke before/after the
change must give identical outputs.  Encoders (seeds 1–2): A-P3 `P107_AP3_c100_views4_800ep`, SimCLR `P91_c100_simclr_views4_800ep`,
VICReg `P91_c100_vicreg_views4_800ep`, selected logistic `P129_JS_c100_a3_k0.5`.  Same strengths (colour 0.1, blur 0.25), n, repeats and
permutations; base images from the CIFAR-100 45k FIT partition.  16 audit units.  At n 2000 about 20 images per class enter each draw — stated,
not tuned.

## 5. Readouts and rules
Each encoder seed is one row (never averaged into "mean ± sd" across two seeds; 100 draws are not encoder seeds).  H and O rejection, null
rejection and flags (> 0.09 at R = 200, any layer / statistic, reported), max-T power, effects with intervals, detection n and response n
listed separately.  No ordering of encoders by "interpretability"; detection ≠ amount of information; non-rejection ≠ absence.

## 6. Cost (estimate; smoke replaces it)
GPU ≈ 1–1.5 h for 13 extractions (one 1-GPU job); CPU ≈ 15–22 h on 16 cores for 25 audit units (P118: power 4–8 min, level 18–45 min per
unit), bundled into ≈ 5 jobs.

## 7. Not claimed
New pretraining; strengths chosen after seeing EVAL; pooling with Table 3; causal or "VCS is more interpretable" statements.
