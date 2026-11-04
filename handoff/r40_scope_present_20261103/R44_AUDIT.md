Publication copy (sample names aliased; numbers and conclusions unchanged).

# R44 audit (zero training, no TEST)

**Cause:** LAYOUT LEAK through the JOINT student centring, not a routing-arithmetic bug and not DEV data in the training graph. The R39 JOINT student vector is u = l2norm(e - mu[batch]) with mu = the PARENT per-native-batch mean of 128 SEQUENTIAL rows of the native eval layout (row r = image r//5, caption r), so an image and its own captions always share a batch (DEV: the true pairs; FIT: the clean grouping k//5, i.e. truth). ROUTE_PRESENT trains the encoder only on present rows (enriched for clean records) and removes the null-row push-down; the encoder output collapses (DEV mean cosine of random distinct items 0.97-0.99 vs 0.76-0.87 for ALL / parent) and the head reads pair identity from the shared batch mean. With layout-free DEV centring (one global mean, or independent random batches for images and captions) the R44 ROUTE_PRESENT units fall to 489-496 (F60), 454.6 (F80), 520.0 (F00), i.e. below the frozen parent H; student-cosine-only DEV is 9-90 rSum (near chance on F80).

| cond | unit | logged 8000 | H | two-stage native | two-stage global | two-stage decoupled | global-H | student-cos native | student-cos global | collapse cos img/cap | rel. dist |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n60 | parent_fresh_student | NA | 505.21 | 505.21 | 505.21 | 505.21 | 0.00 | 505.21 | 496.63 | 0.759/0.409 | NA |
| n60 | RP_J | 567.50 | 505.21 | 567.50 | 489.39 | 484.79 | -15.82 | 89.53 | 40.16 | 0.969/0.981 | 0.07 |
| n60 | RP_NCE | 569.76 | 505.21 | 569.76 | 496.11 | 494.48 | -9.09 | 33.77 | 9.94 | 0.976/0.992 | 0.09 |
| n60 | RR_J | 515.35 | 505.21 | 515.35 | 498.17 | 496.49 | -7.04 | 501.48 | 484.83 | 0.854/0.584 | 0.02 |
| n60 | ALL_J | 517.87 | 505.21 | 517.87 | 496.53 | 496.02 | -8.68 | 504.12 | 485.05 | 0.866/0.616 | 0.02 |
| n60 | PA_J | 516.11 | 505.21 | 516.11 | 496.73 | 495.36 | -8.48 | 503.16 | 486.94 | 0.855/0.586 | 0.02 |
| n00 | parent_fresh_student | NA | 522.07 | 522.07 | 522.07 | 522.07 | 0.00 | 522.07 | 511.85 | 0.830/0.298 | NA |
| n00 | RP_J | 532.52 | 522.07 | 532.52 | 519.96 | 519.92 | -2.11 | 467.79 | 449.29 | 0.932/0.566 | 0.02 |
| n00 | ALL_J | 527.59 | 522.07 | 527.59 | 520.41 | 519.72 | -1.66 | 435.64 | 423.31 | 0.924/0.451 | 0.02 |
| f80in | parent_fresh_student | NA | 464.93 | 464.93 | 464.93 | 464.93 | 0.00 | 464.93 | 453.85 | 0.810/0.639 | NA |
| f80in | RP_J | 542.50 | 464.93 | 542.50 | 454.62 | 452.09 | -10.32 | 12.96 | 5.68 | 0.984/0.995 | 0.09 |
| f80in | ALL_J | 457.55 | 464.93 | 457.55 | 448.15 | 447.22 | -16.79 | 449.31 | 438.64 | 0.844/0.700 | 0.03 |

1. Parameters: same 42 trainable student tensors per peer as R39 ALL (stud[p].trainable(); r39ext identical); all 42 moved; word embedding / BN statistics / unused BN unchanged (frozen_state digest == parent, 0 buffers moved, 0 non-trainable tensors moved) in RP, RR, ALL, PA
2. Leakage: layout: the fixed parent per-native-batch centring vectors (registered R39 JOINT patch) encode which images and captions are co-located in the native sequential layout; FIT: co-location = clean grouping k//5 (truth); DEV: co-location = the true pairs. The student can exploit it once its own output carries little item signal. Mask / p_d: present mask = D36.valid_rows(lists): A mass on the null slot from the OBSERVED map; p_d from observed map + parent candidate tables; record//5 truth not read Native evaluation: the native IN2R evaluation (VSE.forward_emb per-batch mean subtraction over the sequential loader) has the same layout dependence: the parent student-cosine DEV readout with native per-batch centring vs layout-free centring is 505.2 vs 496.6 / 493.6 (F60), 522.1 vs 511.9 / 510.8 (F00), 464.9 vs 453.8 / 452.1 (F80). This affects every native H number (DEV and the TEST tool use the native path) and the frozen-H cache / stage-1 top-100 of all critic lines; it is constant across arms of one parent but not zero.
3. Re-evaluation: table above (native reproduces the logged step-8000 rSum exactly).
4. Scaling: loss_enc = masked row sum / (n p_d): present t2i rows weighted 1/p_d (F60 2.5x, F80 4.4x). The per-peer clip is almost never active (F60 RP frac_clipped 0.0014 at 8000), and AdamW normalises step size per parameter, so the scaling does not make a larger effective lr; encoder grad norms are similar to ROUTE_RANDOM (end window img 0.376 vs 0.437, txt 0.507 vs 0.386). The 3.5x larger parameter distance (0.071 vs 0.021) comes from a consistent gradient direction (collapse toward the layout code), not from the scale.

Scope: all JOINT units (R39, R39X, R40 encoder arms, R41, R42, R43 joint, R44, R45, R46-B) are evaluated with the same centring; under layout-free centring the R39 F60 ALL J falls 517.9 -> 496.5 (H 505.2, i.e. -8.7 instead of +12.7), R41-C F80 ALL J 457.6 -> 448.1 (H 464.9), R39X F00 ALL J 527.6 -> 520.4 (H 522.1). The JOINT gains reported since R39 are therefore not established; frozen-encoder (head-only) units use the frozen-H cache and are not affected by the student centring (only by the native-evaluation layout effect common to all arms).

Caveat: global / decoupled centring changes the student input statistics. Reference points: the parent (same model, only centring changed) loses 9-11 rSum in its student-cosine readout and 0 in the two-stage readout (its head residual is zero, stage 1 = frozen-H cache); the JOINT units lose 7-21 rSum two-stage (ALL / PA / RR) and 12-88 (ROUTE_PRESENT). How much of the ALL drop is layout leak vs distribution shift is not separated by this audit; a JOINT student trained with layout-free centring (e.g. global or shuffled-batch FIT means) would separate them.
