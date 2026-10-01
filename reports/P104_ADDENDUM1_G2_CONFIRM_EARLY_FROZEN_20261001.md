# P104 addendum 1 — early confirmation runs for G2 and its same-initialisation control — FROZEN 2026-10-01T13:19:02Z before these runs

Parent: `P104_V3_FIRST_BATCH_PREREG_FROZEN_20260930.md`.  Owner prompt 2026-10-01 ("为什么不提交后续任务，没有任务了吗") — use the idle GPU quota.

**State at the freeze.**  First-batch seed-0 results in: G1 87.04, G2 **88.66**, G3 87.64, U1 87.56, N2 86.84 (linear-val; kNN 85.32 / 86.96 / 85.38 /
85.58 / 84.84).  N1, G4 and U2 still training.  Spec §10.1 selects ≤ 2 candidates only after all eight finish.

**What is submitted now (4 × 800-epoch, normal QOS, RTX6000PRO / H100):**
1. `P104_G2F_views4_800ep_seed{0,1}` — the §4.1 same-initialisation control: a, b start at (2, −1) and are **learned** (CosineCritic, both trainable);
   everything else identical to G2.  §4.1 explicitly allows this control before the batch completes; it is required before any mechanism statement about G2.
2. `P104_G2_views4_800ep_seed{1,2}` — G2's confirmation seeds, submitted before the selection point.

**Rule kept.**  The §10.1 selection is still made from the eight seed-0 runs alone, by final linear-val (kNN adjacent; < 0.1 → cheaper / simpler).
G2 drops out of the top two only if two of N1, G4, U2 exceed 88.66.  If that happens, G2 seeds 1–2 and G2F are reported as **extra runs outside the
selection** and do not enter any candidate-vs-SimCLR claim.  No hyper-parameter changes after seeing these seeds.  G2F seed 2 follows when the selection is
made (paired with G2 seeds 0–2 for the §4.1 comparison).

**Configs.**  `configs/make_p104_configs.py --uids` (new option; merges into `P104_SHA256.json`, leaves `slurm/p104_units.txt` untouched).  The
generator reproduces the frozen G2 seed-0 YAML byte for byte (sha256 2ddb42ab…).  New hashes: G2 s1 8f5cf1ba…, G2 s2 f1ac61db…, G2F s0 ed4db1ad…, G2F s1 3e904b69….
Seeds 1 / 2 inherit their bases `cifar10_hpK_a5_views4_800ep_vcs_seed{1,2}.yaml` (P35 recipe seeds).
