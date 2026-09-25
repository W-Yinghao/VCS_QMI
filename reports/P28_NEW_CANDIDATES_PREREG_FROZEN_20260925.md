# Pre-registration — new critic-form / view candidates (P28) and seed fills (P29), frozen 2026-09-25 before launch

Owner: "排进去，该补的 seed 也补上".  Candidates from `reports/SYNTHESIS_20260925.md` §5.C; the plan's content (J, mixture reference,
product-of-marginals negatives, tanh output) is unchanged.

## P28 — new candidates on the neg-detach cosine base (`cifar10_hpF_cos_negdetach_vcs_seed0.yaml`: 80.48 / kNN 74.62 / rank 30.9), seed 0, 200 ep
| unit | change | hypothesis |
|---|---|---|
| mono_spline | f = c0 + Σ_k softplus(w_k)·relu(s − t_k), 8 knots on [−1,1], s = ⟨z1,z2⟩, T = tanh f; init f(s) = s (= cosine a=1,b=0) | §4.4: a monotone but non-affine score lets gradient reach pairs across the whole similarity range instead of one threshold |
| diag_metric | T = tanh(a⟨norm(w⊙z1), norm(w⊙z2)⟩+b), w ∈ R¹²⁸ init 1 | §3.1: per-dimension weighting, less freedom than the shared W (P26 N6) |
| views4 | 4 independent views per image; J averaged over the 6 view pairs, each with its own K=8 shifts (negatives still from other UIDs) | §4.4: more positive pairs per step → higher gradient participation; not a new loss (same P, Q) |
| proj_linear | projector = one Linear(512→128, no BN/ReLU); critic unchanged | added after `GEOMETRY_DIAG.md`: plain cosine K=8 reaches SimCLR-like z-uniformity (−3.51 vs −3.84) but its h-uniformity stays at −1.56 vs −2.81; every VCS run has h-uniformity in −1.5…−1.9 while both controls are below −2.5, and h-uniformity orders the VCS runs roughly like linear-val (K=1 −1.51 → neg-detach base −1.79). A 2-layer MLP projector can absorb the spreading; a single linear map forces it into h |
| proj_bnonly | no projector: critic reads L2(BN(h)) with affine-free BN (output_dim = 512) | same hypothesis, stronger form; the BN replaces the failed `cos_on_h` (‖h‖ collapsed to 0.5 without scale control) |

Also implemented (code only, no run): `pairing.sampler=all_pairs_matrix` — the B(B−1) off-diagonal pairs at once for similarity-type critics;
tested equal to K=B−1 cyclic shifts in value and gradient (float64, with and without negative detach).  Future K=255 runs use it.

## P29 — seed fills
| run | source config | seeds |
|---|---|---|
| cosK8 | `cifar10_hpC_crit_cosine_vcs_seed0.yaml` (78.32) | 1, 2 |
| cosK255 | `cifar10_hpF_cos_k255_vcs_seed0.yaml` (79.32) | 1, 2 |
(The neg-detach base seeds 1/2 are in P26; seed 2 = 80.72.)  These pin the critic-form effect and the K effect on 3 seeds, as the external review asked.

## Reading
P28: final linear-val vs 80.48 (HELPS > +1.0 / HURTS < −1.0); rank, kNN, learned spline shape (w, c0) / w-spectrum for diag_metric;
views4 is also compared at equal *compute* (2× encoder forward per step).  P29: per-seed values and mean ± SD; the critic-form claim
("cosine > concat-MLP at K=8") is stated with 3 seeds vs the P10 K=8 MLP seeds.

Configs and sha256: `configs/HPARAM_H_SHA256.json`.
