# Pre-registration — ablations inside the current recipe + optimizer re-check (P39), frozen 2026-09-26 before launch

Owner (2026-09-26): "看看还有没有要提交的实验" — all 8 GPU slots are busy; these units are queued so the slots that free up over the next
1–2 h are not idle.  Current recipe: cosine critic tanh(a⟨z1,z2⟩+b), K = 8, negative partner detached, a0 = 5, 4 views (J averaged over
the 6 view pairs).  Established: a0 = 5 on 3 seeds (P32), 4 views at equal compute (P34) and equal epochs (P29); 3 seeds of a0 = 5 + 4 views
and the views curve are running (P35, P37).

## Units
| unit | recipe | change | compute | compared with | question |
|---|---|---|---|---|---|
| a5_views4_k1_100ep | 4 views, 100 ep | K = 1 (one shifted negative per pair) | 1× | `P37_vcs_a5_views4_100ep` (K = 8) | do negatives still matter once detach + 4 views carry the signal?  (P29 §4: under detach the positives are the unsaturated side) |
| a5_views4_nodetach_100ep | 4 views, 100 ep | negative detach off | 1× | same | does the +2.4 detach effect (2 views) persist with 4 views? |
| a5_views4_b128_100ep | 4 views, 100 ep | batch 128 | 1× | same, and the 2-view 200-ep base (81.56) | equal compute *and* equal optimizer steps (352 × 100 = 176 × 200): is the 4-view gain about pairs per step or steps? |
| a5_lr2e-3 | 2 views, 200 ep | lr 2e-3 | 1× | a0 = 5 base 81.56 ± 0.44 | LR was tuned on the MLP-critic recipe (P15); re-check at the new operating point |
| a5_lr5e-4 | 2 views, 200 ep | lr 5e-4 | 1× | same | " |

Warm-up 10 epochs throughout; B = 128 keeps lr 1e-3 (AdamW; P17 found B = 128 neutral at 2 views with lr fixed).

## Reading
4-view units: final linear vs `P37_vcs_a5_views4_100ep` (running; single seed) with HELPS/HURTS at ±1.0, plus kNN, h-rank, heldout-J,
positive/negative saturation fractions.  k1 ≈ k8 ⇒ negatives are a minor ingredient under this recipe (mechanism claim for the paper);
nodetach ≪ k8 ⇒ detach is necessary independently of views.  LR units vs 81.56 ± 0.44.  Single seed.  No code change since gate 1009237.
sha256: `configs/HPARAM_M_SHA256.json`.

## Not queued (and why)
Controls at equal tuning budget: owner deferred until the VCS recipe is settled.  K = 255 / EMA / predictor / critic capacity / projector
variants: closed by P25–P29.  Seeds for the views curve: after the single-seed curve is read.

## Addendum 2026-09-26 03:00 UTC — batch size / optimizer steps (queued after the b128 result, before these units)
`a5_views4_b128_100ep` = **83.52 / kNN 78.30** vs `a5_views4_100ep` (B = 256) 81.88 / 77.74 at identical compute: halving the batch
(35 100 vs 17 500 optimizer steps) is worth +1.6 linear, and the critic reaches the sharp regime (a 10.1 / b −8.2) that the B = 256 run
only reaches at 200 epochs.  So at fixed compute the 4-view gain is largely a *steps* effect (the equal-epoch 4-view run, 84.54, has the
same 35 k steps at 2× compute).  Three units follow the lead (single seed, lr 1e-3 unchanged, as in the P17 batch sweep):
| unit | views | epochs | B | steps | compute | compared with | question |
|---|---|---|---|---|---|---|---|
| a5_views4_b64_100ep | 4 | 100 | 64 | 70 k | 1× | b128/100ep 83.52 | does the steps gain continue at B = 64? |
| a5_views4_b128_200ep | 4 | 200 | 128 | 70 k | 2× | b256/200ep 84.54 ± 0.16 | does it hold at the recipe's compute? |
| a5_b128 | 2 | 200 | 128 | 70 k | 1× | a0 = 5 base 81.56 ± 0.44 (35 k steps) | is it steps alone, without views?  (P17: B = 128 neutral on the MLP recipe) |
HELPS/HURTS ±1.0 vs the comparators; also kNN, rank, (a, b), saturation.  Configs in `HPARAM_M_SHA256.json`.
