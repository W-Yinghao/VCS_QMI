# Pre-registration — CIFAR-10 SSL development pilot (P0–P4), frozen 2026-09-24

This freezes what will be run and how it will be read **before any GPU compute**.  The binding contract is
`VCS_QMI_SSL_Server_Agent_Spec_v1.md` (sha256 `c89f992a…28a7e3`); this note only pins hashes and the reading rules.

## Question
Does the collaborator's VCS-QMI objective (`loss = -J`, ordinary joint update, tanh pair critic, K=1 cyclic negatives)
train a ResNet-18-CIFAR encoder whose *frozen* 512-d `h` improves over its own random initialization under a fixed 20-epoch
budget, and how does that compare with a matched SimCLR and a matched-128 VICReg run under the same components?

## Units
Three runs, seed 0, 20 epochs, B=256 images, identical encoder/projector initialization, identical fit/selection split.

| config | file sha256 | method |
|---|---|---|
| `configs/cifar10_pilot_vcs.yaml` | `6621d513c45fc9d487228264e80f262f51ee0511bddaaf24aa0b1381bd56199e` | `vcs_qmi` |
| `configs/cifar10_pilot_simclr.yaml` | `2f40f45b2738ead8c803abe4063302cce808bf2e5ae860f2332cb33f1bfb68ca` | `simclr_matched` |
| `configs/cifar10_pilot_vicreg.yaml` | `964845e7373e72fa958d211a9270ff02c1b3062cefafb65533da29057145f304` | `vicreg_matched_128` |

Split: `dev45k_val5k`, seed 20260924, 500/class selection (manifest sha256 recorded in `reports/P0_preflight.md` once created).

## Primary endpoint
`linear_val_top1_pct` of the frozen `h` at the **epoch-20 checkpoint** (fixed, not best-of), probe = Linear(512,10), SGD
lr 0.1 / mom 0.9 / wd 0, 100 epochs cosine→0.001×, seed 20260925, final probe epoch, trained on fit labels, scored on selection.
Reference: the same probe on the epoch-0 (`initial.pt`) checkpoint of the same seed.

## Secondary endpoints / diagnostics
`knn_val_top1_pct` (k=200, T=0.1) at epochs 0/5/10/20; `heldout_J` (VCS; 4 repeats mean±sd, selection images);
`h_effective_rank` / `z_effective_rank` on the first 4096 sorted selection UIDs; per-step `J_raw`, `R_binary`, saturation
fractions, per-module gradient norms; cost (steady-state images/s, train seconds, peak memory).

## Gates before P3 (no GPU pilot without them)
P1: 29 reference tests pass on the server; integration tests 1–10 pass (PARTIAL is reported as such).
P2: three 100-step smoke runs COMPLETED; first-step gradient check PASS for all modules; all losses/grads finite; J∈[-3,1],
J=1−R; in-training evaluation leaves training RNG untouched; GPU resume produces a full trajectory (bit-exactness reported, not gated).

## Pre-committed reading (spec §16.2, §18 D)
- **Encoder improved**: linear-val(ep20) > linear-val(ep0) by a margin that is not within probe noise (single seed → stated as an
  observation, not a significance claim).  If linear-val(ep20) ≤ linear-val(ep0): "no improvement over random features" — a result,
  not a bug, unless a gate failed.
- **Critic learned but features did not**: train `J_raw` and `heldout_J` rise while linear/kNN do not improve → report as
  "pair discrimination improved without transfer to representation quality".  Do not report critic accuracy in place of `h`.
- **Train/validation gap**: `heldout_J` markedly below final train `J_raw` → report as gap; no conclusion about the estimator's
  population properties.
- **Saturation/collapse**: `sat_*_frac` high, `J_raw` near 1, `effective_rank ≤ 2` or trace ≤ 1e-8 twice in a row → flag
  `COLLAPSE_SUSPECTED`; do not add regularizers, EMA, or change the critic.
- **Controls**: SimCLR/VICReg are sanity references for the shared pipeline; a 20-epoch single-seed ranking between methods is
  *not* a method verdict.  VCS below controls is reported as such.
- **Numerical failure** (non-finite loss/grad, OOM): stop, keep artifacts, report; no batch skipping, AMP, or batch reduction.

## What this pilot cannot claim
No test-set accuracy; no statement about 200-epoch behaviour; no "converges to S"; no Shannon-MI language; no statement that
K×B negatives are independent samples.

## Next step after P4 (single option, pre-committed)
If gates pass and nothing in the runs indicates an implementation error: propose the 200-epoch, 3-seed confirmation from the same
initialization with warmup 10 (plan `configs/next_stage_plan.yaml`) and **wait for the owner's decision**.  If an implementation
error is found: fix it, re-run P1–P3, report the patch reason.  Nothing else is started.
