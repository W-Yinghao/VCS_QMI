# P97_controls_16x — neutral results table (observed values only)

Generated 2026-10-01T05:27:52Z. Failed / stopped runs are listed, never dropped.

| run | method | K | seed | epochs done | linear-val (%) | kNN-val (%) | heldout-J (mean±sd) | h-rank | train time (s) | peak GPU MB (alloc/res) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P97_simclr_views4_1600ep_seed0 | simclr_matched | null | 0 | 1600/1600 | 87.64 | 87.56 | null | 180.51 | 31590 | 5269/8140 | COMPLETED |
| P97_vicreg_views4_1600ep_seed0 | vicreg_matched_128 | null | 0 | 1600/1600 | 86.06 | 83.40 | null | 91.04 | 32743 | 5269/8138 | COMPLETED |

## Hyper-parameters per run (from run_manifest.json)

| run | K | critic hidden | last gain | critic lr× | critic wd | proj hidden | proj out | critic input | critic impl | B | lr | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P97_simclr_views4_1600ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 1600 |
| P97_vicreg_views4_1600ep_seed0 | 1 | [512, 512] | 0.1 | 1.0 | 0.0 | 512 | 128 | l2 | None | 256 | 0.001 | 1600 |

## Epoch-0 (random init, same seed) reference and deltas

| run | linear-val ep0 (%) | linear-val final (%) | Δ linear | kNN ep0 (%) | kNN final (%) | Δ kNN | h-rank ep0 | h-rank final | heldout-J ep0 |
|---|---|---|---|---|---|---|---|---|---|
| P97_simclr_views4_1600ep_seed0 | 41.94 | 87.64 | 45.70 | 36.54 | 87.56 | 51.02 | 2.94 | 180.51 | null |
| P97_vicreg_views4_1600ep_seed0 | 41.94 | 86.06 | 44.12 | 36.54 | 83.40 | 46.86 | 2.94 | 91.04 | null |

## kNN trajectory (in-training monitor, selection top-1 %)

- P97_simclr_views4_1600ep_seed0: ep0: 36.54, ep50: 81.44, ep100: 84.62, ep200: 86.94, ep400: 87.20, ep800: 87.16, ep1200: 87.30, ep1600: 87.56
- P97_vicreg_views4_1600ep_seed0: ep0: 36.54, ep50: 80.04, ep100: 83.78, ep200: 84.26, ep400: 83.86, ep800: 83.68, ep1200: 83.36, ep1600: 83.40

## Held-out J trajectory (VCS only)


## Final-epoch training objective values (epoch means)

- P97_simclr_views4_1600ep_seed0: J_raw null, R_binary null, nt_xent 1.8108, vicreg {'vicreg_invariance': None, 'vicreg_variance': None, 'vicreg_covariance': None}
- P97_vicreg_views4_1600ep_seed0: J_raw null, R_binary null, nt_xent null, vicreg {'vicreg_invariance': 0.04137664425585951, 'vicreg_variance': 0.003502667049012546, 'vicreg_covariance': 1.104280960219247}

## Cost

| run | steady step (s) | images/s | views/s | train s | in-train eval s | final eval s | seen base images | critic params | job |
|---|---|---|---|---|---|---|---|---|---|
| P97_simclr_views4_1600ep_seed0 | 0.1114 | 2298 | 4596 | 31590 | 18 | 7 | 71680000 | None | 1015831@node59 |
| P97_vicreg_views4_1600ep_seed0 | 0.1155 | 2217 | 4433 | 32743 | 19 | 7 | 71680000 | None | 1015833@node58 |

## Provenance

- P97_simclr_views4_1600ep_seed0: commit `57a3851f62713eb16f4385410b2d134ead48690d` dirty=True, config `a8d0af0ce16014da430d41beb87888569f3c635db8fdc5dab3e69c76fe41d080`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`
- P97_vicreg_views4_1600ep_seed0: commit `e466fd548daf3ce53b97f157f4f91a75aa5c3d8b` dirty=True, config `8fe764b84430ba8aed49bafa913abcc43f342eca4759073c2489baf71101a711`, split `c35d7cd336e79680c2e2c077e5612ed9eb6779ff70a427f5e085bc4df2d617a9`, init enc `bdc3a4e33883343e29639fce1282566645a76c79099aaadda58244d9f9ccc767`

- identical encoder init across runs: True
- identical split across runs: True

## Per-method aggregation across seeds (only COMPLETED runs with a final evaluation; mean ± sample SD, n seeds)

| method | n | linear-val final (%) | linear-val ep0 (%) | Δ linear | kNN final (%) | h-rank final | heldout-J final |
|---|---|---|---|---|---|---|---|
| simclr_matched | 1 | 87.64 | 41.94 | 45.70 | 87.56 | 180.51 | null |
| vicreg_matched_128 | 1 | 86.06 | 41.94 | 44.12 | 83.40 | 91.04 | null |

## Coverage checks

- P97_simclr_views4_1600ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_1600.json), status COMPLETED, failure None
- P97_vicreg_views4_1600ep_seed0: epoch0 eval True, final eval True (evaluation_epoch_1600.json), status COMPLETED, failure None
