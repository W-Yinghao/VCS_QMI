# R4-T2 — fixed-critic paired-sampling validation: results

Implementation checks (gate for stage 3): PASS; failures: none.

## Stage 1a — finite-state cross-pair: two-sided coverage of the target (%), Wilson-checked; [one-sided S coverage of the jk / img_t readout]

| setting | level | critic | n | target | naive | jk | blk_t | blk_hoeff | bd | K1_jk | K4_jk | K16_jk | S one-sided | mean width jk / img_t | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| finite | independent | fixed_random | 128 | -0.1164 | 99.6 | 94.6 | 94.6 | 100.0 | 100.0 | 99.0 | 97.0 | 95.0 | 100.0 | 0.0873 |  |
| finite | independent | fixed_random | 32 | -0.1164 | 98.4 | 89.6 | 91.8 | 100.0 | 100.0 | 98.6 | 93.8 | 90.8 | 100.0 | 0.1728 | jk under, blk_t under, K16_jk under |
| finite | independent | fixed_random | 512 | -0.1164 | 99.4 | 95.2 | 96.0 | 100.0 | 100.0 | 99.6 | 97.6 | 96.4 | 100.0 | 0.0440 |  |
| finite | independent | optimum | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | optimum | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | optimum | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | prescribed_error | 128 | -0.0100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | prescribed_error | 32 | -0.0100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | prescribed_error | 512 | -0.0100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | zero | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | zero | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | zero | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | strong | fixed_random | 128 | -0.0234 | 99.4 | 94.0 | 94.0 | 100.0 | 100.0 | 100.0 | 97.4 | 95.2 | 100.0 | 0.0666 |  |
| finite | strong | fixed_random | 32 | -0.0234 | 98.4 | 93.0 | 94.8 | 100.0 | 100.0 | 99.4 | 96.4 | 94.6 | 100.0 | 0.1345 | jk under |
| finite | strong | fixed_random | 512 | -0.0234 | 99.0 | 95.8 | 95.6 | 100.0 | 100.0 | 99.4 | 98.2 | 96.0 | 100.0 | 0.0332 |  |
| finite | strong | optimum | 128 | 0.2422 | 90.6 | 95.2 | 92.4 | 100.0 | 100.0 | 98.2 | 97.2 | 96.0 | 93.2 | 0.1212 | naive under, blk_t under |
| finite | strong | optimum | 32 | 0.2422 | 84.8 | 90.8 | 94.0 | 100.0 | 100.0 | 99.0 | 96.0 | 92.2 | 89.0 | 0.2445 | naive under, jk under, K16_jk under |
| finite | strong | optimum | 512 | 0.2422 | 88.0 | 93.6 | 95.0 | 100.0 | 100.0 | 98.2 | 97.0 | 94.6 | 92.0 | 0.0593 | naive under |
| finite | strong | prescribed_error | 128 | 0.2104 | 87.4 | 95.4 | 95.0 | 100.0 | 100.0 | 99.4 | 98.0 | 96.8 | 100.0 | 0.0884 | naive under |
| finite | strong | prescribed_error | 32 | 0.2104 | 80.0 | 92.0 | 92.8 | 100.0 | 100.0 | 97.0 | 95.8 | 93.4 | 97.2 | 0.1865 | naive under, jk under, blk_t under |
| finite | strong | prescribed_error | 512 | 0.2104 | 85.6 | 94.2 | 95.2 | 100.0 | 100.0 | 99.2 | 98.4 | 96.0 | 100.0 | 0.0439 | naive under |
| finite | strong | zero | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | strong | zero | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | strong | zero | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | weak | fixed_random | 128 | -0.0724 | 99.4 | 94.4 | 95.0 | 100.0 | 100.0 | 99.8 | 97.6 | 95.8 | 100.0 | 0.0795 |  |
| finite | weak | fixed_random | 32 | -0.0724 | 99.4 | 90.0 | 95.2 | 100.0 | 100.0 | 99.0 | 96.2 | 91.8 | 100.0 | 0.1550 | jk under, K16_jk under |
| finite | weak | fixed_random | 512 | -0.0724 | 99.0 | 94.0 | 93.8 | 100.0 | 100.0 | 99.8 | 97.2 | 96.2 | 100.0 | 0.0400 |  |
| finite | weak | optimum | 128 | 0.0500 | 96.6 | 97.2 | 96.4 | 100.0 | 100.0 | 99.2 | 96.8 | 97.2 | 96.0 | 0.0762 |  |
| finite | weak | optimum | 32 | 0.0500 | 92.8 | 94.4 | 95.4 | 100.0 | 100.0 | 98.0 | 94.4 | 94.8 | 93.8 | 0.1565 | naive under |
| finite | weak | optimum | 512 | 0.0500 | 92.2 | 92.8 | 94.8 | 100.0 | 100.0 | 98.2 | 95.2 | 94.2 | 93.2 | 0.0378 | naive under, jk under |
| finite | weak | prescribed_error | 128 | 0.0355 | 94.4 | 95.2 | 94.6 | 100.0 | 100.0 | 98.6 | 97.8 | 95.6 | 100.0 | 0.0498 |  |
| finite | weak | prescribed_error | 32 | 0.0355 | 92.6 | 94.4 | 95.4 | 100.0 | 100.0 | 98.8 | 96.4 | 94.2 | 98.6 | 0.1030 | naive under |
| finite | weak | prescribed_error | 512 | 0.0355 | 93.6 | 95.2 | 94.4 | 100.0 | 100.0 | 97.6 | 96.6 | 95.8 | 100.0 | 0.0246 |  |
| finite | weak | zero | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | weak | zero | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | weak | zero | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |

## Stage 2 — existing generators, cross-pair: two-sided coverage of the target (%), Wilson-checked; [one-sided S coverage of the jk / img_t readout]

| setting | level | critic | n | target | naive | jk | blk_t | blk_hoeff | bd | K1_jk | K4_jk | K16_jk | S one-sided | mean width jk / img_t | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gaussian | independent | optimum | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| gaussian | independent | optimum | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| gaussian | independent | optimum | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| gaussian | independent | prescribed_error | 128 | -0.0100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| gaussian | independent | prescribed_error | 32 | -0.0100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| gaussian | independent | prescribed_error | 512 | -0.0100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| gaussian | independent | zero | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| gaussian | independent | zero | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| gaussian | independent | zero | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| gaussian | strong | optimum | 128 | 0.5672 | 92.6 | 94.0 | 94.6 | 100.0 | 100.0 | 96.8 | 96.2 | 95.2 | 92.0 | 0.1378 | naive under |
| gaussian | strong | optimum | 32 | 0.5672 | 90.6 | 91.0 | 94.4 | 100.0 | 100.0 | 96.4 | 94.6 | 92.8 | 92.2 | 0.2837 | naive under, jk under, K16_jk under |
| gaussian | strong | optimum | 512 | 0.5672 | 94.0 | 94.8 | 97.4 | 100.0 | 100.0 | 98.0 | 96.0 | 94.0 | 93.0 | 0.0687 |  |
| gaussian | strong | prescribed_error | 128 | 0.5063 | 90.6 | 93.2 | 94.4 | 100.0 | 100.0 | 97.4 | 96.4 | 93.8 | 100.0 | 0.0957 | naive under |
| gaussian | strong | prescribed_error | 32 | 0.5063 | 88.8 | 92.4 | 94.2 | 100.0 | 100.0 | 96.8 | 95.8 | 92.6 | 99.6 | 0.1945 | naive under, jk under, K16_jk under |
| gaussian | strong | prescribed_error | 512 | 0.5063 | 93.0 | 96.0 | 95.4 | 100.0 | 100.0 | 99.4 | 96.4 | 95.4 | 100.0 | 0.0473 | naive under |
| gaussian | strong | zero | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| gaussian | strong | zero | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| gaussian | strong | zero | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| gaussian | weak | optimum | 128 | 0.0480 | 93.6 | 93.6 | 94.6 | 100.0 | 100.0 | 97.6 | 96.0 | 94.4 | 93.4 | 0.0717 |  |
| gaussian | weak | optimum | 32 | 0.0480 | 93.4 | 93.4 | 95.6 | 100.0 | 100.0 | 98.8 | 97.2 | 94.6 | 92.8 | 0.1454 |  |
| gaussian | weak | optimum | 512 | 0.0480 | 95.0 | 95.0 | 96.2 | 100.0 | 100.0 | 98.4 | 96.4 | 95.0 | 96.0 | 0.0358 |  |
| gaussian | weak | prescribed_error | 128 | 0.0336 | 95.2 | 95.8 | 95.4 | 100.0 | 100.0 | 99.0 | 96.6 | 96.6 | 99.6 | 0.0458 |  |
| gaussian | weak | prescribed_error | 32 | 0.0336 | 94.6 | 95.8 | 95.8 | 100.0 | 100.0 | 98.2 | 97.8 | 96.0 | 98.8 | 0.0950 |  |
| gaussian | weak | prescribed_error | 512 | 0.0336 | 94.8 | 95.0 | 95.2 | 100.0 | 100.0 | 98.4 | 97.4 | 96.0 | 100.0 | 0.0228 |  |
| gaussian | weak | zero | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| gaussian | weak | zero | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| gaussian | weak | zero | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| xor_mixture | independent | optimum | 128 | -0.0000 | 96.6 | 97.0 | 96.2 | 100.0 | 100.0 | 98.6 | 98.8 | 97.2 | 98.0 | 0.0000 | degenerate |
| xor_mixture | independent | optimum | 32 | -0.0000 | 95.2 | 96.0 | 97.6 | 100.0 | 100.0 | 98.6 | 97.8 | 96.2 | 98.4 | 0.0000 | degenerate |
| xor_mixture | independent | optimum | 512 | -0.0000 | 99.4 | 99.4 | 99.0 | 100.0 | 100.0 | 99.8 | 99.8 | 99.8 | 99.0 | 0.0000 | degenerate |
| xor_mixture | independent | prescribed_error | 128 | -0.0100 | 98.8 | 98.8 | 98.2 | 100.0 | 100.0 | 99.2 | 99.6 | 99.4 | 100.0 | 0.0000 | degenerate |
| xor_mixture | independent | prescribed_error | 32 | -0.0100 | 97.0 | 97.4 | 96.6 | 100.0 | 100.0 | 99.0 | 98.4 | 97.4 | 100.0 | 0.0000 | degenerate |
| xor_mixture | independent | prescribed_error | 512 | -0.0100 | 99.6 | 99.6 | 99.0 | 100.0 | 100.0 | 100.0 | 99.8 | 99.6 | 100.0 | 0.0000 | degenerate |
| xor_mixture | independent | zero | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| xor_mixture | independent | zero | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| xor_mixture | independent | zero | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| xor_mixture | strong | optimum | 128 | 0.5788 | 88.8 | 92.8 | 94.8 | 100.0 | 100.0 | 97.6 | 94.8 | 92.4 | 91.0 | 0.1402 | naive under, jk under, K16_jk under |
| xor_mixture | strong | optimum | 32 | 0.5788 | 87.8 | 93.4 | 93.8 | 100.0 | 100.0 | 96.4 | 95.6 | 94.6 | 90.2 | 0.2841 | naive under |
| xor_mixture | strong | optimum | 512 | 0.5788 | 91.2 | 96.0 | 93.0 | 100.0 | 100.0 | 98.6 | 97.0 | 95.4 | 93.2 | 0.0699 | naive under, blk_t under |
| xor_mixture | strong | prescribed_error | 128 | 0.5168 | 86.2 | 93.6 | 92.6 | 100.0 | 100.0 | 97.0 | 95.4 | 94.8 | 100.0 | 0.1039 | naive under, blk_t under |
| xor_mixture | strong | prescribed_error | 32 | 0.5168 | 84.8 | 93.2 | 93.0 | 100.0 | 100.0 | 95.6 | 97.4 | 93.6 | 99.2 | 0.2124 | naive under, blk_t under |
| xor_mixture | strong | prescribed_error | 512 | 0.5168 | 86.2 | 94.8 | 96.4 | 100.0 | 100.0 | 98.6 | 97.2 | 95.2 | 100.0 | 0.0515 | naive under |
| xor_mixture | strong | zero | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| xor_mixture | strong | zero | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| xor_mixture | strong | zero | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| xor_mixture | weak | optimum | 128 | 0.0444 | 94.6 | 95.2 | 96.2 | 100.0 | 100.0 | 98.2 | 98.0 | 95.6 | 95.2 | 0.0702 |  |
| xor_mixture | weak | optimum | 32 | 0.0444 | 92.4 | 93.6 | 95.4 | 100.0 | 100.0 | 97.8 | 95.0 | 93.6 | 94.8 | 0.1424 | naive under |
| xor_mixture | weak | optimum | 512 | 0.0444 | 95.0 | 95.6 | 96.6 | 100.0 | 100.0 | 98.2 | 96.8 | 96.2 | 95.4 | 0.0348 |  |
| xor_mixture | weak | prescribed_error | 128 | 0.0305 | 92.8 | 94.8 | 95.6 | 100.0 | 100.0 | 98.2 | 97.6 | 95.0 | 99.6 | 0.0462 | naive under |
| xor_mixture | weak | prescribed_error | 32 | 0.0305 | 92.4 | 94.0 | 96.8 | 100.0 | 100.0 | 97.2 | 96.0 | 93.8 | 98.2 | 0.0946 | naive under |
| xor_mixture | weak | prescribed_error | 512 | 0.0305 | 93.0 | 93.8 | 95.6 | 100.0 | 100.0 | 99.0 | 97.6 | 94.0 | 100.0 | 0.0230 | naive under |
| xor_mixture | weak | zero | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| xor_mixture | weak | zero | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| xor_mixture | weak | zero | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |

## Stage 1b — finite-state image-local: two-sided coverage of the target (%), Wilson-checked; [one-sided S coverage of the jk / img_t readout]

| setting | level | critic | n | target | img_t | pooled_naive | img_hoeff | boot | S one-sided | mean width jk / img_t | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| finite | independent | fixed_random | 128 | -0.1164 | 94.6 | 99.6 | 100.0 | 94.0 | 100.0 | 0.0412 |  |
| finite | independent | fixed_random | 32 | -0.1164 | 93.4 | 100.0 | 100.0 | 92.0 | 100.0 | 0.0834 | boot under |
| finite | independent | fixed_random | 512 | -0.1164 | 95.6 | 100.0 | 100.0 | 95.4 | 100.0 | 0.0206 |  |
| finite | independent | optimum | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | optimum | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | optimum | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | prescribed_error | 128 | -0.0100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | prescribed_error | 32 | -0.0100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | prescribed_error | 512 | -0.0100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | zero | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | zero | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | independent | zero | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | strong | fixed_random | 128 | -0.0385 | 95.8 | 100.0 | 100.0 | 95.0 | 100.0 | 0.0314 |  |
| finite | strong | fixed_random | 32 | -0.0385 | 93.0 | 99.8 | 100.0 | 92.4 | 100.0 | 0.0645 | img_t under, boot under |
| finite | strong | fixed_random | 512 | -0.0385 | 94.6 | 99.8 | 100.0 | 94.2 | 100.0 | 0.0156 |  |
| finite | strong | optimum | 128 | 0.1284 | 95.2 | 94.2 | 100.0 | 94.8 | 100.0 | 0.0628 |  |
| finite | strong | optimum | 32 | 0.1284 | 95.0 | 95.6 | 100.0 | 94.4 | 100.0 | 0.1283 |  |
| finite | strong | optimum | 512 | 0.1284 | 94.0 | 93.8 | 100.0 | 93.4 | 100.0 | 0.0312 |  |
| finite | strong | prescribed_error | 128 | 0.1153 | 93.4 | 92.0 | 100.0 | 94.0 | 100.0 | 0.0477 | pooled_naive under |
| finite | strong | prescribed_error | 32 | 0.1153 | 94.4 | 92.0 | 100.0 | 93.0 | 100.0 | 0.0979 | pooled_naive under, boot under |
| finite | strong | prescribed_error | 512 | 0.1153 | 94.2 | 92.2 | 100.0 | 93.8 | 100.0 | 0.0238 | pooled_naive under |
| finite | strong | zero | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | strong | zero | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | strong | zero | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | weak | fixed_random | 128 | -0.0796 | 94.2 | 100.0 | 100.0 | 93.8 | 100.0 | 0.0375 |  |
| finite | weak | fixed_random | 32 | -0.0796 | 94.4 | 99.6 | 100.0 | 93.6 | 100.0 | 0.0757 |  |
| finite | weak | fixed_random | 512 | -0.0796 | 96.0 | 100.0 | 100.0 | 96.2 | 100.0 | 0.0186 |  |
| finite | weak | optimum | 128 | 0.0204 | 94.2 | 98.6 | 100.0 | 94.2 | 100.0 | 0.0332 |  |
| finite | weak | optimum | 32 | 0.0204 | 97.2 | 98.8 | 100.0 | 96.2 | 100.0 | 0.0679 |  |
| finite | weak | optimum | 512 | 0.0204 | 95.8 | 98.0 | 100.0 | 95.8 | 100.0 | 0.0165 |  |
| finite | weak | prescribed_error | 128 | 0.0129 | 95.4 | 98.4 | 100.0 | 94.8 | 100.0 | 0.0218 |  |
| finite | weak | prescribed_error | 32 | 0.0129 | 92.6 | 98.6 | 100.0 | 91.4 | 100.0 | 0.0443 | img_t under, boot under |
| finite | weak | prescribed_error | 512 | 0.0129 | 94.2 | 99.0 | 100.0 | 94.4 | 100.0 | 0.0108 |  |
| finite | weak | zero | 128 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | weak | zero | 32 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |
| finite | weak | zero | 512 | 0.0000 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0000 | degenerate |

## Stage 1a — Monte Carlo variance / exact Hoeffding variance (99.9 % bootstrap interval)

| level | critic | n | ζ₁ | E h₂² | ratio | interval |
|---|---|---|---|---|---|---|
| independent | fixed_random | 128 | 1.60e-02 | 1.41e-02 | 0.954 | [0.775, 1.173] |
| independent | fixed_random | 32 | 1.60e-02 | 1.41e-02 | 1.207 | [0.963, 1.467] |
| independent | fixed_random | 512 | 1.60e-02 | 1.41e-02 | 0.984 | [0.797, 1.195] |
| strong | fixed_random | 128 | 9.20e-03 | 1.69e-02 | 1.060 | [0.867, 1.259] |
| strong | fixed_random | 32 | 9.20e-03 | 1.69e-02 | 0.871 | [0.687, 1.063] |
| strong | fixed_random | 512 | 9.20e-03 | 1.69e-02 | 0.983 | [0.799, 1.193] |
| strong | optimum | 128 | 2.95e-02 | 1.22e-01 | 0.966 | [0.785, 1.162] |
| strong | optimum | 32 | 2.95e-02 | 1.22e-01 | 0.958 | [0.799, 1.115] |
| strong | optimum | 512 | 2.95e-02 | 1.22e-01 | 1.055 | [0.858, 1.286] |
| strong | prescribed_error | 128 | 1.59e-02 | 8.69e-02 | 0.983 | [0.805, 1.175] |
| strong | prescribed_error | 32 | 1.59e-02 | 8.69e-02 | 1.078 | [0.866, 1.313] |
| strong | prescribed_error | 512 | 1.59e-02 | 8.69e-02 | 0.994 | [0.796, 1.210] |
| weak | fixed_random | 128 | 1.33e-02 | 1.47e-02 | 0.931 | [0.764, 1.098] |
| weak | fixed_random | 32 | 1.33e-02 | 1.47e-02 | 1.005 | [0.807, 1.214] |
| weak | fixed_random | 512 | 1.33e-02 | 1.47e-02 | 1.016 | [0.811, 1.212] |
| weak | optimum | 128 | 1.19e-02 | 2.53e-02 | 0.898 | [0.729, 1.098] |
| weak | optimum | 32 | 1.19e-02 | 2.53e-02 | 0.972 | [0.779, 1.173] |
| weak | optimum | 512 | 1.19e-02 | 2.53e-02 | 1.072 | [0.879, 1.262] |
| weak | prescribed_error | 128 | 5.02e-03 | 1.49e-02 | 0.913 | [0.752, 1.101] |
| weak | prescribed_error | 32 | 5.02e-03 | 1.49e-02 | 1.054 | [0.873, 1.254] |
| weak | prescribed_error | 512 | 5.02e-03 | 1.49e-02 | 1.014 | [0.830, 1.199] |

## Sparse cyclic shifts — formula / resampled conditional variance, total / predicted variance, evaluations

| stage | setting | level | critic | n | K | formula ratio | total ratio | evals (K) / full | critic s full / K16 |
|---|---|---|---|---|---|---|---|---|---|
| 1a | finite | independent | fixed_random | 128 | 1 | 0.997 | 1.045 | 256 / 16384 | — |
| 1a | finite | independent | fixed_random | 128 | 4 | 0.996 | 1.014 | 640 / 16384 | — |
| 1a | finite | independent | fixed_random | 128 | 16 | 0.999 | 0.993 | 2176 / 16384 | — |
| 1a | finite | independent | fixed_random | 32 | 1 | 1.003 | 0.987 | 64 / 1024 | — |
| 1a | finite | independent | fixed_random | 32 | 4 | 0.997 | 1.012 | 160 / 1024 | — |
| 1a | finite | independent | fixed_random | 32 | 16 | 1.011 | 0.988 | 544 / 1024 | — |
| 1a | finite | independent | fixed_random | 512 | 1 | 0.984 | 1.023 | 1024 / 262144 | — |
| 1a | finite | independent | fixed_random | 512 | 4 | 0.997 | 0.994 | 2560 / 262144 | — |
| 1a | finite | independent | fixed_random | 512 | 16 | 1.006 | 0.989 | 8704 / 262144 | — |
| 1a | finite | independent | optimum | 128 | 1 | — | — | 256 / 16384 | — |
| 1a | finite | independent | optimum | 128 | 4 | — | — | 640 / 16384 | — |
| 1a | finite | independent | optimum | 128 | 16 | — | — | 2176 / 16384 | — |
| 1a | finite | independent | optimum | 32 | 1 | — | — | 64 / 1024 | — |
| 1a | finite | independent | optimum | 32 | 4 | — | — | 160 / 1024 | — |
| 1a | finite | independent | optimum | 32 | 16 | — | — | 544 / 1024 | — |
| 1a | finite | independent | optimum | 512 | 1 | — | — | 1024 / 262144 | — |
| 1a | finite | independent | optimum | 512 | 4 | — | — | 2560 / 262144 | — |
| 1a | finite | independent | optimum | 512 | 16 | — | — | 8704 / 262144 | — |
| 1a | finite | independent | prescribed_error | 128 | 1 | — | 0.000 | 256 / 16384 | — |
| 1a | finite | independent | prescribed_error | 128 | 4 | — | 0.000 | 640 / 16384 | — |
| 1a | finite | independent | prescribed_error | 128 | 16 | — | 0.000 | 2176 / 16384 | — |
| 1a | finite | independent | prescribed_error | 32 | 1 | — | 0.000 | 64 / 1024 | — |
| 1a | finite | independent | prescribed_error | 32 | 4 | — | 0.000 | 160 / 1024 | — |
| 1a | finite | independent | prescribed_error | 32 | 16 | — | 0.000 | 544 / 1024 | — |
| 1a | finite | independent | prescribed_error | 512 | 1 | — | — | 1024 / 262144 | — |
| 1a | finite | independent | prescribed_error | 512 | 4 | — | — | 2560 / 262144 | — |
| 1a | finite | independent | prescribed_error | 512 | 16 | — | — | 8704 / 262144 | — |
| 1a | finite | independent | zero | 128 | 1 | — | — | 256 / 16384 | — |
| 1a | finite | independent | zero | 128 | 4 | — | — | 640 / 16384 | — |
| 1a | finite | independent | zero | 128 | 16 | — | — | 2176 / 16384 | — |
| 1a | finite | independent | zero | 32 | 1 | — | — | 64 / 1024 | — |
| 1a | finite | independent | zero | 32 | 4 | — | — | 160 / 1024 | — |
| 1a | finite | independent | zero | 32 | 16 | — | — | 544 / 1024 | — |
| 1a | finite | independent | zero | 512 | 1 | — | — | 1024 / 262144 | — |
| 1a | finite | independent | zero | 512 | 4 | — | — | 2560 / 262144 | — |
| 1a | finite | independent | zero | 512 | 16 | — | — | 8704 / 262144 | — |
| 1a | finite | strong | fixed_random | 128 | 1 | 0.994 | 0.901 | 256 / 16384 | — |
| 1a | finite | strong | fixed_random | 128 | 4 | 0.997 | 0.998 | 640 / 16384 | — |
| 1a | finite | strong | fixed_random | 128 | 16 | 1.006 | 0.965 | 2176 / 16384 | — |
| 1a | finite | strong | fixed_random | 32 | 1 | 1.016 | 0.954 | 64 / 1024 | — |
| 1a | finite | strong | fixed_random | 32 | 4 | 1.003 | 1.029 | 160 / 1024 | — |
| 1a | finite | strong | fixed_random | 32 | 16 | 0.999 | 1.016 | 544 / 1024 | — |
| 1a | finite | strong | fixed_random | 512 | 1 | 1.006 | 1.052 | 1024 / 262144 | — |
| 1a | finite | strong | fixed_random | 512 | 4 | 1.013 | 1.032 | 2560 / 262144 | — |
| 1a | finite | strong | fixed_random | 512 | 16 | 1.000 | 1.011 | 8704 / 262144 | — |
| 1a | finite | strong | optimum | 128 | 1 | 1.003 | 1.008 | 256 / 16384 | — |
| 1a | finite | strong | optimum | 128 | 4 | 1.001 | 1.017 | 640 / 16384 | — |
| 1a | finite | strong | optimum | 128 | 16 | 1.001 | 0.971 | 2176 / 16384 | — |
| 1a | finite | strong | optimum | 32 | 1 | 1.001 | 0.986 | 64 / 1024 | — |
| 1a | finite | strong | optimum | 32 | 4 | 1.002 | 1.000 | 160 / 1024 | — |
| 1a | finite | strong | optimum | 32 | 16 | 1.000 | 1.015 | 544 / 1024 | — |
| 1a | finite | strong | optimum | 512 | 1 | 0.993 | 1.081 | 1024 / 262144 | — |
| 1a | finite | strong | optimum | 512 | 4 | 1.014 | 0.931 | 2560 / 262144 | — |
| 1a | finite | strong | optimum | 512 | 16 | 0.992 | 0.981 | 8704 / 262144 | — |
| 1a | finite | strong | prescribed_error | 128 | 1 | 1.001 | 1.037 | 256 / 16384 | — |
| 1a | finite | strong | prescribed_error | 128 | 4 | 1.001 | 0.977 | 640 / 16384 | — |
| 1a | finite | strong | prescribed_error | 128 | 16 | 0.994 | 1.023 | 2176 / 16384 | — |
| 1a | finite | strong | prescribed_error | 32 | 1 | 1.010 | 1.058 | 64 / 1024 | — |
| 1a | finite | strong | prescribed_error | 32 | 4 | 0.992 | 1.072 | 160 / 1024 | — |
| 1a | finite | strong | prescribed_error | 32 | 16 | 1.011 | 1.022 | 544 / 1024 | — |
| 1a | finite | strong | prescribed_error | 512 | 1 | 1.006 | 0.939 | 1024 / 262144 | — |
| 1a | finite | strong | prescribed_error | 512 | 4 | 0.990 | 1.002 | 2560 / 262144 | — |
| 1a | finite | strong | prescribed_error | 512 | 16 | 1.000 | 0.991 | 8704 / 262144 | — |
| 1a | finite | strong | zero | 128 | 1 | — | — | 256 / 16384 | — |
| 1a | finite | strong | zero | 128 | 4 | — | — | 640 / 16384 | — |
| 1a | finite | strong | zero | 128 | 16 | — | — | 2176 / 16384 | — |
| 1a | finite | strong | zero | 32 | 1 | — | — | 64 / 1024 | — |
| 1a | finite | strong | zero | 32 | 4 | — | — | 160 / 1024 | — |
| 1a | finite | strong | zero | 32 | 16 | — | — | 544 / 1024 | — |
| 1a | finite | strong | zero | 512 | 1 | — | — | 1024 / 262144 | — |
| 1a | finite | strong | zero | 512 | 4 | — | — | 2560 / 262144 | — |
| 1a | finite | strong | zero | 512 | 16 | — | — | 8704 / 262144 | — |
| 1a | finite | weak | fixed_random | 128 | 1 | 1.003 | 1.075 | 256 / 16384 | — |
| 1a | finite | weak | fixed_random | 128 | 4 | 0.989 | 0.987 | 640 / 16384 | — |
| 1a | finite | weak | fixed_random | 128 | 16 | 0.981 | 1.013 | 2176 / 16384 | — |
| 1a | finite | weak | fixed_random | 32 | 1 | 1.010 | 1.041 | 64 / 1024 | — |
| 1a | finite | weak | fixed_random | 32 | 4 | 0.993 | 1.042 | 160 / 1024 | — |
| 1a | finite | weak | fixed_random | 32 | 16 | 1.007 | 1.002 | 544 / 1024 | — |
| 1a | finite | weak | fixed_random | 512 | 1 | 1.003 | 0.994 | 1024 / 262144 | — |
| 1a | finite | weak | fixed_random | 512 | 4 | 0.998 | 0.997 | 2560 / 262144 | — |
| 1a | finite | weak | fixed_random | 512 | 16 | 0.993 | 0.976 | 8704 / 262144 | — |
| 1a | finite | weak | optimum | 128 | 1 | 1.012 | 0.974 | 256 / 16384 | — |
| 1a | finite | weak | optimum | 128 | 4 | 1.004 | 1.035 | 640 / 16384 | — |
| 1a | finite | weak | optimum | 128 | 16 | 0.998 | 0.993 | 2176 / 16384 | — |
| 1a | finite | weak | optimum | 32 | 1 | 1.003 | 0.973 | 64 / 1024 | — |
| 1a | finite | weak | optimum | 32 | 4 | 1.013 | 1.055 | 160 / 1024 | — |
| 1a | finite | weak | optimum | 32 | 16 | 1.011 | 0.980 | 544 / 1024 | — |
| 1a | finite | weak | optimum | 512 | 1 | 0.987 | 0.870 | 1024 / 262144 | — |
| 1a | finite | weak | optimum | 512 | 4 | 0.997 | 0.981 | 2560 / 262144 | — |
| 1a | finite | weak | optimum | 512 | 16 | 1.003 | 0.971 | 8704 / 262144 | — |
| 1a | finite | weak | prescribed_error | 128 | 1 | 1.001 | 1.029 | 256 / 16384 | — |
| 1a | finite | weak | prescribed_error | 128 | 4 | 0.998 | 1.004 | 640 / 16384 | — |
| 1a | finite | weak | prescribed_error | 128 | 16 | 0.997 | 0.989 | 2176 / 16384 | — |
| 1a | finite | weak | prescribed_error | 32 | 1 | 1.002 | 0.986 | 64 / 1024 | — |
| 1a | finite | weak | prescribed_error | 32 | 4 | 0.996 | 1.005 | 160 / 1024 | — |
| 1a | finite | weak | prescribed_error | 32 | 16 | 1.005 | 1.011 | 544 / 1024 | — |
| 1a | finite | weak | prescribed_error | 512 | 1 | 1.021 | 1.042 | 1024 / 262144 | — |
| 1a | finite | weak | prescribed_error | 512 | 4 | 1.005 | 0.930 | 2560 / 262144 | — |
| 1a | finite | weak | prescribed_error | 512 | 16 | 0.990 | 1.053 | 8704 / 262144 | — |
| 1a | finite | weak | zero | 128 | 1 | — | — | 256 / 16384 | — |
| 1a | finite | weak | zero | 128 | 4 | — | — | 640 / 16384 | — |
| 1a | finite | weak | zero | 128 | 16 | — | — | 2176 / 16384 | — |
| 1a | finite | weak | zero | 32 | 1 | — | — | 64 / 1024 | — |
| 1a | finite | weak | zero | 32 | 4 | — | — | 160 / 1024 | — |
| 1a | finite | weak | zero | 32 | 16 | — | — | 544 / 1024 | — |
| 1a | finite | weak | zero | 512 | 1 | — | — | 1024 / 262144 | — |
| 1a | finite | weak | zero | 512 | 4 | — | — | 2560 / 262144 | — |
| 1a | finite | weak | zero | 512 | 16 | — | — | 8704 / 262144 | — |
| 2 | gaussian | independent | optimum | 128 | 1 | — | — | 256 / 16384 | 0.0116 / 0.0040 |
| 2 | gaussian | independent | optimum | 128 | 4 | — | — | 640 / 16384 | 0.0116 / 0.0040 |
| 2 | gaussian | independent | optimum | 128 | 16 | — | — | 2176 / 16384 | 0.0116 / 0.0040 |
| 2 | gaussian | independent | optimum | 32 | 1 | — | — | 64 / 1024 | 0.0004 / 0.0031 |
| 2 | gaussian | independent | optimum | 32 | 4 | — | — | 160 / 1024 | 0.0004 / 0.0031 |
| 2 | gaussian | independent | optimum | 32 | 16 | — | — | 544 / 1024 | 0.0004 / 0.0031 |
| 2 | gaussian | independent | optimum | 512 | 1 | — | — | 1024 / 262144 | 0.1052 / 0.0070 |
| 2 | gaussian | independent | optimum | 512 | 4 | — | — | 2560 / 262144 | 0.1052 / 0.0070 |
| 2 | gaussian | independent | optimum | 512 | 16 | — | — | 8704 / 262144 | 0.1052 / 0.0070 |
| 2 | gaussian | independent | prescribed_error | 128 | 1 | — | 0.000 | 256 / 16384 | 0.0087 / 0.0035 |
| 2 | gaussian | independent | prescribed_error | 128 | 4 | — | 0.000 | 640 / 16384 | 0.0087 / 0.0035 |
| 2 | gaussian | independent | prescribed_error | 128 | 16 | — | 0.000 | 2176 / 16384 | 0.0087 / 0.0035 |
| 2 | gaussian | independent | prescribed_error | 32 | 1 | — | 0.000 | 64 / 1024 | 0.0004 / 0.0030 |
| 2 | gaussian | independent | prescribed_error | 32 | 4 | — | 0.000 | 160 / 1024 | 0.0004 / 0.0030 |
| 2 | gaussian | independent | prescribed_error | 32 | 16 | — | 0.000 | 544 / 1024 | 0.0004 / 0.0030 |
| 2 | gaussian | independent | prescribed_error | 512 | 1 | — | — | 1024 / 262144 | 0.1404 / 0.0064 |
| 2 | gaussian | independent | prescribed_error | 512 | 4 | — | — | 2560 / 262144 | 0.1404 / 0.0064 |
| 2 | gaussian | independent | prescribed_error | 512 | 16 | — | — | 8704 / 262144 | 0.1404 / 0.0064 |
| 2 | gaussian | independent | zero | 128 | 1 | — | — | 256 / 16384 | 0.0077 / 0.0035 |
| 2 | gaussian | independent | zero | 128 | 4 | — | — | 640 / 16384 | 0.0077 / 0.0035 |
| 2 | gaussian | independent | zero | 128 | 16 | — | — | 2176 / 16384 | 0.0077 / 0.0035 |
| 2 | gaussian | independent | zero | 32 | 1 | — | — | 64 / 1024 | 0.0004 / 0.0029 |
| 2 | gaussian | independent | zero | 32 | 4 | — | — | 160 / 1024 | 0.0004 / 0.0029 |
| 2 | gaussian | independent | zero | 32 | 16 | — | — | 544 / 1024 | 0.0004 / 0.0029 |
| 2 | gaussian | independent | zero | 512 | 1 | — | — | 1024 / 262144 | 0.1044 / 0.0045 |
| 2 | gaussian | independent | zero | 512 | 4 | — | — | 2560 / 262144 | 0.1044 / 0.0045 |
| 2 | gaussian | independent | zero | 512 | 16 | — | — | 8704 / 262144 | 0.1044 / 0.0045 |
| 2 | gaussian | strong | optimum | 128 | 1 | 1.006 | 1.061 | 256 / 16384 | 0.0139 / 0.0044 |
| 2 | gaussian | strong | optimum | 128 | 4 | 0.989 | 1.016 | 640 / 16384 | 0.0139 / 0.0044 |
| 2 | gaussian | strong | optimum | 128 | 16 | 0.988 | 0.995 | 2176 / 16384 | 0.0139 / 0.0044 |
| 2 | gaussian | strong | optimum | 32 | 1 | 1.005 | 1.001 | 64 / 1024 | 0.0004 / 0.0031 |
| 2 | gaussian | strong | optimum | 32 | 4 | 1.008 | 1.018 | 160 / 1024 | 0.0004 / 0.0031 |
| 2 | gaussian | strong | optimum | 32 | 16 | 0.997 | 1.001 | 544 / 1024 | 0.0004 / 0.0031 |
| 2 | gaussian | strong | optimum | 512 | 1 | 1.001 | 0.937 | 1024 / 262144 | 0.1523 / 0.0063 |
| 2 | gaussian | strong | optimum | 512 | 4 | 0.996 | 1.060 | 2560 / 262144 | 0.1523 / 0.0063 |
| 2 | gaussian | strong | optimum | 512 | 16 | 1.006 | 1.004 | 8704 / 262144 | 0.1523 / 0.0063 |
| 2 | gaussian | strong | prescribed_error | 128 | 1 | 0.994 | 0.974 | 256 / 16384 | 0.0147 / 0.0049 |
| 2 | gaussian | strong | prescribed_error | 128 | 4 | 1.007 | 1.046 | 640 / 16384 | 0.0147 / 0.0049 |
| 2 | gaussian | strong | prescribed_error | 128 | 16 | 0.995 | 0.991 | 2176 / 16384 | 0.0147 / 0.0049 |
| 2 | gaussian | strong | prescribed_error | 32 | 1 | 1.001 | 1.030 | 64 / 1024 | 0.0006 / 0.0042 |
| 2 | gaussian | strong | prescribed_error | 32 | 4 | 0.996 | 1.016 | 160 / 1024 | 0.0006 / 0.0042 |
| 2 | gaussian | strong | prescribed_error | 32 | 16 | 0.997 | 1.025 | 544 / 1024 | 0.0006 / 0.0042 |
| 2 | gaussian | strong | prescribed_error | 512 | 1 | 0.992 | 1.031 | 1024 / 262144 | 0.1592 / 0.0071 |
| 2 | gaussian | strong | prescribed_error | 512 | 4 | 0.989 | 0.990 | 2560 / 262144 | 0.1592 / 0.0071 |
| 2 | gaussian | strong | prescribed_error | 512 | 16 | 1.010 | 1.027 | 8704 / 262144 | 0.1592 / 0.0071 |
| 2 | gaussian | strong | zero | 128 | 1 | — | — | 256 / 16384 | 0.0120 / 0.0039 |
| 2 | gaussian | strong | zero | 128 | 4 | — | — | 640 / 16384 | 0.0120 / 0.0039 |
| 2 | gaussian | strong | zero | 128 | 16 | — | — | 2176 / 16384 | 0.0120 / 0.0039 |
| 2 | gaussian | strong | zero | 32 | 1 | — | — | 64 / 1024 | 0.0004 / 0.0031 |
| 2 | gaussian | strong | zero | 32 | 4 | — | — | 160 / 1024 | 0.0004 / 0.0031 |
| 2 | gaussian | strong | zero | 32 | 16 | — | — | 544 / 1024 | 0.0004 / 0.0031 |
| 2 | gaussian | strong | zero | 512 | 1 | — | — | 1024 / 262144 | 0.1515 / 0.0067 |
| 2 | gaussian | strong | zero | 512 | 4 | — | — | 2560 / 262144 | 0.1515 / 0.0067 |
| 2 | gaussian | strong | zero | 512 | 16 | — | — | 8704 / 262144 | 0.1515 / 0.0067 |
| 2 | gaussian | weak | optimum | 128 | 1 | 1.000 | 1.041 | 256 / 16384 | 0.0104 / 0.0040 |
| 2 | gaussian | weak | optimum | 128 | 4 | 0.997 | 0.970 | 640 / 16384 | 0.0104 / 0.0040 |
| 2 | gaussian | weak | optimum | 128 | 16 | 0.996 | 1.013 | 2176 / 16384 | 0.0104 / 0.0040 |
| 2 | gaussian | weak | optimum | 32 | 1 | 0.999 | 0.879 | 64 / 1024 | 0.0004 / 0.0031 |
| 2 | gaussian | weak | optimum | 32 | 4 | 1.006 | 0.963 | 160 / 1024 | 0.0004 / 0.0031 |
| 2 | gaussian | weak | optimum | 32 | 16 | 1.002 | 0.999 | 544 / 1024 | 0.0004 / 0.0031 |
| 2 | gaussian | weak | optimum | 512 | 1 | 1.007 | 1.062 | 1024 / 262144 | 0.1379 / 0.0057 |
| 2 | gaussian | weak | optimum | 512 | 4 | 1.000 | 1.003 | 2560 / 262144 | 0.1379 / 0.0057 |
| 2 | gaussian | weak | optimum | 512 | 16 | 1.006 | 0.975 | 8704 / 262144 | 0.1379 / 0.0057 |
| 2 | gaussian | weak | prescribed_error | 128 | 1 | 1.004 | 0.971 | 256 / 16384 | 0.0098 / 0.0046 |
| 2 | gaussian | weak | prescribed_error | 128 | 4 | 0.998 | 1.009 | 640 / 16384 | 0.0098 / 0.0046 |
| 2 | gaussian | weak | prescribed_error | 128 | 16 | 0.999 | 1.019 | 2176 / 16384 | 0.0098 / 0.0046 |
| 2 | gaussian | weak | prescribed_error | 32 | 1 | 0.997 | 1.099 | 64 / 1024 | 0.0004 / 0.0029 |
| 2 | gaussian | weak | prescribed_error | 32 | 4 | 1.011 | 0.978 | 160 / 1024 | 0.0004 / 0.0029 |
| 2 | gaussian | weak | prescribed_error | 32 | 16 | 1.008 | 1.001 | 544 / 1024 | 0.0004 / 0.0029 |
| 2 | gaussian | weak | prescribed_error | 512 | 1 | 1.014 | 0.967 | 1024 / 262144 | 0.1494 / 0.0068 |
| 2 | gaussian | weak | prescribed_error | 512 | 4 | 1.011 | 1.004 | 2560 / 262144 | 0.1494 / 0.0068 |
| 2 | gaussian | weak | prescribed_error | 512 | 16 | 1.002 | 1.001 | 8704 / 262144 | 0.1494 / 0.0068 |
| 2 | gaussian | weak | zero | 128 | 1 | — | — | 256 / 16384 | 0.0090 / 0.0042 |
| 2 | gaussian | weak | zero | 128 | 4 | — | — | 640 / 16384 | 0.0090 / 0.0042 |
| 2 | gaussian | weak | zero | 128 | 16 | — | — | 2176 / 16384 | 0.0090 / 0.0042 |
| 2 | gaussian | weak | zero | 32 | 1 | — | — | 64 / 1024 | 0.0004 / 0.0030 |
| 2 | gaussian | weak | zero | 32 | 4 | — | — | 160 / 1024 | 0.0004 / 0.0030 |
| 2 | gaussian | weak | zero | 32 | 16 | — | — | 544 / 1024 | 0.0004 / 0.0030 |
| 2 | gaussian | weak | zero | 512 | 1 | — | — | 1024 / 262144 | 0.1514 / 0.0072 |
| 2 | gaussian | weak | zero | 512 | 4 | — | — | 2560 / 262144 | 0.1514 / 0.0072 |
| 2 | gaussian | weak | zero | 512 | 16 | — | — | 8704 / 262144 | 0.1514 / 0.0072 |
| 2 | xor_mixture | independent | optimum | 128 | 1 | — | 1.037 | 256 / 16384 | 0.0097 / 0.0066 |
| 2 | xor_mixture | independent | optimum | 128 | 4 | — | 0.953 | 640 / 16384 | 0.0097 / 0.0066 |
| 2 | xor_mixture | independent | optimum | 128 | 16 | — | 1.008 | 2176 / 16384 | 0.0097 / 0.0066 |
| 2 | xor_mixture | independent | optimum | 32 | 1 | — | 0.991 | 64 / 1024 | 0.0010 / 0.0061 |
| 2 | xor_mixture | independent | optimum | 32 | 4 | — | 0.973 | 160 / 1024 | 0.0010 / 0.0061 |
| 2 | xor_mixture | independent | optimum | 32 | 16 | — | 0.996 | 544 / 1024 | 0.0010 / 0.0061 |
| 2 | xor_mixture | independent | optimum | 512 | 1 | — | 1.011 | 1024 / 262144 | 0.2435 / 0.0093 |
| 2 | xor_mixture | independent | optimum | 512 | 4 | — | 1.003 | 2560 / 262144 | 0.2435 / 0.0093 |
| 2 | xor_mixture | independent | optimum | 512 | 16 | — | 0.996 | 8704 / 262144 | 0.2435 / 0.0093 |
| 2 | xor_mixture | independent | prescribed_error | 128 | 1 | — | 1.067 | 256 / 16384 | 0.0113 / 0.0080 |
| 2 | xor_mixture | independent | prescribed_error | 128 | 4 | — | 1.092 | 640 / 16384 | 0.0113 / 0.0080 |
| 2 | xor_mixture | independent | prescribed_error | 128 | 16 | — | 0.970 | 2176 / 16384 | 0.0113 / 0.0080 |
| 2 | xor_mixture | independent | prescribed_error | 32 | 1 | — | 1.168 | 64 / 1024 | 0.0011 / 0.0068 |
| 2 | xor_mixture | independent | prescribed_error | 32 | 4 | — | 0.958 | 160 / 1024 | 0.0011 / 0.0068 |
| 2 | xor_mixture | independent | prescribed_error | 32 | 16 | — | 1.007 | 544 / 1024 | 0.0011 / 0.0068 |
| 2 | xor_mixture | independent | prescribed_error | 512 | 1 | — | 0.976 | 1024 / 262144 | 0.2100 / 0.0093 |
| 2 | xor_mixture | independent | prescribed_error | 512 | 4 | — | 0.999 | 2560 / 262144 | 0.2100 / 0.0093 |
| 2 | xor_mixture | independent | prescribed_error | 512 | 16 | — | 1.015 | 8704 / 262144 | 0.2100 / 0.0093 |
| 2 | xor_mixture | independent | zero | 128 | 1 | — | — | 256 / 16384 | 0.0107 / 0.0073 |
| 2 | xor_mixture | independent | zero | 128 | 4 | — | — | 640 / 16384 | 0.0107 / 0.0073 |
| 2 | xor_mixture | independent | zero | 128 | 16 | — | — | 2176 / 16384 | 0.0107 / 0.0073 |
| 2 | xor_mixture | independent | zero | 32 | 1 | — | — | 64 / 1024 | 0.0010 / 0.0064 |
| 2 | xor_mixture | independent | zero | 32 | 4 | — | — | 160 / 1024 | 0.0010 / 0.0064 |
| 2 | xor_mixture | independent | zero | 32 | 16 | — | — | 544 / 1024 | 0.0010 / 0.0064 |
| 2 | xor_mixture | independent | zero | 512 | 1 | — | — | 1024 / 262144 | 0.2139 / 0.0089 |
| 2 | xor_mixture | independent | zero | 512 | 4 | — | — | 2560 / 262144 | 0.2139 / 0.0089 |
| 2 | xor_mixture | independent | zero | 512 | 16 | — | — | 8704 / 262144 | 0.2139 / 0.0089 |
| 2 | xor_mixture | strong | optimum | 128 | 1 | 1.007 | 1.011 | 256 / 16384 | 0.0083 / 0.0064 |
| 2 | xor_mixture | strong | optimum | 128 | 4 | 1.000 | 0.997 | 640 / 16384 | 0.0083 / 0.0064 |
| 2 | xor_mixture | strong | optimum | 128 | 16 | 0.997 | 1.020 | 2176 / 16384 | 0.0083 / 0.0064 |
| 2 | xor_mixture | strong | optimum | 32 | 1 | 0.993 | 1.060 | 64 / 1024 | 0.0009 / 0.0055 |
| 2 | xor_mixture | strong | optimum | 32 | 4 | 0.991 | 0.979 | 160 / 1024 | 0.0009 / 0.0055 |
| 2 | xor_mixture | strong | optimum | 32 | 16 | 0.999 | 0.984 | 544 / 1024 | 0.0009 / 0.0055 |
| 2 | xor_mixture | strong | optimum | 512 | 1 | 1.002 | 0.982 | 1024 / 262144 | 0.1575 / 0.0076 |
| 2 | xor_mixture | strong | optimum | 512 | 4 | 1.000 | 0.990 | 2560 / 262144 | 0.1575 / 0.0076 |
| 2 | xor_mixture | strong | optimum | 512 | 16 | 1.000 | 1.030 | 8704 / 262144 | 0.1575 / 0.0076 |
| 2 | xor_mixture | strong | prescribed_error | 128 | 1 | 0.996 | 1.048 | 256 / 16384 | 0.0058 / 0.0064 |
| 2 | xor_mixture | strong | prescribed_error | 128 | 4 | 0.996 | 1.002 | 640 / 16384 | 0.0058 / 0.0064 |
| 2 | xor_mixture | strong | prescribed_error | 128 | 16 | 0.988 | 0.978 | 2176 / 16384 | 0.0058 / 0.0064 |
| 2 | xor_mixture | strong | prescribed_error | 32 | 1 | 1.011 | 1.092 | 64 / 1024 | 0.0010 / 0.0069 |
| 2 | xor_mixture | strong | prescribed_error | 32 | 4 | 0.998 | 1.001 | 160 / 1024 | 0.0010 / 0.0069 |
| 2 | xor_mixture | strong | prescribed_error | 32 | 16 | 0.998 | 1.002 | 544 / 1024 | 0.0010 / 0.0069 |
| 2 | xor_mixture | strong | prescribed_error | 512 | 1 | 0.996 | 0.934 | 1024 / 262144 | 0.1580 / 0.0088 |
| 2 | xor_mixture | strong | prescribed_error | 512 | 4 | 0.991 | 1.038 | 2560 / 262144 | 0.1580 / 0.0088 |
| 2 | xor_mixture | strong | prescribed_error | 512 | 16 | 0.998 | 1.014 | 8704 / 262144 | 0.1580 / 0.0088 |
| 2 | xor_mixture | strong | zero | 128 | 1 | — | — | 256 / 16384 | 0.0057 / 0.0062 |
| 2 | xor_mixture | strong | zero | 128 | 4 | — | — | 640 / 16384 | 0.0057 / 0.0062 |
| 2 | xor_mixture | strong | zero | 128 | 16 | — | — | 2176 / 16384 | 0.0057 / 0.0062 |
| 2 | xor_mixture | strong | zero | 32 | 1 | — | — | 64 / 1024 | 0.0011 / 0.0064 |
| 2 | xor_mixture | strong | zero | 32 | 4 | — | — | 160 / 1024 | 0.0011 / 0.0064 |
| 2 | xor_mixture | strong | zero | 32 | 16 | — | — | 544 / 1024 | 0.0011 / 0.0064 |
| 2 | xor_mixture | strong | zero | 512 | 1 | — | — | 1024 / 262144 | 0.1621 / 0.0079 |
| 2 | xor_mixture | strong | zero | 512 | 4 | — | — | 2560 / 262144 | 0.1621 / 0.0079 |
| 2 | xor_mixture | strong | zero | 512 | 16 | — | — | 8704 / 262144 | 0.1621 / 0.0079 |
| 2 | xor_mixture | weak | optimum | 128 | 1 | 1.003 | 1.026 | 256 / 16384 | 0.0117 / 0.0073 |
| 2 | xor_mixture | weak | optimum | 128 | 4 | 0.990 | 0.993 | 640 / 16384 | 0.0117 / 0.0073 |
| 2 | xor_mixture | weak | optimum | 128 | 16 | 0.997 | 0.973 | 2176 / 16384 | 0.0117 / 0.0073 |
| 2 | xor_mixture | weak | optimum | 32 | 1 | 1.003 | 1.055 | 64 / 1024 | 0.0010 / 0.0061 |
| 2 | xor_mixture | weak | optimum | 32 | 4 | 1.002 | 0.994 | 160 / 1024 | 0.0010 / 0.0061 |
| 2 | xor_mixture | weak | optimum | 32 | 16 | 0.993 | 0.991 | 544 / 1024 | 0.0010 / 0.0061 |
| 2 | xor_mixture | weak | optimum | 512 | 1 | 0.997 | 0.974 | 1024 / 262144 | 0.1856 / 0.0082 |
| 2 | xor_mixture | weak | optimum | 512 | 4 | 0.993 | 0.985 | 2560 / 262144 | 0.1856 / 0.0082 |
| 2 | xor_mixture | weak | optimum | 512 | 16 | 1.006 | 1.011 | 8704 / 262144 | 0.1856 / 0.0082 |
| 2 | xor_mixture | weak | prescribed_error | 128 | 1 | 0.997 | 1.015 | 256 / 16384 | 0.0055 / 0.0063 |
| 2 | xor_mixture | weak | prescribed_error | 128 | 4 | 1.010 | 0.954 | 640 / 16384 | 0.0055 / 0.0063 |
| 2 | xor_mixture | weak | prescribed_error | 128 | 16 | 1.006 | 0.992 | 2176 / 16384 | 0.0055 / 0.0063 |
| 2 | xor_mixture | weak | prescribed_error | 32 | 1 | 0.995 | 1.009 | 64 / 1024 | 0.0011 / 0.0070 |
| 2 | xor_mixture | weak | prescribed_error | 32 | 4 | 1.005 | 1.004 | 160 / 1024 | 0.0011 / 0.0070 |
| 2 | xor_mixture | weak | prescribed_error | 32 | 16 | 0.993 | 1.014 | 544 / 1024 | 0.0011 / 0.0070 |
| 2 | xor_mixture | weak | prescribed_error | 512 | 1 | 1.005 | 0.974 | 1024 / 262144 | 0.1298 / 0.0080 |
| 2 | xor_mixture | weak | prescribed_error | 512 | 4 | 1.007 | 0.958 | 2560 / 262144 | 0.1298 / 0.0080 |
| 2 | xor_mixture | weak | prescribed_error | 512 | 16 | 1.005 | 1.025 | 8704 / 262144 | 0.1298 / 0.0080 |
| 2 | xor_mixture | weak | zero | 128 | 1 | — | — | 256 / 16384 | 0.0111 / 0.0073 |
| 2 | xor_mixture | weak | zero | 128 | 4 | — | — | 640 / 16384 | 0.0111 / 0.0073 |
| 2 | xor_mixture | weak | zero | 128 | 16 | — | — | 2176 / 16384 | 0.0111 / 0.0073 |
| 2 | xor_mixture | weak | zero | 32 | 1 | — | — | 64 / 1024 | 0.0011 / 0.0065 |
| 2 | xor_mixture | weak | zero | 32 | 4 | — | — | 160 / 1024 | 0.0011 / 0.0065 |
| 2 | xor_mixture | weak | zero | 32 | 16 | — | — | 544 / 1024 | 0.0011 / 0.0065 |
| 2 | xor_mixture | weak | zero | 512 | 1 | — | — | 1024 / 262144 | 0.1939 / 0.0088 |
| 2 | xor_mixture | weak | zero | 512 | 4 | — | — | 2560 / 262144 | 0.1939 / 0.0088 |
| 2 | xor_mixture | weak | zero | 512 | 16 | — | — | 8704 / 262144 | 0.1939 / 0.0088 |
