# P68 — final evaluation on the OFFICIAL CIFAR-10 TEST partition (10 000 images), evaluated once — 2026-09-28T13:43:28Z

Head and kNN bank: the 45 000-image fit split, frozen pilot probe hyper-parameters, clean transform; the checkpoint of each run is the one every
decision was made on.  Selection-split numbers (the development metric) are shown next to the test numbers; single-seed cells are marked in the label.

## Per cell (mean ± sd over seeds; n)

| cell | n | test linear % | test kNN % | selection linear % | selection kNN % | runs |
|---|---|---|---|---|---|---|
| VCS|1x|4v/B128/100ep | 3 | 82.98 ± 0.45 | 78.57 ± 0.09 | 83.19 ± 0.40 | 78.43 ± 0.11 | P39_vcs_a5_views4_b128_100ep_seed0, P39_vcs_a5_views4_b128_100ep_seed1, P39_vcs_a5_views4_b128_100ep_seed2 |
| VCS|2x|4v/B256/200ep | 3 | 84.52 ± 0.18 | 81.31 ± 0.29 | 84.54 ± 0.16 | 81.40 ± 0.22 | P35_vcs_a5_views4_seed0, P35_vcs_a5_views4_seed1, P35_vcs_a5_views4_seed2 |
| VCS|4x|8v/200ep | 3 | 85.87 ± 0.07 | 83.69 ± 0.15 | 85.95 ± 0.35 | 83.47 ± 0.38 | P37_vcs_a5_views8_200ep_seed0, P37_vcs_a5_views8_200ep_seed1, P37_vcs_a5_views8_200ep_seed2 |
| VCS|4x|2v/800ep | 3 | 85.28 ± 0.40 | 82.40 ± 0.09 | 85.30 ± 0.21 | 82.33 ± 0.25 | P35_vcs_a5_800ep_seed0, P35_vcs_a5_800ep_seed1, P35_vcs_a5_800ep_seed2 |
| VCS|4x|4v/400ep(single) | 1 | 85.63 | 83.95 | 85.90 | 83.58 | P35_vcs_a5_views4_400ep_seed0 |
| VCS|4x|16v/200ep(single) | 1 | 86.08 | 84.99 | 86.74 | 84.48 | P43_vcs_a5_views16_200ep_seed0 |
| VCS|8x|4v/800ep(frozen-best) | 3 | 86.65 ± 0.26 | 85.30 ± 0.10 | 87.01 ± 0.53 | 85.46 ± 0.12 | P35_vcs_a5_views4_800ep_seed0, P35_vcs_a5_views4_800ep_seed1, P35_vcs_a5_views4_800ep_seed2 |
| VCS|8x|8v/400ep(single) | 1 | 86.86 | 85.83 | 86.76 | 84.94 | P43_vcs_a5_views8_400ep_seed0 |
| VCS|8x|4v/B128/800ep(single) | 1 | 87.09 | 84.92 | 86.70 | 84.54 | P43_vcs_a5_views4_b128_800ep_seed0 |
| VCS|8x|4v/800ep+strong-aug(single) | 1 | 88.26 | 85.79 | 87.78 | 85.58 | P43_vcs_a5_views4_800ep_augstrong_seed0 |
| VCS|16x|4v/1600ep(single) | 1 | 87.60 | 86.43 | 87.50 | 85.84 | P43_vcs_a5_views4_1600ep_seed0 |
| VCS|16x|8v/800ep(single) | 1 | 87.42 | 86.58 | 87.14 | 86.32 | P43_vcs_a5_views8_800ep_seed0 |
| SimCLR|1x|frozen-P5-2v/200ep | 3 | 86.29 ± 0.12 | 84.46 ± 0.20 | 86.09 ± 0.38 | 83.98 ± 0.24 | P5_simclr_seed0, P5_simclr_seed1, P5_simclr_seed2 |
| VICReg|1x|frozen-P5-2v/200ep | 3 | 85.07 ± 0.27 | 82.27 ± 0.10 | 85.46 ± 0.26 | 81.92 ± 0.40 | P5_vicreg_seed0, P5_vicreg_seed1, P5_vicreg_seed2 |
| SimCLR|1x|tuned-4v/B128/100ep | 3 | 86.94 ± 0.12 | 84.85 ± 0.05 | 86.64 ± 0.33 | 84.45 ± 0.06 | P41_simclr_views4_b128_100ep_seed0, P41_simclr_views4_b128_100ep_seed1, P41_simclr_views4_b128_100ep_seed2 |
| SimCLR|2x|tuned-4v/200ep | 3 | 87.99 ± 0.07 | 86.87 ± 0.09 | 87.84 ± 0.42 | 86.52 ± 0.37 | P41_simclr_views4_seed0, P41_simclr_views4_seed1, P41_simclr_views4_seed2 |
| SimCLR|4x|tuned-2v/800ep | 3 | 88.21 ± 0.10 | 87.70 ± 0.51 | 88.35 ± 0.06 | 87.26 ± 0.41 | P41_simclr_800ep_seed0, P41_simclr_800ep_seed1, P41_simclr_800ep_seed2 |
| VICReg|1x|tuned-4v/B128/100ep | 3 | 87.00 ± 0.25 | 84.45 ± 0.17 | 87.12 ± 0.10 | 83.91 ± 0.17 | P41_vicreg_views4_b128_100ep_seed0, P41_vicreg_views4_b128_100ep_seed1, P41_vicreg_views4_b128_100ep_seed2 |
| VICReg|2x|tuned-4v/200ep | 3 | 87.11 ± 0.31 | 84.71 ± 0.17 | 86.73 ± 0.30 | 84.27 ± 0.35 | P41_vicreg_views4_seed0, P41_vicreg_views4_seed1, P41_vicreg_views4_seed2 |
| VICReg|4x|tuned-2v/800ep | 3 | 86.91 ± 0.15 | 84.94 ± 0.16 | 87.28 ± 0.62 | 84.53 ± 0.18 | P41_vicreg_800ep_seed0, P41_vicreg_800ep_seed1, P41_vicreg_800ep_seed2 |
| SimCLR|8x|tuned-4v/800ep | 3 | 88.13 ± 0.07 | 87.98 ± 0.53 | 88.32 ± 0.30 | 87.65 ± 0.40 | P41_simclr_views4_800ep_seed0, P41_simclr_views4_800ep_seed1, P41_simclr_views4_800ep_seed2 |
| VICReg|8x|tuned-4v/800ep | 3 | 86.87 ± 0.06 | 84.78 ± 0.30 | 87.11 ± 0.51 | 84.46 ± 0.08 | P41_vicreg_views4_800ep_seed0, P41_vicreg_views4_800ep_seed1, P41_vicreg_views4_800ep_seed2 |

## Per run

| run | checkpoint | cell | test linear % | test kNN % | selection linear % | selection kNN % | status |
|---|---|---|---|---|---|---|---|
| P39_vcs_a5_views4_b128_100ep_seed0 | epoch_100.pt | VCS|1x|4v/B128/100ep | 83.41 | 78.48 | 83.51999521255493 | 78.3 | evaluated |
| P39_vcs_a5_views4_b128_100ep_seed1 | epoch_100.pt | VCS|1x|4v/B128/100ep | 82.51 | 78.65 | 82.73999691009521 | 78.48 | evaluated |
| P39_vcs_a5_views4_b128_100ep_seed2 | epoch_100.pt | VCS|1x|4v/B128/100ep | 83.01 | 78.59 | 83.30000042915344 | 78.5 | evaluated |
| P35_vcs_a5_views4_seed0 | epoch_200.pt | VCS|2x|4v/B256/200ep | 84.37 | 81.55 | 84.49999690055847 | 81.4 | evaluated |
| P35_vcs_a5_views4_seed1 | epoch_200.pt | VCS|2x|4v/B256/200ep | 84.72 | 80.99 | 84.3999981880188 | 81.18 | evaluated |
| P35_vcs_a5_views4_seed2 | epoch_200.pt | VCS|2x|4v/B256/200ep | 84.47 | 81.40 | 84.71999764442444 | 81.62 | evaluated |
| P37_vcs_a5_views8_200ep_seed0 | epoch_200.pt | VCS|4x|8v/200ep | 85.93 | 83.81 | 86.28000020980835 | 83.66 | evaluated |
| P37_vcs_a5_views8_200ep_seed1 | epoch_200.pt | VCS|4x|8v/200ep | 85.89 | 83.53 | 85.57999730110168 | 83.04 | evaluated |
| P37_vcs_a5_views8_200ep_seed2 | epoch_200.pt | VCS|4x|8v/200ep | 85.80 | 83.74 | 85.97999811172485 | 83.72 | evaluated |
| P35_vcs_a5_800ep_seed0 | epoch_800.pt | VCS|4x|2v/800ep | 84.91 | 82.30 | 85.53999662399292 | 82.62 | evaluated |
| P35_vcs_a5_800ep_seed1 | epoch_800.pt | VCS|4x|2v/800ep | 85.71 | 82.47 | 85.17999649047852 | 82.2 | evaluated |
| P35_vcs_a5_800ep_seed2 | epoch_800.pt | VCS|4x|2v/800ep | 85.21 | 82.44 | 85.17999649047852 | 82.16 | evaluated |
| P35_vcs_a5_views4_400ep_seed0 | epoch_400.pt | VCS|4x|4v/400ep(single) | 85.63 | 83.95 | 85.89999675750732 | 83.58 | evaluated |
| P43_vcs_a5_views16_200ep_seed0 | epoch_200.pt | VCS|4x|16v/200ep(single) | 86.08 | 84.99 | 86.73999905586243 | 84.48 | evaluated |
| P35_vcs_a5_views4_800ep_seed0 | epoch_800.pt | VCS|8x|4v/800ep(frozen-best) | 86.95 | 85.18 | 86.41999959945679 | 85.6 | evaluated |
| P35_vcs_a5_views4_800ep_seed1 | epoch_800.pt | VCS|8x|4v/800ep(frozen-best) | 86.55 | 85.37 | 87.15999722480774 | 85.36 | evaluated |
| P35_vcs_a5_views4_800ep_seed2 | epoch_800.pt | VCS|8x|4v/800ep(frozen-best) | 86.45 | 85.35 | 87.43999600410461 | 85.42 | evaluated |
| P43_vcs_a5_views8_400ep_seed0 | epoch_400.pt | VCS|8x|8v/400ep(single) | 86.86 | 85.83 | 86.75999641418457 | 84.94 | evaluated |
| P43_vcs_a5_views4_b128_800ep_seed0 | epoch_800.pt | VCS|8x|4v/B128/800ep(single) | 87.09 | 84.92 | 86.69999837875366 | 84.54 | evaluated |
| P43_vcs_a5_views4_800ep_augstrong_seed0 | epoch_800.pt | VCS|8x|4v/800ep+strong-aug(single) | 88.26 | 85.79 | 87.77999877929688 | 85.58 | evaluated |
| P43_vcs_a5_views4_1600ep_seed0 | epoch_1600.pt | VCS|16x|4v/1600ep(single) | 87.60 | 86.43 | 87.5 | 85.84 | evaluated |
| P43_vcs_a5_views8_800ep_seed0 | epoch_800.pt | VCS|16x|8v/800ep(single) | 87.42 | 86.58 | 87.1399998664856 | 86.32 | evaluated |
| P5_simclr_seed0 | epoch_200.pt | SimCLR|1x|frozen-P5-2v/200ep | 86.24 | 84.47 | 86.43999695777893 | 84.02 | evaluated |
| P5_simclr_seed1 | epoch_200.pt | SimCLR|1x|frozen-P5-2v/200ep | 86.21 | 84.25 | 85.67999601364136 | 83.72 | evaluated |
| P5_simclr_seed2 | epoch_200.pt | SimCLR|1x|frozen-P5-2v/200ep | 86.43 | 84.65 | 86.13999485969543 | 84.2 | evaluated |
| P5_vicreg_seed0 | epoch_200.pt | VICReg|1x|frozen-P5-2v/200ep | 84.80 | 82.17 | 85.63999533653259 | 81.46 | evaluated |
| P5_vicreg_seed1 | epoch_200.pt | VICReg|1x|frozen-P5-2v/200ep | 85.34 | 82.36 | 85.15999913215637 | 82.18 | evaluated |
| P5_vicreg_seed2 | epoch_200.pt | VICReg|1x|frozen-P5-2v/200ep | 85.08 | 82.27 | 85.57999730110168 | 82.12 | evaluated |
| P41_simclr_views4_b128_100ep_seed0 | epoch_100.pt | SimCLR|1x|tuned-4v/B128/100ep | 86.88 | 84.79 | 86.41999959945679 | 84.5 | evaluated |
| P41_simclr_views4_b128_100ep_seed1 | epoch_100.pt | SimCLR|1x|tuned-4v/B128/100ep | 87.07 | 84.88 | 87.0199978351593 | 84.38 | evaluated |
| P41_simclr_views4_b128_100ep_seed2 | epoch_100.pt | SimCLR|1x|tuned-4v/B128/100ep | 86.86 | 84.88 | 86.4799976348877 | 84.46 | evaluated |
| P41_simclr_views4_seed0 | epoch_200.pt | SimCLR|2x|tuned-4v/200ep | 87.92 | 86.77 | 87.47999668121338 | 86.12 | evaluated |
| P41_simclr_views4_seed1 | epoch_200.pt | SimCLR|2x|tuned-4v/200ep | 88.06 | 86.89 | 88.29999566078186 | 86.84 | evaluated |
| P41_simclr_views4_seed2 | epoch_200.pt | SimCLR|2x|tuned-4v/200ep | 88.00 | 86.94 | 87.73999810218811 | 86.6 | evaluated |
| P41_simclr_800ep_seed0 | epoch_800.pt | SimCLR|4x|tuned-2v/800ep | 88.28 | 87.64 | 88.37999701499939 | 87.12 | evaluated |
| P41_simclr_800ep_seed1 | epoch_800.pt | SimCLR|4x|tuned-2v/800ep | 88.10 | 87.23 | 88.27999830245972 | 86.94 | evaluated |
| P41_simclr_800ep_seed2 | epoch_800.pt | SimCLR|4x|tuned-2v/800ep | 88.25 | 88.24 | 88.37999701499939 | 87.72 | evaluated |
| P41_vicreg_views4_b128_100ep_seed0 | epoch_100.pt | VICReg|1x|tuned-4v/B128/100ep | 86.90 | 84.33 | 87.23999857902527 | 83.72 | evaluated |
| P41_vicreg_views4_b128_100ep_seed1 | epoch_100.pt | VICReg|1x|tuned-4v/B128/100ep | 86.82 | 84.38 | 87.05999851226807 | 83.96 | evaluated |
| P41_vicreg_views4_b128_100ep_seed2 | epoch_100.pt | VICReg|1x|tuned-4v/B128/100ep | 87.28 | 84.65 | 87.05999851226807 | 84.06 | evaluated |
| P41_vicreg_views4_seed0 | epoch_200.pt | VICReg|2x|tuned-4v/200ep | 86.76 | 84.57 | 86.64000034332275 | 84.14 | evaluated |
| P41_vicreg_views4_seed1 | epoch_200.pt | VICReg|2x|tuned-4v/200ep | 87.34 | 84.66 | 87.05999851226807 | 84.0 | evaluated |
| P41_vicreg_views4_seed2 | epoch_200.pt | VICReg|2x|tuned-4v/200ep | 87.22 | 84.90 | 86.4799976348877 | 84.66 | evaluated |
| P41_vicreg_800ep_seed0 | epoch_800.pt | VICReg|4x|tuned-2v/800ep | 86.77 | 85.01 | 86.69999837875366 | 84.36 | evaluated |
| P41_vicreg_800ep_seed1 | epoch_800.pt | VICReg|4x|tuned-2v/800ep | 86.88 | 84.76 | 87.1999979019165 | 84.72 | evaluated |
| P41_vicreg_800ep_seed2 | epoch_800.pt | VICReg|4x|tuned-2v/800ep | 87.07 | 85.06 | 87.93999552726746 | 84.52 | evaluated |
| P41_simclr_views4_800ep_seed0 | epoch_800.pt | SimCLR|8x|tuned-4v/800ep | 88.08 | 87.42 | 88.19999694824219 | 87.2 | evaluated |
| P41_simclr_views4_800ep_seed1 | epoch_800.pt | SimCLR|8x|tuned-4v/800ep | 88.11 | 88.04 | 88.09999823570251 | 87.76 | evaluated |
| P41_simclr_views4_800ep_seed2 | epoch_800.pt | SimCLR|8x|tuned-4v/800ep | 88.21 | 88.48 | 88.65999579429626 | 87.98 | evaluated |
| P41_vicreg_views4_800ep_seed0 | epoch_800.pt | VICReg|8x|tuned-4v/800ep | 86.80 | 84.72 | 87.5 | 84.38 | evaluated |
| P41_vicreg_views4_800ep_seed1 | epoch_800.pt | VICReg|8x|tuned-4v/800ep | 86.91 | 85.11 | 86.5399956703186 | 84.46 | evaluated |
| P41_vicreg_views4_800ep_seed2 | epoch_800.pt | VICReg|8x|tuned-4v/800ep | 86.91 | 84.51 | 87.29999661445618 | 84.54 | evaluated |
