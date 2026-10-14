# P68 — final evaluation on the OFFICIAL CIFAR-10 TEST partition (10 000 images), evaluated once — 2026-10-14T16:11:38Z

Head and kNN bank: the 45 000-image fit split, frozen pilot probe hyper-parameters, clean transform; the checkpoint of each run is the one every
decision was made on.  Selection-split numbers (the development metric) are shown next to the test numbers; single-seed cells are marked in the label.

## Per cell (mean ± sd over seeds; n)

| cell | n | test linear % | test kNN % | selection linear % | selection kNN % | runs |
|---|---|---|---|---|---|---|
| A-P3|8x|4v/800ep(final) | 5 | 88.83 ± 0.10 | 87.75 ± 0.23 | 89.02 ± 0.04 | 87.41 ± 0.08 | P107_AP3_views4_800ep_seed0, P107_AP3_views4_800ep_seed1, P107_AP3_views4_800ep_seed2, P107_AP3_views4_800ep_seed3, P107_AP3_views4_800ep_seed4 |
| SimCLR|8x|4v/800ep | 5 | 88.20 ± 0.13 | 88.07 ± 0.40 | 88.31 ± 0.21 | 87.86 ± 0.43 | P41_simclr_views4_800ep_seed0, P41_simclr_views4_800ep_seed1, P41_simclr_views4_800ep_seed2, P111_simclr_views4_800ep_seed3, P111_simclr_views4_800ep_seed4 |
| JS-AP3|8x|4v/800ep(matched) | 3 | 88.92 ± 0.29 | 88.04 ± 0.06 | 88.81 ± 0.45 | 87.67 ± 0.27 | P114_JSAP3_views4_800ep_seed0, P114_JSAP3_views4_800ep_seed1, P114_JSAP3_views4_800ep_seed2 |

## Per run

| run | checkpoint | cell | test linear % | test kNN % | selection linear % | selection kNN % | status |
|---|---|---|---|---|---|---|---|
| P107_AP3_views4_800ep_seed0 | epoch_800.pt | A-P3|8x|4v/800ep(final) | 88.84 | 87.82 | 89.05999660491943 | 87.3 | evaluated |
| P107_AP3_views4_800ep_seed1 | epoch_800.pt | A-P3|8x|4v/800ep(final) | 88.97 | 87.60 | 88.95999789237976 | 87.48 | evaluated |
| P107_AP3_views4_800ep_seed2 | epoch_800.pt | A-P3|8x|4v/800ep(final) | 88.75 | 87.54 | 89.05999660491943 | 87.36 | evaluated |
| P107_AP3_views4_800ep_seed3 | epoch_800.pt | A-P3|8x|4v/800ep(final) | 88.89 | 88.12 | 88.99999856948853 | 87.46 | evaluated |
| P107_AP3_views4_800ep_seed4 | epoch_800.pt | A-P3|8x|4v/800ep(final) | 88.71 | 87.68 | 89.03999924659729 | 87.44 | evaluated |
| P41_simclr_views4_800ep_seed0 | epoch_800.pt | SimCLR|8x|4v/800ep | 88.08 | 87.42 | 88.19999694824219 | 87.2 | read |
| P41_simclr_views4_800ep_seed1 | epoch_800.pt | SimCLR|8x|4v/800ep | 88.11 | 88.04 | 88.09999823570251 | 87.76 | read |
| P41_simclr_views4_800ep_seed2 | epoch_800.pt | SimCLR|8x|4v/800ep | 88.21 | 88.48 | 88.65999579429626 | 87.98 | read |
| P111_simclr_views4_800ep_seed3 | epoch_800.pt | SimCLR|8x|4v/800ep | 88.21 | 88.22 | 88.27999830245972 | 88.36 | evaluated |
| P111_simclr_views4_800ep_seed4 | epoch_800.pt | SimCLR|8x|4v/800ep | 88.41 | 88.18 | 88.29999566078186 | 88.0 | evaluated |
| P114_JSAP3_views4_800ep_seed0 | epoch_800.pt | JS-AP3|8x|4v/800ep(matched) | 88.72 | 88.02 | 88.71999979019165 | 87.5 | evaluated |
| P114_JSAP3_views4_800ep_seed1 | epoch_800.pt | JS-AP3|8x|4v/800ep(matched) | 89.25 | 88.10 | 89.30000066757202 | 87.98 | evaluated |
| P114_JSAP3_views4_800ep_seed2 | epoch_800.pt | JS-AP3|8x|4v/800ep(matched) | 88.79 | 87.99 | 88.41999769210815 | 87.54 | evaluated |
