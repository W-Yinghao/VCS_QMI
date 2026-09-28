# P68 — final evaluation on the selection split (STAND-IN smoke — not the official test) — 2026-09-28T06:11:28Z

Head and kNN bank: the 45 000-image fit split, frozen pilot probe hyper-parameters, clean transform; the checkpoint of each run is the one every
decision was made on.  Selection-split numbers (the development metric) are shown next to the test numbers; single-seed cells are marked in the label.

## Per cell (mean ± sd over seeds; n)

| cell | n | test linear % | test kNN % | selection linear % | selection kNN % | runs |
|---|---|---|---|---|---|---|
| VCS|1x|4v/B128/100ep | 1 | 83.52 | 78.30 | 83.52 | 78.30 | P39_vcs_a5_views4_b128_100ep_seed0 |

## Per run

| run | checkpoint | cell | test linear % | test kNN % | selection linear % | selection kNN % | status |
|---|---|---|---|---|---|---|---|
| P39_vcs_a5_views4_b128_100ep_seed0 | epoch_100.pt | VCS|1x|4v/B128/100ep | 83.52 | 78.30 | 83.51999521255493 | 78.3 | evaluated |
