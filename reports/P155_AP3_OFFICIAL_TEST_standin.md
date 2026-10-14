# P68 — final evaluation on the selection split (STAND-IN smoke — not the official test) — 2026-10-14T16:09:20Z

Head and kNN bank: the 45 000-image fit split, frozen pilot probe hyper-parameters, clean transform; the checkpoint of each run is the one every
decision was made on.  Selection-split numbers (the development metric) are shown next to the test numbers; single-seed cells are marked in the label.

## Per cell (mean ± sd over seeds; n)

| cell | n | test linear % | test kNN % | selection linear % | selection kNN % | runs |
|---|---|---|---|---|---|---|
| A-P3|8x|4v/800ep(final) | 1 | 89.06 | 87.30 | 89.06 | 87.30 | P107_AP3_views4_800ep_seed0 |

## Per run

| run | checkpoint | cell | test linear % | test kNN % | selection linear % | selection kNN % | status |
|---|---|---|---|---|---|---|---|
| P107_AP3_views4_800ep_seed0 | epoch_800.pt | A-P3|8x|4v/800ep(final) | 89.06 | 87.30 | 89.05999660491943 | 87.3 | evaluated |
