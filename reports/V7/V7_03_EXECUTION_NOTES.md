# V7-03 execution notes (dated; each note is written before the event it governs)

## n1 — 2026-10-14 ~00:10 UTC: DEV metrics written as a by-product of the M-arm export (before any export has run: the first two tuning
## units had been training for ~11 min; an export follows ~1.4 h of training)
`v7_train.py export` (run inside every M unit's job after training) calls the shared `evaluate()` for CAL_SELECT, CAL_PROB and DEV, so it
writes DEV metrics into `<unit>/pred/export.json` and prints a DEV natural Brier line into the unit's job log.  The frozen protocol says no
DEV label is read before all 12 formal units and every export exist.  Rule from now on (no code change: the code is frozen and every job
prints its hash):
- the lr selection reads only the tuning units' `log.json` (CAL_SELECT; `v7_tune_select.py`), never `export.json`;
- no `export.json` and no DEV line of an M job log is opened, grepped or summarised before all 12 formal units and every export exist (the
  monitor reports only a job's last line, `exit=... end=...`);
- the DEV numbers of Table A come only from `v7_score.py table-a` reading `elements_DEV.jsonl` after that point;
- N0 is not affected (the native `eval.py` writes predictions only).
