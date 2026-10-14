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

## n2 — 2026-10-14 ~00:20 UTC: how the amendment-v2 families are scored (before any GenAI-Rank label or prediction of the family exists)
`scripts/v7/v7_score.py rank` takes the family from `--comparators` and applies Holm over it; the code is not changed.  After the owner's scoring
approval and once every prediction file is in the manifest:
- primary (amendment v2): `--ours M2_E --comparators B0_FGA_released_O B1_VQAScore B4T_TIFA B4D_DSG` -> 8 contrasts (4 x 2 generators), Holm over 8;
- attribution: `--ours M2_E --comparators M1_E` -> 2 contrasts, Holm over 2, written to its own output file;
- v1-registered family (reported in full, not primary): `--comparators M1_E B0_FGA_released_O B1_VQAScore`, its own output file.
The scorer never overwrites an output, so each family is one new file; the three invocations read the same sealed labels once each.
