# V7 owner decisions

| when (UTC) | decision | scope | record |
|---|---|---|---|
| 2026-10-13 ~22:55 | allow a sealed split of GenAI-Rank `human_rating.json` | the prompt text only goes to an input file; the ratings move unread into `closed/genai_rank_labels_sealed.jsonl` (mode 0400; counts and sha256 only); scoring still needs a separate approval after the prediction / protocol freeze | `outputs/V7/data/genai_rank_seal_report.json` (source sha256 b4d812ec…, sealed sha256 0eb955b4…, 800 prompts, 14,400 rows, 0 structure problems; CPU job 1038690) |

Context: the Rank prompt ids (1–1611) are not GenAI-Bench's Index, and the label-free image mapping matched 1 of 800 prompts (GenAI-Bench's
per-prompt images are different samples), so the prompt text existed only in the label file.  750 of the 800 Rank prompts are verbatim
GenAI-Bench prompts; the other 50 are parsed separately into the same prompt-only cache.
