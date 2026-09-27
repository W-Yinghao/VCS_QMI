# P57 addendum 1 (DRAFT, to be frozen by the main session) — external target for A-S1 / A-T: Localized Narratives on the Open Images validation split (2026-09-27) — FROZEN 2026-09-27T19:40:28Z before GPU compute (download 1011799 and CPU probes 1011810 / 1011812 disclosed inside; owner: "都补充上")

## 0. Why nocaps was dropped (record)
nocaps (first candidate) was checked on 2026-09-27: its caption annotations carry no licence or terms text anywhere (site pages; the JSON
header shows `"licenses": []`), and the official files (`s3.amazonaws.com/nocaps/...`) return HTTP 403 from this cluster.  Nothing was
downloaded.  The coordinator switched the target to Localized Narratives.

## 1. Data and provenance (recorded before download; the download job writes `data/localized_narratives/PROVENANCE.json`)
- **Annotations:** Localized Narratives (Google), project page https://google.github.io/localized-narratives/ — verbatim: "All the annotations
  available through this website are released under a CC BY 4.0 license."  File: `open_images_validation_localized_narratives.jsonl`
  (https://storage.googleapis.com/localized-narratives/annotations/…, 1 139 215 790 bytes, last-modified 2021-01-25): 41 691 narratives on
  41 620 unique Open Images *validation* images (no test split is touched).
- **Images:** Open Images validation split, https://storage.googleapis.com/openimages/web/factsfigures_v7.html — verbatim: "The annotations
  are licensed by Google LLC under CC BY 4.0 license. The images are listed as having a CC BY 2.0 license." and "Note: while we tried to
  identify images that are licensed under a Creative Commons Attribution license, we make no representations or warranties regarding the
  license status of each image and you should verify the license for each image yourself."  Files fetched from
  `https://open-images-dataset.s3.amazonaws.com/validation/<ImageID>.jpg` (HEAD 200 from this cluster; the `storage.googleapis.com/openimages`
  path returns 403).  Per-image attribution (author, title, original URL, licence) from
  `https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv`; the rows of the selected images are
  kept as `attribution_selected.csv`.
- **Selection:** identity = Open Images image id; first narrative per image; 5 000 ids sampled without replacement from the sorted unique ids
  with seed 20260927 (`selection.json`, `ids.txt`); sha256 of the id list, of the jsonl and of the CSV in `PROVENANCE.json` (appended below
  when the job lands).
- **Not used:** Open Images test split; nocaps; any image whose download fails (counted, listed in `download_failures.txt`).

## 2. What is measured (`scripts/precheck_a_external.py`; jobs `slurm/precheck_a_external_features.sbatch`, `slurm/precheck_a_external.sbatch`)
- Frozen CLIP ViT-B/32 (P49 tower/cache) image features and the narrative's text feature.  Narratives are long (typically > 77 CLIP tokens):
  the CLIP tokenizer truncates at 77 and the **truncation rate is recorded** in the features manifest and printed in the table header.  The
  split is saved as `EXT-EVAL` in the P49 format (the single narrative embedding repeated in the five caption slots so all P49/P57 helpers
  apply unchanged).
- Adapters: the **P58 topic-pairing selection** (lr / epochs read from `reports/P58_precheck_A_wave2.json`) re-trained on each source —
  animal (P49 splits) and indoor (P57 splits) — for vcs / infonce / logistic, 3 seeds; training and SRC-CAL calibration exactly as in P58.
- A-S1 on EXT-EVAL: held-out J (vcs), native ECE ((1+T)/2; logistic sigmoid) vs cosine + Platt fitted on the respective SRC-CAL, R@1; the
  in-domain SRC-EVAL (topic) row repeated next to it.  Pairing on the external target is **exact** (own narrative) — there are no topic labels.
- A-T on EXT-EVAL: matched = own narrative, mismatched = a random other image's narrative (fixed derangement); rules fixed on the source:
  native accept ≥ nominal; cosine + Platt fitted on SRC-CAL task pairs with *random* mismatches (the kind available here) and, as a second
  transfer variant, with coarse-topic mismatches; score + Platt (vcs / logistic); target-fitted cross-fitted oracle (upper reference, not
  deployable); raw CLIP reference.  Nominal precision 0.8 (primary) and 0.7 (secondary) in one run.  Metrics as P57: realized precision vs
  nominal (primary), acceptance, FNR / FPR, balanced error, AUROC, ECE.

## 3. Pre-committed reading (P57's rules restricted to this target)
- A-S1: native calibration *holds on the external target* if native ECE ≤ Platt + 0.01 for every (source, method-with-native) cell with
  J ∈ [0.1, 0.7]; *does not hold* if any such cell has native > Platt + 0.02; else conditional.  Cells with J outside [0.1, 0.7] are reported,
  not read (the exact-pair relation on a strongly aligned target may land above 0.7 — that is then the P58 "exact-pair" regime again and is
  said so).
- A-T: *holds* if the native rule's realized precision is within 0.05 of the nominal and not worse than source-Platt-on-cosine (random-mismatch
  variant) by more than 0.02; *does not hold* if worse by ≥ 0.05; else conditional.  If AUROC on EXT-EVAL is below 0.6 for every embedding the
  task is declared undefined on this target (as for indoor → outdoor in P58).  Acceptance < 5 % → "degenerate acceptance", reading falls to
  conditional.
- Caveats fixed now: only random mismatches exist on this target (no topic-matched hard mismatches), so A-T here is the *easy* variant of P58;
  narrative truncation is a property of the CLIP tokenizer shared by every method; one tower, linear adapters, one seed of the towers.

## 4. Compute
Download: 1 CPU job (jsonl 1.1 GB + 5 000 images ≈ 0.6 GB + CSV).  Features: 1 GPU job (≈ 5 min).  Main run: 1 GPU job — 2 sources × 3 methods
× 3 seeds = 18 adapter trainings (≈ 10 min) + evaluations.  Probe: CPU smoke on 200 images, one method, one seed, one epoch (appended below).

## 5. Provenance as recorded by the download job (CPU job 1011799, `data/localized_narratives/PROVENANCE.json`, 2026-09-27T19:36:38Z)
narratives jsonl sha256 c9713328829c18cef80ed196674036605fe779c3e410b5b955662e35cd061e58 (1 139 215 790 bytes; 41 691 narratives, 41 620 unique
validation images); attribution CSV sha256 ed93a0e121fe345effdfc7359b848dbc64a1ff6778c8c73563157cb500b33a17; selected id list (5 000, seed
20260927) sha256 bd298bfdd0d1cfac69ff72efc83aca377f8de31730a642ff70d8cd44f875a402; 5 000 / 5 000 images downloaded (1 548 910 135 bytes, 0
failures); attribution rows found for all 5 000, every one with License = https://creativecommons.org/licenses/by/2.0/ (author, title,
original Flickr URL kept in `attribution_selected.csv`).

## 6. Probe (disclosed): CPU smoke on 200 images — features job 1011810, run job 1011812 (animal source only, vcs only, 1 epoch, 1 seed)
Truncation rate 0.040 (mean narrative length 30 words; on the smoke subset only 4 % exceed 77 CLIP tokens).  1-epoch adapter: EXT-EVAL held-out
J 0.360, R@1 0.630, native ECE 0.261 vs cosine + Platt 0.277 (both poorly calibrated after one epoch — probe only).  A-T at nominal 0.8:
native accept 0.445 / precision 0.972, source-Platt (random-mismatch calibration) 0.595 / 0.840, oracle 0.445 / 0.972; AUROC 0.986 for every
rule (the random-mismatch task is easy on this target).  Pipeline complete end to end; no design change follows.  Note for the reading: J on
the exact-pairing external target lands at 0.36 after one epoch and will likely exceed 0.7 after the selected 15 / 40 epochs (the P58
exact-pair regime); if so the A-S1 cell is reported and not read, as §3 says.
