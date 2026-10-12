# R6 + R7 data: final report (data steward, 2026-10-12)

**Scope and authorization.** The owner (2026-10-11) asked for all r6 and r7 package data to be downloaded to /projects/EEG-foundation-model, including the non-commercial sets (research use only).
- Data are under `yinghao/data/<item>` and weights under `yinghao/models/<item>`.
- Every item is recorded in `data/DATA_MANIFEST.json`: source, revision, licence, sha256, what is closed, and the authorization.
- Nothing is committed.

| File | Content |
|---|---|
| `reports/R7/R7_DATA_AUDIT.csv` | One row per item: status, source, revision, licence, size, sha256 status, closed parts, used by, path (56 rows, including blocked and deferred sets) |
| `reports/R7/R7_DATA_CLOSED_SHA256.csv` | Every file under a `closed/` folder, with its bytes and sha256 (15,093 files, 10.55 GB) |

## 1. Status

**Done: all 45 manifest items.**

- **r6, 6 items.**
  - Zurich Summer and BF/SHG: the Zenodo md5 matches; the BF/SHG test archive is closed and was never extracted.
  - AV-MNIST: sha256 recorded.
  - CREMA-D: all 14,884 WAV/FLV files match their Git LFS pointer's sha256 and size; the repo working tree is checked out with hard links.
  - EPFL RGB-NIR: listed as a local copy.
- **r7 fixtures, 6 items, and Tier A, 31 items.** Every Hugging Face file matches its published LFS sha256. Every other download has its sha256 recorded, and git repos are pinned to a commit.
- **Tier B, 2 items.**
  - QVHighlights annotations.
  - Moment-DETR features, from the HF mirror the coordinator approved. They carry the flag "unofficial mirror, no official checksum" and the coordinator's `validation_required` rule. A name-only check found features for every train and val video and query.

**Blocked, needs the owner's account.**

| Dataset | What the owner must do |
|---|---|
| SP-DocVQA (DS03) | Register at rrc.cvc.uab.es (Document VQA, Task 1 single page), accept the terms, and share the download links |
| Ego4D NLQ (DS33) | Sign the licence at ego4ddataset.com; AWS credentials arrive by e-mail in about 48 h; then the ego4d CLI fetches the NLQ annotations and features |

**Deferred: Tier C, not started.** None of these fits above the 1.2 TB floor; the owner decides.
- DataComp-small: about 450 GB of images plus 3 GB of metadata.
- CC3M: 130–400 GB.
- Recap-DataComp: needs the DataComp images.
- HowTo100M: 1–2 TB of features; the videos are tens of TB.
- Raw videos for DS29–DS35: 50 GB to over 500 GB per corpus.

**Not fetched: a Tier B gap, stated plainly.** Tier B covered QVHighlights only. The annotations of Charades-STA (DS30), ActivityNet Captions (DS31), HiREST (DS32) and MomentSeeker (DS35) are small and can be fetched on request. Their "official precomputed features" need a source check for each dataset. D6 is not on the current route (R7_ROUTE_DECISION: D1 is the main route).

**Not run.**
- The SNLI-VE jsonl generator: it is code shipped inside the download and needs a review first.
- The ECCV Caption Drive extras: they need gdown.
- PCME++ official weights: never released.
- CREMA-D AudioMP3: skipped by design.

## 2. Space

**`yinghao/data`: 201.7 GB on disk**, measured with du on 2026-10-12 at 01:20Z. ZFS compresses text, so the JSON items take less space than their size.

| Item folder | GB | Item folder | GB | Item folder | GB |
|---|---:|---|---:|---|---:|
| gqa | 48.36 | hpdv2 | 3.78 | geneval | 0.28 |
| coco | 44.08 | crema_d | 2.81 | pope_lmms | 0.25 |
| clevr_v1 | 39.23 | bf_shg_4550300 | 2.38 | snli_ve | 0.22 |
| qvhighlights | 18.65 | epfl_rgb_nir | 1.98 | hallusionbench | 0.19 |
| textvqa | 15.36 | chartqa | 1.83 | llava_mix665k | 0.18 |
| av_mnist_multibench | 4.46 | gqa_lmms | 1.47 | t2i_compbench | 0.13 |
| flickr30k | 4.39 | textvqa_lmms | 0.92 | conceptmix | 0.06 |
| cub_200_2011 | 4.28 | amber | 0.84 | r7_fixtures (R7-20 agent's own folder) | 0.06 |
| genai_bench | 3.95 | scienceqa | 0.63 | pope | 0.05 |
| | | mmbench | 0.52 | cxc, pcmepp | 0.03 each |
| | | zurich_summer_5914759 | 0.28 | mhal_detect, parti | 0.01 each |
| | | | | eccv_caption, sugarcrepe_pp | under 0.01 each |

- **Weights fetched by `data_fetch.py`:** 47.5 GB under `yinghao/models`. Other folders there that the tool did not create take 274.4 GB, for example `vl3` at 198 GB.
- **Free space:** 1528 GB, with the filesystem 96% used. It was 1561 GB when the downloads ended; the drop since then comes from other activity on the shared disk.
- **Floor:** free space never went below the 1.2 TB floor during the downloads, and the guard never fired.

## 3. Archives that could be deleted (listed only; the owner runs deletions)

For every extracted archive, each file it contains was found on disk, in the item folder or under `closed/`, with the archive's size. All 19 archives pass. Each archive's sha256 stays in the manifest, which also holds this list under `deletion_candidates` and the check results under `items.<id>.extraction_check`.

**Candidates: 72.4 GB, all re-downloadable from stable official hosts.**

| Archive | GB |
|---|---:|
| gqa/zips/images.zip | 21.82 |
| clevr_v1/zips/CLEVR_v1.0.zip | 19.02 |
| coco/zips/train2014.zip | 13.51 |
| textvqa/zips/train_val_images.zip | 7.07 |
| coco/zips/val2014.zip | 6.65 |
| gqa/zips/questions1.2.zip | 1.50 |
| cub_200_2011/zips/CUB_200_2011.tgz | 1.15 |
| coco/zips/val2017.zip | 0.82 |
| coco/zips/annotations_trainval2014.zip and annotations_trainval2017.zip | 0.50 |
| zurich_summer_5914759/zips/Zurich_dataset_v1.0.zip | 0.14 |
| snli_ve/zips/snli_1.0.zip | 0.09 |
| pcmepp/zips/coco_annotations.tar.gz | 0.05 |
| gqa/zips/sceneGraphs.zip | 0.04 |
| coco/zips/caption_datasets.zip | 0.04 |

For GQA questions, CLEVR, SNLI and PCME++, deleting the archive also removes the last copy of their test files outside `closed/`.

**One more candidate, not an archive:** `av_mnist_multibench/avmnist/`, 1.50 GB. It is my full extraction, now holding train arrays only. The r6 GAIN pilot reads its own train-only extraction in `extracted/avmnist/` instead.

**Recommend keeping: 11.0 GB.**
- **Google Drive copies:** `avmnist.tar.gz` (1.60 GB), `cvpr2016_cub.tar.gz` (0.86 GB) and `AMBER_images.zip` (0.42 GB). Two Drive links in this package already return 404.
- **QVHighlights mirror tarball:** `moment_detr_features.tar.gz` (8.12 GB), until `validation_required` passes.

## 4. Closed test files (sha256 computed, contents never parsed)

All 15,093 files and their sha256 are in `R7_DATA_CLOSED_SHA256.csv`.

| Item | Closed files | GB |
|---|---|---:|
| bf_shg_4550300 | test.zip, the official test split, never extracted | 1.153 |
| textvqa | test_images.zip and TextVQA_0.5.1_test.json, never extracted | 0.973 |
| hpdv2 | the whole HF repo: test and benchmark files, 55 files | 3.786 |
| gqa | test, test-dev, challenge and submission question files, 7 files | 0.914 |
| clevr_v1 | 15,000 test images and CLEVR_test_questions.json | 2.869 |
| chartqa, scienceqa, mmbench | the test parquets: 1 file, 1 file and 3 files (en, cn, cc) | 0.525 |
| av_mnist_multibench | the official test arrays (image and audio test_data.npy, test_labels.npy, plus the avmnist_temp copies), 5 files | 0.251 |
| snli_ve | snli_1.0_test.jsonl, snli_1.0_test.txt, flickr30k_test.lst | 0.017 |
| cxc | sis/sts/sits `_test` and `_test_raw` CSV files, 6 files | 0.062 |
| pcmepp | precomp test/testall and cxc_annots testall files, 6 files | 0.002 |
| qvhighlights | highlight_test_release.jsonl and highlight_test_with_gt.jsonl (the repo ships the test labels) | 0.001 |

**Test splits that stay open, marked eval-only.** They sit inside mixed files and cannot be separated without parsing: the COCO and Flickr30k Karpathy test sets, and CUB (train_test_split.txt, testclasses.txt).

**Evaluation-only benchmarks, kept open:** POPE, HallusionBench, AMBER, ECCV Caption, SugarCrepe++, GenEval, T2I-CompBench, ConceptMix, PartiPrompts and GenAI-Bench.

**Correction.** The AV-MNIST test arrays were left open by the first extraction, which had no close pattern; they were moved to `closed/` on 2026-10-12. The data steward never read them, and no script in the repo references them. The GAIN pilot is train-only by construction, and its tests enforce this.

## 5. Provenance flags and fixes

- **HallusionBench images:** the first author's Hugging Face upload (rayguan), because the README's Drive zip returns 404. Its HallusionBench.json is byte-identical to the GitHub copy at commit 744007c2.
- **QVHighlights features:** taken from an unofficial mirror; carries `validation_required`.
- **Flickr30k:** the HF mirror nlphuji/flickr30k; the official distribution is by request form.
- **llava_v1_5_mix665k:** carries the coordinator's provenance note ("used only to exclude LLaVA-seen GQA QA pairs before role assignment").
- **GQA readme:** `questions1.2.zip` and `sceneGraphs.zip` both contain a `readme.txt`. The questions readme is restored as `gqa/questions1.2_readme.txt`.
- **Disk speed:** the shared filesystem took 70–200 ms per file operation under load. GPU jobs that read the many-small-file folders (GQA, COCO, CLEVR, QVHighlights features) should copy the archive to node-local disk and extract it there.
