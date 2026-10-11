# R7-00 — intake, scope and permissions — 2026-10-11 (queried ~22:15 UTC)

- **Package:** `VCS_CVPR_V6_Server_r7_VL_Research_Atlas_20261011.zip`, sha256 `1efe9fbe…f0d8a`.  Copied without PDFs to
  `r7/package_r7_20261011/`.
- **Repository:** package snapshot `87a847e` (18:59 UTC); actual HEAD `bdb052d`.  Delta = two r6 commits:
  - `603e5a6` R6-02 controlled validations;
  - `bdb052d` R6-04 route decision (REG on BF/SHG).
- **Owner instruction (verbatim):** "你规划一下，上一个包和…r7…这个包的内容同时做。这个包和上个包的数据全都下载到/projects/EEG-foundation-model，你先开始下载数据和准备代码，适当时候提交"
  (plan it; do r6 and r7 together; download both packages' data to /projects/EEG-foundation-model; start with data and code, submit when
  appropriate).

## Scope and authorization (precedence: owner > approved protocols > package)
| item | r7 package default | owner instruction / state | applied |
|---|---|---|---|
| downloads | no new large downloads; no accepting new data terms | **download both packages' data** to /projects/EEG-foundation-model | Tiered downloads by a data steward with a manifest, under a free-space floor of 1.2 TB on the shared disk (1.7 TB free now).  Web-scale scraping and raw video corpora (> ~100 GB) are listed for the owner, not started.  Account-gated sets (SP-DocVQA via RRC, Ego4D NLQ) need the owner's own registration.  Terms of directed downloads (e.g. CREMA-D, AV-MNIST / ESC-50) recorded as owner-directed, research and non-commercial use only. |
| official test / held-out data | never accessed automatically | "all data" | may be downloaded into `closed/` folders, hashed only, never opened; any read needs an explicitly approved one-shot protocol |
| r6 | REG / GAIN not auto-continued as large experiments | do r6 and r7 together | r6 REG-P1 (one natural route) proceeds: implementation and smoke in progress, frozen before results.  GAIN data downloaded; GAIN natural pilot after REG unless the owner reorders. |
| r7 training | one main route after R7-20; no six-way grid | standing authorization for useful experiments on free GPUs | one main route plus at most one light check, chosen at R7-30 on availability / necessity / prior art, not on a DEV win |
| VL wrap-up | do not cancel or duplicate R5-QN / R5-X1 | — | R5-QN: 3 runs left (aggregation, rescoring and diagnostics chained).  R5-X1: 5 running + 4 queued. |

## Resources
- **GPU:** my quota is 8 GPUs (RTX 6000 Pro / H100 / L40S, excluding node51 / 52 / 60); normal-QOS cap 30, kept <= 27.  Currently R5-QN + R5-X1.
- **Disk:** /projects/EEG-foundation-model 30 T, 1.7 T free.  Data under `yinghao/data/`, weights under `yinghao/models/`.
- **Existing weights / data for VL:** SigLIP 2 / CLIP / FG-CLIP checkpoints (VL1–VL3, R4, R5); COCO-based RefCOCO family images; VL1-21
  proposals; ImageNet-1k at /projects/common/imagenet (not an r7 set).

## Running work (r7 first round)
| R7 task | state |
|---|---|
| R7-00 intake | this file |
| R7-10 literature + data review | literature fork running (R7_LITERATURE_COMPLETION.md); data steward running (data/DATA_MANIFEST.json, R7_DATA_AUDIT.csv to follow) |
| R7-11 CPU reference tests | job 1034586 (isolated) |
| R7-20 D1 / D2 feasibility + D3 finite check | after R7-10 / 11 and Tier-A data |
| R7-30 route decision + frozen protocol | after R7-20 |
