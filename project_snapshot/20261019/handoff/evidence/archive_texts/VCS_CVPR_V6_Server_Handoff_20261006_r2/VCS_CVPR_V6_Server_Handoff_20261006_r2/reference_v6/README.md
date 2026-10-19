# VCS-QMI CVPR manuscript — V6, revised interpretation section

The paper version remains **V6**. `Revised` in this package's directory name only distinguishes this copy from the preceding V6. It does not denote V7 and does not redefine the estimator.

## Read first

- `VCS_QMI_CVPR_Manuscript_v6_Main.pdf`: 7 pages — 6 pages of main text and 1 page of references.
- `VCS_QMI_CVPR_Manuscript_v6.pdf`: 13 pages — main paper, references, and 6 supplementary pages.
- `VCS_QMI_CVPR_Manuscript_v6_EN.md`: complete editable main-paper export, including numbered references.
- `VCS_QMI_CVPR_Supplement_v6_EN.md`: editable supplement.
- `Paper_Positioning_and_Interpretability_CN.pdf` / `.md`: 3-page author note explaining the within-V6 changes and remaining evidence needs.

## What changed

The abstract, introduction, method framing, representation-analysis section, and conclusion now connect an independently checked measurement procedure to understanding what the final SSL representations retain. Section 5.3 is **Measuring What SSL Representations Retain**. It first compares estimators on a fixed external model, then uses a fixed VCS measurement procedure to examine the final VCS and SimCLR encoders. These are explicitly different comparisons.

The main theory block, existing estimator and SSL table values, figure data, and hyperparameter-selection rules are unchanged. A new matched-condition model-analysis table is assembled from already completed color-strength-0.1 results on encoder seeds 1–2; its color and blur details are in the supplement. It does not compare two internal VCS recipes or claim inherent semantic interpretability.

## Source and evidence

The empirical snapshot is unchanged: `W-Yinghao/VCS_QMI@bc28bf89c94d5331de811a83aa8bcbfb77d7d155`. The newly assembled rows were read at this fixed snapshot from `reports/P118_nest_results.md` (blob `2ebe8acb02365c9417d6660a251d272344912bbf`). No latest-HEAD review, new training, checkpoint re-evaluation, server submission, or GitHub write was performed for this revision.

`evidence/` contains the source-row transcription, original and revised claim ledger, within-V6 patch, integrity contract, and local verification outputs. These author materials are not part of an anonymous submission.

## Build

Run `bash source/build.sh` to compile the full LaTeX document to `source/main.pdf`. The root PDFs are the already compiled full version and its main-plus-references extract. Run `python source/export_markdown.py` after compilation to regenerate both Markdown exports; this requires Pandoc. Numerical checks use Python, NumPy, SciPy, and PyMuPDF.

The file `source/author_note_header.tex` is supplied for rebuilding the Chinese author note using Pandoc and XeLaTeX with installed CJK fonts. No font binaries are included.

The paper retains the preceding V6 two-column authoring layout. This package does not assert certification against a particular conference year's submission kit. All current accuracy tables retain their validation designation. Unrun ImageNet and additional measurement comparisons are not filled with predicted results.

## Server handoff

This revision changes manuscript organization, not execution authorization. When integrating it with the existing V6 server handoff, replace the manuscript and positioning references and retain the current job registry and evidence history. Consult the patch and author note; do not cancel or restart jobs solely because of the wording update.
