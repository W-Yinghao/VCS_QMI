# VCS-QMI CVPR manuscript v6 — 2026-10-06

## Files

- `VCS_QMI_CVPR_Manuscript_v6.pdf`: 12 pages: 6 main-text pages, 1 reference page, 5 supplementary pages.
- `VCS_QMI_CVPR_Manuscript_v6_Main.pdf`: 7-page main-text and references extract; use the full PDF for linked supplementary proofs.
- `VCS_QMI_CVPR_Manuscript_v6_EN.md`: editable text rendering of the main paper.
- `Paper_Positioning_and_Interpretability_CN.md` / `.pdf`: author-side whole-paper rationale and experiment placement.
- `source/`: LaTeX, linked BibTeX references, a print bibliography, source figures, and build script.
- `evidence/`: claim/result provenance and independent algebra/arithmetic checks; exclude from anonymous submission.

## Build

Run `bash source/build.sh` with a standard LaTeX distribution and BibTeX. The script supports `bibtex.original` in the supplied environment as a fallback.
Figures are supplied as vector PDFs. The manuscript uses a two-column authoring layout rather than claiming complete target-year submission-format certification. Move its sections and bibliography into the target year's official author kit when the submission metadata is available. No author identities or fabricated paper ID are inserted in the manuscript.

The main PDF contains completed, source-supported results only. Server task status and unrun experiment proposals are confined to the Chinese positioning note and this author package. The current paper is not a claim that the empirical package is submission-complete: ImageNet training and broader independent measurement comparisons still require actual results.

## Version decisions

This version updates the earlier `VCS_QMI_Manuscript_v5.pdf`, replaces the learned-scale/negative-detach main training recipe by the selected fixed-scale/all-view/full-gradient implementation, and unifies its paper name as VCS.
Interpretability is a focused direct measurement application of the estimator, not a new universal saliency method. Oracle compression checks belong to estimator evaluation, and internal recipe comparisons belong to ablations.

The result snapshot is `bc28bf89c94d5331de811a83aa8bcbfb77d7d155`, 2026-10-06 05:58:03 UTC. No GitHub writes, server jobs, or new pretrained models were produced for this update.
