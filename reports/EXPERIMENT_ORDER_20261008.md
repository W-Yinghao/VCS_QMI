# Experiment order under the estimator-centred design — 2026-10-08 (owner: "就按这个设计来")

Rules unchanged: pre-register → freeze → compute; results-only commit then report; development splits only until the final pass.  Matched JS /
logistic is the same-posterior control everywhere (one row), never a tuning target.  No new SSL exploration or robustness cells.

| # | unit | what it delivers for the paper | depends on | compute | state |
|---|---|---|---|---|---|
| **A — running / queued (finish, report, no follow-ups)** |
| A1 | P130 add. 1: R50 CIFAR-100 JS / SimCLR seeds 1–2 | 3-seed R50 table (learning §, strongest SSL result) | — | 6 chains running, ≈ 1–2 days | running |
| A2 | P145 add. 2: STRESS ε 0.20, seed 0 | dose curve, appendix robustness | — | 6 GPU runs | 2 running, 4 queued; if its trigger fires I report it and ask, no seeds submitted |
| A3 | P138 add. 2: K = 16 seeds 3–4 | appendix: sampled-pair non-inferiority at 5 seeds | — | 8 GPU runs | queued |
| A4 | P153: weak crop C100 | appendix ablation (C10 already −2.2) | — | running | running |
| A5 | VL1-00 raw-CLIP DEV table (ViT-B/16-quickgelu) | Table A row 1; feature cache for A6–B3 | — | 1 GPU-h | queued |
| A6 | ReCLIP reproduction, RefCOCOg-UMD **val**, parse + IPS-only | Table B check vs 68.08 / 65.32 | — | 2 GPU runs | queued |
| **B — this week (all CPU except B3 / B6)** |
| B1 | VL1-10 Table A: raw, VCS, JS, softmax, SigLIP-style, RFF × N {1k, 4k, all} × 3 seeds (+ lr grid) | **the VL main table**: common J / calibration on CAL, sample-efficiency curve, Top-1 on DEV | A5 → VL1-10 gate → freeze | 6 CPU jobs, ≈ 2–3 days | gate script ready |
| B2 | VL1-11 prior-corrected ranking: phrase-uniform law (p(r) ∝ #expressions) → rank by 2f + log p(r) vs f on DEV | theory test of O1 §3.4 (a prediction CIFAR cannot test) | B1 checkpoints | evaluation only + 1 refit | to write |
| B3 | VL1-12 estimator-as-probe: same P / Q laws on CLIP vs **FG-CLIP 2 Base** (official RoIAlign region API) vs SigLIP 2 Base (crop adaptation); raw cosine + VCS / JS fits | "how much relation each foundation feature set carries"; Table A on stronger features | FG-CLIP 2 downloaded; venv built | ≈ 1 GPU-h extraction + CPU fits | extractor to write |
| B4 | decisive estimator figure (P151 + P152 panels; MI methods on own targets; log 1024 bound) | estimator § centrepiece | — | CPU script | to write |
| B5 | ablation table (from `status_v7.md`) + mechanism figure (pair-weight share, layer profile P141, J at z P149 add. 1) | learning § appendix + 1 figure | A1 | CPU | to write |
| B6 | **P154 ImageNet-100** (VCS / JS / SimCLR, R18, 200 ep; frozen, gate passed) | non-CIFAR sufficiency result | GPUs free after A1 | 3 × 7.5 GPU-h | **re-queued at the lowest priority (Nice 2000): runs only when no other job of mine needs the GPU** |
| **C — weeks 2–3** |
| C1 | Table B rows on DEV: ReCLIP (after A6 check), FG-CLIP 2 / SigLIP 2 zero-shot given-box | placement among public fine-grained models | A6, B3 | 3 GPU runs | — |
| C2 | Flickr30k Entities: fixture, audit, Table A (raw / VCS / JS / softmax at N 4k, 3 seeds) + ReCLIP | second VL dataset | **owner: image request form** | CPU + 1 GPU-h | blocked on images |
| C3 | optional appendix: Grounding DINO Swin-T OGC zero-shot vs proposer + VCS | grounding feasibility | clean proposer (≈ 1 GPU-day, owner budget D2) | — | not planned unless budgeted |
| C4 | **final pass**: freeze everything; official CIFAR test for the frozen SSL recipes; RefCOCOg UMD test for the frozen VL configurations | the paper's reported numbers | all above | 1 GPU-h | date = owner decision D3 |

Writing items carried: W1–W11 (estimator wording, log 1024, S_plugin on real data, detection ≠ estimation, cross-site J).
