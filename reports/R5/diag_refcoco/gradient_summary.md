# r5 R5-20A — hard-distractor reference mass and score-level gradients (exploratory, DEV)

Score-level derivatives of each model's own training loss at its CAL-selected state; not encoder gradients.  Image-macro over queries with >= 1 valid and >= 1 invalid candidate; seed mean (per-seed values in gradient_diagnostics.jsonl).  T-based fields of softmax models are not identified (b unconstrained).

## refcoco

| model | shared top-1 q share | shared top-1 |g| share | self top-1 q share | self top-1 |g| share | shared top-5 |g| share | invalid T ≥ 0.95 | invalid T + 1 |
|---|---|---|---|---|---|---|---|
| VCS | 0.044 | 0.142 | 0.044 | 0.490 | 0.455 | 0.002 | 0.117 |
| JS | 0.044 | 0.138 | 0.044 | 0.480 | 0.450 | 0.001 | 0.127 |
| set softmax | 0.044 | 0.153 | 0.044 | 0.708 | 0.472 | 0.005 | 0.062 |
| distribution softmax | 0.044 | 0.164 | 0.044 | 0.643 | 0.495 | 0.006 | 0.100 |
| VCS(qnorm) | 0.044 | 0.159 | 0.044 | 0.582 | 0.485 | 0.001 | 0.108 |
| JS(qnorm) | 0.044 | 0.156 | 0.044 | 0.578 | 0.479 | 0.001 | 0.103 |

Missing inputs (not filled):

- /home/infres/yinwang/CS_QMI/outputs/R4_VL0/scores_r4vl2/refcoco_siglip2_b16_js_s0.pt
- /home/infres/yinwang/CS_QMI/outputs/R4_VL0/scores_r4vl2/refcoco_siglip2_b16_js_s1.pt
- /home/infres/yinwang/CS_QMI/outputs/R4_VL0/scores_r4vl2/refcoco_siglip2_b16_js_s2.pt
- /home/infres/yinwang/CS_QMI/outputs/R4_VL0/scores_r4vl2/refcoco_siglip2_b16_vcs_s0.pt
- /home/infres/yinwang/CS_QMI/outputs/R4_VL0/scores_r4vl2/refcoco_siglip2_b16_vcs_s1.pt
- /home/infres/yinwang/CS_QMI/outputs/R4_VL0/scores_r4vl2/refcoco_siglip2_b16_vcs_s2.pt
- /home/infres/yinwang/CS_QMI/outputs/R4_VL2/refcoco_siglip2_b16_js_nuscore_s0.json
- /home/infres/yinwang/CS_QMI/outputs/R4_VL2/refcoco_siglip2_b16_js_nuscore_s1.json
- /home/infres/yinwang/CS_QMI/outputs/R4_VL2/refcoco_siglip2_b16_js_nuscore_s2.json
- /home/infres/yinwang/CS_QMI/outputs/R4_VL2/refcoco_siglip2_b16_vcs_nuscore_s0.json
- /home/infres/yinwang/CS_QMI/outputs/R4_VL2/refcoco_siglip2_b16_vcs_nuscore_s1.json
- /home/infres/yinwang/CS_QMI/outputs/R4_VL2/refcoco_siglip2_b16_vcs_nuscore_s2.json
