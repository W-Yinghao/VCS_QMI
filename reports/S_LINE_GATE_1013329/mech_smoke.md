# S4 mechanism records — /home/infres/yinwang/CS_QMI/ssl_pilot/reports/S_LINE_GATE_1013329/mech_smoke (2026-09-28T20:11:35Z)

Frozen checkpoints, eval mode, same base images / views / shifts for every run (seeds in the JSONs).  gate_M = actual mean 1 − T² over the equal mixture (not 1 − J).  Gradients are per image, loss averaged over the batch.

| run | ep | aug | method | J / NT-Xent / D_CS | gate_M | 1−J | |res| pos / neg | sat pos / neg | cos_z pos / neg (med) | ‖∂x‖ v1 (mean) | ‖∂p‖ (mean) | align h | unif h | erank h | SimCLR neg mass / eff. n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P35_vcs_a5_views4_800ep_seed0 | 800 | own | vcs_qmi | 0.9627 | 0.1262 | 0.0373 | 0.119 / 0.045 | 0.453 / 0.828 | 0.976 / 0.776 | 9.053e-03 | 5.647e-05 | 0.367 | -2.490 | 40.8 | nan / nan |
| P41_simclr_views4_800ep_seed0 | 800 | own | simclr_matched | 1.4166 | nan | nan | nan / nan | nan / nan | 0.884 / -0.009 | 1.245e-01 | 3.919e-03 | 0.451 | -3.229 | 35.3 | 0.674 / 103.8 |
| P43_vcs_a5_views4_800ep_augstrong_seed0 | 800 | own | vcs_qmi | 0.8665 | 0.3396 | 0.1335 | 0.282 / 0.192 | 0.000 / 0.424 | 0.987 / 0.884 | 3.315e-02 | 1.074e-04 | 0.529 | -2.499 | 39.6 | nan / nan |
| P43_vcs_a5_views4_800ep_augstrong_seed0 | 800 | standard | vcs_qmi | 0.9346 | 0.2719 | 0.0654 | 0.195 / 0.142 | 0.000 / 0.488 | 0.991 / 0.882 | 1.253e-02 | 6.390e-05 | 0.388 | -2.465 | 38.8 | nan / nan |
