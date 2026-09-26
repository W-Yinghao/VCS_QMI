# Geometry diagnostic (alignment / uniformity, Wang & Isola) — 2026-09-26T02:13:59Z

Selection set, two train-distribution views (fixed RNG), frozen final checkpoints. Lower alignment = views closer; lower (more negative) uniformity = points spread more evenly on the sphere.

| run | method | ep | critic | K | negdet | z: align | z: unif | z cos+ median | z cos− median | h: align | h: unif | h cos+ med | h cos− med | a / b / thr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P5_simclr_seed0 | simclr_matched | 200 | None | 1 | False | 0.262 | -3.842 | 0.923 | -0.010 | 0.236 | -2.807 | 0.917 | 0.239 | — |
| P5_vicreg_seed0 | vicreg_matched_128 | 200 | None | 1 | False | 0.274 | -3.678 | 0.937 | -0.015 | 0.208 | -2.523 | 0.927 | 0.270 | — |
| P24_vcs_cos_negdetach_seed0 | vcs_qmi | 200 | cosine | 8 | True | 0.090 | -1.563 | 0.965 | 0.575 | 0.192 | -1.792 | 0.922 | 0.516 | 9.81 / -8.09 / 0.825 |
| P26_vcs_a5_learn_seed0 | vcs_qmi | 200 | cosine | 8 | True | 0.101 | -1.560 | 0.959 | 0.584 | 0.208 | -1.793 | 0.913 | 0.525 | 10.08 / -8.22 / 0.816 |
| P28_vcs_views4_seed0 | vcs_qmi | 200 | cosine | 8 | True | 0.101 | -1.675 | 0.958 | 0.553 | 0.225 | -2.025 | 0.906 | 0.463 | 10.65 / -8.62 / 0.809 |
| P35_vcs_a5_views4_seed0 | vcs_qmi | 200 | cosine | 8 | True | 0.109 | -1.651 | 0.954 | 0.565 | 0.238 | -2.017 | 0.899 | 0.472 | 10.83 / -8.70 / 0.803 |
| P37_vcs_a5_views8_100ep_seed0 | vcs_qmi | 100 | cosine | 8 | True | 0.199 | -2.558 | 0.914 | 0.317 | 0.214 | -1.863 | 0.909 | 0.513 | 6.70 / -4.45 / 0.664 |
| P26_vcs_base_800ep_seed0 | vcs_qmi | 800 | cosine | 8 | True | 0.045 | -0.927 | 0.982 | 0.761 | 0.265 | -2.283 | 0.889 | 0.392 | 21.12 / -19.18 / 0.908 |
| P24_vcs_cos_800ep_seed0 | vcs_qmi | 800 | cosine | 8 | False | 0.381 | -3.199 | 0.822 | 0.155 | 0.286 | -1.789 | 0.873 | 0.539 | 13.03 / -6.95 / 0.533 |
