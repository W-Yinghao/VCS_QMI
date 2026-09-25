# Geometry diagnostic (alignment / uniformity, Wang & Isola) — 2026-09-25T16:34:11Z

Selection set, two train-distribution views (fixed RNG), frozen final checkpoints. Lower alignment = views closer; lower (more negative) uniformity = points spread more evenly on the sphere.

| run | method | ep | critic | K | negdet | z: align | z: unif | z cos+ median | z cos− median | h: align | h: unif | h cos+ med | h cos− med | a / b / thr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P5_simclr_seed0 | simclr_matched | 200 | None | 1 | False | 0.262 | -3.842 | 0.923 | -0.010 | 0.236 | -2.807 | 0.917 | 0.239 | — |
| P5_vicreg_seed0 | vicreg_matched_128 | 200 | None | 1 | False | 0.274 | -3.678 | 0.937 | -0.015 | 0.208 | -2.523 | 0.927 | 0.270 | — |
| P5_vcs_seed0 | vcs_qmi | 200 | ordered_concat | 1 | False | 0.178 | -2.047 | 0.942 | 0.368 | 0.161 | -1.511 | 0.940 | 0.575 | — |
| P10_vcs_k8_seed0 | vcs_qmi | 200 | ordered_concat | 8 | False | 0.157 | -1.931 | 0.947 | 0.436 | 0.182 | -1.614 | 0.930 | 0.549 | — |
| P16_vcs_k8_clr10_seed0 | vcs_qmi | 200 | ordered_concat | 8 | False | 0.209 | -2.236 | 0.925 | 0.355 | 0.203 | -1.611 | 0.917 | 0.564 | — |
| P18_vcs_crit_cosine_seed0 | vcs_qmi | 200 | cosine | 8 | False | 0.420 | -3.505 | 0.808 | 0.063 | 0.222 | -1.561 | 0.902 | 0.594 | 8.35 / -3.91 / 0.468 |
| P18_vcs_crit_interact_seed0 | vcs_qmi | 200 | concat_interact | 8 | False | 0.359 | -3.157 | 0.848 | 0.122 | 0.213 | -1.447 | 0.908 | 0.621 | — |
| P24_vcs_cos_k255_seed0 | vcs_qmi | 200 | cosine | 255 | False | 0.423 | -3.510 | 0.806 | 0.064 | 0.225 | -1.574 | 0.900 | 0.592 | 8.49 / -3.95 / 0.465 |
| P24_vcs_cos_negdetach_seed0 | vcs_qmi | 200 | cosine | 8 | True | 0.090 | -1.563 | 0.965 | 0.575 | 0.192 | -1.792 | 0.922 | 0.516 | 9.81 / -8.09 / 0.825 |
| P24_vcs_cos_on_h_seed0 | vcs_qmi | 200 | cosine | 8 | False | 0.368 | -1.659 | 0.838 | 0.568 | 0.548 | -3.293 | 0.738 | 0.143 | 9.27 / -3.94 / 0.425 |
| P8_vcs800_seed0 | vcs_qmi | 800 | ordered_concat | 1 | False | 0.126 | -1.548 | 0.959 | 0.569 | 0.197 | -1.634 | 0.925 | 0.552 | — |
| P16_vcs_k8_clr10_800ep_seed0 | vcs_qmi | 800 | ordered_concat | 8 | False | 0.137 | -1.595 | 0.952 | 0.565 | 0.239 | -1.855 | 0.900 | 0.501 | — |
