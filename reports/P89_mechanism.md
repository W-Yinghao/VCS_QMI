# S4 mechanism records — reports/P89_mechanism (2026-09-29T20:07:11Z)

Frozen checkpoints, eval mode, same base images / views / shifts for every run (seeds in the JSONs).  gate_M = actual mean 1 − T² over the equal mixture (not 1 − J).  Gradients are per image, loss averaged over the batch.

| run | ep | aug | method | J / NT-Xent / D_CS | gate_M | 1−J | |res| pos / neg | sat pos / neg | cos_z pos / neg (med) | ‖∂x‖ v1 (mean) | ‖∂p‖ (mean) | align h | unif h | erank h | SimCLR neg mass / eff. n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P35_vcs_a5_views4_800ep_seed0 | 100 | own | vcs_qmi | 0.9421 | 0.2152 | 0.0579 | 0.147 / 0.126 | 0.021 / 0.597 | 0.955 / 0.600 | 2.134e-03 | 2.607e-05 | 0.234 | -1.985 | 75.1 | nan / nan |
| P35_vcs_a5_views4_800ep_seed0 | 100 | standard | vcs_qmi | 0.9421 | 0.2152 | 0.0579 | 0.147 / 0.126 | 0.021 / 0.597 | 0.955 / 0.600 | 2.134e-03 | 2.607e-05 | 0.234 | -1.985 | 75.1 | nan / nan |
| P35_vcs_a5_views4_800ep_seed0 | 400 | own | vcs_qmi | 0.9677 | 0.1405 | 0.0323 | 0.112 / 0.061 | 0.328 / 0.799 | 0.977 / 0.765 | 2.101e-03 | 1.174e-05 | 0.315 | -2.440 | 136.7 | nan / nan |
| P35_vcs_a5_views4_800ep_seed0 | 400 | standard | vcs_qmi | 0.9677 | 0.1405 | 0.0323 | 0.112 / 0.061 | 0.328 / 0.799 | 0.977 / 0.765 | 2.101e-03 | 1.174e-05 | 0.315 | -2.440 | 136.7 | nan / nan |
| P35_vcs_a5_views4_800ep_seed0 | 800 | own | vcs_qmi | 0.9740 | 0.1167 | 0.0260 | 0.095 / 0.047 | 0.493 / 0.846 | 0.978 / 0.773 | 2.023e-03 | 1.120e-05 | 0.327 | -2.469 | 150.4 | nan / nan |
| P35_vcs_a5_views4_800ep_seed0 | 800 | standard | vcs_qmi | 0.9740 | 0.1167 | 0.0260 | 0.095 / 0.047 | 0.493 / 0.846 | 0.978 / 0.773 | 2.023e-03 | 1.120e-05 | 0.327 | -2.469 | 150.4 | nan / nan |
| P35_vcs_a5_views4_800ep_seed1 | 100 | own | vcs_qmi | 0.9395 | 0.2224 | 0.0605 | 0.153 / 0.130 | 0.017 / 0.586 | 0.954 / 0.604 | 2.265e-03 | 2.757e-05 | 0.237 | -1.982 | 74.6 | nan / nan |
| P35_vcs_a5_views4_800ep_seed1 | 100 | standard | vcs_qmi | 0.9395 | 0.2224 | 0.0605 | 0.153 / 0.130 | 0.017 / 0.586 | 0.954 / 0.604 | 2.265e-03 | 2.757e-05 | 0.237 | -1.982 | 74.6 | nan / nan |
| P35_vcs_a5_views4_800ep_seed1 | 400 | own | vcs_qmi | 0.9659 | 0.1412 | 0.0341 | 0.114 / 0.062 | 0.332 / 0.798 | 0.976 / 0.766 | 2.158e-03 | 1.251e-05 | 0.312 | -2.435 | 130.7 | nan / nan |
| P35_vcs_a5_views4_800ep_seed1 | 400 | standard | vcs_qmi | 0.9659 | 0.1412 | 0.0341 | 0.114 / 0.062 | 0.332 / 0.798 | 0.976 / 0.766 | 2.158e-03 | 1.251e-05 | 0.312 | -2.435 | 130.7 | nan / nan |
| P35_vcs_a5_views4_800ep_seed1 | 800 | own | vcs_qmi | 0.9726 | 0.1178 | 0.0274 | 0.098 / 0.048 | 0.482 / 0.844 | 0.978 / 0.774 | 2.067e-03 | 1.163e-05 | 0.323 | -2.454 | 145.5 | nan / nan |
| P35_vcs_a5_views4_800ep_seed1 | 800 | standard | vcs_qmi | 0.9726 | 0.1178 | 0.0274 | 0.098 / 0.048 | 0.482 / 0.844 | 0.978 / 0.774 | 2.067e-03 | 1.163e-05 | 0.323 | -2.454 | 145.5 | nan / nan |
| P41_simclr_views4_800ep_seed0 | 100 | own | simclr_matched | 2.3555 | nan | nan | nan / nan | nan / nan | 0.914 / -0.008 | 2.002e-02 | 5.693e-04 | 0.270 | -2.948 | 103.6 | 0.886 / 334.9 |
| P41_simclr_views4_800ep_seed0 | 100 | standard | simclr_matched | 2.3555 | nan | nan | nan / nan | nan / nan | 0.914 / -0.008 | 2.002e-02 | 5.693e-04 | 0.270 | -2.948 | 103.6 | 0.886 / 334.9 |
| P41_simclr_views4_800ep_seed0 | 400 | own | simclr_matched | 2.3411 | nan | nan | nan / nan | nan / nan | 0.920 / -0.006 | 2.842e-02 | 7.522e-04 | 0.370 | -3.336 | 150.8 | 0.880 / 355.7 |
| P41_simclr_views4_800ep_seed0 | 400 | standard | simclr_matched | 2.3411 | nan | nan | nan / nan | nan / nan | 0.920 / -0.006 | 2.842e-02 | 7.522e-04 | 0.370 | -3.336 | 150.8 | 0.880 / 355.7 |
| P41_simclr_views4_800ep_seed0 | 800 | own | simclr_matched | 2.3546 | nan | nan | nan / nan | nan / nan | 0.924 / -0.004 | 3.311e-02 | 1.023e-03 | 0.369 | -3.220 | 159.0 | 0.879 / 362.5 |
| P41_simclr_views4_800ep_seed0 | 800 | standard | simclr_matched | 2.3546 | nan | nan | nan / nan | nan / nan | 0.924 / -0.004 | 3.311e-02 | 1.023e-03 | 0.369 | -3.220 | 159.0 | 0.879 / 362.5 |
| P41_simclr_views4_800ep_seed1 | 100 | own | simclr_matched | 2.3522 | nan | nan | nan / nan | nan / nan | 0.916 / -0.011 | 1.950e-02 | 5.623e-04 | 0.275 | -2.956 | 106.5 | 0.886 / 334.6 |
| P41_simclr_views4_800ep_seed1 | 100 | standard | simclr_matched | 2.3522 | nan | nan | nan / nan | nan / nan | 0.916 / -0.011 | 1.950e-02 | 5.623e-04 | 0.275 | -2.956 | 106.5 | 0.886 / 334.6 |
| P41_simclr_views4_800ep_seed1 | 400 | own | simclr_matched | 2.3466 | nan | nan | nan / nan | nan / nan | 0.923 / -0.005 | 2.724e-02 | 7.421e-04 | 0.381 | -3.413 | 146.2 | 0.879 / 354.7 |
| P41_simclr_views4_800ep_seed1 | 400 | standard | simclr_matched | 2.3466 | nan | nan | nan / nan | nan / nan | 0.923 / -0.005 | 2.724e-02 | 7.421e-04 | 0.381 | -3.413 | 146.2 | 0.879 / 354.7 |
| P41_simclr_views4_800ep_seed1 | 800 | own | simclr_matched | 2.3542 | nan | nan | nan / nan | nan / nan | 0.924 / -0.005 | 3.106e-02 | 9.884e-04 | 0.406 | -3.476 | 154.2 | 0.878 / 362.5 |
| P41_simclr_views4_800ep_seed1 | 800 | standard | simclr_matched | 2.3542 | nan | nan | nan / nan | nan / nan | 0.924 / -0.005 | 3.106e-02 | 9.884e-04 | 0.406 | -3.476 | 154.2 | 0.878 / 362.5 |
| P43_vcs_a5_views4_800ep_augstrong_seed0 | 100 | own | vcs_qmi | 0.7920 | 0.4669 | 0.2080 | 0.347 / 0.328 | 0.000 / 0.208 | 0.970 / 0.777 | 6.887e-03 | 4.554e-05 | 0.378 | -2.086 | 69.0 | nan / nan |
| P43_vcs_a5_views4_800ep_augstrong_seed0 | 100 | standard | vcs_qmi | 0.8699 | 0.3904 | 0.1301 | 0.257 / 0.264 | 0.000 / 0.271 | 0.981 / 0.760 | 2.725e-03 | 2.558e-05 | 0.253 | -2.055 | 64.4 | nan / nan |
| P43_vcs_a5_views4_800ep_augstrong_seed0 | 400 | own | vcs_qmi | 0.8481 | 0.3776 | 0.1519 | 0.286 / 0.244 | 0.000 / 0.339 | 0.986 / 0.883 | 8.719e-03 | 2.596e-05 | 0.469 | -2.463 | 121.9 | nan / nan |
| P43_vcs_a5_views4_800ep_augstrong_seed0 | 400 | standard | vcs_qmi | 0.9154 | 0.2993 | 0.0846 | 0.196 / 0.188 | 0.000 / 0.425 | 0.991 / 0.876 | 3.025e-03 | 1.337e-05 | 0.321 | -2.417 | 113.1 | nan / nan |
| P43_vcs_a5_views4_800ep_augstrong_seed0 | 800 | own | vcs_qmi | 0.8735 | 0.3344 | 0.1265 | 0.258 / 0.203 | 0.000 / 0.423 | 0.988 / 0.888 | 8.352e-03 | 2.544e-05 | 0.463 | -2.477 | 131.6 | nan / nan |
| P43_vcs_a5_views4_800ep_augstrong_seed0 | 800 | standard | vcs_qmi | 0.9291 | 0.2668 | 0.0709 | 0.180 / 0.158 | 0.000 / 0.503 | 0.992 / 0.882 | 3.030e-03 | 1.355e-05 | 0.328 | -2.428 | 122.7 | nan / nan |
| P89_simclr_views4_800ep_augstrong_seed0 | 100 | own | simclr_matched | 2.8765 | nan | nan | nan / nan | nan / nan | 0.852 / -0.018 | 3.873e-02 | 4.880e-04 | 0.397 | -2.614 | 97.9 | 0.919 / 296.7 |
| P89_simclr_views4_800ep_augstrong_seed0 | 100 | standard | simclr_matched | 2.4252 | nan | nan | nan / nan | nan / nan | 0.913 / -0.015 | 1.897e-02 | 3.581e-04 | 0.258 | -2.608 | 94.1 | 0.897 / 295.9 |
| P89_simclr_views4_800ep_augstrong_seed0 | 400 | own | simclr_matched | 2.7230 | nan | nan | nan / nan | nan / nan | 0.878 / -0.021 | 4.979e-02 | 4.312e-04 | 0.481 | -3.060 | 135.8 | 0.906 / 309.1 |
| P89_simclr_views4_800ep_augstrong_seed0 | 400 | standard | simclr_matched | 2.2982 | nan | nan | nan / nan | nan / nan | 0.927 / -0.018 | 2.402e-02 | 3.316e-04 | 0.328 | -3.067 | 131.1 | 0.884 / 307.9 |
| P89_simclr_views4_800ep_augstrong_seed0 | 800 | own | simclr_matched | 2.6825 | nan | nan | nan / nan | nan / nan | 0.885 / -0.019 | 5.348e-02 | 4.844e-04 | 0.493 | -3.098 | 143.1 | 0.903 / 313.3 |
| P89_simclr_views4_800ep_augstrong_seed0 | 800 | standard | simclr_matched | 2.2743 | nan | nan | nan / nan | nan / nan | 0.932 / -0.017 | 2.522e-02 | 3.848e-04 | 0.342 | -3.102 | 137.5 | 0.881 / 312.2 |
| P89_simclr_views4_800ep_augstrong_seed1 | 100 | own | simclr_matched | 2.8818 | nan | nan | nan / nan | nan / nan | 0.849 / -0.017 | 3.864e-02 | 4.929e-04 | 0.403 | -2.635 | 98.7 | 0.919 / 297.5 |
| P89_simclr_views4_800ep_augstrong_seed1 | 100 | standard | simclr_matched | 2.4222 | nan | nan | nan / nan | nan / nan | 0.915 / -0.016 | 1.829e-02 | 3.597e-04 | 0.262 | -2.632 | 94.8 | 0.896 / 296.2 |
| P89_simclr_views4_800ep_augstrong_seed1 | 400 | own | simclr_matched | 2.7575 | nan | nan | nan / nan | nan / nan | 0.874 / -0.020 | 5.207e-02 | 4.407e-04 | 0.490 | -3.043 | 138.2 | 0.908 / 308.6 |
| P89_simclr_views4_800ep_augstrong_seed1 | 400 | standard | simclr_matched | 2.3106 | nan | nan | nan / nan | nan / nan | 0.927 / -0.016 | 2.439e-02 | 3.387e-04 | 0.329 | -3.053 | 129.7 | 0.885 / 308.0 |
| P89_simclr_views4_800ep_augstrong_seed1 | 800 | own | simclr_matched | 2.6849 | nan | nan | nan / nan | nan / nan | 0.888 / -0.017 | 5.415e-02 | 4.854e-04 | 0.488 | -3.066 | 141.2 | 0.903 / 312.9 |
| P89_simclr_views4_800ep_augstrong_seed1 | 800 | standard | simclr_matched | 2.2776 | nan | nan | nan / nan | nan / nan | 0.934 / -0.016 | 2.586e-02 | 3.895e-04 | 0.337 | -3.072 | 132.6 | 0.881 / 312.7 |
| P89_vcs_a5_views4_800ep_augstrong_seed1 | 100 | own | vcs_qmi | 0.7885 | 0.4754 | 0.2115 | 0.353 / 0.334 | 0.000 / 0.196 | 0.971 / 0.782 | 6.610e-03 | 4.557e-05 | 0.387 | -2.123 | 69.0 | nan / nan |
| P89_vcs_a5_views4_800ep_augstrong_seed1 | 100 | standard | vcs_qmi | 0.8666 | 0.3990 | 0.1334 | 0.262 / 0.271 | 0.000 / 0.252 | 0.982 / 0.766 | 2.475e-03 | 2.569e-05 | 0.255 | -2.082 | 63.7 | nan / nan |
| P89_vcs_a5_views4_800ep_augstrong_seed1 | 400 | own | vcs_qmi | 0.8515 | 0.3707 | 0.1485 | 0.291 / 0.228 | 0.000 / 0.369 | 0.986 / 0.881 | 8.368e-03 | 2.619e-05 | 0.471 | -2.494 | 122.1 | nan / nan |
| P89_vcs_a5_views4_800ep_augstrong_seed1 | 400 | standard | vcs_qmi | 0.9190 | 0.2959 | 0.0810 | 0.200 / 0.177 | 0.000 / 0.441 | 0.991 / 0.874 | 2.818e-03 | 1.326e-05 | 0.321 | -2.447 | 113.9 | nan / nan |
| P89_vcs_a5_views4_800ep_augstrong_seed1 | 800 | own | vcs_qmi | 0.8731 | 0.3343 | 0.1269 | 0.260 / 0.201 | 0.000 / 0.430 | 0.988 / 0.889 | 8.166e-03 | 2.541e-05 | 0.470 | -2.505 | 132.1 | nan / nan |
| P89_vcs_a5_views4_800ep_augstrong_seed1 | 800 | standard | vcs_qmi | 0.9307 | 0.2652 | 0.0693 | 0.179 / 0.155 | 0.000 / 0.504 | 0.992 / 0.882 | 2.846e-03 | 1.303e-05 | 0.329 | -2.454 | 123.9 | nan / nan |
