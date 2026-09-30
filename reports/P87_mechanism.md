# S4 mechanism records — reports/P87_mechanism (2026-09-30T00:44:29Z)

Frozen checkpoints, eval mode, same base images / views / shifts for every run (seeds in the JSONs).  gate_M = actual mean 1 − T² over the equal mixture (not 1 − J).  Gradients are per image, loss averaged over the batch.

| run | ep | aug | method | J / NT-Xent / D_CS | gate_M | 1−J | |res| pos / neg | sat pos / neg | cos_z pos / neg (med) | ‖∂x‖ v1 (mean) | ‖∂p‖ (mean) | align h | unif h | erank h | SimCLR neg mass / eff. n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P35_vcs_a5_views4_800ep_seed0 | 100 | own | vcs_qmi | 0.9421 | 0.2152 | 0.0579 | 0.147 / 0.126 | 0.021 / 0.597 | 0.956 / 0.600 | 2.134e-03 | 2.607e-05 | 0.234 | -1.985 | 75.1 | nan / nan |
| P35_vcs_a5_views4_800ep_seed0 | 400 | own | vcs_qmi | 0.9677 | 0.1405 | 0.0323 | 0.112 / 0.061 | 0.328 / 0.799 | 0.977 / 0.765 | 2.100e-03 | 1.174e-05 | 0.315 | -2.440 | 136.7 | nan / nan |
| P35_vcs_a5_views4_800ep_seed0 | 800 | own | vcs_qmi | 0.9740 | 0.1167 | 0.0260 | 0.095 / 0.047 | 0.493 / 0.846 | 0.978 / 0.773 | 2.024e-03 | 1.120e-05 | 0.327 | -2.469 | 150.4 | nan / nan |
| P35_vcs_a5_views4_800ep_seed1 | 100 | own | vcs_qmi | 0.9395 | 0.2224 | 0.0605 | 0.153 / 0.130 | 0.017 / 0.586 | 0.954 / 0.604 | 2.265e-03 | 2.757e-05 | 0.237 | -1.982 | 74.6 | nan / nan |
| P35_vcs_a5_views4_800ep_seed1 | 400 | own | vcs_qmi | 0.9659 | 0.1412 | 0.0341 | 0.114 / 0.062 | 0.332 / 0.798 | 0.976 / 0.766 | 2.158e-03 | 1.251e-05 | 0.312 | -2.435 | 130.7 | nan / nan |
| P35_vcs_a5_views4_800ep_seed1 | 800 | own | vcs_qmi | 0.9726 | 0.1178 | 0.0274 | 0.098 / 0.048 | 0.482 / 0.844 | 0.978 / 0.774 | 2.067e-03 | 1.163e-05 | 0.323 | -2.454 | 145.5 | nan / nan |
| P87_kcs_bw0.5_views4_800ep_seed0 | 100 | own | cs_kernel_native | 1.1353 | nan | nan | nan / nan | nan / nan | 0.282 / -0.005 | 6.072e-03 | 4.379e-04 | 0.209 | -0.774 | 9.5 | nan / nan |
| P87_kcs_bw0.5_views4_800ep_seed0 | 400 | own | cs_kernel_native | 1.5604 | nan | nan | nan / nan | nan / nan | 0.999 / -0.081 | 1.138e-02 | 6.306e-04 | 0.397 | -2.572 | 13.0 | nan / nan |
| P87_kcs_bw0.5_views4_800ep_seed0 | 800 | own | cs_kernel_native | 1.5527 | nan | nan | nan / nan | nan / nan | 0.999 / -0.080 | 1.706e-02 | 8.847e-04 | 0.437 | -2.662 | 10.3 | nan / nan |
| P87_kcs_bw0.5_views4_800ep_seed1 | 100 | own | cs_kernel_native | 1.2064 | nan | nan | nan / nan | nan / nan | 0.518 / -0.081 | 9.313e-03 | 1.285e-03 | 0.270 | -1.376 | 11.2 | nan / nan |
| P87_kcs_bw0.5_views4_800ep_seed1 | 400 | own | cs_kernel_native | 1.5299 | nan | nan | nan / nan | nan / nan | 0.999 / -0.089 | 1.354e-02 | 5.374e-04 | 0.386 | -2.567 | 14.6 | nan / nan |
| P87_kcs_bw0.5_views4_800ep_seed1 | 800 | own | cs_kernel_native | 1.5342 | nan | nan | nan / nan | nan / nan | 1.000 / -0.089 | 1.509e-02 | 7.082e-04 | 0.416 | -2.645 | 14.7 | nan / nan |
| P87_skernel_m4096_bw0.5_views4_800ep_seed0 | 100 | own | vcs_qmi | 0.8773 | 0.2622 | 0.1227 | 0.211 / 0.173 | 0.034 / 0.651 | 0.990 / 0.844 | 4.114e-03 | 9.691e-05 | 0.174 | -2.242 | 30.7 | nan / nan |
| P87_skernel_m4096_bw0.5_views4_800ep_seed0 | 400 | own | vcs_qmi | 0.9064 | 0.2066 | 0.0936 | 0.188 / 0.112 | 0.076 / 0.773 | 0.993 / 0.858 | 4.694e-03 | 9.317e-05 | 0.161 | -2.523 | 29.3 | nan / nan |
| P87_skernel_m4096_bw0.5_views4_800ep_seed0 | 800 | own | vcs_qmi | 0.9132 | 0.1895 | 0.0868 | 0.178 / 0.098 | 0.080 / 0.810 | 0.993 / 0.844 | 4.982e-03 | 1.166e-04 | 0.121 | -2.285 | 11.7 | nan / nan |
| P87_skernel_m4096_bw0.5_views4_800ep_seed1 | 100 | own | vcs_qmi | 0.8696 | 0.2787 | 0.1304 | 0.218 / 0.192 | 0.033 / 0.608 | 0.989 / 0.831 | 4.309e-03 | 1.016e-04 | 0.181 | -2.285 | 29.0 | nan / nan |
| P87_skernel_m4096_bw0.5_views4_800ep_seed1 | 400 | own | vcs_qmi | 0.9102 | 0.1931 | 0.0898 | 0.164 / 0.119 | 0.116 / 0.754 | 0.994 / 0.881 | 4.198e-03 | 5.526e-05 | 0.174 | -2.584 | 41.2 | nan / nan |
| P87_skernel_m4096_bw0.5_views4_800ep_seed1 | 800 | own | vcs_qmi | 0.9213 | 0.1720 | 0.0787 | 0.161 / 0.090 | 0.127 / 0.822 | 0.994 / 0.867 | 4.356e-03 | 7.391e-05 | 0.167 | -2.578 | 39.4 | nan / nan |
