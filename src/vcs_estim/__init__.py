"""vcs_estim — controlled estimator experiments of the VCS-QMI programme.

Package v1 / v2 modules (spec §3–§9): `objectives` (J, R, matched JS, posterior MSE), `synthetic` (Gaussian generator with roles, padded
irrelevant dimensions), `candidates` (C0 / C1 / C2 / CQ), `fitting`, `evaluation` (metrics, channel gradients), `convex_mix` + `bounded_core`
(simplex dictionary, residual step, JS mixture), `frozen` (SSL-checkpoint diagnostics), `pairing` (image identity roles), `observation`
(noise), `kernel_cs` (v2 §3 direct CS controls: CS-K-native, S-KDE, S-Kernel RFF / Nystrom, rLS, rLS-tanh, dictionary bound), `run` (probe).
E-line modules (next-round brief E / R1; P85): `data` (staircase settings with analytic PMI, R1 contaminations), `critics` (one joint MLP
for every estimator + oracle critic), `estimators` (VCS, JS, InfoNCE, NWJ, DV/MINE, SMILE on one interface; three negative constructions),
`oracle` (oracle-critic resolution table), `staircase` (cell runner).  Independent of vcs_ssl except for one equivalence test."""
