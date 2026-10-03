# P121 — condition table (finite latent model; exact float64)

| condition | seed | gamma | S | max |f*| | kernel identity rel err | nested max rel residual | J(oracle free pair) | J(const 0) |
|---|---|---|---|---|---|---|---|---|
| core_seed73 | 73 | 1.0 | 0.01870 | 0.508 | 4.8e-16 | 6.2e-16 | 0.01870 | 0.0 |
| weak | 11 | 0.5 | 0.00062 | 0.129 | 4.3e-16 | 6.9e-16 | 0.00062 | 0.0 |
| moderate | 11 | 2.0 | 0.04293 | 1.109 | 6.0e-16 | 5.5e-16 | 0.04293 | 0.0 |
| strong | 11 | 4.0 | 0.15967 | 2.843 | 4.5e-16 | 2.4e-16 | 0.15967 | 0.0 |

## Gram target G* = kappa + log(P/Q)/(2a)

| condition | a | kappa | sym err | diag err (max |G*_ii − 1|) | range excess | neg-eig mass | frac neg eigs | min / max |
|---|---|---|---|---|---|---|---|---|
| core_seed73 | 2.0 | 0.5 | 1.1e-16 | 0.472 | 0.000 | 0.129 | 0.42 | 0.246 / 0.627 |
| core_seed73 | 1.0 | 0.5 | 1.1e-16 | 0.445 | 0.000 | 0.260 | 0.42 | -0.008 / 0.754 |
| core_seed73 | 3.0 | 0.5 | 1.1e-16 | 0.482 | 0.000 | 0.085 | 0.42 | 0.331 / 0.585 |
| core_seed73 | 2.0 | 0.25 | 5.6e-17 | 0.722 | 0.000 | 0.130 | 0.42 | -0.004 / 0.377 |
| core_seed73 | 2.0 | 0.75 | 1.1e-16 | 0.222 | 0.000 | 0.128 | 0.42 | 0.496 / 0.877 |
| weak | 2.0 | 0.5 | 1.1e-16 | 0.496 | 0.000 | 0.008 | 0.50 | 0.458 / 0.564 |
| weak | 1.0 | 0.5 | 1.1e-16 | 0.491 | 0.000 | 0.015 | 0.50 | 0.416 / 0.629 |
| weak | 3.0 | 0.5 | 1.1e-16 | 0.497 | 0.000 | 0.005 | 0.50 | 0.472 / 0.543 |
| weak | 2.0 | 0.25 | 5.6e-17 | 0.746 | 0.000 | 0.008 | 0.50 | 0.208 / 0.314 |
| weak | 2.0 | 0.75 | 1.1e-16 | 0.246 | 0.000 | 0.008 | 0.50 | 0.708 / 0.814 |
| moderate | 2.0 | 0.5 | 1.1e-16 | 0.449 | 0.000 | 0.475 | 0.42 | -0.055 / 0.799 |
| moderate | 1.0 | 0.5 | 1.7e-16 | 0.397 | 0.098 | 0.990 | 0.42 | -0.609 / 1.098 |
| moderate | 3.0 | 0.5 | 1.1e-16 | 0.466 | 0.000 | 0.313 | 0.42 | 0.130 / 0.699 |
| moderate | 2.0 | 0.25 | 8.3e-17 | 0.699 | 0.000 | 0.495 | 0.42 | -0.305 / 0.549 |
| moderate | 2.0 | 0.75 | 1.1e-16 | 0.199 | 0.049 | 0.469 | 0.42 | 0.195 / 1.049 |
| strong | 2.0 | 0.5 | 1.1e-16 | 0.401 | 0.112 | 1.944 | 0.42 | -0.922 / 1.112 |
| strong | 1.0 | 0.5 | 2.2e-16 | 0.724 | 1.343 | 4.300 | 0.42 | -2.343 / 1.724 |
| strong | 3.0 | 0.5 | 1.1e-16 | 0.434 | 0.000 | 1.249 | 0.42 | -0.448 / 0.908 |
| strong | 2.0 | 0.25 | 1.1e-16 | 0.651 | 0.172 | 2.150 | 0.42 | -1.172 / 0.862 |
| strong | 2.0 | 0.75 | 2.2e-16 | 0.362 | 0.362 | 1.873 | 0.42 | -0.672 / 1.362 |

## Unit-vector fits (all starts; J / S; not an optimality certificate; J(const 0) = 0, J(free oracle) = S)

shared = one unit vector per state on both sides (unit diagonal G_uu = 1, a finite-model constraint); two_tower = separate unit vectors per side.

| condition | geometry | a | kappa | d | J/S per start | best J/S | Lipschitz bound holds (all starts) |
|---|---|---|---|---|---|---|---|
| core_seed73 | shared_unit_gram | 1.0 | 0.5 | 2 | -2.984, -2.984, -2.984 | -2.984 | True |
| core_seed73 | shared_unit_gram | 1.0 | 0.5 | 4 | -0.596, -0.598, -0.581 | -0.581 | True |
| core_seed73 | shared_unit_gram | 1.0 | 0.5 | 8 | 0.272, 0.272, 0.272 | 0.272 | True |
| core_seed73 | shared_unit_gram | 2.0 | 0.25 | 2 | -24.295, -23.708, -20.196 | -20.196 | True |
| core_seed73 | shared_unit_gram | 2.0 | 0.25 | 4 | -10.260, -8.084, -8.548 | -8.084 | True |
| core_seed73 | shared_unit_gram | 2.0 | 0.25 | 8 | -4.371, -3.427, -3.658 | -3.427 | True |
| core_seed73 | shared_unit_gram | 2.0 | 0.5 | 2 | -16.199, -17.757, -19.357 | -16.199 | True |
| core_seed73 | shared_unit_gram | 2.0 | 0.5 | 4 | -5.087, -5.554, -5.600 | -5.087 | True |
| core_seed73 | shared_unit_gram | 2.0 | 0.5 | 8 | -1.905, -1.911, -1.968 | -1.905 | True |
| core_seed73 | shared_unit_gram | 2.0 | 0.75 | 2 | -3.057, -5.450, -5.861 | -3.057 | True |
| core_seed73 | shared_unit_gram | 2.0 | 0.75 | 4 | -0.551, -0.600, -0.551 | -0.551 | True |
| core_seed73 | shared_unit_gram | 2.0 | 0.75 | 8 | 0.272, 0.272, 0.272 | 0.272 | True |
| core_seed73 | shared_unit_gram | 3.0 | 0.5 | 2 | -23.506, -31.964, -32.928 | -23.506 | True |
| core_seed73 | shared_unit_gram | 3.0 | 0.5 | 4 | -9.379, -9.030, -10.188 | -9.030 | True |
| core_seed73 | shared_unit_gram | 3.0 | 0.5 | 8 | -3.650, -3.686, -4.044 | -3.650 | True |
| core_seed73 | two_tower_cross_gram | 1.0 | 0.5 | 2 | -4.674, -11.995, -2.925 | -2.925 | True |
| core_seed73 | two_tower_cross_gram | 1.0 | 0.5 | 4 | 0.988, 0.988, 0.941 | 0.988 | True |
| core_seed73 | two_tower_cross_gram | 1.0 | 0.5 | 8 | 0.999, 1.000, 0.999 | 1.000 | True |
| core_seed73 | two_tower_cross_gram | 2.0 | 0.25 | 2 | -3.225, -4.551, -4.994 | -3.225 | True |
| core_seed73 | two_tower_cross_gram | 2.0 | 0.25 | 4 | 0.957, 0.935, 0.945 | 0.957 | True |
| core_seed73 | two_tower_cross_gram | 2.0 | 0.25 | 8 | 0.999, 0.999, 0.999 | 0.999 | True |
| core_seed73 | two_tower_cross_gram | 2.0 | 0.5 | 2 | -12.381, -15.643, -4.608 | -4.608 | True |
| core_seed73 | two_tower_cross_gram | 2.0 | 0.5 | 4 | 0.948, 0.966, 0.990 | 0.990 | True |
| core_seed73 | two_tower_cross_gram | 2.0 | 0.5 | 8 | 0.999, 0.998, 0.999 | 0.999 | True |
| core_seed73 | two_tower_cross_gram | 2.0 | 0.75 | 2 | -8.089, -2.068, -9.101 | -2.068 | True |
| core_seed73 | two_tower_cross_gram | 2.0 | 0.75 | 4 | 0.984, 0.990, 0.989 | 0.990 | True |
| core_seed73 | two_tower_cross_gram | 2.0 | 0.75 | 8 | 0.999, 0.999, 1.000 | 1.000 | True |
| core_seed73 | two_tower_cross_gram | 3.0 | 0.5 | 2 | -16.262, -17.051, -12.643 | -12.643 | True |
| core_seed73 | two_tower_cross_gram | 3.0 | 0.5 | 4 | 0.987, 0.947, 0.988 | 0.988 | True |
| core_seed73 | two_tower_cross_gram | 3.0 | 0.5 | 8 | 0.999, 0.998, 0.998 | 0.999 | True |
| moderate | shared_unit_gram | 1.0 | 0.5 | 2 | -0.556, -0.556, -0.556 | -0.556 | True |
| moderate | shared_unit_gram | 1.0 | 0.5 | 4 | 0.480, 0.493, 0.480 | 0.493 | True |
| moderate | shared_unit_gram | 1.0 | 0.5 | 8 | 0.699, 0.699, 0.699 | 0.699 | True |
| moderate | shared_unit_gram | 2.0 | 0.25 | 2 | -9.421, -10.153, -7.885 | -7.885 | True |
| moderate | shared_unit_gram | 2.0 | 0.25 | 4 | -4.248, -2.750, -3.075 | -2.750 | True |
| moderate | shared_unit_gram | 2.0 | 0.25 | 8 | -0.849, -0.970, -0.861 | -0.849 | True |
| moderate | shared_unit_gram | 2.0 | 0.5 | 2 | -5.946, -7.015, -6.963 | -5.946 | True |
| moderate | shared_unit_gram | 2.0 | 0.5 | 4 | -1.242, -1.298, -1.507 | -1.242 | True |
| moderate | shared_unit_gram | 2.0 | 0.5 | 8 | -0.165, -0.196, -0.156 | -0.156 | True |
| moderate | shared_unit_gram | 2.0 | 0.75 | 2 | -0.576, -0.576, -0.576 | -0.576 | True |
| moderate | shared_unit_gram | 2.0 | 0.75 | 4 | 0.484, 0.495, 0.484 | 0.495 | True |
| moderate | shared_unit_gram | 2.0 | 0.75 | 8 | 0.704, 0.704, 0.704 | 0.704 | True |
| moderate | shared_unit_gram | 3.0 | 0.5 | 2 | -10.569, -10.541, -14.471 | -10.541 | True |
| moderate | shared_unit_gram | 3.0 | 0.5 | 4 | -3.098, -3.913, -2.917 | -2.917 | True |
| moderate | shared_unit_gram | 3.0 | 0.5 | 8 | -0.892, -0.855, -0.937 | -0.855 | True |
| moderate | two_tower_cross_gram | 1.0 | 0.5 | 2 | -1.286, -6.961, -1.522 | -1.286 | True |
| moderate | two_tower_cross_gram | 1.0 | 0.5 | 4 | 0.967, 0.964, 0.941 | 0.967 | True |
| moderate | two_tower_cross_gram | 1.0 | 0.5 | 8 | 0.993, 0.993, 0.993 | 0.993 | True |
| moderate | two_tower_cross_gram | 2.0 | 0.25 | 2 | -0.733, -3.262, -2.061 | -0.733 | True |
| moderate | two_tower_cross_gram | 2.0 | 0.25 | 4 | 0.936, 0.961, 0.963 | 0.963 | True |
| moderate | two_tower_cross_gram | 2.0 | 0.25 | 8 | 0.999, 0.999, 0.998 | 0.999 | True |
| moderate | two_tower_cross_gram | 2.0 | 0.5 | 2 | -4.635, -6.781, -1.864 | -1.864 | True |
| moderate | two_tower_cross_gram | 2.0 | 0.5 | 4 | 0.923, 0.964, 0.963 | 0.964 | True |
| moderate | two_tower_cross_gram | 2.0 | 0.5 | 8 | 0.998, 0.999, 0.999 | 0.999 | True |
| moderate | two_tower_cross_gram | 2.0 | 0.75 | 2 | -12.005, -3.089, -2.235 | -2.235 | True |
| moderate | two_tower_cross_gram | 2.0 | 0.75 | 4 | 0.969, 0.938, 0.965 | 0.969 | True |
| moderate | two_tower_cross_gram | 2.0 | 0.75 | 8 | 0.995, 0.995, 0.995 | 0.995 | True |
| moderate | two_tower_cross_gram | 3.0 | 0.5 | 2 | -7.176, -7.958, -7.299 | -7.176 | True |
| moderate | two_tower_cross_gram | 3.0 | 0.5 | 4 | 0.877, 0.965, 0.391 | 0.965 | True |
| moderate | two_tower_cross_gram | 3.0 | 0.5 | 8 | 0.999, 0.999, 0.999 | 0.999 | True |
| strong | shared_unit_gram | 1.0 | 0.5 | 2 | 0.728, 0.728, 0.728 | 0.728 | True |
| strong | shared_unit_gram | 1.0 | 0.5 | 4 | 0.901, 0.901, 0.897 | 0.901 | True |
| strong | shared_unit_gram | 1.0 | 0.5 | 8 | 0.904, 0.904, 0.904 | 0.904 | True |
| strong | shared_unit_gram | 2.0 | 0.25 | 2 | -1.008, -0.720, -0.433 | -0.433 | True |
| strong | shared_unit_gram | 2.0 | 0.25 | 4 | 0.297, 0.322, 0.335 | 0.335 | True |
| strong | shared_unit_gram | 2.0 | 0.25 | 8 | 0.461, 0.461, 0.461 | 0.461 | True |
| strong | shared_unit_gram | 2.0 | 0.5 | 2 | -0.520, 0.113, 0.113 | 0.113 | True |
| strong | shared_unit_gram | 2.0 | 0.5 | 4 | 0.604, 0.607, 0.611 | 0.611 | True |
| strong | shared_unit_gram | 2.0 | 0.5 | 8 | 0.674, 0.674, 0.674 | 0.674 | True |
| strong | shared_unit_gram | 2.0 | 0.75 | 2 | 0.319, 0.315, -1.453 | 0.319 | True |
| strong | shared_unit_gram | 2.0 | 0.75 | 4 | 0.902, 0.902, 0.906 | 0.906 | True |
| strong | shared_unit_gram | 2.0 | 0.75 | 8 | 0.909, 0.909, 0.909 | 0.909 | True |
| strong | shared_unit_gram | 3.0 | 0.5 | 2 | -0.740, -1.042, -2.966 | -0.740 | True |
| strong | shared_unit_gram | 3.0 | 0.5 | 4 | 0.326, 0.335, 0.310 | 0.335 | True |
| strong | shared_unit_gram | 3.0 | 0.5 | 8 | 0.462, 0.462, 0.462 | 0.462 | True |
| strong | two_tower_cross_gram | 1.0 | 0.5 | 2 | 0.634, 0.778, -0.270 | 0.778 | True |
| strong | two_tower_cross_gram | 1.0 | 0.5 | 4 | 0.951, 0.951, 0.953 | 0.953 | True |
| strong | two_tower_cross_gram | 1.0 | 0.5 | 8 | 0.963, 0.963, 0.963 | 0.963 | True |
| strong | two_tower_cross_gram | 2.0 | 0.25 | 2 | -0.203, 0.213, -0.109 | 0.213 | True |
| strong | two_tower_cross_gram | 2.0 | 0.25 | 4 | 0.972, 0.971, 0.972 | 0.972 | True |
| strong | two_tower_cross_gram | 2.0 | 0.25 | 8 | 0.999, 0.999, 0.999 | 0.999 | True |
| strong | two_tower_cross_gram | 2.0 | 0.5 | 2 | -0.412, -1.461, 0.502 | 0.502 | True |
| strong | two_tower_cross_gram | 2.0 | 0.5 | 4 | 0.966, 0.974, 0.965 | 0.974 | True |
| strong | two_tower_cross_gram | 2.0 | 0.5 | 8 | 0.997, 0.998, 0.998 | 0.998 | True |
| strong | two_tower_cross_gram | 2.0 | 0.75 | 2 | 0.694, 0.769, -0.136 | 0.769 | True |
| strong | two_tower_cross_gram | 2.0 | 0.75 | 4 | 0.964, 0.958, 0.964 | 0.964 | True |
| strong | two_tower_cross_gram | 2.0 | 0.75 | 8 | 0.969, 0.969, 0.969 | 0.969 | True |
| strong | two_tower_cross_gram | 3.0 | 0.5 | 2 | -1.739, -2.474, -1.499 | -1.499 | True |
| strong | two_tower_cross_gram | 3.0 | 0.5 | 4 | 0.971, 0.971, 0.969 | 0.971 | True |
| strong | two_tower_cross_gram | 3.0 | 0.5 | 8 | 0.999, 0.999, 0.999 | 0.999 | True |
| weak | shared_unit_gram | 1.0 | 0.5 | 2 | -173.492, -167.321, -178.396 | -167.321 | True |
| weak | shared_unit_gram | 1.0 | 0.5 | 4 | -79.671, -79.390, -81.099 | -79.390 | True |
| weak | shared_unit_gram | 1.0 | 0.5 | 8 | -38.298, -38.425, -38.300 | -38.298 | True |
| weak | shared_unit_gram | 2.0 | 0.25 | 2 | -841.074, -832.034, -791.466 | -791.466 | True |
| weak | shared_unit_gram | 2.0 | 0.25 | 4 | -383.977, -384.209, -375.660 | -375.660 | True |
| weak | shared_unit_gram | 2.0 | 0.25 | 8 | -176.448, -179.433, -182.548 | -176.448 | True |
| weak | shared_unit_gram | 2.0 | 0.5 | 2 | -481.601, -512.335, -540.535 | -481.601 | True |
| weak | shared_unit_gram | 2.0 | 0.5 | 4 | -235.352, -246.454, -230.010 | -230.010 | True |
| weak | shared_unit_gram | 2.0 | 0.5 | 8 | -117.761, -118.179, -120.593 | -117.761 | True |
| weak | shared_unit_gram | 2.0 | 0.75 | 2 | -166.455, -170.784, -171.414 | -166.455 | True |
| weak | shared_unit_gram | 2.0 | 0.75 | 4 | -80.561, -79.233, -79.469 | -79.233 | True |
| weak | shared_unit_gram | 2.0 | 0.75 | 8 | -38.410, -38.275, -38.274 | -38.274 | True |
| weak | shared_unit_gram | 3.0 | 0.5 | 2 | -1024.990, -1012.822, -1024.769 | -1012.822 | True |
| weak | shared_unit_gram | 3.0 | 0.5 | 4 | -403.580, -374.221, -358.559 | -358.559 | True |
| weak | shared_unit_gram | 3.0 | 0.5 | 8 | -196.922, -171.645, -179.707 | -171.645 | True |
| weak | two_tower_cross_gram | 1.0 | 0.5 | 2 | -160.563, -181.748, -196.210 | -160.563 | True |
| weak | two_tower_cross_gram | 1.0 | 0.5 | 4 | 0.932, 0.903, 0.901 | 0.932 | True |
| weak | two_tower_cross_gram | 1.0 | 0.5 | 8 | 0.996, 0.999, 0.998 | 0.999 | True |
| weak | two_tower_cross_gram | 2.0 | 0.25 | 2 | -113.203, -166.519, -163.696 | -113.203 | True |
| weak | two_tower_cross_gram | 2.0 | 0.25 | 4 | 0.918, 0.943, 0.897 | 0.943 | True |
| weak | two_tower_cross_gram | 2.0 | 0.25 | 8 | 0.991, 0.989, 0.989 | 0.991 | True |
| weak | two_tower_cross_gram | 2.0 | 0.5 | 2 | -371.319, -408.030, -170.804 | -170.804 | True |
| weak | two_tower_cross_gram | 2.0 | 0.5 | 4 | 0.844, 0.913, 0.908 | 0.913 | True |
| weak | two_tower_cross_gram | 2.0 | 0.5 | 8 | 0.989, 0.996, 0.994 | 0.996 | True |
| weak | two_tower_cross_gram | 2.0 | 0.75 | 2 | -818.008, -41.214, -265.492 | -41.214 | True |
| weak | two_tower_cross_gram | 2.0 | 0.75 | 4 | 0.812, 0.879, 0.918 | 0.918 | True |
| weak | two_tower_cross_gram | 2.0 | 0.75 | 8 | 0.994, 0.996, 0.994 | 0.996 | True |
| weak | two_tower_cross_gram | 3.0 | 0.5 | 2 | -521.866, -543.687, -532.964 | -521.866 | True |
| weak | two_tower_cross_gram | 3.0 | 0.5 | 4 | 0.848, 0.890, 0.916 | 0.916 | True |
| weak | two_tower_cross_gram | 3.0 | 0.5 | 8 | 0.998, 0.996, 0.995 | 0.998 | True |
