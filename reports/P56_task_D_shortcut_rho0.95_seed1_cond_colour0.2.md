# Pre-check D — independence-test power on planted nuisances (task_d_shortcut_rho0.95_seed1, model.pt) — 2026-09-27T17:57:13Z

Pool: 5000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_colour0.2 | 0.2 | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.1293 ± 0.0445 | 0.419 | 0.636 |
| cond_colour0.2 (unconditional test) | | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
| cond_colour0.2 | 0.2 | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.1734 ± 0.0215 | 0.265 | 0.666 |
| cond_colour0.2 (unconditional test) | | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
| cond_colour0.2 | 0.2 | 1000 | 50 | 0.62 | 0.04 | 0.42 | 0.66 | 1.00 | 1.00 | 1.00 | 0.1905 ± 0.0143 | 0.187 | 0.683 |
| cond_colour0.2 (unconditional test) | | 1000 | 50 | 0.50 | 0.00 | 0.48 | 0.46 | 1.00 | 1.00 | 1.00 | | | |
