# Pre-check D — independence-test power on planted nuisances (task_d_shortcut_rho0.5_seed0, model.pt) — 2026-09-27T15:06:34Z

Pool: 5000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_colour0.2 | 0.2 | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.72 | 0.30 | 0.10 | -0.0135 ± 0.0354 | 0.419 | 0.516 |
| cond_colour0.2 (unconditional test) | | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.80 | 0.18 | 0.06 | | | |
| cond_colour0.2 | 0.2 | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.98 | 0.62 | 0.40 | 0.0134 ± 0.0183 | 0.265 | 0.520 |
| cond_colour0.2 (unconditional test) | | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.28 | 0.24 | | | |
| cond_colour0.2 | 0.2 | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.96 | 0.78 | 0.0271 ± 0.0122 | 0.187 | 0.542 |
| cond_colour0.2 (unconditional test) | | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.84 | 0.82 | | | |
