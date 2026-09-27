# Pre-check D — independence-test power on planted nuisances (task_d_shortcut_rho0.95_seed0, model.pt) — 2026-09-27T16:19:49Z

Pool: 5000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_colour0.2 | 0.2 | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.1245 ± 0.0469 | 0.419 | 0.635 |
| cond_colour0.2 (unconditional test) | | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
| cond_colour0.2 | 0.2 | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.1730 ± 0.0226 | 0.265 | 0.668 |
| cond_colour0.2 (unconditional test) | | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
| cond_colour0.2 | 0.2 | 1000 | 50 | 0.66 | 0.00 | 0.40 | 0.74 | 1.00 | 1.00 | 1.00 | 0.1924 ± 0.0170 | 0.187 | 0.682 |
| cond_colour0.2 (unconditional test) | | 1000 | 50 | 0.54 | 0.00 | 0.26 | 0.54 | 1.00 | 1.00 | 1.00 | | | |
