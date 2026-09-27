# Pre-check D — independence-test power on planted nuisances (task_d_shortcut_rho0.8_seed1, model.pt) — 2026-09-27T15:15:33Z

Pool: 5000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_colour0.2 | 0.2 | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.92 | 0.28 | 0.0238 ± 0.0383 | 0.419 | 0.529 |
| cond_colour0.2 (unconditional test) | | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.98 | 0.72 | 0.24 | | | |
| cond_colour0.2 | 0.2 | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.96 | 0.0595 ± 0.0291 | 0.265 | 0.569 |
| cond_colour0.2 (unconditional test) | | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.90 | | | |
| cond_colour0.2 | 0.2 | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0818 ± 0.0154 | 0.187 | 0.609 |
| cond_colour0.2 (unconditional test) | | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.96 | | | |
