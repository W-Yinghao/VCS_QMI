# Pre-check D — independence-test power on planted nuisances (task_d_shortcut_rho0.8_seed0, model.pt) — 2026-09-27T15:12:36Z

Pool: 5000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_colour0.2 | 0.2 | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.92 | 0.52 | 0.0305 ± 0.0431 | 0.419 | 0.544 |
| cond_colour0.2 (unconditional test) | | 200 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.88 | 0.42 | | | |
| cond_colour0.2 | 0.2 | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.96 | 0.0748 ± 0.0216 | 0.265 | 0.585 |
| cond_colour0.2 (unconditional test) | | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.94 | | | |
| cond_colour0.2 | 0.2 | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.0967 ± 0.0160 | 0.187 | 0.624 |
| cond_colour0.2 (unconditional test) | | 1000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
