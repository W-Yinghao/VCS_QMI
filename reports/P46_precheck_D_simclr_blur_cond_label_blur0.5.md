# Pre-check D — independence-test power on planted nuisances (P5_simclr_seed0, epoch_200.pt) — 2026-09-27T18:45:22Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cond_label_blur0.5 | 0.5 | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.30 | 0.04 | 0.00 | -0.0047 ± 0.0074 | 0.265 | 0.503 |
| cond_label_blur0.5 (unconditional test) | | 500 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.98 | | | |
| cond_label_blur0.5 | 0.5 | 2000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.98 | 0.18 | 0.24 | 0.0014 ± 0.0027 | 0.132 | 0.508 |
| cond_label_blur0.5 (unconditional test) | | 2000 | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | | | |
