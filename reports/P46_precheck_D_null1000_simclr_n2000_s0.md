# Pre-check D — independence-test power on planted nuisances (P5_simclr_seed0, epoch_200.pt) — 2026-09-27T14:53:26Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 1000).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 0 | 2000 | 1000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.05 | 0.05 | 0.01 | -0.0015 ± 0.0022 | 0.132 | 0.501 |
