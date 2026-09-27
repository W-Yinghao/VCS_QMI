# Pre-check D — independence-test power on planted nuisances (P5_simclr_seed0, epoch_200.pt) — 2026-09-27T18:42:51Z

Pool: 45000 fit images; per instance three disjoint samples of n (fit / eval / independent pool); level δ = 0.05; permutations 200; critic/classifier: MLP(512+2→128→128→1), 300 Adam steps; repeats 100 (n ≥ 2000: 30).

| case | s | n | R | vcs_hoeff (val-picked) | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st | mean J_eval (picked) | τ | c2st acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s1 | 1 | 100 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.96 | 1.00 | 0.27 | -0.0042 ± 0.0808 | 0.592 | 0.539 |
| s1 | 1 | 200 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.83 | 0.0397 ± 0.0548 | 0.419 | 0.572 |
| s1 | 1 | 500 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.99 | 0.1150 ± 0.0281 | 0.265 | 0.616 |
| s1 | 1 | 1000 | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.1452 ± 0.0171 | 0.187 | 0.645 |
| s1 | 1 | 2000 | 30 | 1.00 | 0.40 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.1757 ± 0.0123 | 0.132 | 0.666 |
| s1 | 1 | 5000 | 30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.1996 ± 0.0074 | 0.084 | 0.682 |
