# Pre-check D — synthetic null level check, mode `fixed` — 2026-09-28T02:23:09Z

Gaussian z (dim 512), N ~ Bernoulli(1/2) independent; δ = 0.05; 200 permutations; critics 300 steps; one pool of 45000 items, N drawn once.

| n | R | 95 % band (exact test) | vcs_hoeff | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | 1000 | 0.036–0.064 | 0.000 | 0.000 | 0.000 | 0.000 | 0.059 | 0.059 | 0.001 |
| 5000 | 1000 | 0.036–0.064 | 0.000 | 0.000 | 0.000 | 0.000 | 0.063 | 0.062 | 0.000 |
| 10000 | 1000 | 0.036–0.064 | 0.000 | 0.000 | 0.000 | 0.000 | 0.084 | 0.000 | 0.000 |
