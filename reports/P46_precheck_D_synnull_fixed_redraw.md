# Pre-check D — synthetic null level check, mode `fixed_redraw` — 2026-09-28T04:26:21Z

Gaussian z (dim 512), N ~ Bernoulli(1/2) independent; δ = 0.05; 200 permutations; critics 300 steps; one pool of 45000 items, N drawn once, re-drawn per repeat.

| n | R | 95 % band (exact test) | vcs_hoeff | hoeff lin | hoeff mlp | hoeff closed | vcs_perm | hsic_perm | c2st |
|---|---|---|---|---|---|---|---|---|---|
| 2000 | 1000 | 0.036–0.064 | 0.000 | 0.000 | 0.000 | 0.000 | 0.053 | 0.056 | 0.000 |
| 5000 | 1000 | 0.036–0.064 | 0.000 | 0.000 | 0.000 | 0.000 | 0.061 | 0.048 | 0.000 |
| 10000 | 1000 | 0.036–0.064 | 0.000 | 0.000 | 0.000 | 0.000 | 0.052 | 0.000 | 0.002 |
