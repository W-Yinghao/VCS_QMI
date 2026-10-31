"""R27-Q hyperparameters (one set for all arms). Values marked TRIAL are fixed by the real-data technical trial
(gate/r27q_trial.py, rules in TRIAL_RULES) and then frozen; FROZEN must be True before any formal (non-smoke) unit.
The same values are copied into PROTOCOL_FROZEN.json."""
FROZEN = True          # frozen 2026-10-31 from TRIAL_1078146.json (rules TRIAL_RULES, applied once, all arms)

# relation head (Q2-Q4 identical): g(q, y) = l2(relu(l1([q*y, |q-y|, q.y])))
HIDDEN = 256
FEATURES = 'concat(q*y, |q-y|, q.y) of unit teacher vectors (2*1024+1 = 2049 inputs)'
# P / Q (per peer stage, teacher = the other peer, frozen)
N_FIT = 16384          # max high-confidence pairs used to fit (multiple of CHUNK)
N_VAL = 2048           # max held-out high-confidence pairs (diagnostics only; multiple of CHUNK)
CHUNK = 128            # teacher encoding / centering batch (= host batch size)
# optimisation of the head (same for Q2-Q4)
BATCH = 128            # positive pairs per step; Q grid = BATCH x BATCH independent draws from the P marginals
LR = 1e-3
STEPS = 200            # TRIAL 1078146: rule TRIAL_RULES['steps'] -> 200
# score -> alpha readout: alpha = softmax(beta * score) over the K candidates of a row
BETA_HEAD = 2.0        # TRIAL 1078146: rule TRIAL_RULES['beta'] -> 2.0 (one shared value for Q2/Q3/Q4: pre-link g)
BETA_COS = 16.0        # TRIAL 1078146: same rule -> 16.0 (Q1: teacher cosine)
BETA_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
STEPS_GRID = (100, 200, 400, 800)
SEED_BASE = 7919       # stage seed = SEED_BASE * opt.seed + 2 * epoch + peer (identical for all arms)

TRIAL_RULES = dict(
    steps='smallest S in STEPS_GRID such that, for EACH of the three heads, the held-out improvement of its own objective '
          '(loss at step 0 minus loss at S) is >= 0.95 x its best improvement over STEPS_GRID; one S for all heads',
    beta='temperature scaling on held-out high-confidence pairs (training-visible observed pairing only): each '
         'held-out caption gets K=5 candidates = its observed image + the 4 other distinct observed images of the '
         'held-out pool with highest teacher cosine; NLL(beta) = -mean log softmax(beta*score)[observed]; '
         'BETA_HEAD = argmin over BETA_GRID of the SUM of the Q2/Q3/Q4 NLLs (one shared value); '
         'BETA_COS = argmin over BETA_GRID of the Q1 cosine NLL',
    data='gate start state, stage A of the next epoch (teacher = model B), GMM split recomputed as in training',
)
