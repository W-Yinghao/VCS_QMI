"""R28 selective candidate correction: hyperparameters (one set for all four arms). Written before any R28 GPU run and
before any DEV result of an R28 arm; FROZEN must be True before any formal (non-smoke) unit. The same values are copied
into PROTOCOL_FROZEN.json. Relation-head architecture / P / Q / budget / readout scales are inherited from the frozen R27-Q
config (r27q_config) and asserted equal in the CPU checks; nothing here is chosen on DEV."""
FROZEN = True          # frozen 2026-10-31 before the R28 GPU gate (no R28 training result of any kind seen)

# ---------------------------------------------------------------- relation / residual heads (inherited, same budget as Q2)
HIDDEN = 256           # == r27q_config.HIDDEN (PairHead 2049 -> 256 -> 1, 525057 params)
STEPS = 200            # == r27q_config.STEPS (common frozen budget; no per-arm change)
LR = 1e-3              # == r27q_config.LR
BATCH = 128            # == r27q_config.BATCH
BETA_COS = 16.0        # == r27q_config.BETA_COS (Q1 readout softmax(16 s))
BETA_HEAD = 2.0        # == r27q_config.BETA_HEAD (Q2 readout softmax(2 g))
RESIDUAL_INIT = 'PairHead default init under torch.manual_seed(stage_seed+17) (= the Q2 head init), then the LAST layer ' \
                '(l2 weight and bias) set to zero -> r_phi == 0 at fit start'

# ---------------------------------------------------------------- cosine anchor g0 = a*s + b (R1 and R2 share it)
AFF_RULE = ('per peer stage, on the stage fit P (teacher coordinates, the same pairs that fit the residual) and the FULL '
            'empirical product of its two marginals incl. the diagonal (all n_fit x n_fit caption/image pairs): '
            'minimize the balanced logistic loss E_P softplus(-2 g0) + E_Q softplus(2 g0) over (a, b) (Platt-type scalar '
            'calibration of 0.5 log(p/q), convex; the diagonal is in both P and Q so the optimum is finite); damped Newton '
            'with Armijo backtracking, float64 reductions. a, b are then frozen and only the residual is fitted.')
AFF_CHUNK = 2048       # rows of the product grid per matmul chunk
AFF_A0, AFF_B0 = 1.0, 0.0
AFF_MAX_ITERS = 60
AFF_TOL = 1e-9         # stop when the Newton step (max abs over a, b) is below this
A_MIN = 1.0            # a below this (or non-finite / not converged) -> technical fallback (beta = 16/a would exceed 16)
AFF_FALLBACK = ('R1/R2 only: the residual is not fitted and r = 0 for the whole stage, so alpha_score = softmax(16 s) '
                '(exactly the G1 score readout); counted in rstage_aff_fallback. Same rule for both arms.')

# ---------------------------------------------------------------- support calibrator u (shared by all four arms)
U_FEATURES = ('best_cos', 'ab_overlap', 'ab_top1_agree')
U_FEATURE_DEF = dict(
    best_cos='max over the K candidates of the teacher cosine q.y (R27-Q teacher coordinates; the Q1 score)',
    ab_overlap='|A5 & B5| / 5: overlap of the two C_mix routes\' unique top-5 image-ID lists (A: query . teacher text bank '
               '-> partner observed image IDs; B: query . teacher image bank)',
    ab_top1_agree='1[A route top-1 ID == B route top-1 ID]')
N_CAL_FIT = 4096       # calibration-fit rows (multiple of 128)
N_CAL_REP = 2048       # separate report rows (multiple of 128), disjoint observed images
CAL_SEED_OFFSET = 31   # calibrator generator seed = stage_seed + 31 (independent CPU generator; global RNG untouched)
CAL_RULE = ('pool = stage P rows (teacher high-confidence labeled rows, observed pairing) NOT used by the head/residual '
            'fit or held-out sets; observed-image IDs of the pool shuffled (stage generator) and split in half -> fit / '
            'report groups (no observed image in both); rows of each group shuffled, first N_CAL_FIT / N_CAL_REP taken '
            '(rounded down to 128). Each row: query = its caption in teacher coordinates (unaugmented, centered by its '
            '128-row chunk mean); candidates = the ACTUAL C_mix rule (r27cs_sources.unique_topk + mix_3a_2b on the teacher '
            'queue) with the query\'s own queue record(s) (slot caption ID == query caption ID) excluded from BOTH routes '
            '(direct-copy channel; same for all arms); rows short of 5 unique IDs dropped. weak label = 1[observed image '
            'ID in the 5 candidates]; the observed item is never inserted. Logistic regression on U_FEATURES + intercept, '
            'ridge on the weights only, Newton/IRLS float64 to convergence.')
RIDGE = 1e-4
LOGIT_MAX_ITERS = 100
LOGIT_TOL = 1e-10
MIN_CLASS = 64         # each weak-label class needs >= MIN_CLASS calibration-fit rows, else the fallback
U_FALLBACK = ('intercept-only logistic: u = the calibration-fit weak-label rate (constant for the stage; 0 or 1 if one '
              'class is empty); used when the calibration-fit split has < MIN_CLASS rows of a class, or the Newton fit '
              'is non-finite / not converged. Counted in rstage_u_fallback. Same rule for all arms.')
ECE_BINS = 10

# ---------------------------------------------------------------- fixed hard-candidate diagnostic (training-visible)
HARD_RULE = ('each held-out P caption (the head held-out set): its observed image + the 4 other distinct observed images '
             'of the held-out pool with the highest teacher cosine (= the R27-Q trial calibration set); logged per stage: '
             'observed top-1 / mean rank / NLL under the arm score readout')
