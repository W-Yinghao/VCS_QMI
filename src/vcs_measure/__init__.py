"""P109 (package v4 modules X1, I1, I2): measurements on FROZEN representations — no SSL training happens here.

    common      run loading (read-only), layer features (layer3 pooled, h, z), split helpers
    xmeasure    X1 independent measurement-critic family on two-view pairs (calibrated cosine, pair MLP, closed-form product ridge, RFF ridge, T = 0)
    audit       I1 paired nuisance audit: presence / per-layer retention (shared-permutation max-statistic) / paired prediction effects
    nested      I2 pilot: nested fine-set critics (exact coarse branch) and exact enumeration of the binary-N product term
"""
