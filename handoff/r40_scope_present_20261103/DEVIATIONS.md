Publication copy (sample names aliased; numbers and conclusions unchanged).

# R40 implementation notes and registered choices (numbers-only line; no TEST read)

Fixed before any formal unit (see PROTOCOL_DIFF_VS_R39.json for the full list):
1. Frozen modality in IMAGE_ONLY / TEXT_ONLY is still re-encoded through the same student forward as R39 N-JOINT (parent weights,
   requires_grad False, outside the optimizer and the clip); no cache substitution. Its whole state is in frozen_state() and is asserted
   bit-unchanged at update 1 and at the end of the unit.
2. One-step check (all units, update 1, i.e. the first update with a nonzero critic output): every declared-active tensor must change unless
   its update-1 gradient is exactly zero or its predicted AdamW step (from the post-update optimizer state) is below the fp32 spacing of every
   element (ONE_STEP_CHECK_R40.json, per-tensor values). The R39 rule (every trainable tensor changed vs the parent) is recorded only.
   The comparison is against the state right before update 1 (R39 compared to the parent, i.e. updates 0+1).
3. Direction-sum gradient check tolerance 1e-5 relative (R39X amendment A1) adopted up front.
4. N_PRESENT_ONLY uses the existing R37 production path (data36.valid_rows / presence_probs, objectives.loss_list_masked), checked against
   the package reference (CPU_TEST reference_vs_prod) and with the R39 cosine schedule (R37's constant-schedule PRESENT units are not reused).
5. PRESENT_frozen (head-only) jobs use the R39 frozen-job resources (64G / 2h); JOINT jobs use the R39 JOINT resources (Flickr 64G/8h,
   CC 150G/10h). All formal jobs: normal QOS, nice 100, RTX6000PRO/L40S/H100 excl. node51/53/60, self-resume <= 3.
6. Trainer-role guard additionally refuses the R39 oracle/audit outputs and OUT/diag7/oracle.

CPU check history: 1083445 ALL_PASS (13/13, real data). Earlier: 1083427 (10/13 pass; the 3 end-to-end runs failed in r40ext.predicted_step on a 0-d parameter - np.spacing returned a
numpy scalar; fixed with np.asarray before any GPU job; record kept as CPU_TEST_1083427_failed.json).
