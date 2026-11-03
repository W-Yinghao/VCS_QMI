Publication copy (sample names aliased; numbers and conclusions unchanged).

# R39 implementation notes and registered deviations (numbers-only line; no TEST read)

Registered compatibility patches (production interface, fixed before any formal unit):
1. JOINT centring: native `VSE.forward_emb` subtracts the per-batch mean (128 sequential rows of the native eval layout, per modality)
   before L2. A training batch cannot reproduce that 128-row batch, so the student uses the PARENT's centring vector of the item's
   native batch (stored per batch in prep, `center_<split>_<kind>_peer<P>.npy`), held fixed for FIT and DEV. Measured equivalence of
   student vs native cache at the parent weights: max |diff| 2.0e-5 (Flickr60), 7.0e-5 (CC152K); DEV native rSum recomputed from the
   student vectors equals the cache rSum exactly on both (JOINT_PREP_MANIFEST.json). In-unit tolerance 1e-4 (config joint_rule).
2. JOINT modes: BN running stats (eval), dropout off; the caption BiGRU and both GPO GRUs are in train mode only because cuDNN RNN
   backward requires it (one layer, dropout 0: same forward).
3. The native `vse.MLP` creates an output BatchNorm (`mlp.bns.1`) that its forward never applies; it has no gradient and is kept
   frozen (found by the first JOINT functional check, 1082795/1082796: 42/46 student tensors changed at update 1).
4. COMPLETE/oracle keep the N label dtype (data36.null_lists returns float64 rows, e.g. 1/3 for three observed positives); the first
   CPU post check (1082832) caught a float32 cast that changed unchanged rows by rounding; fixed before any unit.
5. CC152K raw region features (44 GB) are copied to host RAM per JOINT job; the CC functional check (1082821) ran on the NFS mmap and
   measured 14.1 s/update (Flickr60 on GPU: 0.24 s/update); loading line changed before freeze, training code path unchanged.
6. Scratch functional checks (prep jobs) disable the R38 batch-chain reference (they stop at step 20); formal units assert it.

Design choices fixed in configs (package-defined unless noted): COMPLETE real mass 0.5, reciprocal rank <= 5, teachers s1002/s1003
of the same series; head lr 1e-4 -> 1e-5, student lr 1e-5 -> 1e-6 (common cosine, floor 0.1), AdamW wd 1e-4 both groups, one clip per
peer (head + null + student) at 2.0 with the split recorded; 8000 updates, 16 queries/direction/update, sampler [1001, 3], head seed 1001.
CPU-test fixture fix: oracle selftest t2i fixture (1082794 -> 1082800). Package: FILE_SHA256 checked at freeze.

## R39X extension (main session relaying the user 「把你觉得值得提交的都提交了吧」; 0% addition: "everything meaningful is worth running")
- Separate trees r39xf/r39xc (copies of the frozen r39f/r39c; edits: per-cond JOINT parent table, runs_r39x output dir,
  PROTOCOL_FROZEN_R39X.json; r39xf data35.load_cond tolerates a parent without a B-TRUST file (n00, N route only)). R39 21-unit protocol untouched.
- JOINT prep equivalence (student at parent weights vs native cache, max |diff|): f00 2.2e-5, f60_s1002 4.6e-5, f60_s1003 2.2e-5,
  cc_s1002 1.6e-5, cc_s1003 2.2e-5, cc_r37s 2.2e-5 (tol 1e-4). DEV native rSum recomputed from student vectors equals the cache for all
  but f60_s1003 (495.98 vs 496.07: 1 i2t query flips at a near-tie) and cc_r37s (393.2 equal, 3 rank flips at ties); training/eval use the
  frozen-H s_H skip and candidates, so step-0 two-stage ranks are unaffected (asserted in-unit).
- 0% parent: no R38 N reference exists, so 3 N-frozen-shared-cosine cells (R38 recipe, cond n00) are new units; JOINT 0% units have no
  in-unit reference; their init/batch-chain identity with the new frozen cells is compared in the summary.
- R39X amendments (PROTOCOL_FROZEN_R39X.json, technical in-unit checks only): A1 — one-step check accepts exactly-zero-gradient tensors,
  gradient-sum tolerance 1e-6 -> 1e-5 (failed: 1083219 R37S J, 1083222 0% frozen NCE). A2 — also accepts a tensor whose predicted first
  AdamW step is below fp32 resolution (no element > 0.5 spacing); replay 1083287: R37S J peer A img.gpool.gru.bias_ih_l0 update-1 grad
  absmax 2.1e-12, max step/spacing 0.44 (failed: 1083238). Failed attempts kept under runs_r39x/_failed_attempts/.
