# fig_mechanism — caption draft (assembly of frozen results; `scripts/fig_mechanism.py`, job 1030560)

**(a)** Weight the objective puts on contaminated positives (P145, ε 0.10, seed 0): gradient-norm ratio of the replaced to the kept positive pairs
with respect to the projector outputs, second half of training.  The bounded VCS score weights wrong positives like SimCLR's softmax (0.48–0.50);
the matched logistic loss weights them 1.5× more (0.74–0.76).  Accuracy under contamination did not separate the methods (P145 add. 1: no robustness
difference at three seeds), so this is a weighting statement, not a robustness claim.
**(b)** CIFAR-100 fine-label linear readout by site (P141; A-P3 and JS-AP3 5 seeds, SimCLR 3; mean ± sd).  h is the best readout site for every
method.  The VCS / JS projectors discard more fine-label information between h and z (−7.5 / −8.2) than SimCLR's (−3.5); at h, VCS ≥ JS ≥ SimCLR.
**(c)** Common J of the fitted VCS critic at h and at z (P149 + add. 1, seeds 1–2, t 0).  At z every encoder reads J ≈ 0.96–0.97 (near saturation,
|T| > 0.95 on 90–97 % of P), whatever its family; at h the readings are 0.79–0.87.  Fitted J is a fit-limited lower reading; a cross-site
difference is not a data-processing statement (W11).
Reading for the text: the projector output is where the pairwise objective is satisfied (critic-fittable dependence saturates at z for every
encoder), and the backbone keeps the class information (h is best in (b)).  Same-posterior objectives (VCS, matched JS) differ in how they weight
pairs (a), not in what they learn at h (Table: VCS ≈ JS).
