# Supplementary Material — V6

# Proofs of the CS and Compression Identities

## Theorem 1

The mixture dominates both measures, so $p_M+q_M=2$ and $p_M=1+\eta$, $q_M=1-\eta$. Since both measures have unit mass, $\mathbb E_M\eta=0$. Hence $$\int p_M^2\,dM=1+S,\qquad \int q_M^2\,dM=1+S,\qquad \int p_Mq_M\,dM=1-S.$$ Substitution gives Eq. (6). Furthermore, $$\mathbb E_PT-\mathbb E_QT=2\mathbb E_M(T\eta),\qquad \tfrac12\mathbb E_PT^2+\tfrac12\mathbb E_QT^2=\mathbb E_MT^2.$$ It follows that $$J(T)=2\mathbb E_M(T\eta)-\mathbb E_MT^2=S-\mathbb E_M(T-\eta)^2.$$ Since $|\eta|\leq1$, the measurable bounded class contains the maximizer. Equality in the gap identifies it uniquely up to $M$-null sets. For $p,q>0$, $\eta=(p/q-1)/(p/q+1)=\tanh(\frac12\log(p/q))$. A finite neural logit approaches the endpoint values in the closure of its output class.

Equation (11) follows from $dM=(p+q)d\mu/2$. The convex generator is $F(r)=(r-1)^2/[2(r+1)]$ and $F''(r)=4/(r+1)^3>0$. Thus $S$ is an $f$-divergence and contracts under a common Markov kernel. Separate stochastic maps of $X$ and $Y$ also map the product law to the product of the transformed marginals, giving the dependence data-processing inequality.

## Proposition 3

Because $A=g(B)$, the tower property gives $\eta_A=\mathbb E[\eta_B\mid A]$. The difference $\eta_B-\eta_A$ is orthogonal to every square-integrable function of $A$. In particular, $$\mathbb E\eta_B^2=\mathbb E\eta_A^2+\mathbb E(\eta_B-\eta_A)^2.$$ This proves Eq. (15). By the reverse triangle inequality in $L^2$, $$\begin{aligned}
\left|\|T_B-T_A\|_2-\|\eta_B-\eta_A\|_2\right|
&\leq\|(T_B-\eta_B)-(T_A-\eta_A)\|_2\\
&\leq\|T_B-\eta_B\|_2+\|T_A-\eta_A\|_2,
\end{aligned}$$ which proves Eq. (16). Applying Eq. (8) separately to the two observations gives Eq. (17). All quantities use the same underlying balanced experiment and its pushforwards. Two unrelated fitted distributions do not define a compression pair.

# Statistical Error and Pair Sampling

## Independent evaluation

Conditional on $T$, Eq. (12) is unbiased for $J(T)$, so its bias for $S$ is $-\varepsilon_T$. Independence of the two pools gives its variance as the sum of the two variance terms in Eq. (13). Each summand has range width two and variance at most one. The bias–variance identity proves Proposition 2.

Hoeffding’s inequality separately applied to the two pools, followed by a union bound, gives with probability at least $1-\delta$, $$|\widehat J(T)-J(T)|\leq
\sqrt{\frac{2\log(4/\delta)}{n}}+
\sqrt{\frac{2\log(4/\delta)}{m}}.$$ The inequality can be conditioned on a critic fitted on independent data. A random fitted critic also contributes variability through $J(T)$ across fits.

For any $T_\epsilon\in\mathcal T$ with $S-J(T_\epsilon)\leq a_{\mathcal T}+\epsilon$, the empirical optimization condition implies $$J(T_\epsilon)-J(\widehat T)
\leq \widehat J(T_\epsilon)-\widehat J(\widehat T)+2\Delta
\leq\varepsilon_{\mathrm{opt}}+2\Delta.$$ Adding the approximation error and taking $\epsilon\downarrow0$ yields Eq. (14). Symmetrization and contraction apply because $a_\pm(0)=0$ and the two functions are 2-Lipschitz on $[-1,1]$. Standard measurable-class assumptions then give a complexity bound on $\Delta$ [2]. Consistency requires approximation, sampling, and optimization errors to converge to zero; it is not an unconditional property of a fixed neural architecture.

## Shared observations and finite pair budgets

For independent base observations $(X_i,Y_i)$, the off-diagonal pair $(X_i,Y_j)$ with $i\neq j$ has product-marginal law for a fixed pointwise critic. Different pairs share observations, so all-pair counts do not equal independent sample counts. Finite-dataset sampling excludes identical image identities, and training-mode BatchNorm adds a batch-dependent feature map. The independent-pool proposition is not applied to those training terms as though every pair were independent.

A useful conditional identity holds directly for finite pair subsampling. Given $B$ images and fixed view features, let $H_r$ be the average negative contribution for nonzero image shift $r$, including all view combinations. Uniformly sampling $K$ of the $B-1$ shifts without replacement gives $$\widehat Q_K=\frac1K\sum_{r\in\mathcal R_K}H_r,\qquad
\mathbb E[\widehat Q_K\mid\text{features}]=\frac1{B-1}\sum_{r=1}^{B-1}H_r=\widehat Q_{\mathrm{all}}.$$ If $s_H^2$ is the finite-population variance with denominator $B-2$, then $$\operatorname{Var}(\widehat Q_K\mid\text{features})=
\frac1K\left(1-\frac K{B-1}\right)s_H^2.$$ The same conditional expectation identity applies to the gradient of the finite sum. It isolates pair-selection noise, not image-sampling noise or downstream performance. It also applies to other additive pair objectives, including the matched logistic control.

## Transforming the bounded reading

The derivative of $g(s)=2\log[(1+s)/(1-s)]$ is $4/(1-s^2)$. Thus low variance of the bounded reading does not imply uniformly low variance after transformation near $S=1$. Estimation and training report the raw bounded-target reading. A finite empirical $\widehat J$ may be negative; it remains reported without clipping. A log-transformed CS display requires a stated projection into its numerical domain and is a separate output.

# Relationships to Existing Variational Objectives

#### RPC.

Equation (1) has pointwise optimizer $(p-\alpha q)/(\beta p+\gamma q)$ when the denominator is positive [17]. Its balanced instance is precisely $\eta$ and its objective is $J$. This equality holds for matching critics and pair samples; it is not an approximation or an empirical claim of superiority over the same functional.

#### Relative least squares.

The equal-mixture relative ratio is $g^*=p/[(p+q)/2]=1+\eta$. For the least-squares criterion $L(g)=\frac12\mathbb E_Mg^2-\mathbb E_Pg$, substitution gives $$L(1+T)=-\tfrac12-\tfrac12J(T).$$ Thus the raw quadratic problems are affinely related [19]. Fitting a ridge-linear score and subsequently applying $\tanh$ is a two-stage bounded estimator; the ridge step alone is not an exact solution of the nonlinear bounded fit.

#### Variational $f$-divergences.

For $F(r)=(r-1)^2/[2(r+1)]$, let $U=T-T^2/2$ with $-1\leq T\leq1$. On this image, $F^*(U)=T+T^2/2$. The variational objective $\mathbb E_PU-\mathbb E_QF^*(U)$ then equals $J(T)$. This situates the CS construction within general convex-risk and $f$-divergence estimation [9, 11].

#### Matched logistic.

At a fixed observation with posterior mean $\eta$, the half-squared conditional risk has derivative $(T-\eta)(1-T^2)$ in $f$, whereas the matched logistic risk has derivative $T-\eta$, with $T=\tanh f$. Both have the same population posterior optimum. Their curvatures at $f^*=\operatorname{atanh}\eta$ are $(1-\eta^2)^2$ and $1-\eta^2$, respectively. This is a link-dependent optimization difference, consistent with proper composite loss theory [13]; it is not a proof of a universal accuracy or robustness ordering.

# Scoring Geometry and Measurement

For a fixed distribution of original view pairs $W$, consider a deterministic chain $W\to H\to Z\to s$. Let $\eta_U=\mathbb E[C\mid U]$. Repeated conditional-expectation projection yields $$S_W-J(T(s))=
\mathbb E(\eta_W-\eta_H)^2+\mathbb E(\eta_H-\eta_Z)^2+
\mathbb E(\eta_Z-\eta_s)^2+\mathbb E(\eta_s-T(s))^2.$$ The four terms separate representation compression, projection compression, restriction to a scalar relation, and numerical posterior fitting. The identity uses population conditional expectations; differences between several finite fitted readings are not automatically these four terms.

For affine-cosine $f=a(s-\kappa)$, the implied posterior odds are $\exp[2a(s-\kappa)]$. Exact scalar calibration requires $p_s/q_s=\exp[2a(s-\kappa)]$, which is a restriction on the score distributions. A fixed training scorer may instead act as an inductive bias on the encoder. A strictly monotone map of $s$ preserves its measurable information even when its numerical values are poorly calibrated as a posterior.

The positive and negative magnitudes of the scalar ascent derivative $(c-T)(1-T^2)a$ are maximized at $T=-1/3$ and $T=1/3$, respectively. The corresponding similarities are $\kappa\mp\log 2/(2a)$ when these lie in $[-1,1]$. For an unnormalized representation $r_i$ and $z_i=r_i/\|r_i\|$, $$\nabla_{r_i}\bigl[cT-T^2/2\bigr]
=\frac{a(c-T)(1-T^2)}{\|r_i\|}\left(z_j-s_{ij}z_i\right).$$ A large scalar derivative therefore does not guarantee a large encoder gradient: the tangent direction and network Jacobian also matter. These identities describe an instantaneous update, while complete pretraining determines representation quality.

#### What the scorer identifies.

The similarity scorer is unchanged by a common orthogonal transformation. If $R^\top R=I$ and $z'_i=Rz_i$, then $$(z'_i)^\top z'_j=z_i^\top z_j,\qquad
T(z'_i,z'_j)=T(z_i,z_j).$$ Thus all pair scores and the SSL objective remain unchanged while individual coordinates can mix. A corresponding rotation of a linear readout preserves its predictions. This algebraic symmetry does not prevent useful semantic structure; it shows that the objective does not identify a unique coordinate-to-concept assignment. The main analysis instead measures specified relationships in complete observations and checks prediction responses. A claim of greater intrinsic interpretability would require an independently defined semantic or decision-structure criterion.

# Training, Selection, and Measurement Protocols

## Known-target compression protocol

The compression experiment uses $Y$ uniform on ten classes, $P(N=1\mid Y)=1/2\pm0.3$ by class parity, and $b=\mu_Y+(2N-1)(\delta/2)v+\epsilon$, with $\epsilon\sim\mathcal N(0,I_{64})$. The vector $v$ has value $1/\sqrt8$ on the first eight coordinates and zero elsewhere. The chain retains all 64, the first eight, and the first four coordinates. Deleting coordinates 8–63 preserves the conditional dependence exactly. The posterior density ratio is analytic; $S$ is evaluated on a 400,000-unit oracle sample. The displayed moderate condition has $\delta=2$, with a second decrement approximately 0.0441. Fits use five repeated independent splits, 3,000 observations per role, and 300 updates with validation-selected critics. The residual-based nested model includes the coarse prediction at initialization. These specifications distinguish the analytic posterior and exact zero-step property from the numerically integrated positive target.

## Visual implementation

The CIFAR ResNet-18 uses a bias-free $3\times3$, stride-one convolution, no initial max-pool, and global average pooling. The projector is a bias-free $512\to512$ linear layer, BatchNorm and ReLU, followed by a $512\to128$ linear layer with bias and no final BatchNorm. The norm floor is $10^{-8}$. Matrix weights use AdamW decay $10^{-4}$; biases and normalization parameters have zero decay. AdamW uses $(\beta_1,\beta_2)=(0.9,0.999)$ and $\epsilon=10^{-8}$. There are ten warm-up epochs followed by cosine decay to 1% of the peak learning rate. The reported runs use FP32.

Standard augmentation applies random resized cropping to $32\times32$ with scale $[0.2,1]$ and aspect ratio $[0.75,4/3]$, horizontal flip with probability 0.5, color jitter with probability 0.8 (strengths $0.4,0.4,0.4,0.1$), grayscale with probability 0.2, and channelwise normalization by mean and standard deviation $(0.5,0.5,0.5)$. Strong augmentation changes the crop lower bound to 0.08 and jitter strengths to $0.8,0.8,0.8,0.2$. All views are sampled independently from the selected common law.

The main VCS configuration fixes $(a,\kappa)=(2,0.5)$ and computes all ordered same-image, different-view positives and different-image negatives. Both endpoints receive gradients; no momentum branch or queue is used. The logistic control uses the same pair construction. The four-view SimCLR reference averages its two-view NT-Xent over the six unordered view pairs. Thus it is the existing multi-view SimCLR implementation, not the all-token logistic control.

## Hyperparameter selection and evaluation

Each dataset has its own 45k/5k stratified split. The affine grid is $(a,\kappa)\in\{(1.5,.5),(2,.25),(2,.5),(2,.75),(3,.5),(3,.25)\}$. The registered rule retains the default unless the best seed-zero candidate improves linear accuracy by at least 0.30 points on CIFAR-10 or 0.50 on CIFAR-100. This retains $(2,.5)$ for VCS and selects $(2,.25)$ and $(3,.5)$ for logistic. The best VCS CIFAR-100 screen is 60.64 at $(3,.5)$, below the replacement threshold; the rule-selected configuration is therefore not labelled a global optimum. The SimCLR reference uses its earlier selected recipe rather than a new grid of equal cumulative search size.

The standard linear probe reads raw $h$ with a bias, using 100 SGD epochs, learning rate 0.1, momentum 0.9, batch 256, no weight decay, and cosine decay to 0.1% of the peak. The final probe epoch is reported. Weighted $k$NN uses normalized $h$, $k=200$, temperature 0.1, and the 45k reference bank. Dataset selection and development consume validation information; all manuscript scores retain their validation designation. Earlier project evaluation used the official CIFAR-10 test set on an older configuration. Those scores are not mixed with the current validation tables.

## Conditional measurement implementation

The fixed-model solver comparison and the cross-encoder paired-image study are separate experimental sets. Table 3 varies the measuring algorithm on one fixed SimCLR model; Table 4 holds the VCS measuring procedure fixed and examines the final VCS and SimCLR encoders. Results from the two sets are not pooled. For the former, VCS and logistic use identical per-repeat FIT/VAL indices, EVAL draws, and within-class permutations. The common feature map is $\phi=[H(2N-1),1]$. VCS solves the penalized raw linear quadratic criterion, then selects a tanh output scale on VAL. Logistic is numerically fitted by L-BFGS; its short 50-closure control is reported alongside the VCS timing.

For the paired-image audit, 20,000 base images are partitioned by identity into FIT 10,000, VAL 2,000, and EVAL 8,000. Each has clean and modified versions. An analysis classifier is fitted on standardized clean $h$ using FIT labels only. Attribute assignment uses $P(N=1\mid Y)=0.5\pm0.3$ by class parity. Conditional negatives use same-class observations, and the known binary probability permits exact enumeration. Each evaluation permutation preserves the class labels. The layer family is pooled layer3, $h$, $z$, and logits; shared permutations define the maximum statistic. Pooled layer3 is used for layerwise detection, not assumed to be a deterministic parent of $h$.

Detection studies use 100 repeats for power and 200 for the reported null checks. Observed rejection rates are accompanied by their finite-repeat uncertainty; isolated level flags are retained rather than repaired by selectively removing their unfavorable alternatives. Predicted probabilities, margins, accuracy, and flips are computed on paired clean/modified EVAL images. Multi-seed summaries use encoder seeds as the replication unit; repeated image draws are not additional encoder seeds.

# Additional Visual and Measurement Results

| Method                             |           Linear (%) |            $k$NN (%) | Seeds |
|:-----------------------------------|---------------------:|---------------------:|------:|
| VCS                                |       $60.15\pm0.38$ |       $55.90\pm0.31$ |     5 |
| Balanced logistic                  |       $59.35\pm0.42$ |       $55.30\pm0.39$ |     5 |
| Paired difference \[95% interval\] | $+0.80\ [0.06,1.54]$ | $+0.60\ [0.11,1.09]$ |     5 |

Shared-score comparison on CIFAR-100, $(a,\kappa)=(2,0.5)$ and learning rate $10^{-3}$. This tests losses at a common configuration, unlike Table 2’s rule-selected scores.

| Method | Linear, strong (%) | Linear change (points) | $k$NN, strong (%) |
|:-------|-------------------:|-----------------------:|------------------:|
| VCS    |              88.25 |                $-0.78$ |             86.11 |
| SimCLR |              89.49 |                $+1.17$ |             88.71 |

CIFAR-10 strong augmentation, three seeds. Differences are relative to each method’s own standard-augmentation three-seed mean.

| Method            | Layer3, fine | $h$, fine | $r$, fine | $z$, fine |
|:------------------|-------------:|----------:|----------:|----------:|
| VCS               |         57.8 |      59.2 |      55.1 |      55.6 |
| Selected logistic |         57.5 |      59.5 |      55.3 |      55.6 |
| SimCLR            |         58.0 |      56.6 |      54.9 |      56.7 |

CIFAR-100 layerwise task readability under the separately specified standardized linear probe. Values are pooled reported means. VCS uses five encoder seeds; selected logistic and SimCLR use three. These are site-specific task readouts, not measured information losses. $r$ is the raw projector output and $z$ is its normalization.

#### Ablation of score adaptation.

With the same initialization and all-token full-gradient structure, learning the affine scale and offset gives 82.92% linear accuracy on CIFAR-10 and 56.74% on CIFAR-100 over three seeds, compared with 89.03% and 60.00% for the fixed scorer. This is an internal ablation under the specified optimizer, not a second proposed method or a universal statement about critic optimization. Explicit curvature perturbations of the fixed scorer do not improve the tested two-dataset screen. The main manuscript uses a single selected VCS implementation.

#### Other estimator targets.

The controlled benchmark includes Gaussian dependence levels indexed by Shannon mutual information $I\in\{2,4,6,8,10\}$ nats, with separate truth for each estimator. Under in-batch negatives, InfoNCE resolves the adjacent Gaussian levels more finely than VCS in the reported standardized-separation diagnostic, even above its nominal value bound. Single-shuffle DV and NWJ behave poorly in high-dependence conditions; increasing negative coverage materially changes their behavior. Corrected in-batch SMILE behaves comparably to other in-batch MI estimators. These results motivate reporting the sampling rule and estimator target together, rather than treating every neural MI estimator as equally unstable.

## Matched attribute measurements and prediction responses

Table 4 uses the color condition already measured on the two final encoder families. Table 8 gives the corresponding complete color and blur records. The probabilities and accuracy differences are changes from each image’s clean prediction. Responses are measured on 8,000 held-out base-image pairs; tests use repeated subsets of 2,000 observations.

| Encoder / seed | Attribute | $H$ | $O$ | $\Delta p(\mathrm{true})$ \[95% interval\] | $\Delta$acc. | Flips |
|:---------------|:----------|----:|----:|-------------------------------------------:|-------------:|------:|
| VCS / 1        | Color     | .60 | .13 |                  $-.0027\ [-.0043,-.0011]$ |       $-.18$ |   2.8 |
| VCS / 2        | Color     | .58 | .14 |                  $-.0038\ [-.0054,-.0022]$ |       $-.40$ |   2.9 |
| SimCLR / 1     | Color     | .08 | .11 |                  $-.0029\ [-.0043,-.0015]$ |       $-.25$ |   2.2 |
| SimCLR / 2     | Color     | .11 | .05 |                  $-.0027\ [-.0041,-.0014]$ |       $-.51$ |   2.4 |
| VCS / 1        | Blur      | .13 | .09 |                  $+.0001\ [-.0008,+.0009]$ |       $+.14$ |   1.5 |
| VCS / 2        | Blur      | .14 | .08 |                  $+.0005\ [-.0004,+.0014]$ |       $+.03$ |   1.4 |
| SimCLR / 1     | Blur      | .10 | .02 |                  $+.0003\ [-.0004,+.0010]$ |       $+.05$ |   1.1 |
| SimCLR / 2     | Blur      | .06 | .05 |                  $-.0001\ [-.0009,+.0006]$ |       $-.15$ |   1.3 |

**Matched color and blur records.** VCS is the measurement statistic in every row. Color uses strength 0.1; blur uses 0.25. $H$ and $O$ are rejection rates over 100 draws. Probability intervals are the reported paired-image 95% intervals within one encoder seed. Accuracy changes are percentage points; flips are percentages. These intervals do not replace between-encoder-seed replication.

For the displayed VCS measurement columns, two null constructions are evaluated over 200 draws each. Across the color rows, $H$ null rejection rates are 0.04–0.07 and $O$ rates are 0.03–0.08. Across blur rows, they are 0.03–0.07 and 0.04–0.07, respectively. The broader source study records a 0.10 logistic-logit null rate for one VCS-encoder seed and another statistic-level flag in a SimCLR family. Those flags remain part of the source record; neither is used as evidence for a valid, universally calibrated test. The displayed $H/O$ VCS columns are reported with their own checks.

The earlier SimCLR color-strength-0.2 demonstration is retained as a separate condition: accuracy changes by about $-0.7$ points, mean true-class probability by $-0.007$, and approximately 3.8% of predictions flip. It is not combined with the color-strength-0.1 rows or with the separate solver comparison to create a single effect estimate.

#### Scope of the model interpretation.

The measurement procedure is checked independently, then applied to final SSL models. A difference between encoders describes the relationship recoverable by the specified test at its sample and fitting budget. It is not a comparison of two estimators, nor evidence that one encoder is intrinsically more interpretable. Deterministic compression identities apply to maps such as $h\to r\to z$ or $h\to O(h)$ with class and attribute retained. They are not inferred from a sequence of independently pooled layer features. Spatial attribution, causal feature attribution, and semantic coordinate assignment are separate questions from the attribute measurements reported here.
