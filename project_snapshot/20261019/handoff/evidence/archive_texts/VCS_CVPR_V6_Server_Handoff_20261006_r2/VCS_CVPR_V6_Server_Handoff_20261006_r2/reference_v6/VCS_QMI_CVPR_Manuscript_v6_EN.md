# Variational Cauchy–Schwarz Quadratic Mutual Information for Visual Representation Learning

# Abstract

Statistical dependence provides a training signal for visual self-supervised learning and a way to examine the representations it produces. We formulate Cauchy–Schwarz quadratic mutual information under the equal mixture of a joint distribution and the product of its marginals. This reference yields a bounded score with an exact sample-based variational representation. Its optimal critic is $\tanh(\operatorname{PMI}/2)$, and the variational gap is a squared posterior-regression error. We characterize estimation error and express dependence lost under observation compression as a squared difference between optimal critics. For visual learning, we optimize the criterion through a fixed-scale similarity scorer. After pretraining, we fit independent critics to measure attributes retained by VCS and SimCLR representations and relate these measurements to paired prediction responses. Controlled experiments show lower posterior error than the tested kernel CS estimators in high-dimensional settings. Complete pretraining achieves 89.02% linear validation accuracy on CIFAR-10 over five seeds and 60.00% on CIFAR-100 over three seeds. The subsequent analysis finds different color detectability in the two encoders alongside small accuracy changes under the tested color intervention. The estimator thus supports both learning visual features and examining the relationships those features retain.

# Introduction

Augmented views provide the paired observations in visual self-supervised learning; independent image pairs define a reference. A dependence objective distinguishes the two and trains an encoder. After training, accuracy alone does not reveal whether features distinguish a color change, where that evidence is detectable, or how predictions respond. We study learning and measurement through the same dependence estimator.

Cauchy–Schwarz (CS) divergence compares densities through normalized overlap. Its quadratic structure supports kernel-based information bottlenecks and conditional-distribution comparison [21, 22]. Choosing the reference measure relative to which those densities are defined gives a direct neural estimation route without recovering the full densities.

For the joint law $P$ and product law $Q$, we use $M=(P+Q)/2$. Densities relative to $M$ sum to two, reducing CS divergence to a monotone transform of a bounded score $S$. Completing a square gives a sample-based objective $J$, the optimal critic $\tanh(\operatorname{PMI}/2)$, and an exact squared-error expression for $S-J$.

During pretraining, an encoder and projector express paired evidence through a fixed-scale similarity score. After training, new critics measure class-conditional relationships between frozen features and controlled attributes. The same procedure applies to VCS and externally trained representations. Paired image interventions quantify prediction responses, providing a behavioral counterpart to the dependence readings.

Known-target experiments assess estimation error, and complete pretraining assesses representation quality. We check the measurement procedure on a fixed model before applying it to VCS and SimCLR features. This separates the quality of a reading from the properties it describes, consistent with the role of critic inductive bias in dependence-based learning [18].

Our contributions are threefold. We develop the mixture-reference CS formulation and error statements for estimation and observation compression. We instantiate it in multi-view learning with a fixed similarity scale and fully differentiated pair expectations. We compare kernel and neural estimators, complete pretraining on two visual datasets, and measure retained attributes and prediction responses in the resulting SSL representations.

# Related Work

#### Neural and quadratic dependence estimation.

MINE estimates Shannon mutual information using a KL variational representation with an exponential critic moment [3]. Subsequent work analyzes the statistical behavior of variational bounds and stabilization methods [12, 15]. Convex risk minimization and $f$-GAN provide general variational divergence frameworks [9, 11]. Our target is mixture-reference CS dependence rather than Shannon mutual information.

Least-squares mutual information fits an ordinary density ratio [16]; relative least squares replaces its denominator by a mixture distribution [19]. At equal mixture weight, the relative ratio is $1+T^*$ in our notation. Relative Predictive Coding (RPC) studies $$J_{\mathrm{RPC}}(T)=\mathbb E_PT-\alpha\mathbb E_QT-\tfrac\beta2\mathbb E_PT^2-\tfrac\gamma2\mathbb E_QT^2.$$ Its balanced instance $(\alpha,\beta,\gamma)=(1,1,1)$ is exactly our functional [17]. The contribution studied here is the CS-reference construction, its estimation and compression analysis, and its visual instantiations. Appendix C gives the algebraic relationships explicitly.

#### CS criteria and visual representation learning.

CS information bottlenecks and conditional CS use density-overlap estimates for prediction, compression, and comparison [21, 22]. CS-Aligner applies CS to marginal distribution alignment across modalities [20]. SimCLR learns from augmented pairs through contrastive discrimination [4]; VICReg regularizes invariance and feature statistics [1]. Kernel-dependence SSL and efficient multi-view objectives provide related alternatives [8, 14]. We use a shared visual architecture to examine how the quadratic criterion translates into learned features.

#### Interpreting learned representations.

Concept activation methods relate human-defined concepts to model predictions [6]. HSIC attribution measures dependence between input regions and model responses [10]. We examine a different object: class-conditional attribute evidence in SSL representations and its relation to a frozen prediction head. Independent critics provide the measurements, and paired interventions quantify the observed response. The analysis concerns what the learned features retain, rather than assigning semantic meanings to individual embedding coordinates. Proper scoring-rule theory explains why squared and logistic critics can represent the same posterior while exhibiting different fitting behavior [13].

# Variational CS Dependence

## Equal-mixture reference and exact regression

Let $P,Q$ be probability measures on the same measurable space and define $$M=\tfrac12(P+Q),\qquad p_M=\frac{dP}{dM},\quad q_M=\frac{dQ}{dM}.$$ The derivatives exist since $M$ dominates both measures, and $p_M+q_M=2$ almost everywhere. Set $$\eta=\tfrac12(p_M-q_M),\qquad S(P,Q)=\mathbb E_M\eta^2.$$ Then $\mathbb E_M\eta=0$, $|\eta|\leq1$, and $0\leq S\leq1$. Under the squared-ratio convention, define $$D_{\mathrm{CS}}^{(M)}(P,Q)
=-\log\frac{(\int p_Mq_M\,dM)^2}{(\int p_M^2\,dM)(\int q_M^2\,dM)}.$$

#### Theorem 1 (Variational CS representation).

For measurable $T$ with $|T|\leq1$, let $$J(T)=\mathbb E_P[T-\tfrac12T^2]+\mathbb E_Q[-T-\tfrac12T^2].$$ Then $$\begin{aligned}
D_{\mathrm{CS}}^{(M)}(P,Q)&=2\log\frac{1+S(P,Q)}{1-S(P,Q)},\\
S(P,Q)&=\sup_{|T|\leq1}J(T),\\
S(P,Q)-J(T)&=\mathbb E_M(T-\eta)^2.
\end{aligned}$$ The maximizer is $T^*=\eta$, uniquely up to $M$-null sets.

The CS value is infinite at $S=1$. The identities follow from $p_M=1+\eta$, $q_M=1-\eta$ and completing the square. A proof is provided in Appendix A.

For any common dominating measure with densities $p,q$, $$T^*=\frac{p-q}{p+q}=\tanh\!\left(\tfrac12\log\frac pq\right).$$ Density-zero boundaries are interpreted by limits. For $P=P_{XY}$ and $Q=P_XP_Y$, the log ratio is the pointwise mutual information. We denote $I_{\mathrm{VQ}}(X;Y)=S(P_{XY},P_XP_Y)$ and optimize this bounded score directly. The logarithmic transform in Eq. (6) is used when a CS value is required.

Introduce a balanced binary label $C$: $W\mid C=+1\sim P$ and $W\mid C=-1\sim Q$. Its marginal is $M$, and $\mathbb E[C\mid W]=\eta(W)$. Consequently, $$J(T)=1-\mathbb E(C-T(W))^2.$$ This is a bounded posterior-regression problem. Parameterizing $T_\phi=\tanh f_\phi$ gives a sample-based neural estimator. Boundary optima are approached in the closure of this parameterization.

The identity $$S(P,Q)=\tfrac12\int\frac{(p-q)^2}{p+q}\,d\mu$$ identifies $S$ as half the classical triangular discrimination. Its $f$-divergence form gives invariance under bijective reparameterization and contraction under common stochastic maps. In particular, $I_{\mathrm{VQ}}=0$ exactly at independence. The reference construction therefore places CS dependence within an established divergence family and specifies its sample-based regression form.

## Fitting and evaluation error

Write $a_+(t)=t-t^2/2$ and $a_-(t)=-t-t^2/2$. Both have range $[-3/2,1/2]$ on $[-1,1]$. Given a fitted critic and independent evaluation pools $U_i\sim P$, $V_j\sim Q$, define $$\widehat J(T)=\frac1n\sum_{i=1}^n a_+(T(U_i))
+\frac1m\sum_{j=1}^m a_-(T(V_j)).$$

#### Proposition 2 (Estimation error).

If the evaluation pools are mutually independent and independent of the fitted critic, with $\varepsilon_T=\mathbb E_M(T-\eta)^2$, $$\begin{aligned}
\mathbb E[(\widehat J-S)^2\mid T]
={}&\varepsilon_T^2+\frac{\operatorname{Var}_P(a_+(T))}{n}\\
&+\frac{\operatorname{Var}_Q(a_-(T))}{m}
\leq\varepsilon_T^2+\frac1n+\frac1m.
\end{aligned}$$

The bias is exactly $-\varepsilon_T$. Thus a perturbation $T=\eta+e$ changes the population reading by $-\mathbb E_Me^2$: the first-order error cancels. Equation (13) separates this approximation bias from evaluation variance.

For a critic class $\mathcal T$, define $a_{\mathcal T}=\inf_{T\in\mathcal T}\mathbb E_M(T-\eta)^2$ and $\Delta=\sup_{T\in\mathcal T}|\widehat J(T)-J(T)|$. An empirical solution with optimization error $\varepsilon_{\mathrm{opt}}$ satisfies $$S-J(\widehat T)\leq a_{\mathcal T}+2\Delta+\varepsilon_{\mathrm{opt}}.$$ Boundedness permits concentration and standard complexity control of $\Delta$ [2]. Appendix B states the sampling assumptions and distinguishes independent evaluation from shared in-batch observations. These statements concern the estimator for fixed distributions.

## Dependence under representation compression

The same regression geometry describes what a specified observation retains. In the balanced experiment, let $B$ be an observation and $A=g(B)$ a deterministic compression. Let $\eta_B=\mathbb E[C\mid B]$ and $\eta_A=\mathbb E[C\mid A]$.

#### Proposition 3 (Observation loss and finite critics).

With $S_B=\mathbb E\eta_B^2$ and $S_A=\mathbb E\eta_A^2$, $$S_B-S_A=\mathbb E(\eta_B-\eta_A)^2.$$ For fitted critics $T_B,T_A$, define $D_T=\mathbb E(T_B-T_A)^2$ and $\varepsilon_B=\mathbb E(T_B-\eta_B)^2$, $\varepsilon_A=\mathbb E(T_A-\eta_A)^2$. Then $$\left|\sqrt{D_T}-\sqrt{S_B-S_A}\right|
\leq\sqrt{\varepsilon_B}+\sqrt{\varepsilon_A}.$$

This is the orthogonal-projection property of conditional expectation in $L^2$. It strengthens the data-processing statement by identifying the size of the decrease. For independent fitted readings, $$J(T_B)-J(T_A)=S_B-S_A-\varepsilon_B+\varepsilon_A.$$ Unequal fitting errors can therefore reverse a finite-sample ordering. Nested critic classes provide a useful implementation: the richer model includes the fitted coarse model as a candidate and learns an additional residual. We evaluate the resulting accuracy against known compression targets before interpreting visual measurements.

# Learning and Examining Visual Representations

## Multi-view visual learning

For each image $x_i$, draw $V$ independent augmentations $t_v$ from a common law. A shared encoder $e_\theta$ and projector $g_\psi$ give $$h_{vi}=e_\theta(t_v(x_i)),\quad r_{vi}=g_\psi(h_{vi}),\quad z_{vi}=\frac{r_{vi}}{\|r_{vi}\|_2}.$$ We use the similarity scorer $$T_{a,\kappa}(z,z')=\tanh\!\bigl[a(z^\top z'-\kappa)\bigr].$$ The hyperparameters $a>0$ and $\kappa$ fix a numerical scale during pretraining; they can be selected separately for each dataset. A fixed scale makes the encoder express pair discrimination through its geometry rather than through continual rescaling of the final score.

Let $\mathcal P$ contain ordered pairs of distinct views of the same image, and $\mathcal Q$ contain ordered pairs of different images across all views. For a batch of $B$ images, $$|\mathcal P|=VB(V-1),\qquad |\mathcal Q|=V^2B(B-1).$$ The loss is $$\mathcal L_{\mathrm{SSL}}
=\frac{1}{2|\mathcal P|}\sum_{p\in\mathcal P}(1-T_p)^2
+\frac{1}{2|\mathcal Q|}\sum_{q\in\mathcal Q}(1+T_q)^2.$$ It equals $1-\widehat J$. Positive and negative terms are normalized separately, and both endpoints receive gradients. All views share one encoder/projector forward pass and one optimizer update. Pair matrices can be processed in row blocks.

For a pair label $c$ and similarity $s$, the half-squared-loss derivative is $$\frac{\partial\ell_c}{\partial s}
=a(T-c)(1-T^2).$$ The residual and the link sensitivity jointly determine the update. After pretraining, an independently fitted critic reads a specified relationship in the frozen representation. Appendix D relates these roles through an observation decomposition and describes the rotational symmetry of the similarity scorer. The training score shapes pair geometry; the measurement critic evaluates what can be read from the learned features.

#### Matched objective comparison.

To separate scoring geometry from the loss, we train a balanced logistic control on the same logits $f=a(s-\kappa)$: $$\mathcal L_{\mathrm{JS}}=
\operatorname{mean}_{\mathcal P}\operatorname{softplus}(-2f)
+\operatorname{mean}_{\mathcal Q}\operatorname{softplus}(2f).$$ It uses the same posterior coordinate $T=\tanh f$ and matches the logit-gradient scale of $-\widehat J$ at $f=0$. We report shared-parameter and separately selected-parameter comparisons as different experiments.

## Measuring retained attributes

We freeze the trained encoder and projector and fit new critics to study the same SSL representations. For a controlled attribute $N$, image class $Y$, and observation $U$, define $$P_c^U=P_Y P_{U,N\mid Y},\qquad
Q_c^U=P_Y P_{U\mid Y}P_{N\mid Y}.$$ The quantity $\mathcal D_N(U\mid Y)=S(P_c^U,Q_c^U)$ measures attribute dependence beyond class association. An observation can be an encoder feature $H$, a projector output $Z$, or a frozen task output $O(H)$. We compare measurement algorithms on a fixed encoder, then keep the procedure fixed to examine different SSL encoders.

#### Detectability and observation changes.

Clean and attribute-modified versions of each image share a data split. The attribute indicator is randomized according to a specified $P(N\mid Y)$, and critics are fitted and selected outside the evaluation pool. Known binary conditional probabilities permit exact summation of the product expectation. Within-class permutations test conditional dependence; shared permutations and a maximum statistic handle the predefined layer family.

For a deterministic map, such as $H\to Z$ or $H\to O(H)$, Proposition 3 gives a population measure of the dependence change. The attribute and class variables are retained in both observations. We evaluate this operation against known compression targets and use independent evaluation for finite critics. Measurements at separately pooled intermediate layers are reported as layerwise detection, not as an assumed deterministic chain.

#### Prediction response.

The same intervention is applied to a frozen encoder and clean-feature classifier. We record changes in true-class probability and accuracy, and the prediction-flip rate. These accompany attribute detectability: statistical distinguishability differs from response magnitude. For example, $R=\delta N$ preserves dependence for every $\delta>0$ while its numerical response tends to zero. We use both measurements to describe retained attributes and prediction behavior.

# Experiments

## Estimator precision and sampling

We evaluate joint-versus-product estimation on Gaussian, invertibly transformed Gaussian, and mixture channels with computable targets. The primary endpoint is $\mathbb E_M(T-\eta)^2$, which equals the population variational gap. We compare neural VCS, a matched logistic critic, KDE for the same target, random Fourier and Nyström kernel features, and a relative least-squares estimator with bounded output. Hyperparameters use separate fitting and selection samples. Neural results below use eight nonzero cyclic shifts; full in-batch sampling gives similar values in these settings.

<figure>
<embed src="source/figures/estimator_dimension.pdf" />
<figcaption><strong>High-dimensional estimation.</strong> Two signal coordinates are retained while independent dimensions are added. <span class="math inline"><em>S</em> = 0.540</span> throughout, with 4,096 fitting observations. Mean posterior errors are shown for the common target.</figcaption>
</figure>

Figure 1 isolates dimension from dependence strength. At 20 dimensions per variable, VCS has posterior MSE 0.055, compared with 0.353 for kernel features and 0.481 for KDE. At 50 dimensions, the corresponding errors are 0.169, 0.507, and 0.527. Thus VCS retains substantially more of the fixed posterior signal. All methods deteriorate as irrelevant dimensions are added. Matched logistic is comparable at moderate dimension and reaches 0.509 at 100 dimensions, compared with 0.354 for VCS.

<figure>
<embed src="source/figures/estimator_samples.pdf" />
<figcaption><strong>Sample scaling at a fixed target.</strong> Dimension is 20 per variable and <span class="math inline"><em>S</em> = 0.802</span>. Both learned critics improve with the number of fitting observations. The figure compares errors for the same target rather than raw values of different information measures.</figcaption>
</figure>

Figure 2 varies sample count at fixed dimension. VCS error decreases from 0.364 at $N=256$ to 0.029 at $N=16{,}384$; relative LS–tanh decreases from 0.491 to 0.104. Matched logistic reaches 0.026 at the largest sample count. These results support the learned estimation route over the tested kernel implementations, rather than a uniform advantage of squared over logistic fitting.

At $N=4096$, VCS posterior errors at batch sizes 64, 256, and 1024 are 0.064, 0.059, and 0.061 under equal update counts. The effect of fitting-sample count is much larger than this batch effect. The supplementary comparisons retain MINE, NWJ, InfoNCE, and corrected SMILE under their own targets and sampling rules. In-batch InfoNCE resolves dependence levels well; the unstable behavior of single-shuffle DV does not characterize every contrastive estimator.

#### Measuring dependence lost under compression.

A conditional Gaussian attribute model with a coordinate chain $B\to U\to A$ has a known zero-decrement step $B\to U$ and a positive decrement $U\to A$. The dimensions are 64, 8, and 4. Table 1 compares estimators on the same chain. Nested fitting almost removes the spurious decrement caused by underfitting the richer observation. The positive decrement remains detectable and its error decreases. This checks the measurement operation used in Sec. 4.2 before it is applied to learned SSL features, whose population dependence values are unknown.

|                         |           |               |            |
|:------------------------|----------:|--------------:|-----------:|
| Estimator               | Zero-step | Positive-step |  Posterior |
|                         |      bias |          RMSE | MSE at $B$ |
| VCS, separate fits      | $-0.0204$ |        0.0105 |     0.0433 |
| VCS, nested             | $-0.0001$ |        0.0070 |     0.0157 |
| Logistic, separate fits | $-0.0192$ |        0.0109 |     0.0442 |
| Logistic, nested        | $+0.0000$ |        0.0087 |     0.0171 |

**Known compression targets.** In the moderate Gaussian condition, the first decrement is zero and the second has oracle value approximately 0.0441. “Nested” includes the coarse critic and enumerates the binary product term during fitting. The reported readout uses sampled pairs.

## Complete visual self-supervised learning

#### Protocol.

CIFAR-10 and CIFAR-100 each use 45,000 images for pretraining and 5,000 for validation, stratified by class [7]. All results in this section are validation results. Labels define the split and train downstream probes; pretraining uses image identities and augmented views. We use a CIFAR ResNet-18 [5], a $512\to512\to128$ projector, four views, batch size 256, and 800 epochs. The common AdamW learning rate is $10^{-3}$, with ten warm-up epochs and cosine decay. The encoder output $h$ is evaluated by a 100-epoch linear probe and weighted $k$NN ($k=200$). Full details and the selection history are in Appendix E.

| Dataset   | Method            | Scorer $(a,\kappa)$ |     Linear (%) |      $k$NN (%) | Seeds |
|:----------|:------------------|--------------------:|---------------:|---------------:|------:|
| CIFAR-10  | SimCLR            |                   – | $88.32\pm0.30$ | $87.65\pm0.40$ |     3 |
|           | Balanced logistic |          $(2,0.25)$ | $88.97\pm0.34$ | $87.39\pm0.34$ |     3 |
|           | VCS (ours)        |           $(2,0.5)$ | $89.03\pm0.06$ | $87.38\pm0.09$ |     3 |
| CIFAR-100 | SimCLR            |                   – | $58.25\pm0.35$ | $57.21\pm0.37$ |     3 |
|           | Balanced logistic |           $(3,0.5)$ | $60.49\pm0.21$ | $56.31\pm0.15$ |     3 |
|           | VCS (ours)        |           $(2,0.5)$ | $60.00\pm0.25$ | $55.80\pm0.24$ |     3 |

**Visual representation quality.** All entries use four views and 800 epochs on the 45k/5k split. Each block uses three common pretraining seeds. Logistic uses the same all-view pair structure and full gradients as VCS. Its score parameters are selected by the registered per-dataset rule; the shared-scale comparison is reported separately in the supplement.

#### Accuracy and matched objectives.

Table 2 contains one VCS configuration per dataset. VCS exceeds the SimCLR reference on linear evaluation on both datasets; SimCLR has higher $k$NN accuracy. On CIFAR-10, a five-seed confirmation gives $89.02\pm0.04$ for VCS and $88.31\pm0.21$ for SimCLR, with a paired difference of $+0.72$ percentage points and a 95% interval of $[0.48,0.95]$. The two fresh confirmation seeds yield differences of $+0.72$ and $+0.74$.

The balanced logistic comparison identifies the role of the loss within the same visual structure. With separately selected score parameters, CIFAR-10 results are close. On CIFAR-100, the selected logistic configuration leads VCS by 0.49 linear points over three seeds. At the common score $(2,0.5)$, the direction reverses: five-seed VCS and logistic means are 60.15 and 59.35. Score parameterization and the loss therefore interact. The evidence supports competitive visual learning with the quadratic criterion rather than universal accuracy superiority over another proper loss.

#### Augmentation and feature location.

Stronger cropping and color jitter change the relative result. Across three seeds, VCS linear accuracy is 88.25 and SimCLR is 89.49, compared with 89.03 and 88.32 under standard augmentation. A factor study attributes most of the between-method reversal to SimCLR benefiting from smaller crops. These conditions are reported together in Appendix F.

Layerwise CIFAR-100 probes further distinguish representation sites. Under a common standardized readout, the final VCS encoder feature has higher fine-class accuracy than SimCLR, whereas the ordering reverses at the normalized projector output. These measurements concern task readability at each site; they motivate evaluating $h$ and $z$ separately. They do not equate linear-probe changes with information-theoretic compression. Internal scorer and gradient-path ablations are kept separate from the external estimator comparison.

## Measuring What SSL Representations Retain

We first check the measuring algorithm on a fixed external model. We then keep the measurement procedure fixed and compare the final VCS and SimCLR encoders. The first comparison evaluates an estimator; the second uses that estimator to understand learned representations.

#### Measurement check on a fixed model.

For a fixed SimCLR model, VCS and logistic critics use the same feature class $\phi=[H(2N-1),1]$, data splits, and permutations. VCS uses a ridge solution of the raw quadratic problem followed by a validation-selected scale and $\tanh$ output. Logistic uses L-BFGS with a 50-closure budget. The two-stage VCS fit and the finite-budget logistic fit are compared as implemented solvers.

| Attribute  |  $n$ |  VCS | Logistic | VCS s | Log. s |
|:-----------|-----:|-----:|---------:|------:|-------:|
| Blur, 0.5  | 1000 | 0.75 |     0.82 |  0.04 |   0.05 |
| Blur, 0.5  | 2000 | 0.99 |     1.00 |  0.04 |   0.06 |
| Color, 0.2 | 2000 | 0.97 |     0.94 |  0.05 |   0.07 |

**Measurement comparison on one fixed SimCLR model.** Entries are rejection rates over 100 draws at nominal level 0.05. Data, feature class, and permutations are shared. Timing includes fitting and the solver’s selection procedure.

Both estimators detect the specified relationships with modest samples (Table 3); logistic is stronger for the smaller blur sample. Applicable null rejection rates are 0.04–0.07. This checks the measurement procedure on visual features, while Sec. 5.1 evaluates its numerical accuracy against known targets.

#### Retained color evidence and prediction response.

We next examine the final VCS and SimCLR encoders using the same VCS measurement statistic. The paired-image experiment uses 8,000 held-out base images and a classifier fitted on each encoder’s clean features. Table 4 aligns detection and response at the same color-shift strength, 0.1, for encoder seeds 1 and 2. Conditional tests use $n=2000$ with 100 repeated draws; prediction responses use all held-out pairs. The solver comparison in Table 3 is a separate fixed-model experiment.

| Encoder / seed |  $H$ |  $O$ | $\Delta$acc. | Flips |
|:---------------|-----:|-----:|-------------:|------:|
| VCS / 1        | 0.60 | 0.13 |      $-0.18$ |   2.8 |
| VCS / 2        | 0.58 | 0.14 |      $-0.40$ |   2.9 |
| SimCLR / 1     | 0.08 | 0.11 |      $-0.25$ |   2.2 |
| SimCLR / 2     | 0.11 | 0.05 |      $-0.51$ |   2.4 |

**What the SSL representations retain.** Color shift 0.1, using the same VCS measuring procedure for both encoders. $H$ and $O$ are rejection rates for encoder features and frozen logits. Accuracy changes are percentage points; flips are percentages. Each row is one encoder seed, not 100 independent pretrained models.

Color is more readily detected in VCS encoder features than in SimCLR, while detection in VCS logits is much lower. The same intervention changes accuracy by less than 0.6 percentage points in every row. Thus more detectable attribute evidence need not imply a larger accuracy change. Rejection rates describe detectability at the given sample budget, not fractions of retained information or intrinsic interpretability. Appendix F reports probability changes, blur results, and null checks.

# Conclusion

The equal-mixture reference expresses CS dependence as bounded posterior regression with an exact approximation gap and a squared geometry for observation loss. The criterion supports complete multi-view visual learning. Independently fitted critics then examine retained attributes and paired prediction responses in VCS and SimCLR representations. Known-target estimation, complete pretraining, and frozen-model measurements connect the estimator’s statistical properties to the learning and analysis of visual features.

# References

[1] Adrien Bardes, Jean Ponce, Yann LeCun. **VICReg: Variance-invariance-covariance regularization for self-supervised learning**. International conference on learning representations, 2022. https://arxiv.org/abs/2105.04906

[2] Peter L. Bartlett, Shahar Mendelson. **Rademacher and Gaussian complexities: Risk bounds and structural results**. Journal of Machine Learning Research, 2002. https://www.jmlr.org/papers/v3/bartlett02a.html

[3] Mohamed Ishmael Belghazi, Aristide Baratin, Sai Rajeshwar, Sherjil Ozair, Yoshua Bengio, Aaron Courville, R. Devon Hjelm. **Mutual information neural estimation**. International conference on machine learning, 2018. https://proceedings.mlr.press/v80/belghazi18a.html

[4] Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton. **A simple framework for contrastive learning of visual representations**. International conference on machine learning, 2020. https://arxiv.org/abs/2002.05709

[5] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. **Deep residual learning for image recognition**. Proceedings of the IEEE conference on computer vision and pattern recognition, 2016. https://arxiv.org/abs/1512.03385

[6] Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, Rory Sayres. **Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV)**. International conference on machine learning, 2018. https://proceedings.mlr.press/v80/kim18d.html

[7] Alex Krizhevsky, Vinod Nair, Geoffrey Hinton. **CIFAR-10 and CIFAR-100 datasets**. Official dataset documentation, n.d.. https://www.cs.toronto.edu/~kriz/cifar.html

[8] Yazhe Li, Roman Pogodin, Danica J. Sutherland, Arthur Gretton. **Self-supervised learning with kernel dependence maximization**. Advances in neural information processing systems, 2021. https://arxiv.org/abs/2106.08320

[9] XuanLong Nguyen, Martin J. Wainwright, Michael I. Jordan. **Estimating divergence functionals and the likelihood ratio by convex risk minimization**. IEEE Transactions on Information Theory, 2010. https://arxiv.org/abs/0809.0853

[10] Paul Novello, Thomas Fel, David Vigouroux. **Making sense of dependence: Efficient black-box explanations using dependence measure**. Advances in neural information processing systems, 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/1bed04feb85e5f02a7407fa3b191630b-Abstract-Conference.html

[11] Sebastian Nowozin, Botond Cseke, Ryota Tomioka. **f-GAN: Training generative neural samplers using variational divergence minimization**. Advances in neural information processing systems, 2016. https://arxiv.org/abs/1606.00709

[12] Ben Poole, Sherjil Ozair, Aaron Oord, Alexander Alemi, George Tucker. **On variational bounds of mutual information**. International conference on machine learning, 2019. https://proceedings.mlr.press/v97/poole19a.html

[13] Mark D. Reid, Robert C. Williamson. **Composite binary losses**. Journal of Machine Learning Research, 2010. https://www.jmlr.org/papers/v11/reid10a.html

[14] Oscar Skean, Aayush Dhakal, Nathan Jacobs, Luis Gonzalo Sanchez Giraldo. **FroSSL: Frobenius norm minimization for efficient multiview self-supervised learning**. European conference on computer vision, 2024. https://arxiv.org/abs/2310.02903

[15] Jiaming Song, Stefano Ermon. **Understanding the limitations of variational mutual information estimators**. International conference on learning representations, 2020. https://arxiv.org/abs/1910.06222

[16] Taiji Suzuki, Masashi Sugiyama, Takafumi Kanamori, Jun Sese. **Mutual information estimation reveals global associations between stimuli and biological processes**. BMC Bioinformatics, 2009. https://link.springer.com/article/10.1186/1471-2105-10-S1-S52

[17] Yao-Hung Hubert Tsai, Martin Q. Ma, Muqiao Yang, Han Zhao, Louis-Philippe Morency, Ruslan Salakhutdinov. **Self-supervised representation learning with relative predictive coding**. International conference on learning representations, 2021. https://arxiv.org/abs/2103.11275

[18] Michael Tschannen, Josip Djolonga, Paul K. Rubenstein, Sylvain Gelly, Mario Lucic. **On mutual information maximization for representation learning**. International conference on learning representations, 2020. https://arxiv.org/abs/1907.13625

[19] Makoto Yamada, Taiji Suzuki, Takafumi Kanamori, Hirotaka Hachiya, Masashi Sugiyama. **Relative density-ratio estimation for robust distribution comparison**. Advances in neural information processing systems, 2011. https://proceedings.neurips.cc/paper_files/paper/2011/file/d1f255a373a3cef72e03aa9d980c7eca-Paper.pdf

[20] Wenzhe Yin, Zehao Xiao, Pan Zhou, Shujian Yu, Jiayi Shen, Jan-Jakob Sonke, Efstratios Gavves. **Distributional vision-language alignment by Cauchy–Schwarz divergence**. International conference on learning representations, 2026. https://arxiv.org/abs/2502.17028

[21] Shujian Yu, Xi Yu, Sigurd Løkse, Robert Jenssen, José C. Príncipe. **Cauchy–Schwarz divergence information bottleneck for regression**. International conference on learning representations, 2024. https://arxiv.org/abs/2404.17951

[22] Shujian Yu, Hongming Li, Sigurd Løkse, Robert Jenssen, José C. Príncipe. **The conditional Cauchy–Schwarz divergence with applications to time-series data and sequential decision making**. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. https://arxiv.org/abs/2301.08970
