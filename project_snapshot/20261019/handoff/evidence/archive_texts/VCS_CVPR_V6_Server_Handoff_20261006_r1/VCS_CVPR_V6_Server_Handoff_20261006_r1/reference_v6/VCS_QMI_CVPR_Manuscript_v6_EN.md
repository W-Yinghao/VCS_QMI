# Variational Cauchy–Schwarz Quadratic Mutual Information for Visual Representation Learning

## Abstract

Statistical dependence is useful both for learning visual representations and for measuring the relationships they retain. We formulate Cauchy–Schwarz quadratic mutual information under the equal mixture of a joint distribution and the product of its marginals. This reference measure yields a bounded score with an exact, sample-based variational representation. Its optimal critic is $\tanh(\mathop{\mathrm{PMI}}/2)$, and its variational gap is a squared posterior-regression error. We characterize how critic approximation and evaluation sampling affect estimation, and express dependence lost under observation compression as a squared difference between optimal critics. We use the same quadratic criterion for two visual tasks: multi-view self-supervised learning with a fixed-scale similarity scorer, and conditional dependence measurement on frozen representations. Controlled experiments show lower posterior error than the tested kernel CS estimators in high-dimensional settings. The visual implementation achieves 89.02% linear validation accuracy on CIFAR-10 over five seeds and 60.00% on CIFAR-100 over three seeds. Frozen-model audits then relate detectable visual attributes to prediction responses under paired interventions. These results establish a practical CS estimation route with distinct roles in representation learning and analysis.

# Introduction

Augmented observations provide the supervision in much of visual self-supervised learning. Two views of the same image share content, whereas independently paired images provide a reference distribution. A dependence objective formalizes this distinction and supplies a signal for learning an encoder. After training, a related question concerns what the encoder has retained: can a representation reveal a controlled change in color or texture, and does that change also affect its predictions? Both questions require a dependence measure that can be estimated from samples.

Cauchy--Schwarz (CS) divergence compares densities through their normalized overlap. Its quadratic structure has supported information-bottleneck regression and conditional-distribution comparison through kernel estimators [21, 22]. A key design choice is the reference measure relative to which the densities are defined. Choosing this measure appropriately gives a direct route from CS to a neural estimation problem, without estimating the full densities.

We use the equal mixture of the distributions being compared. For the joint law $P$ and the product law $Q$, set $M=(P+Q)/2$. The densities relative to $M$ sum to two, reducing CS divergence to a monotone transform of a bounded score $S$. Completing a square gives an exact variational objective $J$ involving only first and second critic moments. The optimal critic is the bounded transform $\tanh(\mathop{\mathrm{PMI}}/2)$, and $S-J$ equals its squared regression error. The objective therefore connects an explicit dependence target to a concrete statistical learning problem.

This connection is useful in two settings. During visual pretraining, a shared encoder and projector learn through a fixed-scale similarity scorer. The scorer requires paired evidence to be expressed through embedding geometry. During analysis, the encoder is fixed and a separate critic is fitted to measure a specified relationship. Conditioning on image class isolates attribute dependence beyond class association. Comparing measurements before and after a deterministic representation map quantifies the corresponding loss of dependence at the population optimum.

These roles call for different empirical tests. Estimator quality is assessed against known dependence and posterior targets. Representation quality is assessed through complete pretraining and frozen-feature evaluation. Visual audits assess conditional detection and observed prediction responses on the same frozen model. This organization connects the theory to visual use without treating a training score as a substitute for downstream accuracy. The importance of the critic's inductive bias in dependence-based learning has also been established in prior work [18].

Our contributions are threefold. We develop the mixture-reference CS formulation, its exact posterior-regression gap, and error statements for estimation and observation compression. We instantiate the criterion in multi-view visual learning using a fixed similarity scale and fully differentiated, separately normalized pair expectations. We evaluate the estimator against kernel and neural alternatives, complete visual pretraining on two datasets, and apply it to conditional audits of frozen visual models. The audit is a direct measurement application of the same estimator.

# Related Work

#### Neural and quadratic dependence estimation.

MINE estimates Shannon mutual information using a KL variational representation with an exponential critic moment [3]. Subsequent work analyzes the statistical behavior of variational bounds and stabilization methods [12, 15]. Convex risk minimization and $f$-GAN provide general variational divergence frameworks [9, 11]. Our target is mixture-reference CS dependence rather than Shannon mutual information.

Least-squares mutual information fits an ordinary density ratio [16]; relative least squares replaces its denominator by a mixture distribution [19]. At equal mixture weight, the relative ratio is $1+T^*$ in our notation. Relative Predictive Coding (RPC) studies $$J_{\mathrm{RPC}}(T)=\mathbb E_PT-\alpha\mathbb E_QT-\tfrac\beta2\mathbb E_PT^2-\tfrac\gamma2\mathbb E_QT^2.
$$ Its balanced instance $(\alpha,\beta,\gamma)=(1,1,1)$ is exactly our functional C gives the algebraic relationships explicitly.

#### CS criteria and visual representation learning.

CS information bottlenecks and conditional CS use density-overlap estimates for prediction, compression, and comparison [21, 22]. CS-Aligner applies CS to marginal distribution alignment across modalities [20]. SimCLR learns from augmented pairs through contrastive discrimination [4]; VICReg regularizes invariance and feature statistics [1]. Kernel-dependence SSL and efficient multi-view objectives provide related alternatives [8, 14]. We use a shared visual architecture to examine how the quadratic criterion translates into learned features.

#### Interpreting representations through dependence.

Concept activation methods relate human-defined concepts to model predictions [6]. HSIC attribution measures dependence between input regions and model responses [10]. We study a complementary measurement task: class-conditional attribute dependence in a frozen representation and its change under a specified compression. Prediction interventions are reported alongside these measurements. The resulting audit is a use of the same CS estimator, rather than a separate saliency architecture. Proper scoring-rule theory explains why squared and logistic critics can represent the same posterior while exhibiting different fitting behavior [13].

# Variational CS Dependence 

## Equal-mixture reference and exact regression

Let $P,Q$ be probability measures on the same measurable space and define $$M=\tfrac12(P+Q),\qquad p_M=\frac{dP}{dM},\quad q_M=\frac{dQ}{dM}.$$ The derivatives exist since $M$ dominates both measures, and $p_M+q_M=2$ almost everywhere. Set $$\eta=\tfrac12(p_M-q_M),\qquad S(P,Q)=\mathbb E_M\eta^2.$$ Then $\mathbb E_M\eta=0$, $|\eta|\leq1$, and $0\leq S\leq1$. Under the squared-ratio convention, define $$D_{\mathrm{CS}}^{(M)}(P,Q)
=-\log\frac{(\int p_Mq_M\,dM)^2}{(\int p_M^2\,dM)(\int q_M^2\,dM)}.$$

**Theorem 1** (Variational CS representation). *For measurable $T$ with $|T|\leq1$, let $$J(T)=\mathbb E_P[T-\tfrac12T^2]+\mathbb E_Q[-T-\tfrac12T^2].
$$ Then $$\begin{aligned}
D_{\mathrm{CS}}^{(M)}(P,Q)&=2\log\frac{1+S(P,Q)}{1-S(P,Q)},\\
S(P,Q)&=\sup_{|T|\leq1}J(T),\\
S(P,Q)-J(T)&=\mathbb E_M(T-\eta)^2.
\end{aligned}$$ The maximizer is $T^*=\eta$, uniquely up to $M$-null sets.*

The CS value is infinite at $S=1$. The identities follow from $p_M=1+\eta$, $q_M=1-\eta$ and completing the square. A proof is provided in Appendix A.

For any common dominating measure with densities $p,q$, $$T^*=\frac{p-q}{p+q}=\tanh\!\left(\tfrac12\log\frac pq\right).
$$ Density-zero boundaries are interpreted by limits. For $P=P_{XY}$ and $Q=P_XP_Y$, the log ratio is the pointwise mutual information. We denote $I_{\mathrm{VQ}}(X;Y)=S(P_{XY},P_XP_Y)$ and optimize this bounded score directly. The logarithmic transform in Eq. (6) is used when a CS value is required.

Introduce a balanced binary label $C$: $W\mid C=+1\sim P$ and $W\mid C=-1\sim Q$. Its marginal is $M$, and $\mathbb E[C\mid W]=\eta(W)$. Consequently, $$J(T)=1-\mathbb E(C-T(W))^2.
$$ This is a bounded posterior-regression problem. Parameterizing $T_\phi=\tanh f_\phi$ gives a sample-based neural estimator. Boundary optima are approached in the closure of this parameterization.

The identity $$S(P,Q)=\tfrac12\int\frac{(p-q)^2}{p+q}\,d\mu
$$ identifies $S$ as half the classical triangular discrimination. Its $f$-divergence form gives invariance under bijective reparameterization and contraction under common stochastic maps. In particular, $I_{\mathrm{VQ}}=0$ exactly at independence. The reference construction therefore places CS dependence within an established divergence family and specifies its sample-based regression form.

## Fitting and evaluation error

Write $a_+(t)=t-t^2/2$ and $a_-(t)=-t-t^2/2$. Both have range $[-3/2,1/2]$ on $[-1,1]$. Given a fitted critic and independent evaluation pools $U_i\sim P$, $V_j\sim Q$, define $$\widehat J(T)=\frac1n\sum_{i=1}^n a_+(T(U_i))
+\frac1m\sum_{j=1}^m a_-(T(V_j)).
$$

**Proposition 2** (Estimation error). *If the evaluation pools are mutually independent and independent of the fitted critic, with $\varepsilon_T=\mathbb E_M(T-\eta)^2$, $$\begin{aligned}
\mathbb E[(\widehat J-S)^2\mid T]
={}&\varepsilon_T^2+\frac{\mathop{\mathrm{Var}}_P(a_+(T))}{n}\nonumber\\
&+\frac{\mathop{\mathrm{Var}}_Q(a_-(T))}{m}
\leq\varepsilon_T^2+\frac1n+\frac1m.

\end{aligned}$$*

The bias is exactly $-\varepsilon_T$. Thus a perturbation $T=\eta+e$ changes the population reading by $-\mathbb E_Me^2$: the first-order error cancels. Equation (13) separates this approximation bias from evaluation variance.

For a critic class $\mathcal T$, define $a_{\mathcal T}=\inf_{T\in\mathcal T}\mathbb E_M(T-\eta)^2$ and $\Delta=\sup_{T\in\mathcal T}|\widehat J(T)-J(T)|$. An empirical solution with optimization error $\varepsilon_{\mathrm{opt}}$ satisfies $$S-J(\widehat T)\leq a_{\mathcal T}+2\Delta+\varepsilon_{\mathrm{opt}}.
$$ Boundedness permits concentration and standard complexity control of $\Delta$ B states the sampling assumptions and distinguishes independent evaluation from shared in-batch observations. These statements concern the estimator for fixed distributions.

## Dependence under representation compression 

The same regression geometry describes what a specified observation retains. In the balanced experiment, let $B$ be an observation and $A=g(B)$ a deterministic compression. Let $\eta_B=\mathbb E[C\mid B]$ and $\eta_A=\mathbb E[C\mid A]$.

**Proposition 3** (Observation loss and finite critics). *With $S_B=\mathbb E\eta_B^2$ and $S_A=\mathbb E\eta_A^2$, $$S_B-S_A=\mathbb E(\eta_B-\eta_A)^2.
$$ For fitted critics $T_B,T_A$, define $D_T=\mathbb E(T_B-T_A)^2$ and $\varepsilon_B=\mathbb E(T_B-\eta_B)^2$, $\varepsilon_A=\mathbb E(T_A-\eta_A)^2$. Then $$\left|\sqrt{D_T}-\sqrt{S_B-S_A}\right|
\leq\sqrt{\varepsilon_B}+\sqrt{\varepsilon_A}.
$$*

This is the orthogonal-projection property of conditional expectation in $L^2$. It strengthens the data-processing statement by identifying the size of the decrease. For independent fitted readings, $$J(T_B)-J(T_A)=S_B-S_A-\varepsilon_B+\varepsilon_A.
$$ Unequal fitting errors can therefore reverse a finite-sample ordering. Nested critic classes provide a useful implementation: the richer model includes the fitted coarse model as a candidate and learns an additional residual. We evaluate the resulting accuracy against known compression targets before interpreting visual measurements.

# Learning and Auditing Visual Representations

## Multi-view visual learning

For each image $x_i$, draw $V$ independent augmentations $t_v$ from a common law. A shared encoder $e_\theta$ and projector $g_\psi$ give $$h_{vi}=e_\theta(t_v(x_i)),\quad r_{vi}=g_\psi(h_{vi}),\quad z_{vi}=\frac{r_{vi}}{\|r_{vi}\|_2}.$$ We use the similarity scorer $$T_{a,\kappa}(z,z')=\tanh\!\bigl[a(z^\top z'-\kappa)\bigr].
$$ The hyperparameters $a>0$ and $\kappa$ fix a numerical scale during pretraining; they can be selected separately for each dataset. A fixed scale makes the encoder express pair discrimination through its geometry rather than through continual rescaling of the final score.

Let $\mathcal P$ contain ordered pairs of distinct views of the same image, and $\mathcal Q$ contain ordered pairs of different images across all views. For a batch of $B$ images, $$|\mathcal P|=VB(V-1),\qquad |\mathcal Q|=V^2B(B-1).$$ The loss is $$\mathcal L_{\mathrm{SSL}}
=\frac{1}{2|\mathcal P|}\sum_{p\in\mathcal P}(1-T_p)^2
+\frac{1}{2|\mathcal Q|}\sum_{q\in\mathcal Q}(1+T_q)^2.
$$ It equals $1-\widehat J$. Positive and negative terms are normalized separately, and both endpoints receive gradients. All views share one encoder/projector forward pass and one optimizer update. Pair matrices can be processed in row blocks.

For a pair label $c$ and similarity $s$, the half-squared-loss derivative is $$\frac{\partial\ell_c}{\partial s}
=a(T-c)(1-T^2).
$$ The residual and the link sensitivity jointly determine the update. In contrast, a fitted measurement critic approximates the posterior for an already fixed representation. Appendix D relates these two roles through an observation decomposition. Neither role requires interpreting the current training score as the complete dependence optimum.

#### Matched objective comparison.

To separate scoring geometry from the loss, we train a balanced logistic control on the same logits $f=a(s-\kappa)$: $$\mathcal L_{\mathrm{JS}}=
\operatorname{mean}_{\mathcal P}\mathop{\mathrm{softplus}}(-2f)
+\operatorname{mean}_{\mathcal Q}\mathop{\mathrm{softplus}}(2f).$$ It uses the same posterior coordinate $T=\tanh f$ and matches the logit-gradient scale of $-\widehat J$ at $f=0$. We report shared-parameter and separately selected-parameter comparisons as different experiments.

## Conditional audits of frozen models 

The second use fixes the visual model and fits a new critic. For a controlled attribute $N$, image class $Y$, and representation $H$, define $$P_c=P_Y P_{H,N\mid Y},\qquad
Q_c=P_Y P_{H\mid Y}P_{N\mid Y}.
$$ Estimating $S(P_c,Q_c)$ asks whether the representation distinguishes the attribute beyond its association with class. The same construction applies to a frozen task output $O(H)$. Here the interpreted model and the measuring algorithm are independent choices: a VCS critic can audit a SimCLR representation.

We construct clean and attribute-modified versions of each image, randomize the attribute indicator according to a specified $P(N\mid Y)$, and keep all versions of an image in one data split. Critics are fitted and selected outside the evaluation pool. For binary $N$ with known conditional probabilities, the product expectation can be evaluated by summing over its two states. Within-class permutation tests evaluate conditional dependence. Shared permutations and a maximum statistic account for the predefined layer family.

Two outputs are reported. The first is attribute detectability in the representation or task output, with its sampling uncertainty. The second is the model's response to a paired image intervention, including probability change, accuracy change, and prediction flips. This separates statistical dependence from response magnitude. For example, a noiseless encoding $R=\delta N$ preserves dependence for every $\delta>0$ while its numerical response tends to zero. A conditional audit therefore supplies an interpretable measurement, not a claim of causal feature attribution from dependence alone.

# Experiments

## Estimator precision and sampling 

We evaluate joint-versus-product estimation on Gaussian, invertibly transformed Gaussian, and mixture channels with computable targets. The primary endpoint is $\mathbb E_M(T-\eta)^2$, which equals the population variational gap. We compare neural VCS, a matched logistic critic, KDE for the same target, random Fourier and Nyström kernel features, and a relative least-squares estimator with bounded output. Hyperparameters use separate fitting and selection samples. Neural results below use eight nonzero cyclic shifts; full in-batch sampling gives similar values in these settings.

![**High-dimensional estimation.** Two signal coordinates are retained while independent dimensions are added. $S=0.540$ throughout, with 4,096 fitting observations. Mean posterior errors are shown for the common target.](source/figures/estimator_dimension.pdf)

Figure 1 isolates dimension from dependence strength. At 20 dimensions per variable, VCS has posterior MSE 0.055, compared with 0.353 for kernel features and 0.481 for KDE. At 50 dimensions, the corresponding errors are 0.169, 0.507, and 0.527. Thus VCS retains substantially more of the fixed posterior signal. All methods deteriorate as irrelevant dimensions are added. Matched logistic is comparable at moderate dimension and reaches 0.509 at 100 dimensions, compared with 0.354 for VCS.

![**Sample scaling at a fixed target.** Dimension is 20 per variable and $S=0.802$. Both learned critics improve with the number of fitting observations. The figure compares errors for the same target rather than raw values of different information measures.](source/figures/estimator_samples.pdf)

Figure 2 varies sample count at fixed dimension. VCS error decreases from 0.364 at $N=256$ to 0.029 at $N=16{,}384$; relative LS--tanh decreases from 0.491 to 0.104. Matched logistic reaches 0.026 at the largest sample count. These results support the learned estimation route over the tested kernel implementations, rather than a uniform advantage of squared over logistic fitting.

At $N=4096$, VCS posterior errors at batch sizes 64, 256, and 1024 are 0.064, 0.059, and 0.061 under equal update counts. The effect of fitting-sample count is much larger than this batch effect. The supplementary comparisons retain MINE, NWJ, InfoNCE, and corrected SMILE under their own targets and sampling rules. In-batch InfoNCE resolves dependence levels well; the unstable behavior of single-shuffle DV does not characterize every contrastive estimator.

#### Measuring dependence lost under compression.

A conditional Gaussian attribute model with a coordinate chain $B\to U\to A$ has a known zero-decrement step $B\to U$ and a positive decrement $U\to A$. The dimensions are 64, 8, and 4. Table 1 compares estimators on the same chain. Nested fitting almost removes the spurious decrement caused by underfitting the richer observation. The positive decrement remains detectable and its error decreases. This checks the measurement operation used in Sec. 4.2 without assuming a true dependence value for a learned image representation.

|                         |           |               |            |
|:------------------------|----------:|--------------:|-----------:|
| Estimator               | Zero-step | Positive-step |  Posterior |
|                         |      bias |          RMSE | MSE at $B$ |
| VCS, separate fits      | $-0.0204$ |        0.0105 |     0.0433 |
| VCS, nested             | $-0.0001$ |        0.0070 |     0.0157 |
| Logistic, separate fits | $-0.0192$ |        0.0109 |     0.0442 |
| Logistic, nested        | $+0.0000$ |        0.0087 |     0.0171 |

: **Known compression targets.** In the moderate Gaussian condition, the first decrement is zero and the second has oracle value approximately 0.0441. "Nested" includes the coarse critic and enumerates the binary product term during fitting. The reported readout uses sampled pairs.

## Complete visual self-supervised learning 

#### Protocol.

CIFAR-10 and CIFAR-100 each use 45,000 images for pretraining and 5,000 for validation, stratified by class E.

| Dataset   | Method            | Scorer $(a,\kappa)$ |     Linear (%) |      $k$NN (%) | Seeds |
|:----------|:------------------|--------------------:|---------------:|---------------:|------:|
| CIFAR-10  | SimCLR            |                  -- | $88.32\pm0.30$ | $87.65\pm0.40$ |     3 |
|           | Balanced logistic |          $(2,0.25)$ | $88.97\pm0.34$ |          87.39 |     3 |
|           | VCS (ours)        |           $(2,0.5)$ | $89.03\pm0.06$ | $87.38\pm0.09$ |     3 |
| CIFAR-100 | SimCLR            |                  -- | $58.25\pm0.35$ | $57.21\pm0.37$ |     3 |
|           | Balanced logistic |           $(3,0.5)$ | $60.49\pm0.21$ | $56.31\pm0.15$ |     3 |
|           | VCS (ours)        |           $(2,0.5)$ | $60.00\pm0.25$ | $55.80\pm0.24$ |     3 |

#### Accuracy and matched objectives.

Table 2 contains one VCS configuration per dataset. VCS exceeds the SimCLR reference on linear evaluation on both datasets; SimCLR has higher $k$NN accuracy. On CIFAR-10, a five-seed confirmation gives $89.02\pm0.04$ for VCS and $88.31\pm0.21$ for SimCLR, with a paired difference of $+0.72$ percentage points and a 95% interval of $[0.48,0.95]$. The two fresh confirmation seeds yield differences of $+0.72$ and $+0.74$.

The balanced logistic comparison identifies the role of the loss within the same visual structure. With separately selected score parameters, CIFAR-10 results are close. On CIFAR-100, the selected logistic configuration leads VCS by 0.49 linear points over three seeds. At the common score $(2,0.5)$, the direction reverses: five-seed VCS and logistic means are 60.15 and 59.35. Score parameterization and the loss therefore interact. The evidence supports competitive visual learning with the quadratic criterion rather than universal accuracy superiority over another proper loss.

#### Augmentation and feature location.

Stronger cropping and color jitter change the relative result. Across three seeds, VCS linear accuracy is 88.25 and SimCLR is 89.49, compared with 89.03 and 88.32 under standard augmentation. A factor study attributes most of the between-method reversal to SimCLR benefiting from smaller crops. These conditions are reported together in Appendix F.

Layerwise CIFAR-100 probes further distinguish representation sites. Under a common standardized readout, the final VCS encoder feature has higher fine-class accuracy than SimCLR, whereas the ordering reverses at the normalized projector output. These measurements concern task readability at each site; they motivate evaluating $h$ and $z$ separately. They do not equate linear-probe changes with information-theoretic compression. Internal scorer and gradient-path ablations are kept separate from the external estimator comparison.

## Interpreting frozen visual models 

The audit tests the estimator in its measurement role. Its primary comparison fixes the interpreted model and varies the measuring algorithm. We use frozen SimCLR representations and controlled color shifts or blur, with class-conditional pair construction from Eq. (24). This application therefore does not depend on the model having been trained with VCS.

#### Estimator comparison on the same model.

For the fixed SimCLR audit fixture, we fit VCS and logistic critics in the same feature class $\phi=[H(2N-1),1]$, with common splits and permutations. VCS uses a ridge solution of the raw quadratic problem followed by a validation-selected scale and $\tanh$ output. Logistic uses L-BFGS with a 50-closure budget. These are two practical solvers; the ridge initialization is not described as an exact solution of the bounded nonlinear problem.

| Attribute  |  $n$ |  VCS | Logistic | VCS s | Log. s |
|:-----------|-----:|-----:|---------:|------:|-------:|
| Blur, 0.5  | 1000 | 0.75 |     0.82 |  0.04 |   0.05 |
| Blur, 0.5  | 2000 | 0.99 |     1.00 |  0.04 |   0.06 |
| Color, 0.2 | 2000 | 0.97 |     0.94 |  0.05 |   0.07 |

: **Conditional detection on a fixed SimCLR model.** Entries are rejection rates over 100 repeated draws at nominal level 0.05. Each comparison uses identical data, feature class, and permutations. The final columns show observed fitting time in seconds, including the solver's selection procedure.

Table 3 shows that both estimators support conditional detection at modest sample sizes. VCS detects the blur relationship in 99% of the $n=2000$ repeats and the color relationship in 97%. The logistic comparison is close, and is stronger for the smaller blur sample. The broader solver study also finds that higher common squared score does not always give higher permutation power. Numerical score estimation and testing are thus reported as distinct outcomes. The applicable null checks have observed rejection rates between 0.04 and 0.07.

#### Attribute evidence and prediction response.

A separate paired-image audit uses a SimCLR encoder and a clean-feature classifier, with 8,000 held-out base images. A color shift of 0.2 changes its accuracy by approximately $-0.7$ points, changes mean true-class probability by $-0.007$, and flips about 3.8% of predictions. At a smaller color shift of 0.1, the backbone-level VCS test has rejection rates 0.08 and 0.11 on two additional encoder seeds, while the any-layer test rejects in every repeat. These measurements distinguish attribute evidence in earlier observations from its detectability in the evaluated feature.

A joint presentation of detection and paired response is essential: low average accuracy change can coexist with prediction flips, and failure to reject at one layer is not evidence of exact independence. The intervention measures the response to the specified image operation. Large blur changes recognizable class content and is treated as a content-damaging perturbation rather than automatically as a harmful shortcut.

# Conclusion

Choosing the equal-mixture reference expresses CS dependence as bounded posterior regression. The resulting objective has a precise approximation error, a sample-based implementation, and a squared geometry for dependence changes under compression. A fixed-scale visual instantiation supports complete multi-view representation learning, while independently fitted critics measure conditional relationships in frozen models. Controlled estimation, visual pretraining, and paired attribute audits evaluate these capabilities separately. Together they show how the same CS estimator can be used to learn useful visual features and to examine the relationships those features retain.

::: center

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
