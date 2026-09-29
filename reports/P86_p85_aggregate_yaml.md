# P85 — per-method delivery blocks (spec §13.2), representative selected rows

## neural:vcs:product (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: neural_vcs
estimand: S
evaluation_readout: native
loss_scale: -J (P and Q means separately)
reference_measure: mixture_equal
critic_class: JointMLP concat -> 256 ReLU -> 256 ReLU -> 1
gradient_routing: full (data fixed; critic only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 2.326140880584717
evaluation_seconds: 0.16524696350097656
status: completed
```

## neural:vcs:cyclic8 (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: neural_vcs
estimand: S
evaluation_readout: native
loss_scale: -J (P and Q means separately)
reference_measure: mixture_equal
critic_class: JointMLP concat -> 256 ReLU -> 256 ReLU -> 1
gradient_routing: full (data fixed; critic only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 262144
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 6.213787794113159
evaluation_seconds: 0.21868324279785156
status: completed
```

## neural:vcs:inbatch (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: neural_vcs
estimand: S
evaluation_readout: native
loss_scale: -J (P and Q means separately)
reference_measure: mixture_equal
critic_class: JointMLP concat -> 256 ReLU -> 256 ReLU -> 1
gradient_routing: full (data fixed; critic only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 33521664
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 3.4186789989471436
evaluation_seconds: 0.5101406574249268
status: completed
```

## neural:js:product (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: neural_js
estimand: JS
evaluation_readout: native
loss_scale: Deep-InfoMax softplus form (f = log p/q at the optimum)
reference_measure: mixture_equal
critic_class: JointMLP concat -> 256 ReLU -> 256 ReLU -> 1
gradient_routing: full (data fixed; critic only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 1.9640073776245117
evaluation_seconds: 0.2003498077392578
status: completed
```

## neural:js:cyclic8 (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: neural_js
estimand: JS
evaluation_readout: native
loss_scale: Deep-InfoMax softplus form (f = log p/q at the optimum)
reference_measure: mixture_equal
critic_class: JointMLP concat -> 256 ReLU -> 256 ReLU -> 1
gradient_routing: full (data fixed; critic only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 262144
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 6.592891693115234
evaluation_seconds: 0.17366647720336914
status: completed
```

## neural:js:inbatch (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: neural_js
estimand: JS
evaluation_readout: native
loss_scale: Deep-InfoMax softplus form (f = log p/q at the optimum)
reference_measure: mixture_equal
critic_class: JointMLP concat -> 256 ReLU -> 256 ReLU -> 1
gradient_routing: full (data fixed; critic only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 33521664
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 3.426767349243164
evaluation_seconds: 0.47840094566345215
status: completed
```

## neural:infonce:inbatch (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: neural_infonce
estimand: MI (InfoNCE bounded by log(K+1))
evaluation_readout: native
loss_scale: native
reference_measure: native_base
critic_class: JointMLP concat -> 256 ReLU -> 256 ReLU -> 1
gradient_routing: full (data fixed; critic only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 33521664
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 3.4835729598999023
evaluation_seconds: 0.33225440979003906
status: completed
```

## neural:infonce:cyclic8 (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: neural_infonce
estimand: MI (InfoNCE bounded by log(K+1))
evaluation_readout: native
loss_scale: native
reference_measure: native_base
critic_class: JointMLP concat -> 256 ReLU -> 256 ReLU -> 1
gradient_routing: full (data fixed; critic only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 262144
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 6.730095863342285
evaluation_seconds: 0.09490537643432617
status: completed
```

## neural:nwj:product (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: neural_nwj
estimand: MI
evaluation_readout: native
loss_scale: native
reference_measure: native_base
critic_class: JointMLP concat -> 256 ReLU -> 256 ReLU -> 1
gradient_routing: full (data fixed; critic only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 2.1330056190490723
evaluation_seconds: 0.06461381912231445
status: completed
```

## neural:nwj:inbatch (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: neural_nwj
estimand: MI
evaluation_readout: native
loss_scale: native
reference_measure: native_base
critic_class: JointMLP concat -> 256 ReLU -> 256 ReLU -> 1
gradient_routing: full (data fixed; critic only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 33521664
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 3.33959698677063
evaluation_seconds: 0.3188440799713135
status: completed
```

## neural:dv:product (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: neural_dv
estimand: MI
evaluation_readout: native
loss_scale: native
reference_measure: native_base
critic_class: JointMLP concat -> 256 ReLU -> 256 ReLU -> 1
gradient_routing: full (data fixed; critic only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 2.1935737133026123
evaluation_seconds: 0.08005475997924805
status: completed
```

## neural:dv:inbatch (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: neural_dv
estimand: MI
evaluation_readout: native
loss_scale: native
reference_measure: native_base
critic_class: JointMLP concat -> 256 ReLU -> 256 ReLU -> 1
gradient_routing: full (data fixed; critic only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 33521664
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 3.4250550270080566
evaluation_seconds: 0.3247230052947998
status: completed
```

## neural:smile:product (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: neural_smile
estimand: MI
evaluation_readout: native
loss_scale: native
reference_measure: native_base
critic_class: JointMLP concat -> 256 ReLU -> 256 ReLU -> 1
gradient_routing: full (data fixed; critic only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 2.2126314640045166
evaluation_seconds: 0.08323192596435547
status: completed
```

## neural:smile:inbatch (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: neural_smile
estimand: MI
evaluation_readout: native
loss_scale: native
reference_measure: native_base
critic_class: JointMLP concat -> 256 ReLU -> 256 ReLU -> 1
gradient_routing: full (data fixed; critic only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 33521664
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 3.425344944000244
evaluation_seconds: 0.32059741020202637
status: completed
```

## cs_k:mult1 (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: cs_kernel_native
estimand: native_CS (Lebesgue reference)
evaluation_readout: native
loss_scale: D_CS = log A + log B_q - 2 log C
reference_measure: native_base
critic_class: none (Gaussian kernel plug-in, effective kernel sqrt(2) h)
gradient_routing: native autodiff (not used here)
n_independent_units: 4096
n_positive_pairs: 4096
n_negative_pairs: null
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 0.0037665367126464844
evaluation_seconds: 0.0
status: completed
```

## s_kde:common_risk (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: s_kde
estimand: S
evaluation_readout: S_plugin_and_J_common (both kept)
loss_scale: none (plug-in)
reference_measure: mixture_equal
critic_class: product Gaussian KDE eta_hat = tanh(1/2 (log p_hat - log q_hat))
gradient_routing: none
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 0.058892011642456055
evaluation_seconds: 0.24253129959106445
status: completed
```

## s_kde:scott (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: s_kde
estimand: S
evaluation_readout: S_plugin_and_J_common (both kept)
loss_scale: none (plug-in)
reference_measure: mixture_equal
critic_class: product Gaussian KDE eta_hat = tanh(1/2 (log p_hat - log q_hat))
gradient_routing: none
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 0.0
evaluation_seconds: 0.21225428581237793
status: completed
```

## s_kernel:rff:m4096:mult0.5 (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: s_kernel_rff
estimand: S
evaluation_readout: J_common
loss_scale: -J (P and Q means separately)
reference_measure: mixture_equal
critic_class: s_kernel_rff: fixed features + trainable read-out with intercept, tanh
gradient_routing: full (data fixed; read-out only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 1.1585016250610352
evaluation_seconds: 0.1121065616607666
status: completed
```

## s_kernel:nystrom:c512 (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: s_kernel_nystrom
estimand: S
evaluation_readout: J_common
loss_scale: -J (P and Q means separately)
reference_measure: mixture_equal
critic_class: s_kernel_nystrom: fixed features + trainable read-out with intercept, tanh
gradient_routing: full (data fixed; read-out only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 1.60172700881958
evaluation_seconds: 0.16266536712646484
status: completed
```

## rls:raw (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: rls_raw
estimand: S (relative density ratio r_1/2 - 1 = T)
evaluation_readout: J_common (unbounded T: diagnostic only)
loss_scale: RuLSIF alpha = 1/2 LS = -1/2 - J/2
reference_measure: mixture_equal
critic_class: linear read-out on RFF features, closed-form ridge (Cholesky)
gradient_routing: none (closed form)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 0.2069408893585205
evaluation_seconds: 0.014113903045654297
status: completed
```

## rls:tanh (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: rls_tanh
estimand: S
evaluation_readout: J_common
loss_scale: tanh(c T_raw + b0), (c, b0) by Nelder-Mead on J_hat (SELECT)
reference_measure: mixture_equal
critic_class: tanh-wrapped closed-form ridge read-out on RFF features
gradient_routing: none
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: a888332ab06a11c8590bd7ce3b364ad2f4d2ae9e3f27d9e24b5f5ebf65956f76
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 0.32714295387268066
evaluation_seconds: 0.0
status: completed
```

## s_kernel:rff:m4096:mult1 (cell P85_cubic_ds20_dt20_I2_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: b5673a29cd25879f4ffcca23bf326e28936c3c35
source_basis: supplement_af7172c_plus_v2
estimator: s_kernel_rff
estimand: S
evaluation_readout: J_common
loss_scale: -J (P and Q means separately)
reference_measure: mixture_equal
critic_class: s_kernel_rff: fixed features + trainable read-out with intercept, tanh
gradient_routing: full (data fixed; read-out only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: 9d7c0a7f6389727c5c630caaf9bfb702f28c2f2805fcbaa5c008b23d3cfc86d5
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 2.922236442565918
evaluation_seconds: 0.1876380443572998
status: completed
```

## s_kernel:nystrom:c256 (cell P85_gaussian_ds20_dt20_I4_N256_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: s_kernel_nystrom
estimand: S
evaluation_readout: J_common
loss_scale: -J (P and Q means separately)
reference_measure: mixture_equal
critic_class: s_kernel_nystrom: fixed features + trainable read-out with intercept, tanh
gradient_routing: full (data fixed; read-out only)
n_independent_units: 256
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: a78447eda6fb31401bd92b904bcebc8c2d49499930c540280b1d22a129e06733
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 4.7994465827941895
evaluation_seconds: 0.38399791717529297
status: completed
```

## s_kernel:rff:m4096:mult2 (cell P85_gaussian_ds20_dt20_I4_N256_B256_U2000_s1)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: s_kernel_rff
estimand: S
evaluation_readout: J_common
loss_scale: -J (P and Q means separately)
reference_measure: mixture_equal
critic_class: s_kernel_rff: fixed features + trainable read-out with intercept, tanh
gradient_routing: full (data fixed; read-out only)
n_independent_units: 256
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: 4984a4368aaf21cc77f01442d30af0d1577b069766abd982be37d1a28b8e5c50
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 2.1815097332000732
evaluation_seconds: 0.16643095016479492
status: completed
```

## s_kernel:rff:m256:mult0.5 (cell P85_gaussian_ds2_dt100_I1.5_N256_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 1973fe0cb68aec7acbbe184b863a7f8154ce1ca0+dirty
source_basis: supplement_af7172c_plus_v2
estimator: s_kernel_rff
estimand: S
evaluation_readout: J_common
loss_scale: -J (P and Q means separately)
reference_measure: mixture_equal
critic_class: s_kernel_rff: fixed features + trainable read-out with intercept, tanh
gradient_routing: full (data fixed; read-out only)
n_independent_units: 256
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: b04d99f3a432bb40b30e42736b3a50b018b5310ff739667ba9b561c89dd18ad3
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 2.426232099533081
evaluation_seconds: 0.23742103576660156
status: completed
```

## s_kernel:rff:m1024:mult0.5 (cell P85_gaussian_ds2_dt100_I1.5_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 4aa545438c919d6ea673e26d84e6164d83784b1b
source_basis: supplement_af7172c_plus_v2
estimator: s_kernel_rff
estimand: S
evaluation_readout: J_common
loss_scale: -J (P and Q means separately)
reference_measure: mixture_equal
critic_class: s_kernel_rff: fixed features + trainable read-out with intercept, tanh
gradient_routing: full (data fixed; read-out only)
n_independent_units: 4096
n_positive_pairs: 32768
n_negative_pairs: 32768
split_manifest_hash: 54434c0014ab150af7efa17d972b943aa56afa90dafee599fc27858e7b1e42e9
noise_target_kind: none
noise_tau: null
noise_sigma_coordinate: null
fit_seconds: 3.666825294494629
evaluation_seconds: 0.41698670387268066
status: completed
```

