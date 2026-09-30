# P85 — per-method delivery blocks (spec §13.2), representative selected rows

## neural:smile:product (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 764a5bb91a1dff3ce23cc00a4f19f79f1faa75c8
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
fit_seconds: 2.224597930908203
evaluation_seconds: 0.07556319236755371
status: completed
```

## neural:smile:inbatch (cell P85_cubic_ds20_dt20_I10_N4096_B256_U2000_s0)

```yaml
experiment_family: direct_cs
protocol_id: P85
source_commit: 764a5bb91a1dff3ce23cc00a4f19f79f1faa75c8
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
fit_seconds: 3.4607601165771484
evaluation_seconds: 0.3194081783294678
status: completed
```

