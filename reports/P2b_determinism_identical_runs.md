# Two identical continuous runs, same GPU (NVIDIA RTX PRO 6000 Blackwell Server Edition)

- A: `/home/infres/yinwang/CS_QMI/outputs/det_vcs_a_1007761` step 100
- B: `/home/infres/yinwang/CS_QMI/outputs/det_vcs_b_1007761` step 100

- encoder_state: max|Δ| params 4.105e-02 (max rel 1.629e+00); max|Δ| BN buffers 3.046e+02; tensors bit-identical 20/120
- projector_state: max|Δ| params 4.281e-02 (max rel 1.225e+00); max|Δ| BN buffers 2.879e+01; tensors bit-identical 1/8
- critic_state: max|Δ| params 4.575e-02 (max rel 6.723e-01); max|Δ| BN buffers 0.000e+00; tensors bit-identical 0/6
- logged steps: s0: Δloss 0.000e+00 shiftA=97 shiftB=97, s50: Δloss 9.939e-02 shiftA=138 shiftB=138, s99: Δloss 1.241e-01 shiftA=155 shiftB=155
- identical pair shifts at logged steps: True
- init hashes equal: True
