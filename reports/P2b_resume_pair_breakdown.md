# P2 continuous vs interrupted+resumed (job 1007756)

- A: `/home/infres/yinwang/CS_QMI/outputs/smoke_vcs_1007756` step 100
- B: `/home/infres/yinwang/CS_QMI/outputs/smoke_vcs_interrupted_1007756` step 100

- encoder_state: max|Δ| params 4.273e-02 (max rel 1.533e+00); max|Δ| BN buffers 1.651e+02; tensors bit-identical 20/120
- projector_state: max|Δ| params 4.436e-02 (max rel 1.805e+00); max|Δ| BN buffers 1.939e+01; tensors bit-identical 1/8
- critic_state: max|Δ| params 4.210e-02 (max rel 7.337e-01); max|Δ| BN buffers 0.000e+00; tensors bit-identical 0/6
- logged steps: s0: Δloss 0.000e+00 shiftA=97 shiftB=97, s50: Δloss 5.626e-02 shiftA=138 shiftB=138, s99: Δloss 7.795e-02 shiftA=155 shiftB=155
- identical pair shifts at logged steps: True
- init hashes equal: True
