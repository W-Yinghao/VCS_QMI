Publication copy (sample names aliased; numbers and conclusions unchanged).

# R30 完整表：40 轮（自动生成；full DEV；未读 TEST；只列数字，不含解读）

40 轮内 best full-DEV rSum 及同一 epoch 的六项 Recall、final、last5（epoch 记为 completed_cotrain_epochs）。H0/L0/Ls/PN/PV/PB 从共同 H 父状态出发；HN/HB 为 seed1001 延长到 40 轮的阶段路径（修订 A1）；60% seed1002/1003 的 PN/PV/PB 与 HH 为修订 A1 的配对确认。Ls/PN/PV/PB 系数见 COEFFICIENTS.json。未完成记 NA。

| unit | 噪声 | seed | 臂 | best | best epoch | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | final | last5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r30_f00_s1001_H0 | 0% | 1001 | H→H | NA | | | | | | | | | |
| r30_f00_s1001_L0 | 0% | 1001 | H→L | NA | | | | | | | | | |
| r30_f00_s1001_Ls | 0% | 1001 | H→L(s) | NA | | | | | | | | | |
| r30_f00_s1001_PB | 0% | 1001 | H→PB(s) | NA | | | | | | | | | |
| r30_f00_s1001_PN | 0% | 1001 | H→PN(s) | NA | | | | | | | | | |
| r30_f00_s1001_PV | 0% | 1001 | H→PV(s) | NA | | | | | | | | | |
| r30v11_f00_s1001_HB | 0% | 1001 | H→B | NA | | | | | | | | | |
| r30v11_f00_s1001_HN | 0% | 1001 | H→N | NA | | | | | | | | | |
| r30_f60_s1001_H0 | 60% | 1001 | H→H | NA | | | | | | | | | |
| r30_f60_s1001_L0 | 60% | 1001 | H→L | 482.92 | 24 | 75.05 | 92.21 | 95.76 | 53.57 | 79.25 | 87.08 | 481.83 | 481.53 |
| r30_f60_s1001_Ls | 60% | 1001 | H→L(s) | NA | | | | | | | | | |
| r30_f60_s1001_PB | 60% | 1001 | H→PB(s) | NA | | | | | | | | | |
| r30_f60_s1001_PN | 60% | 1001 | H→PN(s) | NA | | | | | | | | | |
| r30_f60_s1001_PV | 60% | 1001 | H→PV(s) | NA | | | | | | | | | |
| r30v11_f60_s1001_HB | 60% | 1001 | H→B | NA | | | | | | | | | |
| r30v11_f60_s1001_HN | 60% | 1001 | H→N | NA | | | | | | | | | |
| r30a1_f60_s1002_PB | 60% | 1002 | H→PB(s) | NA | | | | | | | | | |
| r30a1_f60_s1002_PN | 60% | 1002 | H→PN(s) | NA | | | | | | | | | |
| r30a1_f60_s1002_PV | 60% | 1002 | H→PV(s) | NA | | | | | | | | | |
| r30v11_f60_s1002_HH | 60% | 1002 | H→H | NA | | | | | | | | | |
| r30a1_f60_s1003_PB | 60% | 1003 | H→PB(s) | NA | | | | | | | | | |
| r30a1_f60_s1003_PN | 60% | 1003 | H→PN(s) | NA | | | | | | | | | |
| r30a1_f60_s1003_PV | 60% | 1003 | H→PV(s) | NA | | | | | | | | | |
| r30v11_f60_s1003_HH | 60% | 1003 | H→H | NA | | | | | | | | | |

## 同格配对差（best / last5 / final / i2t R@1 / t2i R@1）

| 格 | 比较 | best | last5 | final | i2t R@1 | t2i R@1 |
|---|---|---|---|---|---|---|
| f00_s1001 | Ls-L0 | NA | NA | NA | NA | NA |
| f00_s1001 | PV-Ls | NA | NA | NA | NA | NA |
| f00_s1001 | PV-PN | NA | NA | NA | NA | NA |
| f00_s1001 | PV-PB | NA | NA | NA | NA | NA |
| f00_s1001 | PN-H0 | NA | NA | NA | NA | NA |
| f00_s1001 | PV-H0 | NA | NA | NA | NA | NA |
| f00_s1001 | PB-H0 | NA | NA | NA | NA | NA |
| f00_s1001 | L0-H0 | NA | NA | NA | NA | NA |
| f00_s1001 | Ls-H0 | NA | NA | NA | NA | NA |
| f00_s1001 | L0-HN | NA | NA | NA | NA | NA |
| f00_s1001 | L0-HB | NA | NA | NA | NA | NA |
| f00_s1001 | HN-H0 | NA | NA | NA | NA | NA |
| f00_s1001 | HB-H0 | NA | NA | NA | NA | NA |
| f60_s1001 | Ls-L0 | NA | NA | NA | NA | NA |
| f60_s1001 | PV-Ls | NA | NA | NA | NA | NA |
| f60_s1001 | PV-PN | NA | NA | NA | NA | NA |
| f60_s1001 | PV-PB | NA | NA | NA | NA | NA |
| f60_s1001 | PN-H0 | NA | NA | NA | NA | NA |
| f60_s1001 | PV-H0 | NA | NA | NA | NA | NA |
| f60_s1001 | PB-H0 | NA | NA | NA | NA | NA |
| f60_s1001 | L0-H0 | NA | NA | NA | NA | NA |
| f60_s1001 | Ls-H0 | NA | NA | NA | NA | NA |
| f60_s1001 | L0-HN | NA | NA | NA | NA | NA |
| f60_s1001 | L0-HB | NA | NA | NA | NA | NA |
| f60_s1001 | HN-H0 | NA | NA | NA | NA | NA |
| f60_s1001 | HB-H0 | NA | NA | NA | NA | NA |
| f60_s1002 | PV-PN | NA | NA | NA | NA | NA |
| f60_s1002 | PV-PB | NA | NA | NA | NA | NA |
| f60_s1002 | PV-HH | NA | NA | NA | NA | NA |
| f60_s1002 | PN-HH | NA | NA | NA | NA | NA |
| f60_s1002 | PB-HH | NA | NA | NA | NA | NA |
| f60_s1003 | PV-PN | NA | NA | NA | NA | NA |
| f60_s1003 | PV-PB | NA | NA | NA | NA | NA |
| f60_s1003 | PV-HH | NA | NA | NA | NA | NA |
| f60_s1003 | PN-HH | NA | NA | NA | NA | NA |
| f60_s1003 | PB-HH | NA | NA | NA | NA | NA |
