"""Interpret both complete trajectory audits; keep dataset scales separate."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

W=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

ep=W/'ANALYSIS_REG_TRAJECTORY_EPFL_20261025.json';e=read(ep)
assert e['status']=='COMPLETE_REVIEW' and e['complete_units']==9
assert all(c['positive_square']['negative_seeds']==0 and c['positive_square']['registered_directional_signal'] for c in e['contrasts'].values())
e['conclusion']='On all3 representations and all3 seeds, POS_QUAD increases primary path error relative to P1; scene leave-one-out signs remain positive. The mean increases are0.040866/0.081150/0.135491 native pixels for exact/BCE/NCE. FullJ-P1 also increases all9; most of the mean endpoint difference is already present without Q, but this additive contrast alone is not an isolated gradient-mechanism decomposition. Initial step errors almost equal; later paths diverge. Exact-trained Q contribution has mixed scene sensitivity and opposite mean path/own-selection signs. No universal harmful reference claim.'
e['analyzer_sha256']=sha(W/'analyze_reg_trajectory_20261025.py');write(ep,e)
txt=W/'ANALYSIS_REG_TRAJECTORY_EPFL_20261025.txt'
text=txt.read_text().replace('完成另一数据集后并列判断，不从单数据集或单seed提前改A1/训练配方。','与已完整BF并列：正对平方项在EPFL的误差增加一致，BF则依表示与候选选择变化；不修改既有A1/KD配方。')
txt.write_text(text+'完整判断：正对平方项相对P1使exact/BCE/NCE的共同起点终点误差分别+0.040866/+0.081150/+0.135491原生像素，九个seed单元均同向，LOSO均同向；完整J相对P1也九单元均较差。参考项增加的均值较小，exact表示上还存在路径与自身能量排序方向相反。首步误差近同、10至100步分开，初始梯度范数不足以区分方向与尺度的动态作用。此为固定CAL位姿优化机制，不等同训练损失机制或新确认性性能。\n')
bf=W/'ANALYSIS_REG_TRAJECTORY_BF_20261025.json';b=read(bf)
cost=b['cost']['receipt_gpu_hours']+e['cost']['receipt_gpu_hours']
summary=dict(status='COMPLETE_REVIEW',utc=datetime.now(timezone.utc).isoformat(),complete_units=18,
    sources={p.name:sha(p) for p in [bf,ep,W/'REG_TRAJECTORY_BF_CURVES_20261025.csv',W/'REG_TRAJECTORY_EPFL_CURVES_20261025.csv']},
    experiments='18fixed30k checkpoints; BF8cores/EPFL10windows;2perturbations;3sharedCOSstarts;P1/POS_QUAD/J each100freshAdam steps. 324problems,2916paths,291600poseupdates.',
    results={ds:dict(means=d['scene_equal_group_means'],contrasts=d['contrasts']) for ds,d in [('bf',b),('epfl',e)]},
    cost=dict(full_receipt_gpu_hours=cost,shared_new_gate_receipt_seconds=11.343969345092773,
        known_success_receipt_gpu_hours=cost+11.343969345092773/3600,old_failed_gate_scheduler_seconds=8,
        note='Old failed scheduler cost and successful receipt costs have different scopes; CPU/source training and unmeasured scheduler overhead separate.'),
    conclusion='EPFL positive-square precision loss is consistent across3representations and3seeds; BF effects vary and own-energy ranking can reverse mean-path comparisons. Reference terms are not uniformly harmful. No pooled dataset claim.',
    next_hypothesis='Separate changing gradient direction from scalar norm history using a2x2 intervention: directionP1/POS_QUAD x normP1/POS_QUAD at each current pose. Reuse two existing natural trajectories and add only two crossed trajectories, same fixed CAL inputs, cached starts,100steps and Adam. No new encoder training or objective-theory change.',
    limits='Development CAL reused; starts/perturbations correlated; no newTEST; norm swapping is a mechanism intervention, not completeVCS or a proposed production loss.')
write(W/'ANALYSIS_REG_TRAJECTORY_ALL18_20261025.json',summary)
(W/'ANALYSIS_REG_TRAJECTORY_ALL18_20261025.txt').write_text('''配准100步轨迹18/18完整结论（2026-10-25）
全部冻结、模型/输入/起点身份、4096位置、100步真实Adam状态、访问域、有限值、单次尝试和成本审计通过。两数据集分别报告。
EPFL：正对平方项相对余弦P1使exact/BCE/NCE表示的三起点平均末误差+0.040866/+0.081150/+0.135491原生像素，三表示×三seed全部为正，逐scene留一均同向。fullJ相对P1也九单元均更差。自身能量选终点的误差同样增加；严格成功率多数不变，NCE1003的J下降一个CAL问题。
BF：正对平方项在VCS表示使三起点均值略改善，却让自身选终点更差；参考项对VCS路径较差、对BCE路径较好。fullJ−P1三表示均未通过seed方向+LOSO的描述性信号。全部选终点严格/宽松100%，不能说明DEV尾部改善。
两数据集都表明路径与排序要分开。EPFL的主要端点差在没有Q时已出现，但不能仅由加法差分断言正对平方项通过何种梯度机制致损；BF也不支持普遍删除参考项。
下一项：梯度方向×梯度范数的2×2机制对照，复用P1/POS_QUAD现有自然轨迹，只补两个交换臂。每一步在各自当前位姿计算两梯度，比较保留方向而交换范数、保留范数而交换方向；同输入/起点/100步Adam。主指标继续同三起点平均误差；次指标统一COS能量选择，避免新增合成梯度没有独立标量能量的问题。先冻结和技术门禁，再入原池。
原论文理论不变；交换梯度只标机制控制。继续A1、shuffled-U和KD固定预算。CAL是反复开发集，seed仅三个，全部scene保留，不把扰动/起点当独立样本。
成本：18正式回执0.369811GPUh；两成功门禁11.343969秒计一次，成功回执合计0.372962GPUh。旧失败门禁调度8秒单列，CPU/源训练/其他历史成本另计。
''')
print('REG_ALL18_COMPLETE',cost)
