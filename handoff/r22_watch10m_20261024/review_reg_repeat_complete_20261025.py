"""Complete interpretation of separately audited BF/EPFL repeat series; stdlib only."""
import hashlib
import json
from pathlib import Path
import statistics as S
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    ep = HERE / 'ANALYSIS_REG_REPEAT_EPFL_20261025.json'
    bp = HERE / 'ANALYSIS_REG_REPEAT_BF_20261025.json'
    e, b = read(ep), read(bp)
    assert b['status'] == 'COMPLETE_REVIEW'
    assert e['complete_units'] == b['complete_units'] == 9
    assert e['freeze_sha256'] == b['freeze_sha256']
    assert e['frozen_files_verified'] == b['frozen_files_verified'] == 476
    assert len({u['job'] for d in (b, e) for u in d['units'].values()}) == 18
    assert e['paths'] == 8640 and b['paths'] == 6912
    supplemental = {}
    for encoder, result in e['results']['epfl'].items():
        units = [u for u in e['units'].values() if u['encoder'] == encoder]
        assert sorted(u['seed'] for u in units) == [1001, 1002, 1003]
        methods = {}
        for method in result['method_means']:
            spreads = [x for u in units for x in u['within_path_spreads'] if x['method'] == method]
            methods[method] = dict(
                within_path_repeat_sd_median=S.median(x['sd'] for x in spreads),
                within_path_repeat_sd_mean=S.mean(x['sd'] for x in spreads),
                within_path_repeat_range_max=max(x['range'] for x in spreads),
                curve={str(t): S.mean(u['summaries'][method][str(r)]['curve'][str(t)]
                                     for u in units for r in range(4)) for t in (0, 1, 5, 10, 25, 50, 100)},
                execution_position_means=[S.mean(u['execution_position_path_means'][method][str(i)] for u in units) for i in range(4)])
        selected = {str(u['seed']): S.mean(u['summaries']['POS_NATIVE'][str(r)]['selected_error']
                    - u['summaries']['P1_NATIVE'][str(r)]['selected_error'] for r in range(4)) for u in units}
        supplemental[encoder] = dict(methods=methods, selected_POS_minus_P1_per_seed=selected,
                                      selected_POS_minus_P1_mean=S.mean(selected.values()))
    passed = [(enc, name) for enc, r in e['results']['epfl'].items() for name, c in r['contrasts'].items()
              if c['precision_qualified_directional_signal']]
    assert len(passed) == 6
    assert all(name in ('direction_POS_minus_P1', 'native_POS_minus_P1') for _, name in passed)
    e['supplemental_review'] = supplemental
    e['conclusions_cn'] = [
        'EPFL九项完成，476冻结文件、全部三seed/10CAL窗口(9个scene簇)/两个扰动/三共同起点、四臂四重复100步、源和输入身份、同节点GPU软件精度、实际梯度范数和比例、平衡顺序及RNG未变均通过。每项仅一次成功尝试；8640路径、864000更新、60480行曲线。',
        '方向POS−P1主效应在VCS/BCE/NCE表示上为+0.044393/+0.088138/+0.131187原生px；每种表示三个seed和四个聚合重复均为正，去任一scene仍为正，最弱seed均超过两倍最大重复SD。三个方向比较通过预登记精度筛选，非显著性检验或确认性结果。',
        '原生POS−P1路径误差+0.046987/+0.090181/+0.140393px，也全部通过同一筛选。范数主效应+0.002594/+0.002043/+0.009206px以及交互−0.000808/+0.011998/−0.008550px均未通过；这不是证明范数和交互为零。12比较共6通过，全部是方向和原生差值。',
        '统一末P1选点后POS−P1误差+0.040400/+0.085796/+0.131700px，九个seed均为正；这是次指标，不替代平均三起点的主终点。原生P1/POS严格成功率VCS同为87.0370%、BCE同为85.1852%、NCE88.8889%/88.4259%；所有宽松成功率100%。连续误差变化不能直接说成大失败尾部改善。',
        '相同方向、不同范数的首步误差几乎一致，后续Adam轨迹分岔；方向差异在50/100步更清楚。VCS原生P1/POS误差从共同3.71893降到0.80883/0.85581。范数匹配改变了完整优化路径，不能把方向与范数主效应解释成独立的训练因果贡献。',
        '重复波动存在：各表示/方法单路径SD中位约0.0022–0.0235原生px，最大极差1.30485px；交叉梯度臂波动通常更大。平衡执行位置没有一致单调模式，尚未隔离CUDA算子根因，不能声称完整系统确定性。',
        '与BF的0/12精度筛选通过不同，EPFL的方向信号在这组CAL与固定编码器上可辨明。正对平方项对局部精修的影响并非仅统一缩小梯度；结果支持当前COS/P1局部读出，不支持把POS或完整J作为默认精修替代。四臂不含完整J，不能据此断言参考项普遍有害，也不能推出完整训练应删除平方项。',
        '本组回执1.383254GPUh；BF另为1.030488GPUh，两组2.413742GPUh。共享门禁57.147047秒只计一次，合计已知2.429616GPUh；历史失败/旧实验/CPU/调度启动成本另列，不记零。无TEST读取、不改变已冻结任务。'
    ]
    e['decision'] = dict(status='EPFL_COMPLETE_REVIEW', registered_primary_screen_passed=6,
        registered_primary_comparisons=12, original_negative_findings_preserved=True,
        next='Retain current COS/P1 local readout. No additional registration training, longer trajectory, or repetition batch now: EPFL direction contrast is resolved at this diagnostic scope; BF remains unresolved. Finish shuffled retrieval3; decide whether to admit the existing matched learning-rate design using its complete allocation evidence. A future registration intervention needs a distinct application hypothesis and a fresh frozen matched design, not a permanent stop or a relaxed old screen.')
    e['status'] = 'COMPLETE_REVIEW'
    e['reviewed_utc'] = datetime.now(timezone.utc).isoformat()
    write(ep, e)
    txt = HERE / 'ANALYSIS_REG_REPEAT_EPFL_20261025.txt'
    marker = '\n完整解释与执行判断\n'
    txt.write_text(txt.read_text().split(marker)[0] + marker + '\n\n'.join(e['conclusions_cn'])
                   + '\n\n下一步：保留COS/P1局部读出，不追加配准长训练或重复次数；完成shuffled检索三seed后，按完整分配证据判断已设计的匹配学习率对照。新的配准干预须另有应用假设和冻结门禁。\n')
    gate_seconds = e['cost']['shared_gate_seconds']
    assert gate_seconds == b['cost']['shared_gate_seconds']
    hours = sum(d['cost']['formal_receipt_gpu_hours'] for d in (b, e))
    all_report = dict(status='COMPLETE_REVIEW', utc=e['reviewed_utc'], freeze_sha256=e['freeze_sha256'],
        datasets_reported_separately=True, complete_units=18, paths=15552, adam_updates=1555200,
        evidence=[dict(path=str(p), sha256=sha(p)) for p in (bp, ep)],
        results={'bf': b['results']['bf'], 'epfl': e['results']['epfl']},
        precision_screen_passed={'bf': 0, 'epfl': 6}, comparisons_per_dataset=12,
        cost=dict(formal_receipt_gpu_hours=hours, shared_gate_seconds_counted_once=gate_seconds,
                  known_formal_plus_gates_gpu_hours=hours + gate_seconds / 3600,
                  excluded_not_zero=['CPU check', 'scheduler/startup overhead', 'prior failed attempts and prior studies']),
        conclusion='EPFL direction and native POS-P1 differences remain positive beyond the registered repeat-spread screen across all three encoders; BF contrasts remain unresolved. Retain COS/P1 local readout; do not pool pixel scales, equate lack of detection with zero, or infer a fullJ/training/reference-term conclusion.',
        next=e['decision']['next'], limitations=e['limitations'])
    write(HERE / 'ANALYSIS_REG_REPEAT_ALL18_20261025.json', all_report)
    (HERE / 'ANALYSIS_REG_REPEAT_ALL18_20261025.txt').write_text(
        '固定代码重复性诊断18/18完成，BF/EPFL分别分析。\n'
        'BF12个分项比较均未通过精度筛选；EPFL方向和原生POS−P1各3个比较通过，共6/12。筛选非显著性检验。\n'
        'EPFL方向主效应VCS/BCE/NCE=+0.044393/+0.088138/+0.131187原生px；范数/交互未辨明。BF不因符号混合认定零效应。\n'
        '保留当前COS/P1精修，暂不追加配准训练或重复；等待shuffled检索三seed完整分配证据决定匹配学习率研究。非永久关闭配准研究。\n'
        f'15552路径，1555200更新；正式回执{hours:.9f}GPUh，共享门禁{gate_seconds:.6f}秒只计一次，已知合计{hours+gate_seconds/3600:.9f}GPUh。其他成本不记零。\n'
        '各数据集完整审计/曲线/逐seed/scene及限制见独立BF、EPFL报告；未读取TEST。\n')
    print(json.dumps(dict(status='COMPLETE_REVIEW', passes=passed, formal_gpu_hours=hours,
                          known_formal_plus_gates_gpu_hours=hours + gate_seconds / 3600)))


if __name__ == '__main__':
    main()
