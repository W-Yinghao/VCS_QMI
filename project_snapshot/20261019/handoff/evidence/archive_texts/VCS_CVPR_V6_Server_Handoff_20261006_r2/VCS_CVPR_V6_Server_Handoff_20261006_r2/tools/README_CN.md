# 包内辅助工具

这些是交接与小型数学工具，不是新的SSL或测量训练代码。

## 标准库工具

```bash
python tools/verify_package.py
python tools/check_contract.py
python tools/check_r2_sync.py
python tools/intake_repo.py --repo /absolute/path/to/VCS_QMI --out /tmp/mv6_intake.json
```

第一个核对本包manifest和参考文件；第二个检查确定数组上的配对/目标/压缩/分数约定；第三个只读扫描指定Git checkout。第三个只接受仓库外输出路径，不发网络请求，不启动任务，不更改branch/HEAD/index；缺少baseline提交时会记录，不能据此推断报告不存在。

## 重跑原V6十项核对

原脚本依赖NumPy、SciPy；版本信息应由实际环境记录。脚本会把输出写在脚本目录，因此先复制到临时目录，避免修改参考文件：

```bash
TMP_CHECK=$(mktemp -d)
cp reference_v6/evidence/verify_math_and_tables.py "$TMP_CHECK/verify_math_and_tables.py"
python "$TMP_CHECK/verify_math_and_tables.py"
# 输出：$TMP_CHECK/verification_results.json
```

不要在本地完整性核验后向 `reference_v6/` 写入server结果。`checks_r1_history/v6_replayed_verification_results.json` 是r1当时的有限代数/报告数组复核记录，不是server科学实验。

## 尚需服务器验证

PyTorch计算图的完整梯度、单机/DDP loss与梯度一致性、真实图像变换与缓存语义、split独立性、conditional null水平、encoder/head来源、现有作业状态均需真实项目检查。本工具的PASS不覆盖这些范围。


## r2增量检查

`check_r2_sync.py`只读取任务依赖、两种比较轴、修订稿已报四行及模板字段；不拟合模型、不检查真实null水平、不调用网络。r2的本地执行记录在checks；r1记录在checks_r1_history，二者不混称同轮验证。
