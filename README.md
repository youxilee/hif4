# HiF4 高精度 4Bit 数值转换方案

2026 华为算法大赛赛题：将 NVFP4（carrier + block scale）数据转换为 HiF4
格式，目标是在 Linear 与 Attention 两条路径上让 HiF4 反量化后的输出
尽可能接近 NVFP4 参考输出（MSE 相对标准 HiF4 的改善即为得分）。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `solution.py` | 比赛提交主体：HiF4 校准与动态量化全部实现（唯一需要提交的文件） |
| `nvfp4_sim.py` | 权威 NVFP4 模拟器：E4M3 向上取整 scale + E2M1 就近取偶 carrier（cuDNN 官方配方） |
| `real_data_eval.py` | 真实 GPT-2 端到端评测：按比赛口径输出 Linear / Attention 得分 |
| `e2e_eval.py` | 小型 Transformer 端到端冒烟评测（logits 误差 + top1） |
| `sweep_eval.py` | 参数/算法开关扫描：真实数据只加载编码一次，逐配置重跑校准+打分 |
| `shape_smoke.py` | 非 GPT-2 几何冒烟：head_dim 128 + GQA、宽 Linear（1536/6144 通道） |
| `time_breakdown.py` | 校准 vs 每样本动态量化耗时拆解，用于 5 分钟预算评估 |
| `test_solution.py` | 离线回归测试：确定性、合法性、重建误差 |
| `block_smooth_probe.py` | 块正交 SmoothQuant 探针：隔离比较 identity 与 H4/H8/H16 |

## 运行方式

```bash
# 离线回归（纯 CPU，无网络）
/opt/anaconda3/bin/python3 test_solution.py

# 真实 GPT-2 评测（original vs current）
/opt/anaconda3/bin/python3 real_data_eval.py --mode amax6 --config both

# 配置扫描
/opt/anaconda3/bin/python3 sweep_eval.py --layers 12
```

## 版本管理约定

- 每个逻辑改动一个 commit，message 用中文一句话概括改动与动机。
- 每个版本的算法要点与实测得分记录在 `CHANGELOG.md`，与 commit 对应。
- `solution.py` 顶部常量即当前版本的参数配置，改动时同步更新 CHANGELOG。
- 提交前必须跑通 `test_solution.py`；评测脚本的对比结果一并记录。
