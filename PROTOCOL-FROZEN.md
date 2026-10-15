# 冻结实验协议（阶段 1 产出，2026-10-09）

本文件为后续所有实验的唯一协议依据。若修改共享前向或初始化方式，旧结果复用
资格需重新判定（见 §5）。

## 1. 本批实验协议（FROZEN-v6-full）

| 项 | 值 |
|---|---|
| 数据 | ARD100（论文视频级划分）：train 59 视频 / 115,278 帧（全帧密度 train_all.txt）；val 6 视频（phantom09/10/28/36/40/101）/ 11,962 帧；test 35 视频 / 71,468 帧 |
| 双模态输入 | RGB 帧同名的净化 FD5 差分图（KLT+RANSAC 单应、中间帧基准三帧差分；净化：阈值 12、边界 12px 置零、增益 3） |
| 划分完整性 | train∩val∩test 视频集合两两不相交（程序化核验）；mask 生成所用 f±2 支持帧天然在同一视频内，无跨集合支持帧 |
| 配对/标签 | RGB 与运动流逐帧同名配对（三集合 100%）；标签 100% 存在、坐标域 [0,1] 且宽高>0，无格式错误 |
| 训练 | 640×640、20 epochs、batch 12、nbs 12 → **accumulate=1（每步更新，~192,140 次）** |
| 优化器 | SGD，lr0=0.01 线性衰减至 1e-4（lrf 0.01），momentum 0.937，weight_decay 5e-4，warmup 3 epochs |
| 增强 | mosaic 1.0、fliplr 0.5、hsv(0.015/0.7/0.4)、scale 0.5、translate 0.1；双流同步（Q 通道不参与 HSV） |
| 初始化 | yolov5s.pt 形状匹配迁移（A: 32/510，B/C: 32/536）+ seed 指定；anchor 由 autoanchor 按训练集聚类 |
| 权重选择 | 每轮验证，best = 验证集 fitness（0.1·AP50 + 0.9·AP50:95）最大；测试集不参与任何选择 |
| 评估 | val.py 标准口径：conf 0.001、NMS IoU 0.6、置信度扫描 AP |
| 运行隔离 | 每次 --name 独立目录（opt.yaml/hyp.yaml/results.csv/权重/完整 stdout 日志），不覆盖 |

## 2. 最终 v6 代码的五个检查项

**① 融合公式**（models/common.py::QMCFusion.forward，代码核对）：
- 基线路径**逐式复刻原版 Concat3**（含原调用参数序：先运动流后外观流）：
  `w = SA(F_m) / (SA(F_r) + 1e-4)`，`base = concat(F_m·(2−w), w·F_r)`
- 修正项：`F_f = base + G · 0.1 · tanh(corr(F_r, F_m, Q))`，
  corrector 末层**零初始化**（起点与 A 数值完全一致），tanh×0.1 有界。

**② Q**：3 通道（q_res/q_cov/q_val），由 `compute_quality_evidence` 从**净化后的运动
差分图在线计算**（非配准真值；真值 Q 存于 q/ 目录但本协议不启用）。数值已 clamp 至
[0,1]，无额外归一化。单一入口：`Q → resize → q_proj(3→4ch) → 拼入 evidence_in`
（=concat(F_r, F_m, qf)），**同时进入 gate、corrector、quality_head 三个分支**；
quality_head 为悬空旁路（U 关闭时无监督、不影响主路径）。B 组以 0.75 常量替换 Q，
其余不变。

**③ 门控 G**：每像素标量（1 通道、sigmoid），初始 bias=4.0（≈0.982）。G 只控制
**修正项的注入幅度**（乘在 tanh 限幅后的修正上）；G→0 时结构退回 Concat3 基线。
G 不作用于基线路径。

**④ 三开关**（opt 字段）：
- U 贡献监督：`--motion-gain-weight`（A/B/C 均 0；教师由 teacher_warmup 门控，未触发）
- 定向退化：`--motion-degrade-local / --motion-degrade-perturb`（均 0）
- 一致性：`--consistency-weight`（0）
- 另：整图失活 `--motion-dropout`（0）

**⑤ 优化/选择**：见 §1（batch/nbs=12/12、accumulate=1、SGD lr0 0.01→1e-4、
best 按验证 fitness）。

## 3. A/B/C（seed 1）配置对照表——基于日志与 checkpoint，非目录名

| 项 | A | B | C |
|---|---|---|---|
| 目录 | fullall_base | fullall_v6_neutralQ | fullall_qmc |
| 日志 cfg | YOLOMG_base.yaml | QMC_uav_s.yaml | QMC_uav_s.yaml |
| 日志 quality_mode | online（Concat3 不接收 Q，无效） | **neutral（0.75 常量）** | **online（实际 Q）** |
| 日志 degrade/aux | 0/0、0/0 | 0/0、0/0 | 0/0、0/0 |
| data/seed/batch | fullall / 1 / 12 | fullall / 1 / 12 | fullall / 1 / 12 |
| 模型摘要 | 403 层 / 2,041,458 | 416 层 / 2,046,024 | 416 层 / 2,046,024 |
| 迁移 | 32/510 | 32/536 | 32/536 |
| 结构指纹 | Concat3 三键 | QMCFusion 26 键 | QMCFusion 26 键，与 B **逐键同形** |
| val AP50/:95 | 0.419 / 0.240 | 0.406 / 0.233 | 0.410 / 0.238 |
| test AP50/:95 | 0.263 / 0.130 | 0.255 / 0.126 | 0.271 / 0.137 |

**共同层初始化核验（如实记录）**：同 seed=1、同预训练权重，但 Concat3 与 QMCFusion
的构建消耗 RNG 的方式不同——**layer 4 之后的共同层初始化抽样序列分歧**（layer 0-3 的
72 个键初始化相同）。事后无法从训练后权重验证初始化相等（训练已改变权重）；
此为结构消融的**固有混淆**，缓解手段是多种子重复（阶段 3），或在后续种子中
实施"共同层初始化注入"（构建后复制 A 的初始化到 C 的共同层再训练）。

## 4. 可复用 / 不可复用清单

**可复用（本协议 seed=1 组成部分）**：fullall_base、fullall_v6_neutralQ、fullall_qmc
——身份核验三重通过（opt.yaml、训练日志身份行、checkpoint 结构指纹）。

**不可复用/不得混入本协议消融**（全部保留为探索记录）：

| 运行 | 不可复用原因 |
|---|---|
| full_base、full_qmc、full_qmc_nodeg_v5、full_qmc_bare_v5、full_qmc_neutralQ_v5、full_qmc_offlineQ_v52 | 38,426 帧隔帧数据（数据量不同）且为 v4/v5 结构 |
| full_qmc_v6、v6_ext12、v6_ext20、full_base_ext12/ext20 | v6 结构但 38k 数据/不同轮数（预算不同） |
| full_qmc_final | v6+退化+辅助开启（配置不同） |
| cmp_*（base/fulldrop/noQ/qmcfull）、fix_base、hd_*、qmc_smoke | 2,259 帧小样本系列 + nbs=64 旧优化设置 |
| ARD100_mask32-640/1280_uavs | 原作者权重的 run 记录（非本文训练，仅作参照） |

## 5. 最终代码快照与复用判定规则

- 快照：`EVIDENCE/seed1_evidence.json` 的 code_snapshot_current（14 个关键文件 MD5）；
  **后续每次训练启动前重新生成代码哈希并存 EVIDENCE/**，日志、配置、权重哈希随 run 固化。
- 复用判定规则：仅当（数据列表 MD5 一致 + opt/hyp 全字段一致 + 日志身份行一致 +
  checkpoint 结构指纹一致 + 代码快照一致或差异被声明不影响该 run 的路径）时，旧结果
  可复用。**若修改共享前向（QMCFusion.forward、Concat3）或初始化方式，本协议全部
  旧结果复用资格自动失效**，需按新版本重训核验组。

## 6. 尚需修复的问题

1. **共同层初始化分歧**（§3）：后续种子实验建议实施初始化注入（从 A 的构建态复制
   共同层随机初始化到 B/C），使结构消融只剩"融合模块本身"的差异；实施前多种子
   结论需标注此混淆。
2. **历史 run 无代码快照**：已用间接证据核验并记录；无法密码学级证明，接受该限制。
3. val 集仅 6 个训练域视频（11,962 帧），对跨视频泛化代表性弱——A/C 在 val/test
   排序相反的事实已出现；阶段 4 的误检专项需在工作阈值下做，不追加 val 视频
   （避免改变已冻结协议）。
4. 训练日志中 AutoAnchor 汇总行存在但本审计正则未捕获（不影响任何结论，仅审计
   脚本细节）。

## 7. 阶段 1 结论

**下一阶段（多种子等）具备条件**：数据/代码/配置/身份/结构五级核验全部通过，
A/B/C(seed1) 判定可复用；协议以本文件为准冻结。
