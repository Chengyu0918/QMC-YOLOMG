# 最终设置配置表（第 1 步核对，2026-10-08）

对象：`runs/train/fullall_base`（A：原版融合）与 `runs/train/fullall_qmc`（C：v6+实际 Q）。
逐项来源：两 run 的 `opt.yaml`/`hyp.yaml` 全字段比对、数据列表 MD5、代码核对。

## 1. 完全一致的项（程序化比对通过）

| 项 | 值 | 证据 |
|---|---|---|
| 训练/验证/测试列表 | 同一份（train_all.txt md5 9fc24db6…、val 27813889…、test 44a067e5…，双流列表成对一致） | MD5 |
| 训练帧数 | 115,278（全部标注帧，无隔帧） | 行数 |
| epochs / batch / nbs | 20 / 12 / 12 | opt.yaml |
| 梯度累积 | accumulate = round(12/12) = 1（每步更新）；warmup 插值区间 [1,1] 恒为 1 | 代码+opt |
| 有效参数更新次数 | ~9,607 步/epoch × 20 ≈ **192,140 次**（两组一致） | 推算 |
| 学习率 | lr0=0.01 线性衰减至 lr0×lrf(0.01)，SGD momentum 0.937，wd 5e-4（×batch×accum/nbs=×1） | hyp.yaml |
| 增强 | mosaic 1.0、fliplr 0.5、hsv、scale 0.5、translate 0.1（其余关闭），双流同步 | hyp.yaml |
| 初始化 | yolov5s.pt（形状匹配迁移，双流结构两组可迁移集合一致）+ seed=1 | opt.yaml |
| 权重选择 | best = 验证集 fitness 最大（w=[0,0,0.1,0.9]，即 0.1·AP50+0.9·AP50:95），每轮验证 | metrics.py |
| **U 贡献监督** | **关闭**（motion_gain_weight=0，teacher_warmup 未触发） | opt.yaml |
| **定向退化** | **关闭**（local=0、perturb=0） | opt.yaml |
| **一致性损失** | **关闭**（consistency_weight=0） | opt.yaml |
| 整图失活 | 关闭（motion_dropout=0） | opt.yaml |

两组 opt.yaml 唯一差异字段：`cfg`（models/YOLOMG_base.yaml vs models/QMC_uav_s.yaml）与
name/save_dir。

## 2. 结构与 Q 的实际用法（代码核对）

**A（fullall_base）**：Concat3 自适应融合（原版），不接收任何 Q 输入
（quality_mode='online' 对其无效，仅是遗留默认值）。

**C（fullall_qmc）**：QMCFusion = **精确复刻 Concat3 基线路径** + 零初始化有界修正项：
`F = Concat3(F_m, F_r) + G·0.1·tanh(corr(F_r, F_m, Q))`。
Q 为**在线估计**（compute_quality_evidence：从净化后的运动差分图计算三通道证据），
单一入口 `q → q_proj(3→4ch) → 拼入 evidence_in`，同时进入 corrector、gate、
quality_head 三个头，**无部分中性化**。gate 初始 bias=4.0（sigmoid≈0.982）、
corrector 末层零初始化 → 训练起点与 A 数值完全一致。quality_head 为悬空旁路头
（U 关闭时无监督、不影响主路径）。

## 3. 预处理与评估口径

- 双模态输入：RGB + 复刻 FD5 差分图（KLT+RANSAC 单应、中间帧基准三帧差分），
  经净化（阈值 12、边界 12px 置零、增益 3）对齐原文样本统计（背景均值 0.44/max 71）。
- 在线 Q 在 dataloader 之后、GPU 上从（增强后的）差分图计算，与输入天然对齐。
- 评估：val.py 标准口径 conf=0.001、NMS IoU=0.6、AP 为置信度扫描；test 仅在
  最终设置上评估一次（A：0.263/0.130；C：0.271/0.137）。
- 结构口径：**本文两组均用 P2–P5 四检测头 + 初始 anchor 5–64px（autoanchor 重聚类）**，
  系 NPS_uav_s.yaml 派生结构，而非原作者 ARD100 专用 dual_uav2.yaml（P3–P5 三头、
  anchor 3–26px）。

## 4. 与原文 0.963 差距的候选解释（不止算力）

1. 训练轮数 20 vs 100（差距主因候选，但非唯一）；
2. **检测头配置不同**：本文四头 P2–P5 vs 原文 ARD 三头 P3–P5+更小 anchor——
   结构口径即不一致，不能视为同一模型的严格复现；
3. 差分图为本文复刻+净化，与作者发布 mask32 的生成细节存在分布差异；
4. batch 12 vs 原文 16（DDP）、有效超参标度不同；
5. anchor 初始化与重聚类差异。
其中 2 表明：**本文 A/C 对照内部有效，但与原文数字的比较只能作为量级参照**，
后续如需严格对标原文，应以其 ARD 配置（dual_uav2）重训基线。

## 5. 第 2 步 Q 对照结果与判定（2026-10-09）

三组同协议（115,278 帧 / 20ep / batch12 / seed 1 / U·退化·一致性全关）：

| 组 | val AP50 / AP50:95 | test AP50 / AP50:95 | test P / R |
|---|---|---|---|
| A 原版 Concat3 | **0.419 / 0.240** | 0.263 / 0.130 | 0.381 / 0.277 |
| B v6 + 中性 Q（0.75 常量） | 0.406 / 0.233 | 0.255 / 0.126 | 0.349 / 0.300 |
| C v6 + 在线 Q | 0.410 / 0.238 | **0.271 / 0.137** | **0.422** / 0.269 |

**净效应分解（同结构/同入口唯一变量差）：**
- **Q 的净效应（C−B）**：test AP50 **+1.6pp**、AP50:95 +1.1pp、**P +7.3pp**
  （0.422 对 0.349）；val +0.3pp。质量证据的收益集中在跨视频泛化与误检抑制。
- **结构净效应（B−A）**：test −0.8pp、val −1.2pp。残差修正结构本身（中性 Q 时）
  无正收益。
- C 对 A 的 test 优势（+0.8pp）= 结构（−0.8）+ Q（+1.6），分解自洽。

**分支判定**：落在"C 稳定优于 B、A"与"B、C 无稳定优势"之间——
test（35 个未见视频）上 C 最优且 C−B 优势（+1.6pp）为 C−A（+0.8pp）的两倍，
**质量证据是反超的来源**；但 val（6 个训练域视频）上 A 仍领先 0.9pp，
"稳定优于"需按第 3 步多随机种子（0/17/42）确认后方可写入结论。
论文主线候选相应为"质量证据引导的残差修正融合"，结构（零初始化有界修正）
作为承载 Q 的必要组件保留、不单独主张收益。

次级观察：B 的 test P（0.349）明显低于 A（0.381）与 C（0.422）——中性常量 Q
条件下修正项可能引入额外误检，佐证 Q 提供的信息具有实际选择性。

全部原始记录：`runs/train/fullall_{base,v6_neutralQ,qmc}/results.csv`、
`_fullall_*_{val,test}.log`。
