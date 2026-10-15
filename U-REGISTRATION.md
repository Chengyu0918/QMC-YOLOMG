# U（运动贡献监督）正式登记与教师协议（阶段 2 产出，2026-10-09）

## 1. U 的正式定义（FROZEN）

**登记名称：任务收益辅助修正监督（auxiliary task-gain supervision for the
correction gate）**。计算式沿用原稿：

    Δ_i = L_i(零运动输入) − L_i(完整运动输入)

两次前向均由**冻结参考教师**执行，使用相同增强后的输入批次、相同标注、
相同匹配位置（取自完整运动前向的一次 build_targets，含跨层 anchor 匹配与
固定困难背景 cell）与相同逐位置损失定义（box CIoU / objectness / classification /
hard-negative objectness，各项乘以与主检测损失一致的超参权重）。

**语义边界（按阶段 2 要求明确，不得越界表述）**：
1. 零运动前向 = 运动差分图置零后仍经 Concat3 基线路径（F_m 以零图特征参与），
   **不是纯 RGB 单流网络**；
2. Δ 度量的是**整条运动通路（Concat3 基线中的 F_m 与修正项之和）对局部检测
   损失的净效应**，而 v6 的 G 只控制残差修正项的注入幅度——**G→0 时退回含完整
   运动流的 Concat3 基线，G 不能关闭运动分支**；
3. 因此该标签是**任务收益的辅助监督信号**，不是修正项收益的真值，不得声称
   "G 学到了关闭运动的能力"。

## 2. 参考教师协议（FROZEN）

| 项 | 值 |
|---|---|
| 教师来源 | **组 A（无门控 Concat3）seed-1 训练权重** `runs/train/fullall_base/weights/best.pt`，md5 `16899d7bc98d…`（EVIDENCE/seed1_evidence.json 一致） |
| 选择依据 | A 即"无待学习门控的固定融合"参考网络（v6 基线路径=原版融合）；无额外预热成本，权重与哈希已固化，可跨组共享 |
| 加载方式 | `--teacher-weights <ckpt>`：训练开始前一次性加载（`ContributionTeacher.from_checkpoint`），eval+requires_grad False |
| 冻结保证 | **全程不刷新**（refresh 方法对 checkpoint 教师为 no-op；epoch 哨兵使其不可达）；参数与 BN 统计与学生训练完全解耦 |
| 共享 | 同一种子内全部 U 开启组使用**同一教师文件**（哈希写入训练日志与 EVIDENCE 快照） |
| 额外成本 | 教师前向每 `--aux-every` 步 2 次（完整/零运动）；与旧实现相同 |

**与旧实现的区别（修复记录）**：旧实现（38k 系列）从学生 EMA 深拷贝并
**每 epoch 刷新**——违反第 3/6 条（教师间接随学生训练改变），且教师欠训练导致
Δ 信号微弱（±0.01 量级）。本修复后教师为 0.42 水平的独立网络，小样本验证中
Δ 信号充分（见 §4）。旧实现的任何 U 结论不迁移到新协议。

## 3. 六项检查结果（对照阶段 2 要求）

| # | 要求 | 结果 | 证据 |
|---|---|---|---|
| 1 | 教师不用学生门控生成标签 | PASS | 教师为独立 Concat3 网络，自测 Δ；代码结构保证 |
| 2 | 同种子 U 组共享同一教师 | PASS（协议级） | 统一 `--teacher-weights` 文件 + 哈希入日志/EVIDENCE |
| 3 | 预热后全程冻结、不每 epoch 刷新 | PASS（修复后） | refresh 对 checkpoint 教师 no-op；日志无刷新记录 |
| 4 | 两次前向同输入/匹配/损失定义 | PASS | forward_passes 单批两次前向、build_targets 一次、同 hyp 权重 |
| 5 | 中性带/归一化/困难背景/权重固定 | PASS | near_zero=0.05、margin=0.01、tau=0.1、hard_neg top-8、正负分组归一化，均为代码常量未改动 |
| 6 | 停止梯度、教师不被学生修改 | PASS | Δ 计算 no_grad、教师参数 requires_grad False、外部文件与训练解耦（文件哈希前后一致） |

## 4. 小样本验证（56 train / 8 val 帧，真实 train.py 4 epochs）

运行：`runs/train/u_smoke_v6`，日志 `_u_smoke_train.log`，命令含于日志首行。

| 检查 | 结果 |
|---|---|
| 损失有限（无 NaN/爆炸） | PASS |
| U 有效监督非长期全空 | PASS（Δ ∈ [0.166, 0.313]，全非零） |
| 门控参数梯度 > 0 | PASS（gate 参数梯度范数正向） |
| 教师全程不变 | PASS（文件 md5 前后一致 + 结构性解耦） |
| 权重保存/加载 | PASS（checkpoint 重载输出逐位一致） |
| 教师来自 checkpoint 且未刷新 | PASS（日志含 protocol, frozen + checkpoint: 行，无 LEGACY 行） |

**全部通过，具备进入正式矩阵的条件。**

## 5. 产物

- 本登记：`U-REGISTRATION.md`
- 代码修改：`utils/contribution.py`（from_checkpoint + refresh no-op）、
  `train.py`（--teacher-weights、一次性加载、移除 per-epoch 刷新路径）
- 验证运行：`runs/train/u_smoke_v6/`（保留）、`_u_smoke_train.log`（保留）
- 代码快照：`EVIDENCE/code_stage2.json`
- 小样本数据（临时，可删）：`datasets/_u_smoke/`、`data/_u_smoke.yaml`
