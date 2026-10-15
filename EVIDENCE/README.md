# EVIDENCE 目录：按新协议固化的实验证据

## seed1_evidence.json（2026-10-09 生成）

对应 A/B/C 三个 seed-1 运行的证据固化，内容：

1. **权重哈希**：best.pt / last.pt 的 MD5 与大小
   - fullall_base best.md5 16899d7b…（2,052,357 参数，融合层=Concat3 三键）
   - fullall_v6_neutralQ best.md5 7827bb01…（2,056,974 参数）
   - fullall_qmc best.md5 97cba11e…（2,056,974 参数）
2. **结构指纹**：B 与 C 的 QMCFusion 全部键与形状**逐键一致**（True）；
   A 的融合层为 Concat3 的 3 个注意力键——三组结构与各自声明的配置相符
3. **代码快照哈希**：train/val/models/common/yolo/datasets/quality/
   contribution/augmentations/loss/metrics + 两个 yaml + 数据配置 + hyp，
   共 14 个文件的当前版本 MD5（作为后续运行的前置版本基线）
4. **数据列表哈希**：六个 txt 的 MD5（与 CONFIG-FINAL.md §1 一致）

## 已知缺口（如实记录）

- 三个 seed-1 运行训练时**未保存代码快照**（期间代码被多轮迭代）。版本同一性
  通过以下间接证据核验：opt.yaml/hyp.yaml 全字段比对（仅 cfg/name 不同）、
  训练日志的 Model Summary 参数量（base 2,041,458 / v6 2,044,469 系列）、
  checkpoint 结构指纹逐键比对。无法做到密码学级证明，已按规范记录。
- 自本次固化起，**每次新训练启动前先跑 `test_code/_evidence_freeze.py` 的
  代码哈希部分**，存入本目录，实现后续运行的严格版本可追溯。

## 每次运行的固定产物位置

| 内容 | 位置 |
|---|---|
| 完整命令/配置 | runs/train/<run>/opt.yaml、hyp.yaml（train.py 自动保存） |
| 训练曲线 | runs/train/<run>/results.csv（逐 epoch）、events.out.tfevents |
| 训练日志 | _<run>_train.log（完整 stdout，含逐 epoch 验证行与 aux 行） |
| 验证/测试指标 | _<run>_val.log、_<run>_test.log |
| 权重 | runs/train/<run>/weights/best.pt（验证集 fitness 选）、last.pt |
| 预测结果 | runs/val/eval_* 目录（val.py 生成） |
| 本协议记录 | CONFIG-FINAL.md（含三组结果、2×2 矩阵与判定） |
