# 事故记录（按规范保留，含失败与修复）

## INC-20261009-01：watcher 误触发导致双训练挤爆（阶段 3/4 交界）

- **时间**：2026-10-09 18:25–18:27
- **经过**：阶段 4 的 M_s1 排队 watcher（`_stage4_watcher.bat`）在等待条件
  `_full3_E_s1_val.log 含 ALL_DONE` 上判断失效——目标文件不存在时 PowerShell
  条件未保持等待，watcher 挂上后**立即**启动了 M_s1。此时 D_s1 仍在训练
  （epoch 6/19），两个 115k 全量训练同时争用 16GB GPU 与系统内存。
- **后果**：M_s1 启动即崩（EOFError + MemoryError，日志 14KB，无有效进度）；
  D_s1 于 epoch 6 中断（last.pt 保存于 epoch 5）。
- **处置**：
  1. 清理全部僵死进程（42→0）；
  2. M_s1 误启动目录删除，失败日志保留为
     `_full3_M_s1_FAILED_watcher_misfire.log`；
  3. 废弃独立 watcher 机制，改为**单文件顺序链**
     `_stage3_resume_chain.bat`（D_s1 从 epoch 6 断点续训 → E_s1 → M_s1，
     串行排队，无并发可能）；
  4. D_s1 续训自动从 last.pt 恢复（teacher_weights 等全部参数在 opt.yaml 内）。
- **损失**：D_s1 约 1 个 epoch（epoch 6）的训练时间；无数据或配置损坏。
- **教训**：单卡排队必须用串行 bat 链而非"等待文件出现"的 watcher；
  watcher 类逻辑需对"文件不存在"显式分支。

## 附：同日早些时候的两次统计失败（已在阶段 4 汇报中记录）

- CPU 11 进程版 compute_q_means 与训练争核僵死（进程清理后 D_s1 恢复）；
- GPU 多线程版产出错误均值 0.026（经 320 帧 mini 验证定位为线程结构 bug，
  弃用；失败脚本 `compute_q_means_gpu.py` 与 `_q_means3.log` 保留）。
  最终采用串行 GPU 版（`compute_q_means_serial.py`），冻结值
  [0.8355, 0.9109, 0.8465] 通过 sanity check。
