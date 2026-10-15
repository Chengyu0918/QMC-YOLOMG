# QMC-YOLOMG

**配准质量与运动贡献联合建模的微小无人机检测方法**

在 YOLOMG 外观—运动双模态框架上实现的改进方法，包含零初始化有界残差修正融合、基于冻结参考网络的局部检测收益监督（U），以及定向退化训练策略。

## 项目结构

```
QMC-YOLOMG/
├── train.py                    # 训练入口（支持 Q/U/退化/一致性全部开关）
├── val.py                      # 验证/测试入口
├── dualdetector.py             # 推理演示
├── eval_gate_gain.py           # 门控-收益相关性评估
├── run_experiments3.py         # 全量对照实验编排
│
├── models/
│   ├── QMC_uav_s.yaml          # 改进模型配置（v6 残差修正融合）
│   ├── YOLOMG_base.yaml        # 原版 Concat3 基线配置
│   ├── common.py               # QMCFusion 模块实现
│   └── yolo.py                 # 模型构建与解析
│
├── utils/
│   ├── quality.py              # 配准质量证据 Q（在线估计/常量/均值）
│   ├── contribution.py         # U 收益监督 + 一致性损失 + 冻结教师
│   ├── datasets.py             # 双模态数据加载（6通道运动+Q）
│   └── loss.py                 # 检测损失
│
├── data/
│   ├── ARD100_fullall.yaml     # 全帧密度数据配置（115,278 训练帧）
│   └── hyps/hyp.scratch-low.yaml
│
├── test_code/
│   ├── prepare_ard100.py       # 数据集重建（FD5 差分管道，11 进程并行）
│   ├── clean_masks.py          # 运动差分图净化
│   ├── prepare_offline_q.py    # 离线真值 Q 生成
│   ├── test_qmc_units.py       # 机制单元测试
│   ├── make_*.py               # 论文图表生成工具
│   └── snapshot_code.py        # 代码版本快照工具
│
├── EVIDENCE/                   # 实验证据链（代码/权重/数据哈希）
├── paper_figures/              # 论文组合图（4 张）
├── paper_single_panels/        # 论文单面板图（30 张）
├── degradation_motion_input/   # 退化+运动输入可视化（60 张）
│
├── TECH-REPORT.md              # 完整技术报告
├── CONFIG-FINAL.md             # 最终配置对照表 + 消融结果
├── PROTOCOL-FROZEN.md          # 冻结实验协议
├── U-REGISTRATION.md           # U（贡献监督）正式登记
├── RUN-GUIDE.md                # 运行指南
└── ...
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 重建数据集（需要 ARD100 原始视频和标注）
python test_code/prepare_ard100.py --out datasets/ARD100_full_clean

# 3. 净化运动差分图
python test_code/clean_masks.py --root datasets/ARD100_full_clean

# 4. 训练（全帧密度、20 epochs、batch 12）
python train.py --data data/ARD100_fullall.yaml \
    --cfg models/QMC_uav_s.yaml \
    --weights yolov5s.pt \
    --epochs 20 --batch-size 12 --imgsz 640 \
    --nbs 12 --device 0

# 5. 验证
python val.py --weights runs/train/<run_name>/weights/best.pt \
    --data data/ARD100_fullall.yaml --task val

# 6. 机制分析（门控-收益相关性）
python eval_gate_gain.py --weights runs/train/<run_name>/weights/best.pt \
    --data data/ARD100_fullall.yaml --split val
```

## 实验结果（seed 1，val 集 11,962 帧）

| 方法 | AP50 | AP50:95 |
|---|---|---|
| YOLOMG（原版） | 0.419 | 0.240 |
| QMC-YOLOMG（完整） | 0.416 | 0.235 |
| **QMC-YOLOMG（常量Q+U）** | **0.421** | **0.244** |

## 关键发现

1. **U（贡献监督）在常量 Q 下有正效应**（+1.4pp），是可主张的创新点
2. **在线 Q 估计在当前设置下无独立收益**（机制分析显示门控无空间分化）
3. **最佳配置为 D 组**（v6 残差修正 + 常量 Q + U 监督）
4. 详细消融矩阵见 `CONFIG-FINAL.md`

## 环境要求

- Python 3.10+
- PyTorch 2.5.1+
- CUDA 12.1+
- OpenCV 5.0+
- 详见 `requirements.txt`

## 数据集

- ARD100 数据集：[BaiduYun](https://pan.baidu.com/s/1ycAoKbzQ1rlzvKr8VRakgw?pwd=1x2z)（提取码：1x2z）
- 原始 YOLOMG 论文：*"YOLOMG: Vision-based Drone-to-Drone Detection with Appearance and Pixel-Level Motion Fusion"*

## License

GPL-3.0（沿用 YOLOv5）
