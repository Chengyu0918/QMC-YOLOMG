# 运行指南：QMC-YOLOMG vs 原版 YOLOMG 全量对照实验

工作目录：`Z:\cy\code\YOLOMG-main\change\YOLOMG-main`（VSCode 打开这个文件夹作为工作区）
环境：conda `adaptivenn`（已装好 torch 2.5.1+cu121 等全部依赖）

## 0. 已经就绪、不需要重跑的东西

| 内容 | 位置 | 状态 |
|---|---|---|
| 全量真实数据（19.9 万帧） | `datasets/ARD100_full/` | ✅ 已生成完毕（用了 28 分钟） |
| 训练清单（隔 3 帧取 1，38,426 帧） | `datasets/ARD100_full/train.txt` | ✅ 当前生效版本 |
| 全量训练清单（115,278 帧） | `datasets/ARD100_full/train_all.txt` | 备用，想用全量见 §5 |
| val 全量 11,962 / test 全量 71,468 | `datasets/ARD100_full/{val,test}.txt` | ✅ |
| 数据配置 | `data/ARD100_full.yaml` | ✅ 路径已写好 |
| 模型配置 | `models/QMC_uav_s.yaml`（改进版）/ `models/YOLOMG_base.yaml`（原版） | ✅ |
| 代码 | train/val/dualdetector/eval_gate_gain/run_experiments3.py | ✅ 全部冒烟通过 |

数据划分与原论文一致：59 个 train 域视频训练（隔帧采样）、6 个 train 域视频做
val（选 best）、35 个 test 域视频共 71,468 帧做最终评估。图像已预缩到长边 640
（与训练 letterbox 完全等价，标签归一化坐标不变），mask 是作者 FD5 管道的
KLT+单应补偿三帧差分。

## 1. 方式一：一键自动跑完全部（推荐）

在 VSCode 里按 F5 选 **"Run ALL full-ARD100 experiments (auto chain)"**，或命令行：

```bat
cd /d Z:\cy\code\YOLOMG-main\change\YOLOMG-main
Z:\cy\anaconda\envs\adaptivenn\python.exe run_experiments3.py
```

它会自动依次执行：① full_base 训练 → ② full_base 测试集评估 →
③ full_qmc 训练 → ④ full_qmc 测试集评估 → ⑤ full_qmc 门控-收益相关性分析，
并把解析好的指标写进 `_results3_final.json`。

预计总时长 **14~16 小时**（base 约 6.5 小时、qmc 约 8.5 小时、评估约 40 分钟），
建议过夜跑。中途可以随时关掉，重跑同一条命令会从头训练该组（如需断点续训见 §4）。

看进度：`_full_base_train.log` / `_full_qmc_train.log`（实时写入），
或者 TensorBoard：`tensorboard --logdir runs\train`。

## 2. 方式二：分步手动跑（想单独控制每组时）

每步都在 `Z:\cy\code\YOLOMG-main\change\YOLOMG-main` 下执行，
python 用 `Z:\cy\anaconda\envs\adaptivenn\python.exe`（下文简写 python）。

### 2.1 原版 YOLOMG（对照组）

```bat
python train.py --data data/ARD100_full.yaml --cfg models/YOLOMG_base.yaml ^
  --weights yolov5s.pt --epochs 18 --batch-size 8 --imgsz 640 --device 0 --workers 8 ^
  --motion-degrade-local 0 --motion-degrade-perturb 0 --motion-gain-weight 0 --consistency-weight 0 ^
  --name full_base --exist-ok
```

### 2.2 完整 QMC-YOLOMG（改进组）

```bat
python train.py --data data/ARD100_full.yaml --cfg models/QMC_uav_s.yaml ^
  --weights yolov5s.pt --epochs 18 --batch-size 8 --imgsz 640 --device 0 --workers 8 ^
  --aux-every 2 ^
  --name full_qmc --exist-ok
```

（默认即：在线质量证据 + 局部衰减 0.25 + 配准扰动 0.15 + Δ 门控监督 0.05 +
一致性约束 0.3，教师 3 epoch 后启动；`--aux-every 2` = 每 2 步做一次教师前向，
大数量集下省一半辅助开销。）

### 2.3 消融组（可选，想补全论文证据表时）

```bat
:: 全图失活基线（ModDrop 式，证明"局部退化优于全图失活"）
python train.py --data data/ARD100_full.yaml --cfg models/QMC_uav_s.yaml ^
  --weights yolov5s.pt --epochs 18 --batch-size 8 --imgsz 640 --device 0 --workers 8 ^
  --motion-degrade-local 0 --motion-degrade-perturb 0 --motion-gain-weight 0 --consistency-weight 0 ^
  --motion-dropout 0.3 --name full_fulldrop --exist-ok

:: 普通门控（无质量证据，证明"配准质量证据有帮助"）
python train.py --data data/ARD100_full.yaml --cfg models/QMC_uav_s.yaml ^
  --weights yolov5s.pt --epochs 18 --batch-size 8 --imgsz 640 --device 0 --workers 8 ^
  --quality-mode neutral --aux-every 2 ^
  --name full_noQ --exist-ok
```

### 2.4 测试集评估（35 个未见视频，71,468 帧）

```bat
python val.py --weights runs/train/full_base/weights/best.pt --data data/ARD100_full.yaml ^
  --task test --imgsz 640 --batch-size 8 --device 0 --workers 6 --name eval_full_base --exist-ok

python val.py --weights runs/train/full_qmc/weights/best.pt --data data/ARD100_full.yaml ^
  --task test --imgsz 640 --batch-size 8 --device 0 --workers 6 --name eval_full_qmc --exist-ok
```

（注意：noQ 组评估必须加 `--quality-mode neutral` 与训练一致。）
输出看日志里的 `all  71468  71468  P  R  mAP@.5  mAP@.5:.95` 行和
`Speed: ... ms inference` 行。

### 2.5 门控-收益相关性（创新点 2 的证据实验）

```bat
python eval_gate_gain.py --weights runs/train/full_qmc/weights/best.pt ^
  --data data/ARD100_full.yaml --split test --imgsz 640 --batch-size 8 --device 0 ^
  --name full_qmc
```

输出 Pearson/Spearman 相关系数与逐样本 CSV（`runs/gate_gain/full_qmc/gate_gain.csv`）。
**判读**：相关系数显著为正（比如 > +0.3）即"门控学到了运动贡献"的证据；
Δ 分布（delta mean±std）应正负分化——运动信息时有益时有害。

### 2.6 推理演示（单帧双模态检测）

```bat
python dualdetector.py --weights runs/train/full_qmc/weights/best.pt --imgsz 640
```

## 3. 结果在哪里看

| 内容 | 位置 |
|---|---|
| 每组最终指标（自动解析） | `_results3_final.json` |
| 训练曲线 | `runs/train/<组名>/results.csv`、TensorBoard |
| 权重 | `runs/train/<组名>/weights/best.pt`（best 按 val 选） |
| 测试评估明细 | `_full_*_val.log`、`runs/val/eval_*` |
| 门控-收益数据 | `runs/gate_gain/<组名>/report.txt` + `gate_gain.csv` |
| 已完成的子采样版对照（供参考） | `RESULTS-QMC.md`（2259 帧版本，2,259/84/2,042） |

跑完 full 版后把 `_results3_final.json` 的数字填进 `RESULTS-QMC.md` 新表即可。

## 4. 常用操作

- **断点续训**：`python train.py --resume runs/train/full_qmc/weights/last.pt`
- **显存不够/想更快**：`--batch-size 4`（其余不变，梯度累积自动补偿）
- **跑不动 18 epoch**：改 `--epochs 12`（时间约 2/3）；两组必须用相同设置对比才有效
- **Windows 下 workers 报错**：把 `--workers 8` 降到 `4`

## 5. 想用完全量（不隔帧）或更大分辨率

当前 train.txt 是隔 3 帧取 1（38,426 帧，为单卡时间预算折衷）。想更接近原文：

```bat
:: 用全量 115,278 帧训练（时间 ×3，约 18h/组）
copy /Y datasets\ARD100_full\train_all.txt datasets\ARD100_full\train.txt
copy /Y datasets\ARD100_full\train_all.txt 的第 2 列流同理：
copy /Y datasets\ARD100_full\train2_all.txt datasets\ARD100_full\train2.txt
del datasets\ARD100_full\train.cache datasets\ARD100_full\train2.cache
```

:: 1280 分辨率（原文设置；batch 必须降 4，约 4× 时长）
在 2.1/2.2 命令里改 `--imgsz 1280 --batch-size 4`，并把 §2.4/2.5 的 `--imgsz` 同步改 1280。
注意 1280 时建议同时把 mosaic 关掉（加 `--hyp data/hyps/hyp.ard100.yaml`），
且我们 64GB 内存下 8 workers 解码 1080p 原图（此时不要用预缩数据——重新用
`--edge 0` 生成数据）才是严格意义的原文复现；粗略预估每组 20~24 小时。

## 6. 预期与判读（诚实版）

- 38k 帧 × 18 epoch 的见次约为原文（115k×100ep）的 1/17，mAP 仍会低于论文的 95%，
  但数据分布、视频划分、mask 管道与原文一致，**两组同条件对比**有效；
- 重点看三个相对结论：
  1. full_qmc vs full_base 的 mAP@.5 / Recall 差；
  2. （若跑消融）qmc > fulldrop、qmc > noQ 是否仍成立；
  3. gate-gain 相关系数是否显著为正（比子采样版的 +0.06~+0.19 更强）。
- 若 full_qmc 低于 full_base，说明该预算下辅助目标代价仍占主导——这本身也是
  可报告的诚实结果，配合"原文预算复现"命令（§5 的 1280 全量）继续补证。
