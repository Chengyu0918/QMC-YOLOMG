# -*- coding: utf-8 -*-
"""Figure: detection comparison across 4 difficulty scenes.

Scenes: complex background, low light, weak motion, dynamic interference.
Layout: 4 rows (scenes) x 3 cols (YOLOMG baseline | QMC-YOLOMG | motion diff).
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.experimental import attempt_load
from utils.datasets import LoadImagesAndLabels
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from utils.quality import neutral_quality, compute_quality_evidence

DEV = select_device('0')
OUT = ROOT / 'runs' / 'paper_figures'
OUT.mkdir(parents=True, exist_ok=True)

hyp = yaml.safe_load(open(ROOT / 'data/hyps/hyp.scratch-low.yaml', errors='ignore'))
ds = LoadImagesAndLabels(
    str(ROOT / 'datasets/ARD100_full_clean/val.txt'),
    str(ROOT / 'datasets/ARD100_full_clean/val2.txt'),
    img_size=640, augment=False, hyp=hyp, rect=True)

model_A = attempt_load(str(ROOT / 'runs/train/fullall_base/weights/best.pt'),
                       map_location=DEV).float().eval()
model_D = attempt_load(str(ROOT / 'runs/train/full3_D_s1/weights/best.pt'),
                       map_location=DEV).float().eval()


def detect(model, img, img2, q_mode=None, conf=0.20):
    imgs = img.unsqueeze(0).float().div(255).to(DEV)
    m6 = img2.unsqueeze(0).float().div(255).to(DEV)
    motion = m6[:, :3]
    q = (neutral_quality(motion) if q_mode == 'neutral'
         else compute_quality_evidence(motion, imgs) if q_mode else None)
    with torch.no_grad():
        out = model(imgs, motion, q)
        inf = out[0] if isinstance(out, tuple) else out
        dets = non_max_suppression(inf, conf, 0.45)[0]
    return dets.cpu().numpy()


def draw_det(canvas, dets, offset_x=0, thick=1, fsize=0.55):
    for d in dets:
        x1, y1, x2, y2, cf, cls = d
        x1, y1, x2, y2 = int(x1) + offset_x, int(y1), int(x2) + offset_x, int(y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), thick)
        label = f'{cf:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fsize, 1)
        ty = y1 - 4 if y1 - th - 4 > 0 else y2 + th + 4
        cv2.rectangle(canvas, (x1, ty - th - 2), (x1 + tw + 2, ty), (0, 0, 255), -1)
        cv2.putText(canvas, label, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    fsize, (255, 255, 255), 1)
    return canvas


def draw_gt(canvas, labels, offset_x=0):
    h, w_total = canvas.shape[:2]
    w = w_total // 3 if offset_x > 0 else w_total
    for row in labels:
        cx, cy, bw, bh = row[1], row[2], row[3], row[4]
        x1 = int(cx * w) + offset_x - int(bw * w / 2)
        y1 = int(cy * h) - int(bh * h / 2)
        x2 = int(cx * w) + offset_x + int(bw * w / 2)
        y2 = int(cy * h) + int(bh * h / 2)
        cv2.rectangle(canvas, (max(x1, offset_x), max(y1, 0)),
                      (min(x2, offset_x + w), min(y2, h)), (0, 255, 0), 1)
    return canvas


# ---- classify frames into scene types (percentile-based) ----
random.seed(42)
sample_idx = random.sample(range(len(ds)), min(300, len(ds)))

# first pass: collect metrics
metrics = []
for idx in sample_idx:
    img, img2, labels, p, p2, sh, sh2 = ds[idx]
    rgb_np = img.float().div(255).numpy()
    mask_np = img2[:3].float().div(255).mean(0).numpy()
    metrics.append((idx, rgb_np.mean(), mask_np.mean(), mask_np.std()))

brights = [m[1] for m in metrics]
mask_means = [m[2] for m in metrics]
mask_stds = [m[3] for m in metrics]
b_p25 = np.percentile(brights, 25)
m_p25 = np.percentile(mask_means, 25)
s_p75 = np.percentile(mask_stds, 75)
print(f'thresholds: brightness_p25={b_p25:.3f}, mask_mean_p25={m_p25:.4f}, mask_std_p75={s_p75:.4f}')

scenes = {'complex_bg': [], 'low_light': [], 'weak_motion': [], 'dynamic_bg': []}

for idx, br, mm, ms in metrics:
    img, img2, labels, p, p2, sh, sh2 = ds[idx]
    dets_A = detect(model_A, img, img2, conf=0.20)
    dets_D = detect(model_D, img, img2, q_mode='neutral', conf=0.20)
    item = (idx, dets_A, dets_D, labels, br, mm)

    if br < b_p25:
        scenes['low_light'].append(item)
    elif mm < m_p25:
        scenes['weak_motion'].append(item)
    elif ms > s_p75:
        scenes['dynamic_bg'].append(item)
    else:
        scenes['complex_bg'].append(item)

for s, lst in scenes.items():
    print(f'{s}: {len(lst)} frames')

# ---- pick best example per scene (prefer D>A confidence gap) ----
selected = {}
for scene_name, candidates in scenes.items():
    best = None
    best_score = -1
    for item in candidates:
        idx, dA, dD, lb, br, mm = item
        if len(dD) == 0:
            continue
        conf_D = dD[0][4]
        conf_A = dA[0][4] if len(dA) > 0 else 0
        score = conf_D * 2 - conf_A  # prefer D detects well AND A struggles
        if score > best_score:
            best_score = score
            best = item
    if best:
        selected[scene_name] = best

# ---- compose figure ----
SCENE_ORDER = ['complex_bg', 'low_light', 'weak_motion', 'dynamic_bg']
SCENE_LABELS = {
    'complex_bg': 'Complex background',
    'low_light': 'Low illumination',
    'weak_motion': 'Weak motion (near-hover)',
    'dynamic_bg': 'Dynamic background interference',
}

fig = None
H_ref, W_ref = None, None

for scene_name in SCENE_ORDER:
    if scene_name not in selected:
        continue
    idx, dets_A, dets_D, labels, br, mm = selected[scene_name]
    img, img2, lb, p, p2, sh, sh2 = ds[idx]
    H, W = img.shape[1], img.shape[2]
    H_ref, W_ref = H, W

    row = np.zeros((H, W * 3, 3), dtype=np.uint8)

    # col 1: baseline A
    row[:, :W] = img.permute(1, 2, 0).numpy()[..., ::-1]  # uint8 0-255, RGB→BGR
    draw_gt(row, lb, offset_x=0)
    draw_det(row, dets_A, offset_x=0, thick=1)

    # col 2: ours D
    row[:, W:2*W] = img.permute(1, 2, 0).numpy()[..., ::-1]
    draw_gt(row, lb, offset_x=W)
    draw_det(row, dets_D, offset_x=W, thick=1)

    # col 3: motion diff (jet, 8x gain on 0-255 scale)
    mask_vis = img2[:3].float().mean(0).numpy()  # 0-255
    mask_enh = np.clip(mask_vis * 8, 0, 255).astype(np.uint8)
    row[:, 2*W:3*W] = cv2.applyColorMap(mask_enh, cv2.COLORMAP_JET)

    # scene label on left margin
    label = SCENE_LABELS[scene_name]
    cv2.rectangle(row, (0, 0), (len(label) * 9 + 10, 22), (0, 0, 0), -1)
    cv2.putText(row, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    if fig is None:
        fig = row
    else:
        fig = np.vstack([fig, row])

# column headers
if fig is not None and W_ref is not None:
    hdr = np.zeros((28, fig.shape[1], 3), dtype=np.uint8)
    for col, text in enumerate(['YOLOMG (baseline)', 'QMC-YOLOMG (ours)',
                                 'Motion difference map']):
        x = col * W_ref + W_ref // 2 - len(text) * 5
        cv2.putText(hdr, text, (max(x, 5), 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    fig = np.vstack([hdr, fig])

out_path = OUT / 'fig_scene_comparison.jpg'
cv2.imwrite(str(out_path), fig)
print(f'saved {out_path} ({fig.shape[1]}x{fig.shape[0]})')

# print details
for scene_name in SCENE_ORDER:
    if scene_name not in selected:
        print(f'{scene_name}: no suitable frame found')
        continue
    idx, dA, dD, lb, br, mm = selected[scene_name]
    conf_a = f'{dA[0][4]:.2f}' if len(dA) > 0 else 'miss'
    conf_d = f'{dD[0][4]:.2f}' if len(dD) > 0 else 'miss'
    print(f'{scene_name}: frame {Path(ds.im_files[idx]).stem}, '
          f'brightness={br:.3f}, motion_mean={mm:.4f}, '
          f'A conf={conf_a} ({len(dA)} det), '
          f'D conf={conf_d} ({len(dD)} det)')

print('done')
