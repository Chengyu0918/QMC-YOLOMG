# -*- coding: utf-8 -*-
"""Generate paper Figure 2 (detection results + failure cases) and
Figure 3 (motion diff, Q evidence, gate visualization correspondence).

Outputs: runs/paper_figures/fig2_detections.jpg, fig3_mechanism.jpg
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
from utils.quality import compute_quality_evidence, neutral_quality

DEV = select_device('0')
OUT = ROOT / 'runs' / 'paper_figures'
OUT.mkdir(parents=True, exist_ok=True)

hyp = yaml.safe_load(open(ROOT / 'data/hyps/hyp.scratch-low.yaml', errors='ignore'))
ds = LoadImagesAndLabels(
    str(ROOT / 'datasets/ARD100_full_clean/val.txt'),
    str(ROOT / 'datasets/ARD100_full_clean/val2.txt'),
    img_size=640, augment=False, hyp=hyp, rect=True)

# best model
model_D = attempt_load(str(ROOT / 'runs/train/full3_D_s1/weights/best.pt'),
                       map_location=DEV).float().eval()
# baseline for comparison
model_A = attempt_load(str(ROOT / 'runs/train/fullall_base/weights/best.pt'),
                       map_location=DEV).float().eval()


def detect(model, img, motion6, q_mode='neutral', conf=0.25):
    imgs = img.unsqueeze(0).float().div(255).to(DEV)
    m6 = motion6.unsqueeze(0).float().div(255).to(DEV)
    motion = m6[:, :3]
    q = (neutral_quality(motion) if q_mode == 'neutral'
         else compute_quality_evidence(motion, imgs) if q_mode else None)
    with torch.no_grad():
        out = model(imgs, motion, q)
        inf = out[0] if isinstance(out, tuple) else out
        # also grab gate if available
        gate = None
        for m in model.modules():
            if type(m).__name__ == 'QMCFusion' and hasattr(m, 'last_gate') and m.last_gate is not None:
                gate = m.last_gate.detach()
        dets = non_max_suppression(inf, conf, 0.45)[0]
    return dets.cpu().numpy(), gate


def draw_det(canvas, dets, color=(0, 0, 255), W=0, thick=2, fsize=0.6):
    for d in dets:
        x1, y1, x2, y2, cf, cls = d
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(canvas, (x1 + W, y1), (x2 + W, y2), color, thick)
        label = f'{cf:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fsize, 1)
        ty = y1 - 4 if y1 - th - 4 > 0 else y2 + th + 4
        cv2.rectangle(canvas, (x1 + W, ty - th - 2), (x1 + W + tw + 2, ty), color, -1)
        cv2.putText(canvas, label, (x1 + W + 1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    fsize, (255, 255, 255), 1)
    return canvas


def draw_gt(canvas, labels, W=0, H_off=0):
    for row in labels:
        h, w = canvas.shape[:2]
        cx, cy, bw, bh = row[1], row[2], row[3], row[4]
        # account for letterbox padding
        x1 = int((cx - bw / 2) * w) + W
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w) + W
        y2 = int((cy + bh / 2) * h)
        if y1 < 0: y1 = 0
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return canvas


# ---- find good and failure frames ----
good_frames = []   # both A and D detect correctly with high conf
fail_frames = []   # D detects but A misses, or both miss (weak motion)
all_frames = list(range(len(ds)))
random.seed(42)

for idx in random.sample(all_frames, min(200, len(all_frames))):
    img, img2, labels, p, p2, sh, sh2 = ds[idx]
    dets_A, _ = detect(model_A, img, img2, q_mode=None, conf=0.25)
    dets_D, _ = detect(model_D, img, img2, q_mode='neutral', conf=0.25)
    nl = len(labels)
    has_det = len(dets_D) > 0
    if has_det and len(dets_A) > 0 and dets_D[0][4] > 0.8:
        good_frames.append((idx, dets_A, dets_D, labels))
    elif has_det and len(dets_A) == 0 and dets_D[0][4] > 0.5:
        fail_frames.append(('D_only', idx, dets_A, dets_D, labels))
    elif not has_det and nl > 0:
        # weak motion frame (mask very dark)
        mask_mean = img2[:3].float().mean().item()
        if mask_mean < 0.01:
            fail_frames.append(('both_miss', idx, dets_A, dets_D, labels))

print(f'good: {len(good_frames)}, fail: {len(fail_frames)}')

# ============ FIGURE 2: Detection Results + Failure Cases ============
# 4 rows x 4 cols: [RGB_A | RGB_D | mask | label]
n_rows_fig2 = 4  # 2 success + 2 failure
fig2_rows = []
# pick 2 success (different videos)
used_vids = set()
for item in good_frames:
    idx, dA, dD, lb = item
    vid = Path(ds.im_files[idx]).stem.rsplit('_', 1)[0]
    if vid not in used_vids and len(fig2_rows) < 2:
        fig2_rows.append(('success', idx, dA, dD, lb))
        used_vids.add(vid)
# pick 2 failure
for item in fail_frames[:2]:
    ftype, idx, dA, dD, lb = item
    if len(fig2_rows) < 4:
        fig2_rows.append((ftype, idx, dA, dD, lb))

fig2 = None
for row_i, (rtype, idx, dets_A, dets_D, labels) in enumerate(fig2_rows):
    img, img2, lb, p, p2, sh, sh2 = ds[idx]
    H, W = img.shape[1], img.shape[2]
    # 4 panels: A result | D result | motion diff (jet) | GT only
    row_canvas = np.zeros((H, W * 4, 3), dtype=np.uint8)

    # panel 1: A detection
    row_canvas[:, :W] = img.permute(1, 2, 0).numpy()[..., ::-1]
    draw_det(row_canvas, dets_A, (0, 0, 255), W=0, thick=1, fsize=0.5)

    # panel 2: D detection
    row_canvas[:, W:2*W] = img.permute(1, 2, 0).numpy()[..., ::-1]
    draw_det(row_canvas, dets_D, (0, 0, 255), W=W, thick=1, fsize=0.5)

    # panel 3: motion diff (jet colormap)
    mask_vis = img2[:3].float().mean(0).numpy()
    mask_enh = np.clip(mask_vis * 8, 0, 255).astype(np.uint8)
    mask_color = cv2.applyColorMap(mask_enh, cv2.COLORMAP_JET)
    mask_color[mask_enh == 0] = [20, 20, 20]
    row_canvas[:, 2*W:3*W] = mask_color

    # panel 4: GT boxes on RGB
    row_canvas[:, 3*W:4*W] = img.permute(1, 2, 0).numpy()[..., ::-1]
    draw_gt(row_canvas, labels, W=3*W)

    # row label
    label_text = {'success': 'Success', 'D_only': 'D detects, A misses',
                  'both_miss': 'Both miss (weak motion)'}.get(rtype, rtype)
    cv2.putText(row_canvas, label_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if fig2 is None:
        fig2 = row_canvas
    else:
        fig2 = np.vstack([fig2, row_canvas])

# add column headers
header = np.zeros((30, fig2.shape[1], 3), dtype=np.uint8)
for col, text in enumerate(['YOLOMG (baseline)', 'QMC-YOLOMG (ours)',
                             'Motion diff', 'Ground truth']):
    x = col * W + W // 2 - len(text) * 5
    cv2.putText(header, text, (max(x, 5), 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
fig2 = np.vstack([header, fig2])
cv2.imwrite(str(OUT / 'fig2_detections.jpg'), fig2)
print(f'saved fig2_detections.jpg ({fig2.shape[1]}x{fig2.shape[0]})')

# ============ FIGURE 3: Motion diff, Q, Gate correspondence ============
# 3 rows x 4 cols: [RGB | motion | Q channels | gate]
fig3_frames = []
for item in good_frames[:3]:
    idx, dA, dD, lb = item
    vid = Path(ds.im_files[idx]).stem.rsplit('_', 1)[0]
    fig3_frames.append(idx)

fig3 = None
for idx in fig3_frames:
    img, img2, labels, p, p2, sh, sh2 = ds[idx]
    H, W = img.shape[1], img.shape[2]
    imgs = img.unsqueeze(0).float().div(255).to(DEV)
    motion6 = img2.unsqueeze(0).float().div(255).to(DEV)
    motion = motion6[:, :3]
    q = compute_quality_evidence(motion, imgs)

    dets, gate = detect(model_D, img, img2, q_mode='neutral')

    row_canvas = np.zeros((H, W * 4, 3), dtype=np.uint8)

    # panel 1: RGB + det
    row_canvas[:, :W] = img.permute(1, 2, 0).numpy()[..., ::-1]
    draw_det(row_canvas, dets, (0, 0, 255), W=0, thick=1, fsize=0.5)

    # panel 2: motion diff (jet, 8x gain on 0-255)
    mask_vis = img2[:3].float().mean(0).numpy()  # raw 0-255
    mask_enh = np.clip(mask_vis * 8, 0, 255).astype(np.uint8)
    row_canvas[:, W:2*W] = cv2.applyColorMap(mask_enh, cv2.COLORMAP_JET)

    # panel 3: Q mean (all 3 channels averaged)
    q_vis = q[0].mean(0).cpu().numpy()
    q_enh = (q_vis * 255).astype(np.uint8)
    row_canvas[:, 2*W:3*W] = cv2.applyColorMap(q_enh, cv2.COLORMAP_VIRIDIS)

    # panel 4: gate
    if gate is not None:
        g = gate[0, 0].cpu().numpy()
        g_resized = cv2.resize(g, (W, H))
        g_uint8 = (g_resized * 255).astype(np.uint8)
        row_canvas[:, 3*W:4*W] = cv2.applyColorMap(g_uint8, cv2.COLORMAP_HOT)
    else:
        row_canvas[:, 3*W:4*W] = 128  # gray if no gate

    if fig3 is None:
        fig3 = row_canvas
    else:
        fig3 = np.vstack([fig3, row_canvas])

header3 = np.zeros((30, fig3.shape[1], 3), dtype=np.uint8)
for col, text in enumerate(['RGB + detection', 'Motion difference',
                             'Q evidence (mean)', 'Gate G']):
    x = col * W + W // 2 - len(text) * 5
    cv2.putText(header3, text, (max(x, 5), 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
fig3 = np.vstack([header3, fig3])
cv2.imwrite(str(OUT / 'fig3_mechanism.jpg'), fig3)
print(f'saved fig3_mechanism.jpg ({fig3.shape[1]}x{fig3.shape[0]})')
print('done')
