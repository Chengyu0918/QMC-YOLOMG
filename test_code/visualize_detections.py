# -*- coding: utf-8 -*-
"""Generate detection visualization images for the paper.

Picks frames where the model detects well (high confidence + correct),
draws RGB + mask + detection boxes side by side.
Outputs to runs/paper_visualizations/
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
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.quality import compute_quality_evidence, neutral_quality

DEV = select_device('0')
OUT = ROOT / 'runs' / 'paper_visualizations'
OUT.mkdir(parents=True, exist_ok=True)

# models to visualize
MODELS = [
    ('A_original', 'fullall_base', None),
    ('D_constQ_U', 'full3_D_s1', 'neutral'),
]
N_SHOW = 6  # frames per model
CONF_THRES = 0.3  # show only confident detections

hyp = yaml.safe_load(open(ROOT / 'data/hyps/hyp.scratch-low.yaml', errors='ignore'))
ds = LoadImagesAndLabels(
    str(ROOT / 'datasets/ARD100_full_clean/val.txt'),
    str(ROOT / 'datasets/ARD100_full_clean/val2.txt'),
    img_size=640, augment=False, hyp=hyp, rect=True)

# pick diverse frames from different videos
video_map = {}
for i in range(len(ds)):
    p = ds.im_files[i]
    vid = Path(p).stem.rsplit('_', 1)[0]
    video_map.setdefault(vid, []).append(i)

videos = sorted(video_map.keys())
print(f'{len(videos)} val videos: {videos}')
selected = []
for v in videos:
    selected.extend(video_map[v][:2])  # up to 2 frames per video
random.seed(42)
random.shuffle(selected)
selected = selected[:N_SHOW * 3]  # extra candidates
print(f'selected {len(selected)} candidate frames')

for tag, run_name, q_mode in MODELS:
    model = attempt_load(str(ROOT / f'runs/train/{run_name}/weights/best.pt'),
                         map_location=DEV).float().eval()
    shown = 0
    for idx in selected:
        if shown >= N_SHOW:
            break
        img, img2, labels, path, path2, sh, sh2 = ds[idx]
        imgs = img.unsqueeze(0).float().div(255).to(DEV)
        motion6 = img2.unsqueeze(0).float().div(255).to(DEV)
        motion = motion6[:, :3]
        q = (neutral_quality(motion) if q_mode == 'neutral'
             else compute_quality_evidence(motion, imgs) if q_mode else None)

        with torch.no_grad():
            out = model(imgs, motion, q)
            inf = out[0] if isinstance(out, tuple) else out
            dets = non_max_suppression(inf, CONF_THRES, 0.45)[0]

        if len(dets) == 0:
            continue

        H, W = img.shape[1], img.shape[2]
        mask_vis = motion[0].mean(0).cpu().numpy()

        # compose: RGB+dets | mask (enhanced) | zoom inset
        canvas = np.zeros((H, W * 2, 3), dtype=np.uint8)
        canvas[:, :W] = img.permute(1, 2, 0).numpy()[..., ::-1]  # BGR

        # enhance mask: adaptive gain + colormap for visibility
        mask_enhanced = np.clip(mask_vis * 8, 0, 255).astype(np.uint8)  # 8x gain
        mask_color = cv2.applyColorMap(mask_enhanced, cv2.COLORMAP_JET)
        # make zero-response areas dark blue (not black)
        mask_color[mask_enhanced == 0] = [20, 20, 20]
        canvas[:, W:] = mask_color

        # draw GT (green) on both
        for row in labels.numpy():
            cx, cy, bw, bh = row[1], row[2], row[3], row[4]
            x1, y1 = int((cx - bw/2) * W), int((cy - bh/2) * H)
            x2, y2 = int((cx + bw/2) * W), int((cy + bh/2) * H)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(canvas, (x1 + W, y1), (x2 + W, y2), (0, 255, 0), 1)

        # draw detections (red, thick) on RGB
        for det in dets:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # draw confidence with larger font + background box for readability
            label = f'{conf:.2f}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ty = y1 - 6 if y1 - th - 6 > 0 else y2 + th + 6
            cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), (0, 0, 255), -1)
            cv2.putText(canvas, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.rectangle(canvas, (x1 + W, y1), (x2 + W, y2), (0, 0, 255), 2)
            # zoom inset: RGB crop | mask crop, stacked vertically
            pad = max((x2 - x1), (y2 - y1)) * 2 + 30
            zx1, zy1 = max(x1 - pad, 0), max(y1 - pad, 0)
            zx2, zy2 = min(x2 + pad, W), min(y2 + pad, H)
            if zx2 > zx1 and zy2 > zy1:
                zoom_rgb = canvas[zy1:zy2, zx1:zx2].copy()
                zoom_mask = canvas[zy1:zy2, W + zx1:W + zx2].copy()
                zoom_h = H // 4
                zoom_w = int(zoom_rgb.shape[1] * zoom_h / zoom_rgb.shape[0])
                zoom_rgb = cv2.resize(zoom_rgb, (zoom_w, zoom_h))
                zoom_mask = cv2.resize(zoom_mask, (zoom_w, zoom_h))
                zoom = np.hstack([zoom_rgb, zoom_mask])
                # place in top-right corner
                zh, zw = zoom.shape[:2]
                canvas[10:10+zh, W*2-zw-10:W*2-10] = zoom
                cv2.rectangle(canvas, (W*2-zw-12, 8), (W*2-8, 12+zh), (255,255,255), 1)

        name = f'{tag}_{Path(path).stem}.jpg'
        cv2.imwrite(str(OUT / name), canvas)
        shown += 1
        print(f'  {tag}: saved {name} ({len(dets)} det, conf {dets[:, 4].max():.2f})')

    del model
    torch.cuda.empty_cache()

print(f'\ndone: {shown * len(MODELS)} images in {OUT}')
