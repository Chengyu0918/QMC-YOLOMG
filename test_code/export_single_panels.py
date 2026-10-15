# -*- coding: utf-8 -*-
"""Export each panel of the motion pipeline figure as an individual image."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'test_code'))

from models.experimental import attempt_load
from utils.datasets import LoadImagesAndLabels
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from utils.quality import neutral_quality
from MOD_Functions import motion_compensate

DEV = select_device('0')
OUT = ROOT / 'runs' / 'paper_single_panels'
OUT.mkdir(parents=True, exist_ok=True)

hyp = yaml.safe_load(open(ROOT / 'data/hyps/hyp.scratch-low.yaml', errors='ignore'))
ds = LoadImagesAndLabels(
    str(ROOT / 'datasets/ARD100_full_clean/val.txt'),
    str(ROOT / 'datasets/ARD100_full_clean/val2.txt'),
    img_size=640, augment=False, hyp=hyp, rect=True)

idx = next(i for i in range(len(ds)) if 'phantom40_1977' in ds.im_files[i])
img_t, img2_6ch, labels, p, p2, sh, sh2 = ds[idx]
mask_3ch = img2_6ch[:3]

# raw frames
ARD = ROOT.parents[1] / 'data' / 'ARD100'
cap = cv2.VideoCapture(str(ARD / 'train_videos' / 'phantom40.mp4'))
raw = {}
fi = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    fi += 1
    if fi in (1975, 1977, 1979):
        raw[fi] = frame
        if len(raw) == 3:
            break
cap.release()

def prep(f):
    return cv2.cvtColor(cv2.GaussianBlur(f, (11, 11), 0), cv2.COLOR_BGR2GRAY)

g_prev, g_curr, g_next = prep(raw[1975]), prep(raw[1977]), prep(raw[1979])
comp_prev, _, _, _, _, _ = motion_compensate(g_prev, g_curr)
comp_next, _, _, _, _, _ = motion_compensate(g_next, g_curr)
diff_prev = cv2.absdiff(g_curr, comp_prev).astype(np.float32)
diff_next = cv2.absdiff(g_curr, comp_next).astype(np.float32)
diff_fused = np.clip((diff_prev + diff_next) / 2 * 3, 0, 255).astype(np.uint8)

# detection
model = attempt_load(str(ROOT / 'runs/train/full3_D_s1/weights/best.pt'),
                     map_location=DEV).float().eval()
imgs = img_t.unsqueeze(0).float().div(255).to(DEV)
motion = mask_3ch.unsqueeze(0).float().div(255).to(DEV)
with torch.no_grad():
    out = model(imgs, motion, neutral_quality(motion))
    dets_w = non_max_suppression(out[0] if isinstance(out, tuple) else out, 0.15, 0.45)[0].cpu().numpy()
    out0 = model(imgs, torch.zeros_like(motion), neutral_quality(torch.zeros_like(motion)))
    dets_wo = non_max_suppression(out0[0] if isinstance(out0, tuple) else out0, 0.15, 0.45)[0].cpu().numpy()

# letterboxed dims
H, W = img_t.shape[1], img_t.shape[2]

def to_bgr(t):
    return t.permute(1, 2, 0).numpy()[..., ::-1]

def draw(img, dets, color=(0, 0, 255), thick=2):
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        cv2.putText(out, f'{d[4]:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
    return out

def jet(m2d, gain=8):
    m = np.clip(m2d.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_JET)

def gray3(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def rs(img):
    return cv2.resize(img, (W, H))

# draw GT on all detection panels
rgb_bgr = to_bgr(img_t)
def draw_gt(img):
    out = img.copy()
    for row in labels.numpy():
        cx, cy, bw, bh = row[1], row[2], row[3], row[4]
        x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
        x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return out

panels = {
    '01_frame_prev':         rs(raw[1975]),
    '02_frame_curr':         rs(raw[1977]),
    '03_frame_next':         rs(raw[1979]),
    '04_comp_prev':          rs(gray3(comp_prev)),
    '05_comp_next':          rs(gray3(comp_next)),
    '06_motion_diff_jet':    jet(mask_3ch.float().mean(0).numpy()),
    '06_motion_diff_gray':   rs(gray3(diff_fused)),
    '07_det_with_motion':    draw_gt(draw(rgb_bgr, dets_w, (0, 0, 255))),
    '08_det_without_motion': draw_gt(draw(rgb_bgr, dets_wo, (0, 0, 255))),
    '09_det_on_diff':        draw(jet(mask_3ch.float().mean(0).numpy()), dets_w, (255, 255, 255)),
}

for name, img in panels.items():
    path = OUT / f'{name}.jpg'
    cv2.imwrite(str(path), img)
    print(f'saved {name}.jpg ({img.shape[1]}x{img.shape[0]})')

# also save individual panels from other figures
import shutil
# scene comparison panels (4 rows)
scene_src = ROOT / 'runs' / 'paper_figures' / 'fig_scene_comparison.jpg'
if scene_src.exists():
    scene = cv2.imread(str(scene_src))
    h, w = scene.shape[:2]
    # skip header (28px), then 4 rows
    row_h = (h - 28) // 4
    col_w = w // 3
    scene_names = ['complex_bg', 'low_light', 'weak_motion', 'dynamic_bg']
    col_names = ['baseline_A', 'ours_D', 'motion_diff']
    for ri, sname in enumerate(scene_names):
        for ci, cname in enumerate(col_names):
            y1 = 28 + ri * row_h
            x1 = ci * col_w
            panel = scene[y1:y1 + row_h, x1:x1 + col_w]
            cv2.imwrite(str(OUT / f'scene_{sname}_{cname}.jpg'), panel)
    print('saved 12 scene panels')

print(f'\nall files in {OUT}')
