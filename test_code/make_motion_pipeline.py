# -*- coding: utf-8 -*-
"""Motion pipeline figure using the dataset's own preprocessing."""
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
OUT = ROOT / 'runs' / 'paper_figures'
OUT.mkdir(parents=True, exist_ok=True)

hyp = yaml.safe_load(open(ROOT / 'data/hyps/hyp.scratch-low.yaml', errors='ignore'))
ds = LoadImagesAndLabels(
    str(ROOT / 'datasets/ARD100_full_clean/val.txt'),
    str(ROOT / 'datasets/ARD100_full_clean/val2.txt'),
    img_size=640, augment=False, hyp=hyp, rect=True)

# find phantom40_1977 in val set
target_stem = 'phantom40_1977'
for i in range(len(ds)):
    if target_stem in ds.im_files[i]:
        idx = i
        break
else:
    print('frame not found in val set')
    sys.exit(1)

print(f'found {target_stem} at index {idx}')

# get dataset-processed inputs (guaranteed correct preprocessing)
img_t, img2_6ch, labels, p, p2, sh, sh2 = ds[idx]
# img_t: (3,H,W) uint8 RGB
# img2_6ch: (6,H,W) uint8 [mask(3ch), Q(3ch)]
mask_3ch = img2_6ch[:3]  # (3,H,W) uint8, the motion diff

# ---- get raw video frames for display ----
ARD = ROOT.parents[1] / 'data' / 'ARD100'
video = 'phantom40'
frame_id = 1977
cap = cv2.VideoCapture(str(ARD / 'train_videos' / f'{video}.mp4'))
raw_frames = {}
fi = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    fi += 1
    if fi in (frame_id - 2, frame_id, frame_id + 2):
        raw_frames[fi] = frame
        if len(raw_frames) == 3:
            break
cap.release()
print(f'read {len(raw_frames)} raw video frames')

# ---- compensation on raw frames ----
def prep(frame):
    return cv2.cvtColor(cv2.GaussianBlur(frame, (11, 11), 0), cv2.COLOR_BGR2GRAY)

g_prev = prep(raw_frames[frame_id - 2])
g_curr = prep(raw_frames[frame_id])
g_next = prep(raw_frames[frame_id + 2])

comp_prev, _, _, _, _, _ = motion_compensate(g_prev, g_curr)
comp_next, _, _, _, _, _ = motion_compensate(g_next, g_curr)
diff_prev = cv2.absdiff(g_curr, comp_prev).astype(np.float32)
diff_next = cv2.absdiff(g_curr, comp_next).astype(np.float32)
diff_fused = np.clip((diff_prev + diff_next) / 2 * 3, 0, 255).astype(np.uint8)

# ---- detection ----
model = attempt_load(str(ROOT / 'runs/train/full3_D_s1/weights/best.pt'),
                     map_location=DEV).float().eval()

imgs = img_t.unsqueeze(0).float().div(255).to(DEV)
motion = mask_3ch.unsqueeze(0).float().div(255).to(DEV)

# WITH motion
with torch.no_grad():
    q = neutral_quality(motion)
    out = model(imgs, motion, q)
    inf = out[0] if isinstance(out, tuple) else out
    dets_with = non_max_suppression(inf, 0.15, 0.45)[0].cpu().numpy()

# WITHOUT motion
zero = torch.zeros_like(motion)
with torch.no_grad():
    out0 = model(imgs, zero, neutral_quality(zero))
    inf0 = out0[0] if isinstance(out0, tuple) else out0
    dets_without = non_max_suppression(inf0, 0.15, 0.45)[0].cpu().numpy()

n_w = len(dets_with)
n_wo = len(dets_without)
print(f'with motion: {n_w} dets' + (f' conf={dets_with[0][4]:.2f}' if n_w else ''))
print(f'without motion: {n_wo} dets' + (f' conf={dets_without[0][4]:.2f}' if n_wo else ''))

# ---- compose figure (3 rows x 3 cols) ----
H, W = img_t.shape[1], img_t.shape[2]  # letterboxed size
GAP = 4

def to_bgr_display(t):
    """torch (3,H,W) uint8 RGB -> numpy (H,W,3) BGR for cv2."""
    return t.permute(1, 2, 0).numpy()[..., ::-1]

def draw_dets_np(img, dets, color=(0, 0, 255), thick=2, scale=1.0):
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2, cf, cls = d
        x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        cv2.putText(out, f'{cf:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
    return out

def jet(mask_2d, gain=8):
    m = np.clip(mask_2d.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_JET)

# row 1: raw frames (resized to match W×H)
r1c1 = cv2.resize(raw_frames[frame_id - 2], (W, H))
r1c2 = cv2.resize(raw_frames[frame_id], (W, H))
r1c3 = cv2.resize(raw_frames[frame_id + 2], (W, H))

# row 2: compensated + diff
r2c1 = cv2.resize(cv2.cvtColor(comp_prev, cv2.COLOR_GRAY2BGR), (W, H))
r2c2 = cv2.resize(cv2.cvtColor(comp_next, cv2.COLOR_GRAY2BGR), (W, H))
mask_2d = mask_3ch.float().mean(0).numpy()  # (H,W) 0-255
r2c3 = jet(mask_2d)

# row 3: detections
rgb_bgr = to_bgr_display(img_t)
r3c1 = draw_dets_np(rgb_bgr, dets_with, (0, 0, 255), thick=2)
r3c2 = draw_dets_np(rgb_bgr, dets_without, (0, 0, 255), thick=2)
det_on_diff = r2c3.copy()
r3c3 = draw_dets_np(det_on_diff, dets_with, (255, 255, 255), thick=2)

# assemble
fig = np.full((H * 3 + GAP * 2 + 25, W * 3 + GAP * 2, 3), 255, dtype=np.uint8)
rows_data = [
    ([r1c1, r1c2, r1c3], ['(a) Frame t-2', '(b) Current frame t', '(c) Frame t+2']),
    ([r2c1, r2c2, r2c3], ['(d) Compensated t-2\u2192t', '(e) Compensated t+2\u2192t', '(f) Motion difference']),
    ([r3c1, r3c2, r3c3], ['(g) Detection WITH motion', '(h) Detection WITHOUT motion', '(i) Detection on diff map']),
]
for ri, (panels, rlabels) in enumerate(rows_data):
    y0 = ri * (H + GAP)
    for ci, (panel, rlabel) in enumerate(zip(panels, rlabels)):
        x0 = ci * (W + GAP)
        fig[y0:y0 + H, x0:x0 + W] = panel
        cv2.putText(fig, rlabel, (x0 + 5, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

out_path = OUT / 'fig_motion_pipeline.jpg'
cv2.imwrite(str(out_path), fig)
print(f'saved {out_path} ({fig.shape[1]}x{fig.shape[0]})')
