# -*- coding: utf-8 -*-
"""Export detection results + degradation visualization panels.

Panels:
  10: clean motion diff
  11: local attenuation degraded diff (with target region highlighted)
  12a: registration perturbation (shift 1px)
  12b: registration perturbation (shift 4px)
  13: detection on clean input
  14: detection on locally attenuated input
  15: detection on perturbed input
"""
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
OUT = ROOT / 'runs' / 'paper_single_panels'

hyp = yaml.safe_load(open(ROOT / 'data/hyps/hyp.scratch-low.yaml', errors='ignore'))
ds = LoadImagesAndLabels(
    str(ROOT / 'datasets/ARD100_full_clean/val.txt'),
    str(ROOT / 'datasets/ARD100_full_clean/val2.txt'),
    img_size=640, augment=False, hyp=hyp, rect=True)

idx = next(i for i in range(len(ds)) if 'phantom40_1977' in ds.im_files[i])
img_t, img2_6ch, labels, p, p2, sh, sh2 = ds[idx]
mask_3ch = img2_6ch[:3]  # (3,H,W) uint8

H, W = img_t.shape[1], img_t.shape[2]
model = attempt_load(str(ROOT / 'runs/train/full3_D_s1/weights/best.pt'),
                     map_location=DEV).float().eval()

imgs = img_t.unsqueeze(0).float().div(255).to(DEV)
motion_clean = mask_3ch.unsqueeze(0).float().div(255).to(DEV)


def detect_and_draw(rgb_tensor, motion_tensor, q_mode='neutral', label=''):
    q = neutral_quality(motion_tensor) if q_mode == 'neutral' else compute_quality_evidence(motion_tensor, rgb_tensor)
    with torch.no_grad():
        out = model(rgb_tensor, motion_tensor, q)
        inf = out[0] if isinstance(out, tuple) else out
        dets = non_max_suppression(inf, 0.15, 0.45)[0].cpu().numpy()
    rgb_np = rgb_tensor[0].permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255
    canvas = rgb_np.astype(np.uint8).copy()
    # GT green
    for row in labels.numpy():
        cx, cy, bw, bh = row[1], row[2], row[3], row[4]
        x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
        x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
    # det red
    for d in dets:
        x1, y1, x2, y2 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(canvas, f'{d[4]:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    conf = f'{dets[0][4]:.2f}' if len(dets) else 'miss'
    print(f'  {label}: {len(dets)} det, conf={conf}')
    return canvas, dets


def jet(m2d, gain=8):
    m = np.clip(m2d * gain, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_JET)


def shift_tensor(t, dx, dy):
    """Shift a (B,C,H,W) tensor by (dx,dy) with zero padding."""
    out = torch.zeros_like(t)
    B, C, H_, W_ = t.shape
    out[:, :, max(dy, 0):H_ + min(dy, 0), max(dx, 0):W_ + min(dx, 0)] = \
        t[:, :, max(-dy, 0):H_ + min(-dy, 0), max(-dx, 0):W_ + min(-dx, 0)]
    return out


# ---- 1) local attenuation ----
motion_atten = motion_clean.clone()
for row in labels.numpy():
    cx, cy, bw, bh = row[1], row[2], row[3], row[4]
    ew, eh = bw * 1.3, bh * 1.3
    x1, x2 = int((cx - ew / 2) * W), int((cx + ew / 2) * W) + 1
    y1, y2 = int((cy - eh / 2) * H), int((cy + eh / 2) * H) + 1
    motion_atten[0, :, max(y1, 0):y2, max(x1, 0):x2] *= 0.35

# ---- 2) registration perturbation (spatial shift of diff map) ----
motion_shift1 = shift_tensor(motion_clean, 2, 1)   # ~1px at feature scale
motion_shift4 = shift_tensor(motion_clean, 8, 4)   # ~4px

# ---- generate panels ----
print('running detections...')

# clean baseline
det_clean, dets_clean = detect_and_draw(imgs, motion_clean, label='clean')
det_atten, dets_atten = detect_and_draw(imgs, motion_atten, label='attenuated')
det_shift1, _ = detect_and_draw(imgs, motion_shift1, label='shift 1px')
det_shift4, _ = detect_and_draw(imgs, motion_shift4, label='shift 4px')

# motion diff panels
mask_2d = mask_3ch.float().mean(0).numpy()
diff_clean_jet = jet(mask_2d)

# attenuated diff with target region highlighted
diff_atten_2d = motion_atten[0].mean(0).cpu().numpy()
diff_atten_jet = jet(diff_atten_2d)
# draw target region outline (yellow dashed)
for row in labels.numpy():
    cx, cy, bw, bh = row[1], row[2], row[3], row[4]
    ew, eh = bw * 1.3, bh * 1.3
    x1, y1 = int((cx - ew / 2) * W), int((cy - eh / 2) * H)
    x2, y2 = int((cx + ew / 2) * W), int((cy + eh / 2) * H)
    cv2.rectangle(diff_atten_jet, (x1, y1), (x2, y2), (0, 255, 255), 1)
    cv2.putText(diff_atten_jet, 'attenuated', (x1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# perturbed diffs
diff_s1_2d = motion_shift1[0].mean(0).cpu().numpy()
diff_s1_jet = jet(diff_s1_2d)
# draw shift arrow
cv2.arrowedLine(diff_s1_jet, (W // 2, H // 2), (W // 2 + 15, H // 2 + 8),
                (255, 255, 255), 2, tipLength=0.3)
cv2.putText(diff_s1_jet, 'shift 2px', (W // 2 + 20, H // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

diff_s4_2d = motion_shift4[0].mean(0).cpu().numpy()
diff_s4_jet = jet(diff_s4_2d)
cv2.arrowedLine(diff_s4_jet, (W // 2, H // 2), (W // 2 + 60, H // 2 + 30),
                (255, 255, 255), 2, tipLength=0.3)
cv2.putText(diff_s4_jet, 'shift 8px', (W // 2 + 65, H // 2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# save all
panels = {
    '10_diff_clean':          diff_clean_jet,
    '11_diff_attenuated':     diff_atten_jet,
    '12a_diff_shift_small':   diff_s1_jet,
    '12b_diff_shift_large':   diff_s4_jet,
    '13_det_clean':           det_clean,
    '14_det_attenuated':      det_atten,
    '15_det_shift_small':     det_shift1,
    '15_det_shift_large':     det_shift4,
}

for name, img in panels.items():
    path = OUT / f'{name}.jpg'
    cv2.imwrite(str(path), img)
    print(f'saved {name}.jpg')

print(f'\nall in {OUT}')
