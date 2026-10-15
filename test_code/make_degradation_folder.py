# -*- coding: utf-8 -*-
"""Export degradation + motion input panels to a dedicated folder.

Folder: runs/degradation_motion_input/
Contents:
  degradation/  — degraded motion diff images (clean, attenuated x3, shifted x3)
  motion_input/ — motion diff (gray, jet) + optical flow visualization
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'test_code'))

from utils.datasets import LoadImagesAndLabels
from MOD_Functions import motion_compensate

OUT = ROOT / 'runs' / 'degradation_motion_input'
DEG = OUT / 'degradation'
MOT = OUT / 'motion_input'
DEG.mkdir(parents=True, exist_ok=True)
MOT.mkdir(parents=True, exist_ok=True)

# ---- load a good frame from val ----
hyp = yaml.safe_load(open(ROOT / 'data/hyps/hyp.scratch-low.yaml', errors='ignore'))
ds = LoadImagesAndLabels(
    str(ROOT / 'datasets/ARD100_full_clean/val.txt'),
    str(ROOT / 'datasets/ARD100_full_clean/val2.txt'),
    img_size=640, augment=False, hyp=hyp, rect=True)

targets = ['phantom40_1977', 'phantom10_0010', 'phantom09_1003']
for target in targets:
    stem = target.rsplit('_', 1)[0]
    fid = int(target.rsplit('_', 1)[1])
    for i in range(len(ds)):
        if target in ds.im_files[i]:
            idx = i
            break
    else:
        continue

    img_t, img2_6ch, labels, p, p2, sh, sh2 = ds[idx]
    mask_3ch = img2_6ch[:3]
    H, W = img_t.shape[1], img_t.shape[2]
    mask_2d = mask_3ch.float().mean(0).numpy()

    # ============ motion_input folder ============
    # gray diff
    cv2.imwrite(str(MOT / f'{target}_diff_gray.jpg'),
                mask_2d.astype(np.uint8))
    # jet diff (4x gain)
    jet = cv2.applyColorMap(np.clip(mask_2d * 4, 0, 255).astype(np.uint8),
                             cv2.COLORMAP_JET)
    cv2.imwrite(str(MOT / f'{target}_diff_jet.jpg'), jet)
    # hot diff (4x gain)
    hot = cv2.applyColorMap(np.clip(mask_2d * 4, 0, 255).astype(np.uint8),
                             cv2.COLORMAP_HOT)
    cv2.imwrite(str(MOT / f'{target}_diff_hot.jpg'), hot)

    # optical flow (from raw video frames)
    ARD = ROOT.parents[1] / 'data' / 'ARD100'
    cap = cv2.VideoCapture(str(ARD / 'train_videos' / f'{stem}.mp4'))
    raw = {}
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1
        if fi in (fid - 2, fid, fid + 2):
            raw[fi] = frame
            if len(raw) == 3:
                break
    cap.release()

    if len(raw) == 3:
        g1 = cv2.cvtColor(cv2.GaussianBlur(raw[fid - 2], (11, 11), 0), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.GaussianBlur(raw[fid], (11, 11), 0), cv2.COLOR_BGR2GRAY)
        g3 = cv2.cvtColor(cv2.GaussianBlur(raw[fid + 2], (11, 11), 0), cv2.COLOR_BGR2GRAY)

        # dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((g1.shape[0], g1.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2       # hue = direction
        hsv[..., 1] = 255                          # saturation
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # value = magnitude
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        flow_resized = cv2.resize(flow_rgb, (W, H))
        cv2.imwrite(str(MOT / f'{target}_optical_flow.jpg'), flow_resized)

        # flow magnitude heatmap
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mag_color = cv2.applyColorMap(cv2.resize(mag_norm, (W, H)), cv2.COLORMAP_JET)
        cv2.imwrite(str(MOT / f'{target}_flow_magnitude.jpg'), mag_color)

        # compensated frames
        comp1, _, _, _, _, _ = motion_compensate(g1, g2)
        comp2, _, _, _, _, _ = motion_compensate(g3, g2)
        diff1 = cv2.absdiff(g2, comp1).astype(np.float32)
        diff2 = cv2.absdiff(g2, comp2).astype(np.float32)
        fused = np.clip((diff1 + diff2) / 2 * 3, 0, 255).astype(np.uint8)
        fused_resized = cv2.resize(fused, (W, H))
        cv2.imwrite(str(MOT / f'{target}_comp_diff_gray.jpg'), fused_resized)
        fused_jet = cv2.applyColorMap(fused_resized, cv2.COLORMAP_JET)
        cv2.imwrite(str(MOT / f'{target}_comp_diff_jet.jpg'), fused_jet)

    # ============ degradation folder ============
    motion_t = mask_3ch.unsqueeze(0).float().div(255)

    def jet_save(m2d, name, gain=4):
        m = np.clip(m2d * gain, 0, 255).astype(np.uint8)
        cv2.imwrite(str(DEG / f'{target}_{name}.jpg'),
                    cv2.applyColorMap(m, cv2.COLORMAP_JET))

    # clean
    jet_save(mask_2d, 'clean')

    # local attenuation at 3 strengths
    for factor, fname in [(0.3, 'atten_03'), (0.5, 'atten_05'), (0.7, 'atten_07')]:
        m_atten = motion_t.clone()
        for row in labels.numpy():
            cx, cy, bw, bh = row[1], row[2], row[3], row[4]
            ew, eh = bw * 1.3, bh * 1.3
            x1, x2 = int((cx - ew / 2) * W), int((cx + ew / 2) * W) + 1
            y1, y2 = int((cy - eh / 2) * H), int((cy + eh / 2) * H) + 1
            m_atten[0, :, max(y1, 0):y2, max(x1, 0):x2] *= factor
        jet_save(m_atten[0].mean(0).cpu().numpy(), fname)
        # with region outline
        m2d = m_atten[0].mean(0).cpu().numpy()
        m_jet = cv2.applyColorMap(np.clip(m2d * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        for row in labels.numpy():
            cx, cy, bw, bh = row[1], row[2], row[3], row[4]
            ew, eh = bw * 1.3, bh * 1.3
            x1, y1 = int((cx - ew / 2) * W), int((cy - eh / 2) * H)
            x2, y2 = int((cx + ew / 2) * W), int((cy + eh / 2) * H)
            cv2.rectangle(m_jet, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.imwrite(str(DEG / f'{target}_{fname}_marked.jpg'), m_jet)

    # spatial shift at 3 strengths
    for dx, dy, fname in [(2, 1, 'shift_1px'), (4, 2, 'shift_2px'), (8, 4, 'shift_4px')]:
        shifted = torch.zeros_like(motion_t)
        shifted[0, :, max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
            motion_t[0, :, max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
        jet_save(shifted[0].mean(0).cpu().numpy(), fname)
        # with arrow
        m2d = shifted[0].mean(0).cpu().numpy()
        m_jet = cv2.applyColorMap(np.clip(m2d * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.arrowedLine(m_jet, (W // 3, H // 3), (W // 3 + dx * 8, H // 3 + dy * 8),
                        (255, 255, 255), 2, tipLength=0.3)
        cv2.putText(m_jet, f'shift ({dx},{dy})px', (W // 3 + dx * 8 + 10, H // 3 + dy * 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(DEG / f'{target}_{fname}_arrow.jpg'), m_jet)

    print(f'{target}: done')

print(f'\nall in {OUT}')
