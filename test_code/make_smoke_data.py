"""Generate a tiny synthetic dual-modal dataset for QMC-YOLOMG smoke tests.

Each clip simulates an airborne camera with ego-motion over a textured
background while a small bright target (drone) moves independently:

    RGB frame      textured background shifted by the camera motion + target
    motion mask    |frame_t - warp(frame_{t-1}, true ego shift)|, i.e. the
                   compensated frame difference the ARD100-style mask
                   pipelines produce; a small deliberate registration error is
                   injected into a fraction of frames so the registration-
                   quality evidence has real structure to work with
    labels         YOLO format (class cx cy w h) of the target box

Usage:
    python test_code/make_smoke_data.py --out datasets/smoke --clips 6 --frames 10
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np


def make_background(rng, size=640):
    """Structured, textured background (smooth noise + edges + blobs)."""
    h = w = size
    small = rng.integers(0, 255, (h // 16, w // 16), dtype=np.uint8)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    # A few strong edges / structures so misregistration leaves visible residual.
    canvas = np.zeros((h, w), np.float32)
    for _ in range(24):
        x1, y1 = rng.integers(0, w, 2)
        x2, y2 = rng.integers(0, w, 2)
        cv2.line(canvas, (x1, y1), (x2, y2), float(rng.integers(40, 200)), rng.integers(1, 3))
    for _ in range(10):
        x, y = rng.integers(0, w, 2)
        r = int(rng.integers(4, 24))
        cv2.circle(canvas, (int(x), int(y)), r, float(rng.integers(30, 160)), -1)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.5)
    img = 0.55 * bg + 0.45 * canvas
    return np.clip(img, 0, 255).astype(np.uint8)


def shift_img(img, dx, dy):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def draw_target(img, x, y, r, rng):
    out = img.copy()
    cv2.circle(out, (int(x), int(y)), int(r), 255, -1)
    cv2.circle(out, (int(x), int(y)), int(r + 1), int(rng.integers(60, 140)), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='datasets/smoke')
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--clips', type=int, default=6)
    ap.add_argument('--frames', type=int, default=10)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--reg-error-prob', type=float, default=0.2,
                    help='fraction of frames whose compensation uses a wrong shift')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    out = Path(args.out).resolve()
    size = args.size

    entries = {'train': [], 'val': []}
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'mask' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    n_train_clips = max(args.clips - 2, 1)
    for clip in range(args.clips):
        split = 'train' if clip < n_train_clips else 'val'
        name = f'clip{clip:02d}'
        bg = make_background(rng, size)
        # Camera ego-motion random walk and target trajectory.
        camx, camy = float(rng.integers(-3, 4)), float(rng.integers(-3, 4))
        tx, ty = [float(v) for v in rng.integers(80, size - 80, 2)]
        tvx, tvy = float(rng.uniform(-4, 4)), float(rng.uniform(-4, 4))
        radius = float(rng.integers(4, 8))

        prev_frame = None
        prev_pose = (camx, camy)
        for f in range(args.frames):
            camx += float(rng.integers(-3, 4))
            camy += float(rng.integers(-3, 4))
            tx = float(np.clip(tx + tvx + rng.uniform(-1, 1), 40, size - 40))
            ty = float(np.clip(ty + tvy + rng.uniform(-1, 1), 40, size - 40))

            world = shift_img(bg, int(round(camx)), int(round(camy)))
            frame = draw_target(world, tx, ty, radius, rng)
            frame = np.clip(frame.astype(np.float32) + rng.normal(0, 2.0, frame.shape), 0, 255).astype(np.uint8)

            if prev_frame is None:
                mask = np.zeros_like(frame)
            else:
                # Compensate previous frame onto current view with the TRUE
                # ego shift (occasionally perturbed -> registration residual).
                ex, ey = 0, 0
                if rng.random() < args.reg_error_prob:
                    ex, ey = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
                dx = (camx - prev_pose[0]) + ex
                dy = (camy - prev_pose[1]) + ey
                compensated = shift_img(prev_frame, int(round(dx)), int(round(dy)))
                # Match the FD5 pipeline: blur both views before differencing
                # so texture noise does not flood the difference map.
                a = cv2.GaussianBlur(frame, (11, 11), 0)
                b = cv2.GaussianBlur(compensated, (11, 11), 0)
                diff = cv2.absdiff(a, b).astype(np.float32)
                diff[diff < 12] = 0  # residual floor: hides interpolation noise
                diff = cv2.GaussianBlur(diff, (0, 0), 1.0)
                mask = np.clip(diff * 2.2, 0, 255).astype(np.uint8)

            stem = f'{name}_{f:03d}'
            cv2.imwrite(str(out / 'images' / split / f'{stem}.jpg'),
                        cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            cv2.imwrite(str(out / 'mask' / split / f'{stem}.jpg'),
                        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            # YOLO label of the target box in frame coordinates.
            x1, y1 = (tx - radius - 2) / size, (ty - radius - 2) / size
            x2, y2 = (tx + radius + 2) / size, (ty + radius + 2) / size
            with open(out / 'labels' / split / f'{stem}.txt', 'w') as fh:
                fh.write(f'0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} '
                         f'{x2 - x1:.6f} {y2 - y1:.6f}\n')
            entries[split].append(stem)

            prev_frame = frame
            prev_pose = (camx, camy)

    # Path list files (absolute, matching the repo's txt conventions).
    for split in ('train', 'val'):
        with open(out / f'{split}.txt', 'w') as fh:
            fh.writelines(str((out / 'images' / split / f'{s}.jpg').resolve()).replace(os.sep, '/') + '\n'
                          for s in entries[split])
        with open(out / f'{split}2.txt', 'w') as fh:
            fh.writelines(str((out / 'mask' / split / f'{s}.jpg').resolve()).replace(os.sep, '/') + '\n'
                          for s in entries[split])

    print(f'smoke dataset ready at {out}: '
          f'{len(entries["train"])} train / {len(entries["val"])} val samples')


if __name__ == '__main__':
    main()
