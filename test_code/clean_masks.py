"""Clean motion masks to match the author's published mask32 statistics.

The author's real mask32 sample (data/Test_images/mask/phantom05_0606.jpg)
has background mean 0.44 and global max 71, with clean borders.  Our FD5
reproduction keeps raw compensated differences, which leaves (a) warp-border
artifacts at image edges (up to 255) and (b) low-level residual noise.

This script applies, IN PLACE by default is avoided (writes to <root>_clean):
  1. residual floor:  diff < floor  -> 0        (author's T = 4 + mean idea)
  2. border zeroing:  <border> px bands -> 0    (warp exposure region)
  3. optional gain:   diff * gain, clip 255     (strengthen weak true motion)

Usage:
  python test_code/clean_masks.py --root datasets/ARD100_local --floor 12 --border 12 --gain 3
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def clean_one(job):
    src, dst, floor, border, gain = job
    m = cv2.imread(str(src), 0)
    if m is None:
        return 0
    m = m.astype(np.float32)
    if floor > 0:
        m[m < floor] = 0
    if border > 0:
        m[:border, :] = 0
        m[-border:, :] = 0
        m[:, :border] = 0
        m[:, -border:] = 0
    if gain != 1:
        m = np.clip(m * gain, 0, 255)
    cv2.imwrite(str(dst), m.astype(np.uint8))
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='e.g. datasets/ARD100_local')
    ap.add_argument('--floor', type=int, default=12)
    ap.add_argument('--border', type=int, default=12)
    ap.add_argument('--gain', type=float, default=3.0)
    ap.add_argument('--workers', type=int, default=11)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    dst_root = Path(str(root) + '_clean')
    jobs = []
    for split in ('train', 'val', 'test'):
        files = sorted((root / 'mask' / split).glob('*.jpg'))
        out_dir = dst_root / 'mask' / split
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs += [(f, out_dir / f.name, args.floor, args.border, args.gain) for f in files]

    n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(clean_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs)):
            n += fut.result()
            if (i + 1) % 2000 == 0:
                print(f'{i+1}/{len(jobs)}', flush=True)
    print(f'cleaned {n} masks -> {dst_root}', flush=True)

    # mirror images + labels + txt lists into the cleaned dataset root
    import shutil
    for split in ('train', 'val', 'test'):
        for sub in ('images', 'labels'):
            src_d = root / sub / split
            dst_d = dst_root / sub / split
            dst_d.mkdir(parents=True, exist_ok=True)
            for f in src_d.iterdir():
                if f.is_file() and not (dst_d / f.name).exists():
                    shutil.copy2(f, dst_d / f.name)
    for txt in root.glob('*.txt'):
        if txt.name.endswith('_all.txt'):
            continue
        (dst_root / txt.name).write_text(
            txt.read_text().replace(str(root).replace('\\', '/'),
                                    str(dst_root).replace('\\', '/')))
    print(f'dataset root ready: {dst_root} (same txt names, mask cleaned)', flush=True)


if __name__ == '__main__':
    main()
