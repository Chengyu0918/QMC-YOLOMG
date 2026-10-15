"""Prepare the real ARD100 dual-modal dataset (full-quantity, parallel).

Reproduces the original YOLOMG data pipeline from the raw videos + VOC
annotations (dataset root: <outer-repo>/data/ARD100):

    RGB frame   every annotated frame, extracted from the video
    motion mask compensated 3-frame difference: for annotated frame f, the
                mask is aligned to f and computed as
                  mean(|gray(f) - comp(gray(f-2))|, |gray(f) - comp(gray(f+2))|)
                (FD5 construction, authors' KLT+homography compensation)
    labels      VOC xml -> YOLO txt; NORMALIZED boxes are invariant to the
                isotropic resize below, so no relabeling is needed

Frames whose largest object area < min_area are skipped (author setting 25).
Images/masks are resized with a long edge of --edge (default 640): the
training letterbox at imgsz 640 maps 1920x1080 -> 640x360 + padding, which is
EXACTLY the transform applied here, so the network sees identical pixels
while dataloader JPEG decode becomes ~8x cheaper.  Set --edge 0 to keep the
original resolution.

Splits follow the paper's video-level division; val = 6 train-domain videos
held out for model selection.  Per-video annotated frames are subsampled
with an even stride (--n-per-* ; 0 = use every annotated frame).

Parallelism: --workers N processes videos concurrently (each video = one
task: sequential decode pass spilling needed frames to tmp, then mask+label
generation).  Requires the __main__ guard (Windows spawn).
"""

import argparse
import io
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'test_code'))

from MOD_Functions import motion_compensate  # authors' KLT + homography

ARD = ROOT.parents[1] / 'data' / 'ARD100'

TRAIN_VIDEOS = ['phantom09', 'phantom10', 'phantom14', 'phantom17', 'phantom19',
                'phantom20', 'phantom28', 'phantom29', 'phantom30', 'phantom32',
                'phantom36', 'phantom40', 'phantom42', 'phantom43', 'phantom44',
                'phantom46', 'phantom63', 'phantom65', 'phantom66', 'phantom68',
                'phantom70', 'phantom71', 'phantom74', 'phantom75', 'phantom76',
                'phantom77', 'phantom78', 'phantom80', 'phantom81', 'phantom82',
                'phantom84', 'phantom85', 'phantom86', 'phantom87', 'phantom89',
                'phantom90', 'phantom101', 'phantom103', 'phantom104', 'phantom105',
                'phantom106', 'phantom107', 'phantom108', 'phantom109', 'phantom111',
                'phantom112', 'phantom114', 'phantom115', 'phantom116', 'phantom117',
                'phantom118', 'phantom120', 'phantom132', 'phantom137', 'phantom138',
                'phantom139', 'phantom140', 'phantom142', 'phantom143', 'phantom145',
                'phantom146', 'phantom147', 'phantom148', 'phantom149', 'phantom150']
VAL_VIDEOS = ['phantom09', 'phantom10', 'phantom28', 'phantom36', 'phantom40', 'phantom101']
TRAIN_VIDEOS = [v for v in TRAIN_VIDEOS if v not in VAL_VIDEOS]
TEST_VIDEOS = ['phantom02', 'phantom03', 'phantom04', 'phantom05', 'phantom08',
               'phantom22', 'phantom39', 'phantom41', 'phantom45', 'phantom47',
               'phantom50', 'phantom54', 'phantom55', 'phantom56', 'phantom57',
               'phantom58', 'phantom60', 'phantom61', 'phantom64', 'phantom73',
               'phantom79', 'phantom92', 'phantom93', 'phantom94', 'phantom95',
               'phantom97', 'phantom102', 'phantom110', 'phantom113', 'phantom119',
               'phantom133', 'phantom135', 'phantom136', 'phantom141', 'phantom144']


def anno_frames(video):
    d = ARD / 'annotations' / video
    out = []
    if d.is_dir():
        for p in d.glob(f'{video}_*.xml'):
            try:
                out.append(int(p.stem.split('_')[-1]))
            except ValueError:
                pass
    return sorted(out)


def parse_area_and_boxes(xml_bytes):
    root = ET.parse(io.BytesIO(xml_bytes)).getroot()
    boxes = []
    for obj in root.iter('object'):
        b = obj.find('bndbox')
        if b is None:
            continue
        boxes.append((float(b.find('xmin').text), float(b.find('ymin').text),
                      float(b.find('xmax').text), float(b.find('ymax').text)))
    area = max((xmax - xmin) * (ymax - ymin) for xmin, ymin, xmax, ymax in boxes) if boxes else 0.0
    return area, boxes, root


def fd5_diff(gray_ref, gray_a, gray_b):
    comp_a, _, _, _, _, _ = motion_compensate(gray_a, gray_ref)
    comp_b, _, _, _, _, _ = motion_compensate(gray_b, gray_ref)
    d1 = cv2.absdiff(gray_ref, comp_a).astype(np.float32)
    d2 = cv2.absdiff(gray_ref, comp_b).astype(np.float32)
    return (d1 + d2) / 2


def process_video(job):
    """Extract + mask + label one video. Returns (split, video, written)."""
    split, video, cap, min_area, out_s, edge = job
    vdir = 'train_videos' if video in TRAIN_VIDEOS or video in VAL_VIDEOS else 'test_videos'
    vpath = ARD / vdir / f'{video}.mp4'
    out = Path(out_s)
    if not vpath.exists():
        return split, video, -1
    frames = anno_frames(video)
    if not frames:
        return split, video, -1
    stride = max(len(frames) // cap, 1) if cap > 0 else 1
    picked = frames[::stride] if cap > 0 else frames

    xmls = {}
    for f in picked:
        p = ARD / 'annotations' / video / f'{video}_{f:04d}.xml'
        if p.exists():
            xmls[f] = p.read_bytes()
    need = set()
    for f in xmls:
        need.update((f - 2, f, f + 2))
    need.discard(0)

    tmpdir = out / '_tmp' / f'{split}_{video}'
    tmpdir.mkdir(parents=True, exist_ok=True)

    # PASS 1: decode sequentially, spill needed frames (downscaled if edge>0).
    cap_v = cv2.VideoCapture(str(vpath))
    idx = 0
    while True:
        ok, frame = cap_v.read()
        if not ok:
            break
        idx += 1
        if idx in need:
            if edge > 0:
                h, w = frame.shape[:2]
                s = edge / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(tmpdir / f'{idx:06d}.jpg'), frame)
    cap_v.release()

    # PASS 2: mask + label for each annotated frame.
    written = 0
    for f, xml in sorted(xmls.items()):
        p_rgb, p_ref, p_nxt = (tmpdir / f'{f + d:06d}.jpg' for d in (0, -2, 2))
        if not (p_rgb.exists() and p_ref.exists() and p_nxt.exists()):
            continue  # window truncated at video boundary
        rgb = cv2.imread(str(p_rgb))
        g_ref = cv2.imread(str(p_ref))
        g_next = cv2.imread(str(p_nxt))
        area, boxes, root = parse_area_and_boxes(xml)
        if not boxes or area < min_area:
            continue
        try:
            gray_ref = cv2.cvtColor(cv2.GaussianBlur(g_ref, (11, 11), 0), cv2.COLOR_BGR2GRAY)
            gray_cur = cv2.cvtColor(cv2.GaussianBlur(rgb, (11, 11), 0), cv2.COLOR_BGR2GRAY)
            gray_nxt = cv2.cvtColor(cv2.GaussianBlur(g_next, (11, 11), 0), cv2.COLOR_BGR2GRAY)
            # KLT on the downscaled view: correspondingly smaller search win,
            # matching the scale the network will actually see.
            diff = fd5_diff(gray_cur, gray_ref, gray_nxt)
        except Exception as e:
            print(f'mask failed {video}_{f}: {e}', flush=True)
            continue
        stem = f'{video}_{f:04d}'
        cv2.imwrite(str(out / 'images' / split / f'{stem}.jpg'), rgb)
        cv2.imwrite(str(out / 'mask' / split / f'{stem}.jpg'),
                    np.clip(diff, 0, 255).astype(np.uint8))
        h = float(root.find('size').find('height').text)
        w = float(root.find('size').find('width').text)
        with open(out / 'labels' / split / f'{stem}.txt', 'w') as fh:
            for xmin, ymin, xmax, ymax in boxes:
                if (xmax - xmin) * (ymax - ymin) < min_area:
                    continue
                cx, cy = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
                bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
                fh.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
        written += 1
    for p in tmpdir.glob('*.jpg'):
        p.unlink()
    tmpdir.rmdir()
    return split, video, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'datasets' / 'ARD100_full'))
    ap.add_argument('--n-per-train', type=int, default=0, help='0 = every annotated frame')
    ap.add_argument('--n-per-val', type=int, default=0)
    ap.add_argument('--n-per-test', type=int, default=0)
    ap.add_argument('--min-area', type=float, default=25.0)
    ap.add_argument('--edge', type=int, default=640,
                    help='long-edge resize (0 = keep original 1080p)')
    ap.add_argument('--workers', type=int, default=11)
    args = ap.parse_args()

    out = Path(args.out).resolve()
    for s in ('train', 'val', 'test'):
        (out / 'images' / s).mkdir(parents=True, exist_ok=True)
        (out / 'mask' / s).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / s).mkdir(parents=True, exist_ok=True)

    jobs = []
    for split, videos, cap in (('train', TRAIN_VIDEOS, args.n_per_train),
                               ('val', VAL_VIDEOS, args.n_per_val),
                               ('test', TEST_VIDEOS, args.n_per_test)):
        for video in videos:
            jobs.append((split, video, cap, args.min_area, str(out), args.edge))

    counts = {s: 0 for s in ('train', 'val', 'test')}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_video, j): j[1] for j in jobs}
        done_n = 0
        for fut in as_completed(futs):
            split, video, written = fut.result()
            done_n += 1
            if written >= 0:
                counts[split] += written
            print(f'[{done_n}/{len(jobs)}] {split}/{video}: kept {written} '
                  f'({time.time()-t0:.0f}s elapsed)', flush=True)

    for split in ('train', 'val', 'test'):
        for folder in ('images', 'mask'):
            suffix = '' if folder == 'images' else '2'
            files = sorted((out / folder / split).glob('*.jpg'))
            with open(out / f'{split}{suffix}.txt', 'w') as fh:
                fh.writelines(str(f).replace(os.sep, '/') + '\n' for f in files)
    print('TOTALS:', counts, flush=True)


if __name__ == '__main__':
    main()
