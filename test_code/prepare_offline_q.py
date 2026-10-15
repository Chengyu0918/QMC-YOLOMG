"""Generate OFFLINE ground-truth registration-quality evidence Q (route A).

For every annotated frame f already present in datasets/ARD100_full_clean,
re-run the SAME KLT+RANSAC homography registration used for the mask
(neighbours f-2 / f+2) and save three evidence channels aligned to f:

    ch0 q_res  exp(-reproj_residual/tau)   — per-grid-point forward/backward
                reprojection error of RANSAC inliers, scattered and smoothed
    ch1 q_cov  inlier coverage             — fraction of inlier grid points
                in a local window
    ch2 q_val  warp valid region           — polygon of the source frame
                projected through H (outside = invalid), like the author's
                motion_compensate mask

Output: datasets/ARD100_full_clean/q/<split>/<stem>.png (uint8, 3ch, 0-255).
Frame selection mirrors prepare_ard100.py exactly (same lists, same stride),
so stems match the existing images/mask one-to-one.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'test_code'))

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

LK = dict(winSize=(15, 15), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))
TAU_PX = 1.5       # reprojection residual (px at long-edge-640 scale) halving q_res
SMOOTH_K = 85      # box filter for scattering grid evidence (640 scale)


def grid_points(shape):
    h, w = shape
    gh, gw = 48, 64
    ys = (np.arange(gh) + 0.5) * h / gh
    xs = (np.arange(gw) + 0.5) * w / gw
    pts = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 1, 2).astype(np.float32)
    return pts


def register(ref_gray, nbr_gray, shape):
    """KLT grid tracking + RANSAC homography -> spatial evidence maps.

    Returns (q_res, q_cov, q_val) float maps [0,1] at the input resolution.
    Registration direction matches motion_compensate(nbr, ref): warp nbr onto ref.
    All scattering is vectorized (np.add.at).
    """
    p0 = grid_points(shape)
    p1, st, err = cv2.calcOpticalFlowPyrLK(nbr_gray, ref_gray, p0, None, **LK)
    good = (st == 1).ravel()
    ones = np.ones(shape, np.float32)
    if good.sum() < 15:
        return ones, ones, ones
    src, dst = p0[good].reshape(-1, 2), p1[good].reshape(-1, 2)
    Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if Hm is None:
        return ones, ones, ones
    inl = inl.ravel().astype(bool)

    h, w = shape
    xs = np.clip(src[:, 0].astype(np.int32), 0, w - 1)
    ys = np.clip(src[:, 1].astype(np.int32), 0, h - 1)
    # reprojection residual; RANSAC outliers contribute the worst-case value
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    resid = np.linalg.norm(proj - dst, axis=1).astype(np.float32)
    resid = np.where(inl, resid, 2.0 * TAU_PX)
    acc = np.zeros(shape, np.float32)
    cnt = np.zeros(shape, np.float32)
    np.add.at(acc, (ys, xs), resid)
    np.add.at(cnt, (ys, xs), 1.0)
    acc = cv2.blur(acc, (SMOOTH_K, SMOOTH_K))
    cnt = cv2.blur(cnt, (SMOOTH_K, SMOOTH_K))
    res_map = acc / np.maximum(cnt, 1e-6)
    q_res = np.exp(-res_map / TAU_PX)

    # inlier coverage: window fraction of grid points that are RANSAC inliers
    inl_map = np.zeros(shape, np.float32)
    all_map = np.zeros(shape, np.float32)
    np.add.at(inl_map, (ys[inl], xs[inl]), 1.0)
    np.add.at(all_map, (ys, xs), 1.0)
    q_cov = cv2.blur(inl_map, (SMOOTH_K, SMOOTH_K)) / \
        np.maximum(cv2.blur(all_map, (SMOOTH_K, SMOOTH_K)), 1e-6)

    # warp valid region (author's mask convention)
    corners = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], np.float32)
    try:
        poly = cv2.perspectiveTransform(corners, np.linalg.inv(Hm)).astype(np.int32)
    except np.linalg.LinAlgError:
        poly = corners.astype(np.int32)
    valid = np.zeros(shape, np.uint8)
    cv2.fillPoly(valid, [poly.reshape(-1, 1, 2)], 1)
    return q_res, np.clip(q_cov, 0, 1), valid.astype(np.float32)


def process_video(job):
    split, video, out_root = job
    vdir = 'train_videos' if video in TRAIN_VIDEOS or video in VAL_VIDEOS else 'test_videos'
    vpath = ARD / vdir / f'{video}.mp4'
    out = Path(out_root)
    if not vpath.exists():
        return split, video, -1
    stems = [p.stem for p in sorted((out / 'mask' / split).glob(f'{video}_*.jpg'))]
    if not stems:
        return split, video, 0
    wanted = sorted(int(s.split('_')[-1]) for s in stems)
    need = set()
    for f in wanted:
        need.update((f - 2, f, f + 2))
    need.discard(0)

    # PASS 1: spill needed frames as grayscale pngs resized to a long edge of
    # 640 (the evidence is low-frequency and the training letterbox targets the
    # same scale; this makes KLT/blurs ~9x cheaper). Output Q maps therefore
    # share the 640-long-edge geometry of the stored mask images.
    tmpdir = out / '_qtmp' / f'{split}_{video}'
    tmpdir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(vpath))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx in need:
            g = cv2.cvtColor(cv2.GaussianBlur(frame, (11, 11), 0), cv2.COLOR_BGR2GRAY)
            h0, w0 = g.shape
            r = 640 / max(h0, w0)
            g = cv2.resize(g, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(tmpdir / f'{idx:06d}.png'), g)
    cap.release()

    # PASS 2: register each wanted frame against its two neighbours.
    written = 0
    for f in wanted:
        p_ref, p_a, p_b = (tmpdir / f'{f + d:06d}.png' for d in (0, -2, 2))
        if not (p_ref.exists() and p_a.exists() and p_b.exists()):
            continue
        g_ref = cv2.imread(str(p_ref), 0)
        g_a = cv2.imread(str(p_a), 0)
        g_b = cv2.imread(str(p_b), 0)
        shape = g_ref.shape[:2]
        try:
            r1, c1, v1 = register(g_ref, g_a, shape)
            r2, c2, v2 = register(g_ref, g_b, shape)
        except Exception as e:
            print(f'q failed {video}_{f}: {e}', flush=True)
            continue
        q = np.stack([(r1 + r2) / 2, (c1 + c2) / 2, (v1 + v2) / 2], -1)
        q = (np.clip(q, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(out / 'q' / split / f'{video}_{f:04d}.png'), q)
        written += 1
    for p in tmpdir.glob('*.png'):
        p.unlink()
    tmpdir.rmdir()
    return split, video, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ROOT / 'datasets/ARD100_full_clean'))
    ap.add_argument('--workers', type=int, default=11)
    args = ap.parse_args()
    out = Path(args.root).resolve()
    for s in ('train', 'val', 'test'):
        (out / 'q' / s).mkdir(parents=True, exist_ok=True)
    jobs = [(split, v, str(out)) for split, vids in
            (('train', TRAIN_VIDEOS), ('val', VAL_VIDEOS), ('test', TEST_VIDEOS)) for v in vids]
    t0 = time.time()
    done = tot = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_video, j) for j in jobs]
        for f in as_completed(futs):
            split, video, written = f.result()
            tot += max(written, 0)
            done += 1
            if done % 10 == 0:
                print(f'[{done}/{len(jobs)}] {tot} q-maps ({time.time()-t0:.0f}s)', flush=True)
    print(f'TOTAL q-maps: {tot} -> {out / "q"}', flush=True)


if __name__ == '__main__':
    main()
