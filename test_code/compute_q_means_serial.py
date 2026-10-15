"""Stage-4 FINAL: serial GPU computation of per-channel Q means over TRAIN split.

Structure identical to the validated mini-check (_q_mini.py): serial imread ->
resize -> batch of 32 -> GPU compute_quality_evidence. Single pass over
train_all.txt. The threaded variant (compute_q_means_gpu.py) produced wrong
values and is kept only as a failed-attempt record.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.quality import compute_quality_evidence  # noqa: E402

lines = [l.strip() for l in (ROOT / 'datasets/ARD100_full_clean/train_all.txt').read_text().splitlines() if l.strip()]
print(f'train frames: {len(lines)}', flush=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', dev, flush=True)

sums = torch.zeros(3)
n = 0
buf_m, buf_r = [], []
for k, line in enumerate(lines):
    mm = cv2.imread(line.replace('/images/', '/mask/'), 0)
    rgb = cv2.imread(line)
    if mm is None or rgb is None or min(mm.shape) < 64:
        continue
    h, w = mm.shape
    s = 640 / max(h, w)
    buf_m.append(cv2.resize(mm, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA))
    buf_r.append(cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA))
    if len(buf_m) >= 32:
        mt = torch.from_numpy(np.stack(buf_m)).float().div(255)[:, None].repeat(1, 3, 1, 1).to(dev)
        rt = torch.from_numpy(np.stack(buf_r)[:, :, ::-1].copy()).permute(0, 3, 1, 2).float().div(255).to(dev)
        with torch.no_grad():
            qv = compute_quality_evidence(mt, rt)
        sums += qv.mean(dim=(0, 2, 3)).cpu() * len(buf_m)
        n += len(buf_m)
        buf_m, buf_r = [], []
        if n % 25600 < 32:
            print(f'{n}/{len(lines)} running: {(sums / n).numpy().round(4)}', flush=True)
if buf_m:
    mt = torch.from_numpy(np.stack(buf_m)).float().div(255)[:, None].repeat(1, 3, 1, 1).to(dev)
    rt = torch.from_numpy(np.stack(buf_r)[:, :, ::-1].copy()).permute(0, 3, 1, 2).float().div(255).to(dev)
    with torch.no_grad():
        qv = compute_quality_evidence(mt, rt)
    sums += qv.mean(dim=(0, 2, 3)).cpu() * len(buf_m)
    n += len(buf_m)

means = (sums / n).tolist()
out = {
    'statistic': 'per-channel mean of online Q over TRAIN split (frozen)',
    'rule': 'serial validated path: imread -> long-edge-640 INTER_AREA resize -> /255 -> '
            'compute_quality_evidence(motion, rgb); train split only, single pass',
    'train_frames_used': n,
    'train_list_md5': hashlib.md5((ROOT / 'datasets/ARD100_full_clean/train_all.txt').read_bytes()).hexdigest(),
    'q_res_mean': means[0], 'q_cov_mean': means[1], 'q_val_mean': means[2],
    'sanity_check': 'first-320 mean [0.858, 0.933, 0.844] reproduced by _q_mini.py',
    'frozen_at': datetime.now().isoformat(timespec='seconds'),
}
(ROOT / 'EVIDENCE').mkdir(exist_ok=True)
(ROOT / 'EVIDENCE/q_channel_means.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
