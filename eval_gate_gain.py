"""Correlate the learned motion gate with the measured detection gain.

Evidence experiment for innovation 2: on a held-out set, run the frozen model
twice per sample (with motion / without motion), measure the per-sample local
detection gain Delta_i = l(appearance only) - l(appearance + motion) at fixed
matched cells, and correlate it with the gate's motion-utilization score.

Outputs a per-sample CSV plus Pearson/Spearman correlations (overall and on
the subset of samples that contain targets).  A positive, clearly non-zero
correlation is the evidence that the gate learned actual motion contribution
rather than an arbitrary attention pattern.

Usage:
    python eval_gate_gain.py --weights runs/train/qmc_smoke/weights/best.pt \
        --data data/smoke.yaml --imgsz 640 --device 0
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.datasets import create_dataloader
from utils.general import check_dataset, check_img_size, colorstr
from utils.loss import ComputeLoss
from utils.quality import compute_quality_evidence
from utils.contribution import measure_motion_gain
from utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(ROOT))


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _rank(x):
    x = np.asarray(x, dtype=np.float64)
    order = x.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def _spearman(a, b):
    return _pearson(_rank(a), _rank(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--data', default='data/smoke.yaml')
    ap.add_argument('--hyp', default='data/hyps/hyp.scratch-low.yaml')
    ap.add_argument('--split', default='val', choices=['val', 'test'],
                    help='which dataset split to correlate on')
    ap.add_argument('--quality-mode', default='online', choices=['online', 'neutral', 'offline'],
                    help='must match the training configuration')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--device', default='')
    ap.add_argument('--half', action='store_true')
    ap.add_argument('--project', default=ROOT / 'runs/gate_gain')
    ap.add_argument('--name', default='exp')
    args = ap.parse_args()

    device = select_device(args.device)
    ckpt = torch.load(args.weights, map_location='cpu')
    model = (ckpt.get('ema') or ckpt['model']).float().to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hyp = yaml.safe_load(open(args.hyp, errors='ignore'))
    model.hyp = hyp
    data = check_dataset(args.data)
    stride = max(int(model.stride.max()), 32)
    imgsz = check_img_size(args.imgsz, s=stride)
    compute_loss = ComputeLoss(model)

    dataloader = create_dataloader(data[args.split], data[f'{args.split}2'], imgsz, args.batch_size,
                                   stride, False, hyp=hyp, cache=False, rect=True,
                                   pad=0.5, workers=2, prefix=f'{args.split}: ',
                                   prefix2=f'{args.split}2: ')[0]

    rows = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='gate-gain', bar_format='{l_bar}{bar:10}{r_bar}')
        for im, im2, targets, paths, _, _, _ in pbar:
            im = im.to(device).float() / 255
            im2 = im2.to(device).float() / 255
            motion, q_off = im2[:, :3], im2[:, 3:]
            targets = targets.to(device)

            if args.quality_mode == 'offline':
                q, q0 = q_off, q_off  # ground-truth evidence is motion-independent
            elif args.quality_mode == 'online':
                q = compute_quality_evidence(motion, im)
                q0 = compute_quality_evidence(torch.zeros_like(motion), im)
            else:
                q = q0 = neutral_quality(motion, im)
            zero = torch.zeros_like(motion)

            out = model(im, motion, q)
            train_out = out[1] if isinstance(out, tuple) else out
            out0 = model(im, zero, q0)
            train_out0 = out0[1] if isinstance(out0, tuple) else out0

            gate = model.motion_gate_score()  # per-sample gate mean (with motion)
            match = compute_loss.build_targets(train_out, targets)
            delta, comps = measure_motion_gain(train_out, train_out0, match,
                                               hyp, bs=im.shape[0])
            if delta is None:
                continue
            n_pos = torch.bincount(targets[:, 0].long(), minlength=im.shape[0])
            for i in range(im.shape[0]):
                rows.append({
                    'image': Path(paths[i]).name,
                    'gate': float(gate[i]),
                    'delta': float(delta[i]),
                    'delta_box': float(comps['box'][0][i]) if 'box' in comps else 0.0,
                    'delta_obj': float(comps['obj'][0][i]) if 'obj' in comps else 0.0,
                    'delta_neg': float(comps['neg'][0][i]) if 'neg' in comps else 0.0,
                    'n_targets': int(n_pos[i]),
                })

    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / 'gate_gain.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    gate = [r['gate'] for r in rows]
    delta = [r['delta'] for r in rows]
    pos = [r for r in rows if r['n_targets'] > 0]
    gate_p = [r['gate'] for r in pos]
    delta_p = [r['delta'] for r in pos]

    report = (
        f"\n{colorstr('gate-gain correlation report')}  ({len(rows)} samples, "
        f"{len(pos)} with targets)\n"
        f"  gate        mean {np.mean(gate):.4f}  std {np.std(gate):.4f}\n"
        f"  delta       mean {np.mean(delta):.5f} std {np.std(delta):.5f}\n"
        f"  Pearson  (all)   {_pearson(gate, delta):+.4f}\n"
        f"  Spearman (all)   {_spearman(gate, delta):+.4f}\n"
        f"  Pearson  (targets)   {_pearson(gate_p, delta_p):+.4f}\n"
        f"  Spearman (targets)   {_spearman(gate_p, delta_p):+.4f}\n"
        f"  per-sample CSV: {csv_path}"
    )
    if np.std(delta) < 1e-6 or np.std(gate) < 1e-6:
        report += ("\n  WARNING: gate or delta is (near-)constant — the model is "
                   "under-trained or not using motion; correlations above are "
                   "numerically meaningless.")
    print(report)
    (save_dir / 'report.txt').write_text(report)


if __name__ == '__main__':
    main()
