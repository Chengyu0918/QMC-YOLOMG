"""Round 3: the headline comparison on the FULL-quantity real ARD100.

Every annotated frame of the paper's video-level split (~43k train / ~10k val
/ ~72k test), 640 input (with the long-edge-640 pre-resize that is exactly
the training letterbox transform), 20 epochs, identical seed/hyp for both
groups:

  full_base     original YOLOMG (Concat3 fusion)
  full_qmc      complete QMC-YOLOMG (online quality evidence + local
                degradation + Delta gate supervision + consistency,
                teacher evaluated every 2nd step)

Then: test-split evaluation + gate-gain correlation for the gated variant.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

DATA = 'data/ARD100_full.yaml'
COMMON = ['--data', DATA, '--weights', 'yolov5s.pt',
          '--epochs', '18', '--batch-size', '8', '--imgsz', '640',
          '--device', '0', '--workers', '8', '--patience', '18',
          '--nbs', '8']

GROUPS = {
    'full_base': ['--cfg', 'models/YOLOMG_base.yaml',
                  '--motion-degrade-local', '0', '--motion-degrade-perturb', '0',
                  '--motion-gain-weight', '0', '--consistency-weight', '0'],
    'full_qmc': ['--cfg', 'models/QMC_uav_s.yaml', '--aux-every', '2'],
}


def sh(args, log):
    print(f'\n>>> {" ".join(args)}\n', flush=True)
    with open(log, 'w', encoding='utf-8') as f:
        p = subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def parse_train(log):
    txt = Path(log).read_text(encoding='utf-8', errors='ignore')
    out = {}
    m = re.search(r'Model Summary: (\d+) layers, (\d+) parameters, \d+ gradients, ([\d.]+) GFLOPs', txt)
    if m:
        out.update(layers=int(m.group(1)), params=int(m.group(2)), gflops=float(m.group(3)))
    m = re.search(r'(\d+) epochs completed in ([\d.]+) hours', txt)
    if m:
        out.update(hours=float(m.group(2)))
    return out


def parse_val(log):
    txt = Path(log).read_text(encoding='utf-8', errors='ignore')
    out = {}
    m = re.search(r'^\s+all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', txt, re.M)
    if m:
        out.update(images=int(m.group(1)), labels=int(m.group(2)),
                   P=float(m.group(3)), R=float(m.group(4)),
                   mAP50=float(m.group(5)), mAP=float(m.group(6)))
    m = re.search(r'Speed: ([\d.]+)ms pre-process, ([\d.]+)ms inference, ([\d.]+)ms NMS', txt)
    if m:
        out.update(pre_ms=float(m.group(1)), inf_ms=float(m.group(2)), nms_ms=float(m.group(3)))
    return out


def parse_gategain(log):
    txt = Path(log).read_text(encoding='utf-8', errors='ignore')
    out = {}
    for key, pat in [('gate_mean', r'gate\s+mean ([\d.]+)'),
                     ('gate_std', r'gate\s+mean [\d.]+\s+std ([\d.]+)'),
                     ('delta_mean', r'delta\s+mean (-?[\d.]+)'),
                     ('delta_std', r'delta\s+mean -?[\d.]+\s+std ([\d.]+)'),
                     ('pearson_all', r'Pearson\s+\(all\)\s+([-+][\d.]+|[\d.]+)'),
                     ('spearman_all', r'Spearman \(all\)\s+([-+][\d.]+|[\d.]+)'),
                     ('pearson_tgt', r'Pearson\s+\(targets\)\s+([-+][\d.]+|[\d.]+)'),
                     ('spearman_tgt', r'Spearman \(targets\)\s+([-+][\d.]+|[\d.]+)')]:
        m = re.search(pat, txt)
        if m:
            out[key] = float(m.group(1))
    return out


def main():
    only = sys.argv[1:] or list(GROUPS)
    results = {}
    for name in only:
        t0 = time.time()
        rc = sh([PY, 'train.py', *COMMON, *GROUPS[name], '--name', name, '--exist-ok'],
                ROOT / f'_{name}_train.log')
        if rc != 0:
            print(f'!! training {name} failed rc={rc}', flush=True)
            continue
        sh([PY, 'val.py', '--weights', f'runs/train/{name}/weights/best.pt',
            '--data', DATA, '--task', 'test', '--imgsz', '640',
            '--batch-size', '8', '--device', '0', '--workers', '6',
            '--name', f'eval_{name}', '--exist-ok'],
           ROOT / f'_{name}_val.log')
        r = parse_train(ROOT / f'_{name}_train.log')
        r.update(parse_val(ROOT / f'_{name}_val.log'))
        if name != 'full_base':
            sh([PY, 'eval_gate_gain.py',
                '--weights', f'runs/train/{name}/weights/best.pt',
                '--data', DATA, '--split', 'test', '--imgsz', '640',
                '--batch-size', '8', '--device', '0',
                '--name', name], ROOT / f'_{name}_gate.log')
            r.update(parse_gategain(ROOT / f'_{name}_gate.log'))
        results[name] = r
        print(f'== {name} done in {(time.time()-t0)/60:.1f} min: {r}', flush=True)
        (ROOT / '_results3_partial.json').write_text(json.dumps(results, indent=2))

    (ROOT / '_results3_final.json').write_text(json.dumps(results, indent=2))
    print('\nROUND3 DONE', flush=True)


if __name__ == '__main__':
    main()
