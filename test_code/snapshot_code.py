"""Snapshot code-version hashes into EVIDENCE/ — run BEFORE every training launch.

Usage:  python test_code/snapshot_code.py <run-name>
Writes EVIDENCE/code_<run-name>.json with MD5 of the 14 protocol-critical files,
plus the six data-list hashes. Compare against EVIDENCE/seed1_evidence.json to
detect any code drift since the frozen protocol.
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ['train.py', 'val.py', 'models/common.py', 'models/yolo.py',
        'utils/datasets.py', 'utils/quality.py', 'utils/contribution.py',
        'utils/augmentations.py', 'utils/loss.py', 'utils/metrics.py',
        'models/QMC_uav_s.yaml', 'models/YOLOMG_base.yaml',
        'data/ARD100_fullall.yaml', 'data/hyps/hyp.scratch-low.yaml']
LISTS = ['train_all.txt', 'train2_all.txt', 'val.txt', 'val2.txt', 'test.txt', 'test2.txt']


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


run = sys.argv[1] if len(sys.argv) > 1 else 'unnamed'
snap = {'run': run, 'time': datetime.now().isoformat(timespec='seconds'),
        'code': {f: md5(ROOT / f) for f in CODE},
        'data_lists': {f: md5(ROOT / 'datasets/ARD100_full_clean' / f) for f in LISTS}}
out = ROOT / 'EVIDENCE' / f'code_{run}.json'
out.write_text(json.dumps(snap, indent=2))
print(f'snapshot written: {out}')
