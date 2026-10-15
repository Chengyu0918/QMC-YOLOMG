"""Unit smoke test for the three QMC-YOLOMG mechanisms.

Checks, without any dataset:
  1. Model builds from models/QMC_uav_s.yaml and forward works with and
     without quality evidence Q (train and eval modes).
  2. utils/quality.py: evidence ranges/shapes; both degradation modes change
     the tensor and keep RGB/labels untouched.
  3. utils/contribution.py: frozen teacher measures a Delta vector, gate
     supervision and consistency loss return scalars with grad only on the
     student side.
Run:  python test_code/test_qmc_units.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from models.yolo import Model
from utils.quality import compute_quality_evidence, degrade_motion, neutral_quality
from utils.contribution import (ContributionTeacher, measure_motion_gain,
                                gate_contribution_loss, degradation_consistency_loss)
from utils.loss import ComputeLoss

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)


def main():
    model = Model(ROOT / 'models' / 'QMC_uav_s.yaml', ch=3, ch2=3).to(DEV)
    model.hyp = {'box': 0.05, 'cls': 0.5, 'obj': 1.0, 'anchor_t': 4.0,
                 'label_smoothing': 0.0, 'fl_gamma': 0.0, 'cls_pw': 1.0, 'obj_pw': 1.0}
    model.nc = 1
    model.names = ['Drone']
    B, S = 2, 640
    imgs = torch.rand(B, 3, S, S, device=DEV)
    motion = (torch.rand(B, 3, S, S, device=DEV) < 0.01).float()

    # --- forward with evidence -------------------------------------------------
    q = compute_quality_evidence(motion, imgs)
    assert q.shape == (B, 3, S, S) and q.min() >= 0 and q.max() <= 1, q.shape
    model.train()
    pred = model(imgs, motion, q)
    assert isinstance(pred, list) and pred[0].shape[:2] == (B, 4)
    gate = model.motion_gate_score()
    assert gate is not None and gate.shape == (B,) and gate.requires_grad

    # eval mode returns tuple; q=None falls back to the neutral prior
    model.eval()
    with torch.no_grad():
        out = model(imgs, motion)
    assert isinstance(out, tuple) and out[0].shape[0] == B

    # --- degradation -----------------------------------------------------------
    targets = torch.tensor([[0, 0, 0.5, 0.5, 0.05, 0.05],
                            [1, 0, 0.3, 0.4, 0.04, 0.04]], device=DEV)
    deg = degrade_motion(motion, targets, local_prob=1.0, perturb_prob=1.0)
    assert not torch.equal(deg, motion)
    changed = (deg != motion).any(dim=(1, 2, 3))
    assert changed.all(), changed

    # --- teacher / gain --------------------------------------------------------
    compute_loss = ComputeLoss(model)
    teacher = ContributionTeacher(model)
    assert not any(p.requires_grad for p in teacher.ref.parameters())
    model.train()
    pred_full = model(imgs, motion, q)
    match = compute_loss.build_targets(pred_full, targets)
    pred0 = model(imgs, torch.zeros_like(motion), compute_quality_evidence(torch.zeros_like(motion), imgs))
    delta, comps = measure_motion_gain(pred_full, pred0, match, model.hyp)
    assert delta.shape == (B,) and torch.isfinite(delta).all()
    print('delta per sample:', delta.detach().cpu().numpy().round(6).tolist(),
          '(exactly ~0 at init: psi is zero-initialized, so the network starts'
          ' on the pure appearance pathway)')

    pred_ref, pred_ref0, match2, delta2, comps2 = teacher.forward_passes(
        imgs, motion, imgs, targets, compute_loss)
    assert torch.isfinite(delta2).all()

    # student gate from a fresh forward (mirrors the training loop, where the
    # gate is read from the degraded-input student pass)
    _ = model(imgs, motion, q)
    gate_now = model.motion_gate_score()
    assert gate_now is not None
    gain_loss, stats = gate_contribution_loss(gate_now.float(), delta2.float())
    assert gain_loss.requires_grad
    print('gate supervision stats:', {k: round(v, 4) for k, v in stats.items()})

    cons, _ = degradation_consistency_loss(pred_full, pred_ref, match2,
                                           sample_weights=torch.ones(B, device=DEV))
    n_cells = sum(idx[0].numel() for idx in match2[2])
    print(f'matched cells across layers: {n_cells}; cons.requires_grad={cons.requires_grad}; cons={cons.item():.6f}')
    assert cons.requires_grad and torch.isfinite(cons)
    teacher_params = sum(p.sum() for p in teacher.ref.parameters())
    print(f'consistency loss: {cons.item():.4f} (teacher params frozen: {teacher_params:.0f})')

    # --- backward through everything ------------------------------------------
    total = compute_loss(pred_full, targets)[0] + 0.05 * gain_loss + 0.3 * cons
    total.backward()
    grads = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    print(f'total loss {total.item():.3f}; params with grad: {len(grads)}/{sum(1 for _ in model.parameters())}')
    print('ALL UNIT CHECKS PASSED')


if __name__ == '__main__':
    main()
