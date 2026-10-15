"""Local detection-gain supervision of the motion gate (innovation 2) and the
degradation-consistency constraint (innovation 3c) for QMC-YOLOMG.

The motion contribution of sample i is defined as

    Delta_i = l_i(appearance only) - l_i(appearance + motion)

where both losses are computed by a FROZEN reference network with the SAME
labels, the SAME anchor/cell matching (taken from the reference run with
motion, i.e. fixed matching relationships) and the SAME per-position loss
definition.  A positive Delta means motion information reduced the local
detection loss at the matched positions (motion helps); a negative Delta means
motion interfered; samples with |Delta| close to zero get a reduced
supervision weight instead of a hard zero/one answer.

The reference network is a detached deep copy of the student, created after
warmup and refreshed only at epoch boundaries, so the gate can never generate
its own supervision signal (no gradient flows through the teacher and the
teacher's weights never follow the gate's within-epoch updates).

The per-position loss covers positive (matched) cells for box regression,
objectness and classification, plus a fixed number of hard background cells
(the highest-objectness unmatched cells), so the signal reflects both target
recognition/localisation gain and hard-background suppression gain.
"""

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.metrics import bbox_iou
from utils.torch_utils import clear_runtime_tensors, de_parallel


class ContributionTeacher:
    """Frozen reference network that measures the local detection gain.

    Registered protocol (stage 2): the teacher is loaded ONCE from a gate-free
    Concat3 checkpoint (the group-A run) via `from_checkpoint`, hashed, and
    NEVER refreshed during or across training — all U-enabled groups within a
    seed share the same teacher file.  The legacy EMA-copy constructor is kept
    only for backward compatibility with old logs and is NOT part of the
    frozen protocol.
    """

    def __init__(self, model):
        src = de_parallel(model)
        # Live gate tensors cached on fusion modules block deepcopy; they are
        # stale at teacher (re)creation time, so drop them safely.
        clear_runtime_tensors(src)
        ref = deepcopy(src).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.motion_dropout = 0.0  # never drop motion inside the teacher
        self.ref = ref
        self.epoch = -1
        self.source = 'student-ema-copy (LEGACY, not protocol)'

    @classmethod
    def from_checkpoint(cls, path, device):
        """Build the protocol teacher from a gate-free Concat3 checkpoint."""
        import hashlib

        ckpt = torch.load(path, map_location='cpu')
        ref = (ckpt.get('ema') or ckpt['model']).float().to(device).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.motion_dropout = 0.0
        h = hashlib.md5(Path(path).read_bytes()).hexdigest()
        obj = cls.__new__(cls)
        obj.ref = ref
        obj.epoch = 10 ** 9  # sentinel: never matches a training epoch -> never refreshed
        obj.source = f'checkpoint:{path} md5:{h[:12]}'
        return obj

    def refresh(self, model, epoch):
        """LEGACY per-epoch refresh — DISABLED under the frozen protocol.
        Checkpoint teachers carry an unreachable epoch sentinel, so this never
        fires for them; kept only so old code paths remain readable."""
        if self.source.startswith('checkpoint:'):
            return self  # protocol teachers are never refreshed
        self.__init__(model)
        self.epoch = epoch
        return self

    @torch.no_grad()
    def forward_passes(self, imgs, motion, rgb, targets, compute_loss, hard_neg_k=8,
                       quality_fn=None):
        """Run the two frozen reference passes.

        Returns (pred_full, pred_nomot, match, delta, comps):
            pred_full    reference prediction WITH motion (also the clean-side
                         teacher for the consistency loss)
            pred_nomot   reference prediction WITHOUT motion
            match        anchor/cell matching from the with-motion pass
            delta        (B,) per-sample gain l(no motion) - l(motion)
            comps        per-component gains (box/obj/cls/neg)
        """
        from utils.quality import compute_quality_evidence
        if quality_fn is None:
            quality_fn = compute_quality_evidence

        zero_motion = torch.zeros_like(motion)
        q_motion = quality_fn(motion, rgb)
        q_zero = quality_fn(zero_motion, rgb)

        # The reference network is in eval mode, so Detect returns
        # (inference_output, train_output); the raw train_output carries the
        # logits needed by the fixed-matching loss evaluation.
        out_full = self.ref(imgs, motion, q_motion)
        out_nomot = self.ref(imgs, zero_motion, q_zero)
        pred_full = out_full[1] if isinstance(out_full, tuple) else out_full
        pred_nomot = out_nomot[1] if isinstance(out_nomot, tuple) else out_nomot

        # Fixed matching: taken once from the with-motion reference pass and
        # used unchanged for both loss evaluations.
        match = compute_loss.build_targets(pred_full, targets)
        delta, comps = measure_motion_gain(pred_full, pred_nomot, match,
                                           compute_loss.hyp, bs=imgs.shape[0])
        return pred_full, pred_nomot, match, delta, comps

    @torch.no_grad()
    def clean_forward(self, imgs, motion, rgb, quality_fn=None):
        """Clean-side teacher pass for the degradation-consistency loss.

        The consistency objective constrains the student (degraded input)
        toward the teacher on the CLEAN input — 'before/after degradation'
        by design.  Separate from forward_passes, whose U labels are measured
        on the degraded input per the stage-5 rule."""
        if quality_fn is None:
            from utils.quality import compute_quality_evidence as quality_fn
        q = quality_fn(motion, rgb)
        out = self.ref(imgs, motion, q)
        pred = out[1] if isinstance(out, tuple) else out
        return pred


def _select_hard_neg_cells(pi, match_i, k):
    """Pick the top-k unmatched cells by objectness logit for one layer.

    The SAME cells must be evaluated on both predictions, otherwise the gain
    comparison is no longer controlled; callers pass the returned indices to
    both evaluations.
    """
    b, a, gj, gi = match_i
    logits = pi[..., 4].detach()
    mask = torch.ones_like(logits, dtype=torch.bool)
    if b.numel():
        mask[b, a, gj, gi] = False
    if not mask.any():
        return None
    logits = logits.masked_fill(~mask, float('-inf'))
    k = min(k, int(mask.sum().item()))
    topv, topi = logits.view(logits.shape[0], -1).topk(k, dim=1)
    sel = torch.isfinite(topv)
    if not sel.any():
        return None
    bi = torch.arange(logits.shape[0], device=pi.device).unsqueeze(1).expand_as(topi)[sel]
    oi = topi[sel]
    cells = pi.shape[2] * pi.shape[3]
    ai = torch.div(oi, cells, rounding_mode='floor') % pi.shape[1]
    rem = oi % cells
    gi_n = rem % pi.shape[3]
    gj_n = torch.div(rem, pi.shape[3], rounding_mode='floor')
    return bi, ai, gj_n, gi_n


def local_detection_losses(pred, match, hyp, hard_neg_cells=None):
    """Per-position detection losses at FIXED matched cells.

    `pred` is the raw Detect training output (list of (B,na,ny,nx,no) tensors,
    logits).  The matching `match` comes from build_targets on the reference
    pass, and the exact same cells (positives AND hard negatives) are
    evaluated for both the with-motion and without-motion predictions, which
    is what makes the difference a controlled comparison.

    Args:
        hard_neg_cells: optional list (per layer) of index tuples produced by
            _select_hard_neg_cells; when given, no new selection is made.

    Returns:
        (parts, hard_neg_cells) where parts maps 'box'/'obj'/'cls'/'neg' to a
        list of (values (N,), sample_ids (N,)) pairs.
    """
    tcls, tbox, indices, anchors = match
    device = pred[0].device
    out = {k: [] for k in ('box', 'obj', 'cls', 'neg')}
    neg_cells = hard_neg_cells if hard_neg_cells is not None else [None] * len(pred)

    for i, pi in enumerate(pred):
        b, a, gj, gi = indices[i]
        n = b.shape[0]
        if n:
            ps = pi[b, a, gj, gi]
            pxy = ps[:, :2].sigmoid() * 2 - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True).clamp(0, 1)
            out['box'].append(((1.0 - iou) * hyp['box'], b))

            tobj = iou.detach().clamp(0).float()  # gr = 1.0 iou-ratio target
            obj_pos = F.binary_cross_entropy_with_logits(ps[:, 4], tobj, reduction='none')
            out['obj'].append((obj_pos * hyp['obj'], b))

            if pi.shape[-1] > 6:  # classification term (nc > 1)
                nc = pi.shape[-1] - 5
                t = torch.zeros_like(ps[:, 5:])
                t[torch.arange(n, device=device), tcls[i]] = 1.0
                cls_pos = F.binary_cross_entropy_with_logits(ps[:, 5:], t, reduction='none').mean(1)
                out['cls'].append((cls_pos * hyp['cls'], b))

        if neg_cells[i] is None:
            neg_cells[i] = _select_hard_neg_cells(pi, indices[i], 8)
        cells = neg_cells[i]
        if cells is not None:
            bi, ai, gj_n, gi_n = cells
            if bi.numel():
                neg_val = F.binary_cross_entropy_with_logits(
                    pi[bi, ai, gj_n, gi_n, 4],
                    pi.new_zeros(bi.numel()), reduction='none')
                out['neg'].append((neg_val * hyp['obj'], bi))

    return {k: v for k, v in out.items() if v}, neg_cells


def per_sample_losses(part_list, bs=None):
    """Average a list of (values, sample_ids) into a per-sample vector.

    `bs` pins the output length so components covering different subsets of
    the batch (e.g. positives only in images 2-3, hard negatives in 0-3) stay
    addable.
    """
    if not part_list:
        return None
    vals = torch.cat([v for v, _ in part_list])
    ids = torch.cat([b for _, b in part_list]).long()
    if bs is None:
        bs = int(ids.max().item()) + 1
    sum_ = vals.new_zeros(bs)
    cnt = vals.new_zeros(bs)
    sum_.index_add_(0, ids, vals)
    cnt.index_add_(0, ids, torch.ones_like(vals))
    return sum_ / cnt.clamp_min(1.0), cnt


def measure_motion_gain(pred_full, pred_nomot, match, hyp, hard_neg_k=8, bs=None):
    """Delta_i = l_i(no motion) - l_i(motion) per sample, per component.

    Both predictions are evaluated at exactly the same matched cells and the
    same hard-negative cells (selected once from the with-motion prediction),
    with the same loss definitions, so a positive delta means the motion
    branch lowered the local detection loss.
    """
    parts_full, neg_cells = local_detection_losses(pred_full, match, hyp)
    parts_nomot, _ = local_detection_losses(pred_nomot, match, hyp, hard_neg_cells=neg_cells)
    delta, comps = None, {}
    for key in parts_full:
        lf = per_sample_losses(parts_full[key], bs)
        ln = per_sample_losses(parts_nomot[key], bs)
        if lf is None or ln is None:
            continue
        d = ln[0] - lf[0]  # positive -> motion reduced the loss
        comps[key] = (d, lf[1])
        delta = d if delta is None else delta + d
    return delta, comps


def gate_contribution_loss(gate_score, delta, margin=0.01, tau=0.1, near_zero=0.05):
    """Supervise the per-sample motion-utilization score with the measured gain.

    near_zero: |delta| below this (in loss units) is treated as
    "uninformative" and its BCE weight is linearly reduced to zero, so neutral
    samples do not force the gate toward an arbitrary value.
    """
    w = (delta.abs() / max(near_zero, 1e-6)).clamp(0, 1)
    target = ((delta - margin) / max(tau, 1e-6)).sigmoid()
    bce = F.binary_cross_entropy(gate_score.clamp(1e-4, 1 - 1e-4), target, reduction='none')
    loss = (w * bce).sum() / w.sum().clamp_min(1.0)
    stats = dict(gate=gate_score.detach().mean().item(),
                 delta=delta.detach().mean().item(),
                 target=target.detach().mean().item(),
                 weight=w.detach().mean().item())
    return loss, stats


def degradation_consistency_loss(student_pred, teacher_pred, match, sample_weights=None, box_w=2.0):
    """Constrain the student (degraded motion) to match the teacher (clean
    motion) at label-supported cells: objectness/class logits via soft-BCE and
    box decoding via smooth-L1 on the sigmoid outputs.  The teacher side is
    fully detached, so only the student receives gradients.

    sample_weights: optional (B,) tensor; cells of sample i are weighted by
    sample_weights[i] so the constraint applies where degradation actually
    happened (or with any other per-sample weighting).
    """
    _, _, indices, _ = match
    losses = student_pred[0].new_zeros(())
    wsum = student_pred[0].new_zeros(())
    count = 0
    for i, (ps, pt) in enumerate(zip(student_pred, teacher_pred)):
        b, a, gj, gi = indices[i]
        n = b.shape[0]
        if n == 0:
            continue
        s = ps[b, a, gj, gi]
        t = pt[b, a, gj, gi].detach()
        obj = F.binary_cross_entropy_with_logits(s[:, 4], t[:, 4].sigmoid(), reduction='none')
        box = F.smooth_l1_loss(s[:, :4].sigmoid(), t[:, :4].sigmoid(), reduction='none').mean(1)
        loss_i = obj + box_w * box
        if s.shape[1] > 6:
            cls = F.binary_cross_entropy_with_logits(s[:, 5:], t[:, 5:].sigmoid(), reduction='none').mean(1)
            loss_i = loss_i + cls
        if sample_weights is not None:
            w = sample_weights.to(loss_i.dtype)[b]
            loss_i = loss_i * w
            wsum = wsum + w.sum()
        losses = losses + loss_i.sum()
        count += n
    if count == 0:
        return student_pred[0].new_zeros(()), student_pred[0].new_zeros(())
    denom = wsum if sample_weights is not None else torch.as_tensor(float(count), device=losses.device)
    return losses / denom.clamp_min(1.0), losses.detach() / count
