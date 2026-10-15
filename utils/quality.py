"""Registration-quality evidence and targeted motion degradation for QMC-YOLOMG.

Innovation 1 (quality evidence): the fusion network receives a per-pixel
registration-quality map Q with three channels computed from the motion
difference image itself:

    q_res   local background reprojection-residual proxy (low-pass energy of
            the difference map; high residual floor -> unreliable registration)
    q_cov   spatial coverage of quiet, inlier-like background (local fraction
            of pixels whose difference response is below a small threshold)
    q_valid warp valid-region estimate (geometric border prior combined with a
            local correlation detector for warp-fill "shadow" artifacts where
            the difference map copies the RGB frame because the compensated
            frame was filled with zeros outside the source image)

When the dataset is regenerated from videos with test_code/generate_quality.py,
the same three channels can be produced from the true registration stage
(RANSAC reprojection residuals, inlier mask, warped polygon).  The online
estimator below keeps the precomputed ARD100/NPS mask data usable without
regeneration and, importantly, remains valid AFTER training-time degradation,
because it is computed from the degraded tensor that the network actually sees.

Innovation 3 (degradation training): two complementary corruptions are applied
to the motion tensor only, never to the RGB input or the labels:

    local attenuation   motion response inside labelled object boxes is scaled
                        down, simulating weak-motion / hovering targets
    registration        the whole difference map is resampled with a random
    perturbation        small affine transform and the newly exposed border is
                        invalidated, emulating a difference map regenerated
                        under a perturbed homography while keeping the spatial
                        structure of registration error
"""

import math

import torch
import torch.nn.functional as F

# Quality-evidence hyperparameters (fixed defaults, tunable via kwargs).
RESIDUAL_TAU = 0.06      # residual floor that halves the quality response
QUIET_THRESH = 0.04      # difference value below which a pixel is "quiet"
BORDER_FRACTION = 0.10   # image border fraction treated as potentially invalid
BOX_KERNEL = 31          # local window (pixels) for residual/coverage estimates
SHADOW_CORR = 0.5        # local correlation above which a region looks warp-filled


def _box_mean(x, k=BOX_KERNEL):
    """Local mean with a k x k box filter (kept symmetric via reflect padding)."""
    p = k // 2
    return F.avg_pool2d(F.pad(x, (p, p, p, p), mode='replicate'), k, stride=1)


def compute_quality_evidence(motion, rgb=None, residual_tau=RESIDUAL_TAU,
                             quiet_thresh=QUIET_THRESH, border_fraction=BORDER_FRACTION,
                             box_kernel=BOX_KERNEL):
    """Estimate a 3-channel registration-quality map from the motion input.

    Args:
        motion: (B,3,H,W) tensor in [0,1]; only its mean luminance is used.
        rgb:    optional (B,3,H,W) tensor in [0,1] enabling the warp-shadow
                correlation term in the validity channel.
    Returns:
        (B,3,H,W) float tensor in [0,1]; higher values mean the registration
        evidence in that region is more trustworthy. Channel order is
        (residual quality, inlier coverage, valid region).
    """
    out_dtype = motion.dtype
    m = motion.float().mean(1, keepdim=True).clamp(0, 1)

    # 1) Local background residual floor.
    res = _box_mean(m, box_kernel)
    q_res = torch.exp(-res / max(residual_tau, 1e-6))

    # 2) Local coverage of quiet (inlier-like) pixels.
    q_cov = _box_mean((m < quiet_thresh).float(), box_kernel)

    # 3) Valid-region estimate.
    b, _, h, w = m.shape
    dy = torch.arange(h, device=m.device, dtype=m.dtype).view(1, 1, h, 1)
    dx = torch.arange(w, device=m.device, dtype=m.dtype).view(1, 1, 1, w)
    margin = max(border_fraction, 1e-6) * min(h, w)
    dist = torch.minimum(torch.minimum(dx, (w - 1) - dx), torch.minimum(dy, (h - 1) - dy))
    q_valid = (dist / margin).clamp(0, 1)

    if rgb is not None:
        g = rgb.float().mean(1, keepdim=True).clamp(0, 1)
        mg = _box_mean(m * g, box_kernel)
        m_mean = _box_mean(m, box_kernel)
        g_mean = _box_mean(g, box_kernel)
        cov = mg - m_mean * g_mean
        var_m = (_box_mean(m * m, box_kernel) - m_mean * m_mean).clamp_min(0)
        var_g = (_box_mean(g * g, box_kernel) - g_mean * g_mean).clamp_min(0)
        corr = cov / (var_m * var_g).clamp_min(1e-8).sqrt()
        shadow = (corr - SHADOW_CORR).clamp(0, 1) / max(1 - SHADOW_CORR, 1e-6)
        q_valid = q_valid * (1 - shadow)

    return torch.cat((q_res, q_cov, q_valid), 1).clamp(0, 1).to(out_dtype)


def neutral_quality(motion, rgb=None):
    """Constant prior (0.75) used when registration evidence is disabled."""
    b, _, h, w = motion.shape
    return torch.full((b, 3, h, w), 0.75, device=motion.device, dtype=motion.dtype)


# Frozen TRAIN-split per-channel means of the online Q evidence (stage 4).
# Source: EVIDENCE/q_channel_means.json (2026-10-09T18:24:25) — computed over
# all 115,278 train_all.txt frames (md5 9fc24db6…) via the serial validated
# path; no val/test frame involved. Frozen for every split and every group.
Q_TRAIN_MEANS = (0.8354769349098206, 0.9108575582504272, 0.8464781641960144)


def mean_quality(motion, rgb=None):
    """Group-M Q input: per-channel constants equal to the TRAIN-split means
    of the online evidence. Removes sample-specific Q information while
    keeping every Q entry/module/channel, and keeps the input distribution
    on-manifold instead of the arbitrary 0.75 used by group B."""
    b, _, h, w = motion.shape
    q = torch.empty((b, 3, h, w), device=motion.device, dtype=motion.dtype)
    for c, v in enumerate(Q_TRAIN_MEANS):
        q[:, c].fill_(v)
    return q


# ---------------------------------------------------------------------------
# Targeted motion degradation (innovation 3a)
# ---------------------------------------------------------------------------

def attenuate_object_motion(motion, targets, probability=0.25, attenuation=0.35, expand=1.3):
    """Scale down the motion response inside labelled object boxes.

    Simulates targets whose motion evidence is weak (slow targets, near-hover
    flight, motion cancelled by camera following).  Labels are normalized
    xywh in the dataset convention (image, class, cx, cy, w, h).
    """
    if probability <= 0 or targets.numel() == 0:
        return motion
    out = motion.clone()
    bs, _, h, w = out.shape
    active = torch.rand(bs, device=out.device) < probability
    for row in targets:
        row = row.to(out.device)
        b = int(row[0].item())
        if b >= bs or not active[b]:
            continue
        cx, cy, bw, bh = row[2:6]
        ew, eh = bw * expand, bh * expand
        x1 = max(int((cx - ew / 2) * w), 0)
        x2 = min(int((cx + ew / 2) * w) + 1, w)
        y1 = max(int((cy - eh / 2) * h), 0)
        y2 = min(int((cy + eh / 2) * h) + 1, h)
        if x2 > x1 and y2 > y1:
            out[b, :, y1:y2, x1:x2] *= attenuation
    return out


def perturb_registration(motion, probability=0.15, max_shift=0.02, max_scale=0.03,
                         max_angle_deg=1.5):
    """Resample the difference map with a random small affine transform.

    This stands in for regenerating the difference image after perturbing the
    homography used for motion compensation: the whole response field (target
    blobs AND background residual structure) is displaced consistently, which
    preserves the spatial structure of registration error, unlike per-pixel
    noise.  The border band exposed by the resampling is zeroed, matching the
    valid-region change of a real perturbed warp.
    """
    if probability <= 0 or motion.ndim != 4:
        return motion
    bs = motion.shape[0]
    active = torch.rand(bs, device=motion.device) < probability
    if not active.any():
        return motion
    dev = motion.device
    act = active.float()
    rad = math.radians(max_angle_deg)
    # Random affine parameters; inactive samples collapse to the identity.
    scale = 1 + (torch.rand(bs, device=dev) * 2 - 1) * max_scale * act
    angle = (torch.rand(bs, device=dev) * 2 - 1) * rad * act
    tx = (torch.rand(bs, device=dev) * 2 - 1) * max_shift * 2 * act
    ty = (torch.rand(bs, device=dev) * 2 - 1) * max_shift * 2 * act
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    theta = torch.zeros(bs, 2, 3, device=dev, dtype=torch.float32)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    x32 = motion.float()
    grid = F.affine_grid(theta, x32.size(), align_corners=False)
    shifted = F.grid_sample(x32, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    ones = torch.ones_like(x32)
    cover = F.grid_sample(ones, grid, mode='nearest', padding_mode='zeros', align_corners=False)
    shifted = shifted * (cover > 0.999).to(shifted.dtype)
    out = torch.where(active[:, None, None, None], shifted.to(motion.dtype), motion)
    return out


def degrade_motion(motion, targets, local_prob=0.25, perturb_prob=0.15,
                   attenuation=0.35):
    """Apply both degradation modes with independent sample probabilities."""
    out = attenuate_object_motion(motion, targets, local_prob, attenuation)
    out = perturb_registration(out, perturb_prob)
    return out
