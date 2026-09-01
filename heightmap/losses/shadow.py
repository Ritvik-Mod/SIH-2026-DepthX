"""Shadow-consistency loss.

Render the shadow the predicted height field would cast under the recorded sun angle,
compare against shadows detected in the image, penalise disagreement.

Two properties make it worth the trouble:
  * it needs no height labels, so it trains on imagery with no ground truth;
  * it is physics, so it holds identically in Philadelphia and Pune -- the one training
    signal here with no domain gap.

Note what it does NOT constrain: adding a constant to H everywhere leaves
H[p + t*u] - H[p] unchanged, so this term says nothing about absolute scale.  That is
exactly the gap L1 fills, which is why the two compose instead of competing, and why
this can only ever be a refinement on top of supervised training.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

LUMA = (0.299, 0.587, 0.114)


def _log_steps(max_dist: float, n: int) -> list[float]:
    """Shadows are contiguous regions, so dense sampling is unnecessary; log spacing
    gets the same detection with 16 samples instead of 200."""
    return sorted({round(float(x), 3) for x in torch.logspace(0, math.log10(max_dist), n)})


def predicted_shadow(height, gsd, sun_elev_deg, sun_azim_deg, tau=1.0,
                     max_steps=16, max_dist_px=200.0):
    """-> (soft_shadow (B,1,H,W), reliable (B,1,H,W) bool)

    Image axes: x = column increasing east, y = row increasing south (row 0 is north).
    Azimuth is a compass bearing clockwise from north, so the unit vector pointing
    TOWARD the sun is (sin az, -cos az).  Check: az=180 (sun due south) gives (0,+1),
    i.e. +y = south, so shadows fall toward -y = north.  Correct.

    Pixel p is shadowed when, somewhere along that direction, the surface rises above
    the sun ray leaving p:   max_t ( H[p+t*u] - H[p] - t*gsd*tan(elev) ) > 0
    Because it compares H at both ends it handles shadows landing on other buildings;
    there is no flat-ground assumption.
    """
    B, _, H, W = height.shape
    dev, dt = height.device, height.dtype
    az = torch.deg2rad(sun_azim_deg.to(dev, dt)).reshape(B, 1, 1)
    el = torch.deg2rad(sun_elev_deg.to(dev, dt)).reshape(B, 1, 1)
    ux, uy = torch.sin(az), -torch.cos(az)
    tan_el = torch.tan(el.clamp(min=math.radians(1.0)))
    g = gsd.to(dev, dt).reshape(B, 1, 1)

    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=dt),
                            torch.arange(W, device=dev, dtype=dt), indexing="ij")
    xs, ys = xs.unsqueeze(0), ys.unsqueeze(0)

    occ = torch.full((B, H, W), -1e4, device=dev, dtype=dt)
    exited = torch.zeros((B, H, W), device=dev, dtype=torch.bool)
    base = height.squeeze(1)

    for t in _log_steps(max_dist_px, max_steps):
        sx, sy = xs + t * ux, ys + t * uy
        inside = (sx >= 0) & (sx <= W - 1) & (sy >= 0) & (sy <= H - 1)
        gx = (2.0 * sx / max(W - 1, 1) - 1.0).clamp(-1, 1)
        gy = (2.0 * sy / max(H - 1, 1) - 1.0).clamp(-1, 1)
        grid = torch.stack([gx, gy], dim=-1)
        hs = F.grid_sample(height, grid, mode="bilinear",
                           padding_mode="border", align_corners=True).squeeze(1)
        cand = hs - base - t * g * tan_el
        occ = torch.maximum(occ, torch.where(inside, cand, torch.full_like(cand, -1e4)))
        exited |= ~inside

    soft = torch.sigmoid(occ / tau).unsqueeze(1)
    # If the ray left the frame and found no occluder, an occluder outside the tile could
    # still be casting here -- we cannot tell, so those pixels are not supervised.
    reliable = (~(exited & (occ <= 0))).unsqueeze(1)
    return soft, reliable


def observed_shadow(image01, tau_dark=0.35, sigma_s=0.08, local=True, bg_ksize=65,
                    tau_rel=0.05, use_blue=True, tau_blue=0.02, sigma_blue=0.03):
    """Soft shadow detection from the image.  Three conjoined cues, because any one
    alone has a large, specific failure mode:

      absolute darkness   -- flags dark roofs and asphalt as shadow
      local darkness      -- a uniformly dark roof is not dark vs its own neighbourhood,
                             so this removes most of that error
      relative blueness   -- shadowed ground is lit by the sky, not the sun, so it is
                             BLUER than its surroundings.  Dark vegetation is greener,
                             not bluer, which is what separates a tree canopy from the
                             shadow it casts.  Without this term, dense tree cover is
                             the single largest source of false positives.

    Deliberately soft throughout: a hard threshold has zero gradient and is brittle.

    This is still the weakest link in the shadow term.  Report it honestly: wet asphalt,
    dark water and freshly-tarred roofs all remain confusable with shadow.
    """
    w = torch.tensor(LUMA, device=image01.device, dtype=image01.dtype).view(1, 3, 1, 1)
    lum = (image01 * w).sum(dim=1, keepdim=True)
    score = torch.sigmoid((tau_dark - lum) / sigma_s)

    if local or use_blue:
        k = int(bg_ksize) | 1
        pad = (k // 2,) * 4
    if local:
        bg = F.avg_pool2d(F.pad(lum, pad, mode="reflect"), k, stride=1)
        score = score * torch.sigmoid(((bg - lum) - tau_rel) / sigma_s)
    if use_blue:
        tot = image01.sum(dim=1, keepdim=True).clamp(min=1e-4)
        blue = image01[:, 2:3] / tot
        bgb = F.avg_pool2d(F.pad(blue, pad, mode="reflect"), k, stride=1)
        score = score * torch.sigmoid(((blue - bgb) - tau_blue) / sigma_blue)
    return score


def shadow_consistency_loss(height, image01, gsd, sun, cfg_shadow, exclude=None):
    """sun: (B,2) [elev_deg, azim_deg]; NaN rows are skipped.  Returns a scalar."""
    elev, azim = sun[:, 0], sun[:, 1]
    usable = torch.isfinite(elev) & torch.isfinite(azim) & (elev > 1.0) & \
             (elev < float(cfg_shadow.max_sun_elev))
    if not bool(usable.any()):
        return height.new_zeros(())

    idx = usable.nonzero(as_tuple=True)[0]
    h, im, g = height[idx].float(), image01[idx].float(), gsd[idx].float()
    soft, reliable = predicted_shadow(h, g, elev[idx], azim[idx],
                                      tau=float(cfg_shadow.tau),
                                      max_steps=int(cfg_shadow.max_steps),
                                      max_dist_px=float(cfg_shadow.max_dist_px))
    obs = observed_shadow(im, float(cfg_shadow.tau_dark), float(cfg_shadow.sigma_s),
                          local=bool(getattr(cfg_shadow, "local_contrast", True)),
                          bg_ksize=int(getattr(cfg_shadow, "bg_ksize", 65)),
                          tau_rel=float(getattr(cfg_shadow, "tau_rel", 0.05)),
                          use_blue=bool(getattr(cfg_shadow, "use_blue", True)),
                          tau_blue=float(getattr(cfg_shadow, "tau_blue", 0.02)),
                          sigma_blue=float(getattr(cfg_shadow, "sigma_blue", 0.03)))
    valid = reliable
    if exclude is not None:
        valid = valid & (~exclude[idx])
    bce = F.binary_cross_entropy(soft.clamp(1e-6, 1 - 1e-6), obs, reduction="none")
    m = valid.to(bce.dtype)
    return (bce * m).sum() / m.sum().clamp(min=1.0)
