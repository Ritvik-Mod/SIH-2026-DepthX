"""Augmentation.

Two hard rules encoded here, both from the design notes:
  * geometric transforms are applied IDENTICALLY to image / height / class / mask;
  * a flip or rotation changes the apparent sun direction, so when the shadow loss
    is active the caller must disable geometric augmentation (config.py enforces it).

Scale augmentation doubles as GSD supervision: crop round(out*s) source pixels and
resize to out, which makes the effective ground sample distance s * gsd_base while
leaving the height VALUES untouched (a 20 m building is 20 m at any resolution).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Sample:
    image: np.ndarray   # (H,W,3) uint8 or float32 0..1
    height: np.ndarray  # (H,W) float32 metres
    cls: np.ndarray     # (H,W) int64
    mask: np.ndarray    # (H,W) bool
    gsd: float          # metres/pixel


def scale_and_crop(s: Sample, out: int, scale_min: float, scale_max: float,
                   rng: np.random.Generator) -> Sample:
    H, W = s.height.shape
    smax = min(scale_max, min(H, W) / out)          # window must fit inside the tile
    smin = max(scale_min, 8.0 / out)                # keep the window sane
    scale = float(rng.uniform(smin, max(smin, smax)))
    win = int(round(out * scale))
    win = max(8, min(win, min(H, W)))
    y0 = int(rng.integers(0, H - win + 1))
    x0 = int(rng.integers(0, W - win + 1))
    return _take_window(s, y0, x0, win, out, scale)


def center_crop(s: Sample, out: int) -> Sample:
    H, W = s.height.shape
    win = min(out, H, W)
    y0, x0 = (H - win) // 2, (W - win) // 2
    return _take_window(s, y0, x0, win, out, win / out)


def _take_window(s: Sample, y0: int, x0: int, win: int, out: int, scale: float) -> Sample:
    img = s.image[y0:y0 + win, x0:x0 + win]
    hgt = s.height[y0:y0 + win, x0:x0 + win]
    cls = s.cls[y0:y0 + win, x0:x0 + win]
    msk = s.mask[y0:y0 + win, x0:x0 + win]
    if win != out:
        interp = cv2.INTER_AREA if win > out else cv2.INTER_LINEAR
        img = cv2.resize(img, (out, out), interpolation=interp)
        hgt = cv2.resize(hgt, (out, out), interpolation=interp)
        cls = cv2.resize(cls.astype(np.int32), (out, out), interpolation=cv2.INTER_NEAREST)
        msk = cv2.resize(msk.astype(np.uint8), (out, out), interpolation=cv2.INTER_NEAREST) > 0
    return Sample(np.ascontiguousarray(img), np.ascontiguousarray(hgt.astype(np.float32)),
                  np.ascontiguousarray(cls.astype(np.int64)),
                  np.ascontiguousarray(msk.astype(bool)), s.gsd * scale)


def geometric(s: Sample, rng: np.random.Generator) -> Sample:
    """Flips and 90-degree rotations only.  Arbitrary angles would interpolate the
    height field and soften exactly the discontinuities the gradient loss protects."""
    if rng.random() < 0.5:
        s = _apply(s, lambda a: a[:, ::-1])
    if rng.random() < 0.5:
        s = _apply(s, lambda a: a[::-1, :])
    k = int(rng.integers(0, 4))
    if k:
        s = _apply(s, lambda a: np.rot90(a, k, axes=(0, 1)))
    return s


def _apply(s: Sample, fn) -> Sample:
    return Sample(np.ascontiguousarray(fn(s.image)), np.ascontiguousarray(fn(s.height)),
                  np.ascontiguousarray(fn(s.cls)), np.ascontiguousarray(fn(s.mask)), s.gsd)


def photometric(img: np.ndarray, rng: np.random.Generator, cfg) -> np.ndarray:
    """Domain randomisation.  Operates on the image only, in 0..1 float.

    The point is to destroy appearance cues that are predictive on US aerial data but
    not causal for height (roof colour, colour balance, sharpness), so the model is
    pushed onto cues that transfer: shadow, footprint geometry, structural regularity.
    """
    x = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
    x = np.clip(x, 0, 1)

    x = np.clip(x * rng.uniform(0.80, 1.25), 0, 1)                       # brightness
    m = x.mean()
    x = np.clip((x - m) * rng.uniform(0.80, 1.25) + m, 0, 1)             # contrast
    g = x.mean(axis=2, keepdims=True)
    x = np.clip(g + (x - g) * rng.uniform(0.70, 1.35), 0, 1)             # saturation
    x = np.clip(x ** rng.uniform(0.80, 1.30), 0, 1)                      # gamma

    if rng.random() < 0.5:                                               # hue
        hsv = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-7, 8))) % 180
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    if rng.random() < cfg.p_haze:                                        # dust / haze
        a = float(rng.uniform(0.0, 0.35))
        warm = np.array([0.80, 0.78, 0.72], np.float32)
        x = np.clip((1 - a) * x + a * warm, 0, 1)

    if rng.random() < cfg.p_gray:                                        # force geometry cues
        x = np.repeat(x.mean(axis=2, keepdims=True), 3, axis=2)

    if rng.random() < cfg.p_blur:
        sig = float(rng.uniform(0.3, 1.0))
        x = cv2.GaussianBlur(x, (0, 0), sig)

    if rng.random() < cfg.p_jpeg:                                        # sensor/codec artefacts
        q = int(rng.integers(45, 96))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            x = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if rng.random() < cfg.p_noise:
        x = np.clip(x + rng.normal(0, float(rng.uniform(0, 0.03)), x.shape).astype(np.float32), 0, 1)

    return x


def normalise(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def denormalise(t):
    """Torch (B,3,H,W) normalised -> 0..1, for the shadow loss's luminance term."""
    import torch
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(1, 3, 1, 1)
    return (t * std + mean).clamp(0, 1)
