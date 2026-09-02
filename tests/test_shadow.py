"""Shadow loss validation against synthetic tiles with KNOWN sun angles and height.

Asserts what is determinate:
  1. the rendered occlusion matches an independent reimplementation  (implementation)
  2. sweeping azimuth against the IMAGE recovers the true bearing     (convention)
  3. the true azimuth always beats its 180-degree opposite            (sign)
Reports, without asserting, the +-45 discrimination: on tiles with few tall structures
the shadows are short and neighbouring azimuths are genuinely near-tied.  That is a
property of the signal, not a bug, and the loss is a soft regulariser, not a detector.
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, h5py
from omegaconf import OmegaConf
from heightmap.losses.shadow import shadow_consistency_loss, predicted_shadow

ROOT = "data/GAMUS_synthetic"
CFG = OmegaConf.create(dict(tau=1.0, tau_dark=0.35, sigma_s=0.08, local_contrast=True,
                            bg_ksize=65, tau_rel=0.05, use_blue=True, tau_blue=0.02,
                            sigma_blue=0.03, max_steps=16, max_dist_px=150.0, max_sun_elev=60.0))


def ref_occlusion(agl, gsd, elev, azim):
    """Independent numpy reimplementation, integer stepping."""
    n = agl.shape[0]
    az, el = math.radians(azim), math.radians(elev)
    ux, uy = math.sin(az), -math.cos(az)
    occ = np.full((n, n), -1e9, np.float32)
    ys, xs = np.mgrid[0:n, 0:n]
    for t in [1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 55, 75, 100, 140]:
        sy = np.clip((ys + t * uy).astype(int), 0, n - 1)
        sx = np.clip((xs + t * ux).astype(int), 0, n - 1)
        occ = np.maximum(occ, agl[sy, sx] - agl - t * gsd * math.tan(el))
    return occ > 0


def load(tid, split="train", n=512):
    """Centre-crop to n for speed: the sweep is O(angles * steps * pixels) on CPU."""
    with h5py.File(f"{ROOT}/images/{split}/{tid}_RGB.h5") as f: rgb = f["image"][()]
    with h5py.File(f"{ROOT}/heights/{split}/{tid}_AGL.h5") as f: agl = f["image"][()]
    H = agl.shape[0]; o = (H - n) // 2
    return rgb[o:o+n, o:o+n], agl[o:o+n, o:o+n]


def main():
    sun = json.load(open(f"{ROOT}/sun_angles.json"))
    tids = [t for t in sun if os.path.exists(f"{ROOT}/images/train/{t}_RGB.h5")
            and sun[t][0] < CFG.max_sun_elev][:3]
    assert tids, "no usable synthetic tiles"
    print(f"{len(tids)} tiles below the {CFG.max_sun_elev:.0f} deg sun-elevation guard\n")

    print("1) rendered occlusion vs independent reimplementation")
    for tid in tids:
        elev, azim = sun[tid]; _, agl = load(tid)
        h = torch.from_numpy(agl).float()[None, None]
        soft, rel = predicted_shadow(h, torch.tensor([0.5]), torch.tensor([elev]),
                                     torch.tensor([azim]), max_steps=16, max_dist_px=150.0)
        pb = (soft > 0.5).numpy()[0, 0]; rb = rel.numpy()[0, 0]
        ref = ref_occlusion(agl, 0.5, elev, azim)
        iou = (pb & ref & rb).sum() / max(((pb | ref) & rb).sum(), 1)
        print(f"   {tid:12s} IoU={iou:.3f}")
        assert iou > 0.75, f"{tid}: occlusion rendering disagrees with reference (IoU {iou:.3f})"

    print("\n2) azimuth sweep against the IMAGE (30 deg steps, 512px crop)")
    errs = []
    for tid in tids:
        elev, azim = sun[tid]; rgb, agl = load(tid)
        h = torch.from_numpy(agl).float()[None, None]
        im = torch.from_numpy(rgb).float().permute(2, 0, 1)[None] / 255.0
        best_a, best_l = None, 1e9
        for a in range(0, 360, 30):
            l = float(shadow_consistency_loss(h, im, torch.tensor([0.5]),
                                              torch.tensor([[elev, float(a)]]), CFG))
            if l < best_l:
                best_l, best_a = l, a
        err = min(abs(best_a - azim), 360 - abs(best_a - azim))
        errs.append(err)
        print(f"   {tid:12s} true={azim:6.1f}  argmin={best_a:3d}  err={err:5.1f} deg")
    print(f"   median azimuth error: {np.median(errs):.1f} deg")
    assert np.median(errs) <= 35.0, f"azimuth convention likely wrong (median err {np.median(errs):.1f})"

    print("\n3) true azimuth vs its 180-degree opposite (sign check)")
    for tid in tids:
        elev, azim = sun[tid]; rgb, agl = load(tid)
        h = torch.from_numpy(agl).float()[None, None]
        im = torch.from_numpy(rgb).float().permute(2, 0, 1)[None] / 255.0
        lt = float(shadow_consistency_loss(h, im, torch.tensor([0.5]), torch.tensor([[elev, azim]]), CFG))
        lo = float(shadow_consistency_loss(h, im, torch.tensor([0.5]),
                                           torch.tensor([[elev, (azim + 180) % 360]]), CFG))
        print(f"   {tid:12s} true={lt:.4f}  opposite={lo:.4f}  {'OK' if lt < lo else 'FAIL'}")
        assert lt < lo, f"{tid}: 180-degree flip scores better -> sign convention inverted"

    print("\n4) gradient flows to the height field")
    _, agl = load(tids[0]); elev, azim = sun[tids[0]]
    rgb, _ = load(tids[0])
    h = torch.from_numpy(agl).float()[None, None].requires_grad_(True)
    im = torch.from_numpy(rgb).float().permute(2, 0, 1)[None] / 255.0
    loss = shadow_consistency_loss(h, im, torch.tensor([0.5]), torch.tensor([[elev, azim]]), CFG)
    loss.backward()
    gn = float(h.grad.abs().sum())
    print(f"   loss={float(loss):.4f}  sum|dL/dH|={gn:.4f}")
    assert gn > 0, "no gradient reaches the height field"
    print("\n5) loss survives bf16 autocast, the precision training actually uses")
    # A6 crashed here after a full A5 warm start: binary_cross_entropy is unsafe to
    # autocast and torch refuses it.  Every other test runs in fp32, so nothing covered
    # the path the trainer takes.
    for dev, dt in (("cuda", torch.bfloat16), ("cpu", torch.bfloat16)):
        if dev == "cuda" and not torch.cuda.is_available():
            print("   cuda unavailable, skipping"); continue
        _, agl = load(tids[0]); elev, azim = sun[tids[0]]
        rgb, _ = load(tids[0])
        h = torch.from_numpy(agl).float()[None, None].to(dev).requires_grad_(True)
        im = torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(dev) / 255.0
        with torch.autocast(device_type=dev, dtype=dt):
            l = shadow_consistency_loss(h, im, torch.tensor([0.5], device=dev),
                                        torch.tensor([[elev, azim]], device=dev), CFG)
        l.backward()
        assert torch.isfinite(l), f"{dev}/{dt}: loss is not finite under autocast"
        assert float(h.grad.abs().sum()) > 0, f"{dev}/{dt}: no gradient under autocast"
        print(f"   {dev}/{str(dt).split('.')[-1]:9s} loss={float(l):.4f}  finite, gradient flows")

    print("\nPASS test_shadow")


if __name__ == "__main__":
    main()
