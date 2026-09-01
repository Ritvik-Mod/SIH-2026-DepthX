#!/usr/bin/env python
"""Is the recovered sun azimuth degenerate under reflection about north-south?

The per-city azimuth histograms from the cached estimates are bimodal, and in all three
cities the two modes straddle roughly 180 deg (DC ~154/194, NYC ~135/215, PHL ~136/236).
Two readings, pointing opposite ways:

  (a) real morning and afternoon sorties.  Then the within-cluster spread is the true
      per-tile error and it is much tighter than the +-23..36 the pooled sd reports,
      and A6 is on firmer ground than it looked.
  (b) a search degeneracy.  occlusion() marches along u = (sin az, -cos az); replacing
      az with 360-az flips u_x and leaves u_y alone -- a MIRROR about the north-south
      axis, not the 180 deg rotation the antipodal test already ruled out.  Rectilinear
      street grids are near mirror-symmetric about that axis, so the two renderings can
      score alike.  Under (b) a large fraction of tiles carry a 60-80 deg azimuth error,
      and the A6 shadow loss would be supervised toward the wrong sun.

(b) forces the modes to straddle EXACTLY 180, since az and 360-az always average to it.
(a) has no reason to.  That is what this measures.

A single 36-point azimuth sweep per tile at the cached elevation yields all of it:
  offset   0  -> IoU at the cached azimuth
  offset 180  -> IoU at the antipode          (confirms the distributional check)
  360-2*az    -> IoU at the mirror 360-az     (the untested degeneracy)
plus the shape of the peak, whose half-width says whether a +-25 deg error is even
large enough to matter to the loss.

The sweep also recovers the per-tile IoU that estimate_sun.py computed and discarded;
--dump-iou writes it out so A6 can weight tiles by confidence instead of trusting a
0.13-IoU estimate as much as a 0.60 one.
"""
import sys, os, json, argparse, math, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, h5py
from tqdm import tqdm
from omegaconf import OmegaConf

from heightmap.data.gamus import index_tiles, IMAGE_KINDS
from heightmap.losses.shadow import predicted_shadow, observed_shadow

# Identical to scripts/estimate_sun.py -- the comparison is only meaningful if the
# detector and renderer settings match the ones that produced the cache.
CFG = OmegaConf.create(dict(tau=1.0, tau_dark=0.35, sigma_s=0.08, local_contrast=True,
                            bg_ksize=65, tau_rel=0.05, use_blue=True, tau_blue=0.02,
                            sigma_blue=0.03, max_steps=16, max_dist_px=150.0))
STEP = 10.0                       # sweep resolution; cached azimuths are multiples of 10
NOFF = int(360 / STEP)


@torch.no_grad()
def sweep(h, obs, gsd, elev, az0, dev):
    """IoU at az0 + k*STEP for every k, in one batched call.  -> (NOFF,) numpy."""
    azs = [(az0 + k * STEP) % 360.0 for k in range(NOFF)]
    hb = h.expand(NOFF, -1, -1, -1)                     # view, not a copy
    soft, rel = predicted_shadow(
        hb, torch.full((NOFF,), gsd, device=dev),
        torch.full((NOFF,), float(elev), device=dev),
        torch.tensor(azs, dtype=torch.float32, device=dev),
        tau=CFG.tau, max_steps=CFG.max_steps, max_dist_px=CFG.max_dist_px)
    p = (soft > 0.5) & rel
    o = (obs > 0.5) & rel
    inter = (p & o).flatten(1).sum(1).float()
    union = (p | o).flatten(1).sum(1).float().clamp(min=1.0)
    return (inter / union).cpu().numpy()


def load_tile(root, split, tid, size):
    with h5py.File(f"{root}/heights/{split}/{tid}_AGL.h5") as f:
        agl = np.asarray(f["image"][()], np.float32)
    img_path = None
    for kind in IMAGE_KINDS:                            # DC/PHL _RGB, NYC _IMG
        cand = f"{root}/images/{split}/{tid}_{kind}.h5"
        if os.path.exists(cand):
            img_path = cand; break
    if img_path is None:
        return None
    with h5py.File(img_path) as f:
        rgb = np.asarray(f["image"][()])
    H = agl.shape[0]; o = max(0, (H - size) // 2); s = min(size, H)
    agl, rgb = agl[o:o+s, o:o+s], rgb[o:o+s, o:o+s]
    if not np.isfinite(agl).all() or agl.max() < 3.0:
        return None
    return agl, rgb


def circ(deg):
    """-> (mean_deg, circular_sd_deg, R) for an array of degrees."""
    a = np.radians(np.asarray(deg, float))
    C, S = np.cos(a).mean(), np.sin(a).mean()
    R = float(math.hypot(C, S))
    mu = math.degrees(math.atan2(S, C)) % 360
    sd = math.degrees(math.sqrt(-2 * math.log(R))) if R > 1e-9 else float("inf")
    return mu, sd, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/GAMUS")
    ap.add_argument("--cache", default="data/GAMUS/sun_angles.json")
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--per-city", type=int, default=200)
    ap.add_argument("--gsd", type=float, default=0.33)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--min-sep", type=float, default=40.0,
                    help="skip the mirror comparison when |wrap(360-2*az)| is below "
                         "this: the mirror is then within the IoU peak's own half-max "
                         "width of az and the two are not distinguishable in principle")
    ap.add_argument("--dump-iou", default="",
                    help="write {tid: [elev, azim, iou, margin_vs_mirror]} here")
    a = ap.parse_args()

    dev = torch.device("cuda" if (a.device != "cpu" and torch.cuda.is_available()) else "cpu")
    print(f"device: {dev}")
    cache = json.load(open(a.cache))

    # ---- exact azimuth histogram, no binning artefact: values are multiples of 10 ----
    by_city = collections.defaultdict(list)
    for k, v in cache.items():
        by_city[k.split("_")[0]].append(v)
    print("\n=== exact azimuth histogram (10 deg resolution) ===")
    for c, v in sorted(by_city.items()):
        az = np.array([x[1] for x in v], float)
        cnt = collections.Counter(int(round(x)) for x in az)
        mu, sd, R = circ(az)
        print(f"\n{c}  n={len(v)}  circ.mean {mu:.1f}  circ.sd {sd:.1f}  R {R:.3f}")
        line = [f"{k:>4d}:{cnt[k]:<5d}" for k in sorted(cnt) if cnt[k]]
        for i in range(0, len(line), 8):
            print("   " + "".join(line[i:i+8]))

    # ---- two-cluster circular split: if the bimodality is real sorties, the spread
    # WITHIN a cluster is the true per-tile error, and the pooled sd overstates it ----
    print("\n=== two-cluster circular split (Lloyd on the circle, 40 restarts) ===")
    print(f"{'city':5s} {'k':>2s} {'n':>5s} {'mean':>7s} {'circ.sd':>8s} {'R':>6s}"
          f" {'elev p50':>9s} {'mirror of other mode?':>22s}")
    for c, v in sorted(by_city.items()):
        az = np.array([x[1] for x in v], float); el = np.array([x[0] for x in v], float)
        rad = np.radians(az); xy = np.stack([np.cos(rad), np.sin(rad)], 1)
        best = None
        rng = np.random.default_rng(0)
        for _ in range(40):
            ctr = xy[rng.choice(len(xy), 2, replace=False)]
            for _it in range(50):
                lab = np.argmax(xy @ ctr.T, 1)
                new = np.stack([xy[lab == k].mean(0) if (lab == k).any() else ctr[k]
                                for k in range(2)])
                new /= np.linalg.norm(new, axis=1, keepdims=True).clip(1e-9)
                if np.allclose(new, ctr):
                    break
                ctr = new
            obj = float((xy * ctr[lab]).sum())
            if best is None or obj > best[0]:
                best = (obj, lab.copy(), ctr.copy())
        _, lab, ctr = best
        mus = [math.degrees(math.atan2(ctr[k][1], ctr[k][0])) % 360 for k in range(2)]
        for k in np.argsort(mus):
            m = lab == k
            if not m.any():
                continue
            mu, sd, R = circ(az[m])
            other = mus[1 - k]
            sep = abs(((360.0 - mu) - other + 180) % 360 - 180)   # is other mode ~= mirror of this one
            print(f"{c:5s} {int(k):2d} {int(m.sum()):5d} {mu:7.1f} {sd:8.1f} {R:6.3f}"
                  f" {np.median(el[m]):9.1f} {sep:19.1f} deg")

    # ---- tid -> split, so a tile can be loaded without knowing which split it is ----
    where = {}
    for split in a.splits:
        for _city, tid in index_tiles(a.root, split):
            where[tid] = split

    # sample evenly through each city's azimuth-sorted list so BOTH modes are covered
    picked = []
    for c, _ in sorted(by_city.items()):
        ids = sorted([t for t in cache if t.split("_")[0] == c and t in where],
                     key=lambda t: cache[t][1])
        if len(ids) > a.per_city:
            ids = [ids[int(round(i * (len(ids) - 1) / (a.per_city - 1)))]
                   for i in range(a.per_city)]
        picked += [(c, t) for t in ids]
    print(f"\nrescoring {len(picked)} tiles, {NOFF} azimuths each")

    res = collections.defaultdict(list)
    dump = {}
    for c, tid in tqdm(picked, disable=None, mininterval=10.0):
        t = load_tile(a.root, where[tid], tid, a.size)
        if t is None:
            continue
        agl, rgb = t
        el, az = float(cache[tid][0]), float(cache[tid][1])
        h = torch.from_numpy(np.nan_to_num(agl)).float()[None, None].to(dev)
        im = torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(dev) / 255.0
        obs = observed_shadow(im, CFG.tau_dark, CFG.sigma_s, CFG.local_contrast,
                              CFG.bg_ksize, CFG.tau_rel, CFG.use_blue,
                              CFG.tau_blue, CFG.sigma_blue)
        v = sweep(h, obs, a.gsd, el, az, dev)
        k_mirror = int(round(((360.0 - 2.0 * az) % 360.0) / STEP)) % NOFF
        k_anti = NOFF // 2
        sep = abs(((360.0 - 2.0 * az) + 180.0) % 360.0 - 180.0)
        res[c].append((v, v[0], v[k_mirror], v[k_anti], int(v.argmax()), az, sep))
        dump[tid] = [el, az, round(float(v[0]), 4), round(float(v[0] - v[k_mirror]), 4)]

    print("\n=== degeneracy: IoU at cached az vs its mirror (360-az) and antipode ===")
    print(f"{'city':5s} {'n':>4s} {'IoU@az':>8s} {'IoU@mir':>8s} {'IoU@anti':>9s}"
          f" {'med margin':>11s} {'mirror wins':>12s} {'mirror within .01':>18s}"
          f" {'argmax==az':>11s}")
    for c, rows in sorted(res.items()):
        keep = [r for r in rows if r[6] >= a.min_sep]
        drop = len(rows) - len(keep)
        am_all = np.array([r[4] for r in rows])
        if not keep:
            print(f"{c:5s} {0:4d}  -- every tile has |sep| < {a.min_sep:.0f} deg, "
                  f"mirror indistinguishable from az by construction "
                  f"(argmax==az {float((am_all==0).mean()):.3f})")
            continue
        at = np.array([r[1] for r in keep]); mi = np.array([r[2] for r in keep])
        an = np.array([r[3] for r in keep]); am = np.array([r[4] for r in keep])
        print(f"{c:5s} {len(keep):4d} {np.median(at):8.3f} {np.median(mi):8.3f}"
              f" {np.median(an):9.3f} {np.median(at-mi):11.3f}"
              f" {float((mi>at).mean()):12.3f} {float((mi>=at-0.01).mean()):18.3f}"
              f" {float((am==0).mean()):11.3f}   ({drop} dropped, |sep|<{a.min_sep:.0f})")

    print("\n=== peak shape: mean normalised IoU vs azimuth offset from cached ===")
    off = [(k if k <= NOFF//2 else k-NOFF) * STEP for k in range(NOFF)]
    order = np.argsort(off); off = np.array(off)[order]
    for c, rows in sorted(res.items()):
        cur = []
        for r in rows:
            v = r[0].astype(float)
            rng = v.max() - v.min()
            cur.append((v - v.min()) / rng if rng > 1e-9 else np.zeros_like(v))
        m = np.mean(cur, 0)[order]
        half = off[m >= 0.5]
        hw = (float(half.min()), float(half.max())) if half.size else (0.0, 0.0)
        print(f"\n{c}  half-max width {hw[0]:+.0f}..{hw[1]:+.0f} deg")
        sel = (np.abs(off) <= 90)
        print("   offset " + " ".join(f"{int(o):>5d}" for o in off[sel]))
        print("   norm   " + " ".join(f"{x:5.2f}" for x in m[sel]))

    if a.dump_iou:
        with open(a.dump_iou, "w") as f:
            json.dump(dump, f, indent=1)
        print(f"\nper-tile IoU + mirror margin -> {a.dump_iou}")


if __name__ == "__main__":
    main()
