"""Measure the DEPLOYED model against reference data: RMSE, MAE, bias and correlation.

    pip install modal numpy pillow h5py
    python scripts/evaluate_live.py --out results/validation.json

Runs every image through the live Modal deployment (the same GPU pipeline the website
uses, called directly so the public per-IP rate limit does not apply) and scores the
returned rasters against ground truth that the model never trained on:

  swisstopo, 10 Swiss scenes (data/swisstopo/)   independent: other country, other sensor,
      hilly terrain with houses. Reference DSM, DTM and AGL from airborne LiDAR
      (swissSURFACE3D / swissALTI3D), 0.33 m/px, same grid as the input.

  GAMUS, 6 Washington DC tiles (data/GAMUS/*/val/)   urban, with per-pixel land-cover
      classes. VALIDATION split: never trained on, but the split the checkpoint was
      selected on, so mildly optimistic. Stated wherever the number is shown.

Three quantities are scored, because they answer different questions:

  AGL        height above ground, the model's own output. The honest measure of skill.
  DSM (ref)  predicted AGL placed on the REFERENCE terrain: isolates the model's error
             inside a full surface model.
  DSM (live) the complete pipeline as a user gets it: predicted AGL on the Copernicus
             GLO-30 DTM. Includes the DEM's own error and the datum difference
             (swisstopo is LN02, Copernicus is EGM2008), reported raw and with the
             single median offset over open ground removed.

A DSM correlation is dominated by the terrain itself and comes out near 1.0 whatever the
model does; the AGL correlation is the one that measures the model.

Strata:
  swisstopo pixels   buildings   = reference AGL > 2.5 m and not green
                     vegetation  = reference AGL > 2.5 m and green (excess-green index)
                     open ground = reference AGL <= 2.5 m
  swisstopo scenes   hilly  = relief >= 80 m   urban = built fraction >= 0.55
                     sparse = built fraction <= 0.36   (a scene can be in two groups)
  GAMUS pixels       official classes: building 3, tree 6, ground/low vegetation/road 1,2,5
"""
import argparse
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SWISS = ROOT / "data" / "swisstopo"
GAMUS = ROOT / "data" / "GAMUS"
OBJECT_M = 2.5


class Acc:
    """Streaming accumulator: MAE, RMSE, bias, Pearson r over any number of pixels."""

    def __init__(self):
        self.n = 0
        self.s = dict(e=0.0, ae=0.0, e2=0.0, x=0.0, y=0.0, xx=0.0, yy=0.0, xy=0.0)

    def add(self, pred, ref):
        pred = pred.astype(np.float64)
        ref = ref.astype(np.float64)
        if pred.size == 0:
            return
        e = pred - ref
        self.n += pred.size
        s = self.s
        s["e"] += e.sum(); s["ae"] += np.abs(e).sum(); s["e2"] += (e * e).sum()
        s["x"] += pred.sum(); s["y"] += ref.sum()
        s["xx"] += (pred * pred).sum(); s["yy"] += (ref * ref).sum(); s["xy"] += (pred * ref).sum()

    def result(self):
        if not self.n:
            return None
        n, s = self.n, self.s
        cov = s["xy"] / n - (s["x"] / n) * (s["y"] / n)
        vx = s["xx"] / n - (s["x"] / n) ** 2
        vy = s["yy"] / n - (s["y"] / n) ** 2
        r = cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
        return {"mae_m": round(s["ae"] / n, 3), "rmse_m": round(float(np.sqrt(s["e2"] / n)), 3),
                "bias_m": round(s["e"] / n, 3), "r": round(float(r), 4), "pixels": int(n)}


def read_tif(p):
    return np.asarray(Image.open(p), dtype=np.float32)


def excess_green(rgb):
    f = rgb.astype(np.float32)
    s = f.sum(-1) + 1e-6
    return (2 * f[..., 1] - f[..., 0] - f[..., 2]) / s


class Live:
    """Calls the deployed GPU class directly and returns the height raster it produced."""

    def __init__(self):
        import modal
        self.cls = modal.Cls.from_name("depthx", "Inference")
        self.state = modal.Dict.from_name("depthx-state")
        self.ids = []

    def run(self, name, body, auto_dem):
        jid = "e" + uuid.uuid4().hex[:11]      # "e" prefix: evaluation runs, easy to clear
        self.ids.append(jid)
        t0 = time.time()
        self.state[f"job:{jid}"] = {"id": jid, "status": "queued", "submitted_at": t0}
        self.state["busy"] = (self.state.get("busy", 0) or 0) + 1
        r = self.cls().run.remote(jid, name, body, {
            "auto_dem": auto_dem, "render_quantity": "dsm" if auto_dem else "agl"})
        if "error" in r:
            raise RuntimeError(r["error"])
        h = np.asarray(Image.open(io.BytesIO(r["files"]["heightmap.tif"])), dtype=np.float32)
        meta = json.loads(r["files"]["metadata.json"])
        return h, meta, r["timing"]


def valid(*arrs):
    m = np.ones(arrs[0].shape, bool)
    for a in arrs:
        m &= np.isfinite(a) & (a > -9000)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "validation.json"))
    a = ap.parse_args()
    live = Live()
    out = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
           "model": "a7_vitl_v0.1.0 (DINOv2 ViT-L + DPT, bins head)",
           "deployment": "Modal, NVIDIA L4, TF32", "scenes": {}, "groups": {}, "timing": []}

    groups = {k: {"agl": Acc(), "dsm_ref": Acc()} for k in
              ("swiss_all", "hilly", "urban", "sparse")}
    strata = {k: Acc() for k in ("swiss_buildings", "swiss_vegetation", "swiss_ground",
                                 "gamus_all", "gamus_buildings", "gamus_trees", "gamus_ground")}
    dsm_live = Acc()
    dsm_live_aligned = Acc()

    # ---------------------------------------------------------------- swisstopo
    for d in sorted(p for p in SWISS.iterdir() if p.is_dir()):
        name = d.name
        info = json.loads((d / f"{name}.json").read_text())
        rgb = np.asarray(Image.open(d / f"{name}_RGB.tif").convert("RGB"))
        agl_t, dsm_t, dtm_t = (read_tif(d / f"{name}_{k}.tif") for k in ("AGL", "DSM", "DTM"))
        body = (d / f"{name}_RGB.tif").read_bytes()

        agl_p, _, tim = live.run("input.tif", body, auto_dem=False)
        dsm_p, meta_dsm, tim2 = live.run("input.tif", body, auto_dem=True)
        out["timing"] += [{"scene": name, "px": list(agl_p.shape), **tim},
                          {"scene": name, "px": list(dsm_p.shape), "dtm": True, **tim2}]

        m = valid(agl_p, agl_t, dsm_t, dtm_t)
        green = excess_green(rgb) > 0.03
        obj = agl_t > OBJECT_M
        sc = {"agl": Acc(), "dsm_ref": Acc()}
        sc["agl"].add(agl_p[m], agl_t[m])
        sc["dsm_ref"].add((agl_p + dtm_t)[m], dsm_t[m])

        relief = info["terrain"]["relief_m"]
        built = info["terrain"]["built_fraction"]
        memb = ["swiss_all"] + (["hilly"] if relief >= 80 else []) + \
               (["urban"] if built >= 0.55 else []) + (["sparse"] if built <= 0.36 else [])
        for g in memb:
            groups[g]["agl"].add(agl_p[m], agl_t[m])
            groups[g]["dsm_ref"].add((agl_p + dtm_t)[m], dsm_t[m])
        strata["swiss_buildings"].add(agl_p[m & obj & ~green], agl_t[m & obj & ~green])
        strata["swiss_vegetation"].add(agl_p[m & obj & green], agl_t[m & obj & green])
        strata["swiss_ground"].add(agl_p[m & ~obj], agl_t[m & ~obj])

        # the full pipeline, as a user gets it
        m2 = valid(dsm_p, dsm_t)
        raw = Acc(); raw.add(dsm_p[m2], dsm_t[m2]); dsm_live.add(dsm_p[m2], dsm_t[m2])
        ground = m2 & ~obj
        off = float(np.median((dsm_p - dsm_t)[ground])) if ground.any() else 0.0
        al = Acc(); al.add((dsm_p - off)[m2], dsm_t[m2]); dsm_live_aligned.add((dsm_p - off)[m2], dsm_t[m2])

        out["scenes"][name] = {
            "dataset": "swisstopo", "relief_m": relief, "built_fraction": built, "groups": memb,
            "agl": sc["agl"].result(), "dsm_reference_terrain": sc["dsm_ref"].result(),
            "dsm_live_pipeline_raw": raw.result(),
            "dsm_live_pipeline_datum_aligned": al.result(), "datum_offset_m": round(off, 2),
            "dtm_source": (meta_dsm.get("terrain") or {}).get("dtm_source"),
        }
        s = out["scenes"][name]
        print(f"{name:13s} relief {relief:5.0f} m  AGL MAE {s['agl']['mae_m']:.2f} RMSE {s['agl']['rmse_m']:.2f} "
              f"r {s['agl']['r']:.3f} | live DSM MAE {s['dsm_live_pipeline_raw']['mae_m']:.2f} "
              f"(aligned {s['dsm_live_pipeline_datum_aligned']['mae_m']:.2f}, offset {off:+.2f})  "
              f"| model {tim['inference_s']}s", flush=True)

    # ---------------------------------------------------------------- GAMUS val (DC)
    for p in sorted((GAMUS / "images" / "val").glob("*_RGB.h5")):
        import h5py
        tile = p.name.replace("_RGB.h5", "")
        rgb = h5py.File(p)["image"][()]
        agl_t = h5py.File(GAMUS / "heights" / "val" / f"{tile}_AGL.h5")["image"][()].astype(np.float32)
        cls = h5py.File(GAMUS / "classes" / "val" / f"{tile}_CLS.h5")["image"][()].astype(np.int16)
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "PNG")
        agl_p, _, tim = live.run("input.png", buf.getvalue(), auto_dem=False)   # GAMUS GSD 0.33 = the assumed default
        out["timing"].append({"scene": tile, "px": list(agl_p.shape), **tim})
        m = valid(agl_p, agl_t) & (cls > 0)
        acc = Acc(); acc.add(agl_p[m], agl_t[m]); strata["gamus_all"].add(agl_p[m], agl_t[m])
        for k, sel in (("gamus_buildings", cls == 3), ("gamus_trees", cls == 6),
                       ("gamus_ground", np.isin(cls, (1, 2, 5)))):
            strata[k].add(agl_p[m & sel], agl_t[m & sel])
        out["scenes"][tile] = {"dataset": "GAMUS val (Washington DC)", "agl": acc.result()}
        r = acc.result()
        print(f"{tile:13s} urban DC      AGL MAE {r['mae_m']:.2f} RMSE {r['rmse_m']:.2f} r {r['r']:.3f} "
              f"| model {tim['inference_s']}s", flush=True)

    for g, v in groups.items():
        out["groups"][g] = {"agl": v["agl"].result(), "dsm_reference_terrain": v["dsm_ref"].result()}
    for k, v in strata.items():
        out["groups"][k] = {"agl": v.result()}
    out["groups"]["swiss_dsm_live_raw"] = {"dsm": dsm_live.result()}
    out["groups"]["swiss_dsm_live_datum_aligned"] = {"dsm": dsm_live_aligned.result()}
    out["evaluation_job_ids"] = live.ids

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    for k in ("swiss_all", "hilly", "urban", "sparse", "swiss_buildings", "swiss_vegetation",
              "swiss_ground", "gamus_all", "gamus_buildings", "gamus_trees", "gamus_ground"):
        r = out["groups"][k]["agl"]
        print(f"  {k:18s} AGL MAE {r['mae_m']:.2f} m  RMSE {r['rmse_m']:.2f} m  bias {r['bias_m']:+.2f}  r {r['r']:.3f}")
    for k in ("swiss_dsm_live_raw", "swiss_dsm_live_datum_aligned"):
        r = out["groups"][k]["dsm"]
        print(f"  {k:28s} DSM MAE {r['mae_m']:.2f} m  RMSE {r['rmse_m']:.2f} m  bias {r['bias_m']:+.2f}  r {r['r']:.4f}")


if __name__ == "__main__":
    main()
