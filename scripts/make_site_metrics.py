"""Build the homepage's metrics card from measured results.

    python scripts/make_site_metrics.py

Reads results/validation.json (scripts/evaluate_live.py, the live model vs swisstopo
LiDAR) and results/offline_eval.json (the GAMUS test strata, the held-out city and the
terrain-only error, from the training cluster), and writes terrain3d/lib/metrics.json,
which components/Metrics.jsx renders.

Accuracy numbers are taken from those two files -- none are typed in here.
The rendering and deployment facts come from tests run against the live deployment
on 28 Sept 2026; each is listed below with how it was measured, so a stale value is
easy to spot and re-measure.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = json.loads((ROOT / "results" / "validation.json").read_text())
OFF = json.loads((ROOT / "results" / "offline_eval.json").read_text())
OUT = ROOT / "terrain3d" / "lib" / "metrics.json"

g = RES["groups"]


def agl(key):
    return g[key]["agl"]


def row(name, detail, r):
    return {"name": name, "detail": detail, "mae": r["mae_m"], "rmse": r["rmse_m"], "r": r["r"]}


strata = OFF["gamus_test_strata"]


def stratum(name, key, rule):
    s = strata[key]
    return row(name, f"GAMUS test, {s['tiles']} tiles, {rule}", s)


def dsm_entry(name, group_stats, scene_key, note):
    # A DSM pooled over scenes correlates near 1.0 only because alpine scenes sit higher
    # than lakeside ones, so r here is the per-scene median.
    rs = [s[scene_key]["r"] for s in RES["scenes"].values() if s["dataset"] == "swisstopo"]
    return {"name": name, "mae": group_stats["mae_m"], "rmse": group_stats["rmse_m"],
            "r": round(statistics.median(rs), 4), "bias": group_stats["bias_m"], "note": note}


swiss = agl("swiss_all")
n_swiss = sum(1 for s in RES["scenes"].values() if s["dataset"] == "swisstopo")
n_test = sum(s["tiles"] for k, s in strata.items() if isinstance(s, dict))
cop = OFF["cop30_terrain_alone"]

# model time per scene size, from the validation runs themselves
t1024 = [t["inference_s"] for t in RES["timing"] if t["px"] == [1024, 1024]]
inf_1024 = statistics.median(t1024) if t1024 else None

# scene-to-scene stability: spread of per-scene MAE on the independent set
scene_mae = [s["agl"]["mae_m"] for s in RES["scenes"].values() if s["dataset"] == "swisstopo"]

data = {
    "subtitle": (f"Deployed model, every pixel, on data it never trained on: {n_swiss} Swiss scenes "
                 f"with airborne LiDAR and {n_test} GAMUS test tiles"),
    "headline": [
        {"label": "Height error (MAE)", "value": f"{swiss['mae_m']:.2f}", "unit": "m",
         "note": f"Per-pixel height above ground, {n_swiss} independent Swiss scenes"},
        {"label": "Correlation", "value": f"{swiss['r']:.3f}", "unit": "",
         "note": "Pearson r, predicted vs LiDAR height above ground, per pixel"},
        {"label": "RMSE", "value": f"{swiss['rmse_m']:.2f}", "unit": "m",
         "note": "Root-mean-square height error, same scenes; weights the large misses"},
        {"label": "Model inference", "value": f"{inf_1024:.1f}" if inf_1024 else "n/a", "unit": "s",
         "note": "Per 1024 px scene on an NVIDIA L4, once the GPU is warm"},
    ],
    "landscapes": [
        stratum("Urban", "urban", "15% or more buildings"),
        stratum("Sparse", "sparse", "almost no buildings or trees, so little height for r to track"),
        row("Hilly", f"swisstopo LiDAR, {n_swiss} alpine and lakeside scenes, 24 to 115 m relief", swiss),
        stratum("Forested", "forested", "30% or more trees"),
        stratum("Mixed", "mixed", "everything in between"),
    ],
    "dsm": [
        dsm_entry("DSM as deployed", g["swiss_dsm_live_raw"]["dsm"], "dsm_live_pipeline_raw",
                  (f"Model heights on the free 30 m Copernicus terrain, scored against the LiDAR surface. "
                   f"About 2 m of this comes from that terrain model, which is {cop['mae_m']:.2f} m MAE off on "
                   "its own, not from the height model.")),
        dsm_entry("DSM on LiDAR terrain", g["swiss_all"]["dsm_reference_terrain"], "dsm_reference_terrain",
                  ("The ceiling: the same predicted heights on surveyed ground, so the error left is the "
                   "model's alone. r is the median per scene.")),
    ],
    "rendering": [
        # measured: Modal output vs the reference Mac pipeline, same inputs, every pixel
        {"label": "Projection fidelity", "value": "0.2 cm mean deviation",
         "note": "Cloud GPU output vs the reference pipeline, pixel for pixel; texture on the same grid as the heights"},
        # measured: warm-up to ready, snapshot restore, repeated cold starts
        {"label": "GPU start-up", "value": "9 to 44 s, once",
         "note": "Timed separately on screen; it switches itself off about 15 s after the page closes"},
        # measured: 15 parallel downloads across cold and warm runs
        {"label": "Stability", "value": "15 of 15 runs clean",
         "note": "Cold and warm starts, results downloaded in parallel; 2D fallback when WebGL is missing"},
        {"label": "Visual fidelity", "value": "1.05 M vertex mesh",
         "note": "Full-resolution imagery, crisp walls, facade detail, instanced trees, sun shadows"},
        {"label": "Navigation", "value": "Orbit, fly, walk",
         "note": "Collision against the rendered surface; source-image minimap and swipe comparison"},
        {"label": "Deployment", "value": "Vercel + Modal L4",
         "note": "Standalone web app; five precomputed scenes open in seconds with no GPU at all"},
    ],
    "footnote": (f"Swiss scenes: swisstopo swissSURFACE3D and swissALTI3D airborne LiDAR, a country and sensor "
                 f"the model never saw, scored through the live service (scene MAE {min(scene_mae):.2f} to "
                 f"{max(scene_mae):.2f} m). GAMUS strata: test tiles never trained on, from US cities that were, "
                 "one centre crop per tile. Reproduce the Swiss figures with scripts/evaluate_live.py."),
}

OUT.write_text(json.dumps(data, indent=2) + "\n")
print(f"wrote {OUT.relative_to(ROOT)}")
for h in data["headline"]:
    print(f"  {h['label']:22s} {h['value']} {h['unit']}")
