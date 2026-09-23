# Making the reconstruction more realistic with real-world priors

Written 22 Sept 2026. Companion to `MASTER_PLAN.md`; read §2 gap 5 ("predicted
heightmaps are soft") and §3 (dataset roster) first, because this note is
largely about closing that gap with data rather than with loss functions.

Every claim here is either measured in this repo or cited. Where something is
not verified it says **UNVERIFIED** and names what would settle it.

---

## 0 · What `gods-eye-view` actually is, and why that matters

The repository referenced as the starting point —
[bilawalsidhu/gods-eye-view](https://github.com/bilawalsidhu/gods-eye-view) —
**is not a depth model and contains no reconstruction of any kind.** It is a
vanilla-JS + Vite front end over **CesiumJS**, rendering **Google Photorealistic
3D Tiles**, with live feeds (OpenSky, CelesTrak, USGS, AISStream, NASA FIRMS)
and an OpenAI Realtime voice layer on top.

That is worth stating plainly because it changes what can be borrowed:

- Its realism is **fetched, not inferred.** The buildings, the trees and the
  terrain are Google's photogrammetric mesh, reconstructed from aerial capture
  Google flew itself. There is no monocular anything.
- It needs a **place**, not an image. You cannot hand it a photo.
- It needs a **billing-enabled Google Maps key**. Photorealistic 3D Tiles is a
  metered Maps Platform SKU: roughly 1,000 free calls a month, then about
  $6 CPM. See
  [Map Tiles API usage and billing](https://developers.google.com/maps/documentation/tile/usage-and-billing).
- Coverage is the ~2,500 cities Google has flown. Rural India is not in it.

So it cannot replace DepthWizard, and nothing in it makes a single-image
prediction sharper. **But its architecture is exactly the right idea for a
second path**, and that is the recommendation below.

---

## 1 · Two paths, and they must coexist

| | **Path A — inference** (what exists) | **Path B — recognised site** (proposed) |
|---|---|---|
| Input | any nadir image | an image whose location is known |
| Works offline | yes | no |
| Coverage | global, unconditional | wherever the prior dataset reaches |
| Role | **the deliverable** | enrichment, and a free accuracy check |

Path A stays the default and stays untouched. Path B is a toggle.

**The trap, and it is a serious one.** SIH26175 scores 50% on *DSM accuracy vs
LiDAR across landscapes*. If the renderer silently snaps building heights to
Overture or Open Buildings values, the reported accuracy is no longer measuring
the model — it is measuring somebody else's survey. Path B must therefore be:

1. off by default,
2. impossible to enable without it being visible in the UI,
3. excluded from every number quoted as model accuracy, and
4. reported per building: which footprints were snapped to a fetched polygon
   and which are pure inference.

Without (4) the accuracy figure becomes uninterpretable, and for a scoring
rubric that distinction is the whole game.

---

## 2 · Getting a location

Path B needs latitude and longitude. There are two cases and they are very
different.

### 2a · Georeferenced input — already solved, small change

`lib/load.js:pixelSpacingFromGeoTIFF` already reads `ModelPixelScale`, and
deliberately **rejects** a geographic CRS because degrees are not metres. The
tags needed for a position — `ModelTiepoint`, `GeoKeyDirectory` — are in the
same `image.fileDirectory` and are currently unread. The `geotiff` library
exposes `image.getBoundingBox()` and `image.geoKeys`.

Three of the five files in `final_test_files/` are georeferenced
(`02_`, `04_`, `05_`), so this covers the majority of the demo set. Reprojecting
UTM → WGS84 needs a small `proj4` dependency, or can be done for UTM only in
about thirty lines.

**This is the cheapest useful work in this document.**

### 2b · Non-georeferenced input — do not bet on a model

`06_nongeo_india_dense_urban.jpeg` and `07_nongeo_nyc_rowhouses.png` carry no
position. The academic answer is **cross-view geo-localization**, and the field
is active — see
[GDAOSU/Awesome-Cross-View-Methods](https://github.com/GDAOSU/Awesome-Cross-View-Methods)
for the current survey, and
[CV-Cities (2025)](https://doi.org/10.3390/s25010044) for the OSM-fused
approach.

**Honest verdict: none of it is reliable enough to build a product path on.**
These methods retrieve from a *gallery* — a pre-built index of candidate
satellite tiles for a known city — and their top-1 accuracy falls off a cliff
when the gallery is the whole planet. A wrong match is worse than no match,
because it fabricates confident, plausible, wrong buildings.

**Recommendation: ask the user.** A small map picker beside the upload box,
defaulted from the GeoTIFF when there is one. One click from a human beats a
model that is right 60% of the time, costs nothing to build, and never lies.

---

## 3 · The data sources, ranked by value per unit of effort

### Tier 1 — building footprints and heights

This is the single biggest realism win available, and it attacks
`MASTER_PLAN.md` gap 5 directly. The current `sharpenHeights()` is a
morphological *guess* at where a building edge is. A footprint polygon does not
guess — it knows.

**1. [Overture Maps — buildings](https://docs.overturemaps.org/guides/buildings/)**
— best single source. ~2.3 billion buildings with `height` and `num_floors`
attributes, merged from OSM, Microsoft, Esri and Google. CC-BY 4.0 / ODbL.
GeoParquet on S3 and Azure, queryable **by bounding box** with DuckDB, the
`overturemaps` Python client, or a
[REST API](https://www.overturemapsapi.com/). A bbox query for one tile
returns in seconds, which makes it usable live in `serve/app.py`.

**2. [Google Open Buildings 2.5D Temporal](https://sites.research.google/gr/open-buildings/temporal/)**
— the one that matters for this problem statement. Building presence, counts
**and heights**, annual 2016–2023, ~4 m effective resolution, covering Africa,
South Asia, South-East Asia and Latin America — **which includes India**, the
exact gap `MASTER_PLAN.md` §2 lists as open. CC-BY 4.0 / ODbL, via
[Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1)
or direct GCS download. Note it is derived from **Sentinel-2 at 10 m**, so the
heights are coarse and quantised — good enough to anchor a storey count, not to
score a roofline against.

**3. Microsoft Global ML Building Footprints** — ~1.4 billion footprints, ODbL,
no heights at all. Geometry only, but the geometry is very clean.

**4. OSM / Cesium OSM Buildings** — `building:levels` where a mapper has filled
it in. Patchy, but exact where present, and free.

**How to use them — and how not to.** Do *not* overwrite the predicted height
field. Use the polygons as **snap targets** in a post-process:

- the polygon edge gives a true vertical wall, replacing the ~2 m model ramp;
- the polygon interior gives a single flat roof plane, killing the dome;
- a known `height` or `num_floors` anchors metric scale on that footprint;
- everything outside a matched polygon stays pure inference, untouched.

Architecturally this belongs in the same display-only style as the existing
canopy layer and the disaster layer: a separate delta array plus a mask,
subtracted by the shader and by the collision sampler, never written back into
`heights`. The mask is also what makes the per-building provenance report in
§1(4) fall out for free.

### Tier 2 — the sides of buildings

This is what the question was really about: the model sees only the roof, so
every facade is invented. It currently *is* invented — `createTerrainMaterial`
draws a procedural storey/bay/window grid, seeded per building. It reads well,
but it is decoration.

**5. [Mapillary](https://www.mapillary.com/developer)** — the open option. Over
2.4 billion crowd-sourced street-level images, free API, CC-BY-SA, and real
coverage in Indian cities. The pipeline: for each footprint edge, query images
whose camera position and compass bearing face that edge, fetch, rectify against
the edge, and use the result as a facade texture. Coverage is uneven and the
rectification is the hard part, so treat this as a stretch goal — but it is the
only *open* route to a real facade.

**6. Google Street View Static API** — far better coverage, metered, needs
billing. Same pipeline, fewer gaps.

**7. [Google Photorealistic 3D Tiles](https://mapsplatform.google.com/maps-products/map-tiles/)**
— what `gods-eye-view` uses. Real glTF meshes of real buildings.

The most valuable use of this is **not** as a render source. It is as a
**validation layer**: where Google has flown, load their mesh beside ours and
show them side by side. That is simultaneously the strongest possible demo
("here is ours, from one image, next to Google's photogrammetry") and a free
independent accuracy check on tiles where no LiDAR exists. Behind a toggle,
clearly labelled, ~1,000 free calls a month.

### Tier 3 — cars, trees and detail

**8. Cars: detect them, do not fetch them.** At 0.33 m/px a car is roughly
13 × 5 px — small but very learnable, and there is no global vehicle dataset to
fetch anyway. The existing auxiliary semantic head already segments the scene;
a car class plus instanced low-poly meshes, placed the same way
`createTreeLayer` places trees, needs no external data and no network.

**9. Trees: replace the greenness heuristic with crown detection.**
`planTreeClumps` currently finds *mounds that are green and raised* and fills
each with a spaced group — which is a good heuristic and is honest about being
one. [DeepForest](https://deepforest.readthedocs.io/) is an open, pretrained
individual-tree-crown detector for RGB aerial imagery; it returns per-tree
bounding boxes, so the layer would go from "one green mound, guess six trees" to
"eleven trees at these coordinates with these crown diameters". Height still
comes from the DSM. This is a contained, high-visibility upgrade.

**10. Roof form.** Pitched versus flat changes a silhouette more than any
texture does. DFC2023 **Track 1** carries roof-type labels and is already noted
in `MASTER_PLAN.md` §3 as "a different taxonomy, not our task" — which is true
for the height model, but it is exactly the label the *renderer* wants.

---

## 4 · Suggested order of work

1. **Read lat/lon from the GeoTIFF** (§2a). Half a day. Unlocks everything else.
2. **Map picker for non-georeferenced input** (§2b). Half a day. Honest, and it
   sidesteps a research problem entirely.
3. **Overture bbox fetch + footprint snapping** (§3.1). The big one. Do it as a
   display-only delta layer, with a provenance mask, off by default.
4. **Google 3D Tiles side-by-side validation view** (§3.7). Cheap, and it is the
   demo that wins arguments.
5. **DeepForest tree crowns** (§3.9) and **car detection** (§3.8). Independent
   of everything above; neither needs a location.
6. **Mapillary facades** (§3.5). Only once 1–3 exist, because it needs
   footprints to rectify against.

Items 5 and 6 are the only ones that improve a *non*-georeferenced scene, which
is worth remembering: for `06_nongeo_india_dense_urban.jpeg`, better tree and
car estimation is the entire available upside.

---

## 5 · What none of this fixes

Nothing above makes the **model** better. Every item is a renderer or a
post-process. The softness in the predicted height field is still there
underneath; it is being covered up by better geometry from elsewhere, and on
any tile the prior datasets do not cover, the output is exactly what it is
today. `MASTER_PLAN.md` §6 remains the place where that problem actually gets
solved.
