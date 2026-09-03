# Output contract — height estimation stage

**Owner:** Ritvik · **Consumers:** Divyanshu (calibration → DSM), Vineeth (TDA), Aakarsh (3D)
**Status:** frozen. Changes require a message in the group before they land.

Run `python scripts/make_contract_sample.py` to get a valid sample file to build
against right now — you do not need to wait for the model.

---

## Files

For an input scene `foo`, the stage emits three files:

| File | Purpose |
|---|---|
| `foo.tif` | **the deliverable** — float32, 3 bands, georeferenced when the input was |
| `foo.json` | sidecar: units, datum, provenance, value range, limitations |
| `foo_preview.png` | colour-mapped image for humans and slides. **Not** an interchange format |

## Raster bands

| Band | Name | Units | Notes |
|---|---|---|---|
| 1 | `agl_metres` | metres | height above **local ground** |
| 2 | `sigma_metres` | metres | per-pixel uncertainty (1σ); `-9999` if unavailable |
| 3 | `agl_normalised` | — | band 1 robust-percentile scaled to [0,1] |

- dtype **float32** throughout, DEFLATE-compressed, internally tiled 256×256
- nodata = **`-9999.0`**, set in the raster metadata. Mask it before any statistics
- band 3 exists only as a fallback for consumers who prefer to calibrate from scratch

## The one thing that must not be got wrong

> Band 1 is **AGL — height above local ground**, not elevation above sea level.
>
> To obtain a DSM: **`DSM = DTM + agl_metres`**. **Add** the terrain model. Do not subtract.

Subtracting produces a scene that looks plausible and is completely wrong, and no
automated check will catch it. The sidecar states this in `quantity` and `quantity_note`.

## Sidecar schema (`schema_version: "1.0"`)

```json
{
  "schema_version": "1.0",
  "produced_by": {"model": "...", "checkpoint": "...", "git_commit": "..."},
  "quantity": "AGL",
  "quantity_note": "height above LOCAL GROUND. To obtain a DSM, ADD a terrain model (DTM). Do not subtract.",
  "sign_convention": "larger value = taller",
  "units": "metres",
  "dtype": "float32",
  "nodata": -9999.0,
  "bands": {"1": "agl_metres", "2": "sigma_metres", "3": "agl_normalised"},
  "crs": "EPSG:32643",
  "transform": [0.5, 0.0, 512000.0, 0.0, -0.5, 1420000.0],
  "gsd_m": 0.5,
  "value_range": {"min": 0.0, "max": 61.3, "p1": 0.0, "p50": 3.9, "p99": 34.7},
  "normalisation": {"method": "robust_percentile", "p_low": 1, "p_high": 99, "lo": 0.0, "hi": 34.7},
  "sun": {"elevation_deg": 48.2, "azimuth_deg": 152.7, "source": "metadata|estimated"},
  "inference": {"tile": 518, "overlap": 0.25, "tta": true, "levelled": true, "n_tiles": 42},
  "assumptions": ["..."],
  "known_limitations": ["..."]
}
```

## Notes per consumer

**Divyanshu — calibration**
- Band 1 is already in metres, so your step is *residual correction + terrain*, not
  scale invention. That is strictly more reliable than fitting scale from scratch.
- Weight your fit by band 2 — low-confidence pixels should not drive it.
- **SRTM is not bare earth.** It was measured by radar in 2000 and sits near rooftop
  level over cities. Subtracting it as a DTM partly cancels buildings. Prefer
  Copernicus GLO-30 or CartoDEM, and ground-filter if you can.
- **Check vertical datums match.** Mixing ellipsoidal (GPS-native) with geoid-referenced
  (EGM96/EGM2008) sources adds a constant offset of tens of metres across India. It
  cancels in my AGL; it does not cancel in your DSM.

**Vineeth — TDA**
- Read band 1 as **float32**. Do not quantise to 8-bit: ties and plateaus become
  degenerate critical points and the persistence computation becomes ambiguous and slow.
- The field is **not pre-smoothed** — smoothing is your step, and doing it upstream
  would destroy the low-persistence features you are measuring.
- Mask `-9999` before building any filtration; NaN/sentinel values break gudhi/ripser.
- Sign convention: **larger = taller**, so sublevel filtration starts at the ground.
- Band 2 distinguishes "spiky because the surface is genuinely spiky" from "spiky
  because the model is unsure" — worth using before deciding what is noise.

**Aakarsh — 3D**
- Band 3 (`[0,1]`) is the convenient one for displacement mapping; band 1 if you want
  true metres in the scene.
- `gsd_m` converts pixels to metres for the horizontal axes — use it so the terrain is
  not vertically exaggerated by accident.
- `transform` + `crs` are present when the input was georeferenced; absent otherwise.

## Class ids (GAMUS convention, used throughout)

`0` unlabelled · `1` ground · `2` low-vegetation · **`3` building** · `4` water ·
`5` road · `6` tree

Building-only metrics use id `3`. Training ignores id `0`.

## Guarantees

- Output raster is the **same H×W as the input image**, whatever its size.
- Large scenes are tiled with overlap and levelled against a coarse whole-scene pass, so
  there are no seam steps at tile boundaries.
- Values are finite and ≥ 0 except at explicit `-9999` nodata.
