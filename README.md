# 2D → 3D Terrain Viewer (Next.js + Three.js)

Step 4 of the reconstruction pipeline: takes the upstream hand-off
(`heightmap.tif` + `texture.png` + `metadata.json`), builds a textured 3D
terrain mesh, and lets you orbit or fly through it.

Everything runs client-side in the browser. There is no server, no
preprocessing step, and no API. You drop the same four files you were handed
onto the page and the scene builds.

## Run it

```bash
npm install
npm run dev
```

Open <http://localhost:3000>, drag all four hand-off files onto the drop zone
(`heightmap_preview.png` is detected and ignored), then hit **Build 3D scene**.

Production: `npm run build && npm start`.

Requires Node 18+. Tested on Node 22 with Next 14.2.

## Controls

| | |
|---|---|
| **Orbit** (default) | drag to rotate, scroll to zoom, right-drag to pan |
| **Fly** | click the canvas to capture the mouse, `WASD` to move, `Space`/`Ctrl` for altitude, `Shift` to boost, `Esc` to release |
| Vertical exaggeration | `0` → `2.5x`. **1.00x is true metric scale.** |
| Replay 2D → 3D | re-runs the reveal animation that raises the flat image into terrain |
| Mesh resolution | 256 / 512 / 1024 segments |
| Surface | RGB imagery or a height colour ramp; wireframe toggle |
| Sun azimuth / elevation | moves the shadows live |

The HUD reports camera altitude, the surface height directly below, and the
difference (true AGL clearance) — which is meaningful because the raster *is*
an AGL product.

## How the data is handled

**Heights are read as real metres and never rescaled.** `lib/load.js` uses
[geotiff.js](https://geotiffjs.github.io/) to decode the tiled, deflate-compressed,
single-band float32 raster into a `Float32Array` directly in the browser. There
is no normalise/denormalise round trip anywhere in the codebase.

`heightmap_preview.png` is deliberately never read as data. It is 8-bit
normalised: 35.8 m across 256 levels quantises to 0.14 m steps, which would
flatten roof pitch and porch detail into staircases.

**Nodata** is taken from the `GDAL_NODATA` tag (falling back to the `-9999`
convention) and patched to the scene floor so the mesh stays watertight. Your
current file has zero nodata pixels; later crops may not.

**No DTM, so the ground plane is y = 0.** `metadata.json` states
`DSM = DTM + heightmap`, but with `georeferenced: false` there is no terrain
model to add. Rendering AGL directly on flat ground is both correct for this
input and better looking — buildings sit on a level street grid instead of on a
regional slope. When a georeferenced scene arrives, add the DTM inside
`buildTerrainGeometry` and subtract a datum offset to keep vertices near the
origin (pushing them to absolute elevation wrecks depth precision).

**Ground extent** comes from `resolution x pixel_spacing_m` read out of
`metadata.json` at runtime — nothing is hardcoded to 1024 or 0.33. World units
are metres, 1:1. Your scene is 337.92 m square with 35.79 m of relief.

## Implementation notes

**Exaggeration is `mesh.scale.y`, not a rebuild.** The mesh is baked once at
true scale; the slider only changes a non-uniform scale. Three's `normalMatrix`
is the inverse transpose, so lighting stays correct automatically and the
slider (and the reveal animation) costs nothing. This is why the reveal can run
at 60fps on a 263k-vertex mesh.

**Mesh density is decoupled from raster density.** The full 1024² float array
stays in memory and vertex heights are sampled from it bilinearly, so 512
segments still reflects the whole raster rather than throwing away every other
row.

**Slope shading.** Gradients here reach 3.2 m per 0.33 m pixel (~84°), so the
single continuous mesh has near-vertical quads standing in for building walls,
with roof texture smeared down them — inherent to 2.5D displacement. A
`wallness` term injected into `MeshStandardMaterial` via `onBeforeCompile`
darkens and desaturates those faces so they read as walls. The vertex shader
corrects the normal for the current exaggeration analytically
(`n.y / uExag`), so it stays accurate as you move the slider.

**Alignment.** Raster row 0 → the first `PlaneGeometry` row → local `+Y` →
after `rotateX(-π/2)` it lands at `-Z`, and with three's default `flipY` it maps
to texture `v = 1` = image row 0. Verified against the real file: vertex 0 is at
`(-169, 13.79, -169)` with UV `(0, 1)`, and `raster[0,0]` is 13.79 m. If you
ever swap the loader, re-check by loading the preview PNG as the texture — every
white blob must sit on a hill.

**Lighting.** The satellite texture already contains baked illumination, so
ambient is generous (hemisphere at 1.15) and the directional light exists mostly
to cast shadows. Shadows are the strongest "this is real geometry" cue in the
whole demo — the sun azimuth slider is worth more than any UI polish.

## Layout

```
app/
  layout.jsx        shell + metadata
  page.jsx          upload ⇄ viewer switch
  globals.css       all styling
components/
  UploadPanel.jsx   drop zone, file classification, pixel-spacing override
  TerrainViewer.jsx scene, camera, lights, shadows, render loop, reveal
  ControlPanel.jsx  sliders and toggles
  Hud.jsx           FPS / altitude / AGL readout
lib/
  load.js           geotiff decode, nodata, metadata, file classification
  terrain.js        bilinear sampler, geometry builder, slope-shaded material
  flyController.js  pointer-lock WASD camera
```

## Performance

512 segments (263k verts) is the default and holds 60fps with shadows on
integrated graphics. 1024 (1.05M verts, 2.1M tris) is fine on a discrete GPU but
the shadow pass doubles the cost — switch to it for the "and here it is at full
raster resolution" moment, not as the default.

## Tuning for the demo

Resist exaggeration above ~1.5x. At 1:1 the scene already has a 1:9.4 relief
ratio and reads dramatically in perspective; pushed further, the 30–36 m tree
canopy turns into spikes and the reconstruction stops looking credible. Frame
your hero pass low over the rooftops along the street grid, where the geometry
is sharp — the woods on the left are blobby domes, which is what the upstream
depth model produced, not something the renderer can fix.
