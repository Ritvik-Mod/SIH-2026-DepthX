// Reads the upstream hand-off files entirely in the browser.
// heightmap.tif holds REAL METRES (AGL). Nothing here normalises or rescales.

const DEFAULT_PIXEL_SPACING_M = 0.33;

export async function loadMetadata(file) {
  if (!file) return null;
  try {
    return JSON.parse(await file.text());
  } catch {
    return null;
  }
}

/**
 * TWO EXPORT PATHS EXIST UPSTREAM AND THEY DISAGREE ON BAND COUNT.
 *
 *   scripts/export_for_threejs.py  ->  1 band  (agl_metres)
 *   heightmap/export.py            ->  3 bands (agl_metres, sigma_metres, agl_normalised)
 *                                      -- this is the contract written in SPEC.md
 *
 * Every bundle handed over so far has been single-band, so the old hard
 * rejection ("Expected a single-band height raster, got 3 bands") never fired.
 * It would fire the moment anyone exported through the documented path, and the
 * viewer would refuse a file that is perfectly valid. Band 1 is agl_metres in
 * BOTH, so read band 1 and carry on.
 *
 * Band 3 is deliberately NOT used: it is percentile-normalised to [0,1] and is
 * a fallback for consumers who want to calibrate from scratch. Loading it as
 * geometry would silently rescale the whole scene.
 */
function pickHeightBand(rasters, image) {
  if (rasters.length === 1) return { band: rasters[0], index: 0, note: 'single-band raster' };

  // Prefer a band literally described as the AGL band, if the writer set it.
  const descs = image.fileDirectory?.GDAL_METADATA || '';
  const named = /agl_metres/i.test(descs);
  return {
    band: rasters[0],
    index: 0,
    note: `${rasters.length}-band raster; read band 1${named ? ' (agl_metres)' : ''}, ignored the rest`,
  };
}

/**
 * Pixel spacing straight from the GeoTIFF when it is georeferenced.
 *
 * ModelPixelScale is [scaleX, scaleY, scaleZ] in CRS units. For a projected CRS
 * (UTM and friends) those units are metres and the value is usable directly. For
 * a geographic CRS the units are DEGREES, and treating 8.9e-6 degrees as 8.9e-6
 * metres would shrink a 338 m tile to under a millimetre -- so that case is
 * rejected rather than guessed at.
 */
function pixelSpacingFromGeoTIFF(image) {
  try {
    const scale = image.fileDirectory?.ModelPixelScale;
    if (!scale || !scale.length) return null;
    const sx = Math.abs(Number(scale[0]));
    if (!Number.isFinite(sx) || sx <= 0) return null;
    if (sx < 1e-3) return null;             // degrees, not metres -- do not guess
    if (sx > 1000) return null;             // implausible for a viewer tile
    return sx;
  } catch {
    return null;
  }
}

export async function loadHeightmap(file, onProgress) {
  onProgress?.('Decoding GeoTIFF…');
  // dynamic import keeps geotiff out of the SSR bundle
  const { fromBlob } = await import('geotiff');
  const tiff = await fromBlob(file);
  const image = await tiff.getImage();

  const width = image.getWidth();
  const height = image.getHeight();
  const samples = image.getSamplesPerPixel();

  onProgress?.(`Reading ${width}x${height} raster…`);
  const rasters = await image.readRasters({ interleave: false });
  const { band, note } = pickHeightBand(rasters, image);
  const data = band instanceof Float32Array ? band : Float32Array.from(band);

  // nodata handling: GDAL_NODATA tag, falling back to the -9999 convention.
  const tag = image.fileDirectory?.GDAL_NODATA;
  const nodata = tag != null ? parseFloat(tag) : -9999;
  const isNodata = (v) => !Number.isFinite(v) || v === nodata || v <= -9000;

  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (isNodata(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min)) throw new Error('Height raster contains no valid pixels.');

  // Nodata is filled with the scene minimum rather than 0: a tile whose ground
  // sits at 2 m would otherwise get 2 m-deep pits wherever data is missing, and
  // those read as real holes in the mesh.
  let sum = 0;
  let nodataPixels = 0;
  for (let i = 0; i < data.length; i++) {
    if (isNodata(data[i])) { data[i] = min; nodataPixels++; }
    sum += data[i];
  }

  const geoSpacing = pixelSpacingFromGeoTIFF(image);

  return {
    data,
    width,
    height,
    min,
    max,
    mean: sum / data.length,
    nodataPixels,
    bands: samples,
    bandNote: note,
    geoSpacing,
    georeferenced: !!geoSpacing,
    gdalMetadata: image.fileDirectory?.GDAL_METADATA ?? null,
  };
}

/**
 * Returns { source, flipY, width, height }.
 *
 * three.js IGNORES Texture.flipY for ImageBitmap sources -- the flip has to be
 * baked in at decode time instead, or the RGB ends up vertically mirrored onto
 * the geometry (roads textured onto roofs). So we decode with
 * imageOrientation: 'flipY' and then tell three not to flip again. If the
 * browser does not support the option we fall back to an HTMLImageElement,
 * where three's own flipY works reliably.
 */
export async function loadTextureBitmap(file, onProgress) {
  onProgress?.('Decoding RGB texture…');

  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: 'flipY' });
    return { source: bitmap, flipY: false, width: bitmap.width, height: bitmap.height };
  } catch {
    const url = URL.createObjectURL(file);
    const img = await new Promise((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error('Could not decode the RGB texture.'));
      el.src = url;
    });
    return { source: img, flipY: true, width: img.naturalWidth, height: img.naturalHeight };
  }
}

// Single drop zone -> figure out which file is which.
// heightmap_preview.png is deliberately ignored: it is 8-bit normalised and is
// NOT a data source (35.8 m across 256 levels would quantise roofs to 0.14 m steps).
export function classifyFiles(fileList) {
  const out = { heightmap: null, texture: null, metadata: null, ignored: [] };
  for (const f of fileList) {
    const name = f.name.toLowerCase();
    if (/\.(tif|tiff)$/.test(name)) out.heightmap = f;
    else if (/\.json$/.test(name)) {
      // manifest.json describes a whole batch and carries no pixel spacing for
      // this scene; taking it as the sidecar silently falls back to the default.
      if (name.includes('manifest')) out.ignored.push(f.name);
      else out.metadata = f;
    } else if (/\.(png|jpg|jpeg|webp)$/.test(name)) {
      if (name.includes('preview') || name.includes('heightmap')) out.ignored.push(f.name);
      else out.texture = f;
    } else out.ignored.push(f.name);
  }
  return out;
}

export function resolvePixelSpacing(metadata) {
  const v = Number(metadata?.pixel_spacing_m);
  return Number.isFinite(v) && v > 0 ? v : DEFAULT_PIXEL_SPACING_M;
}

/**
 * Warnings worth showing the user before they build a scene. None of these are
 * fatal; all of them have produced a confusing render at least once.
 */
export function sanityWarnings({ hm, bitmap, metadata, spacing }) {
  const w = [];
  if (hm.bands > 1) w.push(hm.bandNote);
  if (bitmap && (bitmap.width !== hm.width || bitmap.height !== hm.height)) {
    w.push(`Texture is ${bitmap.width}x${bitmap.height} but the raster is ${hm.width}x${hm.height}; UVs stretch to fit.`);
  }
  if (hm.nodataPixels > hm.data.length * 0.02) {
    w.push(`${((100 * hm.nodataPixels) / hm.data.length).toFixed(1)}% of pixels are nodata, filled with the scene minimum.`);
  }
  const q = metadata?.quantity;
  if (q && q !== 'AGL') {
    w.push(`metadata says quantity="${q}", not "AGL" — heights may be absolute elevation, not height above ground.`);
  }
  if (metadata?.field_source === 'ground_truth_lidar') {
    w.push('This bundle is GROUND TRUTH LiDAR, not a model prediction.');
  }
  if (metadata?.split === 'train') {
    w.push('This tile is from the TRAIN split — for pipeline development only, never for accuracy claims.');
  }
  if (hm.geoSpacing && Math.abs(hm.geoSpacing - spacing) > 0.01) {
    w.push(`GeoTIFF says ${hm.geoSpacing.toFixed(3)} m/px but ${spacing} m/px is selected.`);
  }
  if (hm.max > 400) w.push(`Max height ${hm.max.toFixed(0)} m is unusually tall — check the units.`);
  return w;
}

/**
 * Is this upload georeferenced? Decided in the browser, before anything is sent.
 *
 * The service already skips the DTM stage for an image with no georeferencing,
 * so this is not a safety check -- it is so the page can SAY which of the two
 * will happen before the user waits a minute to find out. Only the GeoTIFF
 * header is read: fromBlob slices lazily, so a 40 MB file costs a few kB here.
 *
 * Georeferenced means a raster-to-model transform (tiepoint or matrix) AND a
 * coordinate system. A TIFF with neither is just a picture that happens to be
 * a TIFF, and PNG / JPEG cannot carry either.
 */
export async function probeGeoreference(file) {
  if (!file) return null;
  if (!/\.(tif|tiff)$/i.test(file.name)) {
    return { georeferenced: false, reason: 'PNG and JPEG carry no georeferencing' };
  }
  try {
    const { fromBlob } = await import('geotiff');
    const image = await (await fromBlob(file)).getImage();
    const fd = image.fileDirectory || {};
    const keys = image.geoKeys || {};
    const hasTransform = !!(fd.ModelTiepoint || fd.ModelTransformation);
    const epsg = keys.ProjectedCSTypeGeoKey || keys.GeographicTypeGeoKey || null;
    const hasCrs = !!epsg || Object.keys(keys).length > 0;
    const scale = fd.ModelPixelScale ? Number(fd.ModelPixelScale[0]) : null;
    return {
      georeferenced: hasTransform && hasCrs,
      crs: epsg && epsg !== 32767 ? `EPSG:${epsg}` : null,
      // metres only for a projected CRS; degrees are not a pixel spacing
      gsd: keys.ProjectedCSTypeGeoKey && scale > 1e-3 ? scale : null,
      width: image.getWidth(),
      height: image.getHeight(),
      reason: hasTransform && hasCrs ? null : 'no coordinate reference in this TIFF',
    };
  } catch {
    return { georeferenced: false, reason: 'could not read the TIFF header' };
  }
}

/** Fetch with byte-level progress, so a 5 MB scene shows it is moving. */
async function fetchWithProgress(url, onBytes) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url.split('/').pop()}: HTTP ${r.status}`);
  if (!r.body || !onBytes) return r.blob();
  const reader = r.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.byteLength;
    onBytes(got);
  }
  return new Blob(chunks);
}

/**
 * Open a precomputed sample scene: the same three files the service returns,
 * served statically, run through exactly the loaders an upload goes through.
 * Nothing about the viewer knows or cares that no model was called.
 */
export async function loadSample(sample, onProgress) {
  const base = sample.base;
  const total = sample.bytes || 0;
  const got = { hm: 0, tex: 0 };
  const report = () => {
    const mb = (got.hm + got.tex) / 1e6;
    onProgress?.(total ? `Downloading ${mb.toFixed(1)} / ${(total / 1e6).toFixed(1)} MB` : 'Downloading…');
  };
  report();

  const [hmBlob, texBlob, metaRes] = await Promise.all([
    fetchWithProgress(`${base}/heightmap.tif`, (n) => { got.hm = n; report(); }),
    fetchWithProgress(`${base}/texture.jpg`, (n) => { got.tex = n; report(); }),
    fetch(`${base}/metadata.json`),
  ]);
  const metadata = metaRes.ok ? await metaRes.json() : null;

  const hm = await loadHeightmap(new File([hmBlob], 'heightmap.tif', { type: 'image/tiff' }), onProgress);
  const bitmap = await loadTextureBitmap(new File([texBlob], 'texture.jpg', { type: 'image/jpeg' }), onProgress);
  onProgress?.('Building mesh…');

  return {
    heights: hm.data,
    width: hm.width,
    height: hm.height,
    min: hm.min,
    max: hm.max,
    mean: hm.mean,
    nodataPixels: hm.nodataPixels,
    pixelSpacing: hm.geoSpacing ?? resolvePixelSpacing(metadata),
    bitmap,
    metadata,
    name: sample.name,
    sampleId: sample.id,
  };
}
