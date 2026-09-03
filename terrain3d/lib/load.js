// Reads the upstream hand-off files entirely in the browser.
// heightmap.tif is a tiled, deflate-compressed, single-band float32 TIFF whose
// values are REAL METRES (AGL). Nothing here normalises or rescales them.

const DEFAULT_PIXEL_SPACING_M = 0.33;

export async function loadMetadata(file) {
  if (!file) return null;
  try {
    return JSON.parse(await file.text());
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
  if (samples !== 1) {
    throw new Error(`Expected a single-band height raster, got ${samples} bands.`);
  }

  onProgress?.(`Reading ${width}x${height} raster…`);
  const [raster] = await image.readRasters({ interleave: false });
  const data = raster instanceof Float32Array ? raster : Float32Array.from(raster);

  // nodata handling: GDAL_NODATA tag, falling back to the -9999 convention.
  const tag = image.fileDirectory?.GDAL_NODATA;
  const nodata = tag != null ? parseFloat(tag) : -9999;

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let nodataPixels = 0;

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!Number.isFinite(v) || v === nodata || v <= -9000) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min)) throw new Error('Height raster contains no valid pixels.');

  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (!Number.isFinite(v) || v === nodata || v <= -9000) {
      data[i] = min;
      nodataPixels++;
    }
    sum += data[i];
  }

  return {
    data,
    width,
    height,
    min,
    max,
    mean: sum / data.length,
    nodataPixels,
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
    else if (/\.json$/.test(name)) out.metadata = f;
    else if (/\.(png|jpg|jpeg|webp)$/.test(name)) {
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