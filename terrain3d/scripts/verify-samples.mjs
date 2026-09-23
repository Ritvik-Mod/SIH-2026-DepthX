/**
 * Decodes every packaged sample with geotiff.js -- the same reader the viewer
 * uses -- and compares it value-for-value with the array package_samples.py
 * wrote. A hand-written TIFF encoder is exactly the kind of thing that works on
 * the first scene and silently corrupts the fourth, so nothing ships unchecked.
 *
 *   node terrain3d/scripts/verify-samples.mjs <inference_dir>
 */
import fs from 'node:fs';
import path from 'node:path';
import { fromFile } from 'geotiff';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const samples = JSON.parse(fs.readFileSync(path.join(here, '../lib/samples.json'), 'utf8'));
const inference = process.argv[2];

function readNpy(file) {
  const b = fs.readFileSync(file);
  const hlen = b.readUInt16LE(8);
  const header = b.subarray(10, 10 + hlen).toString('latin1');
  if (!/'descr': '<f4'/.test(header)) throw new Error(`unexpected dtype in ${file}: ${header}`);
  const data = b.subarray(10 + hlen);
  return new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
}

let failed = 0;
for (const s of samples) {
  const dir = path.join(here, '../public', s.base);
  const img = await (await fromFile(path.join(dir, 'heightmap.tif'))).getImage();
  const [band] = await img.readRasters({ interleave: false });
  const fd = img.fileDirectory;
  const want = readNpy(path.join(inference, s.id, 'packaged_heights.npy'));

  let diffs = 0;
  let maxErr = 0;
  for (let i = 0; i < want.length; i++) {
    const e = Math.abs(band[i] - want[i]);
    if (e > 0) diffs++;
    if (e > maxErr) maxErr = e;
  }
  const meta = JSON.parse(fs.readFileSync(path.join(dir, 'metadata.json'), 'utf8'));
  const checks = {
    size: img.getWidth() === s.width && img.getHeight() === s.height,
    values: band.length === want.length && diffs === 0,
    pixelScale: Math.abs(fd.ModelPixelScale[0] - s.gsd) < 1e-3,
    nodataTag: String(fd.GDAL_NODATA).startsWith('-9999'),
    predictor: fd.Predictor === 3 && fd.Compression === 8,
    metaSpacing: Math.abs(meta.pixel_spacing_m - fd.ModelPixelScale[0]) < 1e-9,
    texture: fs.existsSync(path.join(dir, 'texture.jpg')) && fs.existsSync(path.join(dir, 'thumb.jpg')),
  };
  const ok = Object.values(checks).every(Boolean);
  if (!ok) failed++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${s.id.padEnd(20)} ${img.getWidth()}x${img.getHeight()}`
    + `  ${diffs} differing / ${want.length} values (max err ${maxErr})`
    + `  gsd ${fd.ModelPixelScale[0].toFixed(4)}`
    + (ok ? '' : `  ${JSON.stringify(checks)}`));
}
console.log(failed ? `\n${failed} FAILED` : '\nall samples decode bit-exactly');
process.exit(failed ? 1 : 0);
