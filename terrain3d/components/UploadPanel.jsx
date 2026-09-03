'use client';

import { useCallback, useRef, useState } from 'react';
import { classifyFiles, loadHeightmap, loadMetadata, loadTextureBitmap, resolvePixelSpacing, sanityWarnings } from '@/lib/load';

export default function UploadPanel({ onReady }) {
  const inputRef = useRef(null);
  const [picked, setPicked] = useState({ heightmap: null, texture: null, metadata: null, ignored: [] });
  const [spacing, setSpacing] = useState(0.33);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [warnings, setWarnings] = useState([]);

  const accept = useCallback(async (fileList) => {
    setError('');
    const files = Array.from(fileList);
    const next = classifyFiles(files);
    setPicked((prev) => ({
      heightmap: next.heightmap ?? prev.heightmap,
      texture: next.texture ?? prev.texture,
      metadata: next.metadata ?? prev.metadata,
      ignored: next.ignored,
    }));
    if (next.metadata) {
      const md = await loadMetadata(next.metadata);
      setSpacing(resolvePixelSpacing(md));
    }
  }, []);

  const build = useCallback(async () => {
    if (!picked.heightmap || !picked.texture) {
      setError('Both a .tif height raster and an RGB .png texture are required.');
      return;
    }
    try {
      const metadata = await loadMetadata(picked.metadata);
      const hm = await loadHeightmap(picked.heightmap, setStatus);
      const bitmap = await loadTextureBitmap(picked.texture, setStatus);

      // Prefer the GeoTIFF's own pixel scale when it carries one: it is the
      // file's ground truth, whereas the sidecar and the manual field are both
      // things a human can get wrong.
      const effSpacing = hm.geoSpacing ?? spacing;
      if (hm.geoSpacing && Math.abs(hm.geoSpacing - spacing) > 0.001) {
        setSpacing(hm.geoSpacing);
      }

      setWarnings(sanityWarnings({ hm, bitmap, metadata, spacing: effSpacing }));
      setStatus('Building mesh…');

      onReady({
        heights: hm.data,
        width: hm.width,
        height: hm.height,
        min: hm.min,
        max: hm.max,
        mean: hm.mean,
        nodataPixels: hm.nodataPixels,
        pixelSpacing: spacing,
        bitmap,
        metadata,
      });
    } catch (e) {
      console.error(e);
      setStatus('');
      setError(e?.message || 'Failed to read the input files.');
    }
  }, [picked, spacing, onReady]);

  const row = (label, file, note) => (
    <div className={`slot ${file ? 'ok' : ''}`}>
      <div className="slotDot" />
      <div>
        <b>{label}</b>
        <small>{file ? file.name : note}</small>
      </div>
    </div>
  );

  return (
    <div className="uploadWrap">
      <div className="uploadCard">
        <h1>2D → 3D terrain viewer</h1>
        <p className="lede">
          Drop the reconstruction hand-off here: the float32 <code>heightmap.tif</code> (metres AGL),
          the matching RGB <code>texture.png</code>, and <code>metadata.json</code>.
          The heights are read as real metres — nothing is normalised.
        </p>

        <div
          className={`drop ${dragging ? 'dragging' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); accept(e.dataTransfer.files); }}
          onClick={() => inputRef.current?.click()}
        >
          <strong>Drop all files here</strong>
          <span>or click to browse — you can select them together</span>
          <input
            ref={inputRef} type="file" multiple hidden
            accept=".tif,.tiff,.png,.jpg,.jpeg,.json"
            onChange={(e) => accept(e.target.files)}
          />
        </div>

        <div className="slots">
          {row('Height raster (.tif)', picked.heightmap, 'required · float32, single band, metres')}
          {row('RGB texture (.png)', picked.texture, 'required · same grid as the raster')}
          {row('metadata.json', picked.metadata, 'optional · supplies pixel spacing')}
        </div>

        {picked.ignored.length > 0 && (
          <p className="muted">
            Ignored: {picked.ignored.join(', ')} — <code>heightmap_preview.png</code> is 8-bit
            normalised and is never used as a data source.
          </p>
        )}

        <label className="field">
          <span>Pixel spacing (m/px)</span>
          <input
            type="number" step="0.01" min="0.01" value={spacing}
            onChange={(e) => setSpacing(parseFloat(e.target.value) || 0.33)}
          />
          <small>Read from metadata.json when present. Sets the real-world ground extent.</small>
        </label>

        <button className="wide primary big" onClick={build}>Build 3D scene</button>

        {warnings.length > 0 && (
          <ul className="muted" style={{ margin: '10px 0 0', paddingLeft: 18, lineHeight: 1.55 }}>
            {warnings.map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        )}

        {status && <p className="status">{status}</p>}
        {error && <p className="error">{error}</p>}
      </div>
    </div>
  );
}
