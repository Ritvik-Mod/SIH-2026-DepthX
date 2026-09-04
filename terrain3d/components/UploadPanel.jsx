'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { classifyFiles, loadHeightmap, loadMetadata, loadTextureBitmap, resolvePixelSpacing, sanityWarnings } from '@/lib/load';
import { API_BASE, checkHealth, predictFromImage } from '@/lib/api';

/**
 * Two ways in, one viewer.
 *
 *   photo  (default)  one top-down image -> the inference service -> the same bundle
 *   files  (fallback) a prepared heightmap.tif + texture.png, no service involved
 *
 * The fallback is kept deliberately. It is the path that works with no backend
 * reachable at all -- on a plane, behind a blocked port, or when the demo machine is
 * the judge's laptop -- and it is how a previously exported bundle is re-opened.
 */
export default function UploadPanel({ onReady }) {
  const [mode, setMode] = useState('photo');
  return (
    <div className="uploadWrap">
      <div className="uploadCard">
        <h1>2D → 3D terrain viewer</h1>
        <div style={tabsStyle}>
          <button style={tabStyle(mode === 'photo')} onClick={() => setMode('photo')}>
            Upload a photo
          </button>
          <button style={tabStyle(mode === 'files')} onClick={() => setMode('files')}>
            Load prepared files
          </button>
        </div>
        {mode === 'photo' ? <PhotoMode onReady={onReady} /> : <FilesMode onReady={onReady} />}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ photo mode */

function PhotoMode({ onReady }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [gsd, setGsd] = useState('');
  const [autoDem, setAutoDem] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    checkHealth().then(setHealth).catch(() => setHealth({ ok: false }));
  }, []);

  const go = useCallback(async () => {
    if (!file) { setError('Pick an image first.'); return; }
    setError(''); setBusy(true);
    try {
      const out = await predictFromImage(file, {
        gsd: gsd ? parseFloat(gsd) : null,
        autoDem,
        // Ticking the box is a request to SEE the terrain, so render the DSM too.
        // Fetching a DEM and then still drawing AGL would look identical on screen
        // and the toggle would appear to do nothing.
        renderQuantity: autoDem ? 'dsm' : 'agl',
        onProgress: setStatus,
      });
      setStatus('Reading the height raster…');
      const hm = await loadHeightmap(out.heightmapFile, setStatus);
      const bitmap = await loadTextureBitmap(out.textureFile, setStatus);
      const spacing = hm.geoSpacing ?? resolvePixelSpacing(out.metadata);
      setStatus('Building mesh…');
      onReady({
        heights: hm.data, width: hm.width, height: hm.height,
        min: hm.min, max: hm.max, mean: hm.mean, nodataPixels: hm.nodataPixels,
        pixelSpacing: spacing, bitmap, metadata: out.metadata,
      });
    } catch (e) {
      console.error(e);
      setStatus('');
      setError(e?.message || 'Inference failed.');
    } finally {
      setBusy(false);
    }
  }, [file, gsd, autoDem, onReady]);

  return (
    <>
      <p className="lede">
        Drop one top-down image. It is sent to the height model, which returns the
        terrain — no metadata required. A GeoTIFF keeps its real-world scale; a PNG or
        JPG is rendered at an assumed {' '}
        <code>0.33 m/px</code>, so the shape is right and the metres are conditional.
      </p>

      <div
        className={`drop ${dragging ? 'dragging' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); setFile(e.dataTransfer.files?.[0] ?? null); }}
        onClick={() => inputRef.current?.click()}
      >
        <strong>{file ? file.name : 'Drop a satellite or aerial image'}</strong>
        <span>{file ? `${(file.size / 1e6).toFixed(1)} MB — click to change` : '.tif, .png or .jpg — click to browse'}</span>
        <input
          ref={inputRef} type="file" hidden accept=".tif,.tiff,.png,.jpg,.jpeg"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
      </div>

      <label className="field">
        <span>Pixel spacing (m/px) — optional</span>
        <input
          type="number" step="0.01" min="0.01" value={gsd} placeholder="read from the file, else 0.33"
          onChange={(e) => setGsd(e.target.value)}
        />
        <small>The model is scale-conditioned: this is what tells it a 40 px roof is a shed and not a warehouse.</small>
      </label>

      <label style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '10px 0 14px',
                      fontSize: 13, cursor: 'pointer' }}>
        <input type="checkbox" checked={autoDem} onChange={(e) => setAutoDem(e.target.checked)}
               style={{ width: 15, height: 15, flex: 'none', margin: 0 }} />
        <span>Sit it on real terrain (DSM) <span className="muted">— georeferenced input only; ground follows the true land surface instead of being flat</span></span>
      </label>

      <button className="wide primary big" onClick={go} disabled={busy || !file}>
        {busy ? 'Working…' : 'Generate 3D terrain'}
      </button>

      {status && <p className="status">{status}</p>}
      {error && <p className="error">{error}</p>}
      {health && !health.ok && (
        <p className="error">
          No inference service at <code>{API_BASE}</code>. Start it, or use “Load prepared files”.
        </p>
      )}
      {health?.ok && (
        <p className="muted">
          {health.model_warm
            ? <>Service up · model warm on <code>{health.device}</code></>
            : <>Service up · model loads on the first request (~5 s)</>}
        </p>
      )}
    </>
  );
}

/* ------------------------------------------------- prepared-files mode (unchanged) */

function FilesMode({ onReady }) {
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
    <>
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
    </>
  );
}

const tabsStyle = { display: 'flex', gap: 6, margin: '0 0 14px' };
const tabStyle = (active) => ({
  flex: 1, padding: '9px 12px', borderRadius: 8, cursor: 'pointer', fontSize: 13,
  fontWeight: active ? 600 : 500,
  border: `1px solid ${active ? 'rgba(120,170,255,.55)' : 'rgba(255,255,255,.14)'}`,
  background: active ? 'rgba(120,170,255,.16)' : 'transparent',
  color: active ? '#dce9ff' : 'inherit',
});
