'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  classifyFiles, loadHeightmap, loadMetadata, loadTextureBitmap,
  probeGeoreference, resolvePixelSpacing, sanityWarnings,
} from '@/lib/load';
import { checkHealth, getApiBase, predictFromImage, setApiBase } from '@/lib/api';
import Icon from './Icons';

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
    <section className="card upload" aria-labelledby="upload-title">
      <div className="cardHead">
        <h2 id="upload-title">Reconstruct an image</h2>
        <div className="tabsLight" role="tablist" aria-label="Input type">
          <button type="button" role="tab" aria-selected={mode === 'photo'}
                  className={mode === 'photo' ? 'on' : ''} onClick={() => setMode('photo')}>
            Image
          </button>
          <button type="button" role="tab" aria-selected={mode === 'files'}
                  className={mode === 'files' ? 'on' : ''} onClick={() => setMode('files')}>
            Prepared bundle
          </button>
        </div>
      </div>
      {mode === 'photo' ? <PhotoMode onReady={onReady} /> : <FilesMode onReady={onReady} />}
    </section>
  );
}

const baseName = (f) => (f?.name || '').replace(/\.[^.]+$/, '');

/* ------------------------------------------------------------------ photo mode */

function PhotoMode({ onReady }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [geo, setGeo] = useState(null);          // result of probeGeoreference
  const [gsd, setGsd] = useState('');
  // On by default: it is what makes a georeferenced scene sit on real ground,
  // and for an image with no georeferencing it is simply skipped.
  const [autoDem, setAutoDem] = useState(true);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [health, setHealth] = useState(null);
  const [apiBase, setBase] = useState('');
  const [editingApi, setEditingApi] = useState(false);
  const [apiDraft, setApiDraft] = useState('');

  const probe = useCallback(() => {
    const b = getApiBase();
    setBase(b);
    setApiDraft(b);
    setHealth(null);
    checkHealth(b).then(setHealth).catch(() => setHealth({ ok: false }));
  }, []);

  // getApiBase() reads window, so it can only run after mount -- calling it during
  // render would differ between the server-rendered HTML and the browser and React
  // would throw a hydration mismatch.
  useEffect(() => { probe(); }, [probe]);

  const pick = useCallback((f) => {
    setFile(f ?? null);
    setError('');
    setGeo(null);
    if (!f) return;
    probeGeoreference(f).then((g) => setGeo(g ?? { georeferenced: false }));
  }, []);

  const saveApi = useCallback(() => {
    setApiBase(apiDraft);
    setEditingApi(false);
    probe();
  }, [apiDraft, probe]);

  // The DTM stage needs a georeferenced input. The service skips it on its own
  // when there is none, but asking only when it can apply keeps the request and
  // the rendered quantity honest: AGL for a plain photo, DSM for a GeoTIFF.
  const useDem = autoDem && !!geo?.georeferenced;

  const go = useCallback(async () => {
    if (!file) { setError('Choose an image first.'); return; }
    setError(''); setBusy(true);
    try {
      const out = await predictFromImage(file, {
        base: apiBase || undefined,
        gsd: gsd ? parseFloat(gsd) : null,
        autoDem: useDem,
        // Asking for terrain is a request to SEE the terrain, so render the DSM too.
        renderQuantity: useDem ? 'dsm' : 'agl',
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
        pixelSpacing: spacing, bitmap, metadata: out.metadata, name: baseName(file),
      });
    } catch (e) {
      console.error(e);
      setStatus('');
      setError(e?.message || 'Inference failed.');
    } finally {
      setBusy(false);
    }
  }, [file, gsd, useDem, apiBase, onReady]);

  const geoLine = !file ? null
    : !geo ? { tone: 'muted', text: 'Reading file header…' }
    : geo.georeferenced
      ? { tone: 'ok', text: `Georeferenced${geo.crs ? ` · ${geo.crs}` : ''}${geo.gsd ? ` · ${geo.gsd.toFixed(3)} m/px` : ''}` }
      : { tone: 'muted', text: `Not georeferenced — ${geo.reason || 'no coordinate reference'}. Heights are relative to local ground (AGL) at an assumed scale.` };

  return (
    <div className="uploadBody">
      <div
        className={`drop ${dragging ? 'dragging' : ''} ${file ? 'hasFile' : ''}`}
        role="button"
        tabIndex={0}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files?.[0]); }}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
      >
        <span className="dropIcon"><Icon name={file ? 'image' : 'upload'} size={18} /></span>
        <span className="dropText">
          <b>{file ? file.name : 'Drop a top-down satellite or aerial image'}</b>
          <small>{file ? `${(file.size / 1e6).toFixed(1)} MB · click to replace` : 'GeoTIFF, PNG or JPEG · up to 40 MB'}</small>
        </span>
        <input
          ref={inputRef} type="file" hidden accept=".tif,.tiff,.png,.jpg,.jpeg"
          onChange={(e) => pick(e.target.files?.[0])}
        />
      </div>

      {geoLine && (
        <p className={`geoLine ${geoLine.tone}`}>
          <Icon name="globe" size={13} />{geoLine.text}
        </p>
      )}

      <label className="check">
        <input type="checkbox" checked={autoDem} onChange={(e) => setAutoDem(e.target.checked)} />
        <span className="checkBox" aria-hidden="true" />
        <span>
          <b>Add real terrain (DTM)</b>
          <small>
            Georeferenced inputs sit on the Copernicus GLO-30 surface instead of flat ground.
            Skipped automatically when the image has no georeferencing.
          </small>
        </span>
      </label>

      <details className="more">
        <summary>Pixel spacing</summary>
        <label className="field">
          <span>Ground sample distance (m/px)</span>
          <input
            type="number" step="0.01" min="0.01" value={gsd}
            placeholder={geo?.gsd ? `${geo.gsd.toFixed(3)} from the file` : '0.33 assumed'}
            onChange={(e) => setGsd(e.target.value)}
          />
          <small>
            Optional. The model is scale-conditioned: this is what tells it a 40 px roof is a
            shed and not a warehouse.
          </small>
        </label>
      </details>

      <button type="button" className="btnPrimary" onClick={go} disabled={busy || !file}>
        {busy ? <><span className="spinner" />Reconstructing…</> : <>Generate 3D scene<Icon name="arrowRight" size={15} /></>}
      </button>

      {status && <p className="statusLine">{status}</p>}
      {error && <p className="errLine">{error}</p>}

      {!editingApi && (
        <p className={`service ${health ? (health.ok ? 'up' : 'down') : ''}`}>
          <span className="serviceDot" aria-hidden="true" />
          <span className="serviceText">
            {!health ? 'Checking model service…'
              : health.ok
                ? (health.model_warm ? <>Model service online · warm on <code>{health.device}</code></>
                                     : <>Model service online · loads on first request</>)
                : <>Model service unreachable · samples still work</>}
          </span>
          <a onClick={() => setEditingApi(true)} role="button" tabIndex={0}>Address</a>
        </p>
      )}

      {/* The backend address can change under us -- a free tunnel gets a new hostname
          every restart -- and it is compiled into the build, so without this the fix
          is a Vercel edit plus a redeploy. Set it here and this browser remembers. */}
      {editingApi && (
        <div className="field apiEdit">
          <span>Model service address</span>
          <input
            type="text" value={apiDraft} spellCheck={false}
            placeholder="https://something.trycloudflare.com"
            onChange={(e) => setApiDraft(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && saveApi()}
          />
          <small>
            Saved in this browser only. Opening the page with <code>?api=&lt;address&gt;</code>{' '}
            does the same without a redeploy.
          </small>
          <div className="btnRowLight">
            <button type="button" className="btnSecondary" onClick={saveApi}>Save &amp; test</button>
            <button type="button" className="btnText" onClick={() => { setApiDraft(apiBase); setEditingApi(false); }}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------- prepared-files mode */

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
      setError('Both a .tif height raster and an RGB texture are required.');
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
        pixelSpacing: effSpacing,
        bitmap,
        metadata,
        name: baseName(picked.texture),
      });
    } catch (e) {
      console.error(e);
      setStatus('');
      setError(e?.message || 'Failed to read the input files.');
    }
  }, [picked, spacing, onReady]);

  const slot = (label, file, note) => (
    <div className={`slot ${file ? 'ok' : ''}`}>
      <span className="slotDot" />
      <span className="slotText">
        <b>{label}</b>
        <small>{file ? file.name : note}</small>
      </span>
    </div>
  );

  return (
    <div className="uploadBody">
      <div
        className={`drop ${dragging ? 'dragging' : ''}`}
        role="button"
        tabIndex={0}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); accept(e.dataTransfer.files); }}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
      >
        <span className="dropIcon"><Icon name="folder" size={18} /></span>
        <span className="dropText">
          <b>Drop heightmap.tif, texture and metadata.json</b>
          <small>A bundle exported by the pipeline — no model service needed</small>
        </span>
        <input
          ref={inputRef} type="file" multiple hidden
          accept=".tif,.tiff,.png,.jpg,.jpeg,.json"
          onChange={(e) => accept(e.target.files)}
        />
      </div>

      <div className="slots">
        {slot('Height raster', picked.heightmap, 'float32 GeoTIFF, metres')}
        {slot('Texture', picked.texture, 'RGB, same grid')}
        {slot('metadata.json', picked.metadata, 'optional · pixel spacing')}
      </div>

      {picked.ignored.length > 0 && (
        <p className="noteLine">
          Ignored {picked.ignored.join(', ')}. <code>heightmap_preview.png</code> is 8-bit
          normalised and never used as data.
        </p>
      )}

      <label className="field">
        <span>Pixel spacing (m/px)</span>
        <input
          type="number" step="0.01" min="0.01" value={spacing}
          onChange={(e) => setSpacing(parseFloat(e.target.value) || 0.33)}
        />
        <small>Read from the GeoTIFF or metadata.json when present.</small>
      </label>

      <button type="button" className="btnPrimary" onClick={build}>
        Build 3D scene<Icon name="arrowRight" size={15} />
      </button>

      {warnings.length > 0 && (
        <ul className="warnList">
          {warnings.map((w, i) => <li key={i}>{w}</li>)}
        </ul>
      )}
      {status && <p className="statusLine">{status}</p>}
      {error && <p className="errLine">{error}</p>}
    </div>
  );
}
