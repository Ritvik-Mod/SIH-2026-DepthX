'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  loadHeightmap, loadTextureBitmap, probeGeoreference, resolvePixelSpacing,
} from '@/lib/load';
import { checkHealth, getApiBase, isDefaultApiBase, predictFromImage, setApiBase } from '@/lib/api';
import { engageGpu, initGpu, noteGpuState, refreshGpu, subscribeGpu, warmUp } from '@/lib/gpu';
import Icon from './Icons';

const baseName = (f) => (f?.name || '').replace(/\.[^.]+$/, '');
const secs = (ms) => `${(Math.max(0, ms) / 1000).toFixed(1)} s`;

/* ------------------------------------------------------------------ GPU control */

/**
 * GPU status and the Warm up button.
 *
 * The point of separating these from the upload: a cold GPU takes tens of seconds
 * to start, and the model itself a few. Shown as one spinner, the start-up reads as
 * a slow model. Warming up first, with its own timer, makes the difference visible.
 */
function GpuControl({ gpu, now }) {
  if (gpu.status === 'unsupported' || gpu.status === 'unknown') return null;
  const starting = gpu.status === 'starting';
  const warm = gpu.status === 'warm';
  const label = warm ? 'GPU ready'
    : starting ? `Starting GPU ${gpu.since ? secs(now - gpu.since) : ''}`
    : gpu.status === 'offline' ? 'GPU unreachable'
    : 'GPU asleep';
  return (
    <div className="gpuCtl">
      <span className={`gpuState ${gpu.status}`}>
        <span className="gpuDot" aria-hidden="true" />
        {label}
        {gpu.gpuType && warm && <small>{gpu.gpuType}</small>}
      </span>
      {!warm && gpu.status !== 'offline' && (
        <button type="button" className="btnSecondary btnWarm" onClick={warmUp} disabled={starting}>
          {starting ? <span className="spinner" /> : <Icon name="sun" size={14} />}
          {starting ? 'Warming up' : 'Warm up GPU'}
        </button>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ timings */

/**
 * Live timers for one reconstruction. The server's own figures replace the
 * client's estimates as soon as they arrive, so the final numbers are exact.
 */
function Timings({ run, now }) {
  if (!run) return null;
  const j = run.job || {};
  const t = j.timing;
  const done = run.phase === 'download' || run.phase === 'done';
  const running = run.phase === 'running';

  const startRow = run.gpuBefore === 'warm'
    ? { value: 'already warm', state: 'ok' }
    : t?.wait_s != null
      ? { value: `${t.wait_s.toFixed(1)} s`, state: 'ok' }
      : run.uploadedAt
        ? { value: secs((run.runningAt || now) - run.uploadedAt), state: running || done ? 'ok' : 'live' }
        : { value: 'waiting', state: 'idle' };

  const inferRow = t?.inference_s != null
    ? { value: `${t.inference_s.toFixed(1)} s`, state: 'ok' }
    : run.runningAt
      ? { value: secs(now - run.runningAt), state: 'live' }
      : { value: '', state: 'idle' };

  const restRow = t?.pipeline_s != null && t?.inference_s != null
    ? { value: `${Math.max(0, t.pipeline_s - t.inference_s).toFixed(1)} s`, state: 'ok' }
    : { value: '', state: running ? 'live' : 'idle' };

  const rows = [
    { label: 'Upload', ...(run.uploadedAt ? { value: secs(run.uploadedAt - run.t0), state: 'ok' }
                                          : { value: secs(now - run.t0), state: 'live' }) },
    { label: 'GPU start-up', ...startRow },
    { label: 'Model inference', ...inferRow },
    { label: 'Terrain and files', ...restRow },
    { label: 'Download and build', ...(run.phase === 'download' ? { value: secs(now - run.doneAt), state: 'live' }
      : run.phase === 'done' ? { value: 'done', state: 'ok' } : { value: '', state: 'idle' }) },
  ];

  // After a failure nothing is live any more: the step that was running failed.
  if (run.phase === 'failed') {
    rows.forEach((r) => { if (r.state === 'live') { r.state = 'fail'; r.value = 'failed'; } });
  }

  return (
    <div className="timings" aria-live="polite">
      {rows.map((r) => (
        <div key={r.label} className={`tRow ${r.state}`}>
          <span className="tMark" aria-hidden="true">
            {r.state === 'live' ? <span className="spinner" />
              : r.state === 'ok' ? <Icon name="check" size={12} />
              : r.state === 'fail' ? <Icon name="close" size={11} /> : null}
          </span>
          <span className="tLabel">{r.label}</span>
          <b>{r.value}</b>
        </div>
      ))}
      {j.untrained && (
        <p className="tWarn">Timing test build: the model weights are not uploaded yet, so these heights are not real.</p>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ the card */

export default function UploadPanel({ onReady }) {
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
  const [gpu, setGpu] = useState({ status: 'unknown' });
  const [run, setRun] = useState(null);
  const [now, setNow] = useState(() => Date.now());

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
  useEffect(() => { probe(); initGpu(); return subscribeGpu(setGpu); }, [probe]);

  // One clock for every live timer on the card, running only while something is timing.
  const ticking = busy || gpu.status === 'starting';
  useEffect(() => {
    if (!ticking) return undefined;
    const id = setInterval(() => setNow(Date.now()), 100);
    return () => clearInterval(id);
  }, [ticking]);

  const pick = useCallback((f) => {
    setFile(f ?? null);
    setError('');
    setGeo(null);
    if (!f) return;
    probeGeoreference(f).then((g) => setGeo(g ?? { georeferenced: false }));
  }, []);

  const saveApi = useCallback((value) => {
    setApiBase(value);
    setEditingApi(false);
    probe();
    refreshGpu();
  }, [probe]);

  const custom = apiBase && !isDefaultApiBase(apiBase);

  // The DTM stage needs a georeferenced input. The service skips it on its own
  // when there is none, but asking only when it can apply keeps the request and
  // the rendered quantity honest: AGL for a plain photo, DSM for a GeoTIFF.
  const useDem = autoDem && !!geo?.georeferenced;

  const go = useCallback(async () => {
    if (!file) { setError('Choose an image first.'); return; }
    setError(''); setBusy(true);
    const t0 = Date.now();
    setRun({ t0, phase: 'upload' });
    try {
      const out = await predictFromImage(file, {
        base: apiBase || undefined,
        gsd: gsd ? parseFloat(gsd) : null,
        autoDem: useDem,
        // Asking for terrain is a request to SEE the terrain, so render the DSM too.
        renderQuantity: useDem ? 'dsm' : 'agl',
        onProgress: setStatus,
        onUpload: (u) => {
          engageGpu(u.gpu);
          setRun((r) => ({ ...r, phase: 'queued', uploadedAt: Date.now(), gpuBefore: u.gpu }));
        },
        onJob: (j) => {
          if (j.gpu) noteGpuState(j.gpu);
          setRun((r) => {
            const next = { ...r, job: j };
            if (j.status === 'running' && !r.runningAt) { next.runningAt = Date.now(); next.phase = 'running'; }
            if (j.status === 'done') { next.phase = 'download'; next.doneAt = Date.now(); }
            return next;
          });
        },
      });
      setStatus('Reading the height raster…');
      const hm = await loadHeightmap(out.heightmapFile, setStatus);
      const bitmap = await loadTextureBitmap(out.textureFile, setStatus);
      const spacing = hm.geoSpacing ?? resolvePixelSpacing(out.metadata);
      setRun((r) => ({ ...r, phase: 'done' }));
      setStatus('Building mesh…');
      const timing = out.job?.timing
        ? { ...out.job.timing, cold: !!out.job.cold, totalS: (Date.now() - t0) / 1000 }
        : { totalS: (Date.now() - t0) / 1000 };
      onReady({
        heights: hm.data, width: hm.width, height: hm.height,
        min: hm.min, max: hm.max, mean: hm.mean, nodataPixels: hm.nodataPixels,
        pixelSpacing: spacing, bitmap, metadata: out.metadata, name: baseName(file),
        timing,
      });
    } catch (e) {
      console.error(e);
      setStatus('');
      setError(e?.message || 'Inference failed.');
      // stop every live timer: a spinner that outlives the failure reads as "still working"
      setRun((r) => (r ? { ...r, phase: 'failed', failedAt: Date.now() } : r));
    } finally {
      setBusy(false);
    }
  }, [file, gsd, useDem, apiBase, onReady]);

  const geoLine = !file ? null
    : !geo ? { tone: 'muted', text: 'Reading file header…' }
    : geo.georeferenced
      ? { tone: 'ok', text: `Georeferenced${geo.crs ? ` · ${geo.crs}` : ''}${geo.gsd ? ` · ${geo.gsd.toFixed(3)} m/px` : ''}` }
      : { tone: 'muted', text: `Not georeferenced: ${geo.reason || 'no coordinate reference'}. Heights are relative to local ground (AGL) at an assumed scale.` };

  const serviceText = !health ? 'Checking model service…'
    : !health.ok ? 'Model service unreachable. The sample scenes still work.'
    : health.backend === 'modal' ? `Model service online · NVIDIA ${health.gpu_type || 'GPU'} on Modal`
    : health.model_warm ? <>Model service online · warm on <code>{health.device}</code></>
    : 'Model service online · loads on first request';

  return (
    <section className="card upload" aria-labelledby="upload-title">
      <div className="cardHead">
        <h2 id="upload-title">Reconstruct an image</h2>
        <GpuControl gpu={gpu} now={now} />
      </div>

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

        <Timings run={run} now={now} />

        {status && busy && <p className="statusLine">{status}</p>}
        {error && <p className="errLine">{error}</p>}

        {!editingApi && (
          <p className={`service ${health ? (health.ok ? 'up' : 'down') : ''}`}>
            <span className="serviceDot" aria-hidden="true" />
            <span className="serviceText">{serviceText}</span>
            <a onClick={() => setEditingApi(true)} role="button" tabIndex={0}>Address</a>
          </p>
        )}

        {/* A non-default address must never be silent: it is the one thing that can
            make a working service look broken from this page. */}
        {!editingApi && custom && (
          <p className="customApi">
            Using a custom address for this tab: <code>{apiBase}</code>{' '}
            <a onClick={() => saveApi('')} role="button" tabIndex={0}>Use the default</a>
          </p>
        )}

        {/* An override for pointing this browser at another backend (for instance the
            old Mac + tunnel setup) without a redeploy. */}
        {editingApi && (
          <div className="field apiEdit">
            <span>Model service address</span>
            <input
              type="text" value={apiDraft} spellCheck={false}
              placeholder="https://chandlerismod--depthx-api.modal.run"
              onChange={(e) => setApiDraft(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && saveApi(apiDraft)}
            />
            <small>
              For testing another backend. Applies to this tab only and is forgotten when
              it closes. <code>?api=&lt;address&gt;</code> in the URL does the same.
            </small>
            <div className="btnRowLight">
              <button type="button" className="btnSecondary" onClick={() => saveApi(apiDraft)}>Save and test</button>
              <button type="button" className="btnSecondary" onClick={() => saveApi('')}>Use the default</button>
              <button type="button" className="btnText" onClick={() => { setApiDraft(apiBase); setEditingApi(false); }}>
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
