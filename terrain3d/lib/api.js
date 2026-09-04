/**
 * Client for the DepthX inference service (serve/app.py).
 *
 * One photo in, the same {heightmap.tif, texture.png, metadata.json} bundle out that
 * the manual upload path already understands -- so the viewer, the mesh builder and
 * every sanity check downstream stay exactly as they were. This module only replaces
 * where the files come from.
 *
 * Upload returns a job id rather than the result: a large scene takes ~30 s, which is
 * longer than a serverless function or an idle proxy will hold a connection open. We
 * poll instead, which also gives the UI something honest to display.
 *
 * Set NEXT_PUBLIC_DEPTHX_API to the service URL (an ngrok/cloudflared address when the
 * backend is not on this machine). Without it we assume a local server on :8000.
 */

const BUILD_TIME_BASE =
  process.env.NEXT_PUBLIC_DEPTHX_API?.replace(/\/$/, '') || 'http://localhost:8000';
const TOKEN = process.env.NEXT_PUBLIC_DEPTHX_TOKEN || '';
const STORE_KEY = 'depthx_api_base';

const clean = (u) => (u || '').trim().replace(/\/$/, '');

/**
 * Where the model service lives, decided AT RUNTIME rather than at build time.
 *
 * NEXT_PUBLIC_* values are compiled into the JavaScript when Vercel builds, so a
 * backend whose address changes -- a free Cloudflare quick tunnel gets a new hostname
 * every restart -- would need an environment-variable edit and a full redeploy each
 * time. That is a two-minute outage at the worst possible moment.
 *
 * Resolution order, first hit wins:
 *   1. ?api=https://…   in the URL, which is also remembered for next time
 *   2. whatever was remembered in this browser
 *   3. the build-time NEXT_PUBLIC_DEPTHX_API
 *   4. localhost:8000
 *
 * So when the tunnel changes you open the site once with ?api=<new url>, or paste it
 * into the box on the upload card, and this browser keeps using it. Sharing that same
 * ?api= link points anyone else's browser at it too, without touching Vercel.
 *
 * localStorage is per-browser and per-device, so it is a convenience, never the
 * source of truth: the build-time value stays the sane default for a fresh visitor.
 */
export function getApiBase() {
  if (typeof window === 'undefined') return BUILD_TIME_BASE;   // server render
  try {
    const q = clean(new URLSearchParams(window.location.search).get('api'));
    if (q) {
      try { window.localStorage.setItem(STORE_KEY, q); } catch {}
      return q;
    }
    const saved = clean(window.localStorage.getItem(STORE_KEY));
    if (saved) return saved;
  } catch {
    // private browsing, or storage disabled -- fall through to the built-in default
  }
  return BUILD_TIME_BASE;
}

/** Remember an override for this browser. Empty string clears it. */
export function setApiBase(url) {
  const v = clean(url);
  try {
    if (v) window.localStorage.setItem(STORE_KEY, v);
    else window.localStorage.removeItem(STORE_KEY);
  } catch {}
  return v || BUILD_TIME_BASE;
}

export const API_BASE = BUILD_TIME_BASE;      // the default, for display only
export const DEFAULT_API_BASE = BUILD_TIME_BASE;

const headers = () => (TOKEN ? { 'X-DepthX-Token': TOKEN } : {});

export async function checkHealth(base = getApiBase()) {
  const r = await fetch(`${base}/api/health`, { headers: headers() });
  if (!r.ok) throw new Error(`service returned ${r.status}`);
  return r.json();
}

/**
 * Upload one image, wait for the model, and return blobs the loader can read.
 *
 * onProgress(text) is called at each step; the caller decides what to show.
 */
export async function predictFromImage(file, {
  base = getApiBase(), gsd = null, autoDem = false, renderQuantity = 'agl',
  onProgress = () => {}, pollMs = 1200, timeoutMs = 10 * 60 * 1000,
} = {}) {
  const form = new FormData();
  form.append('image', file);
  if (gsd) form.append('gsd', String(gsd));
  form.append('auto_dem', autoDem ? 'true' : 'false');
  form.append('render_quantity', renderQuantity);

  onProgress('Uploading image…');
  const up = await fetch(`${base}/api/predict`, { method: 'POST', body: form, headers: headers() });
  if (!up.ok) throw new Error(await readError(up));
  const { job_id: jobId } = await up.json();

  onProgress('Queued for the model…');
  const deadline = Date.now() + timeoutMs;
  let job;
  for (;;) {
    if (Date.now() > deadline) throw new Error('Timed out waiting for the model.');
    await sleep(pollMs);
    const r = await fetch(`${base}/api/jobs/${jobId}`, { headers: headers() });
    if (!r.ok) throw new Error(await readError(r));
    job = await r.json();
    if (job.status === 'done') break;
    if (job.status === 'error') throw new Error(job.error || 'Inference failed.');
    onProgress(job.status === 'running' ? 'Running inference…' : 'Queued for the model…');
  }

  onProgress('Fetching height raster…');
  const [heightmap, texture, metadata] = await Promise.all([
    fetchBlob(`${base}${job.files['heightmap.tif']}`),
    fetchBlob(`${base}${job.files['texture.png']}`),
    fetch(`${base}${job.files['metadata.json']}`, { headers: headers() }).then((r) => r.json()),
  ]);

  // Name the blobs: classifyFiles() and the GeoTIFF reader both key off extensions,
  // and a bare Blob has no name at all.
  return {
    jobId,
    manifest: job.manifest,
    metadata,
    heightmapFile: new File([heightmap], 'heightmap.tif', { type: 'image/tiff' }),
    textureFile: new File([texture], 'texture.png', { type: 'image/png' }),
  };
}

async function fetchBlob(url) {
  const r = await fetch(url, { headers: headers() });
  if (!r.ok) throw new Error(await readError(r));
  return r.blob();
}

async function readError(r) {
  try {
    const j = await r.json();
    return j.detail || `${r.status} ${r.statusText}`;
  } catch {
    return `${r.status} ${r.statusText}`;
  }
}

const sleep = (ms) => new Promise((res) => setTimeout(res, ms));
