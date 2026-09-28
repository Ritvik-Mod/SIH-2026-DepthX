/**
 * Client for the DepthX inference service: serve/modal_app.py on Modal (an L4 GPU that
 * runs only while someone is using the site), or serve/app.py on any machine.
 *
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
 * The default is the Modal deployment: its address is permanent, so it is simply the
 * built-in value. ?api=<url> still overrides it -- that is how
 * the old Mac + tunnel setup can be pointed at in an emergency.
 */

export const MODAL_BASE = 'https://chandlerismod--depthx-api.modal.run';

// Deliberately NOT read from NEXT_PUBLIC_DEPTHX_API any more. That variable is compiled
// in at build time, and a stale value left in Vercel from the tunnel era silently
// pointed the live site at a hostname that no longer existed -- "GPU unreachable" for
// every visitor, with correct code deployed. The Modal address never changes, so it is
// simply the default; ?api=<url> remains the way to test another backend.
const BUILD_TIME_BASE = MODAL_BASE;
const TOKEN = process.env.NEXT_PUBLIC_DEPTHX_TOKEN || '';
const STORE_KEY = 'depthx_api_base';

const clean = (u) => (u || '').trim().replace(/\/$/, '');

/**
 * Where the model service lives.
 *
 * The default is the Modal deployment, whose address is permanent. An override exists
 * for testing against another backend (the old Mac + tunnel, a local serve/app.py):
 *
 *   1. ?api=https://...   in the URL
 *   2. an address set with the "Address" editor on the upload card
 *   3. MODAL_BASE
 *
 * Overrides last for THIS TAB ONLY (sessionStorage). They used to be remembered in
 * localStorage indefinitely, which made sense while the backend was a tunnel whose
 * hostname changed every restart -- and became a trap once it stopped changing: a
 * tunnel address typed in weeks ago silently outranked the permanent default, so the
 * page showed "GPU unreachable" and hid the Warm up button for no visible reason.
 * Any such leftover is cleared the first time this runs.
 */
function forgetLegacyOverride() {
  try { window.localStorage.removeItem(STORE_KEY); } catch {}
}

export function getApiBase() {
  if (typeof window === 'undefined') return BUILD_TIME_BASE;   // server render
  forgetLegacyOverride();
  try {
    const q = clean(new URLSearchParams(window.location.search).get('api'));
    if (q) {
      try { window.sessionStorage.setItem(STORE_KEY, q); } catch {}
      return q;
    }
    const saved = clean(window.sessionStorage.getItem(STORE_KEY));
    if (saved) return saved;
  } catch {
    // storage disabled -- fall through to the built-in default
  }
  return BUILD_TIME_BASE;
}

/** Override the address for this tab. Empty string goes back to the default. */
export function setApiBase(url) {
  const v = clean(url);
  forgetLegacyOverride();
  // An address chosen in the editor (or a reset to the default) has to beat ?api= in
  // the URL, which is read first -- so drop the parameter from the address bar.
  try {
    const u = new URL(window.location.href);
    if (u.searchParams.has('api')) {
      u.searchParams.delete('api');
      window.history.replaceState(null, '', u.pathname + (u.search || '') + u.hash);
    }
  } catch {}
  try {
    if (v && v !== BUILD_TIME_BASE) window.sessionStorage.setItem(STORE_KEY, v);
    else window.sessionStorage.removeItem(STORE_KEY);
  } catch {}
  return v || BUILD_TIME_BASE;
}

/** Is this the built-in address, or an override? */
export function isDefaultApiBase(base) {
  return clean(base) === BUILD_TIME_BASE;
}

export const API_BASE = BUILD_TIME_BASE;      // the default, for display only
export const DEFAULT_API_BASE = BUILD_TIME_BASE;

const headers = () => (TOKEN ? { 'X-DepthX-Token': TOKEN } : {});

export async function checkHealth(base = getApiBase()) {
  const r = await fetch(`${base}/api/health`, { headers: headers() });
  if (!r.ok) throw new Error(`service returned ${r.status}`);
  return r.json();
}

/* ------------------------------------------------------------------ GPU control
 * Only the Modal backend has these. Each call is cheap and never starts a GPU by
 * itself except warmupGpu, which exists to do exactly that.
 */

export async function warmupGpu(base = getApiBase()) {
  const r = await fetch(`${base}/api/warmup`, { method: 'POST', headers: headers() });
  if (!r.ok) throw new Error(await readError(r));
  return r.json();
}

export async function keepaliveGpu(client, base = getApiBase()) {
  const r = await fetch(`${base}/api/keepalive`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers() },
    body: JSON.stringify({ client }),
  });
  if (!r.ok) throw new Error(await readError(r));
  return r.json();
}

/** Page is closing: sendBeacon survives the unload where fetch would be cancelled. */
export function releaseGpu(client, base = getApiBase()) {
  try {
    const body = new Blob([JSON.stringify({ client })], { type: 'text/plain' });
    return navigator.sendBeacon(`${base}/api/release`, body);
  } catch {
    return false;
  }
}

/**
 * Upload one image, wait for the model, and return blobs the loader can read.
 *
 * onProgress(text) is called at each step; onJob(job) receives every status poll,
 * which carries the server-side timings (GPU start-up, inference, pipeline).
 */
export async function predictFromImage(file, {
  base = getApiBase(), gsd = null, autoDem = false, renderQuantity = 'agl',
  onProgress = () => {}, onJob = () => {}, onUpload = () => {}, pollMs = 700,
  timeoutMs = 10 * 60 * 1000,
} = {}) {
  const form = new FormData();
  form.append('image', file);
  if (gsd) form.append('gsd', String(gsd));
  form.append('auto_dem', autoDem ? 'true' : 'false');
  form.append('render_quantity', renderQuantity);

  onProgress('Uploading image…');
  const up = await fetch(`${base}/api/predict`, { method: 'POST', body: form, headers: headers() });
  if (!up.ok) throw new Error(await readError(up));
  const upJson = await up.json();
  const { job_id: jobId } = upJson;
  onUpload(upJson);

  onProgress('Queued for the model…');
  const deadline = Date.now() + timeoutMs;
  let job;
  for (;;) {
    if (Date.now() > deadline) throw new Error('Timed out waiting for the model.');
    await sleep(pollMs);
    const r = await fetch(`${base}/api/jobs/${jobId}`, { headers: headers() });
    if (!r.ok) throw new Error(await readError(r));
    job = await r.json();
    onJob(job);
    if (job.status === 'done') break;
    if (job.status === 'error') throw new Error(job.error || 'Inference failed.');
    onProgress(job.status === 'running' ? 'Running inference…' : 'Queued for the model…');
  }

  onProgress('Fetching height raster…');
  // The Modal service sends the texture as JPEG (a third of the bytes); the Mac
  // service still sends PNG. Take whichever this backend offers.
  const texName = job.files['texture.jpg'] ? 'texture.jpg' : 'texture.png';
  const [heightmap, texture, metadata] = await Promise.all([
    fetchBlob(`${base}${job.files['heightmap.tif']}`),
    fetchBlob(`${base}${job.files[texName]}`),
    fetchBlob(`${base}${job.files['metadata.json']}`).then((b) => b.text()).then((t) => JSON.parse(t)),
  ]);

  // Name the blobs: classifyFiles() and the GeoTIFF reader both key off extensions,
  // and a bare Blob has no name at all.
  return {
    jobId,
    job,
    manifest: job.manifest,
    metadata,
    heightmapFile: new File([heightmap], 'heightmap.tif', { type: 'image/tiff' }),
    textureFile: new File([texture], texName, { type: texName.endsWith('.jpg') ? 'image/jpeg' : 'image/png' }),
  };
}

/**
 * Download one result file, retrying a transient failure. A 404/5xx straight after
 * a job finishes is the case that made a first run look stuck, so two retries with
 * a short pause cover it, and a final failure names the file and the status instead
 * of just the file.
 */
async function fetchBlob(url, attempts = 3) {
  const name = url.split('/').pop();
  let last = '';
  for (let i = 0; i < attempts; i++) {
    try {
      const r = await fetch(url, { headers: headers() });
      if (r.ok) return r.blob();
      last = `HTTP ${r.status} (${await readError(r)})`;
      if (r.status < 500 && r.status !== 404) break;       // a real refusal: do not retry
    } catch (e) {
      last = e?.message || 'network error';
    }
    await sleep(800 * (i + 1));
  }
  throw new Error(`Could not download the result file ${name}: ${last}.`);
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
