/**
 * The GPU's lifecycle as the page sees it: asleep, starting, warm.
 *
 * One module-level controller rather than component state, because the page swaps
 * between the upload card and the 3D viewer: a keep-alive owned by the upload card
 * would stop the moment a scene opened, the GPU would go back to sleep, and the next
 * upload would pay the cold start again.
 *
 * The rules that keep the bill honest (see serve/modal_app.py for the server side):
 *
 *   - loading the page never starts a GPU. Only the Warm up button or an upload does;
 *   - once this page has used the GPU, it sends a heartbeat every 25 s, but ONLY
 *     while the tab is visible and someone has touched the page in the last ten
 *     minutes. A tab left open overnight lets the GPU sleep;
 *   - closing the page sends a release beacon, and the server shuts the GPU down
 *     at once if no other open page is using it.
 *
 * Against the old Mac + tunnel backend none of this applies: its health reply has no
 * `gpu` field, the status reads "unsupported", and the UI hides the GPU controls.
 */
import { checkHealth, getApiBase, keepaliveGpu, releaseGpu, warmupGpu } from './api';

const HEARTBEAT_MS = 25_000;
const IDLE_MS = 10 * 60_000;
const POLL_MS = 1_000;

const s = {
  status: 'unknown',   // unknown | asleep | starting | warm | unsupported | offline
  since: null,         // client ms when the current status began (for timers)
  gpuType: null,
  loadS: null,         // server-reported time to put the model on the GPU
  error: null,
  untrained: false,
};

const listeners = new Set();
let started = false;
let engaged = false;         // has this page used the GPU? only then heartbeat / release
let lastActivity = Date.now();
let pollTimer = null;
let beatTimer = null;
let clientId = null;

const emit = () => listeners.forEach((fn) => fn({ ...s }));

function set(patch) {
  const statusChanged = patch.status && patch.status !== s.status;
  Object.assign(s, patch);
  if (statusChanged && !patch.since) s.since = Date.now();
  emit();
}

function fromServer(g, h) {
  if (!g) return;
  const patch = { error: g.error || null, untrained: !!g.untrained };
  if (g.state !== s.status) {
    patch.status = g.state;
    // A GPU already starting when we first looked began before this page did;
    // anchor the timer to the server's own clock offset rather than to now.
    if (g.state === 'starting' && g.since && h?.now) {
      patch.since = Date.now() - Math.max(0, (h.now - g.since) * 1000);
    }
  }
  if (g.gpu_load_s != null) patch.loadS = g.gpu_load_s;
  set(patch);
}

async function refresh() {
  try {
    const h = await checkHealth(getApiBase());
    if (!h.gpu) { set({ status: 'unsupported' }); return; }
    set({ gpuType: h.gpu_type || null });
    fromServer(h.gpu, h);
  } catch {
    set({ status: 'offline' });
  }
}

function pollWhileStarting() {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    await refresh();
    if (s.status !== 'starting') clearInterval(pollTimer);
  }, POLL_MS);
}

async function beat() {
  if (!engaged || document.visibilityState !== 'visible') return;
  if (Date.now() - lastActivity > IDLE_MS) return;
  if (s.status !== 'warm' && s.status !== 'starting') return;
  try {
    const r = await keepaliveGpu(clientId, getApiBase());
    fromServer(r.gpu);
  } catch {
    /* a missed heartbeat only means the GPU may sleep sooner */
  }
}

function markActive() { lastActivity = Date.now(); }

/** Start watching. Safe to call many times; only the first call does anything. */
export function initGpu() {
  if (started || typeof window === 'undefined') return;
  started = true;
  try {
    clientId = sessionStorage.getItem('depthx_client')
      || Math.random().toString(36).slice(2, 12);
    sessionStorage.setItem('depthx_client', clientId);
  } catch {
    clientId = Math.random().toString(36).slice(2, 12);
  }
  ['pointerdown', 'keydown', 'wheel', 'touchstart', 'pointermove'].forEach((ev) =>
    window.addEventListener(ev, markActive, { passive: true }));
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') { markActive(); refresh(); }
  });
  // pagehide fires on close, reload and navigation away, including on mobile,
  // where beforeunload often does not.
  window.addEventListener('pagehide', () => { if (engaged) releaseGpu(clientId, getApiBase()); });
  beatTimer = setInterval(beat, HEARTBEAT_MS);
  refresh().then(() => { if (s.status === 'starting') pollWhileStarting(); });
}

/** The Warm up button. */
export async function warmUp() {
  engaged = true;
  markActive();
  if (s.status === 'warm' || s.status === 'starting') return;
  set({ status: 'starting', since: Date.now(), error: null });
  try {
    const r = await warmupGpu(getApiBase());
    fromServer(r.gpu);
  } catch (e) {
    set({ status: 'offline', error: e?.message || 'Could not reach the model service.' });
    return;
  }
  pollWhileStarting();
}

/** An upload was sent: the GPU is in use from here on, whether it was warm or not. */
export function engageGpu(gpuStateAtUpload) {
  engaged = true;
  markActive();
  if (gpuStateAtUpload && gpuStateAtUpload !== 'warm' && s.status !== 'starting') {
    set({ status: 'starting', since: Date.now() });
    pollWhileStarting();
  }
}

/** Re-read the GPU state, e.g. after the service address changed. */
export function refreshGpu() {
  set({ status: 'unknown', since: null, error: null });
  refresh().then(() => { if (s.status === 'starting') pollWhileStarting(); });
}

/** Server-reported GPU state from a job poll. */
export function noteGpuState(g) { fromServer(g); }

export function getGpu() { return { ...s }; }

export function subscribeGpu(fn) {
  listeners.add(fn);
  fn({ ...s });
  return () => listeners.delete(fn);
}
