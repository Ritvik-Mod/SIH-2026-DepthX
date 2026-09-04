// Can this browser actually give us a WebGL context?
//
// Why this exists: THREE.WebGLRenderer throws from its constructor when the
// context cannot be created, and that throw used to escape all the way out of
// React and replace the entire page with Next's generic "Application error".
// The 3D view is not the only thing on this site -- the upload form and the
// prepared-files loader matter too -- so the failure has to be caught and
// turned into a degraded view rather than a dead page.
//
// The usual cause is NOT a missing GPU. It is Chrome's GPU process being
// disabled or failing to start, which reports as GL_VENDOR = Disabled and
// VENDOR = 0xffff. Recent Chrome also removed the automatic SwiftShader
// software fallback for WebGL, so machines that silently software-rendered a
// year ago now fail outright.

export function probeWebGL() {
  if (typeof window === 'undefined') return { ok: true, reason: null };   // SSR: assume yes
  // ?nowebgl=1 forces the 2D path. Kept in for two reasons: it is the only way
  // to test the fallback on a machine whose WebGL works, and it gives us a
  // reliable way to show the 2D view deliberately.
  try {
    if (new URLSearchParams(window.location.search).get('nowebgl') === '1') {
      return { ok: false, reason: 'Forced by ?nowebgl=1.' };
    }
  } catch {}
  if (typeof WebGLRenderingContext === 'undefined') {
    return { ok: false, reason: 'This browser has no WebGL support at all.' };
  }
  let canvas;
  try {
    canvas = document.createElement('canvas');
    const gl =
      canvas.getContext('webgl2') ||
      canvas.getContext('webgl') ||
      canvas.getContext('experimental-webgl');
    if (!gl) {
      return {
        ok: false,
        reason:
          'The browser knows about WebGL but refused to create a context. ' +
          'This is almost always hardware acceleration being switched off, ' +
          'or the GPU process failing to start.',
      };
    }
    // Some setups hand back a context that is already lost.
    if (typeof gl.isContextLost === 'function' && gl.isContextLost()) {
      return { ok: false, reason: 'The WebGL context was created but was immediately lost.' };
    }
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    const vendor = dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : null;
    const rend = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : null;
    // Free it again so we do not hold one of the browser's limited contexts.
    const lose = gl.getExtension('WEBGL_lose_context');
    if (lose) lose.loseContext();
    return { ok: true, reason: null, vendor, renderer: rend };
  } catch (e) {
    return { ok: false, reason: e?.message || String(e) };
  }
}
