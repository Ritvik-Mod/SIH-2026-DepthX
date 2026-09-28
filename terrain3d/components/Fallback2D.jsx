'use client';

// Shown when WebGL is unavailable. NOT an error page: the model ran, the
// heightmap is in memory, and everything except the flythrough still works.
// A 2D canvas needs no GPU, so we render the same float32 height field as a
// shaded relief image and keep the numbers on screen.

import { useEffect, useRef, useState, useCallback } from 'react';

const RAMP = [
  [0.00, [ 32,  58,  84]],
  [0.12, [ 46, 102,  92]],
  [0.32, [ 92, 140,  78]],
  [0.55, [176, 166,  96]],
  [0.78, [198, 138,  92]],
  [1.00, [244, 240, 232]],
];

function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0]) {
      const [t0, c0] = RAMP[i - 1];
      const [t1, c1] = RAMP[i];
      const f = (t - t0) / (t1 - t0 || 1);
      return [
        c0[0] + (c1[0] - c0[0]) * f,
        c0[1] + (c1[1] - c0[1]) * f,
        c0[2] + (c1[2] - c0[2]) * f,
      ];
    }
  }
  return RAMP[RAMP.length - 1][1];
}

export default function Fallback2D({ dataset, onReset, reason }) {
  const canvasRef = useRef(null);
  const [view, setView] = useState('relief');   // relief | height | texture
  const [exag, setExag] = useState(3);
  const [az, setAz] = useState(315);
  const [el, setEl] = useState(45);

  const {
    heights, width, height, min, max, mean,
    pixelSpacing, bitmap, nodataPixels, metadata,
  } = dataset || {};
  const quantity = metadata?.quantity || 'AGL';

  const draw = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv || !heights) return;
    cv.width = width;
    cv.height = height;
    const ctx = cv.getContext('2d');
    if (!ctx) return;

    if (view === 'texture' && bitmap?.source) {
      // loadTextureBitmap returns { source, flipY }, not a raw drawable.
      // flipY === false means createImageBitmap already flipped it for WebGL's
      // texture convention (v = 0 at the bottom), so a 2D canvas has to flip it
      // back to get the image the right way up. flipY === true is the
      // HTMLImageElement path, which was never flipped.
      ctx.save();
      if (!bitmap.flipY) { ctx.translate(0, height); ctx.scale(1, -1); }
      ctx.drawImage(bitmap.source, 0, 0, width, height);
      ctx.restore();
      return;
    }

    const img = ctx.createImageData(width, height);
    const d = img.data;
    const span = (max - min) || 1;

    // Light direction from azimuth/elevation, in the raster's own frame.
    const ar = (az * Math.PI) / 180;
    const er = (el * Math.PI) / 180;
    const lx = Math.cos(er) * Math.sin(ar);
    const ly = -Math.cos(er) * Math.cos(ar);
    const lz = Math.sin(er);

    const sp = pixelSpacing > 0 ? pixelSpacing : 1;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        const h = heights[i];
        const t = (h - min) / span;
        let r, g, b;

        if (view === 'height') {
          const v = Math.round(255 * Math.max(0, Math.min(1, t)));
          r = g = b = v;
        } else {
          // central differences -> surface normal -> Lambert term
          const xl = x > 0 ? heights[i - 1] : h;
          const xr = x < width - 1 ? heights[i + 1] : h;
          const yu = y > 0 ? heights[i - width] : h;
          const yd = y < height - 1 ? heights[i + width] : h;
          const dzdx = ((xr - xl) * exag) / (2 * sp);
          const dzdy = ((yd - yu) * exag) / (2 * sp);
          const len = Math.sqrt(dzdx * dzdx + dzdy * dzdy + 1) || 1;
          const nx = -dzdx / len, ny = -dzdy / len, nz = 1 / len;
          let shade = nx * lx + ny * ly + nz * lz;
          shade = 0.25 + 0.75 * Math.max(0, shade);       // ambient floor
          const c = ramp(t);
          r = c[0] * shade; g = c[1] * shade; b = c[2] * shade;
        }
        const o = i * 4;
        d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [heights, width, height, min, max, pixelSpacing, bitmap, view, exag, az, el]);

  // The whole point of this component is to be the thing that still works, so
  // it must not be able to throw during render.
  const safeDraw = useCallback(() => {
    try { draw(); } catch (e) { console.error('[DepthX] 2D draw failed:', e); }
  }, [draw]);

  useEffect(() => { safeDraw(); }, [safeDraw]);

  const save = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const a = document.createElement('a');
    a.href = cv.toDataURL('image/png');
    a.download = `depthx_${view}.png`;
    a.click();
  }, [view]);

  const pill = (on) => ({
    padding: '5px 11px', borderRadius: 999, fontSize: 11, cursor: 'pointer',
    fontFamily: 'ui-monospace, monospace', letterSpacing: '0.03em',
    border: `1px solid ${on ? 'rgba(95,178,255,0.55)' : 'rgba(255,255,255,0.12)'}`,
    background: on ? 'rgba(95,178,255,0.16)' : 'rgba(17,21,26,0.86)',
    color: on ? '#8ecbff' : 'rgba(255,255,255,0.55)', transition: 'all 120ms ease',
  });
  const row = { display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' };
  const lbl = { fontSize: 10, opacity: 0.5, fontFamily: 'ui-monospace, monospace' };

  return (
    <div style={{
      position: 'fixed', inset: 0, background: '#0c0f13', color: '#e8edf2',
      display: 'flex', flexDirection: 'column',
      font: '13px/1.5 ui-sans-serif, system-ui, sans-serif', overflow: 'auto',
    }}>
      <div style={{
        padding: '14px 18px', borderBottom: '1px solid rgba(255,255,255,0.08)',
        background: 'rgba(255,180,60,0.07)',
      }}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>
          3D view unavailable on this machine. Showing the result in 2D instead.
        </div>
        <div style={{ opacity: 0.75, fontSize: 12 }}>
          The model ran and the height data loaded correctly. Only the WebGL
          flythrough could not start.{reason ? ` Browser said: ${reason}` : ''}
        </div>
        <details style={{ marginTop: 8, fontSize: 12, opacity: 0.8 }}>
          <summary style={{ cursor: 'pointer' }}>How to enable the 3D view</summary>
          <ol style={{ margin: '8px 0 0 18px', lineHeight: 1.7 }}>
            <li>Open <code>chrome://settings/system</code> and turn on
                “Use graphics acceleration when available”, then fully quit and reopen Chrome.</li>
            <li>Open <code>chrome://gpu</code>. The “Graphics Feature Status” block
                names the exact reason WebGL is off.</li>
            <li>Open <code>chrome://flags/#ignore-gpu-blocklist</code>, set it to
                Enabled, and relaunch.</li>
            <li>No usable GPU at all (virtual machine or remote desktop)? Launch Chrome
                with <code>--enable-unsafe-swiftshader</code> to render in software.</li>
            <li>Or just try Firefox, which has its own software fallback.</li>
          </ol>
        </details>
      </div>

      <div style={{ padding: '10px 18px', display: 'flex', gap: 18, ...row }}>
        <div style={row}>
          <span style={lbl}>VIEW</span>
          {['relief', 'height', 'texture'].map((v) => (
            <button key={v} onClick={() => setView(v)} style={pill(view === v)}
                    disabled={v === 'texture' && !bitmap?.source}>{v}</button>
          ))}
        </div>
        {view === 'relief' && (
          <>
            <div style={row}>
              <span style={lbl}>EXAG {exag.toFixed(1)}×</span>
              <input type="range" min="1" max="12" step="0.5" value={exag}
                     onChange={(e) => setExag(+e.target.value)} />
            </div>
            <div style={row}>
              <span style={lbl}>SUN {az}° / {el}°</span>
              <input type="range" min="0" max="360" value={az}
                     onChange={(e) => setAz(+e.target.value)} />
              <input type="range" min="5" max="85" value={el}
                     onChange={(e) => setEl(+e.target.value)} />
            </div>
          </>
        )}
        <div style={{ ...row, marginLeft: 'auto' }}>
          <button onClick={save} style={pill(false)}>save png</button>
          <button onClick={onReset} style={pill(false)}>load another</button>
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', justifyContent: 'center',
                    alignItems: 'flex-start', padding: '0 18px 18px' }}>
        <canvas ref={canvasRef} style={{
          maxWidth: '100%', maxHeight: '100%', objectFit: 'contain',
          border: '1px solid rgba(255,255,255,0.1)', borderRadius: 6,
          imageRendering: 'auto',
        }} />
      </div>

      <div style={{
        padding: '10px 18px', borderTop: '1px solid rgba(255,255,255,0.08)',
        display: 'flex', gap: 22, flexWrap: 'wrap',
        fontFamily: 'ui-monospace, monospace', fontSize: 11, opacity: 0.72,
      }}>
        <span>{quantity} range {min?.toFixed(2)} – {max?.toFixed(2)}
          {quantity === 'rDSM' ? '' : ' m'}</span>
        <span>mean {mean?.toFixed(2)}</span>
        <span>{width}×{height} px</span>
        <span>{pixelSpacing?.toFixed(3)} m/px</span>
        <span>{(width * pixelSpacing)?.toFixed(0)} × {(height * pixelSpacing)?.toFixed(0)} m</span>
        {nodataPixels ? <span>nodata {nodataPixels}</span> : null}
      </div>
    </div>
  );
}
