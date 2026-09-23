'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Icon from './Icons';

/* ==========================================================================
 * The source image, always on screen.
 *
 * A minimap in the corner carries the camera's position and heading, so the 3D
 * view can always be related back to the photograph it came from. Clicking it
 * opens a comparison view: the photo, the reconstructed height field, or a
 * swipe between the two, with zoom, pan, a scale bar and a live height readout
 * under the cursor.
 * ========================================================================== */

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// Browsers refuse canvases much past ~16.7 Mpx (Safari especially). Anything
// larger is displayed from a downscaled copy; coordinates stay in raster units.
const MAX_CANVAS_PX = 16e6;
const displayScale = (w, h) => Math.min(1, Math.sqrt(MAX_CANVAS_PX / (w * h)));

/**
 * The texture arrives as { source, flipY }. flipY === false means
 * createImageBitmap already flipped it for WebGL's texture convention, so a 2D
 * canvas has to flip it back or the minimap shows the scene upside down --
 * the same rule Fallback2D follows.
 */
function usePhotoUrl(bitmap) {
  const [url, setUrl] = useState(null);
  useEffect(() => {
    const src = bitmap?.source;
    if (!src) return undefined;
    let made = null;
    let cancelled = false;
    // Deferred so the 3D reveal gets the first frames, not a JPEG encoder.
    const id = setTimeout(() => {
      const w = src.width || src.naturalWidth;
      const h = src.height || src.naturalHeight;
      if (!w || !h) return;
      const k = displayScale(w, h);
      const c = document.createElement('canvas');
      c.width = Math.round(w * k);
      c.height = Math.round(h * k);
      const ctx = c.getContext('2d');
      if (!ctx) return;
      if (bitmap.flipY === false) {
        ctx.translate(0, c.height);
        ctx.scale(1, -1);
      }
      ctx.drawImage(src, 0, 0, c.width, c.height);
      c.toBlob((blob) => {
        if (cancelled || !blob) return;
        made = URL.createObjectURL(blob);
        setUrl(made);
      }, 'image/jpeg', 0.9);
    }, 300);
    return () => {
      cancelled = true;
      clearTimeout(id);
      if (made) URL.revokeObjectURL(made);
    };
  }, [bitmap]);
  return url;
}

/* --------------------------------------------------------------- height map */

const toSrgb = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);

// The same polynomial as the terrain shader's height ramp (tViridis), encoded
// to sRGB the way the renderer encodes it, so the two views use one palette.
const LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const r = 0.280 + t * (0.105 + t * (-0.330 + t * 1.900));
    const g = 0.010 + t * (1.410 + t * (-1.010 + t * 0.480));
    const b = 0.330 + t * (1.380 + t * (-3.320 + t * 2.180));
    lut[i * 3] = 255 * toSrgb(clamp(r, 0, 1));
    lut[i * 3 + 1] = 255 * toSrgb(clamp(g, 0, 1));
    lut[i * 3 + 2] = 255 * toSrgb(clamp(b, 0, 1));
  }
  return lut;
})();

const LEGEND_CSS = (() => {
  const stops = [];
  for (let i = 0; i <= 8; i++) {
    const k = Math.round((i / 8) * 255) * 3;
    stops.push(`rgb(${LUT[k]},${LUT[k + 1]},${LUT[k + 2]}) ${(i / 8) * 100}%`);
  }
  return `linear-gradient(90deg, ${stops.join(', ')})`;
})();

/** 2nd-98th percentile, so one mast or one pit does not flatten the colours. */
function percentileRange(heights) {
  const n = heights.length;
  const step = Math.max(1, Math.floor(n / 200000));
  const a = [];
  for (let i = 0; i < n; i += step) if (Number.isFinite(heights[i])) a.push(heights[i]);
  a.sort((x, y) => x - y);
  if (!a.length) return [0, 1];
  const lo = a[Math.floor(a.length * 0.02)];
  const hi = a[Math.min(a.length - 1, Math.floor(a.length * 0.98))];
  return hi - lo > 1e-6 ? [lo, hi] : [a[0], a[a.length - 1] + 1e-3];
}

function renderHeightImage(heights, w, h, [lo, hi]) {
  return new Promise((resolve) => {
    const k = displayScale(w, h);
    const cw = Math.round(w * k);
    const ch = Math.round(h * k);
    const c = document.createElement('canvas');
    c.width = cw;
    c.height = ch;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(cw, ch);
    const d = img.data;
    const inv = 255 / (hi - lo);
    for (let y = 0; y < ch; y++) {
      const sy = Math.min(h - 1, Math.floor(y / k)) * w;
      for (let x = 0; x < cw; x++) {
        const v = heights[sy + Math.min(w - 1, Math.floor(x / k))];
        const li = clamp(Math.round((v - lo) * inv), 0, 255) * 3;
        const o = (y * cw + x) * 4;
        d[o] = LUT[li];
        d[o + 1] = LUT[li + 1];
        d[o + 2] = LUT[li + 2];
        d[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    c.toBlob(resolve, 'image/png');
  });
}

/* ------------------------------------------------------------ camera marker */

function camUV(cam, extentX, extentZ) {
  // Raster row 0 is world -Z (see sampleGround), and the image draws row 0 at
  // the top, so +Z is down the picture and +X is to the right.
  return { u: cam.x / extentX + 0.5, v: cam.z / extentZ + 0.5 };
}

function CameraGlyph({ heading = 0 }) {
  return (
    <svg className="camGlyph" width="36" height="36" viewBox="-18 -18 36 36"
         style={{ transform: `rotate(${heading}rad)` }} aria-hidden="true">
      <path d="M0 0 L16 -9 A 18.5 18.5 0 0 1 16 9 Z" className="camFov" />
      <circle r="3.6" className="camDot" />
    </svg>
  );
}

/* -------------------------------------------------------------------- modal */

function niceLength(m) {
  const p = Math.pow(10, Math.floor(Math.log10(m)));
  const f = m / p;
  return (f >= 5 ? 5 : f >= 2 ? 2 : 1) * p;
}

function SourceModal({
  photoUrl, heights, width, height, extentX, extentZ, camera, name, quantity, units, onClose,
}) {
  const [view, setView] = useState('photo');
  const [heightUrl, setHeightUrl] = useState(null);
  const [range, setRange] = useState(null);
  const [t, setT] = useState({ s: 1, x: 0, y: 0 });
  const [fitS, setFitS] = useState(1);
  const [stage, setStage] = useState({ w: 1, h: 1 });
  const [swipe, setSwipe] = useState(0.5);
  const [hover, setHover] = useState(null);
  const [animate, setAnimate] = useState(false);
  const stageRef = useRef(null);
  const drag = useRef(null);

  const hLabel = quantity === 'DSM' ? 'Elevation' : 'Height';

  const fit = useCallback((anim = true) => {
    const el = stageRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const s = Math.min(r.width / width, r.height / height) * 0.94;
    setStage({ w: r.width, h: r.height });
    setFitS(s);
    setAnimate(anim);
    setT({ s, x: (r.width - width * s) / 2, y: (r.height - height * s) / 2 });
  }, [width, height]);

  useEffect(() => {
    fit(false);
    const onResize = () => fit(false);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [fit]);

  // The height image is only built if someone asks for it.
  useEffect(() => {
    if (view === 'photo' || heightUrl) return undefined;
    let cancelled = false;
    const id = setTimeout(async () => {
      const r = percentileRange(heights);
      const blob = await renderHeightImage(heights, width, height, r);
      if (cancelled || !blob) return;
      setRange(r);
      setHeightUrl(URL.createObjectURL(blob));
    }, 30);
    return () => { cancelled = true; clearTimeout(id); };
  }, [view, heightUrl, heights, width, height]);

  useEffect(() => () => { if (heightUrl) URL.revokeObjectURL(heightUrl); }, [heightUrl]);

  const zoomAt = useCallback((factor, cx, cy, anim = false) => {
    setAnimate(anim);
    setT((p) => {
      const s = clamp(p.s * factor, fitS * 0.5, 64);
      const k = s / p.s;
      return { s, x: cx - (cx - p.x) * k, y: cy - (cy - p.y) * k };
    });
  }, [fitS]);

  const zoomCentre = (factor) => zoomAt(factor, stage.w / 2, stage.h / 2, true);

  const oneToOne = () => {
    setAnimate(true);
    setT((p) => {
      const cx = stage.w / 2;
      const cy = stage.h / 2;
      const k = 1 / p.s;
      return { s: 1, x: cx - (cx - p.x) * k, y: cy - (cy - p.y) * k };
    });
  };

  // Wheel has to be non-passive to stop the page scrolling underneath.
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return undefined;
    const onWheel = (e) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * 0.0016), e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [zoomAt]);

  // Keys are taken in the CAPTURE phase so the scene's own shortcuts -- the
  // number keys that toggle layers -- cannot fire behind an open modal.
  useEffect(() => {
    const onKey = (e) => {
      const mine = ['Escape', 'Equal', 'NumpadAdd', 'Minus', 'NumpadSubtract', 'Digit0',
                    'Digit1', 'Digit2', 'Digit3', 'Digit4', 'Digit5'];
      if (!mine.includes(e.code)) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      if (e.code === 'Escape') onClose();
      else if (e.code === 'Equal' || e.code === 'NumpadAdd') zoomCentre(1.5);
      else if (e.code === 'Minus' || e.code === 'NumpadSubtract') zoomCentre(1 / 1.5);
      else if (e.code === 'Digit0') fit(true);
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  });

  const readAt = (clientX, clientY) => {
    const r = stageRef.current.getBoundingClientRect();
    const ix = (clientX - r.left - t.x) / t.s;
    const iy = (clientY - r.top - t.y) / t.s;
    if (ix < 0 || iy < 0 || ix >= width || iy >= height) return null;
    const h = heights[Math.floor(iy) * width + Math.floor(ix)];
    return {
      x: (ix / width - 0.5) * extentX,
      z: (iy / height - 0.5) * extentZ,
      h,
    };
  };

  const onPointerDown = (e) => {
    if (e.button !== 0) return;
    const r = stageRef.current.getBoundingClientRect();
    const onKnob = e.target.closest?.('.swipeKnob');
    drag.current = onKnob
      ? { kind: 'swipe', left: r.left, w: r.width }
      : { kind: 'pan', x: e.clientX, y: e.clientY, tx: t.x, ty: t.y };
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e) => {
    const d = drag.current;
    if (d?.kind === 'pan') {
      setAnimate(false);
      setT((p) => ({ ...p, x: d.tx + e.clientX - d.x, y: d.ty + e.clientY - d.y }));
    } else if (d?.kind === 'swipe') {
      setSwipe(clamp((e.clientX - d.left) / d.w, 0, 1));
    }
    setHover(readAt(e.clientX, e.clientY));
  };

  const onPointerUp = () => { drag.current = null; };

  const onDoubleClick = (e) => {
    const r = stageRef.current.getBoundingClientRect();
    zoomAt(2, e.clientX - r.left, e.clientY - r.top, true);
  };

  // Scale bar: a round number of metres that spans roughly 110 screen pixels.
  const mPerPx = extentX / width / t.s;
  const barM = niceLength(110 * mPerPx);
  const barPx = barM / mPerPx;

  const { u, v } = camUV(camera, extentX, extentZ);
  const camIn = u >= 0 && u <= 1 && v >= 0 && v <= 1;
  const clipPx = clamp((swipe * stage.w - t.x) / t.s, 0, width);

  const layerStyle = {
    width, height,
    transform: `translate(${t.x}px, ${t.y}px) scale(${t.s})`,
  };

  return (
    <div className="scrim" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-label="Source image comparison">
        <header className="modalHead">
          <div className="modalTitle">
            <Icon name="image" size={15} />
            <b>{name || 'Source image'}</b>
            <span>{width} × {height} px · {extentX.toFixed(0)} × {extentZ.toFixed(0)} m</span>
          </div>

          <div className="tabs" role="tablist" aria-label="View">
            {[['photo', 'Photo'], ['height', 'Height map'], ['swipe', 'Swipe compare']].map(([k, l]) => (
              <button key={k} type="button" role="tab" aria-selected={view === k}
                      className={view === k ? 'on' : ''} onClick={() => setView(k)}>{l}</button>
            ))}
          </div>

          <div className="zoomCtl">
            <button type="button" className="iconBtn" onClick={() => zoomCentre(1 / 1.5)}
                    title="Zoom out  ( − )" aria-label="Zoom out"><Icon name="minus" size={14} /></button>
            <output title="Zoom relative to fit">{Math.round((t.s / fitS) * 100)}%</output>
            <button type="button" className="iconBtn" onClick={() => zoomCentre(1.5)}
                    title="Zoom in  ( + )" aria-label="Zoom in"><Icon name="plus" size={14} /></button>
            <button type="button" className="iconBtn" onClick={() => fit(true)}
                    title="Fit  ( 0 )" aria-label="Fit to window"><Icon name="fit" size={14} /></button>
            <button type="button" className="textBtn" onClick={oneToOne}
                    title="One image pixel per screen pixel">1:1</button>
          </div>

          <button type="button" className="iconBtn modalClose" onClick={onClose}
                  aria-label="Close  ( Esc )" title="Close  ( Esc )"><Icon name="close" size={15} /></button>
        </header>

        <div
          className={`stage ${drag.current?.kind === 'pan' ? 'grabbing' : ''}`}
          ref={stageRef}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onPointerLeave={() => setHover(null)}
          onDoubleClick={onDoubleClick}
        >
          <div className={`stageLayer ${animate ? 'animate' : ''} ${t.s > 2.5 ? 'pixelated' : ''}`}
               style={layerStyle}>
            {view !== 'height' && photoUrl && (
              <img src={photoUrl} alt="" draggable={false} style={{ width, height }} />
            )}
            {view !== 'photo' && heightUrl && (
              <img
                src={heightUrl}
                alt=""
                draggable={false}
                style={{ width, height, clipPath: view === 'swipe' ? `inset(0 0 0 ${clipPx}px)` : undefined }}
              />
            )}
          </div>

          {view !== 'photo' && !heightUrl && <div className="stageNote">Rendering height map…</div>}
          {view !== 'height' && !photoUrl && <div className="stageNote">Preparing image…</div>}

          {view === 'swipe' && (
            <div className="swipeLine" style={{ left: swipe * stage.w }}>
              <span className="swipeTag swipeTagL">Photo</span>
              <span className="swipeTag swipeTagR">Height</span>
              <span className="swipeKnob" aria-label="Drag to compare"><Icon name="chevronLeft" size={12} /><Icon name="chevronRight" size={12} /></span>
            </div>
          )}

          {camIn && (
            <span className="camMark" style={{ left: t.x + u * width * t.s, top: t.y + v * height * t.s }}
                  title="Current 3D camera position">
              <CameraGlyph heading={camera.heading} />
            </span>
          )}

          <div className="scaleBar" aria-hidden="true">
            <i style={{ width: barPx }} />
            <span>{barM >= 1000 ? `${barM / 1000} km` : `${barM} m`}</span>
          </div>
        </div>

        <footer className="modalFoot">
          <span className="readout">
            {hover ? (
              <>
                <span>X <b>{hover.x.toFixed(1)}</b> m</span>
                <span>Z <b>{hover.z.toFixed(1)}</b> m</span>
                <span>{hLabel} <b>{Number.isFinite(hover.h) ? hover.h.toFixed(2) : '—'}</b> {units}</span>
              </>
            ) : (
              <span className="muted">Scroll to zoom · drag to pan · double-click to zoom in · hover to read {hLabel.toLowerCase()}</span>
            )}
          </span>
          {view !== 'photo' && range && (
            <span className="legend">
              <span>{range[0].toFixed(1)}</span>
              <i style={{ background: LEGEND_CSS }} />
              <span>{range[1].toFixed(1)} {units}</span>
            </span>
          )}
        </footer>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ minimap */

export default function SourceImage({
  bitmap, heights, width, height, extentX, extentZ, camera, name, quantity, units = 'm',
  onOpenChange,
}) {
  const photoUrl = usePhotoUrl(bitmap);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    onOpenChange?.(open);
    return () => { if (open) onOpenChange?.(false); };
  }, [open, onOpenChange]);

  const side = 172;
  const aspect = width / height;
  const boxW = aspect >= 1 ? side : Math.round(side * aspect);
  const boxH = aspect >= 1 ? Math.round(side / aspect) : side;

  const { u, v } = camUV(camera, extentX, extentZ);
  const inside = u >= 0 && u <= 1 && v >= 0 && v <= 1;

  return (
    <>
      <button
        type="button"
        className="minimap"
        style={{ width: boxW, height: boxH }}
        onClick={() => setOpen(true)}
        aria-label="Open the source image to compare"
        title="Compare with the source image"
      >
        {photoUrl ? <img src={photoUrl} alt="" draggable={false} /> : <span className="minimapWait" />}
        <span
          className={`camMark ${inside ? '' : 'isOut'}`}
          style={{ left: `${clamp(u, 0, 1) * 100}%`, top: `${clamp(v, 0, 1) * 100}%` }}
        >
          <CameraGlyph heading={camera.heading} />
        </span>
        <span className="minimapBar">
          <span><Icon name="image" size={12} />Source</span>
          <Icon name="expand" size={12} />
        </span>
      </button>

      {open && (
        <SourceModal
          {...{ photoUrl, heights, width, height, extentX, extentZ, camera, name, quantity, units }}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}
