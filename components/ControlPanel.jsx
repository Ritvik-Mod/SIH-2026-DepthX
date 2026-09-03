'use client';

import { useState } from 'react';

export default function ControlPanel(props) {
  const {
    mode, setMode, exag, setExag, segments, setSegments,
    sharpen, setSharpen, flat, setFlat, overlay, setOverlay,
    shading, setShading, wireframe, setWireframe,
    shadows, setShadows, sunAz, setSunAz, sunEl, setSunEl,
    replay, snapshot, onReset, stats,
  } = props;

  const [open, setOpen] = useState(true);

  return (
    <div className={`panel ${open ? '' : 'collapsed'}`}>
      <button className="panelToggle" onClick={() => setOpen(!open)}>
        {open ? '‹' : '›'}
      </button>

      {open && (
        <div className="panelBody">
          <h2>Reconstructed scene</h2>
          <div className="statGrid">
            <span>Grid</span><b>{stats.width} x {stats.height} px</b>
            <span>Ground</span><b>{stats.extentX.toFixed(0)} x {stats.extentZ.toFixed(0)} m</b>
            <span>GSD</span><b>{stats.pixelSpacing} m/px</b>
            <span>AGL range</span><b>{stats.min.toFixed(2)} – {stats.max.toFixed(2)} m</b>
            <span>Mean AGL</span><b>{stats.mean.toFixed(2)} m</b>
            <span>Relief</span><b>1 : {(stats.extentX / Math.max(stats.max, 0.01)).toFixed(0)}</b>
          </div>

          <div className="row">
            <button className={mode === 'orbit' ? 'seg on' : 'seg'} onClick={() => setMode('orbit')}>
              Orbit
            </button>
            <button className={mode === 'fly' ? 'seg on' : 'seg'} onClick={() => setMode('fly')}>
              Fly (WASD)
            </button>
          </div>

          <h2 style={{ marginTop: 20 }}>Geometry</h2>

          <label className="field">
            <span>Edge sharpening <b>{Math.round(sharpen * 100)}%</b></span>
            <input
              type="range" min="0" max="1" step="0.05"
              value={sharpen} onChange={(e) => setSharpen(parseFloat(e.target.value))}
            />
            <small>
              Snaps the depth model&apos;s ~2 m edge ramps to either ground or roof level so
              walls become vertical. 0% shows the raw upstream raster.
            </small>
          </label>

          <label className="field">
            <span>Vertical exaggeration <b>{exag.toFixed(2)}x</b></span>
            <input
              type="range" min="0" max="2.5" step="0.05"
              value={exag} onChange={(e) => setExag(parseFloat(e.target.value))}
            />
            <small>Auto-set from this tile&apos;s relief. 1.00x is true metric scale.</small>
          </label>

          <label className="field">
            <span>Mesh resolution</span>
            <select value={segments} onChange={(e) => setSegments(parseInt(e.target.value, 10))}>
              <option value={256}>256 (66k verts – fastest)</option>
              <option value={512}>512 (263k verts)</option>
              <option value={1024}>1024 (1.05M verts – full raster)</option>
            </select>
            <small>1024 keeps the narrow gaps between row houses from merging.</small>
          </label>

          <button className="wide primary" onClick={replay}>Replay 2D → 3D reveal</button>

          <h2 style={{ marginTop: 20 }}>Shading</h2>

          <label className="check">
            <input type="checkbox" checked={flat} onChange={(e) => setFlat(e.target.checked)} />
            <span>Faceted shading (crisp walls)</span>
          </label>

          <label className="field">
            <span>Surface</span>
            <select value={shading} onChange={(e) => setShading(e.target.value)}>
              <option value="texture">RGB imagery</option>
              <option value="height">Height colour ramp</option>
            </select>
          </label>

          <label className="check">
            <input type="checkbox" checked={wireframe} onChange={(e) => setWireframe(e.target.checked)} />
            <span>Wireframe</span>
          </label>

          <label className="check">
            <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} />
            <span>Height map overlay (alignment check)</span>
          </label>
          {overlay && (
            <p className="muted">
              Every bright rectangle must sit exactly on a raised block. If they are offset or
              inverted, the texture orientation is wrong — not the geometry.
            </p>
          )}

          <label className="check">
            <input type="checkbox" checked={shadows} onChange={(e) => setShadows(e.target.checked)} />
            <span>Sun shadows</span>
          </label>

          <label className="field">
            <span>Sun azimuth <b>{sunAz}°</b></span>
            <input type="range" min="0" max="360" step="1" value={sunAz}
              onChange={(e) => setSunAz(parseInt(e.target.value, 10))} />
          </label>

          <label className="field">
            <span>Sun elevation <b>{sunEl}°</b></span>
            <input type="range" min="8" max="85" step="1" value={sunEl}
              onChange={(e) => setSunEl(parseInt(e.target.value, 10))} />
          </label>

          <div className="row">
            <button className="seg" onClick={snapshot}>Save PNG</button>
            <button className="seg" onClick={onReset}>New scene</button>
          </div>
        </div>
      )}
    </div>
  );
}