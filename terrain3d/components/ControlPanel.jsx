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
            <span>Grid</span>
            <b>{stats.width} x {stats.height} px</b>

            <span>Ground</span>
            <b>{stats.extentX.toFixed(0)} x {stats.extentZ.toFixed(0)} m</b>

            <span>GSD</span>
            <b>{stats.pixelSpacing} m/px</b>

            {/* The raster is AGL, a DSM or an rDSM depending on how it was made, and
                they mean different things: AGL is height above the ground beneath you,
                a DSM is elevation above sea level. Labelling a 22 m ground elevation
                as "AGL" would read as a 22 m-tall street. */}
            <span>{stats.quantity} range</span>
            <b>
              {stats.min.toFixed(2)} – {stats.max.toFixed(2)}
              {stats.quantity === 'rDSM' ? '' : ' m'}
            </b>

            <span>Mean {stats.quantity}</span>
            <b>{stats.mean.toFixed(2)}{stats.quantity === 'rDSM' ? '' : ' m'}</b>

            <span>Relief</span>
            <b>
              1 : {(stats.extentX / Math.max(stats.max, 0.01)).toFixed(0)}
            </b>
          </div>

          {/* CAMERA */}

          <div className="row">
            <button
              className={mode === 'orbit' ? 'seg on' : 'seg'}
              onClick={() => setMode('orbit')}
            >
              Orbit
            </button>

            <button
              className={mode === 'fly' ? 'seg on' : 'seg'}
              onClick={() => setMode('fly')}
            >
              Fly
            </button>

            <button
              className={mode === 'walk' ? 'seg on' : 'seg'}
              onClick={() => setMode('walk')}
            >
              Walk
            </button>
          </div>

          {(mode === 'fly' || mode === 'walk') && (
            <p className="muted" style={{ marginTop: 8 }}>
              Click the scene to capture the mouse. <b>W A S D</b> move
              {mode === 'fly' ? ', ' : ' · '}
              {mode === 'fly' ? <><b>Space</b> / <b>Ctrl</b> altitude, </> : <><b>Space</b> jump · </>}
              <b>Shift</b> sprint · <b>Alt</b> slow · <b>scroll</b> speed ·{' '}
              <b>F</b> swap fly/walk · <b>R</b> recover · <b>Esc</b> release.
              The camera cannot pass below the surface.
            </p>
          )}

          {/* GEOMETRY */}

          <h2 style={{ marginTop: 20 }}>Geometry</h2>

          <label className="field">
            <span>
              Edge sharpening <b>{Math.round(sharpen * 100)}%</b>
            </span>

            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={sharpen}
              onChange={(e) => setSharpen(parseFloat(e.target.value))}
            />

            <small>
              Snaps the depth model&apos;s ~2 m edge ramps to either ground or
              roof level so walls become vertical. 0% shows the raw upstream
              raster.
            </small>
          </label>

          {/* <label className="field">
            <span>
              Vertical exaggeration <b>{exag.toFixed(2)}x</b>
            </span>

            <input
              type="range"
              min="0"
              max="2.5"
              step="0.05"
              value={exag}
              onChange={(e) => setExag(parseFloat(e.target.value))}
            />

            <small>
              Auto-set from this tile&apos;s relief. 1.00x is true metric scale.
            </small>
          </label> */}

          <label className="field">
            <span>Mesh resolution</span>

            <select
              value={segments}
              onChange={(e) =>
                setSegments(parseInt(e.target.value, 10))
              }
            >
              <option value={256}>256 (66k verts – fastest)</option>
              <option value={512}>512 (263k verts)</option>
              <option value={1024}>1024 (1.05M verts – full raster)</option>
            </select>

            <small>
              1024 keeps the narrow gaps between row houses from merging.
            </small>
          </label>

          <button className="wide primary" onClick={replay}>
            Replay 2D → 3D reveal
          </button>

          {/* SHADING */}

          <h2 style={{ marginTop: 20 }}>Shading</h2>

          <div className="featureToggles">

            <button
              type="button"
              className={`featureToggle ${flat ? 'active' : ''}`}
              onClick={() => setFlat(!flat)}
            >
              <span className="featureIcon">◇</span>
              <span className="featureLabel">
                <b>Edges</b>
                <small>Crisp walls</small>
              </span>
              <span className="featureIndicator" />
            </button>

            <button
              type="button"
              className={`featureToggle ${wireframe ? 'active' : ''}`}
              onClick={() => setWireframe(!wireframe)}
            >
              <span className="featureIcon">▦</span>
              <span className="featureLabel">
                <b>Wireframe</b>
                <small>Mesh structure</small>
              </span>
              <span className="featureIndicator" />
            </button>

            <button
              type="button"
              className={`featureToggle ${overlay ? 'active' : ''}`}
              onClick={() => setOverlay(!overlay)}
            >
              <span className="featureIcon">▧</span>
              <span className="featureLabel">
                <b>Height</b>
                <small>Alignment overlay</small>
              </span>
              <span className="featureIndicator" />
            </button>

            <button
              type="button"
              className={`featureToggle ${shadows ? 'active' : ''}`}
              onClick={() => setShadows(!shadows)}
            >
              <span className="featureIcon">◒</span>
              <span className="featureLabel">
                <b>Shadows</b>
                <small>Sun lighting</small>
              </span>
              <span className="featureIndicator" />
            </button>

          </div>

          {overlay && (
            <p className="muted">
              Every bright rectangle must sit exactly on a raised block. If
              they are offset or inverted, the texture orientation is wrong —
              not the geometry.
            </p>
          )}

          <label className="field">
            <span>Surface</span>

            <select
              value={shading}
              onChange={(e) => setShading(e.target.value)}
            >
              <option value="texture">RGB imagery</option>
              <option value="height">Height colour ramp</option>
            </select>
          </label>

          {/* LIGHTING */}

          <label className="field">
            <span>
              Sun azimuth <b>{sunAz}°</b>
            </span>

            <input
              type="range"
              min="0"
              max="360"
              step="1"
              value={sunAz}
              onChange={(e) =>
                setSunAz(parseInt(e.target.value, 10))
              }
            />
          </label>

          <label className="field">
            <span>
              Sun elevation <b>{sunEl}°</b>
            </span>

            <input
              type="range"
              min="8"
              max="85"
              step="1"
              value={sunEl}
              onChange={(e) =>
                setSunEl(parseInt(e.target.value, 10))
              }
            />
          </label>

          {/* BOTTOM ACTIONS */}

          <div className="row">
            <button className="seg" onClick={snapshot}>
              Save PNG
            </button>

            <button className="seg" onClick={onReset}>
              New scene
            </button>
          </div>

        </div>
      )}
    </div>
  );
}