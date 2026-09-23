'use client';

import { useState } from 'react';
import Icon from './Icons';
import { BrandMark } from './Brand';

/* ------------------------------------------------------------------ controls */

function Section({ title, children }) {
  return (
    <section className="sec">
      <h3 className="secTitle">{title}</h3>
      {children}
    </section>
  );
}

/**
 * Segmented control with a sliding indicator. The indicator is positioned from
 * the selected index, so switching animates rather than jumping.
 */
function Segmented({ value, options, onChange, label }) {
  const idx = Math.max(0, options.findIndex((o) => o.value === value));
  return (
    <div
      className="segd"
      role="radiogroup"
      aria-label={label}
      style={{ '--n': options.length, '--i': idx }}
    >
      <span className="segdThumb" aria-hidden="true" />
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          className={o.value === value ? 'on' : ''}
          onClick={() => onChange(o.value)}
          title={o.title}
        >
          {o.icon && <Icon name={o.icon} size={14} />}
          <span>{o.label}</span>
        </button>
      ))}
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, format }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <label className="ctl">
      <span className="ctlRow">
        <span>{label}</span>
        <output>{format(value)}</output>
      </span>
      <input
        className="range"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ '--p': `${pct}%` }}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

function Switch({ label, hint, checked, onChange }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      className="swRow"
      onClick={() => onChange(!checked)}
    >
      <span className="swText">
        <span>{label}</span>
        {hint && <small>{hint}</small>}
      </span>
      <span className={`sw ${checked ? 'on' : ''}`} aria-hidden="true"><i /></span>
    </button>
  );
}

const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '—');

/* --------------------------------------------------------------------- panel */

export default function ControlPanel(props) {
  const {
    mode, setMode, segments, setSegments,
    sharpen, setSharpen, flat, setFlat, overlay, setOverlay,
    shading, setShading, wireframe, setWireframe,
    shadows, setShadows, sunAz, setSunAz, sunEl, setSunEl,
    replay, snapshot, onReset, stats, sceneName,
  } = props;

  const [open, setOpen] = useState(true);

  const q = stats.quantity;
  const unit = q === 'rDSM' ? '' : ' m';
  // An AGL raster is height above ground; a DSM is elevation above the datum.
  // Labelling a 434 m elevation as a "height" reads as a 434 m tower.
  const rangeLabel = q === 'DSM' ? 'Elevation' : q === 'rDSM' ? 'Relative' : 'Height';

  return (
    <aside className={`side ${open ? '' : 'isCollapsed'}`} aria-label="Scene controls">
      <header className="sideHead">
        <button
          type="button"
          className="sideBrand"
          onClick={() => !open && setOpen(true)}
          tabIndex={open ? -1 : 0}
          aria-label={open ? undefined : 'Expand controls'}
        >
          <BrandMark size={20} />
          <span className="sideBrandText">DepthWizard</span>
        </button>
        <button
          type="button"
          className="iconBtn sideToggle"
          onClick={() => setOpen((o) => !o)}
          aria-label={open ? 'Collapse controls' : 'Expand controls'}
          title={open ? 'Collapse' : 'Expand'}
        >
          <Icon name={open ? 'chevronLeft' : 'chevronRight'} size={15} />
        </button>
      </header>

      <div className="sideBody" aria-hidden={!open}>
        <div className="sceneId">
          <div className="sceneName" title={sceneName}>{sceneName || 'Reconstructed scene'}</div>
          <div className="sceneTags">
            <span className="tag tagAccent">{q}</span>
            {stats.georeferenced
              ? <span className="tag">{stats.crs || 'Georeferenced'}</span>
              : <span className="tag tagMuted">Not georeferenced</span>}
          </div>
        </div>

        <Section title="Scene">
          <dl className="kv">
            <dt>Raster</dt><dd>{stats.width} × {stats.height}<small> px</small></dd>
            <dt>Extent</dt><dd>{fmt(stats.extentX, 0)} × {fmt(stats.extentZ, 0)}<small> m</small></dd>
            <dt>GSD</dt><dd>{fmt(stats.pixelSpacing, 3)}<small> m/px</small></dd>
            <dt>{rangeLabel}</dt>
            <dd>{fmt(stats.min)} – {fmt(stats.max)}<small>{unit}</small></dd>
            <dt>Relief</dt><dd>{fmt(stats.relief)}<small>{unit}</small></dd>
            <dt>Mean</dt><dd>{fmt(stats.mean)}<small>{unit}</small></dd>
            {stats.nodataPixels > 0 && (
              <><dt>No-data</dt><dd>{stats.nodataPixels.toLocaleString()}<small> px</small></dd></>
            )}
          </dl>
        </Section>

        <Section title="Navigation">
          <Segmented
            label="Camera mode"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'orbit', label: 'Orbit', icon: 'orbit' },
              { value: 'fly', label: 'Fly', icon: 'fly' },
              { value: 'walk', label: 'Walk', icon: 'walk' },
            ]}
          />
          <p className="hint">
            {mode === 'orbit' ? (
              <>Drag to orbit · scroll to zoom · right-drag to pan.</>
            ) : (
              <>
                Click the scene to capture the mouse. <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move
                {mode === 'fly'
                  ? <> · <kbd>Space</kbd>/<kbd>Ctrl</kbd> altitude</>
                  : <> · <kbd>Space</kbd> jump</>}
                {' '}· <kbd>Shift</kbd> sprint · <kbd>F</kbd> fly/walk · <kbd>R</kbd> recover · <kbd>Esc</kbd> release.
              </>
            )}
          </p>
        </Section>

        <Section title="Geometry">
          <Slider
            label="Edge sharpening"
            value={sharpen}
            min={0}
            max={1}
            step={0.05}
            onChange={setSharpen}
            format={(v) => `${Math.round(v * 100)}%`}
          />
          <div className="ctl">
            <span className="ctlRow"><span>Mesh resolution</span></span>
            <Segmented
              label="Mesh resolution"
              value={segments}
              onChange={setSegments}
              options={[
                { value: 256, label: '256', title: '66k vertices — fastest' },
                { value: 512, label: '512', title: '263k vertices' },
                { value: 1024, label: '1024', title: '1.05M vertices — full detail' },
              ]}
            />
          </div>
        </Section>

        <Section title="Shading">
          <div className="ctl">
            <span className="ctlRow"><span>Surface</span></span>
            <Segmented
              label="Surface"
              value={shading}
              onChange={setShading}
              options={[
                { value: 'texture', label: 'Imagery' },
                { value: 'height', label: 'Height ramp' },
              ]}
            />
          </div>
          <div className="swList">
            <Switch label="Crisp walls" hint="Faceted normals" checked={flat} onChange={setFlat} />
            <Switch label="Shadows" hint="Directional sun" checked={shadows} onChange={setShadows} />
            <Switch label="Wireframe" hint="Mesh structure" checked={wireframe} onChange={setWireframe} />
            <Switch label="Alignment check" hint="Height as texture" checked={overlay} onChange={setOverlay} />
          </div>
          {overlay && (
            <p className="hint">
              Every bright block should sit exactly on a raised block. If it is offset,
              the texture orientation is wrong — not the geometry.
            </p>
          )}
        </Section>

        <Section title="Lighting">
          <Slider
            label="Sun azimuth"
            value={sunAz}
            min={0}
            max={360}
            step={1}
            onChange={(v) => setSunAz(Math.round(v))}
            format={(v) => `${Math.round(v)}°`}
          />
          <Slider
            label="Sun elevation"
            value={sunEl}
            min={8}
            max={85}
            step={1}
            onChange={(v) => setSunEl(Math.round(v))}
            format={(v) => `${Math.round(v)}°`}
          />
        </Section>
      </div>

      <footer className="sideFoot" aria-hidden={!open}>
        <div className="btnRow">
          <button type="button" className="btn" onClick={replay}>
            <Icon name="replay" size={14} />Replay reveal
          </button>
          <button type="button" className="btn" onClick={snapshot}>
            <Icon name="download" size={14} />PNG
          </button>
        </div>
        <button type="button" className="btn btnGhost" onClick={onReset}>
          <Icon name="back" size={14} />New scene
        </button>
      </footer>
    </aside>
  );
}
