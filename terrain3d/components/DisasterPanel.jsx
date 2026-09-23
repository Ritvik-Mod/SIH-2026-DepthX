'use client';

import { useMemo } from 'react';
import {
  CYCLONE_CATEGORIES,
  GUST_FACTOR,
  SCENARIOS,
  hollandWind,
  mmiFromPga,
  pgaFromMagnitude,
  romanMmi,
} from '@/lib/disaster';

/**
 * Controls and impact readout for the disaster simulation.
 *
 * Two things this panel is deliberately strict about.
 *
 * It always shows the DERIVED physical quantity next to the input, not just
 * the input: magnitude alone tells you nothing about a site, so the panel
 * shows the PGA and the Mercalli intensity that magnitude and distance
 * actually produce here. "Category 4" likewise means nothing without a
 * distance from the eye, so the Holland profile result is on screen beside it.
 *
 * And it states what the model does not include. A tool that renders a city
 * falling down has an obligation not to look more authoritative than it is.
 */
export default function DisasterPanel({
  sim,
  setSim,
  ready,
  busy,
  impact,
  progress,
  playing,
  onRun,
  onReset,
  onClose,
}) {
  const set = (patch) => setSim((s) => ({ ...s, ...patch }));

  /* Live derived quantities, so the sliders read as physics not as numbers. */
  const derived = useMemo(() => {
    if (sim.kind === 'earthquake') {
      const pga = pgaFromMagnitude(sim.magnitude, sim.distanceKm);
      return { pga, mmi: mmiFromPga(pga) };
    }
    if (sim.kind === 'cyclone') {
      const cat = CYCLONE_CATEGORIES.find((c) => c.cat === sim.category) || CYCLONE_CATEGORIES[2];
      const sustained = hollandWind(cat.windMs, sim.eyeKm);
      return {
        sustained,
        gust: sustained * GUST_FACTOR,
        surge: (cat.surge[0] + cat.surge[1]) / 2,
        cat,
      };
    }
    return {};
  }, [sim]);

  const pick = (kind) => {
    if (sim.kind === kind) return;
    set({ kind, ...SCENARIOS[kind].defaults });
  };

  return (
    <div className="simPanel">
      <div className="simHead">
        <b>Disaster simulation</b>
        <button className="simX" onClick={onClose} title="Close and restore the scene">
          ×
        </button>
      </div>

      <div className="simPicker">
        {Object.entries(SCENARIOS).map(([key, s]) => (
          <button
            key={key}
            className={`simCard ${sim.kind === key ? 'active' : ''}`}
            onClick={() => pick(key)}
            disabled={!ready}
          >
            <span className="simIcon">{s.icon}</span>
            <span>{s.label}</span>
          </button>
        ))}
      </div>

      {!ready && (
        <p className="muted simNote">
          Waiting for the scene analysis — the simulation works on detected
          buildings, not on pixels, so it needs the footprints first.
        </p>
      )}

      {/* ------------------------------------------------------ earthquake */}
      {ready && sim.kind === 'earthquake' && (
        <>
          <label className="field">
            <span>
              Moment magnitude <b>M<sub>w</sub> {sim.magnitude.toFixed(1)}</b>
            </span>
            <input
              type="range" min="4.5" max="8.2" step="0.1" value={sim.magnitude}
              onChange={(e) => set({ magnitude: parseFloat(e.target.value) })}
            />
          </label>

          <label className="field">
            <span>
              Epicentral distance <b>{sim.distanceKm} km</b>
            </span>
            <input
              type="range" min="2" max="120" step="1" value={sim.distanceKm}
              onChange={(e) => set({ distanceKm: parseInt(e.target.value, 10) })}
            />
          </label>

          <div className="simDerived">
            <div><span>PGA</span><b>{derived.pga?.toFixed(3)} g</b></div>
            <div><span>Intensity</span><b>MMI {romanMmi(derived.mmi)}</b></div>
          </div>

          <p className="muted simNote">
            Joyner &amp; Boore (1981) attenuation, Wald et&nbsp;al. (1999) intensity,
            HAZUS-style fragility by storey count. No soil class, so no site
            amplification and no liquefaction.
          </p>
        </>
      )}

      {/* ----------------------------------------------------------- flood */}
      {ready && sim.kind === 'flood' && (
        <>
          <label className="field">
            <span>
              Water stage <b>{sim.stage.toFixed(1)} m</b>
            </span>
            <input
              type="range" min="0.2" max="20" step="0.1" value={sim.stage}
              onChange={(e) => set({ stage: parseFloat(e.target.value) })}
            />
            <small>
              Metres above the scene datum. Water only reaches ground it is
              hydraulically connected to — an enclosed courtyard stays dry
              until the stage tops the buildings around it.
            </small>
          </label>

          <p className="muted simNote">
            Priority-Flood connectivity on the reconstructed surface. A level
            pool, not a routed flow: no velocity, no timing, no rainfall.
          </p>
        </>
      )}

      {/* --------------------------------------------------------- cyclone */}
      {ready && sim.kind === 'cyclone' && (
        <>
          <label className="field">
            <span>
              Saffir-Simpson <b>{derived.cat?.label}</b>
            </span>
            <input
              type="range" min="1" max="5" step="1" value={sim.category}
              onChange={(e) => set({ category: parseInt(e.target.value, 10) })}
            />
          </label>

          <label className="field">
            <span>
              Distance from the eye <b>{sim.eyeKm} km</b>
            </span>
            <input
              type="range" min="2" max="160" step="1" value={sim.eyeKm}
              onChange={(e) => set({ eyeKm: parseInt(e.target.value, 10) })}
            />
            <small>
              Holland (1980) radial profile, R<sub>max</sub> 35 km. A category 5
              eyewall 120 km away is a windy afternoon, and the slider shows it.
            </small>
          </label>

          <label className="field">
            <span>
              Wind bearing <b>{sim.windDirDeg}°</b>
            </span>
            <input
              type="range" min="0" max="359" step="1" value={sim.windDirDeg}
              onChange={(e) => set({ windDirDeg: parseInt(e.target.value, 10) })}
            />
          </label>

          <label className="simCheck">
            <input
              type="checkbox" checked={sim.surge}
              onChange={(e) => set({ surge: e.target.checked })}
            />
            <span>
              Storm surge <span className="muted">— {derived.surge?.toFixed(1)} m,
              the historical band for this category</span>
            </span>
          </label>

          <div className="simDerived">
            <div><span>Sustained</span><b>{derived.sustained?.toFixed(0)} m/s</b></div>
            <div><span>3-s gust</span><b>{derived.gust?.toFixed(0)} m/s</b></div>
          </div>
        </>
      )}

      {/* ------------------------------------------------------- transport */}
      {ready && sim.kind && (
        <>
          <div className="row">
            <button className="seg on" onClick={onRun} disabled={busy}>
              {busy ? 'Computing…' : playing ? 'Restart' : 'Run event'}
            </button>
            <button className="seg" onClick={onReset} disabled={busy}>
              Reset
            </button>
          </div>

          <div className="simBar">
            <div className="simBarFill" style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
        </>
      )}

      {/* ---------------------------------------------------------- impact */}
      {impact && (
        <div className="simImpact">
          <h3>Impact</h3>

          {impact.kind === 'earthquake' && (
            <>
              <div className="simStat"><span>Shaking</span><b>{impact.pga.toFixed(3)} g · MMI {romanMmi(impact.mmi)}</b></div>
              <div className="simStat"><span>Structures</span><b>{impact.buildings}</b></div>
              <div className="simStat"><span>Collapsed</span><b className="bad">{impact.collapsed}</b></div>
              <div className="simStat"><span>Extensive or worse</span><b className="warn">{impact.severe}</b></div>
              <div className="simStat"><span>Footprint lost</span><b>{fmtArea(impact.collapsedAreaM2)}</b></div>
              <div className="simStat"><span>Debris</span><b>{fmtVol(impact.debrisVolumeM3)}</b></div>
              <DamageBar counts={impact.counts} total={impact.buildings} />
            </>
          )}

          {impact.kind === 'flood' && (
            <>
              <div className="simStat"><span>Stage</span><b>{impact.stage.toFixed(1)} m above datum</b></div>
              <div className="simStat"><span>Area inundated</span><b>{fmtArea(impact.wetAreaM2)} · {(impact.wetFraction * 100).toFixed(0)}%</b></div>
              <div className="simStat"><span>Mean depth</span><b>{impact.meanDepth.toFixed(2)} m</b></div>
              <div className="simStat"><span>Deepest</span><b>{impact.maxDepth.toFixed(2)} m</b></div>
              <div className="simStat"><span>Buildings reached</span><b className="warn">{impact.inundated} / {impact.buildings}</b></div>
              <div className="simStat"><span>Over roof level</span><b className="bad">{impact.submerged}</b></div>
            </>
          )}

          {impact.kind === 'cyclone' && (
            <>
              <div className="simStat"><span>Sustained wind</span><b>{impact.sustainedMs.toFixed(0)} m/s</b></div>
              <div className="simStat"><span>3-s gust</span><b>{impact.gust.toFixed(0)} m/s</b></div>
              <div className="simStat"><span>Roof damage</span><b className="warn">{impact.roofDamaged} / {impact.buildings}</b></div>
              <div className="simStat"><span>Roof lost</span><b className="bad">{impact.roofStripped}</b></div>
              {impact.flood && (
                <>
                  <div className="simStat"><span>Surge stage</span><b>{impact.flood.stage.toFixed(1)} m</b></div>
                  <div className="simStat"><span>Buildings reached</span><b className="warn">{impact.flood.inundated}</b></div>
                </>
              )}
            </>
          )}

          <p className="muted simNote">
            A plausible scenario from published relationships, computed on a
            height field inferred from one image. Not an engineering
            assessment, and not a basis for any real decision.
          </p>
        </div>
      )}
    </div>
  );
}

/* Damage-state histogram — the shape of the distribution says more than any
   single count, and it is the standard way loss estimates are presented. */
function DamageBar({ counts, total }) {
  if (!total) return null;
  const labels = ['None', 'Slight', 'Moderate', 'Extensive', 'Complete'];
  const colors = ['#3f8f62', '#8fbf4a', '#e2c044', '#e08b3c', '#d9534f'];
  return (
    <>
      <div className="simHist">
        {counts.map((c, i) =>
          c > 0 ? (
            <div
              key={i}
              style={{ width: `${(c / total) * 100}%`, background: colors[i] }}
              title={`${labels[i]}: ${c}`}
            />
          ) : null
        )}
      </div>
      <div className="simLegend">
        {counts.map((c, i) =>
          c > 0 ? (
            <span key={i}>
              <i style={{ background: colors[i] }} />
              {labels[i]} {c}
            </span>
          ) : null
        )}
      </div>
    </>
  );
}

const fmtArea = (m2) =>
  m2 >= 10000 ? `${(m2 / 10000).toFixed(2)} ha` : `${Math.round(m2)} m²`;

const fmtVol = (m3) =>
  m3 >= 1000 ? `${(m3 / 1000).toFixed(1)}k m³` : `${Math.round(m3)} m³`;
