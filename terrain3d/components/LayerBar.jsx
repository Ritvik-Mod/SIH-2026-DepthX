'use client';

import { useEffect } from 'react';
import Icon from './Icons';

/**
 * Scene-layer toggles, top centre.
 *
 * They used to be a row of small pills in the bottom-left corner, underneath the
 * help text: easy to miss and awkward to reach. These are the switches people
 * flip most while exploring, so they sit where the eye lands first, and each
 * has a number-key shortcut.
 *
 * Level mounds only means something once trees are on, so it stays in place
 * but disabled rather than appearing and disappearing -- a toolbar whose
 * buttons shift sideways under the cursor is worse than one with a grey button.
 */
export default function LayerBar({
  showHud, setShowHud, windows, setWindows, edges, setEdges,
  trees, setTrees, level, setLevel,
}) {
  const items = [
    { key: '1', label: 'Windows', icon: 'windows', on: windows, set: setWindows,
      hint: 'Facade detail on detected buildings' },
    { key: '2', label: 'Roof edges', icon: 'edges', on: edges, set: setEdges,
      hint: 'Highlight the crease along roof lines' },
    { key: '3', label: 'Trees', icon: 'trees', on: trees, set: setTrees,
      hint: 'Replace canopy mounds with individual trees' },
    { key: '4', label: 'Level mounds', icon: 'level', on: trees && level, set: setLevel,
      disabled: !trees, hint: trees ? 'Flatten the DSM mound under each tree clump' : 'Turn on Trees first' },
  ];

  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT'
                || t.isContentEditable)) return;
      const n = { Digit1: 1, Digit2: 2, Digit3: 3, Digit4: 4, Digit5: 5 }[e.code];
      if (!n) return;
      if (n === 1) setWindows((v) => !v);
      if (n === 2) setEdges((v) => !v);
      if (n === 3) setTrees((v) => !v);
      if (n === 4 && trees) setLevel((v) => !v);
      if (n === 5) setShowHud((v) => !v);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [trees, setWindows, setEdges, setTrees, setLevel, setShowHud]);

  return (
    <div className="layerBar" role="toolbar" aria-label="Scene layers">
      <span className="layerLabel"><Icon name="layers" size={14} />Layers</span>
      {items.map((it) => (
        <button
          key={it.key}
          type="button"
          className={`layerBtn ${it.on ? 'on' : ''}`}
          aria-pressed={it.on}
          disabled={it.disabled}
          onClick={() => it.set((v) => !v)}
          title={`${it.hint}  ·  key ${it.key}`}
        >
          <Icon name={it.icon} size={15} />
          <span>{it.label}</span>
          <kbd>{it.key}</kbd>
        </button>
      ))}
      <span className="layerSep" aria-hidden="true" />
      <button
        type="button"
        className={`layerBtn ${showHud ? 'on' : ''}`}
        aria-pressed={showHud}
        onClick={() => setShowHud((v) => !v)}
        title="Telemetry readout  ·  key 5"
      >
        <Icon name="hud" size={15} />
        <span>HUD</span>
        <kbd>5</kbd>
      </button>
    </div>
  );
}
