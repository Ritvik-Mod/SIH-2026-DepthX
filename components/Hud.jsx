'use client';

export default function Hud({ hud, mode }) {
  return (
    <>
      <div className="hud">
        <div><span>FPS</span><b>{hud.fps}</b></div>
        <div><span>CAM ALT</span><b>{hud.alt.toFixed(1)} m</b></div>
        <div><span>SURFACE</span><b>{hud.ground.toFixed(1)} m</b></div>
        <div><span>AGL</span><b>{(hud.alt - hud.ground).toFixed(1)} m</b></div>
        <div><span>X / Z</span><b>{hud.x.toFixed(0)} / {hud.z.toFixed(0)}</b></div>
      </div>

      <div className="help">
        {mode === 'fly'
          ? 'Click the scene to capture the mouse · W A S D move · Space / Ctrl altitude · Shift boost · Esc release'
          : 'Drag to orbit · Scroll to zoom · Right-drag to pan · switch to Fly for a first-person pass'}
      </div>
    </>
  );
}
