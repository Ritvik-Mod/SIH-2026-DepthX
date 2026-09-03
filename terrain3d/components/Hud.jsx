'use client';

/**
 * Read-outs and the contextual help line.
 *
 * AGL is the number that matters while flying: it is the camera's height above
 * the RENDERED surface, so it goes to ~2 m and stops when the collision floor
 * catches you. If it ever reads negative, the floor has failed and that is a
 * bug worth reporting -- it should not be reachable.
 */
export default function Hud({ hud, mode, trees = false }) {
  const flying = mode === 'fly' || mode === 'walk';
  const walking = hud.moveMode === 'walk';

  const help = !flying
    ? 'Drag to orbit · Scroll to zoom · Right-drag to pan · switch to Fly or Walk for a first-person pass'
    : hud.locked
      ? (walking
          ? 'W A S D walk · Space jump · Shift sprint · Alt slow · Scroll speed · F to fly · R recover · Esc release'
          : 'W A S D move · Space / Ctrl altitude · Shift sprint · Alt slow · Scroll speed · F to walk · R recover · Esc release')
      : 'Click the scene to capture the mouse';

  return (
    <>
      <div className="hud">
        <div><span>FPS</span><b>{hud.fps}</b></div>
        <div><span>CAM ALT</span><b>{hud.alt.toFixed(1)} m</b></div>
        <div><span>SURFACE</span><b>{hud.ground.toFixed(1)} m</b></div>
        <div>
          <span>AGL</span>
          <b style={{ color: hud.agl < 0 ? '#ff6b6b' : undefined }}>
            {(hud.agl ?? hud.alt - hud.ground).toFixed(1)} m
          </b>
        </div>
        <div><span>X / Z</span><b>{hud.x.toFixed(0)} / {hud.z.toFixed(0)}</b></div>
        {flying && <div><span>SPEED</span><b>{(hud.speed ?? 0).toFixed(1)} m/s</b></div>}
        {flying && (
          <div>
            <span>MODE</span>
            <b style={{ color: walking ? '#ffd479' : '#8ecbff' }}>
              {walking ? (hud.grounded ? 'WALK' : 'FALL') : 'FLY'}
            </b>
          </div>
        )}
        {trees && <div><span>TREES</span><b>{hud.trees ?? 0}</b></div>}
      </div>

      <div className="help">{help}</div>

      {/* Pointer lock is not obvious to a first-time user, and without it the
          keys do nothing at all -- which reads as "the fly mode is broken". */}
      {flying && !hud.locked && (
        <div
          style={{
            position: 'absolute', top: '50%', left: '50%',
            transform: 'translate(-50%,-50%)', zIndex: 6, pointerEvents: 'none',
            padding: '14px 22px', borderRadius: 12, textAlign: 'center',
            background: 'rgba(17,21,26,0.9)', border: '1px solid rgba(95,178,255,0.4)',
            backdropFilter: 'blur(10px)', color: '#8ecbff',
            fontFamily: 'ui-monospace, monospace', fontSize: 13, letterSpacing: '0.04em',
          }}
        >
          CLICK TO CAPTURE MOUSE
          <div style={{ color: 'rgba(255,255,255,0.45)', fontSize: 11, marginTop: 6 }}>
            Esc releases it again
          </div>
        </div>
      )}

      {/* A centre reticle makes pitch legible; without one it is genuinely hard
          to tell level flight from a shallow dive over a flat tile. */}
      {flying && hud.locked && (
        <div
          style={{
            position: 'absolute', top: '50%', left: '50%',
            transform: 'translate(-50%,-50%)', zIndex: 6, pointerEvents: 'none',
            width: 14, height: 14, opacity: 0.5,
          }}
        >
          <div style={{ position: 'absolute', left: 6, top: 0, width: 2, height: 14, background: '#fff' }} />
          <div style={{ position: 'absolute', top: 6, left: 0, height: 2, width: 14, background: '#fff' }} />
        </div>
      )}
    </>
  );
}
