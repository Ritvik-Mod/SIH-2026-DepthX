'use client';

/**
 * Telemetry and the contextual help line.
 *
 * AGL is the number that matters while flying: it is the camera's height above
 * the RENDERED surface, so it goes to ~2 m and stops when the collision floor
 * catches you. If it ever reads negative, the floor has failed and that is a
 * bug worth reporting -- it should not be reachable.
 */
function Row({ label, value, unit, tone }) {
  return (
    <div className="hudRow">
      <span>{label}</span>
      <b className={tone || ''}>
        {value}
        {unit && <small>{unit}</small>}
      </b>
    </div>
  );
}

export default function Hud({ hud, mode, trees = false }) {
  const flying = mode === 'fly' || mode === 'walk';
  const walking = hud.moveMode === 'walk';
  const agl = hud.agl ?? hud.alt - hud.ground;

  const help = !flying
    ? 'Drag to orbit · Scroll to zoom · Right-drag to pan · Fly or Walk for first person'
    : hud.locked
      ? (walking
          ? 'WASD walk · Space jump · Shift sprint · Alt slow · F fly · R recover · Esc release'
          : 'WASD move · Space / Ctrl altitude · Shift sprint · Alt slow · F walk · R recover · Esc release')
      : 'Click the scene to capture the mouse';

  return (
    <>
      <div className="hud" aria-label="Telemetry">
        <div className="hudHead">
          <span className="liveDot" aria-hidden="true" />
          <span>Telemetry</span>
          <b>{hud.fps}<small> fps</small></b>
        </div>
        <Row label="Camera" value={hud.alt.toFixed(1)} unit=" m" />
        <Row label="Surface" value={hud.ground.toFixed(1)} unit=" m" />
        <Row label="Above ground" value={agl.toFixed(1)} unit=" m" tone={agl < 0 ? 'bad' : ''} />
        <Row label="X / Z" value={`${hud.x.toFixed(0)} / ${hud.z.toFixed(0)}`} unit=" m" />
        {flying && <Row label="Speed" value={(hud.speed ?? 0).toFixed(1)} unit=" m/s" />}
        {flying && (
          <Row
            label="Mode"
            value={walking ? (hud.grounded ? 'Walk' : 'Falling') : 'Fly'}
            tone={walking ? 'warn' : 'accent'}
          />
        )}
        {trees && <Row label="Trees" value={(hud.trees ?? 0).toLocaleString()} />}
      </div>

      <div className="helpBar">{help}</div>

      {/* Pointer lock is not obvious to a first-time user, and without it the
          keys do nothing at all -- which reads as "the fly mode is broken". */}
      {flying && !hud.locked && (
        <div className="lockPrompt">
          <b>Click to capture the mouse</b>
          <small>Esc releases it again</small>
        </div>
      )}

      {/* A centre reticle makes pitch legible; without one it is genuinely hard
          to tell level flight from a shallow dive over a flat tile. */}
      {flying && hud.locked && <div className="reticle" aria-hidden="true"><i /><i /></div>}
    </>
  );
}
