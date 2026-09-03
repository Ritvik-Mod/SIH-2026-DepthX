import * as THREE from 'three';

/**
 * Pointer-lock camera with two movement models:
 *
 *   FLY   free 6-DoF, no gravity. Minecraft-creative feel: momentum, world-space
 *         vertical, sprint, and a terrain floor you cannot pass through.
 *   WALK  gravity, jump, step-up. Ground-level first-person, which is what the
 *         problem statement actually asks for ("first-person navigation").
 *
 * ---------------------------------------------------------------------------
 * WHY THE OLD VERSION BROKE WHEN YOU FLEW DOWNWARDS
 *
 * Three faults compounded, and only the first is obvious:
 *
 *  1. There was no collision of any kind. `camera.position.addScaledVector(...)`
 *     ran unconditionally, so holding Ctrl walked the camera straight through
 *     the terrain and kept going.
 *
 *  2. TerrainViewer puts a solid plinth box directly beneath the surface
 *     (`BoxGeometry(extentX, plinthH, extentZ)` at y = -plinthH/2). Fall through
 *     the terrain and you are INSIDE that box. Box normals point outward and
 *     backface culling is on, so from the inside the walls vanish and you are
 *     left staring at fog with no horizon -- which reads as "the renderer broke".
 *
 *  3. The terrain material is single-sided too, so from below the ground itself
 *     is invisible. Nothing on screen tells you which way is up.
 *
 * The fix is a real floor: `getGroundHeight(x, z)` is injected by the viewer,
 * which already samples the RENDERED surface (canopy levelling and mesh.scale.y
 * included), so the collision height always matches what is drawn. Both the
 * position and the downward velocity are clamped -- clamping position alone
 * leaves velocity accumulating, so the camera sticks to the floor and then
 * launches when you look up.
 * ---------------------------------------------------------------------------
 */

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

export class FlyController {
  constructor(camera, domElement, options = {}) {
    this.camera = camera;
    this.dom = domElement;
    this.enabled = false;
    this.locked = false;

    // --- injected world knowledge -------------------------------------------
    // Returns the rendered surface height (metres, world space) at a world XZ.
    // Defaults to a flat floor at 0 so the controller still behaves sanely if
    // the viewer forgets to supply it.
    this.getGroundHeight = options.getGroundHeight || (() => 0);
    this.bounds = options.bounds || null;      // {minX,maxX,minZ,maxZ}
    this.maxAltitude = options.maxAltitude ?? 4000;

    // --- movement -----------------------------------------------------------
    this.mode = 'fly';                 // 'fly' | 'walk'
    this.baseSpeed = 28;               // m/s, overwritten by the viewer per scene
    this.sprintFactor = 3.5;
    this.crawlFactor = 0.25;
    this.accel = 9;                    // higher = snappier start
    this.damping = 7;                  // higher = shorter coast
    this.speedScaleWithAltitude = true;

    // --- collision ----------------------------------------------------------
    // Clearance is generous in fly mode: the terrain is a heightfield sampled
    // bilinearly, so a roof edge between two samples can sit slightly above the
    // interpolated value, and a tight clearance lets a corner poke through the
    // near plane. 2 m is comfortably more than the camera's 0.5 m near plane.
    this.flyClearance = 2.0;
    this.eyeHeight = 1.7;              // walk mode
    this.stepUp = 0.6;                 // walk: climb kerbs/steps without jumping
    this.gravity = 24;                 // m/s^2, snappier than real 9.81
    this.jumpSpeed = 8.5;
    this.grounded = false;

    // --- look ---------------------------------------------------------------
    this.lookSpeed = 0.0022;
    this.invertY = false;
    this.yaw = 0;
    this.pitch = 0;

    this.velocity = new THREE.Vector3();
    this.keys = new Set();
    this._euler = new THREE.Euler(0, 0, 0, 'YXZ');
    this._fwd = new THREE.Vector3();
    this._right = new THREE.Vector3();
    this._dir = new THREE.Vector3();

    // reported to the HUD
    this.currentSpeed = 0;
    this.altitudeAGL = 0;

    this._bind();
  }

  _bind() {
    this._onKeyDown = (e) => {
      if (!this.enabled) return;
      // Only swallow keys we actually use, and only while pointer-locked, so
      // the page stays usable (Ctrl+R, Cmd+S, tab focus) when it is not.
      const used = [
        'Space', 'ControlLeft', 'ControlRight', 'ShiftLeft', 'ShiftRight',
        'KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'KeyF', 'KeyR',
        'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
      ];
      if (this.locked && used.includes(e.code)) e.preventDefault();
      if (e.repeat) return;

      this.keys.add(e.code);

      // F toggles fly / walk. Only while locked, so pressing F in a text field
      // elsewhere on the page cannot teleport the camera.
      if (e.code === 'KeyF' && this.locked) this.setMode(this.mode === 'fly' ? 'walk' : 'fly');
      if (e.code === 'KeyR' && this.locked) this.resetToGround();
      if (e.code === 'Space' && this.mode === 'walk' && this.grounded) {
        this.velocity.y = this.jumpSpeed;
        this.grounded = false;
      }
    };

    this._onKeyUp = (e) => this.keys.delete(e.code);

    this._onMouseMove = (e) => {
      if (!this.enabled || !this.locked) return;
      // Guard against the enormous movementX spikes some browsers emit on the
      // first event after a pointer-lock transition, which snap the view around.
      const dx = clamp(e.movementX, -200, 200);
      const dy = clamp(e.movementY, -200, 200);
      this.yaw -= dx * this.lookSpeed;
      this.pitch -= (this.invertY ? -dy : dy) * this.lookSpeed;
      const lim = Math.PI / 2 - 0.015;
      this.pitch = clamp(this.pitch, -lim, lim);
    };

    // Scroll adjusts travel speed, the way every flight/CAD viewer does it.
    // Multiplicative so one notch feels the same at 5 m/s and at 500 m/s.
    this._onWheel = (e) => {
      if (!this.enabled || !this.locked) return;
      e.preventDefault();
      const k = Math.exp(-Math.sign(e.deltaY) * -0.12);
      this.baseSpeed = clamp(this.baseSpeed * k, 0.5, 2000);
    };

    this._onLockChange = () => {
      this.locked = document.pointerLockElement === this.dom;
      // Keys held at the moment focus is lost would otherwise stay "down"
      // forever and the camera would drift on its own after Esc.
      if (!this.locked) this.keys.clear();
      this.onLockChange?.(this.locked);
    };

    this._onLockError = () => {
      this.locked = false;
      this.onLockChange?.(false);
    };

    this._onClick = () => {
      if (this.enabled && !this.locked) this.dom.requestPointerLock?.();
    };

    // Losing the window (alt-tab) leaves keys stuck down exactly like losing
    // the pointer lock does.
    this._onBlur = () => this.keys.clear();

    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);
    window.addEventListener('blur', this._onBlur);
    document.addEventListener('mousemove', this._onMouseMove);
    document.addEventListener('pointerlockchange', this._onLockChange);
    document.addEventListener('pointerlockerror', this._onLockError);
    this.dom.addEventListener('click', this._onClick);
    this.dom.addEventListener('wheel', this._onWheel, { passive: false });
  }

  /** Adopt whatever the orbit camera was looking at, so toggling is seamless. */
  syncFromCamera() {
    this._euler.setFromQuaternion(this.camera.quaternion, 'YXZ');
    this.yaw = this._euler.y;
    this.pitch = this._euler.x;
  }

  setEnabled(on) {
    this.enabled = on;
    if (on) {
      this.syncFromCamera();
      // Entering fly mode from an orbit camera that happens to sit below a roof
      // would otherwise start you inside geometry.
      this._resolveGround(0, true);
    } else if (document.pointerLockElement === this.dom) {
      document.exitPointerLock();
    }
    this.velocity.set(0, 0, 0);
    this.keys.clear();
  }

  setMode(mode) {
    if (mode === this.mode) return;
    this.mode = mode;
    this.velocity.set(0, 0, 0);
    this.grounded = false;
    if (mode === 'walk') {
      // Drop to standing height at the current XZ rather than teleporting.
      const g = this.getGroundHeight(this.camera.position.x, this.camera.position.z);
      this.camera.position.y = Math.max(this.camera.position.y, g + this.eyeHeight);
    }
    this.onModeChange?.(mode);
  }

  /** Escape hatch: lift the camera to a safe height above the surface. */
  resetToGround() {
    const g = this.getGroundHeight(this.camera.position.x, this.camera.position.z);
    this.camera.position.y = g + (this.mode === 'walk' ? this.eyeHeight : Math.max(25, this.flyClearance * 6));
    this.velocity.set(0, 0, 0);
  }

  /**
   * The floor. Returns true if contact was made.
   *
   * `hard` forces the camera up even when it is only slightly below, which is
   * what we want on mode entry; during normal updates we only correct downward
   * penetration so a deliberate descent still feels smooth.
   */
  _resolveGround(dt, hard = false) {
    const p = this.camera.position;
    const ground = this.getGroundHeight(p.x, p.z);
    if (!Number.isFinite(ground)) return false;

    const floor = ground + (this.mode === 'walk' ? this.eyeHeight : this.flyClearance);

    if (p.y < floor) {
      p.y = floor;
      // Kill only the downward component. Zeroing the whole vector here would
      // stop horizontal motion dead every frame you skim the surface, which
      // feels like the camera is catching on nothing.
      if (this.velocity.y < 0) this.velocity.y = 0;
      this.grounded = true;
      return true;
    }

    if (hard) {
      p.y = Math.max(p.y, floor);
      return true;
    }

    // Small tolerance so walking down a gentle slope does not flicker between
    // grounded and airborne every frame.
    this.grounded = p.y <= floor + 0.05;
    return false;
  }

  update(dt) {
    if (!this.enabled) return;
    dt = Math.min(dt, 0.1); // a long frame must not teleport the camera through a wall

    this.camera.quaternion.setFromEuler(
      this._euler.set(this.pitch, this.yaw, 0, 'YXZ')
    );

    const k = this.keys;
    this._fwd.set(0, 0, -1).applyQuaternion(this.camera.quaternion);
    this._right.set(1, 0, 0).applyQuaternion(this.camera.quaternion);

    const dir = this._dir.set(0, 0, 0);

    if (this.mode === 'walk') {
      // Walking is horizontal: looking at your feet must not drive you into the
      // ground. Project forward onto the XZ plane.
      this._fwd.y = 0;
      if (this._fwd.lengthSq() < 1e-6) this._fwd.set(0, 0, -1).applyQuaternion(this.camera.quaternion);
      this._fwd.normalize();
      this._right.y = 0;
      this._right.normalize();
    }

    if (k.has('KeyW') || k.has('ArrowUp')) dir.add(this._fwd);
    if (k.has('KeyS') || k.has('ArrowDown')) dir.sub(this._fwd);
    if (k.has('KeyD') || k.has('ArrowRight')) dir.add(this._right);
    if (k.has('KeyA') || k.has('ArrowLeft')) dir.sub(this._right);

    if (this.mode === 'fly') {
      // World-space vertical, independent of where you are looking -- this is
      // the part that makes creative-mode flight feel controllable.
      if (k.has('Space') || k.has('KeyE')) dir.y += 1;
      if (k.has('ControlLeft') || k.has('ControlRight') || k.has('KeyQ')) dir.y -= 1;
    }

    const sprint = k.has('ShiftLeft') || k.has('ShiftRight');
    const crawl = k.has('AltLeft') || k.has('AltRight');

    // Speed scales with height above the surface, the way Google Earth does it:
    // crossing a 340 m tile at altitude should be quick, and inspecting a facade
    // from 3 m away should be slow, without touching a slider.
    const ground = this.getGroundHeight(this.camera.position.x, this.camera.position.z);
    this.altitudeAGL = Number.isFinite(ground) ? this.camera.position.y - ground : this.camera.position.y;
    let altScale = 1;
    if (this.speedScaleWithAltitude && this.mode === 'fly') {
      altScale = clamp(0.35 + this.altitudeAGL / 60, 0.35, 6);
    }

    let target = this.baseSpeed * altScale;
    if (sprint) target *= this.sprintFactor;
    if (crawl) target *= this.crawlFactor;

    // Exponential smoothing, not the linear (1 - k*dt) the old version used.
    // Linear damping goes NEGATIVE past dt = 1/k -- at damping 8 that is 125 ms,
    // i.e. any frame under 8 fps flipped the velocity sign and the camera jerked
    // backwards. exp() is unconditionally stable and frame-rate independent.
    const aK = 1 - Math.exp(-this.accel * dt);
    const dK = 1 - Math.exp(-this.damping * dt);

    if (dir.lengthSq() > 0) {
      dir.normalize().multiplyScalar(target);
      if (this.mode === 'walk') {
        const vy = this.velocity.y;
        this.velocity.lerp(dir, aK);
        this.velocity.y = vy;         // gravity owns the vertical axis in walk mode
      } else {
        this.velocity.lerp(dir, aK);
      }
    } else if (this.mode === 'walk') {
      this.velocity.x -= this.velocity.x * dK;
      this.velocity.z -= this.velocity.z * dK;
    } else {
      this.velocity.multiplyScalar(1 - dK);
    }

    if (this.mode === 'walk') this.velocity.y -= this.gravity * dt;

    // ---- integrate, then resolve, in that order ----------------------------
    const p = this.camera.position;
    const prevY = p.y;
    p.addScaledVector(this.velocity, dt);

    // Horizontal bounds. Soft: the position is clamped and the velocity into
    // the wall is cancelled, so you slide along it instead of stopping dead.
    if (this.bounds) {
      const b = this.bounds;
      if (p.x < b.minX) { p.x = b.minX; if (this.velocity.x < 0) this.velocity.x = 0; }
      if (p.x > b.maxX) { p.x = b.maxX; if (this.velocity.x > 0) this.velocity.x = 0; }
      if (p.z < b.minZ) { p.z = b.minZ; if (this.velocity.z < 0) this.velocity.z = 0; }
      if (p.z > b.maxZ) { p.z = b.maxZ; if (this.velocity.z > 0) this.velocity.z = 0; }
    }

    if (p.y > this.maxAltitude) { p.y = this.maxAltitude; if (this.velocity.y > 0) this.velocity.y = 0; }

    // Step-up: in walk mode a small rise (kerb, doorstep, low wall) should be
    // climbed rather than blocking you, otherwise a heightfield city is
    // unwalkable. Anything taller than stepUp still blocks.
    if (this.mode === 'walk') {
      const g = this.getGroundHeight(p.x, p.z);
      if (Number.isFinite(g)) {
        const rise = (g + this.eyeHeight) - prevY;
        if (rise > 0 && rise <= this.stepUp && this.grounded) p.y = g + this.eyeHeight;
      }
    }

    this._resolveGround(dt);

    // Final safety net. If anything at all -- a NaN sample, a mid-flight mesh
    // rebuild at a new resolution, a resolution change under the camera -- has
    // left us below the surface, recover rather than leaving the user inside
    // the plinth with no way to tell which way is up.
    const gFinal = this.getGroundHeight(p.x, p.z);
    if (Number.isFinite(gFinal) && p.y < gFinal) {
      p.y = gFinal + (this.mode === 'walk' ? this.eyeHeight : this.flyClearance);
      this.velocity.y = 0;
    }
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z)) {
      p.set(0, 100, 0);
      this.velocity.set(0, 0, 0);
    }

    this.currentSpeed = this.velocity.length();
  }

  dispose() {
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
    window.removeEventListener('blur', this._onBlur);
    document.removeEventListener('mousemove', this._onMouseMove);
    document.removeEventListener('pointerlockchange', this._onLockChange);
    document.removeEventListener('pointerlockerror', this._onLockError);
    this.dom.removeEventListener('click', this._onClick);
    this.dom.removeEventListener('wheel', this._onWheel);
  }
}
