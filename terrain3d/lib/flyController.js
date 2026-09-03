import * as THREE from 'three';

/**
 * Pointer-lock fly camera: mouse to look, WASD to move, Space / Ctrl for
 * altitude, Shift to boost. Deliberately not three's FlyControls (which rolls)
 * or FirstPersonControls (which fights OrbitControls over the DOM element).
 */
export class FlyController {
  constructor(camera, domElement) {
    this.camera = camera;
    this.dom = domElement;
    this.enabled = false;
    this.locked = false;

    this.speed = 28; // m/s
    this.boost = 4;
    this.lookSpeed = 0.0022;
    this.damping = 8;

    this.yaw = 0;
    this.pitch = 0;
    this.velocity = new THREE.Vector3();
    this.keys = new Set();

    this._onKeyDown = (e) => {
      if (!this.enabled) return;
      this.keys.add(e.code);
      if (['Space', 'ControlLeft'].includes(e.code)) e.preventDefault();
    };
    this._onKeyUp = (e) => this.keys.delete(e.code);
    this._onMouseMove = (e) => {
      if (!this.enabled || !this.locked) return;
      this.yaw -= e.movementX * this.lookSpeed;
      this.pitch -= e.movementY * this.lookSpeed;
      const lim = Math.PI / 2 - 0.02;
      this.pitch = Math.max(-lim, Math.min(lim, this.pitch));
    };
    this._onLockChange = () => {
      this.locked = document.pointerLockElement === this.dom;
      if (!this.locked) this.keys.clear();
    };
    this._onClick = () => {
      if (this.enabled && !this.locked) this.dom.requestPointerLock();
    };

    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);
    document.addEventListener('mousemove', this._onMouseMove);
    document.addEventListener('pointerlockchange', this._onLockChange);
    this.dom.addEventListener('click', this._onClick);
  }

  // adopt whatever the orbit camera was looking at, so toggling modes is seamless
  syncFromCamera() {
    const e = new THREE.Euler().setFromQuaternion(this.camera.quaternion, 'YXZ');
    this.yaw = e.y;
    this.pitch = e.x;
  }

  setEnabled(on) {
    this.enabled = on;
    if (on) this.syncFromCamera();
    else if (document.pointerLockElement === this.dom) document.exitPointerLock();
    this.velocity.set(0, 0, 0);
    this.keys.clear();
  }

  update(dt) {
    if (!this.enabled) return;

    this.camera.quaternion.setFromEuler(new THREE.Euler(this.pitch, this.yaw, 0, 'YXZ'));

    const dir = new THREE.Vector3();
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);

    if (this.keys.has('KeyW')) dir.add(fwd);
    if (this.keys.has('KeyS')) dir.sub(fwd);
    if (this.keys.has('KeyD')) dir.add(right);
    if (this.keys.has('KeyA')) dir.sub(right);
    if (this.keys.has('Space') || this.keys.has('KeyE')) dir.y += 1;
    if (this.keys.has('ControlLeft') || this.keys.has('KeyQ')) dir.y -= 1;

    const boosting = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight');
    const target = this.speed * (boosting ? this.boost : 1);

    if (dir.lengthSq() > 0) {
      dir.normalize().multiplyScalar(target);
      this.velocity.lerp(dir, Math.min(1, this.damping * dt));
    } else {
      this.velocity.multiplyScalar(Math.max(0, 1 - this.damping * dt));
    }

    this.camera.position.addScaledVector(this.velocity, dt);
  }

  dispose() {
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
    document.removeEventListener('mousemove', this._onMouseMove);
    document.removeEventListener('pointerlockchange', this._onLockChange);
    this.dom.removeEventListener('click', this._onClick);
  }
}
