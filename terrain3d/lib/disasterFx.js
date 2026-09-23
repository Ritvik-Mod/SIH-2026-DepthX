/* ==========================================================================
 * DISASTER SIMULATION — rendering half
 *
 * lib/disaster.js decides what happens. This draws it: the water surface,
 * the rain, the dust and debris, the ground shake, the wind through the
 * instanced forest, and the weather that goes with each event.
 *
 * Two rules hold throughout, and they are what make this safe to bolt onto a
 * working viewer:
 *
 *  1. EVERYTHING IS ADDITIVE AND REVERSIBLE. Nothing owned by the base scene
 *     is replaced. The environment values this does override — sky, fog, sun
 *     and hemisphere light — are snapshotted on construction and written back
 *     verbatim by clear(), so switching the simulation off is not "set them
 *     to something that looks right again", it is the original numbers.
 *
 *  2. THE SURFACE IS ONE SOURCE OF TRUTH. The shader subtracts a damage
 *     texture; the collision sampler subtracts the SAME array it was built
 *     from, scaled by the SAME animation amount exposed here as
 *     `surfaceAmount`. A collapsed building you can still walk into would be a
 *     worse bug than no simulation at all.
 * ========================================================================== */

import * as THREE from 'three';

/* Streak length is velocity x this, in seconds — a shutter time, effectively. */
const RAIN_STREAK = 0.035;

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function smoothstep(e0, e1, x) {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/* -------------------------------------------------------------------------- */
/* Soft round sprite, generated rather than fetched                           */
/* -------------------------------------------------------------------------- */

function makeSpriteTexture() {
  const s = 64;
  const c = document.createElement('canvas');
  c.width = s;
  c.height = s;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  grd.addColorStop(0, 'rgba(255,255,255,1)');
  grd.addColorStop(0.45, 'rgba(255,255,255,0.55)');
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, s, s);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/* ========================================================================== */
/* Water surface                                                              */
/* ========================================================================== */

/**
 * One plane across the whole tile at the current stage.
 *
 * The shape of the flood is NOT cut into this geometry — it comes from the
 * spill field in the fragment shader (see computeSpillMap), so changing the
 * stage is a uniform write and never a rebuild. Where terrain stands above the
 * water the depth buffer hides the plane for free, which is what gives a
 * pixel-sharp shoreline around every building without any geometry work.
 */
function makeWaterMesh(extentX, extentZ) {
  const uniforms = {
    uField: { value: null },      // R spill level, G surface elevation (metres)
    uLevel: { value: 0 },
    uExtent: { value: new THREE.Vector2(extentX, extentZ) },
    uTime: { value: 0 },
    uSunDir: { value: new THREE.Vector3(0.5, 0.7, 0.4) },
    uSky: { value: new THREE.Color(0.62, 0.77, 0.88) },
    uShallow: { value: new THREE.Color(0.32, 0.40, 0.33) },
    uDeep: { value: new THREE.Color(0.05, 0.11, 0.15) },
    uOpacity: { value: 0.0 },
    uChop: { value: 0.35 },       // surface agitation: still pool -> storm surge
  };

  const material = new THREE.ShaderMaterial({
    uniforms,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    vertexShader: `
      varying vec3 vW;
      void main() {
        vW = (modelMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform sampler2D uField;
      uniform float uLevel;
      uniform vec2 uExtent;
      uniform float uTime;
      uniform vec3 uSunDir;
      uniform vec3 uSky;
      uniform vec3 uShallow;
      uniform vec3 uDeep;
      uniform float uOpacity;
      uniform float uChop;
      varying vec3 vW;

      void main() {
        vec2 uv = vec2(vW.x / uExtent.x + 0.5, vW.z / uExtent.y + 0.5);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) discard;

        vec2 field = texture2D(uField, uv).rg;   // spill, ground elevation

        // Hydraulic connectivity, softened by a few centimetres so the
        // shoreline is a wet edge rather than a stencil.
        float wet = 1.0 - smoothstep(uLevel - 0.12, uLevel + 0.12, field.x);
        if (wet < 0.02) discard;

        float depth = max(uLevel - field.y, 0.0);

        // --- surface: three crossed wave trains, differing in scale and speed,
        // so nothing reads as a repeating tile ---
        vec2 p = vW.xz;
        float a =
            sin(p.x * 0.55 + uTime * 1.6) * 0.5
          + sin(p.y * 0.47 - uTime * 1.3) * 0.5
          + sin((p.x + p.y) * 0.21 + uTime * 0.7) * 0.8
          + sin((p.x - p.y) * 0.93 - uTime * 2.4) * 0.22;

        // Analytic normal from the same sum — no normal map, no seams.
        float dx =
            cos(p.x * 0.55 + uTime * 1.6) * 0.55 * 0.5
          + cos((p.x + p.y) * 0.21 + uTime * 0.7) * 0.21 * 0.8
          + cos((p.x - p.y) * 0.93 - uTime * 2.4) * 0.93 * 0.22;
        float dz =
            cos(p.y * 0.47 - uTime * 1.3) * 0.47 * 0.5
          + cos((p.x + p.y) * 0.21 + uTime * 0.7) * 0.21 * 0.8
          - cos((p.x - p.y) * 0.93 - uTime * 2.4) * 0.93 * 0.22;

        float amp = uChop * 0.06;
        vec3 n = normalize(vec3(-dx * amp, 1.0, -dz * amp));

        vec3 viewDir = normalize(cameraPosition - vW);
        float fres = pow(1.0 - max(dot(n, viewDir), 0.0), 4.0);

        // Beer-Lambert: shallow water shows the bed, deep water does not.
        float clarity = exp(-depth / 1.7);
        vec3 body = mix(uDeep, uShallow, clarity);

        vec3 col = mix(body, uSky * 1.05, clamp(fres * 0.9 + 0.06, 0.0, 1.0));

        // Specular glint off the sun, the cue that reads as "liquid".
        vec3 h = normalize(uSunDir + viewDir);
        col += vec3(1.0, 0.96, 0.88) * pow(max(dot(n, h), 0.0), 120.0) * 0.85;

        // Foam where the water runs out, and along wave crests in a surge.
        float shore = 1.0 - smoothstep(0.0, 0.45, depth);
        float crest = smoothstep(0.55, 1.05, a * uChop);
        float foam = clamp(shore * 0.85 + crest * uChop * 0.5, 0.0, 1.0);
        col = mix(col, vec3(0.88, 0.92, 0.94), foam * 0.7);

        // Thin water is nearly invisible; deep water is not.
        float alpha = uOpacity * wet * mix(0.35, 0.94, clamp(depth / 1.2, 0.0, 1.0));
        alpha = max(alpha, uOpacity * wet * foam * 0.8);

        gl_FragColor = vec4(col, clamp(alpha, 0.0, 1.0));
        #include <colorspace_fragment>
      }
    `,
  });

  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(extentX, extentZ, 1, 1), material);
  mesh.rotation.x = -Math.PI / 2;
  mesh.renderOrder = 2;
  mesh.frustumCulled = false;
  mesh.visible = false;
  mesh.name = 'DisasterWater';
  return mesh;
}

/* ========================================================================== */
/* Rain                                                                       */
/* ========================================================================== */

/**
 * Rain as line segments rather than sprites: a raindrop at 8 m/s seen at 60 fps
 * IS a streak, and a round dot reads as snow no matter how small you make it.
 *
 * The field is a box that follows the camera and wraps, so a few thousand
 * drops cover any view without ever needing more of them.
 */
class RainField {
  constructor(count, boxSize) {
    this.count = count;
    this.box = boxSize;
    this.pos = new Float32Array(count * 3);
    this.speed = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      this.pos[i * 3] = (Math.random() - 0.5) * boxSize;
      this.pos[i * 3 + 1] = Math.random() * boxSize;
      this.pos[i * 3 + 2] = (Math.random() - 0.5) * boxSize;
      this.speed[i] = 14 + Math.random() * 11;
    }

    const geo = new THREE.BufferGeometry();
    this.verts = new Float32Array(count * 6);
    geo.setAttribute('position', new THREE.BufferAttribute(this.verts, 3));

    const mat = new THREE.LineBasicMaterial({
      color: 0xbcd4e4,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });

    this.mesh = new THREE.LineSegments(geo, mat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 3;
    this.mesh.visible = false;
    this.mesh.name = 'DisasterRain';
  }

  update(dt, camera, windX, windZ, intensity) {
    const m = this.mesh.material;
    m.opacity = 0.34 * intensity;
    this.mesh.visible = intensity > 0.01;
    if (!this.mesh.visible) return;

    const b = this.box;
    const half = b / 2;
    const cx = camera.position.x;
    const cy = camera.position.y;
    const cz = camera.position.z;

    for (let i = 0; i < this.count; i++) {
      const o = i * 3;
      const vy = -this.speed[i];

      this.pos[o] += windX * dt;
      this.pos[o + 1] += vy * dt;
      this.pos[o + 2] += windZ * dt;

      // Wrap into a box centred on the camera. Relative coordinates mean the
      // field never drifts away, however far the user flies.
      let x = this.pos[o] - cx;
      let y = this.pos[o + 1] - cy;
      let z = this.pos[o + 2] - cz;

      if (x < -half) x += b; else if (x > half) x -= b;
      if (z < -half) z += b; else if (z > half) z -= b;
      if (y < -half) y += b; else if (y > half) y -= b;

      this.pos[o] = x + cx;
      this.pos[o + 1] = y + cy;
      this.pos[o + 2] = z + cz;

      const v = o * 2;
      this.verts[v] = this.pos[o];
      this.verts[v + 1] = this.pos[o + 1];
      this.verts[v + 2] = this.pos[o + 2];
      this.verts[v + 3] = this.pos[o] - windX * RAIN_STREAK;
      this.verts[v + 4] = this.pos[o + 1] - vy * RAIN_STREAK;
      this.verts[v + 5] = this.pos[o + 2] - windZ * RAIN_STREAK;
    }

    this.mesh.geometry.attributes.position.needsUpdate = true;
  }

  dispose() {
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
  }
}

/* ========================================================================== */
/* Dust and wind-blown debris                                                 */
/* ========================================================================== */

/**
 * One pool serving two jobs: dust billowing off collapsed buildings after a
 * quake, and litter torn loose and carried downwind in a cyclone. Both are a
 * few thousand points with a life, a velocity and a drag, so they share an
 * integrator and differ only in how they are seeded.
 */
class ParticleField {
  constructor(count, sprite) {
    this.count = count;
    this.pos = new Float32Array(count * 3);
    this.vel = new Float32Array(count * 3);
    this.life = new Float32Array(count);
    this.maxLife = new Float32Array(count);
    this.size = new Float32Array(count);
    this.cursor = 0;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    this.alpha = new Float32Array(count);
    geo.setAttribute('aAlpha', new THREE.BufferAttribute(this.alpha, 1));
    geo.setAttribute('aSize', new THREE.BufferAttribute(this.size, 1));

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uMap: { value: sprite },
        uColor: { value: new THREE.Color(0.62, 0.58, 0.50) },
        uOpacity: { value: 1 },
        uScale: { value: 1 },
      },
      transparent: true,
      depthWrite: false,
      vertexShader: `
        attribute float aAlpha;
        attribute float aSize;
        uniform float uScale;
        varying float vA;
        void main() {
          vA = aAlpha;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = aSize * uScale * 300.0 / max(-mv.z, 1.0);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        uniform sampler2D uMap;
        uniform vec3 uColor;
        uniform float uOpacity;
        varying float vA;
        void main() {
          vec4 t = texture2D(uMap, gl_PointCoord);
          float a = t.a * vA * uOpacity;
          if (a < 0.004) discard;
          gl_FragColor = vec4(uColor, a);
          #include <colorspace_fragment>
        }
      `,
    });

    this.mesh = new THREE.Points(geo, mat);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 3;
    this.mesh.visible = false;
    this.mesh.name = 'DisasterParticles';
  }

  emit(x, y, z, vx, vy, vz, life, size) {
    const i = this.cursor;
    this.cursor = (this.cursor + 1) % this.count;
    const o = i * 3;
    this.pos[o] = x;
    this.pos[o + 1] = y;
    this.pos[o + 2] = z;
    this.vel[o] = vx;
    this.vel[o + 1] = vy;
    this.vel[o + 2] = vz;
    this.life[i] = life;
    this.maxLife[i] = life;
    this.size[i] = size;
  }

  update(dt, gravity, drag, windX, windZ) {
    let live = 0;
    for (let i = 0; i < this.count; i++) {
      if (this.life[i] <= 0) {
        this.alpha[i] = 0;
        continue;
      }
      live++;
      this.life[i] -= dt;

      const o = i * 3;
      // Exponential drag towards the ambient wind, integrated as a simple
      // relaxation — stable at any dt, which a naive Euler drag is not.
      const k = 1 - Math.exp(-drag * dt);
      this.vel[o] += (windX - this.vel[o]) * k;
      this.vel[o + 2] += (windZ - this.vel[o + 2]) * k;
      this.vel[o + 1] += gravity * dt - this.vel[o + 1] * k * 0.6;

      this.pos[o] += this.vel[o] * dt;
      this.pos[o + 1] += this.vel[o + 1] * dt;
      this.pos[o + 2] += this.vel[o + 2] * dt;

      const f = Math.max(0, this.life[i] / this.maxLife[i]);
      // Fade in briefly then out, so nothing ever pops into existence.
      this.alpha[i] = Math.min(1, (1 - f) * 6) * f * f;
    }

    this.mesh.visible = live > 0;
    if (live > 0) {
      this.mesh.geometry.attributes.position.needsUpdate = true;
      this.mesh.geometry.attributes.aAlpha.needsUpdate = true;
      this.mesh.geometry.attributes.aSize.needsUpdate = true;
    }
    return live;
  }

  clear() {
    this.life.fill(0);
    this.alpha.fill(0);
    this.mesh.visible = false;
  }

  dispose() {
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
  }
}

/* ========================================================================== */
/* The simulation runtime                                                     */
/* ========================================================================== */

export class DisasterFX {
  constructor({
    scene, sun, hemi, extentX, extentZ,
    getTrees, getScaleY, getLevelled, uniforms,
  }) {
    this.scene = scene;
    this.sun = sun;
    this.hemi = hemi;
    this.extentX = extentX;
    this.extentZ = extentZ;
    this.getTrees = getTrees;
    this.getScaleY = getScaleY;
    // Mound levelling decides whether a tree stands on the ground or on top
    // of its own DSM mound; the wind animation re-seats every instance, so it
    // has to honour the same choice or the forest jumps when a gust hits.
    this.getLevelled = getLevelled || (() => true);
    this.u = uniforms;              // the terrain material's uniform block

    /*
     * Snapshot of everything this class is allowed to override. clear()
     * writes these back, so "off" is the original scene and not an
     * approximation of it.
     */
    this.base = {
      sky: scene.background?.isColor ? scene.background.getHex() : 0x9fc4e0,
      fogColor: scene.fog ? scene.fog.color.getHex() : 0x9fc4e0,
      fogDensity: scene.fog?.density ?? 0.0016,
      sunIntensity: sun.intensity,
      sunColor: sun.color.getHex(),
      hemiIntensity: hemi.intensity,
      hemiSky: hemi.color.getHex(),
      hemiGround: hemi.groundColor.getHex(),
    };

    this.sprite = makeSpriteTexture();
    this.water = makeWaterMesh(extentX, extentZ);
    this.rain = new RainField(2600, 70);
    this.particles = new ParticleField(2400, this.sprite);

    scene.add(this.water);
    scene.add(this.rain.mesh);
    scene.add(this.particles.mesh);

    // ---- simulation state ----
    this.kind = null;
    this.params = {};
    this.playing = false;
    this.t = 0;
    this.duration = 12;
    this.settled = false;

    this.drop = null;             // Float32Array of metres, for collision
    this.damageTexture = null;
    this.waterTexture = null;
    this.waterMeta = null;
    this.targetStage = 0;       // metres of water ABOVE the scene datum
    this.minHeight = 0;         // the datum itself, in raster units
    this.stage = 0;             // the animated stage

    this.surfaceAmount = 0;       // 0..1, shared by the shader and the sampler
    this.level = 0;
    this.windMs = 0;
    this.shake = 0;
    this.emitters = [];           // collapsed-building centroids, world XZ

    this._offset = new THREE.Vector3();
    this._saved = new THREE.Vector3();
    this._roll = 0;
    this._shaking = false;
    this._dustClock = 0;
    this._treeWind = { dx: 1, dz: 0, lean: 0, defoliate: 0, downedP: 0, gust: 0, phase: 0 };
  }

  /* ---------------------------------------------------------------- inputs */

  /**
   * Hand over a freshly computed damage field. `drop` and `texture` must be
   * two views of the same numbers — the shader reads one and the collision
   * sampler reads the other.
   */
  setDamage(drop, texture) {
    if (this.damageTexture && this.damageTexture !== texture) this.damageTexture.dispose();
    this.drop = drop;
    this.damageTexture = texture;
    this.u.uDamage.value = texture;
  }

  /**
   * The scene's lowest surface. Below this there is no water to draw at all,
   * and drawing a plane at a level nothing reaches leaves a sheet hanging in
   * mid air over a dry tile.
   */
  setDatum(minHeight) {
    this.minHeight = minHeight;
  }

  /**
   * The stage the event is working towards, in METRES ABOVE THE DATUM.
   *
   * Not an absolute level: a DSM tile whose ground sits at 210 m would
   * otherwise animate the water up from sea level through the whole scene
   * before anything visible happened.
   */
  setTargetStage(stage) {
    this.targetStage = stage;
  }

  setWaterField(texture, meta) {
    if (this.waterTexture && this.waterTexture !== texture) this.waterTexture.dispose();
    this.waterTexture = texture;
    this.waterMeta = meta;
    this.water.material.uniforms.uField.value = texture;
    this.u.uWaterField.value = texture;
  }

  /** Where dust should billow from: world XZ of each collapsed footprint. */
  setEmitters(list) {
    this.emitters = list || [];
  }

  setScenario(kind, params, { duration } = {}) {
    const changed = kind !== this.kind;
    this.kind = kind;
    this.params = { ...params };
    if (duration) this.duration = duration;

    if (!kind) {
      this.clear();
      return;
    }
    if (changed) {
      this.playing = false;
      this.settled = false;
      this.t = 0;
      this.particles.clear();
    }
    this._sync();
  }

  play() {
    if (!this.kind) return;
    this.t = 0;
    this.playing = true;
    this.settled = false;
    this.particles.clear();
  }

  /** Jump straight to the end state, for scrubbing parameters after an event. */
  settle() {
    this.playing = false;
    this.settled = true;
    this.t = this.duration;
    this._sync();
  }

  reset() {
    this.playing = false;
    this.settled = false;
    this.t = 0;
    this.shake = 0;
    this.particles.clear();
    this._sync();
  }

  get progress() {
    if (this.settled) return 1;
    if (!this.playing) return 0;
    return Math.min(1, this.t / this.duration);
  }

  /* -------------------------------------------------------------- lifecycle */

  /** Back to the untouched scene. Idempotent. */
  clear() {
    this.kind = null;
    this.playing = false;
    this.settled = false;
    this.t = 0;
    this.shake = 0;
    this.surfaceAmount = 0;
    this.stage = 0;
    this.level = 0;
    this.windMs = 0;

    this.u.uDamageOn.value = 0;
    this.u.uWaterOn.value = 0;
    this.u.uWaterLevel.value = 0;
    this.u.uWetness.value = 0;

    this.water.visible = false;
    this.rain.mesh.visible = false;
    this.rain.mesh.material.opacity = 0;
    this.particles.clear();

    const b = this.base;
    if (this.scene.background?.isColor) this.scene.background.setHex(b.sky);
    if (this.scene.fog) {
      this.scene.fog.color.setHex(b.fogColor);
      this.scene.fog.density = b.fogDensity;
    }
    this.sun.intensity = b.sunIntensity;
    this.sun.color.setHex(b.sunColor);
    this.hemi.intensity = b.hemiIntensity;
    this.hemi.color.setHex(b.hemiSky);
    this.hemi.groundColor.setHex(b.hemiGround);

    this._treeWind.lean = 0;
    this._treeWind.defoliate = 0;
    this._treeWind.downedP = 0;
    this._applyTreeWind(null);
  }

  dispose() {
    this.clear();
    this.scene.remove(this.water);
    this.scene.remove(this.rain.mesh);
    this.scene.remove(this.particles.mesh);
    this.water.geometry.dispose();
    this.water.material.dispose();
    this.rain.dispose();
    this.particles.dispose();
    this.sprite.dispose();
    if (this.damageTexture) this.damageTexture.dispose();
    if (this.waterTexture) this.waterTexture.dispose();
    this.damageTexture = null;
    this.waterTexture = null;
  }

  /* ------------------------------------------------------------ the timeline */

  /*
   * Each scenario's envelope, evaluated from progress alone so that scrubbing
   * a slider mid-event stays coherent and the settled state is exactly the
   * state at progress = 1.
   */
  _sync() {
    const p = this.progress;
    const k = this.kind;

    if (k === 'earthquake') {
      // Shaking leads; the buildings come down during the strong-motion
      // window, not at the first tremor and not after the dust settles.
      const env = p <= 0 ? 0
        : p < 0.14 ? p / 0.14
        : p < 0.62 ? 1
        : Math.max(0, 1 - (p - 0.62) / 0.38);
      this.shake = env * Math.min(1.3, (this.params.pga || 0) * 1.5);
      this.surfaceAmount = smoothstep(0.16, 0.66, p);
      this.stage = 0;
      this.level = this.minHeight;
      this.windMs = 0;
    } else if (k === 'flood') {
      this.shake = 0;
      this.surfaceAmount = 0;
      this.stage = (this.targetStage || 0) * smoothstep(0, 1, p);
      this.level = this.minHeight + this.stage;
      this.windMs = 0;
    } else if (k === 'cyclone') {
      this.shake = 0;
      this.surfaceAmount = smoothstep(0.28, 0.82, p);
      this.stage = (this.targetStage || 0) * smoothstep(0.18, 0.95, p);
      this.level = this.minHeight + this.stage;
      this.windMs = (this.params.sustainedMs || 0) * smoothstep(0, 0.45, p);
    } else {
      this.shake = 0;
      this.surfaceAmount = 0;
      this.stage = 0;
      this.level = this.minHeight;
      this.windMs = 0;
    }

    this._writeUniforms();
  }

  _writeUniforms() {
    const u = this.u;
    const scaleY = this.getScaleY();

    u.uDamageOn.value = this.drop && this.surfaceAmount > 0 ? this.surfaceAmount : 0;

    const flooding = (this.kind === 'flood' || this.kind === 'cyclone') && this.waterTexture;
    const wet = flooding && this.stage > 0.02;

    u.uWaterOn.value = wet ? 1 : 0;
    u.uWaterLevel.value = this.level;

    this.water.visible = !!wet;
    if (wet) {
      this.water.position.y = this.level * scaleY;
      const wu = this.water.material.uniforms;
      wu.uLevel.value = this.level;
      wu.uOpacity.value = 1;
      wu.uChop.value = this.kind === 'cyclone' ? 0.35 + Math.min(1, this.windMs / 55) * 0.9 : 0.3;
      wu.uSunDir.value.copy(this.sun.position).normalize();
      wu.uSky.value.copy(this.scene.background?.isColor
        ? this.scene.background
        : wu.uSky.value);
    }

    // Rain leaves everything slick and dark; that reads more strongly than the
    // raindrops themselves do.
    u.uWetness.value = this.kind === 'cyclone'
      ? Math.min(1, this.windMs / 30) * 0.85
      : 0;
  }

  /* ------------------------------------------------------------------ frame */

  update(dt, camera) {
    if (!this.kind) return;

    if (this.playing) {
      this.t += dt;
      if (this.t >= this.duration) {
        this.t = this.duration;
        this.playing = false;
        this.settled = true;
      }
    }

    this._sync();

    const time = performance.now() / 1000;
    this.water.material.uniforms.uTime.value = time;

    this._environment();
    this._shakeStep(time);
    this._dust(dt, camera);
    this._wind(dt, camera, time);
  }

  /* Sky, fog and light for the event in progress. */
  _environment() {
    const b = this.base;
    const scene = this.scene;

    if (this.kind === 'cyclone') {
      // Storm: the sun goes, the sky goes grey-green, visibility collapses.
      const s = Math.min(1, this.windMs / 45);
      const sky = new THREE.Color(b.sky).lerp(new THREE.Color(0x4a5560), 0.45 + s * 0.4);
      if (scene.background?.isColor) scene.background.copy(sky);
      if (scene.fog) {
        scene.fog.color.copy(sky);
        scene.fog.density = lerp(b.fogDensity, 0.0075, s);
      }
      this.sun.intensity = lerp(b.sunIntensity, 0.28, 0.35 + s * 0.6);
      this.sun.color.setHex(0xc8d2dc);
      this.hemi.intensity = lerp(b.hemiIntensity, 0.62, s);
      this.hemi.color.setHex(0x8b9aa6);
    } else if (this.kind === 'earthquake') {
      // A dust pall builds while things are coming down and then hangs.
      const d = this.surfaceAmount;
      const sky = new THREE.Color(b.sky).lerp(new THREE.Color(0xa89880), d * 0.55);
      if (scene.background?.isColor) scene.background.copy(sky);
      if (scene.fog) {
        scene.fog.color.copy(sky);
        scene.fog.density = lerp(b.fogDensity, 0.0052, d);
      }
      this.sun.intensity = lerp(b.sunIntensity, b.sunIntensity * 0.62, d);
      this.sun.color.setHex(0xffe9c9);
      this.hemi.intensity = b.hemiIntensity;
      this.hemi.color.setHex(b.hemiSky);
    } else if (this.kind === 'flood') {
      // Overcast, but nothing dramatic: the water is the story.
      const d = this.targetStage > 0 ? this.stage / this.targetStage : 0;
      const sky = new THREE.Color(b.sky).lerp(new THREE.Color(0x7d8b96), d * 0.4);
      if (scene.background?.isColor) scene.background.copy(sky);
      if (scene.fog) {
        scene.fog.color.copy(sky);
        scene.fog.density = lerp(b.fogDensity, 0.0028, d);
      }
      this.sun.intensity = lerp(b.sunIntensity, b.sunIntensity * 0.78, d);
      this.hemi.intensity = b.hemiIntensity;
    }
  }

  /* --------------------------------------------------------------- shaking */

  /*
   * Real strong motion peaks at a few centimetres of displacement, which on a
   * 300 m tile is invisible. This is DELIBERATELY EXAGGERATED for legibility —
   * the frequency content is honest (1-6 Hz, where damaging motion lives) but
   * the amplitude is not, and nothing downstream reads it as data.
   */
  _shakeStep(time) {
    if (this.shake <= 0.001) {
      this._offset.set(0, 0, 0);
      this._roll = 0;
      return;
    }
    const a = this.shake;
    const t = time;
    this._offset.set(
      (Math.sin(t * 17.2) * 0.55 + Math.sin(t * 33.7) * 0.28 + Math.sin(t * 7.3) * 0.5) * a * 0.42,
      (Math.sin(t * 24.1) * 0.5 + Math.sin(t * 41.3) * 0.22) * a * 0.26,
      (Math.cos(t * 15.9) * 0.55 + Math.cos(t * 29.4) * 0.3 + Math.cos(t * 6.1) * 0.5) * a * 0.42
    );
    this._roll = Math.sin(t * 11.4) * a * 0.008;
  }

  /**
   * Applied immediately before the draw call and undone immediately after, so
   * OrbitControls and the fly controller never see a camera they did not set.
   * Touching camera.position inside the control update would fight damping and
   * make the view drift.
   */
  applyShake(camera) {
    if (this.shake <= 0.001) return;
    this._saved.copy(camera.position);
    camera.position.add(this._offset);
    if (this._roll) camera.rotateZ(this._roll);
    camera.updateMatrixWorld();
    this._shaking = true;
  }

  clearShake(camera) {
    if (!this._shaking) return;
    if (this._roll) camera.rotateZ(-this._roll);
    camera.position.copy(this._saved);
    camera.updateMatrixWorld();
    this._shaking = false;
  }

  /* ------------------------------------------------------------------ dust */

  _dust(dt, camera) {
    const scaleY = this.getScaleY();

    if (this.kind === 'earthquake') {
      this.particles.mesh.material.uniforms.uColor.value.setRGB(0.66, 0.60, 0.51);
      this.particles.mesh.material.uniforms.uScale.value = 1.7;

      // Emit only while things are actually falling.
      const rate = this.shake > 0.05 && this.surfaceAmount < 0.999 ? 220 * this.shake : 0;
      this._dustClock += rate * dt;
      const emits = Math.floor(this._dustClock);
      this._dustClock -= emits;

      for (let i = 0; i < emits && this.emitters.length; i++) {
        const e = this.emitters[(Math.random() * this.emitters.length) | 0];
        const spread = e.r;
        this.particles.emit(
          e.x + (Math.random() - 0.5) * spread,
          e.base * scaleY + Math.random() * 2,
          e.z + (Math.random() - 0.5) * spread,
          (Math.random() - 0.5) * 7,
          1.5 + Math.random() * 5,
          (Math.random() - 0.5) * 7,
          3.5 + Math.random() * 4.5,
          0.05 + Math.random() * 0.09
        );
      }
      this.particles.update(dt, 0.55, 0.45, 0, 0);
    } else if (this.kind === 'cyclone') {
      this.particles.mesh.material.uniforms.uColor.value.setRGB(0.72, 0.70, 0.62);
      this.particles.mesh.material.uniforms.uScale.value = 1.1;

      const w = this.windMs;
      const rate = w > 12 ? Math.min(260, (w - 12) * 9) : 0;
      this._dustClock += rate * dt;
      const emits = Math.floor(this._dustClock);
      this._dustClock -= emits;

      const dir = ((this.params.windDirDeg ?? 135) * Math.PI) / 180;
      const wx = Math.sin(dir) * w;
      const wz = Math.cos(dir) * w;

      for (let i = 0; i < emits; i++) {
        // Seeded upwind of the camera so debris flies past rather than away.
        const d = 45;
        this.particles.emit(
          camera.position.x - (wx / Math.max(w, 1)) * d + (Math.random() - 0.5) * 60,
          Math.max(0, camera.position.y - 20) + Math.random() * 40,
          camera.position.z - (wz / Math.max(w, 1)) * d + (Math.random() - 0.5) * 60,
          wx * 0.7,
          (Math.random() - 0.3) * 4,
          wz * 0.7,
          2.2 + Math.random() * 2.2,
          0.035 + Math.random() * 0.06
        );
      }
      this.particles.update(dt, -1.2, 1.4, wx * 0.85, wz * 0.85);
    } else {
      this.particles.update(dt, 0.4, 0.5, 0, 0);
    }
  }

  /* ------------------------------------------------------------------ wind */

  _wind(dt, camera, time) {
    if (this.kind !== 'cyclone') {
      if (this.rain.mesh.visible) this.rain.update(dt, camera, 0, 0, 0);
      if (this._treeWind.lean !== 0) {
        this._treeWind.lean = 0;
        this._treeWind.defoliate = 0;
        this._treeWind.downedP = 0;
        this._applyTreeWind(null);
      }
      return;
    }

    const w = this.windMs;
    const gust = w * 1.3;
    const dir = ((this.params.windDirDeg ?? 135) * Math.PI) / 180;
    const dx = Math.sin(dir);
    const dz = Math.cos(dir);

    this.rain.update(dt, camera, dx * w * 0.8, dz * w * 0.8, Math.min(1, w / 26));

    const tw = this._treeWind;
    tw.dx = dx;
    tw.dz = dz;
    tw.gust = gust;
    tw.lean = Math.min(0.85, (gust * gust) / 4200);
    tw.defoliate = Math.min(1, Math.max(0, (gust - 24) / 26));
    tw.downedP = Math.min(0.85, Math.max(0, (gust - 42) / 40)) * this.surfaceAmount;
    // A gust front travels; the phase makes the canopy ripple rather than
    // breathe in unison, which is the giveaway of a fake wind.
    tw.phase = time * 1.7;

    this._applyTreeWind(tw);
  }

  _applyTreeWind(wind) {
    const trees = this.getTrees?.();
    if (!trees) return;
    const sy = this.getScaleY();
    for (const child of trees.children) {
      if (typeof child.userData.rebase === 'function') {
        child.userData.rebase(sy, this.getLevelled(), wind);
      }
    }
  }
}
