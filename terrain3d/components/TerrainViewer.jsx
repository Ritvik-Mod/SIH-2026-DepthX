'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  buildTerrainGeometry,
  createTerrainMaterial,
  createTreeLayer,
  makeHeightCheckTexture,
  sampleBilinear,
  sharpenHeights,
  suggestExaggeration,
} from '@/lib/terrain';
import { FlyController } from '@/lib/flyController';
import { probeWebGL } from '@/lib/webgl';
import {
  CYCLONE_CATEGORIES,
  GUST_FACTOR,
  SCENARIOS,
  buildInventory,
  computeQuakeDamage,
  computeSpillMap,
  computeWindDamage,
  floodImpact,
  hollandWind,
  makeDamageTexture,
  makeWaterTexture,
  pgaFromMagnitude,
} from '@/lib/disaster';
import { DisasterFX } from '@/lib/disasterFx';
import ControlPanel from './ControlPanel';
import DisasterPanel from './DisasterPanel';
import Hud from './Hud';
import Fallback2D from './Fallback2D';

const SKY = 0x9fc4e0;

/**
 * Where dust should billow from after a quake: the world-space centroid of
 * every building that came down, with a radius from its own footprint so a
 * warehouse raises a bigger cloud than a shopfront.
 *
 * The pixel -> world mapping here has to match the one the mesh was built
 * with (see worldFromPixel in lib/terrain.js), or the dust rises next to the
 * ruin rather than out of it.
 */
function collapseEmitters(inventory, width, height, extentX, extentZ) {
  const out = [];
  for (const b of inventory) {
    if (!(b.damage >= 3)) continue;
    out.push({
      x: (b.cx / (width - 1) - 0.5) * extentX,
      z: (b.cy / (height - 1) - 0.5) * extentZ,
      base: b.base,
      r: Math.max(4, Math.sqrt(b.areaM2) * 0.9),
    });
  }
  return out;
}

export default function TerrainViewer({ dataset, onReset }) {
  const mountRef = useRef(null);
  const worldRef = useRef(null);

  const { heights, width, height, min, max, mean, pixelSpacing, bitmap, nodataPixels } = dataset;
  const extentX = width * pixelSpacing;
  const extentZ = height * pixelSpacing;

  const [mode, setMode] = useState('orbit');
  const [exag, setExag] = useState(() => suggestExaggeration(max, extentX));
  const [segments, setSegments] = useState(1024);
  const [sharpen, setSharpen] = useState(1);
  const [flat, setFlat] = useState(true);
  const [overlay, setOverlay] = useState(false);
  const [shading, setShading] = useState('texture');
  const [wireframe, setWireframe] = useState(false);
  const [shadows, setShadows] = useState(true);
  const [sunAz, setSunAz] = useState(135);
  const [sunEl, setSunEl] = useState(45);
  const [busy, setBusy] = useState(true);
  // non-null once WebGL has proved unavailable; switches the whole view to 2D
  const [glError, setGlError] = useState(null);
  const [work, setWork] = useState(null);
  const [hud, setHud] = useState({
    fps: 0, alt: 0, ground: 0, x: 0, z: 0, trees: 0,
    speed: 0, agl: 0, moveMode: 'fly', grounded: false, locked: false,
  });
  // mirrors FlyController.mode so the panel can drive it without a re-init
  const [moveMode, setMoveMode] = useState('fly');

  // ------------------------------------------------------- opt-in features
  // Nothing below is active unless switched on, and every one is display-only:
  // the height array is never written to. `windows` and `edges` are pure
  // shading. `trees` adds an overlay; `level` lowers the RENDERED surface of a
  // detected tree mound and does nothing at all until trees are on.
  const [showHud, setShowHud] = useState(true);
  const [windows, setWindows] = useState(true);
  const [edges, setEdges] = useState(true);
  const [trees, setTrees] = useState(false);
  const [level, setLevel] = useState(true);

  // ------------------------------------------------ disaster simulation
  // Also opt-in, also display-only, and also incapable of touching `heights`.
  // The whole feature is off until simOpen goes true, and closing it puts
  // every uniform and every environment value back where it started.
  const [simOpen, setSimOpen] = useState(false);
  const [sim, setSim] = useState({ kind: null });
  const [simReady, setSimReady] = useState(false);   // footprints available yet
  const [simBusy, setSimBusy] = useState(false);
  const [simProgress, setSimProgress] = useState(0);
  const [simPlaying, setSimPlaying] = useState(false);
  const [impact, setImpact] = useState(null);
  // bumped when the analysis pass republishes footprints, so the inventory
  // rebuilds without the scenario sliders being in that dependency list
  const [siteVersion, setSiteVersion] = useState(0);

  // read by the analysis pass, which must not re-run when a toggle flips
  const optsRef = useRef({ trees: false, level: true, windows: true, edges: true });
  optsRef.current = { trees, level, windows, edges };

  // The simulation needs the building footprints the analysis pass produces,
  // so opening it is a third reason to run that pass -- but only the boolean
  // is in the dependency list, never the scenario parameters, or every slider
  // tick would re-segment the scene.
  const analyse = windows || trees || simOpen;

  // ------------------------------------------------------------ init once
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    // WebGL is not guaranteed. Hardware acceleration can be switched off, the
    // GPU process can fail to start, and a VM or remote desktop may have no
    // passthrough at all -- and recent Chrome no longer falls back to software
    // rendering on its own. THREE.WebGLRenderer throws from its constructor in
    // every one of those cases, and that throw used to escape React and blank
    // the entire site, taking the upload form down with it. Detect it, and
    // degrade to the 2D relief view instead.
    const probe = probeWebGL();
    if (!probe.ok) { setGlError(probe.reason); return; }

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    } catch (e) {
      setGlError(e?.message || 'WebGL context creation failed.');
      return;
    }
    // A GPU reset mid-session drops the context and the canvas silently freezes.
    renderer.domElement.addEventListener('webglcontextlost', (ev) => {
      ev.preventDefault();
      setGlError('The WebGL context was lost (the GPU driver reset).');
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(SKY);
    scene.fog = new THREE.FogExp2(SKY, 0.0016);

    const camera = new THREE.PerspectiveCamera(
      60,
      mount.clientWidth / mount.clientHeight,
      0.5,
      6000
    );
    camera.position.set(-extentX * 0.55, max * 3.2 + 70, extentZ * 0.7);

    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.08;
    orbit.maxPolarAngle = Math.PI * 0.495;
    orbit.target.set(0, max * 0.3, 0);
    orbit.maxDistance = Math.max(extentX, extentZ) * 3;

    // The controller needs a floor. `getGroundHeight` is attached just below,
    // once sampleGround exists -- it reads the RENDERED surface (canopy levelling
    // and mesh.scale.y included), so collision always matches what is on screen.
    // Bounds are the tile plus a margin: flying off into empty fog and losing the
    // scene entirely is a worse failure than being gently stopped.
    const margin = Math.max(extentX, extentZ) * 0.35;
    const fly = new FlyController(camera, renderer.domElement, {
      bounds: {
        minX: -extentX / 2 - margin, maxX: extentX / 2 + margin,
        minZ: -extentZ / 2 - margin, maxZ: extentZ / 2 + margin,
      },
      maxAltitude: Math.max(600, max * 6 + Math.max(extentX, extentZ) * 1.5),
    });
    fly.baseSpeed = Math.max(12, extentX / 18);
    fly.onModeChange = (m) => setMoveMode(m);

    // the satellite texture already contains baked illumination, so ambient is
    // generous and the directional light exists mainly to cast shadows
    const hemi = new THREE.HemisphereLight(0xdfefff, 0x4a4237, 1.05);
    scene.add(hemi);

    const sun = new THREE.DirectionalLight(0xfff4e0, 2.0);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const s = Math.max(extentX, extentZ) * 0.62;
    Object.assign(sun.shadow.camera, { left: -s, right: s, top: s, bottom: -s, near: 1, far: s * 8 });
    sun.shadow.bias = -0.0006;
    sun.shadow.normalBias = 0.35;
    scene.add(sun);
    scene.add(sun.target);

    const plinthH = Math.max(12, max * 0.5);
    // DoubleSide matters: if anything ever does put the camera inside this box,
    // single-sided walls disappear and the user is left in featureless fog with
    // no way to tell which way is up. With both sides drawn they at least see a
    // dark room and can hit R to recover.
    const plinth = new THREE.Mesh(
      new THREE.BoxGeometry(extentX, plinthH, extentZ),
      new THREE.MeshStandardMaterial({ color: 0x2b2f36, roughness: 1, side: THREE.DoubleSide })
    );
    plinth.position.y = -plinthH / 2 - 0.05;
    scene.add(plinth);

    // bitmap is { source, flipY } -- three ignores flipY for ImageBitmap, so the
    // flip is already baked in at decode time and flipY comes back false
    const texture = new THREE.Texture(bitmap.source);
    texture.flipY = bitmap.flipY;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
    texture.generateMipmaps = true;
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.needsUpdate = true;

    // grayscale height texture for the alignment self-check
    const checkTexture = makeHeightCheckTexture(heights, width, height, min, max);

    // extents let the shader sample the site / canopy masks from world XZ
    const material = createTerrainMaterial({
      map: texture, maxHeight: max, extentX, extentZ,
    });
    material.userData.uniforms.uSky.value.setHex(SKY);

    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.scale.y = exag;
    // shadows render through their own depth material; this one mirrors the
    // mound levelling so a levelled mound stops casting a mound-shaped shadow
    mesh.customDepthMaterial = material.userData.depthMaterial;
    scene.add(mesh);

    const world = {
      renderer, scene, camera, orbit, fly, sun, mesh, material, texture, checkTexture, plinth,
      heights,
      rise: null,
      targetExag: exag,
      didFirstBuild: false,
      raf: 0,
      frames: 0,
      fpsClock: 0,
      // opt-in enrichment; all null until the analysis pass runs
      treesGroup: null,
      treeCount: 0,
      canopyTexture: null,
      siteTexture: null,
      canopyDelta: null,
      canopyDeltaData: null,
      canopyDeltaMax: 0,
      levelled: false,
    };
    worldRef.current = world;

    const sampleGround = (wx, wz) => {
      const fx = wx / extentX + 0.5;
      const fy = wz / extentZ + 0.5; // +Z is the last raster row after the -90 deg rotation
      let h = sampleBilinear(world.heights, width, height, fx, fy);
      // while mounds are levelled the drawn surface sits below the height
      // array, so the readout follows the surface actually on screen
      if (world.canopyDelta) {
        h -= sampleBilinear(world.canopyDelta, width, height, fx, fy);
      }
      // The simulation lowers the RENDERED surface through the vertex shader.
      // The floor follows the same array, scaled by the same animation amount,
      // so a building that has come down is one you can walk over -- and one
      // still standing is one you cannot walk through.
      const d = world.disaster;
      if (d && d.drop && d.surfaceAmount > 0) {
        h -= sampleBilinear(d.drop, width, height, fx, fy) * d.surfaceAmount;
      }
      return h * world.mesh.scale.y;
    };

    // THE FIX for falling through the terrain: give the controller a real floor.
    // sampleGround already accounts for mound levelling and the vertical
    // exaggeration, so the collision surface can never drift from the drawn one.
    fly.getGroundHeight = sampleGround;

    // Instanced forests bake their transforms, so any change to the vertical
    // exaggeration -- slider or reveal animation -- must re-seat them or they
    // float and sink away from the terrain.
    const syncTreesY = () => {
      if (!world.treesGroup) return;
      const sy = world.mesh.scale.y;
      for (const child of world.treesGroup.children) {
        if (typeof child.userData.rebase === 'function') {
          child.userData.rebase(sy, world.levelled);
        }
      }
    };
    world.syncTreesY = syncTreesY;

    // The single place the opt-in uniforms are set, so "off" always means
    // exactly the original render.
    const applyOptions = (o) => {
      const u = world.material.userData.uniforms;
      const hasSite = !!world.siteTexture;
      const hasMask = !!world.canopyTexture;

      u.uSiteOn.value = hasSite ? 1 : 0;
      u.uWindows.value = o.windows && hasSite ? 1 : 0;
      u.uEdge.value = o.edges ? 1 : 0;

      if (world.treesGroup) world.treesGroup.visible = !!o.trees;

      const levelling = !!(o.trees && o.level && hasMask && world.canopyDeltaMax > 0);
      world.levelled = levelling;

      u.uCanopyOn.value = o.trees && hasMask ? 1 : 0;
      u.uDeltaMax.value = levelling ? world.canopyDeltaMax : 0;
      u.uCanopyFlatten.value = levelling ? 1 : 0;
      world.canopyDelta = levelling ? world.canopyDeltaData : null;

      syncTreesY();
      world.material.needsUpdate = true;
    };
    world.applyOptions = applyOptions;

    // Built once and kept for the life of the scene. It is inert until a
    // scenario is set: no water, no particles, no uniform is off its default.
    world.disaster = new DisasterFX({
      scene,
      sun,
      hemi,
      extentX,
      extentZ,
      uniforms: material.userData.uniforms,
      getTrees: () => world.treesGroup,
      getScaleY: () => world.mesh.scale.y,
      getLevelled: () => world.levelled,
    });
    world.disaster.setDatum(min);

    const onResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);

    let last = performance.now();
    const loop = () => {
      world.raf = requestAnimationFrame(loop);
      const now = performance.now();
      // Two clocks on purpose. `dt` is clamped so that a stall cannot teleport
      // an animation across the scene in one step. `rawDt` is honest wall time,
      // and the FPS readout has to use that one: accumulating the clamped
      // value made the counter bottom out at ~10 and report 16 fps on a
      // software renderer genuinely managing 0.3 -- which is a lie told
      // exactly when the number matters most.
      const rawDt = (now - last) / 1000;
      const dt = Math.min(rawDt, 0.1);
      last = now;

      if (world.rise) {
        world.rise.t += dt;
        const k = Math.min(world.rise.t / world.rise.dur, 1);
        const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
        world.mesh.scale.y = 0.0001 + e * world.targetExag;
        syncTreesY();
        if (k >= 1) world.rise = null;
      }

      if (world.fly.enabled) world.fly.update(dt);
      else world.orbit.update();

      // After the controls have settled and before the draw: the shake is a
      // render-time offset that is undone immediately afterwards, so neither
      // OrbitControls' damping nor the fly controller ever sees a camera
      // position it did not itself set.
      world.disaster.update(dt, camera);
      world.disaster.applyShake(camera);
      renderer.render(scene, camera);
      world.disaster.clearShake(camera);

      world.frames++;
      world.fpsClock += rawDt;
      if (world.fpsClock > 0.4) {
        const g = sampleGround(camera.position.x, camera.position.z);
        setHud({
          fps: Math.round(world.frames / world.fpsClock),
          alt: camera.position.y,
          ground: g,
          agl: camera.position.y - g,
          x: camera.position.x,
          z: camera.position.z,
          trees: world.treeCount,
          speed: world.fly.enabled ? world.fly.currentSpeed : 0,
          moveMode: world.fly.mode,
          grounded: world.fly.grounded,
          locked: world.fly.locked,
        });
        world.frames = 0;
        world.fpsClock = 0;
      }
    };
    loop();

    return () => {
      cancelAnimationFrame(world.raf);
      window.removeEventListener('resize', onResize);
      world.disaster.dispose();
      fly.dispose();
      orbit.dispose();
      world.mesh.geometry.dispose();
      material.dispose();
      material.userData.depthMaterial.dispose();
      texture.dispose();
      checkTexture.dispose();
      if (world.treesGroup) {
        scene.remove(world.treesGroup);
        world.treesGroup.traverse((o) => {
          if (!o.isMesh) return;
          o.geometry.dispose();
          if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose());
          else o.material.dispose();
        });
      }
      if (world.canopyTexture) world.canopyTexture.dispose();
      if (world.siteTexture) world.siteTexture.dispose();
      plinth.geometry.dispose();
      plinth.material.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ------------------------------------------------- edge sharpening pass
  useEffect(() => {
    setBusy(true);
    // defer a tick so the "sharpening" chip actually paints before we block
    const id = setTimeout(() => {
      const out =
        sharpen > 0
          ? sharpenHeights(heights, width, height, { radius: 4, minStep: 1.2, strength: sharpen })
          : heights;
      if (worldRef.current) worldRef.current.heights = out;
      setWork(out);
      setBusy(false);
    }, 30);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sharpen]);

  // ------------------------------------------------------------- geometry
  useEffect(() => {
    const world = worldRef.current;
    if (!world || !work) return;
    const old = world.mesh.geometry;
    world.mesh.geometry = buildTerrainGeometry({
      heights: work, width, height, segments, extentX, extentZ,
    });
    old.dispose();
    if (!world.didFirstBuild) {
      world.didFirstBuild = true;
      world.targetExag = exag;
      world.rise = { t: 0, dur: 2.6 };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [work, segments]);

  // ----------------------------------------------- scene analysis (opt-in)
  // One pass yields both the building information the facade shader needs and
  // the tree clumps, so flipping either toggle afterwards is free. Keyed on the
  // heights, never on a toggle.
  useEffect(() => {
    const world = worldRef.current;
    if (!world || !work) return;

    const disposeGroup = (g) => {
      g.traverse((o) => {
        if (!o.isMesh) return;
        o.geometry.dispose();
        if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose());
        else o.material.dispose();
      });
    };

    const clear = () => {
      const u = world.material.userData.uniforms;
      if (world.treesGroup) {
        world.scene.remove(world.treesGroup);
        disposeGroup(world.treesGroup);
        world.treesGroup = null;
      }
      world.treeCount = 0;
      if (world.canopyTexture) { world.canopyTexture.dispose(); world.canopyTexture = null; }
      if (world.siteTexture) { world.siteTexture.dispose(); world.siteTexture = null; }
      world.canopyDelta = null;
      world.canopyDeltaData = null;
      world.canopyDeltaMax = 0;
      world.levelled = false;
      world.site = null;
      world.inventory = null;
      world.spill = null;
      u.uSite.value = null;
      u.uCanopy.value = null;
      u.uSiteOn.value = 0;
      u.uWindows.value = 0;
      u.uCanopyOn.value = 0;
      u.uCanopyFlatten.value = 0;
      u.uDeltaMax.value = 0;
      world.material.needsUpdate = true;
    };

    clear();
    if (!analyse) return;

    let cancelled = false;

    (async () => {
      try {
        const r = await createTreeLayer({
          source: bitmap.source,
          width,
          height,
          heights: work,          // placement: the array the mesh is built from
          detectHeights: heights, // detection: the RAW dsm, before sharpening
          extentX,
          extentZ,
          verticalScale: exag,    // settled value, not a mid-animation scale
          maxTrees: 1400,
          textureFlipY: bitmap.flipY, // whether the decoded RGB needs flipping
        });

        if (cancelled) {
          disposeGroup(r.group);
          r.maskTexture.dispose();
          r.siteTexture.dispose();
          return;
        }

        const u = world.material.userData.uniforms;

        world.siteTexture = r.siteTexture;
        u.uSite.value = r.siteTexture;
        u.uGroundMin.value = r.groundMin;
        u.uGroundSpan.value = r.groundSpan;

        world.treesGroup = r.group;
        world.treeCount = r.count;
        world.canopyTexture = r.maskTexture;
        world.canopyDeltaData = r.flattenDelta;
        world.canopyDeltaMax = r.deltaMax;
        u.uCanopy.value = r.maskTexture;

        world.site = r.site;
        world.scene.add(r.group);
        world.applyOptions(optsRef.current);
        setSiteVersion((v) => v + 1);

        console.log(
          '[scene]', r.stats.buildings, 'buildings ·',
          r.stats.clumps, 'tree clumps ->', r.count, 'trees', r.stats
        );
      } catch (err) {
        console.error('Scene analysis failed:', err);
      }
    })();

    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [work, analyse, width, height, extentX, extentZ, bitmap]);

  // ------------------------------------------------------- feature toggles
  useEffect(() => {
    const world = worldRef.current;
    if (!world || !world.applyOptions) return;
    world.applyOptions({ trees, level, windows, edges });
  }, [trees, level, windows, edges]);

  // --------------------------------------------------------- exaggeration
  useEffect(() => {
    const world = worldRef.current;
    if (!world || world.rise) return;
    world.targetExag = exag;
    world.mesh.scale.y = exag;
    if (world.syncTreesY) world.syncTreesY();
  }, [exag]);

  // -------------------------------------------------------------- shading
  useEffect(() => {
    const world = worldRef.current;
    if (!world) return;
    const u = world.material.userData.uniforms;
    u.uColorMix.value = shading === 'height' ? 1 : 0;
    u.uWall.value = shading === 'height' || overlay ? 0.2 : 0.92;
    u.uFlat.value = flat ? 1 : 0;
    world.material.map = overlay ? world.checkTexture : world.texture;
    world.material.wireframe = wireframe;
    world.material.needsUpdate = true;
  }, [shading, wireframe, flat, overlay]);

  // ------------------------------------------------------------------ sun
  useEffect(() => {
    const world = worldRef.current;
    if (!world) return;
    const r = Math.max(extentX, extentZ) * 1.4;
    const az = (sunAz * Math.PI) / 180;
    const el = (sunEl * Math.PI) / 180;
    world.sun.position.set(
      Math.cos(el) * Math.sin(az) * r,
      Math.sin(el) * r,
      Math.cos(el) * Math.cos(az) * r
    );
    world.sun.target.position.set(0, 0, 0);
    world.sun.castShadow = shadows;
    world.renderer.shadowMap.enabled = shadows;
    world.material.needsUpdate = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sunAz, sunEl, shadows]);

  // ----------------------------------------------------------------- mode
  useEffect(() => {
    const world = worldRef.current;
    if (!world) return;
    const flying = mode === 'fly' || mode === 'walk';
    world.orbit.enabled = !flying;
    if (flying) world.fly.setMode(mode === 'walk' ? 'walk' : 'fly');
    world.fly.setEnabled(flying);
  }, [mode]);

  // F toggles fly/walk from inside pointer lock; keep the panel in step with it
  useEffect(() => {
    if (mode !== 'fly' && mode !== 'walk') return;
    if (moveMode !== mode) setMode(moveMode);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [moveMode]);

  /* ------------------------------------------- simulation: scene inventory
   * Runs once per scene, not per slider. Two things are prepared here and
   * both are expensive enough that doing them on a drag would be felt:
   *
   *   the building inventory -- height, storeys, footprint area and a stable
   *   seed per detected structure, so damage is decided per BUILDING;
   *
   *   the spill field -- one Priority-Flood pass giving, for every cell, the
   *   water level at which it first connects to the edge of the tile. After
   *   this, any flood stage is a single comparison per cell, which is what
   *   makes the stage slider free.
   */
  useEffect(() => {
    const world = worldRef.current;
    if (!world || !simOpen || !world.site || !work) return;

    setSimReady(false);
    let cancelled = false;

    // A tick, so the panel paints "waiting for the analysis" before the main
    // thread goes away for a few hundred milliseconds.
    const id = setTimeout(() => {
      if (cancelled) return;
      const pxPerM = Math.min(width / extentX, height / extentZ);

      world.inventory = buildInventory({
        heights: work,
        groundBase: world.site.groundBase,
        buildingSeed: world.site.buildingSeed,
        buildingBlobs: world.site.buildingBlobs,
        width,
        pxPerM,
      });

      const sp = computeSpillMap(work, width, height, { maxDim: 512 });
      world.spill = sp;
      world.disaster.setWaterField(
        makeWaterTexture(sp.spill, sp.elev, sp.sw, sp.sh),
        sp
      );

      console.log('[sim]', world.inventory.length, 'structures ·',
        `${sp.sw}x${sp.sh} hydrology grid`);
      setSimReady(true);
    }, 24);

    return () => { cancelled = true; clearTimeout(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simOpen, siteVersion, work, width, height, extentX, extentZ]);

  /* ------------------------------------------------ simulation: scenario
   * Cheap by comparison -- everything costly was cached above -- so this may
   * re-run on every slider tick, behind a short debounce.
   */
  useEffect(() => {
    const world = worldRef.current;
    if (!world || !world.disaster) return;
    const d = world.disaster;

    if (!simOpen || !sim.kind || !simReady || !world.inventory) {
      d.setScenario(null);
      setImpact(null);
      return;
    }

    setSimBusy(true);
    let cancelled = false;

    const id = setTimeout(() => {
      if (cancelled) return;
      try {
        const pxPerM = Math.min(width / extentX, height / extentZ);
        const inv = world.inventory;
        let stats = null;
        let stage = 0;
        const params = { windDirDeg: sim.windDirDeg ?? 135 };

        if (sim.kind === 'earthquake') {
          const pga = pgaFromMagnitude(sim.magnitude, sim.distanceKm);
          const r = computeQuakeDamage({
            inventory: inv,
            heights: work,
            groundBase: world.site.groundBase,
            width, height, pxPerM, pga,
          });
          d.setDamage(r.drop, makeDamageTexture(r.drop, r.state, width, height));
          d.setEmitters(collapseEmitters(inv, width, height, extentX, extentZ));
          d.setTargetStage(0);
          params.pga = pga;
          stats = r.stats;
        } else if (sim.kind === 'flood') {
          d.setDamage(null, null);
          d.setEmitters([]);
          stage = sim.stage;
          d.setTargetStage(stage);
          stats = floodImpact({
            ...world.spill,
            inventory: inv,
            pixelSpacing,
            level: min + stage,
            datum: min,
          });
        } else if (sim.kind === 'cyclone') {
          const cat = CYCLONE_CATEGORIES.find((c) => c.cat === sim.category)
            || CYCLONE_CATEGORIES[2];
          const sustained = hollandWind(cat.windMs, sim.eyeKm);
          const r = computeWindDamage({
            inventory: inv,
            heights: work,
            width, height,
            sustainedMs: sustained,
            windDirDeg: sim.windDirDeg,
          });
          d.setDamage(r.drop, makeDamageTexture(r.drop, r.state, width, height));
          d.setEmitters([]);
          // Surge scaled by the local wind rather than the category alone: a
          // distant cat 5 does not push a 6 m wall of water onto this site.
          stage = sim.surge
            ? ((cat.surge[0] + cat.surge[1]) / 2) * Math.min(1, sustained / cat.windMs)
            : 0;
          d.setTargetStage(stage);
          params.sustainedMs = sustained;
          stats = {
            ...r.stats,
            flood: stage > 0
              ? floodImpact({
                  ...world.spill, inventory: inv, pixelSpacing,
                  level: min + stage, datum: min,
                })
              : null,
          };
        }

        d.setScenario(sim.kind, params, {
          duration: SCENARIOS[sim.kind].defaults.duration,
        });
        setImpact(stats);
      } catch (err) {
        console.error('Disaster simulation failed:', err);
        d.setScenario(null);
        setImpact(null);
      } finally {
        setSimBusy(false);
      }
    }, 70);

    return () => { cancelled = true; clearTimeout(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [simOpen, simReady, sim, work, width, height, extentX, extentZ, pixelSpacing, min]);

  /* The timeline lives in the render loop; this samples it for the panel at a
     rate a human can read, rather than re-rendering React at 60 Hz. */
  useEffect(() => {
    if (!simOpen) { setSimProgress(0); setSimPlaying(false); return; }
    const id = setInterval(() => {
      const d = worldRef.current?.disaster;
      if (!d) return;
      setSimProgress(d.progress);
      setSimPlaying(d.playing || d.settled);
    }, 140);
    return () => clearInterval(id);
  }, [simOpen]);

  const runSim = useCallback(() => { worldRef.current?.disaster?.play(); }, []);
  const resetSim = useCallback(() => { worldRef.current?.disaster?.reset(); }, []);
  const closeSim = useCallback(() => {
    setSimOpen(false);
    setSim({ kind: null });
    setImpact(null);
  }, []);

  const replay = useCallback(() => {
    const world = worldRef.current;
    if (!world) return;
    world.targetExag = exag;
    world.rise = { t: 0, dur: 2.6 };
  }, [exag]);

  const snapshot = useCallback(() => {
    const world = worldRef.current;
    if (!world) return;
    world.renderer.render(world.scene, world.camera);
    const a = document.createElement('a');
    a.href = world.renderer.domElement.toDataURL('image/png');
    a.download = 'terrain3d.png';
    a.click();
  }, []);

  const pill = (on) => ({
    padding: '5px 11px',
    borderRadius: 999,
    fontSize: 11,
    cursor: 'pointer',
    fontFamily: 'ui-monospace, monospace',
    letterSpacing: '0.03em',
    border: `1px solid ${on ? 'rgba(95,178,255,0.55)' : 'rgba(255,255,255,0.12)'}`,
    background: on ? 'rgba(95,178,255,0.16)' : 'rgba(17,21,26,0.86)',
    color: on ? '#8ecbff' : 'rgba(255,255,255,0.55)',
    backdropFilter: 'blur(10px)',
    transition: 'all 120ms ease',
  });

  if (glError) {
    return <Fallback2D dataset={dataset} onReset={onReset} reason={glError} />;
  }

  return (
    <div className="viewer">
      <div ref={mountRef} className="canvasMount" />
      <ControlPanel
        {...{
          mode, setMode, exag, setExag, segments, setSegments,
          sharpen, setSharpen, flat, setFlat, overlay, setOverlay,
          shading, setShading, wireframe, setWireframe,
          shadows, setShadows, sunAz, setSunAz, sunEl, setSunEl,
          replay, snapshot, onReset,
        }}
        stats={{ width, height, min, max, mean, pixelSpacing, extentX, extentZ, nodataPixels,
                 quantity: dataset?.metadata?.quantity || 'AGL' }}
      />
      {/* Feature toggles live here so ControlPanel.jsx stays untouched. */}
      <div
        style={{
          position: 'absolute', bottom: 16, left: 16, zIndex: 5,
          display: 'flex', gap: 6, flexWrap: 'wrap', maxWidth: 340,
        }}
      >
        <button style={pill(showHud)} onClick={() => setShowHud((v) => !v)}>HUD</button>
        <button style={pill(windows)} onClick={() => setWindows((v) => !v)}>WINDOWS</button>
        <button style={pill(edges)} onClick={() => setEdges((v) => !v)}>EDGES</button>
        <button style={pill(trees)} onClick={() => setTrees((v) => !v)}>TREES</button>
        {trees && (
          <button style={pill(level)} onClick={() => setLevel((v) => !v)}>LEVEL MOUNDS</button>
        )}
        <button
          style={{ ...pill(simOpen), borderColor: simOpen ? 'rgba(255,140,90,0.6)' : undefined,
                   background: simOpen ? 'rgba(255,140,90,0.16)' : 'rgba(17,21,26,0.86)',
                   color: simOpen ? '#ffb38c' : 'rgba(255,255,255,0.55)' }}
          onClick={() => (simOpen ? closeSim() : setSimOpen(true))}
        >
          DISASTER
        </button>
      </div>

      {simOpen && (
        <DisasterPanel
          sim={sim}
          setSim={setSim}
          ready={simReady}
          busy={simBusy}
          impact={impact}
          progress={simProgress}
          playing={simPlaying}
          onRun={runSim}
          onReset={resetSim}
          onClose={closeSim}
        />
      )}

      {showHud && <Hud hud={hud} mode={mode} trees={trees} />}
      {busy && (
        <div
          style={{
            position: 'absolute', bottom: 52, left: '50%', transform: 'translateX(-50%)',
            padding: '7px 14px', borderRadius: 999, fontSize: 11.5,
            background: 'rgba(17,21,26,0.86)', border: '1px solid rgba(255,255,255,0.1)',
            backdropFilter: 'blur(10px)', color: '#5fb2ff', fontFamily: 'ui-monospace, monospace',
          }}
        >
          Sharpening edges…
        </div>
      )}
    </div>
  );
}