'use client';

import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
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
import ControlPanel from './ControlPanel';
import LayerBar from './LayerBar';
import Hud from './Hud';
import SourceImage from './SourceImage';
import Fallback2D from './Fallback2D';

const SKY = 0x9fc4e0;

export default function TerrainViewer({ dataset, onReset }) {
  const mountRef = useRef(null);
  const worldRef = useRef(null);

  const {
    heights: rawHeights, width, height, min, max, mean, pixelSpacing, bitmap, nodataPixels, metadata,
  } = dataset;
  const extentX = width * pixelSpacing;
  const extentZ = height * pixelSpacing;
  const quantity = metadata?.quantity || 'AGL';

  // ---------------------------------------------------------------- datum
  // Everything below renders RELATIVE TO THE SCENE'S LOWEST SURFACE.
  //
  // An AGL raster already has ground at ~0 m, but a DSM (DTM + AGL, the
  // georeferenced path) sits at its true elevation -- a Sikkim tile spans
  // ~1,500-1,700 m. The camera height, the orbit target, the fog, the plinth
  // and the fly ceiling were all written as multiples of `max`, i.e. of
  // ALTITUDE rather than RELIEF, so on a DSM the orbit target landed ~480 m
  // underneath the terrain and the fog, tuned for a 300 m scene, dissolved the
  // whole thing into sky colour. The pipeline's own metadata says it outright:
  // "subtract min_height_m before rendering if your viewer expects 0 = ground".
  //
  // AGL tiles keep datum 0, so they render exactly as before. The shader only
  // ever compares heights with other heights from this same array, so the
  // shift is invisible to it; the HUD adds the datum back for display.
  const datum = useMemo(() => (quantity === 'DSM' || min > 5 ? min : 0), [quantity, min]);
  const heights = useMemo(
    () => (datum ? rawHeights.map((v) => v - datum) : rawHeights),
    [rawHeights, datum]
  );
  const relief = max - datum;   // top of the scene, render space
  const floorH = min - datum;   // bottom of the scene, render space (0 for a DSM)

  const [mode, setMode] = useState('orbit');
  // A DSM is shown at true metric scale, which is what it always got before
  // (suggestExaggeration on an absolute max of ~1,600 m always returned 1).
  const [exag, setExag] = useState(() => (datum ? 1 : suggestExaggeration(relief, extentX)));
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

  // read by the analysis pass, which must not re-run when a toggle flips
  const optsRef = useRef({ trees: false, level: true, windows: true, edges: true });
  optsRef.current = { trees, level, windows, edges };

  const analyse = windows || trees;

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

    // ---- framing, from the scene's own size rather than fixed numbers ----
    const span = Math.max(extentX, extentZ);
    // Unchanged for an ordinary urban tile (relief << extent). The cap only
    // bites on mountain relief, where relief * 3.2 put the camera so high the
    // scene became a speck at the bottom of the frame.
    const camY = Math.min(relief * 3.2 + 70, relief + span * 0.9);
    const target0 = new THREE.Vector3(0, relief * 0.3, 0);
    const camPos0 = new THREE.Vector3(-extentX * 0.55, camY, extentZ * 0.7);
    const frameDist = camPos0.distanceTo(target0);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(SKY);
    // Haze scaled to the framing distance: ~24% at the starting viewpoint for
    // EVERY scene. The old fixed 0.0016 was right for a 338 m tile and swallowed
    // anything much bigger -- at 1.7 km it is 99.9% fog, which reads as empty sky.
    const fogDensity = Math.min(0.004, Math.max(0.00025, 0.53 / frameDist));
    scene.fog = new THREE.FogExp2(SKY, fogDensity);

    const camera = new THREE.PerspectiveCamera(
      60,
      mount.clientWidth / mount.clientHeight,
      0.5,
      Math.max(6000, frameDist * 8)
    );
    camera.position.copy(camPos0);

    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.08;
    orbit.maxPolarAngle = Math.PI * 0.495;
    orbit.target.copy(target0);
    // Never smaller than the starting distance, or OrbitControls silently
    // clamps the opening shot on the first update.
    orbit.maxDistance = Math.max(span * 3, frameDist * 2);

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
      maxAltitude: Math.max(600, relief * 6 + span * 1.5),
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
    // Covers tall relief too: a low sun over a 400 m valley wall throws its
    // shadow well outside the tile's horizontal footprint.
    const s = Math.max(span, relief * 1.2) * 0.62;
    Object.assign(sun.shadow.camera, { left: -s, right: s, top: s, bottom: -s, near: 1, far: s * 8 });
    sun.shadow.bias = -0.0006;
    sun.shadow.normalBias = 0.35;
    scene.add(sun);
    scene.add(sun.target);

    const plinthH = Math.max(12, relief * 0.5);
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
    const checkTexture = makeHeightCheckTexture(heights, width, height, floorH, relief);

    // extents let the shader sample the site / canopy masks from world XZ
    const material = createTerrainMaterial({
      map: texture, maxHeight: relief, extentX, extentZ,
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

    const onResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);

    const fwd = new THREE.Vector3();
    let last = performance.now();
    const loop = () => {
      world.raf = requestAnimationFrame(loop);
      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;

      if (world.rise) {
        world.rise.t += dt;
        const k = Math.min(world.rise.t / world.rise.dur, 1);
        const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
        world.mesh.scale.y = 0.0001 + e * world.targetExag;
        syncTreesY();
        if (k >= 1) world.rise = null;
      }

      // Paused while the source-image modal covers the scene: redrawing a
      // million-vertex terrain nobody can see only steals the frame budget the
      // modal's own pan and zoom need to stay smooth.
      if (world.paused) return;

      if (world.fly.enabled) world.fly.update(dt);
      else world.orbit.update();

      renderer.render(scene, camera);

      world.frames++;
      world.fpsClock += dt;
      if (world.fpsClock > 0.4) {
        const g = sampleGround(camera.position.x, camera.position.z);
        camera.getWorldDirection(fwd);
        setHud({
          fps: Math.round(world.frames / world.fpsClock),
          // Render space is relative to the datum; the readout is not.
          alt: camera.position.y + datum,
          ground: g + datum,
          agl: camera.position.y - g,
          x: camera.position.x,
          z: camera.position.z,
          heading: Math.atan2(fwd.z, fwd.x),
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

        world.scene.add(r.group);
        world.applyOptions(optsRef.current);

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
    const r = Math.max(extentX, extentZ, relief) * 1.4;
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
    a.download = `depthwizard-${dataset?.sampleId || 'scene'}.png`;
    a.click();
  }, []);

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
        sceneName={dataset?.name}
        timing={dataset?.timing}
        stats={{ width, height, min, max, mean, pixelSpacing, extentX, extentZ, nodataPixels,
                 quantity, relief: max - min, datum,
                 georeferenced: !!metadata?.georeferenced, crs: metadata?.crs || null }}
      />

      {/* Scene layers. These used to be a row of pills in the bottom-left
          corner, under the help text and easy to miss; they are the toggles
          people reach for most, so they sit top-centre where the eye lands. */}
      <LayerBar
        {...{ showHud, setShowHud, windows, setWindows, edges, setEdges,
              trees, setTrees, level, setLevel }}
      />

      {showHud && <Hud hud={hud} mode={mode} trees={trees} />}

      <SourceImage
        bitmap={bitmap}
        heights={rawHeights}
        width={width}
        height={height}
        extentX={extentX}
        extentZ={extentZ}
        camera={hud}
        name={dataset?.name}
        quantity={quantity}
        onOpenChange={(open) => { if (worldRef.current) worldRef.current.paused = open; }}
        units={quantity === 'rDSM' ? '' : 'm'}
      />

      {busy && <div className="busyChip"><span className="spinner" />Sharpening edges</div>}
    </div>
  );
}
