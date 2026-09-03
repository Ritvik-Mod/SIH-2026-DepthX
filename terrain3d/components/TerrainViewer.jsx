'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  buildTerrainGeometry,
  createTerrainMaterial,
  makeHeightCheckTexture,
  sampleBilinear,
  sharpenHeights,
  suggestExaggeration,
} from '@/lib/terrain';
import { FlyController } from '@/lib/flyController';
import ControlPanel from './ControlPanel';
import Hud from './Hud';

const SKY = 0x9fc4e0;

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
  const [work, setWork] = useState(null);
  const [hud, setHud] = useState({ fps: 0, alt: 0, ground: 0, x: 0, z: 0 });

  // ------------------------------------------------------------ init once
  useEffect(() => {
    const mount = mountRef.current;
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
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

    const fly = new FlyController(camera, renderer.domElement);
    fly.speed = Math.max(18, extentX / 12);

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
    const plinth = new THREE.Mesh(
      new THREE.BoxGeometry(extentX, plinthH, extentZ),
      new THREE.MeshStandardMaterial({ color: 0x2b2f36, roughness: 1 })
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

    const material = createTerrainMaterial({ map: texture, maxHeight: max });
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.scale.y = exag;
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
    };
    worldRef.current = world;

    const sampleGround = (wx, wz) => {
      const fx = wx / extentX + 0.5;
      const fy = wz / extentZ + 0.5; // +Z is the last raster row after the -90 deg rotation
      return sampleBilinear(world.heights, width, height, fx, fy) * world.mesh.scale.y;
    };

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
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;

      if (world.rise) {
        world.rise.t += dt;
        const k = Math.min(world.rise.t / world.rise.dur, 1);
        const e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
        world.mesh.scale.y = 0.0001 + e * world.targetExag;
        if (k >= 1) world.rise = null;
      }

      if (world.fly.enabled) world.fly.update(dt);
      else world.orbit.update();

      renderer.render(scene, camera);

      world.frames++;
      world.fpsClock += dt;
      if (world.fpsClock > 0.4) {
        setHud({
          fps: Math.round(world.frames / world.fpsClock),
          alt: camera.position.y,
          ground: sampleGround(camera.position.x, camera.position.z),
          x: camera.position.x,
          z: camera.position.z,
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
      texture.dispose();
      checkTexture.dispose();
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

  // --------------------------------------------------------- exaggeration
  useEffect(() => {
    const world = worldRef.current;
    if (!world || world.rise) return;
    world.targetExag = exag;
    world.mesh.scale.y = exag;
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
    const flying = mode === 'fly';
    world.orbit.enabled = !flying;
    world.fly.setEnabled(flying);
  }, [mode]);

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
        stats={{ width, height, min, max, mean, pixelSpacing, extentX, extentZ, nodataPixels }}
      />
      <Hud hud={hud} mode={mode} />
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