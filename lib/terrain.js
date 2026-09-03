import * as THREE from 'three';

// fx, fy in [0,1]. fy = 0 is raster ROW 0, which is also texture v = 1 and the
// first row of PlaneGeometry -- see README "Alignment" before changing this.
export function sampleBilinear(data, w, h, fx, fy) {
  const x = Math.min(Math.max(fx, 0), 1) * (w - 1);
  const y = Math.min(Math.max(fy, 0), 1) * (h - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, w - 1);
  const y1 = Math.min(y0 + 1, h - 1);
  const tx = x - x0;
  const ty = y - y0;
  const a = data[y0 * w + x0];
  const b = data[y0 * w + x1];
  const c = data[y1 * w + x0];
  const d = data[y1 * w + x1];
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

// separable local min / max over a square window of radius r
function localMinMax(src, w, h, r) {
  const n = src.length;
  const rowMin = new Float32Array(n);
  const rowMax = new Float32Array(n);

  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let mn = Infinity;
      let mx = -Infinity;
      const a = x - r < 0 ? 0 : x - r;
      const b = x + r > w - 1 ? w - 1 : x + r;
      for (let i = a; i <= b; i++) {
        const v = src[row + i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      rowMin[row + x] = mn;
      rowMax[row + x] = mx;
    }
  }

  const outMin = new Float32Array(n);
  const outMax = new Float32Array(n);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let mn = Infinity;
      let mx = -Infinity;
      const a = y - r < 0 ? 0 : y - r;
      const b = y + r > h - 1 ? h - 1 : y + r;
      for (let j = a; j <= b; j++) {
        const k = j * w + x;
        const p = rowMin[k];
        const q = rowMax[k];
        if (p < mn) mn = p;
        if (q > mx) mx = q;
      }
      const k = y * w + x;
      outMin[k] = mn;
      outMax[k] = mx;
    }
  }
  return { min: outMin, max: outMax };
}

/**
 * Edge sharpening (morphological toggle contrast).
 *
 * The depth model ramps every building edge over ~5-6 px (~2 m at 0.33 m GSD),
 * so ~14% of the raster sits in the no-man's-land between road level and roof
 * level. Extruded directly, that becomes a dome instead of a box -- the
 * "half-lifted building" look. For each pixel we take the local min and local
 * max over a small window and snap the value to whichever it is closer to, so
 * the ramp collapses to a one-cell vertical wall.
 *
 * Guards:
 *  - only applied where (localMax - localMin) > minStep, so flat ground, gentle
 *    terrain and pitched roofs are left alone;
 *  - `strength` blends between the original and the snapped value, so it can be
 *    backed off if it looks too blocky.
 */
export function sharpenHeights(src, w, h, opts = {}) {
  const { radius = 3, minStep = 1.2, strength = 1 } = opts;
  if (strength <= 0) return src;

  const { min: lo, max: hi } = localMinMax(src, w, h, radius);
  const out = new Float32Array(src.length);

  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    const a = lo[i];
    const b = hi[i];
    if (b - a < minStep) {
      out[i] = v;
      continue;
    }
    const snapped = v - a < b - v ? a : b;
    out[i] = v + (snapped - v) * strength;
  }
  return out;
}

/**
 * Builds the terrain at exaggeration 1.0 (true metres). Vertical exaggeration is
 * applied later as mesh.scale.y, so three's normalMatrix keeps lighting correct
 * for free and no rebuild / normal recomputation is ever needed.
 */
export function buildTerrainGeometry({ heights, width, height, segments, extentX, extentZ }) {
  const geo = new THREE.PlaneGeometry(extentX, extentZ, segments, segments);
  const pos = geo.attributes.position;
  const N = segments + 1;

  for (let iy = 0; iy < N; iy++) {
    const fy = iy / segments;
    const row = iy * N;
    for (let ix = 0; ix < N; ix++) {
      pos.setZ(row + ix, sampleBilinear(heights, width, height, ix / segments, fy));
    }
  }

  geo.rotateX(-Math.PI / 2); // XY plane -> XZ ground, +Z displacement -> +Y up
  geo.computeVertexNormals();
  geo.computeBoundingBox();
  return geo;
}

/**
 * Suggested vertical exaggeration: aim for roughly a 1:12 relief-to-extent
 * ratio so low-rise tiles still read in perspective. A 35 m scene needs none;
 * a 17 m scene over the same 338 m footprint wants ~1.6x.
 */
export function suggestExaggeration(maxHeight, extent) {
  if (!(maxHeight > 0)) return 1;
  const k = extent / 12 / maxHeight;
  return Math.min(1.8, Math.max(1, Math.round(k * 20) / 20));
}

/**
 * Grayscale texture built straight from the height array, for the alignment
 * self-check: every bright rectangle must sit exactly on a raised block.
 * Raster row 0 is written to the LAST texture row so that, with flipY = false,
 * it lands at v = 1 -- matching the geometry's first row.
 */
export function makeHeightCheckTexture(heights, w, h, min, max) {
  const span = Math.max(max - min, 1e-6);
  const buf = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    const src = y * w;
    const dst = (h - 1 - y) * w;
    for (let x = 0; x < w; x++) {
      const g = Math.max(0, Math.min(255, ((heights[src + x] - min) / span) * 255)) | 0;
      const o = (dst + x) * 4;
      buf[o] = g;
      buf[o + 1] = g;
      buf[o + 2] = g;
      buf[o + 3] = 255;
    }
  }
  const tex = new THREE.DataTexture(buf, w, h, THREE.RGBAFormat);
  tex.flipY = false;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

/**
 * MeshStandardMaterial with three additions injected via onBeforeCompile.
 *
 * 1. Per-pixel GEOMETRIC normals from screen-space derivatives of the
 *    interpolated world / view position. Exact flat (faceted) shading with no
 *    extra memory and no non-indexed geometry, which removes the melted-candle
 *    fillets that smooth vertex normals produce across a two-cell ramp.
 * 2. A facade treatment: near-vertical faces get a solid concrete colour
 *    instead of roof texture stretched down the wall, which is what caused the
 *    vertical streaking. Wallness comes from the WORLD-space geometric normal,
 *    so it stays correct at any exaggeration automatically.
 * 3. A viridis height ramp, cross-faded with uColorMix.
 */
export function createTerrainMaterial({ map, maxHeight }) {
  const mat = new THREE.MeshStandardMaterial({
    map,
    roughness: 0.95,
    metalness: 0.0,
  });

  mat.userData.uniforms = {
    uMaxH: { value: Math.max(maxHeight, 0.001) },
    uFlat: { value: 1.0 },   // 0 = smooth vertex normals, 1 = faceted
    uWall: { value: 0.92 },  // facade strength on near-vertical faces
    uColorMix: { value: 0.0 },
    uFacade: { value: new THREE.Color(0.42, 0.41, 0.39) },
  };

  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, mat.userData.uniforms);

    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
        uniform float uMaxH;
        varying vec3 vWPos;
        varying vec3 vVPos;
        varying float vHNorm;`
      )
      .replace(
        '#include <project_vertex>',
        `#include <project_vertex>
        vVPos = mvPosition.xyz;
        vWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
        // object-space height, so the colour ramp ignores exaggeration
        vHNorm = clamp(position.y / uMaxH, 0.0, 1.0);`
      );

    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
        uniform float uFlat;
        uniform float uWall;
        uniform float uColorMix;
        uniform vec3 uFacade;
        varying vec3 vWPos;
        varying vec3 vVPos;
        varying float vHNorm;
        vec3 tViridis(float t) {
          return clamp(vec3(
            0.280 + t * (0.105 + t * (-0.330 + t * 1.900)),
            0.010 + t * (1.410 + t * (-1.010 + t * 0.480)),
            0.330 + t * (1.380 + t * (-3.320 + t * 2.180))
          ), 0.0, 1.0);
        }`
      )
      .replace(
        '#include <map_fragment>',
        `#include <map_fragment>
        // true geometric normal of this triangle, in world space
        vec3 tGeoW = normalize(cross(dFdx(vWPos), dFdy(vWPos)));
        // ~57 to ~72 degrees: below that it is roof or ground, above it is wall
        float tWall = smoothstep(0.45, 0.72, 1.0 - abs(tGeoW.y));

        diffuseColor.rgb = mix(diffuseColor.rgb, tViridis(vHNorm), uColorMix);

        float tLum = dot(diffuseColor.rgb, vec3(0.299, 0.587, 0.114));
        vec3 tFacade = mix(uFacade, vec3(tLum), 0.3) * (0.62 + 0.38 * vHNorm);
        diffuseColor.rgb = mix(diffuseColor.rgb, tFacade, tWall * uWall);`
      )
      .replace(
        '#include <normal_fragment_begin>',
        `#include <normal_fragment_begin>
        vec3 tGeoV = normalize(cross(dFdx(vVPos), dFdy(vVPos)));
        if (tGeoV.z < 0.0) tGeoV = -tGeoV; // keep it facing the camera
        normal = normalize(mix(normal, tGeoV, uFlat));`
      );
  };

  return mat;
}