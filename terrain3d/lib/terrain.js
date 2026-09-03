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
 * MeshStandardMaterial with additions injected via onBeforeCompile.
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
 *
 * OPT-IN EXTRAS (4-6). Every one is gated by a uniform that defaults to 0, and
 * each is written as an identity when its gate is 0 -- mix(x, y, 0) or `if
 * (gate > 0.5)`. With the defaults below the compiled output is the same
 * picture as items 1-3 alone, so nothing here can disturb the base render:
 *
 * 4. uWindows: procedural facade detail -- storeys, windows, shopfront glazing,
 *    frames, slab lines, plinth. Gated per pixel on the building channel of
 *    uSite so it can only ever appear on detected structures, never on terrain
 *    cliffs or tree mounds.
 * 5. uCanopyOn / uCanopyFlatten: tree-clump overlay. The tint recolours mound
 *    walls as foliage; the flatten lowers the RENDERED surface of a mound to
 *    the ground beneath it. Flattening is display-only -- it happens in the
 *    vertex shader and the height array is never written to.
 * 6. uEdge: crease highlight along roof edges, from the screen-space
 *    derivative of the geometric normal. Sharpens edges in shading, where it
 *    cannot damage the reconstruction.
 */
export function createTerrainMaterial({ map, maxHeight, extentX = 1, extentZ = 1 }) {
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

    // world XZ -> mask UV, for uSite and uCanopy
    uExtent: { value: new THREE.Vector2(extentX, extentZ) },

    // --- opt-in extras: every gate is 0, so the base render is untouched ---
    uSite: { value: null },       // R ground level, G building mask, B seed
    uSiteOn: { value: 0.0 },
    uGroundMin: { value: 0.0 },
    uGroundSpan: { value: 1.0 },
    uWindows: { value: 0.0 },     // facade detail strength
    uCanopy: { value: null },     // R clump footprint, G flatten amount
    uCanopyOn: { value: 0.0 },    // foliage tint on mound walls
    uCanopyFlatten: { value: 0.0 },
    uDeltaMax: { value: 0.0 },    // metres, scale of uCanopy.g
    uCanopyTint: { value: new THREE.Color(0.11, 0.17, 0.09) },
    uSky: { value: new THREE.Color(0.62, 0.77, 0.88) }, // glass reflection
    uEdge: { value: 0.0 },        // roof-edge crease highlight
  };

  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, mat.userData.uniforms);

    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
        uniform float uMaxH;
        uniform sampler2D uCanopy;
        uniform float uCanopyFlatten;
        uniform float uDeltaMax;
        uniform vec2 uExtent;
        varying vec3 vWPos;
        varying vec3 vVPos;
        varying float vHNorm;
        varying float vLocalY;`
      )
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
        // Lower a tree mound to the ground beneath it. mesh.scale.y is applied
        // later by the model matrix, so this is in raw metres and stays correct
        // at any exaggeration. Identity while the gate is 0.
        if (uCanopyFlatten > 0.0 && uDeltaMax > 0.0) {
          vec2 tFUv = vec2(transformed.x / uExtent.x + 0.5,
                           transformed.z / uExtent.y + 0.5);
          transformed.y -= texture2D(uCanopy, tFUv).g * uDeltaMax * uCanopyFlatten;
        }`
      )
      .replace(
        '#include <project_vertex>',
        `#include <project_vertex>
        vVPos = mvPosition.xyz;
        vWPos = (modelMatrix * vec4(transformed, 1.0)).xyz;
        // raw metres: facade storeys must not stretch with exaggeration
        vLocalY = transformed.y;
        // object-space height, so the colour ramp ignores exaggeration
        vHNorm = clamp(transformed.y / uMaxH, 0.0, 1.0);`
      );

    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
        uniform float uFlat;
        uniform float uWall;
        uniform float uColorMix;
        uniform vec3 uFacade;
        uniform vec2 uExtent;
        uniform sampler2D uSite;
        uniform float uSiteOn;
        uniform float uGroundMin;
        uniform float uGroundSpan;
        uniform float uWindows;
        uniform sampler2D uCanopy;
        uniform float uCanopyOn;
        uniform vec3 uCanopyTint;
        uniform vec3 uSky;
        uniform float uEdge;
        varying vec3 vWPos;
        varying vec3 vVPos;
        varying float vHNorm;
        varying float vLocalY;
        float tHash(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }
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

        // ---- tree-clump footprint (identity while uCanopyOn is 0) ----
        float tCanopy = 0.0;
        if (uCanopyOn > 0.5) {
          vec2 tCUv = vec2(vWPos.x / uExtent.x + 0.5, vWPos.z / uExtent.y + 0.5);
          tCanopy = texture2D(uCanopy, tCUv).r;
        }
        // inside a clump a wall is shaded foliage, not concrete
        tFacade = mix(tFacade, uCanopyTint * (0.55 + 0.45 * vHNorm), tCanopy);

        diffuseColor.rgb = mix(diffuseColor.rgb, tFacade, tWall * uWall);
        diffuseColor.rgb *= mix(1.0, 0.84, tCanopy * (1.0 - tWall));

        // ---- building facades (identity while uSiteOn or uWindows is 0) ----
        float tGlassAmt = 0.0;
        float tWinLit = 0.0;
        if (uSiteOn > 0.5 && uWindows > 0.0) {
          vec2 tSUv = vec2(vWPos.x / uExtent.x + 0.5, vWPos.z / uExtent.y + 0.5);
          vec4 tSite = texture2D(uSite, tSUv);
          // gated on the building channel, so detail can only land on
          // structures -- never on terrain cliffs or tree mounds. Fades with
          // uWall too, so the height-ramp and overlay views stay clean.
          float tFace = tWall * tSite.g * uWall * uWindows;
          if (tFace > 0.01) {
            float tBase = uGroundMin + tSite.r * uGroundSpan;
            float tSeed = tSite.b;
            float tUp = max(vLocalY - tBase, 0.0); // metres above its own base
            // horizontal coord along the wall: pick the axis the wall does NOT
            // face, so the grid runs continuously round corners
            float tPickZ = step(abs(tGeoW.x), abs(tGeoW.z));
            float tU = mix(vWPos.z, vWPos.x, tPickZ);
            float tFloorH = 3.3 + tSeed * 0.9;   // per-building storey height
            float tBayW = 2.4 + tSeed * 1.2;     // per-building bay width
            vec2 tCell = vec2(tU / tBayW, tUp / tFloorH);
            vec2 tId = floor(tCell);
            vec2 tF = fract(tCell);
            float tR = tHash(tId + tSeed * 37.0);
            float tIsGround = 1.0 - step(tFloorH, tUp);
            float tWw = mix(0.46 + 0.10 * tR, 0.74, tIsGround);
            float tWh = mix(0.42, 0.58, tIsGround);
            float tWy = mix(0.60, 0.44, tIsGround);
            float tHasWin = step(0.12, tR);      // some bays are blank wall
            float tIn = step(abs(tF.x - 0.5), tWw * 0.5)
                      * step(abs(tF.y - tWy), tWh * 0.5) * tHasWin;
            float tFrame = clamp(step(abs(tF.x - 0.5), tWw * 0.5 + 0.055)
                               * step(abs(tF.y - tWy), tWh * 0.5 + 0.055)
                               * tHasWin - tIn, 0.0, 1.0);
            float tSlab = 1.0 - smoothstep(0.0, 0.055, tF.y);
            vec3 tGlass = mix(vec3(0.05, 0.068, 0.088), uSky * 0.6,
                              0.28 + 0.34 * tR);
            vec3 tBuilt = tFacade * (0.94 + 0.13 * tR);
            tBuilt = mix(tBuilt, tFacade * 0.7, tSlab * 0.55);
            tBuilt = mix(tBuilt, tFacade * 1.3 + 0.03, tFrame);
            tBuilt = mix(tBuilt, tGlass, tIn);
            tBuilt = mix(tBuilt, tFacade * 0.58, 1.0 - smoothstep(0.0, 1.1, tUp));
            diffuseColor.rgb = mix(diffuseColor.rgb, tBuilt, tFace);
            tGlassAmt = tIn * tFace;
            tWinLit = tGlassAmt * step(0.9, tHash(tId + tSeed * 91.0))
                    * (1.0 - tIsGround);
          }
        }

        // ---- roof-edge crease (identity while uEdge is 0) ----
        // the geometric normal turns sharply at a roof edge, so its
        // screen-space derivative spikes there
        float tCrease = clamp(length(vec2(dFdx(tGeoW.y), dFdy(tGeoW.y))) * 1.6,
                              0.0, 1.0);
        diffuseColor.rgb += tCrease * 0.055 * uEdge;`
      )
      .replace(
        '#include <roughnessmap_fragment>',
        `#include <roughnessmap_fragment>
        roughnessFactor = mix(roughnessFactor, 0.16, tGlassAmt);`
      )
      .replace(
        '#include <metalnessmap_fragment>',
        `#include <metalnessmap_fragment>
        metalnessFactor = mix(metalnessFactor, 0.42, tGlassAmt);`
      )
      .replace(
        '#include <emissivemap_fragment>',
        `#include <emissivemap_fragment>
        totalEmissiveRadiance += tWinLit * vec3(1.0, 0.84, 0.58) * 0.75;`
      )
      .replace(
        '#include <normal_fragment_begin>',
        `#include <normal_fragment_begin>
        vec3 tGeoV = normalize(cross(dFdx(vVPos), dFdy(vVPos)));
        if (tGeoV.z < 0.0) tGeoV = -tGeoV; // keep it facing the camera
        normal = normalize(mix(normal, tGeoV, uFlat));`
      );
  };

  // Shadows render through a separate depth material, which would otherwise
  // still see the un-flattened mound and cast a cylinder-shaped shadow. This
  // mirrors the vertex displacement and shares the SAME uniform objects, so it
  // cannot drift out of sync with the visible surface.
  const depth = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
  depth.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, mat.userData.uniforms);
    shader.vertexShader = shader.vertexShader
      .replace(
        '#include <common>',
        `#include <common>
        uniform sampler2D uCanopy;
        uniform float uCanopyFlatten;
        uniform float uDeltaMax;
        uniform vec2 uExtent;`
      )
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
        if (uCanopyFlatten > 0.0 && uDeltaMax > 0.0) {
          vec2 tFUv = vec2(transformed.x / uExtent.x + 0.5,
                           transformed.z / uExtent.y + 0.5);
          transformed.y -= texture2D(uCanopy, tFUv).g * uDeltaMax * uCanopyFlatten;
        }`
      );
  };
  mat.userData.depthMaterial = depth;

  return mat;
}

/* ========================================================================= */
/* VISUAL ENRICHMENT                                                        */
/*                                                                           */
/* Everything below this point is deliberately separate from the DSM.       */
/* We NEVER write back into the heightmap.                                  */
/* ========================================================================= */

/* -------------------------------------------------------------------------- */
/* RGB analysis                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Converts the already-decoded RGB source into ImageData.
 *
 * This is used only for object placement.
 * It does NOT modify the terrain texture.
 */
export async function decodeRGBPixels(source, width, height) {
  const canvas = document.createElement('canvas');

  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d', {
    willReadFrequently: true,
  });

  if (!ctx) {
    throw new Error('Could not create RGB analysis canvas.');
  }

  ctx.drawImage(source, 0, 0, width, height);

  return ctx.getImageData(
    0,
    0,
    width,
    height
  );
}

/* -------------------------------------------------------------------------- */
/* RGB / raster orientation                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Puts decoded RGB pixels into RASTER row order.
 *
 * buildTerrainGeometry samples the heightmap at fy = iy/segments, while
 * PlaneGeometry's UV runs v = 1 - fy (verified against the geometry).
 * So for the texture to line up on screen:
 *
 *   texture.flipY === true   -> image row 0 IS raster row 0   (no flip)
 *   texture.flipY === false  -> image row 0 IS raster row h-1 (flip)
 *
 * The second case is what happens when an ImageBitmap is created with
 * imageOrientation: 'flipY' and the texture flag is cleared to match.
 * Drawing that pre-flipped bitmap to a canvas therefore yields pixels
 * upside down relative to the heightmap — and comparing green pixels
 * against mirrored heights scatters objects onto rooftops.
 *
 * This is decided from the flag, not guessed from the data: a large
 * lawn landing on a large roof can outscore the correct orientation.
 */
export function orientImageData(imageData, textureFlipY) {
  const flip = textureFlipY === false;

  if (!flip) return imageData;

  const { width: w, height: h, data } = imageData;

  const out = new Uint8ClampedArray(data.length);

  const rowBytes = w * 4;

  for (let y = 0; y < h; y++) {
    const src = (h - 1 - y) * rowBytes;

    out.set(
      data.subarray(src, src + rowBytes),
      y * rowBytes
    );
  }

  return {
    width: w,
    height: h,
    data: out,
  };
}

/* -------------------------------------------------------------------------- */
/* Spatial helpers                                                           */
/* -------------------------------------------------------------------------- */

function worldFromPixel(
  x,
  y,
  width,
  height,
  extentX,
  extentZ
) {
  return {
    x:
      (x / (width - 1) - 0.5) *
      extentX,

    z:
      (y / (height - 1) - 0.5) *
      extentZ,
  };
}

/* -------------------------------------------------------------------------- */
/* Tree geometry                                                             */
/* -------------------------------------------------------------------------- */

function makeTreeVariant(variant) {
  const group = new THREE.Group();

  const trunkMaterial =
    new THREE.MeshStandardMaterial({
      color:
        variant === 0
          ? 0x5a3825
          : variant === 1
            ? 0x68452d
            : 0x4c3022,

      roughness: 1,
      metalness: 0,
    });

  const leafMaterial =
    new THREE.MeshStandardMaterial({
      color:
        variant === 0
          ? 0x315d32
          : variant === 1
            ? 0x3d7139
            : 0x28542d,

      roughness: 0.95,
      metalness: 0,
    });

  const trunkHeight =
    variant === 0
      ? 3.2
      : variant === 1
        ? 2.6
        : 3.8;

  const trunkRadius =
    variant === 2
      ? 0.42
      : 0.34;

  const trunk = new THREE.Mesh(
    new THREE.CylinderGeometry(
      trunkRadius,
      trunkRadius * 1.25,
      trunkHeight,
      6
    ),
    trunkMaterial
  );

  trunk.position.y =
    trunkHeight / 2;

  trunk.castShadow = true;
  trunk.receiveShadow = true;

  group.add(trunk);

  const crownHeight =
    variant === 1
      ? 5.8
      : variant === 2
        ? 7.0
        : 6.3;

  const crownRadius =
    variant === 1
      ? 2.5
      : variant === 2
        ? 2.8
        : 2.4;

  if (variant === 0) {
    const crown = new THREE.Mesh(
      new THREE.ConeGeometry(
        crownRadius,
        crownHeight,
        7
      ),
      leafMaterial
    );

    crown.position.y =
      trunkHeight +
      crownHeight * 0.48;

    crown.castShadow = true;
    crown.receiveShadow = true;

    group.add(crown);
  } else if (variant === 1) {
    for (let i = 0; i < 2; i++) {
      const crown = new THREE.Mesh(
        new THREE.IcosahedronGeometry(
          crownRadius * (1 - i * 0.12),
          1
        ),
        leafMaterial
      );

      crown.position.set(
        (i === 0 ? -0.35 : 0.35),
        trunkHeight +
        2.5 +
        i * 1.4,
        i === 0 ? 0 : 0.2
      );

      crown.castShadow = true;
      crown.receiveShadow = true;

      group.add(crown);
    }
  } else {
    const crown = new THREE.Mesh(
      new THREE.IcosahedronGeometry(
        crownRadius,
        1
      ),
      leafMaterial
    );

    crown.position.y =
      trunkHeight +
      crownRadius;

    crown.scale.y = 1.25;

    crown.castShadow = true;
    crown.receiveShadow = true;

    group.add(crown);
  }

  return group;
}

/* -------------------------------------------------------------------------- */
/* Deterministic hash                                                         */
/* -------------------------------------------------------------------------- */

/*
 * Stable pseudo-random value from raster coordinates, so a rebuild
 * with the same inputs produces the same forest (no re-shuffling).
 */
function hash2D(x, y) {
  const s =
    Math.sin(x * 127.1 + y * 311.7) *
    43758.5453;

  return s - Math.floor(s);
}

/* -------------------------------------------------------------------------- */
/* Fast separable morphology                                                  */
/* -------------------------------------------------------------------------- */

/*
 * O(n) sliding-window min/max along one axis using a monotonic deque.
 * Used to build a ground-surface estimate cheaply, even at large radii.
 */
function extreme1D(
  src,
  dst,
  n,
  stride,
  offset,
  r,
  isMin,
  dq
) {
  let head = 0;
  let tail = 0;

  for (let i = 0; i < n + r; i++) {
    if (i < n) {
      const v = src[offset + i * stride];

      while (tail > head) {
        const w =
          src[offset + dq[tail - 1] * stride];

        if (isMin ? w >= v : w <= v) {
          tail--;
        } else {
          break;
        }
      }

      dq[tail++] = i;
    }

    const j = i - r;

    if (j >= 0) {
      while (dq[head] < j - r) head++;

      dst[offset + j * stride] =
        src[offset + dq[head] * stride];
    }
  }
}

function morph2D(src, w, h, r, isMin) {
  if (r < 1) return src.slice();

  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);

  const dq = new Int32Array(Math.max(w, h));

  for (let y = 0; y < h; y++) {
    extreme1D(src, tmp, w, 1, y * w, r, isMin, dq);
  }

  for (let x = 0; x < w; x++) {
    extreme1D(tmp, out, h, w, x, r, isMin, dq);
  }

  return out;
}

/**
 * Ground-surface (bare-earth) estimate: a morphological opening.
 *
 * Erosion pushes the surface down under every object smaller than the
 * radius; the matching dilation restores the terrain's own slopes. What
 * remains is roughly "the street level beneath this pixel".
 */
function groundSurface(heights, w, h, r) {
  const eroded = morph2D(heights, w, h, r, true);

  return morph2D(eroded, w, h, r, false);
}

/* -------------------------------------------------------------------------- */
/* Statistics                                                                 */
/* -------------------------------------------------------------------------- */

function samplePercentile(values, p, step = 1) {
  const picked = [];

  for (let i = 0; i < values.length; i += step) {
    const v = values[i];

    if (Number.isFinite(v)) picked.push(v);
  }

  if (!picked.length) return 0;

  picked.sort((a, b) => a - b);

  const k = Math.min(
    picked.length - 1,
    Math.max(0, Math.round((p / 100) * (picked.length - 1)))
  );

  return picked[k];
}

/**
 * Otsu's method: split a set of values into two classes by maximising
 * between-class variance.
 *
 * Used to separate green canopy from grey roof without any absolute
 * threshold. Percentiles were tried first and don't work here: one
 * bright sunlit lawn owns the top of the distribution and drags the
 * cut above every canopy sitting in shade. Otsu looks for the gap
 * between the two populations instead of a fixed position in the tail.
 */
function otsuSplit(values, mask, n, lo, hi, bins = 192) {
  const span = hi - lo;

  if (!(span > 1e-9)) return null;

  const hist = new Float64Array(bins);

  let total = 0;

  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;

    const v = values[i];

    if (!Number.isFinite(v)) continue;

    let b = Math.floor(((v - lo) / span) * bins);

    if (b < 0) b = 0;
    if (b >= bins) b = bins - 1;

    hist[b]++;
    total++;
  }

  if (total < 32) return null;

  let sumAll = 0;

  for (let b = 0; b < bins; b++) sumAll += hist[b] * b;

  let wB = 0;
  let sumB = 0;

  let best = -1;
  let bestB = 0;

  for (let b = 0; b < bins - 1; b++) {
    wB += hist[b];

    if (wB === 0) continue;

    const wF = total - wB;

    if (wF === 0) break;

    sumB += hist[b] * b;

    const mB = sumB / wB;
    const mF = (sumAll - sumB) / wF;

    const between = wB * wF * (mB - mF) * (mB - mF);

    if (between > best) {
      best = between;
      bestB = b;
    }
  }

  const threshold = lo + ((bestB + 1) / bins) * span;

  let above = 0;
  let meanAbove = 0;
  let meanBelow = 0;
  let belowCount = 0;

  for (let b = 0; b < bins; b++) {
    const centre = lo + ((b + 0.5) / bins) * span;

    if (b > bestB) {
      above += hist[b];
      meanAbove += hist[b] * centre;
    } else {
      belowCount += hist[b];
      meanBelow += hist[b] * centre;
    }
  }

  return {
    threshold,
    fractionAbove: above / total,
    meanAbove: above ? meanAbove / above : 0,
    meanBelow: belowCount ? meanBelow / belowCount : 0,
    separation:
      (above ? meanAbove / above : 0) -
      (belowCount ? meanBelow / belowCount : 0),
  };
}

/* -------------------------------------------------------------------------- */
/* Connected components                                                       */
/* -------------------------------------------------------------------------- */

/*
 * 8-connected labelling with an explicit stack (no recursion, so a
 * city-sized blob can't blow the call stack).
 */
function labelComponents(mask, w, h, minPixels) {
  const labels = new Int32Array(mask.length).fill(-1);

  const stack = new Int32Array(mask.length);

  const blobs = [];

  for (let seed = 0; seed < mask.length; seed++) {
    if (!mask[seed] || labels[seed] !== -1) continue;

    const id = blobs.length;

    let sp = 0;

    stack[sp++] = seed;
    labels[seed] = id;

    const pixels = [];

    while (sp > 0) {
      const i = stack[--sp];

      pixels.push(i);

      const x = i % w;
      const y = (i - x) / w;

      for (let oy = -1; oy <= 1; oy++) {
        const ny = y + oy;

        if (ny < 0 || ny >= h) continue;

        for (let ox = -1; ox <= 1; ox++) {
          const nx = x + ox;

          if (nx < 0 || nx >= w) continue;

          const k = ny * w + nx;

          if (!mask[k] || labels[k] !== -1) continue;

          labels[k] = id;

          stack[sp++] = k;
        }
      }
    }

    if (pixels.length < minPixels) {
      /* Too small to be anything — unlabel it again. */
      for (const i of pixels) labels[i] = -1;
      continue;
    }

    blobs.push({
      id,
      pixels: Int32Array.from(pixels),
    });
  }

  return { labels, blobs };
}

/* -------------------------------------------------------------------------- */
/* Canopy mask + flatten texture                                              */
/* -------------------------------------------------------------------------- */

/* Separable box blur, used to soften both the mask and the flatten ramp. */
function boxBlur(src, w, h, r) {
  if (r < 1) return src;

  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);

  for (let y = 0; y < h; y++) {
    const row = y * w;

    for (let x = 0; x < w; x++) {
      let sum = 0;
      let c = 0;

      for (let o = -r; o <= r; o++) {
        const nx = x + o;

        if (nx < 0 || nx >= w) continue;

        sum += src[row + nx];
        c++;
      }

      tmp[row + x] = sum / c;
    }
  }

  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      let sum = 0;
      let c = 0;

      for (let o = -r; o <= r; o++) {
        const ny = y + o;

        if (ny < 0 || ny >= h) continue;

        sum += tmp[ny * w + x];
        c++;
      }

      out[y * w + x] = sum / c;
    }
  }

  return out;
}

/**
 * Packs the tree-clump overlay data into one RGBA texture:
 *
 *   R = soft clump footprint, slightly dilated. The fragment shader
 *       uses it to tint any remaining mound wall as foliage.
 *   G = how far to lower the terrain here, normalised by deltaMax.
 *       The vertex shader uses it to flatten the merged canopy mound
 *       down to the ground beneath it, so the trees stand on open
 *       ground instead of on top of a green cylinder.
 *
 * The delta is stored in a single 8-bit channel scaled to the largest
 * delta actually present, which keeps quantisation around 5 cm and —
 * unlike a 16-bit split across two channels — stays correct under
 * linear filtering.
 *
 * This lowers the RENDERED surface only. The heightmap is never
 * written to; the delta is applied in the shader and subtracted in the
 * collision sampler so the two agree.
 */
/**
 * Builds the two canopy channels as plain arrays.
 *
 * Split out from the texture so it can be tested directly, because the
 * ordering here is load-bearing: the blur runs BEFORE the building
 * protection, never after. Blurring a delta that stops at a wall
 * spreads lowering back across that wall by the blur radius, which
 * bites a notch out of the building. Protection is therefore applied
 * last, as a hard multiply.
 */
export function composeCanopyChannels(
  mask,
  delta,
  protect,
  w,
  h,
  dilatePx
) {
  const hard = Float32Array.from(mask);

  /* Dilated + softened footprint, for wall tinting. */
  let tint = boxBlur(
    morph2D(hard, w, h, dilatePx, false),
    w,
    h,
    Math.max(1, Math.round(dilatePx * 0.75))
  );

  /*
   * Flatten ramp: NOT dilated, so surrounding ground is never pulled
   * down. Blurring the delta and multiplying by the blurred footprint
   * gives a smooth shoulder at the clump edge instead of a cliff.
   */
  const soft = boxBlur(hard, w, h, dilatePx);

  const deltaBlur = boxBlur(delta, w, h, dilatePx);

  const out = new Float32Array(w * h);

  let deltaMax = 0;

  for (let i = 0; i < out.length; i++) {
    /*
     * Hard stop on and around buildings. Applied after every blur so
     * nothing can leak into a wall, and applied to the tint too — a
     * building must never be painted as foliage either.
     */
    if (protect && protect[i]) {
      out[i] = 0;
      tint[i] = 0;
      continue;
    }

    const v = deltaBlur[i] * soft[i];

    out[i] = v;

    if (v > deltaMax) deltaMax = v;
  }

  return { tint, delta: out, deltaMax };
}

function buildCanopyTexture(
  mask,
  delta,
  protect,
  w,
  h,
  dilatePx
) {
  const hard = Float32Array.from(mask);

  const {
    tint,
    delta: deltaOut,
    deltaMax,
  } = composeCanopyChannels(
    mask,
    delta,
    protect,
    w,
    h,
    dilatePx
  );

  const inv = deltaMax > 1e-6 ? 1 / deltaMax : 0;

  const buf = new Uint8Array(w * h * 4);

  for (let i = 0; i < hard.length; i++) {
    const o = i * 4;

    buf[o] =
      Math.max(0, Math.min(255, Math.round(tint[i] * 255))) | 0;

    buf[o + 1] =
      Math.max(
        0,
        Math.min(
          255,
          Math.round(deltaOut[i] * inv * 255)
        )
      ) | 0;

    buf[o + 2] = 0;
    buf[o + 3] = 255;
  }

  const tex = new THREE.DataTexture(
    buf,
    w,
    h,
    THREE.RGBAFormat
  );

  /*
   * Sampled from world XZ (not the mesh UVs), so raster row 0 must map
   * to v = 0 — hence flipY stays false.
   */
  tex.flipY = false;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;

  return { texture: tex, deltaMax };
}

/**
 * Packs per-pixel building information into one RGBA texture:
 *
 *   R = ground level under this pixel, normalised — lets the shader
 *       measure height above a building's own base, so storeys line up
 *       from the pavement instead of from sea level.
 *   G = building mask, gating facade detail to actual structures.
 *   B = per-building seed, so every block gets its own storey height,
 *       bay width and window rhythm.
 *
 * Nearest filtering on purpose: interpolating the seed channel across
 * a boundary would blend two buildings' grids into a seam.
 */
function buildSiteTexture(
  buildingMask,
  buildingSeed,
  groundBase,
  w,
  h
) {
  const n = w * h;

  let lo = Infinity;
  let hi = -Infinity;

  for (let i = 0; i < n; i++) {
    const v = groundBase[i];

    if (!Number.isFinite(v)) continue;

    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }

  if (!Number.isFinite(lo)) {
    lo = 0;
    hi = 1;
  }

  const span = Math.max(hi - lo, 0.001);

  const buf = new Uint8Array(n * 4);

  for (let i = 0; i < n; i++) {
    const o = i * 4;

    buf[o] =
      Math.max(
        0,
        Math.min(
          255,
          Math.round(((groundBase[i] - lo) / span) * 255)
        )
      ) | 0;

    buf[o + 1] = buildingMask[i] ? 255 : 0;
    buf[o + 2] = buildingSeed[i];
    buf[o + 3] = 255;
  }

  const tex = new THREE.DataTexture(buf, w, h, THREE.RGBAFormat);

  tex.flipY = false;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;

  return { texture: tex, groundMin: lo, groundSpan: span };
}

/* -------------------------------------------------------------------------- */
/* Vegetation index                                                           */
/* -------------------------------------------------------------------------- */

/**
 * Excess-green index, normalised by total brightness.
 *
 *   (2g - r - b) / (r + g + b)
 *
 * Dividing by brightness is the important part: a canopy in shadow and
 * the same canopy in sunlight give nearly the same value, whereas raw
 * green-dominance collapses in shade. Real scenes have olive/khaki
 * canopies, not saturated green, so absolute thresholds don't travel —
 * this index is only ever compared against the scene's own
 * distribution (see planTreeClumps).
 */
function excessGreen(data, i) {
  const r = data[i];
  const g = data[i + 1];
  const b = data[i + 2];

  const sum = r + g + b;

  if (sum < 24) return 0;

  return (2 * g - r - b) / sum;
}

/* -------------------------------------------------------------------------- */
/* Tree clump detection                                                       */
/* -------------------------------------------------------------------------- */

/**
 * Finds tree clumps and decides where individual trees go.
 *
 * The DSM merges neighbouring trees into one smooth green-capped mound,
 * so this works on whole mounds rather than pixels:
 *
 *   1. Estimate the bare-earth surface (morphological opening) and
 *      subtract it -> nDSM, "height above the street beneath me".
 *   2. Score greenness with a brightness-normalised excess-green index,
 *      and threshold it against THIS scene's own distribution.
 *   3. Mark pixels that are both raised and green, and label them into
 *      connected blobs. Each blob is one mound.
 *   4. Accept a blob as a tree clump only if it stands ON THE GROUND:
 *        - the low percentile of nDSM over the blob and a skirt around
 *          it must be near zero (a rooftop patch sits on a raised
 *          surface, so it fails);
 *        - the local ground estimate must not itself be a platform
 *          (catches roofs too wide for the opening to escape).
 *      Deliberately not a "is it surrounded by tall things" test — that
 *      cannot tell a courtyard from a rooftop, and courtyards are
 *      exactly where trees live.
 *   5. Fill each accepted mound with a spaced group of trees, sized to
 *      the mound, so the mound is hidden inside the foliage.
 *
 * The height band is derived from the scene: the tallest structure sets
 * the scale, and trees live between roughly "taller than a car" and the
 * midpoint up to that tallest structure.
 *
 * Nothing here writes back into the heightmap.
 */
export function planTreeClumps({
  imageData,
  width,
  height,
  detect,
  place,
  extentX,
  extentZ,
  verticalScale = 1,
  maxTrees = 1200,
  minTreeHeight = null,
  maxTreeHeight = null,
  greenThreshold = null,
}) {
  /*
   * imageData must already be in raster row order — see
   * orientImageData. Orientation is not guessed here.
   */
  const n = width * height;

  const pxPerM = Math.min(width / extentX, height / extentZ);

  const mPerPx = 1 / pxPerM;

  const placeHeights = place || detect;

  /* ---------------------------------------------------------------- */
  /* 1. Bare-earth estimate and normalised DSM                        */
  /* ---------------------------------------------------------------- */

  /*
   * ~22 m: wider than a tree clump or a small building, so the opening
   * reaches genuine street level, but narrow enough that real terrain
   * slope isn't mistaken for object height.
   */
  const groundRadius = Math.max(
    3,
    Math.min(90, Math.round(22 * pxPerM))
  );

  /* ~60 m: escapes even a large building footprint. */
  const districtRadius = Math.max(
    groundRadius + 2,
    Math.min(220, Math.round(60 * pxPerM))
  );

  const groundDetect = groundSurface(
    detect,
    width,
    height,
    groundRadius
  );

  const districtDetect = groundSurface(
    detect,
    width,
    height,
    districtRadius
  );

  const groundPlace =
    placeHeights === detect
      ? groundDetect
      : groundSurface(
          placeHeights,
          width,
          height,
          groundRadius
        );

  const nDSM = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    nDSM[i] = detect[i] - groundDetect[i];
  }

  /* ---------------------------------------------------------------- */
  /* 2. Scene-relative height band                                    */
  /* ---------------------------------------------------------------- */

  const tallest = Math.max(
    3,
    samplePercentile(nDSM, 99.5, 7)
  );

  /* Anything under this is a car, a kerb, or DSM noise. */
  const objectFloor = Math.max(1.2, tallest * 0.035);

  const bandLow =
    minTreeHeight != null
      ? minTreeHeight
      : Math.max(2.0, tallest * 0.05);

  const bandHigh =
    maxTreeHeight != null
      ? maxTreeHeight
      : Math.min(
          35,
          Math.max(12, tallest * 0.65)
        );

  const raised = new Uint8Array(n);

  let raisedCount = 0;

  for (let i = 0; i < n; i++) {
    if (nDSM[i] >= objectFloor) {
      raised[i] = 1;
      raisedCount++;
    }
  }

  /* ---------------------------------------------------------------- */
  /* 3. Greenness, with the RGB/raster orientation verified           */
  /* ---------------------------------------------------------------- */

  const pix = imageData.data;

  const exgRaw = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    exgRaw[i] = excessGreen(pix, i * 4);
  }


  /* Mild smoothing: fills single-pixel gaps without erasing canopies. */
  const gr = Math.max(1, Math.round(0.8 * pxPerM));

  const exgTmp = new Float32Array(n);
  const exg = new Float32Array(n);

  for (let y = 0; y < height; y++) {
    const row = y * width;

    for (let x = 0; x < width; x++) {
      let sum = 0;
      let c = 0;

      for (let o = -gr; o <= gr; o++) {
        const nx = x + o;

        if (nx < 0 || nx >= width) continue;

        sum += exgRaw[row + nx];
        c++;
      }

      exgTmp[row + x] = sum / c;
    }
  }

  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      let sum = 0;
      let c = 0;

      for (let o = -gr; o <= gr; o++) {
        const ny = y + o;

        if (ny < 0 || ny >= height) continue;

        sum += exgTmp[ny * width + x];
        c++;
      }

      exg[y * width + x] = sum / c;
    }
  }

  /* ---------------------------------------------------------------- */
  /* 4. Scene-adaptive green threshold                                */
  /* ---------------------------------------------------------------- */

  /*
   * Measured on real output: shadowed olive canopies score ~0.08-0.12
   * on this index while asphalt and concrete sit near 0.02. The gap is
   * reliable; its absolute position is not.
   *
   * The split is computed over RAISED pixels only. That matters: lawns
   * are the greenest thing in most scenes but they are flat, so
   * including them pushes the cut above every canopy in shade. Among
   * raised pixels the two populations are precisely the ones that need
   * separating — grey roofs and green canopies.
   */
  const step = Math.max(1, Math.floor(n / 240000));

  const exgBackground = samplePercentile(exg, 50, step);

  const split = otsuSplit(
    exg,
    raised,
    n,
    Math.min(exgBackground, 0),
    Math.max(
      exgBackground + 0.06,
      samplePercentile(exg, 99.5, step)
    )
  );

  const autoThreshold = split
    ? Math.max(
        split.threshold,
        exgBackground + 0.015
      )
    : exgBackground + 0.05;

  const greenCut =
    greenThreshold != null
      ? greenThreshold
      : autoThreshold;

  /*
   * Guard against a scene with no vegetation at all, where Otsu would
   * happily split noise in half. Require a real gap between the two
   * classes, and a plausible share of raised pixels on the green side.
   */
  const hasVegetation =
    greenThreshold != null ||
    (!!split &&
      split.separation >= 0.03 &&
      split.fractionAbove >= 0.001 &&
      split.fractionAbove <= 0.7);

  /* ---------------------------------------------------------------- */
  /* 5. Building mask                                                 */
  /* ---------------------------------------------------------------- */

  /*
   * Definite built structure: standing well above the district ground
   * and not green. Measuring against the WIDE opening matters here —
   * the narrow one cannot escape a large footprint, so a big roof
   * would read as ground level.
   *
   * This exists for two reasons. It keeps tree clumps from claiming
   * building pixels, so flattening a clump can never carve a bite out
   * of a roof. And it gates the facade shading, so windows appear on
   * buildings and not on terrain cliffs.
   */
  const buildingMinH = Math.max(2.5, objectFloor * 1.4);

  const buildingRaw = new Uint8Array(n);

  for (let i = 0; i < n; i++) {
    if (detect[i] - districtDetect[i] <= buildingMinH) continue;
    if (hasVegetation && exg[i] >= greenCut) continue;

    buildingRaw[i] = 1;
  }

  /* Drop speckle: a building is at least ~12 m². */
  const minBuildingPixels = Math.max(
    8,
    Math.round(12 * pxPerM * pxPerM)
  );

  const buildings = labelComponents(
    buildingRaw,
    width,
    height,
    minBuildingPixels
  );

  const buildingMask = new Uint8Array(n);

  /*
   * A stable per-building random value, so each block gets its own
   * storey height, bay width and window rhythm instead of the whole
   * city sharing one grid.
   */
  const buildingSeed = new Uint8Array(n);

  /*
   * Thickness of the blob each pixel belongs to, as 2 x area /
   * perimeter — roughly how wide the shape is.
   *
   * A tree mound's shadowed flank also lands in buildingRaw, and this
   * is what separates the two: a flank is a thin skirt wrapped around
   * a canopy, a couple of metres wide however long it runs, while a
   * building is chunky. Area alone does not work — a hedgerow flank
   * along a street is hundreds of square metres, the same as a small
   * building.
   */
  const buildingThickness = new Float32Array(n);

  for (const b of buildings.blobs) {
    const x0 = b.pixels[0] % width;
    const y0 = (b.pixels[0] - x0) / width;

    const seed = Math.round(hash2D(x0 * 0.37, y0 * 0.73) * 255);

    /* Perimeter: blob pixels with at least one neighbour outside. */
    let perim = 0;

    for (const p of b.pixels) {
      const x = p % width;
      const y = (p - x) / width;

      let edge = false;

      for (let k = 0; k < 4 && !edge; k++) {
        const nx = x + (k === 0 ? -1 : k === 1 ? 1 : 0);
        const ny = y + (k === 2 ? -1 : k === 3 ? 1 : 0);

        if (
          nx < 0 || nx >= width ||
          ny < 0 || ny >= height ||
          !buildingRaw[ny * width + nx]
        ) {
          edge = true;
        }
      }

      if (edge) perim++;
    }

    const thickness =
      perim > 0
        ? (2 * b.pixels.length / perim) * mPerPx
        : b.pixels.length * mPerPx;

    for (const p of b.pixels) {
      buildingMask[p] = 1;
      buildingSeed[p] = seed;
      buildingThickness[p] = thickness;
    }
  }

  /* ---------------------------------------------------------------- */
  /* 6. Raised-and-green mask, then blobs                             */
  /* ---------------------------------------------------------------- */

  const vegMask = new Uint8Array(n);

  if (hasVegetation) {
    for (let i = 0; i < n; i++) {
      if (!raised[i]) continue;
      if (exg[i] < greenCut) continue;

      /*
       * Exclude building pixels, but NOT a dilated ring — dilation
       * here would shave the edges off legitimate canopy caps that
       * happen to sit beside a wall.
       */
      if (buildingMask[i]) continue;

      vegMask[i] = 1;
    }
  }

  /* A clump smaller than ~3 m² is noise, not a tree. */
  const minBlobPixels = Math.max(
    4,
    Math.round(3 * pxPerM * pxPerM)
  );

  const { labels, blobs } = labelComponents(
    vegMask,
    width,
    height,
    minBlobPixels
  );

  /* ---------------------------------------------------------------- */
  /* 7. Classify each blob                                            */
  /* ---------------------------------------------------------------- */

  /*
   * Each blob is only the canopy CAP — the pixels that are green and
   * raised. The DSM mound underneath is wider: its flanks are
   * shadowed, or carry smeared road texture, so they are not green.
   *
   * So every blob is first grown across the contiguous raised ground
   * around it, giving the mound's real extent. That region is used for
   * two things:
   *
   *   - the footing test. A fixed-radius ring around the cap does not
   *     work: on a wide ridge the ring never escapes the mound, so the
   *     footing reads as elevated and a perfectly good clump gets
   *     rejected as "on a structure". Measured over the whole mound
   *     instead, the low end reaches the street where the mound meets
   *     it, and stays high when the mound is really a rooftop patch —
   *     because the fill is blocked by the building guard and so
   *     cannot escape the roof.
   *
   *   - the flatten footprint, so the whole mound is levelled instead
   *     of just its cap, which would leave the flanks standing as a
   *     ridge with trees perched on top.
   */
  const growLimit = Math.max(2, Math.round(9 * pxPerM));

  const skirtFloor = Math.max(0.6, objectFloor * 0.5);

  /* Visit stamps, so the fill needs no clearing between blobs. */
  const visit = new Int32Array(n);
  const seenEdge = new Int32Array(n);
  const depth = new Int32Array(n);
  const queue = new Int32Array(n);

  /*
   * Above this width, a non-green blob is real built structure and the
   * mound fill will not cross it. Below it, the blob is a mound skirt.
   */
  const thinFlankM = 6.0;

  const growMound = (blob, capCeiling) => {
    const stamp = blob.id + 1;

    let qh = 0;
    let qt = 0;

    for (const p of blob.pixels) {
      visit[p] = stamp;
      depth[p] = 0;
      queue[qt++] = p;
    }

    const extra = [];

    /*
     * Where the mound runs out and open ground begins. This is the
     * mound's footing: the mound's own lowest pixel is no use, since a
     * steep-sided mound starts several metres up, and its true base is
     * the ground beside it rather than any pixel within it.
     */
    const groundEdge = [];

    let blockedByBuilding = 0;

    while (qh < qt) {
      const i = queue[qh++];

      const d = depth[i];

      if (d >= growLimit) continue;

      const x = i % width;
      const y = (i - x) / width;

      for (let k = 0; k < 4; k++) {
        const nx = x + (k === 0 ? -1 : k === 1 ? 1 : 0);
        const ny = y + (k === 2 ? -1 : k === 3 ? 1 : 0);

        if (nx < 0 || nx >= width) continue;
        if (ny < 0 || ny >= height) continue;

        const q = ny * width + nx;

        if (visit[q] === stamp) continue;

        /*
         * A chunky structure is a hard wall for the fill. Thin
         * non-green blobs are not: those are overwhelmingly the
         * mound's own shadowed flanks, which is precisely what needs
         * levelling.
         */
        if (
          buildingMask[q] &&
          buildingThickness[q] > thinFlankM
        ) {
          blockedByBuilding++;
          continue;
        }

        /*
         * Never climb above the canopy that seeded this fill. A mound
         * flank is lower than its own cap; a building next to it is
         * not. This is what stops the fill walking up a wall and
         * levelling part of a building.
         */
        if (nDSM[q] > capCeiling) {
          blockedByBuilding++;
          continue;
        }

        /*
         * Not raised any more: the mound ends here, so this pixel is
         * open ground and records the footing.
         */
        if (nDSM[q] <= skirtFloor) {
          if (seenEdge[q] !== stamp) {
            seenEdge[q] = stamp;
            groundEdge.push(nDSM[q]);
          }

          continue;
        }

        visit[q] = stamp;
        depth[q] = d + 1;

        queue[qt++] = q;

        extra.push(q);
      }
    }

    return { extra, groundEdge, blockedByBuilding };
  };

  const accepted = [];

  const rejected = {
    tooLow: 0,
    tooTall: 0,
    onStructure: 0,
    onPlatform: 0,
    tooBig: 0,
  };

  for (const blob of blobs) {
    const px = blob.pixels;

    const own = new Float32Array(px.length);

    let platformSum = 0;

    for (let i = 0; i < px.length; i++) {
      own[i] = nDSM[px[i]];

      platformSum +=
        groundDetect[px[i]] - districtDetect[px[i]];
    }

    const topH = samplePercentile(own, 88);

    /*
     * Is the local ground estimate itself a platform? On a building
     * too wide for the 22 m opening to escape, the "ground" under a
     * rooftop bush is the roof, so nDSM alone would call it a 3 m
     * tree. The wider estimate escapes and exposes the lift.
     */
    const platformLift = platformSum / px.length;

    if (platformLift > Math.max(2.5, objectFloor)) {
      rejected.onPlatform++;
      continue;
    }

    /* The mound this cap sits on, plus where it meets open ground. */
    const { extra, groundEdge } = growMound(
      blob,
      topH * 1.3 + 1.0
    );

    /*
     * A mound that never reaches open ground is standing on
     * something. A rooftop patch is exactly this case: the fill is
     * walled in by the building guard on every side and so can never
     * find a footing, while a mound on the street finds one as soon
     * as it runs out of height — even if a building presses against
     * its other side.
     */
    const minEdge = Math.max(3, Math.round(pxPerM));

    if (groundEdge.length < minEdge) {
      rejected.onStructure++;
      continue;
    }

    const footing = Math.max(
      0,
      samplePercentile(
        Float32Array.from(groundEdge),
        25
      )
    );

    if (footing > Math.max(1.8, objectFloor * 1.15)) {
      rejected.onStructure++;
      continue;
    }

    const relH = topH - footing;

    if (relH < bandLow) {
      rejected.tooLow++;
      continue;
    }

    if (relH > bandHigh) {
      rejected.tooTall++;
      continue;
    }

    const areaM2 = px.length * mPerPx * mPerPx;

    if (areaM2 > 60000) {
      rejected.tooBig++;
      continue;
    }

    accepted.push({
      blob,
      extra,
      relH,
      areaM2,
    });
  }

  /* ---------------------------------------------------------------- */
  /* 8. Reclaim mound flanks from the building mask                   */
  /* ---------------------------------------------------------------- */

  /*
   * A mound's shadowed flank looks exactly like built structure to a
   * colour test: raised and not green. Now that the clumps are known,
   * every pixel belonging to an accepted mound is taken back out of
   * the building mask.
   *
   * This has to happen here rather than earlier, and it matters twice
   * over: those flanks would otherwise be protected from flattening —
   * leaving the ridge standing under the trees, which is exactly the
   * artefact this fixes — and they would also be handed to the facade
   * shader, growing windows on a tree mound.
   */
  let reclaimed = 0;

  for (const item of accepted) {
    for (const p of item.blob.pixels) {
      if (buildingMask[p]) {
        buildingMask[p] = 0;
        buildingSeed[p] = 0;
        reclaimed++;
      }
    }

    for (const p of item.extra) {
      if (buildingMask[p]) {
        buildingMask[p] = 0;
        buildingSeed[p] = 0;
        reclaimed++;
      }
    }
  }

  /*
   * Guard rings, built from the CLEANED mask so they protect real
   * buildings and nothing else.
   */
  const guardPx = Math.max(1, Math.round(1.2 * pxPerM));

  const buildingGuardF = morph2D(
    Float32Array.from(buildingMask),
    width,
    height,
    guardPx,
    false
  );

  const buildingGuard = new Uint8Array(n);

  for (let i = 0; i < n; i++) {
    if (buildingGuardF[i] > 0.5) buildingGuard[i] = 1;
  }

  /*
   * A wider ring, covering the guard plus the blur shoulder the
   * canopy channels apply later. Everything inside it is off limits:
   * no lowering, no foliage tint, and no trees planted — a canopy
   * this close to a wall would intersect it anyway.
   */
  const protectPx =
    guardPx + Math.max(1, Math.round(1.6 * pxPerM));

  const protectF = morph2D(
    Float32Array.from(buildingMask),
    width,
    height,
    protectPx,
    false
  );

  const protectMask = new Uint8Array(n);

  for (let i = 0; i < n; i++) {
    if (protectF[i] > 0.5) protectMask[i] = 1;
  }

  /* ---------------------------------------------------------------- */
  /* 9. Fill each clump with a spaced group of trees                  */
  /* ---------------------------------------------------------------- */

  /* Bigger clumps first, so the cap spends itself where it shows. */
  accepted.sort((a, b) => b.areaM2 - a.areaM2);

  const spots = [];

  const clumpMask = new Uint8Array(n);

  /*
   * How far the rendered surface should drop at each clump pixel, so
   * the merged canopy mound is replaced by the ground under it.
   */
  const flattenDelta = new Float32Array(n);

  const cellIndex = new Map();

  const addToGrid = (cell, idx) => {
    let list = cellIndex.get(cell);

    if (!list) {
      list = [];
      cellIndex.set(cell, list);
    }

    list.push(idx);
  };

  for (const item of accepted) {
    const { blob, relH } = item;

    /*
     * Level the WHOLE mound — cap plus the flanks found by the growth
     * pass — otherwise the flanks stay up as a ridge under the trees.
     */
    for (const p of blob.pixels) {
      clumpMask[p] = 1;

      /*
       * Belt and braces: even if a clump pixel somehow reached a
       * building, the rendered surface is never lowered there.
       */
      flattenDelta[p] = buildingGuard[p]
        ? 0
        : Math.max(
            0,
            placeHeights[p] - groundPlace[p]
          );
    }

    for (const p of item.extra) {
      clumpMask[p] = 1;

      flattenDelta[p] = buildingGuard[p]
        ? 0
        : Math.max(
            0,
            placeHeights[p] - groundPlace[p]
          );
    }

    /*
     * Spacing scales with clump height: tall canopies are wider, so
     * they need more room. Overlap is deliberate — it is what hides
     * the DSM mound inside the foliage.
     */
    const spacingM = Math.max(
      2.0,
      Math.min(7.0, relH * 0.45)
    );

    const spacingPx = spacingM * pxPerM;
    const spacingSq = spacingPx * spacingPx;

    const cellSize = Math.max(1, spacingPx);

    /* Deterministic shuffle: same scene, same forest, every rebuild. */
    const order = Array.from(blob.pixels);

    order.sort(
      (a, b) =>
        hash2D(a % width, (a / width) | 0) -
        hash2D(b % width, (b / width) | 0)
    );

    const budget = Math.max(
      1,
      Math.ceil(item.areaM2 / (spacingM * spacingM * 0.85))
    );

    let placedHere = 0;

    for (const p of order) {
      if (placedHere >= budget) break;
      if (spots.length >= maxTrees) break;

      const x = p % width;
      const y = (p - x) / width;

      /* Too close to a building — drop the tree rather than clip it. */
      if (protectMask[p]) continue;

      const cx = Math.floor(x / cellSize);
      const cy = Math.floor(y / cellSize);

      let ok = true;

      for (let oy = -1; oy <= 1 && ok; oy++) {
        for (let ox = -1; ox <= 1 && ok; ox++) {
          const list = cellIndex.get(
            `${cx + ox}|${cy + oy}`
          );

          if (!list) continue;

          for (const idx of list) {
            const s = spots[idx];

            const dx = s.px - x;
            const dy = s.py - y;

            if (dx * dx + dy * dy < spacingSq) {
              ok = false;
              break;
            }
          }
        }
      }

      if (!ok) continue;

      const world = worldFromPixel(
        x,
        y,
        width,
        height,
        extentX,
        extentZ
      );

      const h1 = hash2D(x, y);
      const h2 = hash2D(y + 19.7, x + 3.1);
      const h3 = hash2D(x * 0.5 + 7.3, y * 0.5 + 11.9);

      /*
       * Plant at the estimated ground under this pixel, NOT on the
       * mound's own surface — otherwise trees stack on top of the
       * mound instead of replacing it.
       */
      const base = groundPlace[p];

      /* Slightly overshoot the mound so canopies cap it off. */
      const treeH = relH * (1.0 + h2 * 0.22);

      const idx = spots.length;

      spots.push({
        px: x,
        py: y,

        x: world.x,
        z: world.z,

        base,

        /*
         * The mound's own surface here. Used when mound levelling is
         * switched off, so the tree stands on top of the mound rather
         * than being buried inside it.
         */
        surf: placeHeights[p],

        treeH,

        variant: Math.min(2, Math.floor(h1 * 3)),

        rot: h1 * Math.PI * 2,

        jitterXZ: 0.9 + h3 * 0.25,
        jitterY: 0.94 + h2 * 0.16,

        tint: 0.86 + h3 * 0.28,
      });

      addToGrid(`${cx}|${cy}`, idx);

      placedHere++;
    }

    if (spots.length >= maxTrees) break;
  }

  return {
    spots,

    clumpMask,
    flattenDelta,
    protectMask,

    buildingMask,
    buildingSeed,

    /* Ground level under each pixel, for aligning facade storeys. */
    groundBase: districtDetect,

    stats: {
      clumps: accepted.length,
      blobs: blobs.length,
      buildings: buildings.blobs.length,
      moundPixelsReclaimed: reclaimed,
      trees: spots.length,

      tallest: +tallest.toFixed(2),
      objectFloor: +objectFloor.toFixed(2),
      bandLow: +bandLow.toFixed(2),
      bandHigh: +bandHigh.toFixed(2),

      greenCut: +greenCut.toFixed(4),
      exgBackground: +exgBackground.toFixed(4),

      greenSeparation: split
        ? +split.separation.toFixed(4)
        : 0,

      greenFractionOfRaised: split
        ? +split.fractionAbove.toFixed(4)
        : 0,

      hasVegetation,

      raisedFraction: +(raisedCount / n).toFixed(3),

      rejected,
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Tree instancing                                                            */
/* -------------------------------------------------------------------------- */

/*
 * Merges a variant's trunk + crown parts into one non-indexed geometry
 * with baked vertex colours, so a whole forest is 3 draw calls instead
 * of thousands of little groups.
 */
function mergeTreeVariant(variant) {
  const group = makeTreeVariant(variant);

  const parts = [];

  let vertexCount = 0;

  group.updateMatrixWorld(true);

  group.traverse((obj) => {
    if (!obj.isMesh) return;

    const geo = obj.geometry
      .clone()
      .applyMatrix4(obj.matrixWorld)
      .toNonIndexed();

    const color = new THREE.Color();

    if (Array.isArray(obj.material)) {
      color.copy(obj.material[0].color);
    } else {
      color.copy(obj.material.color);
    }

    parts.push({ geo, color });

    vertexCount += geo.attributes.position.count;
  });

  const position = new Float32Array(vertexCount * 3);
  const normal = new Float32Array(vertexCount * 3);
  const color = new Float32Array(vertexCount * 3);

  let v = 0;

  for (const part of parts) {
    const pa = part.geo.attributes.position.array;
    const na = part.geo.attributes.normal.array;

    const count = part.geo.attributes.position.count;

    position.set(pa, v * 3);
    normal.set(na, v * 3);

    for (let i = 0; i < count; i++) {
      color[(v + i) * 3] = part.color.r;
      color[(v + i) * 3 + 1] = part.color.g;
      color[(v + i) * 3 + 2] = part.color.b;
    }

    v += count;

    part.geo.dispose();
  }

  /* Free the scratch variant. */
  group.traverse((obj) => {
    if (!obj.isMesh) return;

    obj.geometry.dispose();

    if (Array.isArray(obj.material)) {
      obj.material.forEach((m) => m.dispose());
    } else {
      obj.material.dispose();
    }
  });

  const merged = new THREE.BufferGeometry();

  merged.setAttribute(
    'position',
    new THREE.BufferAttribute(position, 3)
  );

  merged.setAttribute(
    'normal',
    new THREE.BufferAttribute(normal, 3)
  );

  merged.setAttribute(
    'color',
    new THREE.BufferAttribute(color, 3)
  );

  merged.computeBoundingBox();

  const nominalHeight = Math.max(
    0.001,
    merged.boundingBox.max.y
  );

  merged.userData.nominalHeight = nominalHeight;

  return merged;
}

/* -------------------------------------------------------------------------- */
/* Tree layer                                                                 */
/* -------------------------------------------------------------------------- */

export async function createTreeLayer({
  source,
  width,
  height,
  heights,
  detectHeights = null,
  extentX,
  extentZ,
  verticalScale = 1,
  maxTrees = 1200,
  minTreeHeight = null,
  maxTreeHeight = null,
  textureFlipY = true,
  greenThreshold = null,
}) {
  const imageData = orientImageData(
    await decodeRGBPixels(source, width, height),
    textureFlipY
  );

  const group = new THREE.Group();
  group.name = 'VegetationEnrichment';

  const plan = planTreeClumps({
    imageData,

    width,
    height,

    /*
     * Detection uses the RAW DSM: edge sharpening flattens the very
     * canopy lumpiness the classifier depends on.
     */
    detect: detectHeights || heights,

    /* Planting uses the array the visible mesh was built from. */
    place: heights,

    extentX,
    extentZ,

    verticalScale,

    maxTrees,
    minTreeHeight,
    maxTreeHeight,
    greenThreshold,
  });

  const geometries = [
    mergeTreeVariant(0),
    mergeTreeVariant(1),
    mergeTreeVariant(2),
  ];

  const buckets = [[], [], []];

  for (const spot of plan.spots) {
    buckets[spot.variant].push(spot);
  }

  const matrix = new THREE.Matrix4();
  const quat = new THREE.Quaternion();
  const pos = new THREE.Vector3();
  const scl = new THREE.Vector3();
  const tint = new THREE.Color();

  for (let v = 0; v < 3; v++) {
    const list = buckets[v];

    if (!list.length) {
      geometries[v].dispose();
      continue;
    }

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.92,
      metalness: 0,
    });

    const inst = new THREE.InstancedMesh(
      geometries[v],
      material,
      list.length
    );

    inst.castShadow = true;
    inst.receiveShadow = true;

    inst.frustumCulled = false;

    const nominal =
      geometries[v].userData.nominalHeight;

    /*
     * Re-seat every instance for a given vertical exaggeration. Height
     * follows the terrain's own Y stretch so canopy tops keep matching
     * the mound they replace; girth does not stretch with it.
     */
    const rebase = (sy, levelled = true) => {
      for (let i = 0; i < list.length; i++) {
        const s = list[i];

        const base = s.treeH / nominal;

        pos.set(
          s.x,
          (levelled ? s.base : s.surf) * sy - 0.15,
          s.z
        );

        quat.setFromAxisAngle(
          new THREE.Vector3(0, 1, 0),
          s.rot
        );

        scl.set(
          base * s.jitterXZ,
          base * s.jitterY * sy,
          base * s.jitterXZ
        );

        matrix.compose(pos, quat, scl);

        inst.setMatrixAt(i, matrix);
      }

      inst.instanceMatrix.needsUpdate = true;

      inst.computeBoundingSphere();
    };

    rebase(verticalScale, true);

    for (let i = 0; i < list.length; i++) {
      const t = list[i].tint;

      tint.setRGB(t, t, t);

      inst.setColorAt(i, tint);
    }

    if (inst.instanceColor) {
      inst.instanceColor.needsUpdate = true;
    }

    inst.userData = {
      type: 'treeInstances',
      rebase,
    };

    group.add(inst);
  }

  const { texture: maskTexture, deltaMax } =
    buildCanopyTexture(
      plan.clumpMask,
      plan.flattenDelta,
      plan.protectMask,
      width,
      height,
      Math.max(
        1,
        Math.round(
          1.5 * Math.min(width / extentX, height / extentZ)
        )
      )
    );

  const site = buildSiteTexture(
    plan.buildingMask,
    plan.buildingSeed,
    plan.groundBase,
    width,
    height
  );

  return {
    group,

    count: plan.spots.length,

    maskTexture,

    siteTexture: site.texture,
    groundMin: site.groundMin,
    groundSpan: site.groundSpan,

    /* Metres, matching the texture's green channel scale. */
    deltaMax,

    /*
     * Unscaled per-pixel drop, for the collision sampler: what you
     * see and what you can walk into must be the same surface.
     */
    flattenDelta: plan.flattenDelta,

    stats: {
      ...plan.stats,
      deltaMax: +deltaMax.toFixed(2),
    },
  };
}