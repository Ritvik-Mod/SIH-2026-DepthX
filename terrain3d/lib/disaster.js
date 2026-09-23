/* ==========================================================================
 * DISASTER SIMULATION — analysis half
 *
 * This file computes WHAT HAPPENS. lib/disasterFx.js draws it.
 *
 * The rule the rest of this codebase already lives by applies here without
 * exception: the height array is never written to. Everything below produces
 * a separate per-pixel DELTA (metres) plus a per-pixel damage state, which the
 * shader subtracts and the collision sampler subtracts, from the SAME array —
 * so what you see and what you can walk into can never drift apart. Switch the
 * simulation off and every gate returns to 0, which is byte-for-byte the
 * original render.
 *
 * On the models used. None of this is a certified engineering tool and it must
 * not be presented as one. Each number below is a published, citable
 * relationship used at its intended scale, with the source named, so the
 * output is defensible as a *plausible* scenario rather than invented:
 *
 *   ground motion   Joyner & Boore (1981) PGA attenuation
 *   intensity       Wald et al. (1999) PGA -> Modified Mercalli
 *   fragility       HAZUS-MH-style lognormal curves (illustrative medians)
 *   wind field      Holland (1980) parametric cyclone profile
 *   surge           Saffir-Simpson historical surge bands
 *   hydrology       Priority-Flood (Barnes, Lehman & Mulla 2014)
 *
 * What is NOT modelled, and should be said out loud when demoing: no soil
 * classification (so no site amplification and no liquefaction), no structural
 * type beyond a height proxy, no hydrodynamics (the flood is a level pool, not
 * a routed flow), and no rainfall-runoff. Those need inputs a single nadir
 * image does not carry.
 * ========================================================================== */

import * as THREE from 'three';

/* -------------------------------------------------------------------------- */
/* Deterministic noise                                                        */
/* -------------------------------------------------------------------------- */

/*
 * Every random draw in this file is a hash of something stable — a building's
 * seed, a pixel coordinate — never Math.random(). Re-running the same scenario
 * on the same scene has to give the same collapse, or the impact numbers in
 * the report would not match the buildings on screen.
 */
function hash1(n) {
  const s = Math.sin(n * 127.1) * 43758.5453;
  return s - Math.floor(s);
}

function hash2(x, y) {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}

/** Standard normal CDF — Abramowitz & Stegun 26.2.17, |error| < 7.5e-8. */
export function normalCdf(z) {
  const sign = z < 0 ? -1 : 1;
  const x = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * x);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-x * x);
  return 0.5 * (1 + sign * y);
}

/** P(damage >= state) for a lognormal fragility curve with median `med` (g). */
function fragility(pga, med, beta) {
  if (!(pga > 0) || !(med > 0)) return 0;
  return normalCdf(Math.log(pga / med) / beta);
}

/* ==========================================================================
 * GROUND MOTION
 * ========================================================================== */

/**
 * Peak ground acceleration in g, from magnitude and epicentral distance.
 *
 * Joyner & Boore (1981), the random-horizontal-component relation:
 *
 *   log10 PGA[g] = -1.02 + 0.249 M - log10 r - 0.00255 r
 *   r = sqrt(d^2 + 7.3^2)
 *
 * The 7.3 km is a saturation term: without it the equation sends PGA to
 * infinity directly above the rupture, which is not what happens. Calibrated
 * for 5.0 <= M <= 7.7 on rock/soil in western North America, so it is a
 * generic estimate, not a site-specific one — there is no soil class here to
 * amplify with.
 */
export function pgaFromMagnitude(magnitude, distanceKm) {
  const m = Math.min(Math.max(magnitude, 4), 8.5);
  const r = Math.sqrt(distanceKm * distanceKm + 7.3 * 7.3);
  const log10 = -1.02 + 0.249 * m - Math.log10(r) - 0.00255 * r;
  return Math.pow(10, log10);
}

/**
 * Modified Mercalli Intensity from PGA — Wald et al. (1999), the regression
 * behind ShakeMap's instrumental intensity:
 *
 *   MMI = 3.66 log10(PGA[cm/s^2]) - 1.66      (valid for MMI V and above)
 *
 * Below MMI V the relation flattens and PGA stops predicting what people feel,
 * so it is clamped rather than extrapolated.
 */
export function mmiFromPga(pgaG) {
  const cms2 = pgaG * 980.665;
  if (cms2 <= 0) return 1;
  const mmi = 3.66 * Math.log10(cms2) - 1.66;
  return Math.min(12, Math.max(1, mmi));
}

export const ROMAN = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII'];

export function romanMmi(mmi) {
  return ROMAN[Math.min(12, Math.max(1, Math.round(mmi)))] || 'I';
}

/* ==========================================================================
 * BUILDING INVENTORY
 * ========================================================================== */

export const DAMAGE_STATES = ['none', 'slight', 'moderate', 'extensive', 'complete'];

/*
 * Fragility medians in g, by a height proxy for structural type. HAZUS-MH
 * publishes these per model building type and seismic design level; there is
 * no way to read a construction type off a nadir photo, so height is the only
 * proxy available and these are the moderate-code curves for the type that
 * dominates each band.
 *
 * ILLUSTRATIVE. Right order of magnitude, right ordering between bands, not a
 * substitute for an engineering assessment of any specific building.
 */
const FRAGILITY = [
  // 1-3 storeys: mostly unreinforced masonry / non-engineered concrete
  { maxStoreys: 3, beta: 0.64, med: [0.13, 0.18, 0.28, 0.40] },
  // 4-7 storeys: mid-rise reinforced concrete frame
  { maxStoreys: 7, beta: 0.66, med: [0.17, 0.24, 0.43, 0.66] },
  // 8+: engineered high-rise, longer period, stiffer design requirements
  { maxStoreys: Infinity, beta: 0.68, med: [0.20, 0.29, 0.54, 0.88] },
];

function fragilityClass(storeys) {
  for (const f of FRAGILITY) if (storeys <= f.maxStoreys) return f;
  return FRAGILITY[FRAGILITY.length - 1];
}

/*
 * How much of a building is left standing in each damage state, as a fraction
 * of its original height. A "complete" pancake collapse does not go to zero:
 * the debris pile is roughly a sixth of the original height once the floor
 * slabs stack, which is why search-and-rescue is measured in storeys of rubble
 * and not in a flat slab.
 */
const RESIDUAL = [1.0, 1.0, 0.94, 0.68, 0.17];

/**
 * Per-building geometry, measured once from the scene analysis.
 *
 * `groundBase` is the WIDE morphological opening (districtDetect) — the same
 * surface the facade shader measures storeys from, so a building's base here
 * is the pavement it stands on rather than sea level.
 */
export function buildInventory({
  heights,
  groundBase,
  buildingSeed,
  buildingBlobs,
  width,
  pxPerM,
}) {
  const areaPerPx = 1 / (pxPerM * pxPerM);
  const list = [];

  for (const blob of buildingBlobs) {
    const px = blob.pixels;
    const n = px.length;
    if (!n) continue;

    // Mean base, and a high percentile for the roof: the max would latch onto
    // a single noisy pixel or an aerial mast and call a bungalow a tower.
    let baseSum = 0;
    const tops = new Float32Array(n);
    let cx = 0;
    let cy = 0;
    let minX = Infinity; let maxX = -Infinity;
    let minY = Infinity; let maxY = -Infinity;

    for (let i = 0; i < n; i++) {
      const p = px[i];
      baseSum += groundBase[p];
      tops[i] = heights[p];
      const x = p % width;
      const y = (p - x) / width;
      cx += x;
      cy += y;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }

    tops.sort();
    const base = baseSum / n;
    const top = tops[Math.min(n - 1, Math.floor(n * 0.9))];
    const h = Math.max(0, top - base);

    // 3.3 m floor-to-floor, matching the facade shader's storey grid.
    const storeys = Math.max(1, Math.round(h / 3.3));

    list.push({
      id: blob.id,
      pixels: px,
      base,
      height: h,
      storeys,
      areaM2: n * areaPerPx,
      seed: buildingSeed[px[0]] / 255,
      cx: cx / n,
      cy: cy / n,
      minX, maxX, minY, maxY,
    });
  }

  return list;
}

/* ==========================================================================
 * EARTHQUAKE
 * ========================================================================== */

/**
 * Assigns a damage state to every building and paints the resulting change in
 * surface height into a delta array.
 *
 * The damage state is drawn by inverse transform sampling: each building's
 * stable seed is treated as a uniform variate and compared against the
 * fragility curves, so the same scene under the same shaking always fails the
 * same way, and raising the magnitude can only ever make a building worse —
 * never randomly better, which is what a fresh random draw per frame would do.
 *
 * The delta is signed. Collapsed footprints go DOWN to a rubble pile; a short
 * apron around them goes UP, because the material has to end up somewhere and
 * a collapse that leaves the street pristine looks wrong.
 */
export function computeQuakeDamage({
  inventory,
  heights,
  groundBase,
  width,
  height,
  pxPerM,
  pga,
}) {
  const n = width * height;
  const drop = new Float32Array(n);
  const state = new Float32Array(n);

  const counts = [0, 0, 0, 0, 0];
  let collapsedArea = 0;
  let debrisVolume = 0;
  const collapsedEdge = [];

  for (const b of inventory) {
    const f = fragilityClass(b.storeys);
    const u = b.seed;

    // Highest state whose exceedance probability still covers this draw.
    let ds = 0;
    for (let k = 3; k >= 0; k--) {
      if (fragility(pga, f.med[k], f.beta) >= u) {
        ds = k + 1;
        break;
      }
    }

    b.damage = ds;
    counts[ds]++;

    if (ds === 0) continue;

    const residual = RESIDUAL[ds];
    const target = b.base + b.height * residual;

    if (ds >= 3) collapsedArea += b.areaM2;

    const sNorm = ds / 4;
    let lost = 0;

    for (let i = 0; i < b.pixels.length; i++) {
      const p = b.pixels[i];

      // Per-pixel, so a pitched or stepped roof collapses to a pile that still
      // follows its own footprint instead of becoming a flat slab.
      const x = p % width;
      const y = (p - x) / width;
      const rubble = ds >= 3 ? (hash2(x * 0.21, y * 0.17) - 0.5) * 1.1 * sNorm : 0;

      const d = Math.max(0, heights[p] - (target + rubble));
      drop[p] = d;
      state[p] = sNorm;
      lost += d;
    }

    if (ds >= 3) {
      debrisVolume += (lost / (pxPerM * pxPerM)) * 0.35; // ~35% of the void becomes loose spoil
      collapsedEdge.push(b);
    }
  }

  // ---- debris apron -------------------------------------------------------
  // A ring of raised ground around each collapsed footprint. Written as a
  // NEGATIVE drop, because the shader subtracts this array: -0.4 m of "drop"
  // is 0.4 m of rubble in the street.
  const apronPx = Math.max(1, Math.round(3.5 * pxPerM));
  if (apronPx > 0 && collapsedEdge.length) {
    const seen = new Uint8Array(n);
    let frontier = [];

    for (const b of collapsedEdge) {
      for (let i = 0; i < b.pixels.length; i++) {
        seen[b.pixels[i]] = 1;
        frontier.push(b.pixels[i]);
      }
    }

    for (let ring = 1; ring <= apronPx; ring++) {
      const next = [];
      const amount = 0.55 * (1 - (ring - 1) / apronPx);

      for (const p of frontier) {
        const x = p % width;
        const y = (p - x) / width;

        for (let k = 0; k < 4; k++) {
          const nx = x + (k === 0 ? -1 : k === 1 ? 1 : 0);
          const ny = y + (k === 2 ? -1 : k === 3 ? 1 : 0);
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

          const q = ny * width + nx;
          if (seen[q]) continue;
          seen[q] = 1;

          // Only pile debris on ground — never on a neighbouring roof.
          if (heights[q] - groundBase[q] > 1.5) continue;

          drop[q] = -amount * (0.6 + 0.4 * hash2(nx * 0.31, ny * 0.29));
          state[q] = Math.max(state[q], 0.45);
          next.push(q);
        }
      }
      frontier = next;
      if (!frontier.length) break;
    }
  }

  const mmi = mmiFromPga(pga);

  return {
    drop,
    state,
    stats: {
      kind: 'earthquake',
      pga,
      mmi,
      buildings: inventory.length,
      counts,
      collapsed: counts[4],
      severe: counts[3] + counts[4],
      collapsedAreaM2: collapsedArea,
      debrisVolumeM3: debrisVolume,
    },
  };
}

/* ==========================================================================
 * TROPICAL CYCLONE
 * ========================================================================== */

/*
 * Saffir-Simpson: 1-minute sustained wind in m/s (lower bound of each
 * category) and the historical storm-surge band in metres. Surge is NOT a
 * function of wind alone — bathymetry, forward speed and the angle of approach
 * dominate — so these are the classic reference bands and nothing more.
 */
export const CYCLONE_CATEGORIES = [
  { cat: 1, windMs: 33, surge: [1.2, 1.5], label: 'Category 1' },
  { cat: 2, windMs: 43, surge: [1.8, 2.4], label: 'Category 2' },
  { cat: 3, windMs: 50, surge: [2.7, 3.7], label: 'Category 3' },
  { cat: 4, windMs: 58, surge: [4.0, 5.5], label: 'Category 4' },
  { cat: 5, windMs: 70, surge: [5.5, 7.5], label: 'Category 5' },
];

/**
 * Holland (1980) parametric radial wind profile:
 *
 *   V(r) = Vmax * sqrt( (Rmax/r)^B * exp(1 - (Rmax/r)^B) )
 *
 * The whole point of including it is that "Category 4" on its own says nothing
 * about a site — a cat 4 eyewall 90 km away is a breezy afternoon. B is the
 * Holland shape parameter; 1.3 is a mid-range value for a mature system.
 */
export function hollandWind(vmax, rKm, rmaxKm = 35, B = 1.3) {
  const r = Math.max(rKm, 0.5);
  const ratio = Math.pow(rmaxKm / r, B);
  if (r <= rmaxKm) {
    // Inside the radius of maximum wind the profile falls away towards the
    // calm eye, roughly linearly in the inner core.
    return vmax * (r / rmaxKm);
  }
  return vmax * Math.sqrt(ratio * Math.exp(1 - ratio));
}

/** 3-second gust from 1-minute sustained wind — the usual over-land factor. */
export const GUST_FACTOR = 1.3;

/*
 * Wind damage to a roof, as a fraction of cover lost, from the 3-second gust.
 * Thresholds follow the descriptive Saffir-Simpson damage scale rather than a
 * component-level fragility, which would need a roof type nobody can see from
 * orbit.
 */
function roofLoss(gust, u, storeys) {
  // Taller buildings sit in faster flow but are engineered for it; low-rise
  // sheet and tile roofs go first. Net effect is close to flat, with the
  // low-rise band slightly more exposed.
  const bias = storeys <= 2 ? 1.15 : storeys <= 6 ? 1.0 : 0.85;
  const g = gust * bias;
  if (g < 38) return 0;
  const t = Math.min(1, (g - 38) / 34); // saturates around 72 m/s
  return t * t * (0.35 + 0.65 * u);
}

/**
 * Cyclone damage: roof loss on buildings, plus the tree response the FX layer
 * needs to lean, strip and fell the instanced forest.
 */
export function computeWindDamage({
  inventory,
  heights,
  width,
  height,
  sustainedMs,
  windDirDeg,
}) {
  const n = width * height;
  const drop = new Float32Array(n);
  const state = new Float32Array(n);

  const gust = sustainedMs * GUST_FACTOR;
  let roofed = 0;
  let stripped = 0;

  for (const b of inventory) {
    // A different draw from the seismic one, or the same buildings would
    // always be the unlucky ones in every scenario.
    const u = hash1(b.seed * 311.7 + 17.3);
    const loss = roofLoss(gust, u, b.storeys);

    b.windLoss = loss;
    if (loss <= 0.01) continue;

    roofed++;
    if (loss > 0.5) stripped++;

    // Roof cover and the top of the parapet: a metre or so, not a storey.
    // Stripping a roof does not shorten the building.
    const d = Math.min(1.8, 0.35 + loss * 1.4);

    for (let i = 0; i < b.pixels.length; i++) {
      const p = b.pixels[i];
      const x = p % width;
      const y = (p - x) / width;
      // Peeled, not levelled — the loss is patchy across the roof.
      const patch = hash2(x * 0.13, y * 0.19);
      if (patch > 1 - loss) {
        drop[p] = Math.min(d, Math.max(0, heights[p] - b.base));
        state[p] = Math.min(1, 0.35 + loss * 0.65);
      } else {
        state[p] = Math.min(1, loss * 0.4);
      }
    }
  }

  return {
    drop,
    state,
    gust,
    stats: {
      kind: 'cyclone',
      sustainedMs,
      gust,
      buildings: inventory.length,
      roofDamaged: roofed,
      roofStripped: stripped,
      windDirDeg,
    },
  };
}

/**
 * Per-tree wind response, by the same thresholds field surveys report after
 * landfall: defoliation first, then branch loss, then snapping and uprooting.
 *
 * Returns a closure rather than an array because the FX layer already owns the
 * instance list and re-seats it every frame for the gust animation.
 */
export function treeWindResponse(gust) {
  const defoliate = Math.min(1, Math.max(0, (gust - 24) / 26));
  const downedP = Math.min(0.85, Math.max(0, (gust - 42) / 40));
  // Static lean scales with dynamic pressure, which goes as V^2.
  const lean = Math.min(0.85, (gust * gust) / 4200);
  return { defoliate, downedP, lean, gust };
}

/* ==========================================================================
 * FLOOD / STORM SURGE
 * ========================================================================== */

/**
 * Priority-Flood (Barnes, Lehman & Mulla 2014).
 *
 * For every cell it computes the SPILL LEVEL: the lowest water surface at
 * which that cell becomes hydraulically connected to the edge of the tile.
 * Formally the minimum, over all paths from the boundary, of the maximum
 * elevation along the path.
 *
 * Why this rather than "flood everything below level L":
 *
 *   - a courtyard enclosed by six-storey blocks does not fill until the water
 *     tops the blocks, and a naive threshold fills it instantly;
 *   - a roof hollow, a light well and the inside of a stadium are all below
 *     level L and none of them flood;
 *   - ground behind an embankment stays dry until the embankment is overtopped,
 *     which is the single most important behaviour in real flood mapping.
 *
 * And it is computed ONCE. Afterwards every water level is a single comparison
 * per cell, so the stage slider is free — no re-flooding per frame.
 *
 * Run on a decimated grid: a waterline does not need 0.33 m cells, and the
 * heap is O(n log n). Cell means, not minima, so a wall several pixels thick
 * survives decimation as a barrier.
 */
export function computeSpillMap(heights, w, h, { maxDim = 512 } = {}) {
  const step = Math.max(1, Math.ceil(Math.max(w, h) / maxDim));
  const sw = Math.max(2, Math.floor(w / step));
  const sh = Math.max(2, Math.floor(h / step));
  const n = sw * sh;

  const elev = new Float32Array(n);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      let sum = 0;
      let c = 0;
      for (let dy = 0; dy < step; dy++) {
        const sy = y * step + dy;
        if (sy >= h) break;
        const row = sy * w;
        for (let dx = 0; dx < step; dx++) {
          const sx = x * step + dx;
          if (sx >= w) break;
          sum += heights[row + sx];
          c++;
        }
      }
      elev[y * sw + x] = c ? sum / c : 0;
    }
  }

  const spill = new Float32Array(n);
  const done = new Uint8Array(n);

  // Binary min-heap over (key = spill level, value = cell index).
  const keys = new Float64Array(n + 1);
  const vals = new Int32Array(n + 1);
  let size = 0;

  const push = (k, v) => {
    let i = ++size;
    keys[i] = k;
    vals[i] = v;
    while (i > 1) {
      const p = i >> 1;
      if (keys[p] <= keys[i]) break;
      const tk = keys[p]; const tv = vals[p];
      keys[p] = keys[i]; vals[p] = vals[i];
      keys[i] = tk; vals[i] = tv;
      i = p;
    }
  };

  const pop = () => {
    const top = vals[1];
    keys[1] = keys[size];
    vals[1] = vals[size];
    size--;
    let i = 1;
    for (;;) {
      const l = i << 1;
      const r = l + 1;
      let m = i;
      if (l <= size && keys[l] < keys[m]) m = l;
      if (r <= size && keys[r] < keys[m]) m = r;
      if (m === i) break;
      const tk = keys[m]; const tv = vals[m];
      keys[m] = keys[i]; vals[m] = vals[i];
      keys[i] = tk; vals[i] = tv;
      i = m;
    }
    return top;
  };

  // Seed with the whole boundary: water arrives from outside the tile.
  for (let x = 0; x < sw; x++) {
    for (const y of [0, sh - 1]) {
      const i = y * sw + x;
      if (done[i]) continue;
      done[i] = 1;
      spill[i] = elev[i];
      push(elev[i], i);
    }
  }
  for (let y = 0; y < sh; y++) {
    for (const x of [0, sw - 1]) {
      const i = y * sw + x;
      if (done[i]) continue;
      done[i] = 1;
      spill[i] = elev[i];
      push(elev[i], i);
    }
  }

  while (size > 0) {
    const i = pop();
    const x = i % sw;
    const y = (i - x) / sw;
    const level = spill[i];

    for (let k = 0; k < 4; k++) {
      const nx = x + (k === 0 ? -1 : k === 1 ? 1 : 0);
      const ny = y + (k === 2 ? -1 : k === 3 ? 1 : 0);
      if (nx < 0 || nx >= sw || ny < 0 || ny >= sh) continue;

      const q = ny * sw + nx;
      if (done[q]) continue;
      done[q] = 1;

      // The water has to clear whichever is higher: the barrier it came over,
      // or this cell itself.
      spill[q] = Math.max(level, elev[q]);
      push(spill[q], q);
    }
  }

  return { spill, elev, sw, sh, step };
}

/**
 * Water-field texture. R = spill level, G = surface elevation, both in metres.
 *
 * Half float rather than 8-bit: quantising a 40 m range into 256 levels puts
 * 0.16 m steps in the water surface, and on a shallow slope that terraces the
 * waterline into visible contour rings.
 *
 * flipY stays false and raster row 0 lands at v = 0, matching the canopy and
 * site textures — all three are sampled from world XZ, not from the mesh UVs.
 */
export function makeWaterTexture(spill, elev, sw, sh) {
  const buf = new Uint16Array(sw * sh * 2);
  for (let i = 0; i < sw * sh; i++) {
    buf[i * 2] = THREE.DataUtils.toHalfFloat(spill[i]);
    buf[i * 2 + 1] = THREE.DataUtils.toHalfFloat(elev[i]);
  }
  const tex = new THREE.DataTexture(buf, sw, sh, THREE.RGFormat, THREE.HalfFloatType);
  tex.flipY = false;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Damage texture. R = metres of surface change (signed: positive lowers the
 * surface, negative raises it as debris), G = damage state in [0,1].
 *
 * NEAREST on purpose. Interpolating the drop across a footprint boundary would
 * drag half a collapsed building out into the street as a smooth ramp, and
 * interpolating the state would paint the pavement beside a ruin as damaged.
 */
export function makeDamageTexture(drop, state, w, h) {
  const n = w * h;
  const buf = new Uint16Array(n * 2);
  for (let i = 0; i < n; i++) {
    buf[i * 2] = THREE.DataUtils.toHalfFloat(drop[i]);
    buf[i * 2 + 1] = THREE.DataUtils.toHalfFloat(state[i]);
  }
  const tex = new THREE.DataTexture(buf, w, h, THREE.RGFormat, THREE.HalfFloatType);
  tex.flipY = false;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

/**
 * What a given water level actually does to this scene.
 *
 * Area comes off the decimated hydrology grid, which is what defines "wet".
 * Building exposure is counted on the full-resolution footprints, because a
 * building is inundated when water reaches the ground it stands on — its roof
 * being dry is not relevant to anyone inside it.
 */
export function floodImpact({
  spill, elev, sw, sh, step, inventory, pixelSpacing, level, datum = 0,
}) {
  const cellArea = (step * pixelSpacing) ** 2;
  const cellM = step * pixelSpacing;

  /*
   * How far out to look for the street.
   *
   * A building's own footprint is a barrier in the spill field -- the only
   * way water reaches the middle of a roof is by topping the roof -- so
   * sampling the spill AT a building always reports its own height and every
   * building comes back dry. What decides whether a building floods is
   * whether the ground it stands on is reachable, so the test is the lowest
   * spill level anywhere within a short walk of the footprint.
   *
   * 12 m is about the width of a street: far enough to escape the building
   * itself, near enough that the answer still belongs to this building.
   * The failure mode is optimistic rather than pessimistic -- a low area
   * within 12 m but behind a wall would count -- which is the right way round
   * for a hazard readout.
   */
  const pad = Math.max(1, Math.round(12 / Math.max(cellM, 0.01)));

  let wetCells = 0;
  let depthSum = 0;
  let deepest = 0;

  for (let i = 0; i < sw * sh; i++) {
    if (spill[i] > level) continue;
    const d = level - elev[i];
    if (d <= 0) continue;
    wetCells++;
    depthSum += d;
    if (d > deepest) deepest = d;
  }

  let inundated = 0;
  let submerged = 0;

  for (const b of inventory) {
    const head = level - b.base;
    if (head <= 0) {
      b.flood = 0;
      continue;
    }

    const x0 = Math.max(0, Math.floor(b.minX / step) - pad);
    const x1 = Math.min(sw - 1, Math.floor(b.maxX / step) + pad);
    const y0 = Math.max(0, Math.floor(b.minY / step) - pad);
    const y1 = Math.min(sh - 1, Math.floor(b.maxY / step) + pad);

    let reach = Infinity;
    for (let y = y0; y <= y1 && reach > level; y++) {
      const row = y * sw;
      for (let x = x0; x <= x1; x++) {
        const v = spill[row + x];
        if (v < reach) {
          reach = v;
          if (reach <= level) break;
        }
      }
    }

    if (reach > level) {
      b.flood = 0;
      continue;
    }

    b.flood = head;
    inundated++;
    if (head >= b.height * 0.9) submerged++;
  }

  const totalCells = sw * sh;

  return {
    kind: 'flood',
    /*
     * BOTH numbers, deliberately. `level` is the absolute water surface in
     * raster units, which is what the shader and the spill field compare
     * against. `stage` is height above the scene datum, which is what the
     * slider sets and what a person means by "three metres of flooding".
     * On a DSM tile whose ground sits at 210 m those differ by 210, and
     * showing the wrong one under the label "stage" made the readout
     * disagree with the control that produced it.
     */
    level,
    stage: level - datum,
    wetFraction: wetCells / totalCells,
    wetAreaM2: wetCells * cellArea,
    meanDepth: wetCells ? depthSum / wetCells : 0,
    maxDepth: deepest,
    buildings: inventory.length,
    inundated,
    submerged,
  };
}

/* ==========================================================================
 * SCENARIO PRESETS
 * ========================================================================== */

/**
 * Default parameters per scenario. Chosen so that pressing "Run" once, with
 * nothing touched, produces a visible and defensible event on a typical tile
 * rather than an anticlimax the user has to go hunting for in the sliders.
 */
export const SCENARIOS = {
  earthquake: {
    label: 'Earthquake',
    icon: '◈',
    blurb: 'Ground motion, fragility-based collapse and debris',
    defaults: { magnitude: 7.0, distanceKm: 10, duration: 14 },
  },
  flood: {
    label: 'Flood',
    icon: '≈',
    blurb: 'Connected-water inundation at a chosen stage',
    defaults: { stage: 3.0, duration: 12 },
  },
  cyclone: {
    label: 'Cyclone',
    icon: '◎',
    blurb: 'Holland wind field, roof loss, tree damage and surge',
    defaults: { category: 3, eyeKm: 30, windDirDeg: 135, surge: true, duration: 16 },
  },
};
