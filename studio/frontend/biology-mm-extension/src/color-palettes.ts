/**
 * Precomputed 256-entry physics color palettes and fast RGBA buffer fillers for Biology-MM.
 */

import { Color } from '@molstar/core/util/color';

export const PALETTE_NAMES = [
  'plasma',
  'thermal-div',
  'electrostatic-bwr',
  'bond-viridis',
  'stress-inferno',
  'ternary-triad',
] as const;

export type PaletteName = (typeof PALETTE_NAMES)[number];

export const DEFAULT_FALLBACK_COLOR = Color(0xcccccc);
const FALLBACK_R = 204;
const FALLBACK_G = 204;
const FALLBACK_B = 204;

const PALETTE_STOPS_HEX: Record<PaletteName, readonly string[]> = {
  plasma: [
    '#0d0887',
    '#46039f',
    '#7201a8',
    '#9c179e',
    '#bd3786',
    '#d8576b',
    '#ed7953',
    '#fb9f3a',
    '#f0f921',
  ],
  'thermal-div': ['#2563eb', '#60a5fa', '#e2e8f0', '#fb923c', '#dc2626'],
  'electrostatic-bwr': ['#1d4ed8', '#60a5fa', '#f8fafc', '#f87171', '#b91c1c'],
  'bond-viridis': [
    '#440154',
    '#482878',
    '#3e4989',
    '#31688e',
    '#26828e',
    '#1f9e89',
    '#35b779',
    '#6ece58',
    '#b5de2b',
    '#fde725',
  ],
  'stress-inferno': [
    '#000004',
    '#1b0c41',
    '#4a0c6b',
    '#781c6d',
    '#a52c60',
    '#cf4446',
    '#ed6925',
    '#fb9b06',
    '#fcffa4',
  ],
  // Strictly calibrated: -1 = crimson (#dc2626), 0 = slate-silver (#94a3b8), +1 = cyan (#06b6d4)
  'ternary-triad': ['#dc2626', '#94a3b8', '#06b6d4'],
};

interface PaletteLut {
  readonly rgba: Uint8Array;
  readonly colors: readonly Color[];
}

function parseHexRgb(hex: string): [number, number, number] {
  const clean = hex.startsWith('#') ? hex.slice(1) : hex;
  const n = parseInt(clean, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function buildLut(stopsHex: readonly string[], exactMidpointIndex?: number): PaletteLut {
  const stops = stopsHex.map(parseHexRgb);
  const rgba = new Uint8Array(256 * 4);
  const colors: Color[] = new Array<Color>(256);

  if (exactMidpointIndex !== undefined && stops.length === 3) {
    const [r0, g0, b0] = stops[0];
    const [r1, g1, b1] = stops[1];
    const [r2, g2, b2] = stops[2];
    const mid = exactMidpointIndex;
    const upperSpan = 255 - mid;

    for (let i = 0; i < 256; i++) {
      let r: number;
      let g: number;
      let b: number;
      if (i <= mid) {
        const u = i / mid;
        r = Math.round(r0 + (r1 - r0) * u);
        g = Math.round(g0 + (g1 - g0) * u);
        b = Math.round(b0 + (b1 - b0) * u);
      } else {
        const u = (i - mid) / upperSpan;
        r = Math.round(r1 + (r2 - r1) * u);
        g = Math.round(g1 + (g2 - g1) * u);
        b = Math.round(b1 + (b2 - b1) * u);
      }
      const off = i * 4;
      rgba[off] = r;
      rgba[off + 1] = g;
      rgba[off + 2] = b;
      rgba[off + 3] = 255;
      colors[i] = Color.fromRgb(r, g, b);
    }
    return { rgba, colors };
  }

  const lastStop = stops.length - 1;
  for (let i = 0; i < 256; i++) {
    const pos = (i / 255) * lastStop;
    const idx = Math.min(Math.floor(pos), lastStop - 1);
    const frac = pos - idx;
    const [r0, g0, b0] = stops[idx];
    const [r1, g1, b1] = stops[idx + 1];
    const r = Math.round(r0 + (r1 - r0) * frac);
    const g = Math.round(g0 + (g1 - g0) * frac);
    const b = Math.round(b0 + (b1 - b0) * frac);

    const off = i * 4;
    rgba[off] = r;
    rgba[off + 1] = g;
    rgba[off + 2] = b;
    rgba[off + 3] = 255;
    colors[i] = Color.fromRgb(r, g, b);
  }

  return { rgba, colors };
}

const PALETTE_LUTS: Record<PaletteName, PaletteLut> = {
  plasma: buildLut(PALETTE_STOPS_HEX.plasma),
  'thermal-div': buildLut(PALETTE_STOPS_HEX['thermal-div']),
  'electrostatic-bwr': buildLut(PALETTE_STOPS_HEX['electrostatic-bwr']),
  'bond-viridis': buildLut(PALETTE_STOPS_HEX['bond-viridis']),
  'stress-inferno': buildLut(PALETTE_STOPS_HEX['stress-inferno']),
  // Index 128 corresponds to Math.round(0.5 * 255), so t = 0.5 hits exact #94a3b8
  'ternary-triad': buildLut(PALETTE_STOPS_HEX['ternary-triad'], 128),
};

const PALETTE_STOP_COLORS: Record<PaletteName, readonly Color[]> = {
  plasma: PALETTE_STOPS_HEX.plasma.map(Color.fromHexStyle),
  'thermal-div': PALETTE_STOPS_HEX['thermal-div'].map(Color.fromHexStyle),
  'electrostatic-bwr': PALETTE_STOPS_HEX['electrostatic-bwr'].map(Color.fromHexStyle),
  'bond-viridis': PALETTE_STOPS_HEX['bond-viridis'].map(Color.fromHexStyle),
  'stress-inferno': PALETTE_STOPS_HEX['stress-inferno'].map(Color.fromHexStyle),
  'ternary-triad': PALETTE_STOPS_HEX['ternary-triad'].map(Color.fromHexStyle),
};

/**
 * Sample a Mol* Color from the 256-entry LUT for `t` in `[0, 1]`.
 * Clamps out-of-range values and returns `DEFAULT_FALLBACK_COLOR` for `NaN`.
 */
export function samplePaletteColor(palette: PaletteName, t: number): Color {
  if (Number.isNaN(t)) return DEFAULT_FALLBACK_COLOR;
  const lut = PALETTE_LUTS[palette].colors;
  if (t <= 0) return lut[0];
  if (t >= 1) return lut[255];
  const idx = (t * 255 + 0.5) | 0;
  return lut[idx];
}

/**
 * Fill an RGBA byte buffer (`4 * scalars.length` bytes) from scalar values using a 256-entry LUT.
 * Handles `min === max` (maps finite scalars to midpoint), `NaN` values (opaque fallback gray),
 * and an optional `activeMask` (inactive slots are written as transparent `[0, 0, 0, 0]`).
 */
export function fillRgbaFromScalars(
  scalars: ArrayLike<number>,
  min: number,
  max: number,
  palette: PaletteName,
  outRgba: Uint8Array,
  activeMask?: ArrayLike<number | boolean>,
): void {
  const count = Math.min(scalars.length, outRgba.length >>> 2);
  const lut = PALETTE_LUTS[palette].rgba;
  const hasValidRange = Number.isFinite(min) && Number.isFinite(max) && max > min;
  const invSpan = hasValidRange ? 1 / (max - min) : 0;

  for (let i = 0; i < count; i++) {
    const outOff = i * 4;
    if (activeMask && !activeMask[i]) {
      outRgba[outOff] = 0;
      outRgba[outOff + 1] = 0;
      outRgba[outOff + 2] = 0;
      outRgba[outOff + 3] = 0;
      continue;
    }

    const v = scalars[i];
    if (Number.isNaN(v)) {
      outRgba[outOff] = FALLBACK_R;
      outRgba[outOff + 1] = FALLBACK_G;
      outRgba[outOff + 2] = FALLBACK_B;
      outRgba[outOff + 3] = 255;
      continue;
    }

    let idx: number;
    if (!hasValidRange) {
      if (v === -Infinity) idx = 0;
      else if (v === Infinity) idx = 255;
      else idx = 128;
    } else if (v <= min) {
      idx = 0;
    } else if (v >= max) {
      idx = 255;
    } else {
      idx = ((v - min) * invSpan * 255 + 0.5) | 0;
      if (idx > 255) idx = 255;
    }

    const lutOff = idx * 4;
    outRgba[outOff] = lut[lutOff];
    outRgba[outOff + 1] = lut[lutOff + 1];
    outRgba[outOff + 2] = lut[lutOff + 2];
    outRgba[outOff + 3] = 255;
  }
}

/** Return CSS hex color stops for UI legend rendering. */
export function getPaletteStopsHex(palette: PaletteName): readonly string[] {
  return PALETTE_STOPS_HEX[palette];
}

/** Return Mol* Color stops for ScaleLegend rendering. */
export function getPaletteStopsColor(palette: PaletteName): readonly Color[] {
  return PALETTE_STOP_COLORS[palette];
}

/** Return the underlying 256x4 RGBA lookup table for a palette. */
export function getPaletteLutRgba(palette: PaletteName): Uint8Array {
  return PALETTE_LUTS[palette].rgba;
}
