/**
 * Physics overlays for Biology-MM:
 *  (a) Triclinic/orthorhombic PBC unit-cell wireframe from `boxH`
 *  (b) PBC-safe covalent bonds from `TopologyInitMessage` / `OverlayVectorsMessage`, scaled & colored by `bondOrders`
 *  (c) Top-K ROI force vectors (`vectorStarts`, `vectorDirs`, `vectorMags`) as 3D shaft+cone arrows
 */

import type { PluginContext } from '@molstar/plugin/context';
import { PluginStateObject } from '@molstar/plugin/state/objects';
import { ShapeRepresentation3D } from '@molstar/plugin/state/transforms/shape/representation';
import { setSubtreeVisibility } from '@molstar/plugin/behavior/static/state';
import { StateTransformer, type StateObjectSelector } from '@molstar/core/state';
import { ParamDefinition as PD } from '@molstar/core/util/param-definition';
import { Color } from '@molstar/core/util/color';
import { Vec3 } from '@molstar/core/math/linear-algebra';
import type { Shape } from '@molstar/model/model/shape';
import * as ShapeUtils from '@molstar/graphics/geo/shape/shape';
import type { ShapeProvider } from '@molstar/graphics/geo/shape/provider';
import { Lines } from '@molstar/graphics/geo/geometry/lines/lines';
import { LinesBuilder } from '@molstar/graphics/geo/geometry/lines/lines-builder';
import { Cylinders } from '@molstar/graphics/geo/geometry/cylinders/cylinders';
import { CylindersBuilder } from '@molstar/graphics/geo/geometry/cylinders/cylinders-builder';
import { Mesh } from '@molstar/graphics/geo/geometry/mesh/mesh';
import { MeshBuilder } from '@molstar/graphics/geo/geometry/mesh/mesh-builder';
import { addCylinder } from '@molstar/graphics/geo/geometry/mesh/builder/cylinder';
import type { OverlayVectorsMessage, TopologyInitMessage } from './protocol.js';

export interface OverlayVisibility {
  showBox: boolean;
  showBonds: boolean;
  showForces: boolean;
}

export interface ForceVectorBuildOptions {
  /** Global length multiplier in Å per force unit (default 1.0). */
  scale?: number;
  /** Maximum number of vectors to render (default all in message). */
  maxVectors?: number;
  /** Shaft radius in Å (default 0.08). */
  radius?: number;
  /** Previous shape for geometry buffer reuse. */
  prevShape?: Shape<Mesh>;
}

const Transform = StateTransformer.builderFactory('biology-mm');

export const BiologyMmShapeProvider = Transform({
  name: 'biology-mm-shape-provider',
  display: { name: 'Biology-MM Overlay' },
  from: PluginStateObject.Root,
  to: PluginStateObject.Shape.Provider,
  params: {
    label: PD.Text('Overlay'),
    provider: PD.Value<ShapeProvider<any, any, any>>(undefined as any, { isHidden: true }),
  },
})({
  canAutoUpdate: () => true,
  apply({ params }) {
    return new PluginStateObject.Shape.Provider(params.provider, { label: params.label });
  },
  update({ b, newParams }) {
    b.data = newParams.provider;
    b.label = newParams.label;
    return StateTransformer.UpdateResult.Updated;
  },
});

const BOX_EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [2, 3], [4, 5], [6, 7], // along a
  [0, 2], [1, 3], [4, 6], [5, 7], // along b
  [0, 4], [1, 5], [2, 6], [3, 7], // along c
];

/**
 * Computes the 12 wireframe edge segments (72 floats: 12 * [x0,y0,z0,x1,y1,z1])
 * for a triclinic or orthorhombic cell defined by row-major 3x3 `boxH`.
 */
export function computeBoxWireframeSegments(
  boxH: Float32Array,
  origin: ArrayLike<number> = [0, 0, 0],
): Float32Array {
  const ox = origin[0] ?? 0;
  const oy = origin[1] ?? 0;
  const oz = origin[2] ?? 0;

  // Lattice vectors are columns of row-major boxH
  const ax = boxH[0], ay = boxH[3], az = boxH[6];
  const bx = boxH[1], by = boxH[4], bz = boxH[7];
  const cx = boxH[2], cy = boxH[5], cz = boxH[8];

  const corners = new Float32Array(24);
  let idx = 0;
  for (let ic = 0; ic <= 1; ic++) {
    for (let ib = 0; ib <= 1; ib++) {
      for (let ia = 0; ia <= 1; ia++) {
        corners[idx++] = ox + ia * ax + ib * bx + ic * cx;
        corners[idx++] = oy + ia * ay + ib * by + ic * cy;
        corners[idx++] = oz + ia * az + ib * bz + ic * cz;
      }
    }
  }

  const segments = new Float32Array(BOX_EDGES.length * 6);
  for (let e = 0; e < BOX_EDGES.length; e++) {
    const [u, v] = BOX_EDGES[e];
    const u3 = u * 3;
    const v3 = v * 3;
    const s6 = e * 6;
    segments[s6] = corners[u3];
    segments[s6 + 1] = corners[u3 + 1];
    segments[s6 + 2] = corners[u3 + 2];
    segments[s6 + 3] = corners[v3];
    segments[s6 + 4] = corners[v3 + 1];
    segments[s6 + 5] = corners[v3 + 2];
  }
  return segments;
}

export function buildBoxWireframeShape(
  boxH: Float32Array,
  prevShape?: Shape<Lines>,
  origin: ArrayLike<number> = [0, 0, 0],
  color: Color = Color(0x60a5fa),
): Shape<Lines> {
  const segments = computeBoxWireframeSegments(boxH, origin);
  const builder = LinesBuilder.create(12, 12, prevShape?.geometry);
  for (let e = 0; e < 12; e++) {
    const s6 = e * 6;
    builder.add(
      segments[s6],
      segments[s6 + 1],
      segments[s6 + 2],
      segments[s6 + 3],
      segments[s6 + 4],
      segments[s6 + 5],
      0,
    );
  }
  const lines = builder.getLines();
  return ShapeUtils.create(
    'PBC Unit Cell',
    { boxH, origin },
    lines,
    () => color,
    () => 1.5,
    () => 'PBC Unit Cell',
  );
}

/**
 * Maps a covalent bond order (0..3+) to a distinct perceptual RGB color.
 */
export function bondOrderToColor(order: number): Color {
  const clamped = Math.max(0, Math.min(3, Number.isFinite(order) ? order : 1));
  if (clamped < 0.75) {
    // Weak / partial bond: slate blue
    return Color(0x64748b);
  }
  if (clamped < 1.35) {
    // Single bond: cool cyan-teal
    return Color(0x38bdf8);
  }
  if (clamped < 1.85) {
    // Aromatic / conjugated (~1.5): emerald green
    return Color(0x34d399);
  }
  if (clamped < 2.5) {
    // Double bond (~2.0): amber gold
    return Color(0xfbbf24);
  }
  // Triple / high-order bond (~3.0): coral orange
  return Color(0xf97316);
}

/**
 * Maps a covalent bond order (0..3+) to cylinder radius in Å.
 */
export function bondOrderToRadius(order: number): number {
  const clamped = Math.max(0.2, Math.min(3.0, Number.isFinite(order) ? order : 1.0));
  return 0.08 + 0.055 * clamped;
}

function invertRowMajor3x3(h: Float32Array, out: Float32Array): boolean {
  const a00 = h[0], a01 = h[1], a02 = h[2];
  const a10 = h[3], a11 = h[4], a12 = h[5];
  const a20 = h[6], a21 = h[7], a22 = h[8];

  const c00 = a11 * a22 - a12 * a21;
  const c01 = -(a10 * a22 - a12 * a20);
  const c02 = a10 * a21 - a11 * a20;

  const det = a00 * c00 + a01 * c01 + a02 * c02;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) return false;

  const invDet = 1.0 / det;
  out[0] = c00 * invDet;
  out[1] = -(a01 * a22 - a02 * a21) * invDet;
  out[2] = (a01 * a12 - a02 * a11) * invDet;
  out[3] = c01 * invDet;
  out[4] = (a00 * a22 - a02 * a20) * invDet;
  out[5] = -(a00 * a12 - a02 * a10) * invDet;
  out[6] = c02 * invDet;
  out[7] = -(a00 * a21 - a01 * a20) * invDet;
  out[8] = (a00 * a11 - a01 * a10) * invDet;
  return true;
}

/**
 * Builds PBC-safe covalent bond cylinders, skipping bonds that wrap across a periodic boundary
 * (fractional displacement > 0.5 in any lattice axis) and scaling/coloring by `bondOrders`.
 */
export function buildBondsShape(
  coords: Float32Array,
  bondPairs: Uint32Array,
  bondOrders?: Float32Array,
  boxH?: Float32Array,
  prevShape?: Shape<Cylinders>,
): Shape<Cylinders> {
  const nAtoms = Math.floor(coords.length / 3);
  const nBonds = Math.floor(bondPairs.length / 2);
  const builder = CylindersBuilder.create(Math.max(1, nBonds), 1024, prevShape?.geometry);

  const invH = new Float32Array(9);
  const hasPbc = boxH !== undefined && boxH.length >= 9 && invertRowMajor3x3(boxH, invH);

  const groupOrders: number[] = [];
  for (let b = 0; b < nBonds; b++) {
    const i = bondPairs[b * 2];
    const j = bondPairs[b * 2 + 1];
    if (i >= nAtoms || j >= nAtoms) continue;

    const i3 = i * 3;
    const j3 = j * 3;
    const x0 = coords[i3], y0 = coords[i3 + 1], z0 = coords[i3 + 2];
    const x1 = coords[j3], y1 = coords[j3 + 1], z1 = coords[j3 + 2];
    if (
      !Number.isFinite(x0) || !Number.isFinite(y0) || !Number.isFinite(z0) ||
      !Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(z1)
    ) {
      continue;
    }

    const dx = x1 - x0;
    const dy = y1 - y0;
    const dz = z1 - z0;
    const distSq = dx * dx + dy * dy + dz * dz;
    if (distSq < 1e-10) continue;

    if (hasPbc) {
      const fx = invH[0] * dx + invH[1] * dy + invH[2] * dz;
      const fy = invH[3] * dx + invH[4] * dy + invH[5] * dz;
      const fz = invH[6] * dx + invH[7] * dy + invH[8] * dz;
      if (Math.abs(fx) > 0.5 || Math.abs(fy) > 0.5 || Math.abs(fz) > 0.5) {
        continue;
      }
    }

    const order = bondOrders && b < bondOrders.length ? bondOrders[b] : 1.0;
    const groupId = groupOrders.length;
    groupOrders.push(order);
    builder.add(x0, y0, z0, x1, y1, z1, bondOrderToRadius(order), true, true, 2, groupId);
  }

  const cylinders = builder.getCylinders();
  return ShapeUtils.create(
    'Covalent Bonds',
    { nBonds: groupOrders.length },
    cylinders,
    (g) => bondOrderToColor(groupOrders[g] ?? 1.0),
    (g) => bondOrderToRadius(groupOrders[g] ?? 1.0),
    (g) => `Bond #${g} (order ${(groupOrders[g] ?? 1.0).toFixed(2)})`,
    undefined,
    Math.max(1, groupOrders.length),
  );
}

function forceMagToColor(normMag: number): Color {
  const t = Math.max(0, Math.min(1, normMag));
  // Interpolate cyan (#38bdf8) -> amber (#fbbf24) -> crimson (#ef4444)
  let r: number, g: number, b: number;
  if (t < 0.5) {
    const u = t * 2;
    r = Math.round(0x38 + (0xfb - 0x38) * u);
    g = Math.round(0xbd + (0xbf - 0xbd) * u);
    b = Math.round(0xf8 + (0x24 - 0xf8) * u);
  } else {
    const u = (t - 0.5) * 2;
    r = Math.round(0xfb + (0xef - 0xfb) * u);
    g = Math.round(0xbf + (0x44 - 0xbf) * u);
    b = Math.round(0x24 + (0x44 - 0x24) * u);
  }
  return Color.fromRgb(r, g, b);
}

/**
 * Builds 3D force vector arrows (shaft + cone tip) for non-zero ROI force vectors.
 */
export function buildForceVectorsShape(
  vectorStarts: Float32Array,
  vectorDirs: Float32Array,
  vectorMags: Float32Array,
  options: ForceVectorBuildOptions = {},
): Shape<Mesh> {
  const scale = options.scale ?? 1.0;
  const radius = options.radius ?? 0.08;
  const count = Math.min(
    vectorMags.length,
    Math.floor(vectorStarts.length / 3),
    Math.floor(vectorDirs.length / 3),
    options.maxVectors ?? vectorMags.length,
  );

  let maxMag = 0;
  for (let i = 0; i < count; i++) {
    const m = vectorMags[i];
    if (Number.isFinite(m) && m > maxMag) maxMag = m;
  }

  const state = MeshBuilder.createState(Math.max(64, count * 32), 1024, options.prevShape?.geometry);
  const start = Vec3();
  const shaftEnd = Vec3();
  const tipEnd = Vec3();
  const dir = Vec3();
  const mags: number[] = [];

  for (let i = 0; i < count; i++) {
    const mag = vectorMags[i];
    if (!Number.isFinite(mag) || mag <= 1e-8) continue;

    const i3 = i * 3;
    const sx = vectorStarts[i3], sy = vectorStarts[i3 + 1], sz = vectorStarts[i3 + 2];
    const dx = vectorDirs[i3], dy = vectorDirs[i3 + 1], dz = vectorDirs[i3 + 2];
    if (
      !Number.isFinite(sx) || !Number.isFinite(sy) || !Number.isFinite(sz) ||
      !Number.isFinite(dx) || !Number.isFinite(dy) || !Number.isFinite(dz)
    ) {
      continue;
    }

    const dirLen = Math.hypot(dx, dy, dz);
    if (dirLen < 1e-12) continue;

    Vec3.set(start, sx, sy, sz);
    Vec3.set(dir, dx / dirLen, dy / dirLen, dz / dirLen);

    const totalLen = Math.max(0.25, Math.min(8.0, mag * scale));
    const headLen = Math.min(0.45, totalLen * 0.32);
    const shaftLen = Math.max(0.05, totalLen - headLen);

    Vec3.scaleAndAdd(shaftEnd, start, dir, shaftLen);
    Vec3.scaleAndAdd(tipEnd, start, dir, totalLen);

    state.currentGroup = mags.length;
    mags.push(mag);

    addCylinder(state, start, shaftEnd, 1, {
      radiusTop: radius,
      radiusBottom: radius,
      radialSegments: 8,
      topCap: false,
      bottomCap: true,
    });
    addCylinder(state, shaftEnd, tipEnd, 1, {
      radiusTop: 0,
      radiusBottom: radius * 2.2,
      radialSegments: 10,
      topCap: false,
      bottomCap: true,
    });
  }

  const mesh = MeshBuilder.getMesh(state);
  const invMax = maxMag > 1e-12 ? 1.0 / maxMag : 1.0;

  return ShapeUtils.create(
    'ROI Force Vectors',
    { count: mags.length },
    mesh,
    (g) => forceMagToColor((mags[g] ?? 0) * invMax),
    () => 1,
    (g) => `Force Vector #${g}: ${(mags[g] ?? 0).toFixed(3)}`,
    undefined,
    Math.max(1, mags.length),
  );
}

interface OverlaySlot<G extends Lines | Cylinders | Mesh> {
  providerRef?: StateObjectSelector<PluginStateObject.Shape.Provider>;
  reprRef?: StateObjectSelector<PluginStateObject.Shape.Representation3D>;
  lastShape?: Shape<G>;
}

export class PhysicsOverlayManager {
  private readonly plugin: PluginContext;
  private visibility: OverlayVisibility = {
    showBox: true,
    showBonds: true,
    showForces: false,
  };

  private boxSlot: OverlaySlot<Lines> = {};
  private bondsSlot: OverlaySlot<Cylinders> = {};
  private forcesSlot: OverlaySlot<Mesh> = {};

  private lastBoxH?: Float32Array;
  private lastBoxOrigin: [number, number, number] = [0, 0, 0];
  private lastCoords?: Float32Array;
  private lastBondPairs?: Uint32Array;
  private lastBondOrders?: Float32Array;

  constructor(plugin: PluginContext) {
    this.plugin = plugin;
  }

  getVisibility(): OverlayVisibility {
    return { ...this.visibility };
  }

  async setVisibility(vis: Partial<OverlayVisibility>): Promise<void> {
    this.visibility = { ...this.visibility, ...vis };
    this.applySlotVisibility(this.boxSlot, this.visibility.showBox);
    this.applySlotVisibility(this.bondsSlot, this.visibility.showBonds);
    this.applySlotVisibility(this.forcesSlot, this.visibility.showForces);
  }

  private applySlotVisibility(slot: OverlaySlot<any>, visible: boolean): void {
    const ref = slot.providerRef?.ref;
    if (!ref || !this.plugin.state?.data?.cells?.has(ref)) return;
    setSubtreeVisibility(this.plugin.state.data, ref, !visible);
  }

  async setBoxOrigin(origin: [number, number, number]): Promise<void> {
    if (
      !Array.isArray(origin) ||
      origin.length < 3 ||
      !Number.isFinite(origin[0]) ||
      !Number.isFinite(origin[1]) ||
      !Number.isFinite(origin[2])
    ) {
      return;
    }
    const next: [number, number, number] = [origin[0], origin[1], origin[2]];
    const unchanged =
      Math.abs(next[0] - this.lastBoxOrigin[0]) < 1e-6 &&
      Math.abs(next[1] - this.lastBoxOrigin[1]) < 1e-6 &&
      Math.abs(next[2] - this.lastBoxOrigin[2]) < 1e-6;
    this.lastBoxOrigin = next;
    if (!unchanged && this.lastBoxH && this.visibility.showBox) {
      await this.updateBox(this.lastBoxH, this.lastBoxOrigin);
    }
  }

  async updateBox(boxH: Float32Array, origin?: [number, number, number]): Promise<void> {
    this.lastBoxH = new Float32Array(boxH);
    if (
      origin &&
      origin.length >= 3 &&
      Number.isFinite(origin[0]) &&
      Number.isFinite(origin[1]) &&
      Number.isFinite(origin[2])
    ) {
      this.lastBoxOrigin = [origin[0], origin[1], origin[2]];
    }
    const boxOrigin: [number, number, number] = [
      this.lastBoxOrigin[0],
      this.lastBoxOrigin[1],
      this.lastBoxOrigin[2],
    ];
    const provider: ShapeProvider<{ boxH: Float32Array; origin: [number, number, number] }, Lines, Lines.Params> = {
      label: 'PBC Unit Cell',
      data: { boxH: this.lastBoxH, origin: boxOrigin },
      params: Lines.Params,
      geometryUtils: Lines.Utils,
      getShape: (_ctx, data, _props, prev) => {
        const shape = buildBoxWireframeShape(data.boxH, prev, data.origin);
        this.boxSlot.lastShape = shape;
        return shape;
      },
    };
    await this.upsertOverlay(this.boxSlot, 'PBC Unit Cell', provider, this.visibility.showBox);
  }

  async updateTopology(msg: TopologyInitMessage, coords?: Float32Array): Promise<void> {
    this.lastBoxH = new Float32Array(msg.boxH);
    this.lastBondPairs = msg.bonds;
    if (coords) {
      this.lastCoords = coords;
    } else if (msg.coords) {
      this.lastCoords = msg.coords;
    }

    await this.updateBox(this.lastBoxH);
    if (this.lastCoords && this.lastBondPairs && this.lastBondPairs.length >= 2) {
      await this.renderBonds();
    }
  }

  async updateCoords(coords: Float32Array, boxH?: Float32Array): Promise<void> {
    this.lastCoords = coords;
    if (boxH) {
      await this.updateBox(boxH);
    }
    if (this.lastBondPairs && this.lastBondPairs.length >= 2 && this.visibility.showBonds) {
      await this.renderBonds();
    }
  }

  async updateOverlays(msg: OverlayVectorsMessage, coords?: Float32Array, boxH?: Float32Array): Promise<void> {
    if (coords) this.lastCoords = coords;
    if (boxH) this.lastBoxH = new Float32Array(boxH);

    if (msg.nBonds > 0 && msg.bondPairs.length >= 2) {
      this.lastBondPairs = msg.bondPairs;
      this.lastBondOrders = msg.bondOrders;
      if (this.lastCoords) {
        await this.renderBonds();
      }
    }

    const provider: ShapeProvider<OverlayVectorsMessage, Mesh, Mesh.Params> = {
      label: 'ROI Force Vectors',
      data: msg,
      params: Mesh.Params,
      geometryUtils: Mesh.Utils,
      getShape: (_ctx, data, _props, prev) => {
        const shape = buildForceVectorsShape(data.vectorStarts, data.vectorDirs, data.vectorMags, {
          prevShape: prev,
        });
        this.forcesSlot.lastShape = shape;
        return shape;
      },
    };
    await this.upsertOverlay(this.forcesSlot, 'ROI Force Vectors', provider, this.visibility.showForces);
  }

  private async renderBonds(): Promise<void> {
    if (!this.lastCoords || !this.lastBondPairs) return;
    const coords = this.lastCoords;
    const bondPairs = this.lastBondPairs;
    const bondOrders = this.lastBondOrders;
    const boxH = this.lastBoxH;

    const provider: ShapeProvider<
      { coords: Float32Array; bondPairs: Uint32Array; bondOrders?: Float32Array; boxH?: Float32Array },
      Cylinders,
      Cylinders.Params
    > = {
      label: 'Covalent Bonds',
      data: { coords, bondPairs, bondOrders, boxH },
      params: Cylinders.Params,
      geometryUtils: Cylinders.Utils,
      getShape: (_ctx, data, _props, prev) => {
        const shape = buildBondsShape(data.coords, data.bondPairs, data.bondOrders, data.boxH, prev);
        this.bondsSlot.lastShape = shape;
        return shape;
      },
    };
    await this.upsertOverlay(this.bondsSlot, 'Covalent Bonds', provider, this.visibility.showBonds);
  }

  private async upsertOverlay<G extends Lines | Cylinders | Mesh>(
    slot: OverlaySlot<G>,
    label: string,
    provider: ShapeProvider<any, G, any>,
    visible: boolean,
  ): Promise<void> {
    const state = this.plugin.state?.data;
    if (!state) return;

    const builder = this.plugin.build();
    if (slot.providerRef && state.cells.has(slot.providerRef.ref)) {
      builder.to(slot.providerRef).update(BiologyMmShapeProvider, () => ({ label, provider }));
    } else {
      const providerSel = builder
        .toRoot()
        .apply(BiologyMmShapeProvider, { label, provider }, { state: { isHidden: !visible } });
      const reprSel = providerSel.apply(ShapeRepresentation3D);
      slot.providerRef = providerSel.selector;
      slot.reprRef = reprSel.selector;
    }

    await builder.commit();
    this.applySlotVisibility(slot, visible);
  }

  async dispose(): Promise<void> {
    const state = this.plugin.state?.data;
    if (!state) return;

    const builder = this.plugin.build();
    let hasDeletes = false;
    for (const slot of [this.boxSlot, this.bondsSlot, this.forcesSlot]) {
      const ref = slot.providerRef?.ref;
      if (ref && state.cells.has(ref)) {
        builder.delete(ref);
        hasDeletes = true;
      }
      slot.providerRef = undefined;
      slot.reprRef = undefined;
      slot.lastShape = undefined;
    }

    if (hasDeletes) {
      await builder.commit();
    }
  }
}
