import { describe, expect, it, vi } from 'vitest';
import { Mat4, Vec3 } from '@molstar/core/math/linear-algebra';
import { Grid } from '@molstar/model/model/volume';
import type { PluginContext } from '@molstar/plugin/context';
import {
  MsgType,
  type OverlayVectorsMessage,
  type TriumHeader,
  type VolumeGrid3DMessage,
} from '../protocol.js';
import {
  PhysicsOverlayManager,
  bondOrderToColor,
  bondOrderToRadius,
  buildBondsShape,
  buildBoxWireframeShape,
  buildForceVectorsShape,
  computeBoxWireframeSegments,
} from '../physics-overlays.js';
import {
  TernaryVolumeBridge,
  computeVolumeGridMatrix,
  createVolumeFromGridMessage,
} from '../volume-bridge.js';

function makeHeader(msgType: number, nAtoms = 0): TriumHeader {
  return {
    msgType,
    flags: 0,
    frameId: 1,
    nAtoms,
    scalarMin: 0,
    scalarMax: 1,
    simTimePs: 0.5,
    payloadBytes: 0,
    reserved: 0,
  };
}

function createMockPlugin() {
  const cells = new Map<string, any>();
  const transforms = new Map<string, any>();
  const children = new Map<string, any>();
  let nextId = 1;
  const deletedRefs: string[] = [];
  const committedOps: string[] = [];

  transforms.set('root', { ref: 'root', parent: 'root' });
  children.set('root', new Set<string>());

  const makeChain = (ref: string): any => {
    const sel = { ref };
    return {
      selector: sel,
      apply: (_transform: any, _params: any, _opts?: any) => {
        const childRef = `node-${nextId++}`;
        cells.set(childRef, {
          transform: { ref: childRef },
          state: { isHidden: false },
          params: {},
        });
        transforms.set(childRef, { ref: childRef, parent: ref });
        if (!children.has(childRef)) children.set(childRef, new Set<string>());
        if (!children.has(ref)) children.set(ref, new Set<string>());
        children.get(ref)!.add(childRef);
        committedOps.push(`apply:${childRef}`);
        return makeChain(childRef);
      },
      update: (_transform: any, _fn: any) => {
        committedOps.push(`update:${ref}`);
        return makeChain(ref);
      },
    };
  };

  const stateData = {
    cells,
    transforms,
    tree: {
      transforms,
      children: {
        get: (ref: string) => children.get(ref) ?? new Set<string>(),
      },
    },
    build: () => ({
      toRoot: () => makeChain('root'),
      to: (sel: { ref: string }) => makeChain(sel.ref),
      delete: (ref: string) => {
        deletedRefs.push(ref);
        cells.delete(ref);
        transforms.delete(ref);
        return {
          commit: async () => {
            committedOps.push(`delete:${ref}`);
          },
        };
      },
      commit: async () => {
        committedOps.push('commit');
      },
    }),
    updateCellState: (ref: string, patch: any) => {
      const cell = cells.get(ref);
      if (cell) cell.state = { ...cell.state, ...patch };
    },
  };

  const plugin = {
    state: {
      data: stateData,
    },
    build: () => stateData.build(),
    representation: {
      volume: {
        registry: {
          has: () => true,
          types: [['direct-volume', 'Direct Volume']],
          default: {
            name: 'direct-volume',
            provider: {
              defaultColorTheme: { name: 'uniform' },
              defaultSizeTheme: { name: 'uniform' },
            },
          },
          get: () => ({
            defaultColorTheme: { name: 'uniform' },
            defaultSizeTheme: { name: 'uniform' },
            getParams: () => ({}),
          }),
        },
        themes: {
          colorThemeRegistry: {
            has: () => true,
            default: { name: 'uniform', provider: {} },
            types: [['uniform', 'Uniform']],
            get: () => ({ getParams: () => ({}) }),
          },
          sizeThemeRegistry: {
            has: () => true,
            default: { name: 'uniform', provider: {} },
            types: [['uniform', 'Uniform']],
            get: () => ({ getParams: () => ({}) }),
          },
        },
      },
    },
  } as unknown as PluginContext;

  return { plugin, cells, deletedRefs, committedOps };
}

describe('physics-overlays', () => {
  it('computes 12 wireframe segments for orthorhombic and triclinic boxH', () => {
    const orthoBoxH = new Float32Array([
      10, 0, 0,
      0, 20, 0,
      0, 0, 30,
    ]);
    const segs = computeBoxWireframeSegments(orthoBoxH);
    expect(segs.length).toBe(72); // 12 edges * 6 floats

    // First edge along a: (0,0,0) -> (10,0,0)
    expect(Array.from(segs.slice(0, 6))).toEqual([0, 0, 0, 10, 0, 0]);

    const shape = buildBoxWireframeShape(orthoBoxH);
    expect(shape.geometry.kind).toBe('lines');
    expect(shape.geometry.lineCount).toBe(12);

    // Triclinic box with shear
    const triBoxH = new Float32Array([
      10, 2, 0,
      0, 15, 3,
      0, 0, 20,
    ]);
    const triSegs = computeBoxWireframeSegments(triBoxH, [1, 2, 3]);
    expect(triSegs.length).toBe(72);
    // Origin corner is [1,2,3], +a is [11,2,3]
    expect(Array.from(triSegs.slice(0, 6))).toEqual([1, 2, 3, 11, 2, 3]);
  });

  it('scales and colors covalent bonds by bondOrders and filters PBC-wrapping bonds', () => {
    const cWeak = bondOrderToColor(0.4);
    const cSingle = bondOrderToColor(1.0);
    const cAromatic = bondOrderToColor(1.5);
    const cDouble = bondOrderToColor(2.0);
    const cTriple = bondOrderToColor(3.0);

    expect(new Set([cWeak, cSingle, cAromatic, cDouble, cTriple]).size).toBe(5);
    expect(bondOrderToRadius(0.5)).toBeLessThan(bondOrderToRadius(1.0));
    expect(bondOrderToRadius(1.0)).toBeLessThan(bondOrderToRadius(2.0));
    expect(bondOrderToRadius(2.0)).toBeLessThan(bondOrderToRadius(3.0));

    // Box of 20x20x20:
    // Atom 0 (1, 1, 1) - Atom 1 (2.5, 1, 1): valid in-box bond (dx = 1.5, fx = 0.075 <= 0.5)
    // Atom 0 (1, 1, 1) - Atom 2 (19.5, 1, 1): PBC-wrapped bond (dx = 18.5, fx = 0.925 > 0.5) -> skipped!
    const coords = new Float32Array([
      1, 1, 1,
      2.5, 1, 1,
      19.5, 1, 1,
    ]);
    const bondPairs = new Uint32Array([0, 1, 0, 2]);
    const bondOrders = new Float32Array([2.0, 1.0]);
    const boxH = new Float32Array([20, 0, 0, 0, 20, 0, 0, 0, 20]);

    const shape = buildBondsShape(coords, bondPairs, bondOrders, boxH);
    expect(shape.geometry.kind).toBe('cylinders');
    expect(shape.geometry.cylinderCount).toBe(1);
    expect(shape.getColor(0, 0)).toBe(cDouble);
    expect(shape.getSize(0, 0)).toBeCloseTo(bondOrderToRadius(2.0), 5);
  });

  it('generates mesh arrows only for non-zero force vectors', () => {
    const starts = new Float32Array([
      0, 0, 0,
      5, 5, 5,
      10, 10, 10,
    ]);
    const dirs = new Float32Array([
      1, 0, 0,
      0, 1, 0,
      0, 0, 1,
    ]);
    const mags = new Float32Array([2.5, 0.0, 4.0]); // middle vector is zero

    const shape = buildForceVectorsShape(starts, dirs, mags);
    expect(shape.geometry.kind).toBe('mesh');
    expect(shape.groupCount).toBe(2);
    expect(shape.geometry.vertexCount).toBeGreaterThan(0);
    expect(shape.geometry.triangleCount).toBeGreaterThan(0);
    // Higher magnitude should produce a different color than lower magnitude
    expect(shape.getColor(0, 0)).not.toBe(shape.getColor(1, 0));
  });

  it('manages visibility and dispose cleanly on PhysicsOverlayManager', async () => {
    const { plugin, cells, deletedRefs } = createMockPlugin();
    const manager = new PhysicsOverlayManager(plugin);

    expect(manager.getVisibility()).toEqual({
      showBox: true,
      showBonds: true,
      showForces: false,
    });

    await manager.updateBox(new Float32Array([10, 0, 0, 0, 10, 0, 0, 0, 10]));
    const overlayMsg: OverlayVectorsMessage = {
      kind: 'overlays',
      header: makeHeader(MsgType.OVERLAY_VECTORS),
      nVectors: 1,
      nBonds: 1,
      vectorStarts: new Float32Array([0, 0, 0]),
      vectorDirs: new Float32Array([0, 0, 1]),
      vectorMags: new Float32Array([1.5]),
      bondPairs: new Uint32Array([0, 1]),
      bondOrders: new Float32Array([1.5]),
    };
    await manager.updateOverlays(overlayMsg, new Float32Array([0, 0, 0, 1.4, 0, 0]));

    expect(cells.size).toBeGreaterThanOrEqual(3);

    await manager.setVisibility({ showBox: false, showForces: true });
    expect(manager.getVisibility()).toEqual({
      showBox: false,
      showBonds: true,
      showForces: true,
    });

    await manager.dispose();
    expect(deletedRefs.length).toBe(3);
  });
});

describe('volume-bridge', () => {
  it('constructs Volume Grid, Tensor, stats, and grid-to-cartesian matrix from VolumeGrid3DMessage', () => {
    const dims: [number, number, number] = [2, 2, 2];
    const origin = new Float32Array([1, 2, 3]);
    const boxH = new Float32Array([
      10, 0, 0,
      0, 20, 0,
      0, 0, 30,
    ]);
    const grid = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const msg: VolumeGrid3DMessage = {
      kind: 'volume-grid3d',
      header: makeHeader(MsgType.VOLUME_GRID3D),
      dims,
      origin,
      boxH,
      grid,
      channelNameId: 5,
    };

    const mat = computeVolumeGridMatrix(dims, origin, boxH);
    const p0 = Vec3.transformMat4(Vec3(), Vec3.create(0, 0, 0), mat);
    expect(Array.from(p0)).toEqual([1, 2, 3]);

    const pMax = Vec3.transformMat4(Vec3(), Vec3.create(2, 2, 2), mat);
    expect(Array.from(pMax)).toEqual([11, 22, 33]);

    const vol = createVolumeFromGridMessage(msg, 'Ternary Field');
    expect(vol.label).toBe('Ternary Field');
    expect(vol.grid.cells.space.dimensions).toEqual([2, 2, 2]);
    expect(vol.grid.cells.data).toBe(grid);
    expect(vol.grid.stats.min).toBe(1);
    expect(vol.grid.stats.max).toBe(8);
    expect(vol.grid.stats.mean).toBeCloseTo(4.5, 6);
    expect(vol.grid.stats.sigma).toBeGreaterThan(0);

    const g2c = Grid.getGridToCartesianTransform(vol.grid);
    expect(Mat4.areEqual(g2c, mat, 1e-6)).toBe(true);
  });

  it('mounts, toggles visibility, and disposes TernaryVolumeBridge', async () => {
    const { plugin, deletedRefs } = createMockPlugin();
    const bridge = new TernaryVolumeBridge(plugin);

    expect(bridge.isVisible()).toBe(false);
    bridge.setVisible(true);
    expect(bridge.isVisible()).toBe(true);

    const msg: VolumeGrid3DMessage = {
      kind: 'volume-grid3d',
      header: makeHeader(MsgType.VOLUME_GRID3D),
      dims: [2, 2, 2],
      origin: new Float32Array([0, 0, 0]),
      boxH: new Float32Array([8, 0, 0, 0, 8, 0, 0, 0, 8]),
      grid: new Float32Array([0, 1, 0, -1, 0, 1, -1, 0]),
      channelNameId: 5,
    };

    await bridge.updateFromMessage(msg);
    expect(bridge.getLastVolume()).toBeDefined();

    const updateSpy = vi.spyOn(plugin.state.data, 'updateCellState');
    bridge.setVisible(false);
    expect(bridge.isVisible()).toBe(false);
    expect(updateSpy).toHaveBeenCalled();

    await bridge.dispose();
    expect(bridge.getLastVolume()).toBeUndefined();
    expect(deletedRefs.length).toBe(1);
  });
});
