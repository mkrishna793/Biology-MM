/**
 * Volumetric 3D field bridge for Biology-MM:
 * Converts `VolumeGrid3DMessage` (`dims`, `origin`, `boxH`, `grid`) into a Mol* v6 `Volume`/`Grid`
 * and mounts/updates a `direct-volume` `VolumeRepresentation3D`.
 */

import type { PluginContext } from '@molstar/plugin/context';
import { PluginStateObject } from '@molstar/plugin/state/objects';
import { VolumeRepresentation3D } from '@molstar/plugin/state/transforms/volume/representation';
import { createVolumeRepresentationParams } from '@molstar/plugin/state/helpers/volume-representation-params';
import { setSubtreeVisibility } from '@molstar/plugin/behavior/static/state';
import { StateTransformer, type StateObjectSelector } from '@molstar/core/state';
import { ParamDefinition as PD } from '@molstar/core/util/param-definition';
import { Mat4, Tensor } from '@molstar/core/math/linear-algebra';
import { arrayStats } from '@molstar/core/util/array';
import { CustomProperties } from '@molstar/model/model/custom-property';
import type { ModelFormat } from '@molstar/model/formats/format';
import type { Grid, Volume } from '@molstar/model/model/volume';
import type { VolumeGrid3DMessage } from './protocol.js';

const Transform = StateTransformer.builderFactory('biology-mm');

export const BiologyMmVolumeData = Transform({
  name: 'biology-mm-volume-data',
  display: { name: 'Biology-MM 3D Field' },
  from: PluginStateObject.Root,
  to: PluginStateObject.Volume.Data,
  params: {
    label: PD.Text('3D Field'),
    volume: PD.Value<Volume>(undefined as any, { isHidden: true }),
  },
})({
  canAutoUpdate: () => true,
  apply({ params }) {
    const dims = params.volume.grid.cells.space.dimensions;
    return new PluginStateObject.Volume.Data(params.volume, {
      label: params.label,
      description: `${dims[0]}\u00D7${dims[1]}\u00D7${dims[2]}`,
    });
  },
  update({ b, newParams }) {
    const dims = newParams.volume.grid.cells.space.dimensions;
    b.data = newParams.volume;
    b.label = newParams.label;
    b.description = `${dims[0]}\u00D7${dims[1]}\u00D7${dims[2]}`;
    return StateTransformer.UpdateResult.Updated;
  },
  dispose({ b }) {
    b?.data.customProperties.dispose();
  },
});

/**
 * Computes the column-major 4x4 grid-to-Cartesian transform matrix from
 * grid dimensions `[nx, ny, nz]`, `origin` `[ox, oy, oz]`, and row-major 3x3 `boxH`.
 */
export function computeVolumeGridMatrix(
  dims: readonly [number, number, number],
  origin: Float32Array,
  boxH: Float32Array,
): Mat4 {
  const nx = Math.max(1, dims[0]);
  const ny = Math.max(1, dims[1]);
  const nz = Math.max(1, dims[2]);

  const m = Mat4();
  // Column 0: a / nx
  m[0] = boxH[0] / nx;
  m[1] = boxH[3] / nx;
  m[2] = boxH[6] / nx;
  m[3] = 0;

  // Column 1: b / ny
  m[4] = boxH[1] / ny;
  m[5] = boxH[4] / ny;
  m[6] = boxH[7] / ny;
  m[7] = 0;

  // Column 2: c / nz
  m[8] = boxH[2] / nz;
  m[9] = boxH[5] / nz;
  m[10] = boxH[8] / nz;
  m[11] = 0;

  // Column 3: origin
  m[12] = origin[0] ?? 0;
  m[13] = origin[1] ?? 0;
  m[14] = origin[2] ?? 0;
  m[15] = 1;

  return m;
}

/**
 * Converts a `VolumeGrid3DMessage` into a Mol* v6 `Volume` object with `Tensor`, `Grid`,
 * `stats` (`min`, `max`, `mean`, `sigma`), and grid-to-Cartesian `Mat4` transform.
 */
export function createVolumeFromGridMessage(msg: VolumeGrid3DMessage, label = '3D Field'): Volume {
  const space = Tensor.Space(msg.dims, [0, 1, 2], Float32Array);
  const cells = Tensor.create(space, Tensor.Data1(msg.grid));
  const stats = msg.grid.length > 0 ? arrayStats(msg.grid) : { min: 0, max: 0, mean: 0, rms: 0 };
  const matrix = computeVolumeGridMatrix(msg.dims, msg.origin, msg.boxH);

  const grid: Grid = {
    transform: { kind: 'matrix', matrix },
    cells,
    stats: {
      min: stats.min,
      max: stats.max,
      mean: stats.mean,
      sigma: stats.rms,
    },
  };

  const sourceData: ModelFormat<VolumeGrid3DMessage> = {
    kind: 'biology-mm-volume',
    name: label,
    data: msg,
  };

  return {
    label,
    grid,
    instances: [{ transform: Mat4.identity() }],
    sourceData,
    customProperties: new CustomProperties(),
    _propertyData: Object.create(null),
    _localPropertyData: Object.create(null),
  };
}

export class TernaryVolumeBridge {
  private readonly plugin: PluginContext;
  private visible = false;
  private dataRef?: StateObjectSelector<PluginStateObject.Volume.Data>;
  private lastVolume?: Volume;

  constructor(plugin: PluginContext) {
    this.plugin = plugin;
  }

  isVisible(): boolean {
    return this.visible;
  }

  getLastVolume(): Volume | undefined {
    return this.lastVolume;
  }

  setVisible(visible: boolean): void {
    this.visible = visible;
    const ref = this.dataRef?.ref;
    if (!ref || !this.plugin.state?.data?.cells?.has(ref)) return;
    setSubtreeVisibility(this.plugin.state.data, ref, !visible);
  }

  async updateFromMessage(msg: VolumeGrid3DMessage): Promise<void> {
    const label = `3D Field (ch ${msg.channelNameId})`;
    const volume = createVolumeFromGridMessage(msg, label);
    this.lastVolume = volume;

    const state = this.plugin.state?.data;
    if (!state) return;

    const builder = this.plugin.build();
    if (this.dataRef && state.cells.has(this.dataRef.ref)) {
      builder.to(this.dataRef).update(BiologyMmVolumeData, () => ({ label, volume }));
    } else {
      const dataSel = builder
        .toRoot()
        .apply(BiologyMmVolumeData, { label, volume }, { state: { isHidden: !this.visible } });

      const reprParams = createVolumeRepresentationParams(this.plugin, volume, {
        type: 'direct-volume',
      });
      dataSel.apply(VolumeRepresentation3D, reprParams);

      this.dataRef = dataSel.selector;
    }

    await builder.commit();
    this.setVisible(this.visible);
  }

  async dispose(): Promise<void> {
    const state = this.plugin.state?.data;
    if (state && this.dataRef?.ref && state.cells.has(this.dataRef.ref)) {
      await this.plugin.build().delete(this.dataRef.ref).commit();
    }
    this.dataRef = undefined;
    this.lastVolume = undefined;
  }
}
