/**
 * Native Mol* v6 Structure Bridge for Biology-MM Studio.
 *
 * Loads backend mmCIF topology into Mol*'s full native Structure hierarchy
 * (Polymer Cartoon + Ligand Ball & Stick + Ion + Water + Unit Cell + Quick Styles),
 * drives zero-flicker in-place coordinate & physics-channel updates through a
 * `MutableLiveTrajectory`, and extracts arbitrary imported/downloaded Mol*
 * structures into Biology-MM simulation payloads.
 */

import { Cell } from '@molstar/core/math/geometry/spacegroup/cell';
import { Vec3 } from '@molstar/core/math/linear-algebra';
import { StateTransformer } from '@molstar/core/state';
import { ParamDefinition as PD } from '@molstar/core/util/param-definition';
import { ModelSymmetry } from '@molstar/model/formats/structure/property/symmetry';
import {
  Coordinates,
  type Frame,
  Model,
  type Trajectory,
  Unit,
} from '@molstar/model/model/structure';
import { setSubtreeVisibility } from '@molstar/plugin/behavior/static/state';
import type { PluginContext } from '@molstar/plugin/context';
import { PluginStateObject as SO } from '@molstar/plugin/state/objects';
import type { PaletteName } from './color-palettes.js';
import {
  BIOLOGY_MM_SCALARS_KEY,
  PHYSICS_CHANNEL_TO_STRUCT_THEME,
  type BiologyMmStructureScalarMap,
} from './physics-color-themes.js';
import type { PhysicsChannelName } from './protocol.js';

export interface ExtractedEntityMeta {
  readonly id: string;
  readonly type: string;
  readonly description: string;
}

export interface ExtractedMolstarStructure {
  readonly title: string;
  readonly coords_ang: [number, number, number][];
  readonly elements: string[];
  readonly charges: number[];
  readonly bonds: [number, number][];
  readonly box_ang: [number, number, number];
  readonly structure_metadata: {
    readonly group_pdb: string[];
    readonly atom_id: string[];
    readonly comp_id: string[];
    readonly asym_id: string[];
    readonly entity_id: string[];
    readonly seq_id: number[];
    readonly entities: ExtractedEntityMeta[];
  };
  readonly truncated: boolean;
  readonly original_atom_count: number;
}

/**
 * Extract atomic geometry, multi-entity metadata, formal charges, covalent bonds,
 * and unit-cell dimensions from the currently loaded Mol* `Structure` in the
 * exact `ActiveStructurePayload` schema expected by Biology-MM Studio's backend.
 *
 * When both an external user-imported structure and a live Biology-MM simulation
 * structure are present, prefers the most recently imported external structure
 * (`excludeStructureRef` filters out the live Biology-MM structure).
 */
export function extractActiveMolstarStructure(
  plugin: PluginContext,
  excludeStructureRef?: string,
  maxAtoms = 100_000_000,
): ExtractedMolstarStructure | null {
  const structures = plugin.managers?.structure?.hierarchy?.current?.structures;
  if (!structures || structures.length === 0) return null;

  const external = excludeStructureRef
    ? structures.filter((s) => s.cell.transform.ref !== excludeStructureRef)
    : structures;
  const targetRef = external.length > 0 ? external[external.length - 1] : structures[structures.length - 1];
  const structure = targetRef?.cell?.obj?.data;
  if (!structure) return null;

  const title = targetRef.cell.obj?.label || structure.label || 'Imported Mol* Structure';

  let originalAtomCount = 0;
  for (const u of structure.units) {
    if (Unit.isAtomic(u)) originalAtomCount += u.elements.length;
  }
  if (originalAtomCount === 0) return null;

  const coords_ang: [number, number, number][] = [];
  const elements: string[] = [];
  const charges: number[] = [];
  const bonds: [number, number][] = [];

  const group_pdb: string[] = [];
  const atom_id: string[] = [];
  const comp_id: string[] = [];
  const asym_id: string[] = [];
  const entity_id: string[] = [];
  const seq_id: number[] = [];
  const entitiesMap = new Map<string, ExtractedEntityMeta>();

  const unitAtomToGlobal = new Map<number, Map<number, number>>();
  const tmpVec = Vec3();
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  let truncated = false;
  let cellBoxAng: [number, number, number] | undefined;

  for (const unit of structure.units) {
    if (!Unit.isAtomic(unit)) continue;
    const { elements: els, conformation, model } = unit;
    const { atoms, residues, chains, residueAtomSegments, chainAtomSegments, index: hIndex } = model.atomicHierarchy;
    const entityData = model.entities.data;

    if (!cellBoxAng && model.sourceData) {
      const sym = ModelSymmetry.Provider.get(model);
      const cellSize = sym?.spacegroup.cell.size;
      if (cellSize && cellSize[0] > 0 && cellSize[1] > 0 && cellSize[2] > 0) {
        cellBoxAng = [cellSize[0], cellSize[1], cellSize[2]];
      }
    }

    const localMap = new Map<number, number>();
    unitAtomToGlobal.set(unit.id, localMap);

    for (let uIdx = 0, len = els.length; uIdx < len; uIdx++) {
      if (elements.length >= maxAtoms) {
        truncated = true;
        break;
      }
      const el = els[uIdx];
      conformation.position(el, tmpVec);
      const xAng = tmpVec[0];
      const yAng = tmpVec[1];
      const zAng = tmpVec[2];

      if (xAng < minX) minX = xAng;
      if (yAng < minY) minY = yAng;
      if (zAng < minZ) minZ = zAng;
      if (xAng > maxX) maxX = xAng;
      if (yAng > maxY) maxY = yAng;
      if (zAng > maxZ) maxZ = zAng;

      const globalIdx = elements.length;
      localMap.set(uIdx, globalIdx);

      const rI = residueAtomSegments.index[el];
      const cI = chainAtomSegments.index[el];
      const eI = hIndex.getEntityFromChain(cI);

      const sym = String(atoms.type_symbol.value(el) || 'C');
      const atomName = String(atoms.label_atom_id.value(el) || sym || 'CA');
      const compName = String(atoms.label_comp_id.value(el) || atoms.auth_comp_id.value(el) || 'MOL');
      const charge = atoms.pdbx_formal_charge.isDefined ? Number(atoms.pdbx_formal_charge.value(el) || 0) : 0;

      const seqIdRaw = residues.label_seq_id.value(rI);
      const authSeqRaw = residues.auth_seq_id.value(rI);
      const seqIdVal = seqIdRaw > 0 ? seqIdRaw : Math.max(1, authSeqRaw || 1);
      const groupPdbVal = String(residues.group_PDB.value(rI) || 'ATOM');
      const chainIdVal = String(chains.label_asym_id.value(cI) || chains.auth_asym_id.value(cI) || 'A');
      const entIdVal = eI >= 0 ? String(entityData.id.value(eI) || '1') : '1';
      const entTypeVal = eI >= 0 ? String(entityData.type.value(eI) || 'polymer') : 'polymer';
      const entDescVal =
        eI >= 0 && entityData.pdbx_description.isDefined
          ? String(entityData.pdbx_description.value(eI) || compName)
          : compName;

      if (!entitiesMap.has(entIdVal)) {
        entitiesMap.set(entIdVal, {
          id: entIdVal,
          type: entTypeVal,
          description: entDescVal,
        });
      }

      coords_ang.push([xAng, yAng, zAng]);
      elements.push(sym);
      charges.push(Number.isFinite(charge) ? charge : 0);
      group_pdb.push(groupPdbVal);
      atom_id.push(atomName);
      comp_id.push(compName);
      asym_id.push(chainIdVal);
      entity_id.push(entIdVal);
      seq_id.push(seqIdVal);
    }

    const intraBonds = unit.bonds;
    for (let e = 0, ec = intraBonds.edgeCount; e < ec; e++) {
      const ga = localMap.get(intraBonds.a[e]);
      const gb = localMap.get(intraBonds.b[e]);
      if (ga !== undefined && gb !== undefined && ga < gb) {
        bonds.push([ga, gb]);
      }
    }

    if (truncated) break;
  }

  if (structure.interUnitBonds && structure.interUnitBonds.edgeCount > 0) {
    const seen = new Set<string>();
    for (const [a, b] of bonds) seen.add(`${a}:${b}`);
    for (const edge of structure.interUnitBonds.edges) {
      const ga = unitAtomToGlobal.get(edge.unitA)?.get(edge.indexA);
      const gb = unitAtomToGlobal.get(edge.unitB)?.get(edge.indexB);
      if (ga === undefined || gb === undefined || ga === gb) continue;
      const lo = Math.min(ga, gb);
      const hi = Math.max(ga, gb);
      const key = `${lo}:${hi}`;
      if (!seen.has(key)) {
        seen.add(key);
        bonds.push([lo, hi]);
      }
    }
  }

  const fallbackBoxAng: [number, number, number] = [
    Math.max(20.0, maxX - minX + 12.0),
    Math.max(20.0, maxY - minY + 12.0),
    Math.max(20.0, maxZ - minZ + 12.0),
  ];

  return {
    title,
    coords_ang,
    elements,
    charges,
    bonds,
    box_ang: cellBoxAng ?? fallbackBoxAng,
    structure_metadata: {
      group_pdb,
      atom_id,
      comp_id,
      asym_id,
      entity_id,
      seq_id,
      entities: Array.from(entitiesMap.values()),
    },
    truncated,
    original_atom_count: originalAtomCount,
  };
}

/**
 * Single-frame mutable `Trajectory` adapter that preserves the object reference
 * expected by Mol*'s `ModelFromTrajectory.update` (`cache.trajectory === a.data`),
 * enabling flicker-free in-place `Structure` and representation updates.
 */
class MutableLiveTrajectory implements Trajectory {
  readonly duration = 1;
  readonly frameCount = 1;
  private currentModel: Model;

  constructor(initialModel: Model) {
    this.currentModel = initialModel;
  }

  get representative(): Model {
    return this.currentModel;
  }

  getFrameAtIndex(_i: number): Model {
    return this.currentModel;
  }

  setCurrentModel(model: Model): void {
    this.currentModel = model;
  }
}

const Transform = StateTransformer.builderFactory('biology-mm');

export const BiologyMmLiveTrajectoryTransform = Transform({
  name: 'biology-mm-live-trajectory',
  display: {
    name: 'Biology-MM Live Trajectory',
    description: 'Adapts static mmCIF topology to live frame-by-frame simulation updates',
  },
  from: SO.Molecule.Trajectory,
  to: SO.Molecule.Trajectory,
  params: {
    version: PD.Numeric(0, {}, { isHidden: true }),
    label: PD.Text('Biology-MM Simulation', { isHidden: true }),
  },
})({
  canAutoUpdate: () => true,
  apply({ a, params }) {
    const baseModel = a.data.representative;
    const traj = new MutableLiveTrajectory(baseModel);
    return new SO.Molecule.Trajectory(traj, {
      label: params.label || a.label || 'Biology-MM Simulation',
      description: `${baseModel.atomicHierarchy.atoms._rowCount} atoms`,
    });
  },
  update({ b, newParams }) {
    b.label = newParams.label || b.label;
    return StateTransformer.UpdateResult.Updated;
  },
});

export class BiologyMmStructureBridge {
  private readonly plugin: PluginContext;

  private dataRef: string | undefined;
  private liveTrajRef: string | undefined;
  private structureRef: string | undefined;
  private unitcellRef: string | undefined;

  private baseModel: Model | undefined;
  private mutableTraj: MutableLiveTrajectory | undefined;
  private currentLabel = 'Biology-MM Simulation';
  private frameVersion = 0;

  private xBuf: Float32Array | undefined;
  private yBuf: Float32Array | undefined;
  private zBuf: Float32Array | undefined;

  private colorMode: 'default' | PhysicsChannelName = 'default';
  private palette: PaletteName = 'plasma';
  private unitCellVisible = true;
  private isUpdating = false;
  private disposed = false;

  constructor(plugin: PluginContext) {
    this.plugin = plugin;
  }

  get isLoaded(): boolean {
    return Boolean(this.baseModel && this.mutableTraj && this.liveTrajRef);
  }

  get liveStructureRef(): string | undefined {
    return this.structureRef;
  }

  get currentColorMode(): 'default' | PhysicsChannelName {
    return this.colorMode;
  }

  setPendingAppearance(mode: 'default' | PhysicsChannelName, palette: PaletteName, showUnitCell?: boolean): void {
    this.colorMode = mode;
    this.palette = palette;
    if (showUnitCell !== undefined) {
      this.unitCellVisible = showUnitCell;
    }
  }

  async loadSessionStructure(
    cifText: string,
    label: string,
    initialScalars?: BiologyMmStructureScalarMap,
  ): Promise<boolean> {
    if (this.disposed) return false;
    await this.dispose();

    const state = this.plugin.state?.data;
    if (!state) return false;

    this.currentLabel = label;
    this.isUpdating = true;
    try {
      const rawData = await this.plugin.builders.data.rawData({ data: cifText, label });
      if (!rawData?.ref) return false;
      this.dataRef = rawData.ref;

      const parsedTraj = await this.plugin.builders.structure.parseTrajectory(rawData, 'mmcif');
      if (!parsedTraj?.ref) return false;

      const liveTrajSel = await state
        .build()
        .to(parsedTraj)
        .apply(BiologyMmLiveTrajectoryTransform, { version: 0, label })
        .commit();

      const trajObj = liveTrajSel.cell?.obj?.data;
      if (!(trajObj instanceof MutableLiveTrajectory)) {
        return false;
      }

      this.liveTrajRef = liveTrajSel.ref;
      this.mutableTraj = trajObj;
      this.baseModel = trajObj.representative;
      this.frameVersion = 0;

      if (initialScalars) {
        this.attachScalars(this.baseModel, initialScalars);
      }

      const presetRes = await this.plugin.builders.structure.hierarchy.applyPreset(liveTrajSel, 'default', {
        showUnitcell: true,
      });

      this.structureRef = presetRes?.structureProperties?.ref ?? presetRes?.structure?.ref;
      this.unitcellRef = presetRes?.unitcell?.ref;

      this.setUnitCellVisible(this.unitCellVisible);

      if (this.colorMode !== 'default') {
        await this.applyColorThemeToLiveStructure();
      }

      this.plugin.canvas3d?.requestCameraReset();
      return true;
    } finally {
      this.isUpdating = false;
    }
  }

  async updateLiveFrame(
    coordsAng: Float32Array,
    count: number,
    boxH: Float32Array,
    scalars: BiologyMmStructureScalarMap,
    simTimePs: number,
  ): Promise<void> {
    if (this.disposed || this.isUpdating) return;
    if (!this.baseModel || !this.mutableTraj || !this.liveTrajRef) return;

    const state = this.plugin.state?.data;
    if (!state || !state.cells.has(this.liveTrajRef)) return;

    const modelAtoms = this.baseModel.atomicHierarchy.atoms._rowCount;
    if (count <= 0 || count !== modelAtoms || coordsAng.length < count * 3) return;

    this.isUpdating = true;
    try {
      if (!this.xBuf || this.xBuf.length !== count) {
        this.xBuf = new Float32Array(count);
        this.yBuf = new Float32Array(count);
        this.zBuf = new Float32Array(count);
      }
      const x = this.xBuf;
      const y = this.yBuf!;
      const z = this.zBuf!;

      for (let i = 0; i < count; i++) {
        const i3 = i * 3;
        x[i] = coordsAng[i3];
        y[i] = coordsAng[i3 + 1];
        z[i] = coordsAng[i3 + 2];
      }

      let cell: Cell | undefined;
      if (boxH.length >= 9 && boxH[0] > 0.5 && boxH[4] > 0.5 && boxH[8] > 0.5) {
        cell = Cell.create(
          Vec3.create(boxH[0], boxH[4], boxH[8]),
          Vec3.create(Math.PI / 2, Math.PI / 2, Math.PI / 2),
        );
      }

      const time = { value: simTimePs, unit: 'ps' as const };
      const frame: Frame = {
        elementCount: count,
        time,
        x,
        y,
        z,
        cell,
        xyzOrdering: { isIdentity: true },
      };

      const stepTime = { value: 0.001, unit: 'ps' as const };
      const nextTraj = Model.trajectoryFromModelAndCoordinates(
        this.baseModel,
        Coordinates.create([frame], stepTime, time),
      );
      const nextModel = nextTraj.representative;

      this.attachScalars(this.baseModel, scalars);
      this.attachScalars(nextModel, scalars);
      this.mutableTraj.setCurrentModel(nextModel);

      await state
        .build()
        .to(this.liveTrajRef)
        .update({ version: ++this.frameVersion, label: this.currentLabel })
        .commit();
    } finally {
      this.isUpdating = false;
    }
  }

  async setColorMode(mode: 'default' | PhysicsChannelName, palette: PaletteName): Promise<void> {
    this.colorMode = mode;
    this.palette = palette;
    if (!this.isLoaded) return;
    await this.applyColorThemeToLiveStructure();
  }

  setUnitCellVisible(visible: boolean): void {
    this.unitCellVisible = visible;
    const state = this.plugin.state?.data;
    if (!state || !this.unitcellRef || !state.cells.has(this.unitcellRef)) return;
    setSubtreeVisibility(state, this.unitcellRef, !visible);
  }

  async dispose(): Promise<void> {
    const refToDelete = this.dataRef;
    this.dataRef = undefined;
    this.liveTrajRef = undefined;
    this.structureRef = undefined;
    this.unitcellRef = undefined;
    this.baseModel = undefined;
    this.mutableTraj = undefined;
    this.xBuf = undefined;
    this.yBuf = undefined;
    this.zBuf = undefined;

    const state = this.plugin.state?.data;
    if (refToDelete && state && state.cells.has(refToDelete)) {
      try {
        await state.build().delete(refToDelete).commit();
      } catch {
        // Ignore teardown errors if plugin state is already disposing
      }
    }
  }

  destroy(): void {
    this.disposed = true;
    void this.dispose();
  }

  private attachScalars(model: Model, scalars: BiologyMmStructureScalarMap): void {
    model._staticPropertyData[BIOLOGY_MM_SCALARS_KEY] = scalars;
    model._dynamicPropertyData[BIOLOGY_MM_SCALARS_KEY] = scalars;
  }

  private async applyColorThemeToLiveStructure(): Promise<void> {
    const structures = this.plugin.managers?.structure?.hierarchy?.current?.structures;
    const compManager = this.plugin.managers?.structure?.component;
    if (!structures || structures.length === 0 || !compManager) return;

    const liveStruct = this.structureRef
      ? structures.find((s) => s.cell.transform.ref === this.structureRef)
      : undefined;
    if (!liveStruct) return;
    const components = liveStruct.components;
    if (!components || components.length === 0) return;

    if (this.colorMode === 'default') {
      await compManager.updateRepresentationsTheme(components, { color: 'default' });
    } else {
      const themeName = PHYSICS_CHANNEL_TO_STRUCT_THEME[this.colorMode];
      await compManager.updateRepresentationsTheme(components, {
        color: themeName as any,
        colorParams: {
          palette: this.palette,
          domain: { name: 'auto', params: {} },
        },
      });
    }
  }
}


export interface ExtractedMolstarBinaryStructure {
  readonly title: string;
  readonly nAtoms: number;
  readonly buffer: ArrayBuffer;
  readonly box_ang?: [number, number, number];
}

export function extractActiveMolstarStructureBinary(
  plugin: PluginContext,
  excludeStructureRef?: string,
  options?: { mode?: string; temperature?: number; dt?: number },
): ExtractedMolstarBinaryStructure | null {
  const structures = plugin.managers?.structure?.hierarchy?.current?.structures;
  if (!structures || structures.length === 0) return null;

  const external = excludeStructureRef
    ? structures.filter((s) => s.cell.transform.ref !== excludeStructureRef)
    : structures;
  const targetRef = external.length > 0 ? external[external.length - 1] : structures[structures.length - 1];
  const structure = targetRef?.cell?.obj?.data;
  if (!structure) return null;

  const title = targetRef.cell.obj?.label || structure.label || 'Imported Mol* Structure';

  let totalAtoms = 0;
  for (const u of structure.units) {
    if (Unit.isAtomic(u)) totalAtoms += u.elements.length;
  }
  if (totalAtoms === 0) return null;

  const titleBytes = new TextEncoder().encode(title);
  const modeStr = options?.mode ?? 'metal_verlet';
  const modeBytes = new TextEncoder().encode(modeStr);

  const headerSize = 48; // multiple of 4 and 8
  const coordsBytes = totalAtoms * 3 * 4; // Float32 (always multiple of 4)
  const elemBytes = totalAtoms; // Uint8
  const totalBufferSize = headerSize + coordsBytes + elemBytes + titleBytes.length + modeBytes.length;

  const ab = new ArrayBuffer(totalBufferSize);
  const view = new DataView(ab);
  const u8View = new Uint8Array(ab);

  // Magic 'TBIN'
  view.setUint8(0, 0x54); // 'T'
  view.setUint8(1, 0x42); // 'B'
  view.setUint8(2, 0x49); // 'I'
  view.setUint8(3, 0x4e); // 'N'
  view.setUint32(4, totalAtoms, true);
  view.setUint32(8, titleBytes.length, true);
  view.setFloat32(12, options?.temperature ?? 300.0, true);
  view.setFloat32(16, options?.dt ?? 0.001, true);
  view.setUint32(20, modeBytes.length, true);
  view.setUint32(24, 0x01, true); // flags: has_elements = 1

  let cellBoxAng: [number, number, number] | undefined;
  for (const u of structure.units) {
    if (Unit.isAtomic(u) && u.model.sourceData) {
      const sym = ModelSymmetry.Provider.get(u.model);
      const cSize = sym?.spacegroup.cell.size;
      if (cSize && cSize[0] > 0) {
        cellBoxAng = [cSize[0], cSize[1], cSize[2]];
        break;
      }
    }
  }

  view.setFloat32(28, cellBoxAng ? cellBoxAng[0] : 0.0, true);
  view.setFloat32(32, cellBoxAng ? cellBoxAng[1] : 0.0, true);
  view.setFloat32(36, cellBoxAng ? cellBoxAng[2] : 0.0, true);
  view.setUint32(40, 0, true); // reserved1 (align header to exactly 48 bytes)
  view.setUint32(44, 0, true); // reserved2

  // Place float32Coords immediately after the 48-byte header to guarantee 4-byte alignment
  const coordsOffset = headerSize;
  const float32Coords = new Float32Array(ab, coordsOffset, totalAtoms * 3);
  const elemOffset = coordsOffset + coordsBytes;

  let strOffset = elemOffset + elemBytes;
  u8View.set(titleBytes, strOffset);
  strOffset += titleBytes.length;
  u8View.set(modeBytes, strOffset);

  const tmpVec = Vec3();
  let atomIdx = 0;

  for (const unit of structure.units) {
    if (!Unit.isAtomic(unit)) continue;
    const { elements: els, conformation, model } = unit;
    const type_symbol = model.atomicHierarchy.atoms.type_symbol;

    for (let uIdx = 0, len = els.length; uIdx < len; ++uIdx) {
      const el = els[uIdx];
      conformation.position(el, tmpVec);
      const cBase = atomIdx * 3;
      float32Coords[cBase + 0] = tmpVec[0];
      float32Coords[cBase + 1] = tmpVec[1];
      float32Coords[cBase + 2] = tmpVec[2];

      const sym = type_symbol.value(el);
      u8View[elemOffset + atomIdx] = sym ? sym.charCodeAt(0) : 67; // 'C'
      atomIdx++;
    }
  }

  return {
    title,
    nAtoms: totalAtoms,
    buffer: ab,
    box_ang: cellBoxAng,
  };
}


export interface ExtractedMolstarSymmetryBinary {
  readonly title: string;
  readonly nAsuAtoms: number;
  readonly nTransforms: number;
  readonly totalAtoms: number;
  readonly buffer: ArrayBuffer;
  readonly box_ang?: [number, number, number];
}

export interface SymmetryBinaryOptions {
  readonly excludeStructureRef?: string;
  readonly mode?: string;
  readonly temperature?: number;
  readonly dt?: number;
}

function matrixKey(m: ArrayLike<number>): string {
  const parts: string[] = new Array(16);
  for (let i = 0; i < 16; i++) {
    const v = Math.abs(m[i]) < 1e-5 ? 0 : m[i];
    parts[i] = v.toFixed(4);
  }
  return parts.join(',');
}

/**
 * Detect when the active Mol* structure consists of multiple symmetry-related copies
 * of a common asymmetric unit (e.g., viral capsid Assembly 1) and pack a strictly
 * 4-byte-aligned `TSYM` binary payload:
 *
 *   48-byte header `<4sIIIffIIfffI`:
 *     magic='TSYM', n_asu_atoms, n_transforms, title_len, temperature, dt,
 *     mode_len, flags (0x01 = has_elements), bx, by, bz, reserved=0
 *   Byte offset 48:
 *     Float32Array ASU coords (3 * n_asu_atoms)
 *     Float32Array 4x4 row-major transforms (16 * n_transforms)
 *     Uint8Array elements (n_asu_atoms, when flags & 0x01)
 *     UTF-8 title bytes, UTF-8 mode bytes
 *
 * Returns `null` when the structure does not have >1 uniform symmetry transforms.
 */
export function extractActiveMolstarSymmetryBinary(
  plugin: PluginContext,
  optionsOrExclude?: string | SymmetryBinaryOptions,
  optionsArg?: { mode?: string; temperature?: number; dt?: number },
): ExtractedMolstarSymmetryBinary | null {
  const excludeStructureRef =
    typeof optionsOrExclude === 'string' ? optionsOrExclude : optionsOrExclude?.excludeStructureRef;
  const options =
    typeof optionsOrExclude === 'string' ? optionsArg : optionsOrExclude;

  const structures = plugin.managers?.structure?.hierarchy?.current?.structures;
  if (!structures || structures.length === 0) return null;

  const external = excludeStructureRef
    ? structures.filter((s) => s.cell.transform.ref !== excludeStructureRef)
    : structures;
  const targetRef = external.length > 0 ? external[external.length - 1] : structures[structures.length - 1];
  const structure = targetRef?.cell?.obj?.data;
  if (!structure) return null;

  const atomicUnits: Unit.Atomic[] = [];
  for (const u of structure.units) {
    if (Unit.isAtomic(u) && u.elements.length > 0) {
      atomicUnits.push(u);
    }
  }
  if (atomicUnits.length <= 1) return null;

  // Group atomic units by `unit.invariantId` (falling back to model + element count).
  // In Mol* biological assemblies (e.g. 5J7V Assembly 1 with 2,760 units, whether
  // rendered as Gaussian Surface, Spacefill, or Cartoon), symmetry copies of the same
  // ASU subunit share `invariantId` and differ only by `unit.conformation.operator.matrix`.
  // Grouping by invariantId avoids ever allocating a 40M-atom array in Chrome and stays
  // resilient even if a structure contains minor non-replicated hetero/solvent units.
  const invariantGroups = new Map<string, { units: Unit.Atomic[]; atomCount: number; totalGroupAtoms: number }>();
  for (const u of atomicUnits) {
    const invKey =
      typeof u.invariantId === 'number'
        ? `inv:${u.invariantId}`
        : `mod:${u.model?.id ?? '0'}:${u.elements.length}`;
    let group = invariantGroups.get(invKey);
    if (!group) {
      group = { units: [], atomCount: u.elements.length, totalGroupAtoms: 0 };
      invariantGroups.set(invKey, group);
    }
    group.units.push(u);
    group.totalGroupAtoms += u.elements.length;
  }

  // Pick the dominant replicated invariant group (must have >1 symmetry copies)
  let primaryGroup: { units: Unit.Atomic[]; atomCount: number; totalGroupAtoms: number } | null = null;
  for (const group of invariantGroups.values()) {
    if (group.units.length <= 1) continue;
    if (!primaryGroup || group.totalGroupAtoms > primaryGroup.totalGroupAtoms) {
      primaryGroup = group;
    }
  }
  if (!primaryGroup) return null;

  // Collect all 4x4 operator matrices from the primary replicated group
  const transforms: ArrayLike<number>[] = [];
  const primaryOpKeys = new Set<string>();
  for (const u of primaryGroup.units) {
    const mat = u.conformation.operator.matrix;
    transforms.push(mat);
    primaryOpKeys.add(matrixKey(mat));
  }
  const nTransforms = transforms.length;
  if (nTransforms <= 1) return null;

  // Include any co-replicated ASU subunit groups sharing the exact same transform set
  const asuUnits: Unit.Atomic[] = [primaryGroup.units[0]];
  let nAsuAtoms = primaryGroup.atomCount;
  for (const group of invariantGroups.values()) {
    if (group === primaryGroup || group.units.length !== nTransforms) continue;
    let matchesPrimaryOps = true;
    for (const u of group.units) {
      if (!primaryOpKeys.has(matrixKey(u.conformation.operator.matrix))) {
        matchesPrimaryOps = false;
        break;
      }
    }
    if (matchesPrimaryOps) {
      asuUnits.push(group.units[0]);
      nAsuAtoms += group.atomCount;
    }
  }
  if (nAsuAtoms <= 0) return null;

  const totalAtoms = nAsuAtoms * nTransforms;
  const title = targetRef.cell.obj?.label || structure.label || 'Imported Mol* Symmetry Assembly';
  const titleBytes = new TextEncoder().encode(title);
  const modeStr = options?.mode ?? 'metal_verlet';
  const modeBytes = new TextEncoder().encode(modeStr);

  let cellBoxAng: [number, number, number] | undefined;
  for (const u of asuUnits) {
    if (u.model.sourceData) {
      const sym = ModelSymmetry.Provider.get(u.model);
      const cSize = sym?.spacegroup.cell.size;
      if (cSize && cSize[0] > 0) {
        cellBoxAng = [cSize[0], cSize[1], cSize[2]];
        break;
      }
    }
  }

  const headerSize = 48; // strictly 4-byte & 8-byte aligned
  const coordsBytes = nAsuAtoms * 3 * 4;
  const transformsBytes = nTransforms * 16 * 4;
  const elemBytes = nAsuAtoms;
  const totalBufferSize =
    headerSize + coordsBytes + transformsBytes + elemBytes + titleBytes.length + modeBytes.length;

  const ab = new ArrayBuffer(totalBufferSize);
  const view = new DataView(ab);
  const u8View = new Uint8Array(ab);

  // 48-byte header `<4sIIIffIIfffI`
  view.setUint8(0, 0x54); // 'T'
  view.setUint8(1, 0x53); // 'S'
  view.setUint8(2, 0x59); // 'Y'
  view.setUint8(3, 0x4d); // 'M'
  view.setUint32(4, nAsuAtoms, true);
  view.setUint32(8, nTransforms, true);
  view.setUint32(12, titleBytes.length, true);
  view.setFloat32(16, options?.temperature ?? 300.0, true);
  view.setFloat32(20, options?.dt ?? 0.001, true);
  view.setUint32(24, modeBytes.length, true);
  view.setUint32(28, 0x01, true); // flags: has_elements = 1
  view.setFloat32(32, cellBoxAng ? cellBoxAng[0] : 0.0, true);
  view.setFloat32(36, cellBoxAng ? cellBoxAng[1] : 0.0, true);
  view.setFloat32(40, cellBoxAng ? cellBoxAng[2] : 0.0, true);
  view.setUint32(44, 0, true); // reserved = 0

  // Byte offset 48: Float32Array ASU coordinates (3 * n_asu_atoms)
  const coordsOffset = headerSize;
  const float32Coords = new Float32Array(ab, coordsOffset, nAsuAtoms * 3);

  // Immediately after coords: Float32Array 4x4 row-major transforms (16 * n_transforms)
  const transformsOffset = coordsOffset + coordsBytes;
  const float32Transforms = new Float32Array(ab, transformsOffset, nTransforms * 16);

  // After transforms: Uint8Array elements (n_asu_atoms)
  const elemOffset = transformsOffset + transformsBytes;

  // After elements: UTF-8 title and mode
  let strOffset = elemOffset + elemBytes;
  u8View.set(titleBytes, strOffset);
  strOffset += titleBytes.length;
  u8View.set(modeBytes, strOffset);

  // Populate ASU coordinates (untransformed invariant positions) and element symbols
  const tmpVec = Vec3();
  let atomIdx = 0;
  for (const unit of asuUnits) {
    const { elements: els, conformation, model } = unit;
    const type_symbol = model.atomicHierarchy.atoms.type_symbol;
    const hasInvariant = typeof conformation.invariantPosition === 'function';
    const xCol = model.atomicConformation.x;
    const yCol = model.atomicConformation.y;
    const zCol = model.atomicConformation.z;

    for (let uIdx = 0, len = els.length; uIdx < len; ++uIdx) {
      const el = els[uIdx];
      const cBase = atomIdx * 3;
      if (hasInvariant) {
        conformation.invariantPosition(el, tmpVec);
        float32Coords[cBase + 0] = tmpVec[0];
        float32Coords[cBase + 1] = tmpVec[1];
        float32Coords[cBase + 2] = tmpVec[2];
      } else {
        float32Coords[cBase + 0] = xCol[el];
        float32Coords[cBase + 1] = yCol[el];
        float32Coords[cBase + 2] = zCol[el];
      }
      const sym = type_symbol.value(el);
      u8View[elemOffset + atomIdx] = sym ? sym.charCodeAt(0) : 67; // 'C'
      atomIdx++;
    }
  }

  // Populate 4x4 row-major transforms (transposing Mol*'s column-major Mat4)
  for (let t = 0; t < nTransforms; t++) {
    const m = transforms[t];
    const o = t * 16;
    // Row 0
    float32Transforms[o + 0] = m[0];
    float32Transforms[o + 1] = m[4];
    float32Transforms[o + 2] = m[8];
    float32Transforms[o + 3] = m[12];
    // Row 1
    float32Transforms[o + 4] = m[1];
    float32Transforms[o + 5] = m[5];
    float32Transforms[o + 6] = m[9];
    float32Transforms[o + 7] = m[13];
    // Row 2
    float32Transforms[o + 8] = m[2];
    float32Transforms[o + 9] = m[6];
    float32Transforms[o + 10] = m[10];
    float32Transforms[o + 11] = m[14];
    // Row 3
    float32Transforms[o + 12] = m[3];
    float32Transforms[o + 13] = m[7];
    float32Transforms[o + 14] = m[11];
    float32Transforms[o + 15] = m[15];
  }

  return {
    title,
    nAsuAtoms,
    nTransforms,
    totalAtoms,
    buffer: ab,
    box_ang: cellBoxAng,
  };
}
