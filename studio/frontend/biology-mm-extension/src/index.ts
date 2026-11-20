/**
 * Public entry point for `@molstar/biology-mm-extension`.
 * Explicit named exports only (no `export *` cycles, `verbatimModuleSyntax: true`, `.js` relative specifiers).
 */

export {
  TRIUM_MAGIC,
  HEADER_BYTES,
  MsgType,
  FrameFlags,
  PHYSICS_CHANNELS,
  parseHeader,
  decodeInt16Fractional,
  decodeTriumMessage,
} from './protocol.js';
export type {
  PhysicsChannelName,
  TriumHeader,
  TopologyInitMessage,
  FrameFullMessage,
  FrameTilePatchMessage,
  OverlayVectorsMessage,
  VolumeGrid3DMessage,
  GcmcSlotDeltaMessage,
  TriumMessage,
} from './protocol.js';

export { GcmcSlotPool, defaultRadiusForAtomicNumber } from './gcmc-slot-pool.js';
export type { GcmcDeltaOptions, GcmcDeltaResult } from './gcmc-slot-pool.js';

export { GpuParticleStreamUpdater } from './gpu-buffer-updater.js';
export type { ScalarRange } from './gpu-buffer-updater.js';

export {
  BiologyMmLiveBehavior,
  BiologyMmStreamController,
  getBiologyMmStreamController,
} from './live-stream-behavior.js';
export type {
  StreamFrameStats,
  AtomPickedEvent,
  StreamControllerOptions,
} from './live-stream-behavior.js';

export {
  PALETTE_NAMES,
  DEFAULT_FALLBACK_COLOR,
  samplePaletteColor,
  fillRgbaFromScalars,
  getPaletteStopsHex,
  getPaletteStopsColor,
  getPaletteLutRgba,
} from './color-palettes.js';
export type { PaletteName } from './color-palettes.js';

export {
  ForceMagnitudeThemeProvider,
  KineticTempThemeProvider,
  ChargePotentialThemeProvider,
  BondOrderThemeProvider,
  AtomicStressThemeProvider,
  TernaryStateThemeProvider,
  PHYSICS_THEME_PROVIDERS,
  PHYSICS_CHANNEL_TO_THEME,
  BIOLOGY_MM_SCALARS_KEY,
  StructureForceMagnitudeThemeProvider,
  StructureKineticTempThemeProvider,
  StructureChargePotentialThemeProvider,
  StructureBondOrderThemeProvider,
  StructureAtomicStressThemeProvider,
  StructureTernaryStateThemeProvider,
  STRUCTURE_PHYSICS_THEME_PROVIDERS,
  PHYSICS_CHANNEL_TO_STRUCT_THEME,
  createPhysicsColorThemeParams,
  registerPhysicsColorThemes,
} from './physics-color-themes.js';
export type {
  PhysicsThemeName,
  StructurePhysicsThemeName,
  BiologyMmStructureScalarEntry,
  BiologyMmStructureScalarMap,
  PhysicsColorThemeParams,
  PhysicsThemePluginTarget,
} from './physics-color-themes.js';

export {
  BiologyMmLiveTrajectoryTransform,
  BiologyMmStructureBridge,
  extractActiveMolstarStructure,
  extractActiveMolstarStructureBinary,
} from './structure-bridge.js';
export type { ExtractedEntityMeta, ExtractedMolstarStructure, ExtractedMolstarBinaryStructure } from './structure-bridge.js';

export {
  BiologyMmShapeProvider,
  PhysicsOverlayManager,
  computeBoxWireframeSegments,
  buildBoxWireframeShape,
  bondOrderToColor,
  bondOrderToRadius,
  buildBondsShape,
  buildForceVectorsShape,
} from './physics-overlays.js';
export type { OverlayVisibility, ForceVectorBuildOptions } from './physics-overlays.js';

export {
  BiologyMmVolumeData,
  TernaryVolumeBridge,
  computeVolumeGridMatrix,
  createVolumeFromGridMessage,
} from './volume-bridge.js';
