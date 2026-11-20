/**
 * Mol* v6 PluginBehavior & Stream Controller for Biology-MM live simulation streaming.
 *
 * Mounts and updates a zero-allocation `ParticleList` + `ParticlesRepresentation3D` on
 * `PluginContext`, throttles rendering via `requestAnimationFrame` (dropping stale frames),
 * integrates `PhysicsOverlayManager`, `TernaryVolumeBridge`, and physics color themes,
 * and exposes RxJS observables (`frameStats$`, `atomPicked$`, `activeChannel$`, `activePalette$`).
 */

import { OrderedSet } from '@molstar/core/data/int';
import { StateTransformer } from '@molstar/core/state';
import { ParamDefinition as PD } from '@molstar/core/util/param-definition';
import { SpacefillParticlesParams, SpacefillParticlesRepresentationProvider } from '@molstar/graphics/repr/particles/representation/spacefill';
import { ParticleSizeThemeProvider } from '@molstar/graphics/theme/size/particle-size';
import { Particle, type ParticleList } from '@molstar/model/model/particles/particle-list';
import { StructureElement, Unit } from '@molstar/model/model/structure';
import { PluginBehavior } from '@molstar/plugin/behavior/behavior';
import type { PluginContext } from '@molstar/plugin/context';
import { PluginStateObject as SO, PluginStateTransform } from '@molstar/plugin/state/objects';
import { ParticlesRepresentation3D } from '@molstar/plugin/state/transforms/particles/representation';
import { BehaviorSubject, Subject, type Observable, type Subscription } from 'rxjs';
import type { PaletteName } from './color-palettes.js';
import { GpuParticleStreamUpdater } from './gpu-buffer-updater.js';
import {
  registerPhysicsColorThemes,
  PHYSICS_CHANNEL_TO_THEME,
  type BiologyMmStructureScalarEntry,
  type BiologyMmStructureScalarMap,
} from './physics-color-themes.js';
import { PhysicsOverlayManager, type OverlayVisibility } from './physics-overlays.js';
import {
  decodeTriumMessage,
  MsgType,
  parseHeader,
  PHYSICS_CHANNELS,
  type OverlayVectorsMessage,
  type PhysicsChannelName,
  type VolumeGrid3DMessage,
} from './protocol.js';
import { BiologyMmStructureBridge } from './structure-bridge.js';
import { TernaryVolumeBridge } from './volume-bridge.js';

export interface StreamFrameStats {
  readonly frameId: number;
  readonly simTimePs: number;
  readonly nAtoms: number;
  readonly activeCount: number;
  readonly droppedFrames: number;
  readonly fps: number;
  readonly scalarMin: number;
  readonly scalarMax: number;
  readonly reversed: boolean;
  readonly activeChannel: PhysicsChannelName;
  readonly activePalette: PaletteName;
  readonly lastDirtyAtomCount: number;
  readonly lastPacketBytes: number;
}

export interface AtomPickedEvent {
  readonly index: number;
  readonly atomicNumber: number;
  readonly coords: [number, number, number];
  readonly scalarValue: number | undefined;
  readonly channel: PhysicsChannelName;
}

export interface StreamControllerOptions {
  readonly capacity?: number;
  readonly initialChannel?: PhysicsChannelName;
  readonly initialPalette?: PaletteName;
  readonly scheduleFrame?: (cb: () => void) => number;
  readonly cancelFrame?: (id: number) => void;
}

const BiologyMmParticleListTransform = PluginStateTransform.BuiltIn({
  name: 'biology-mm-particle-list',
  display: { name: 'Biology-MM Particle List', description: 'Live streamed particle buffer' },
  from: SO.Root,
  to: SO.Particle.List,
  params: {
    list: PD.Value<ParticleList>(undefined as any, { isHidden: true }),
  },
})({
  canAutoUpdate() {
    return true;
  },
  apply({ params }) {
    return new SO.Particle.List(params.list, {
      label: params.list.label || 'Biology-MM Live Stream',
      description: `${params.list.count} particles`,
    });
  },
  update({ b, newParams }) {
    b.data = newParams.list;
    b.label = newParams.list.label || 'Biology-MM Live Stream';
    b.description = `${newParams.list.count} particles`;
    return StateTransformer.UpdateResult.Updated;
  },
});

const controllerByPlugin = new WeakMap<PluginContext, BiologyMmStreamController>();

export class BiologyMmStreamController {
  static readonly NATIVE_STRUCTURE_ATOM_LIMIT = 25_000;
  static readonly COMPACT_SCAFFOLD_ATOM_THRESHOLD = 250_000;

  readonly plugin: PluginContext;
  readonly updater: GpuParticleStreamUpdater;
  readonly overlayManager: PhysicsOverlayManager;
  readonly volumeBridge: TernaryVolumeBridge;
  readonly structureBridge: BiologyMmStructureBridge;

  private readonly _frameStats$: BehaviorSubject<StreamFrameStats>;
  private readonly _atomPicked$ = new Subject<AtomPickedEvent>();
  private readonly _activeChannel$: BehaviorSubject<PhysicsChannelName>;
  private readonly _activePalette$: BehaviorSubject<PaletteName>;

  readonly frameStats$: Observable<StreamFrameStats>;
  readonly atomPicked$: Observable<AtomPickedEvent>;
  readonly activeChannel$: Observable<PhysicsChannelName>;
  readonly activePalette$: Observable<PaletteName>;

  private readonly scheduleFrame: (cb: () => void) => number;
  private readonly cancelFrame: (id: number) => void;

  private unregisterThemes: (() => void) | undefined;
  private clickSub: Subscription | undefined;

  private listRef: string | undefined;
  private reprRef: string | undefined;

  private useNativeStructure = false;
  private preserveExternalScaffold = false;
  private defaultSpacefillParams: ReturnType<typeof PD.getDefaultValues<typeof SpacefillParticlesParams>> | undefined;
  private structureColorMode: 'default' | PhysicsChannelName = 'default';
  private desiredOverlays: OverlayVisibility = {
    showBox: true,
    showBonds: true,
    showForces: false,
  };

  private rafId: number | undefined;
  private pendingRender = false;
  private needsFrameCommit = false;
  private needsTopologyCommit = false;
  private needsThemeCommit = false;
  private isCommitting = false;

  private pendingOverlayMsg: OverlayVectorsMessage | undefined;
  private pendingVolumeMsg: VolumeGrid3DMessage | undefined;

  private droppedFrames = 0;
  private lastDirtyAtomCount = 0;
  private lastPacketBytes = 0;
  private lastRenderTimeMs = 0;
  private smoothedFps = 0;
  private disposed = false;

  constructor(plugin: PluginContext, options: StreamControllerOptions = {}) {
    this.plugin = plugin;
    this.updater = new GpuParticleStreamUpdater(options.capacity ?? 1_000_000);
    this.overlayManager = new PhysicsOverlayManager(plugin);
    this.volumeBridge = new TernaryVolumeBridge(plugin);
    this.structureBridge = new BiologyMmStructureBridge(plugin);

    const initialChannel = options.initialChannel ?? 'force-magnitude';
    const initialPalette = options.initialPalette ?? 'plasma';
    this.updater.setActiveChannel(initialChannel);

    this._activeChannel$ = new BehaviorSubject<PhysicsChannelName>(initialChannel);
    this._activePalette$ = new BehaviorSubject<PaletteName>(initialPalette);
    this._frameStats$ = new BehaviorSubject<StreamFrameStats>({
      frameId: 0,
      simTimePs: 0,
      nAtoms: 0,
      activeCount: 0,
      droppedFrames: 0,
      fps: 0,
      scalarMin: 0,
      scalarMax: 1,
      reversed: false,
      activeChannel: initialChannel,
      activePalette: initialPalette,
      lastDirtyAtomCount: 0,
      lastPacketBytes: 0,
    });

    this.frameStats$ = this._frameStats$.asObservable();
    this.atomPicked$ = this._atomPicked$.asObservable();
    this.activeChannel$ = this._activeChannel$.asObservable();
    this.activePalette$ = this._activePalette$.asObservable();

    this.scheduleFrame =
      options.scheduleFrame ??
      ((cb) =>
        typeof globalThis.requestAnimationFrame === 'function'
          ? globalThis.requestAnimationFrame(cb)
          : (globalThis.setTimeout(cb, 16) as unknown as number));

    this.cancelFrame =
      options.cancelFrame ??
      ((id) => {
        if (typeof globalThis.cancelAnimationFrame === 'function') {
          globalThis.cancelAnimationFrame(id);
        } else {
          globalThis.clearTimeout(id);
        }
      });

    this.ensurePluginRegistrations();
    this.subscribeInteraction();
  }

  private ensurePluginRegistrations(): void {
    const particleScope = this.plugin.representation?.particles;
    if (!particleScope) return;

    if (!this.unregisterThemes) {
      this.unregisterThemes = registerPhysicsColorThemes(this.plugin);
    }

    if (!particleScope.registry.has(SpacefillParticlesRepresentationProvider)) {
      particleScope.registry.add(SpacefillParticlesRepresentationProvider);
    }
    if (!particleScope.themes.sizeThemeRegistry.has(ParticleSizeThemeProvider)) {
      particleScope.themes.sizeThemeRegistry.add(ParticleSizeThemeProvider);
    }
  }

  private subscribeInteraction(): void {
    const click$ = this.plugin.behaviors?.interaction?.click;
    if (!click$) return;

    this.clickSub = click$.subscribe((ev) => {
      const loci = ev.current?.loci;
      if (!loci) return;

      let index = -1;
      const rawLoci: unknown = loci;
      if (Particle.isLocation(rawLoci)) {
        index = rawLoci.index;
      } else if (Particle.isLoci(loci) && !Particle.isLociEmpty(loci)) {
        index = OrderedSet.getAt(loci.indices, 0);
      } else if (StructureElement.Loci.is(loci) && !StructureElement.Loci.isEmpty(loci)) {
        const loc = StructureElement.Loci.getFirstLocation(loci);
        if (loc && Unit.isAtomic(loc.unit)) {
          const srcIdx = loc.unit.model.atomicHierarchy.atomSourceIndex.value(loc.element);
          index = srcIdx >= 0 && srcIdx < this.updater.count ? srcIdx : loc.element;
        }
      }

      if (index < 0 || index >= this.updater.count) return;
      const c3 = index * 3;
      const channel = this._activeChannel$.value;
      const scalarBuf = this.updater.scalarChannels.get(channel);

      this._atomPicked$.next({
        index,
        atomicNumber: this.updater.slotPool.atomicNumbers[index] ?? 0,
        coords: [this.updater.coords[c3], this.updater.coords[c3 + 1], this.updater.coords[c3 + 2]],
        scalarValue: scalarBuf ? scalarBuf[index] : undefined,
        channel,
      });
    });
  }

  ingestPacket(buffer: ArrayBuffer): void {
    if (this.disposed) return;

    const header = parseHeader(new DataView(buffer), 0);

    switch (header.msgType) {
      case MsgType.TOPOLOGY_INIT: {
        const msg = decodeTriumMessage(buffer);
        if (msg.kind === 'topology') {
          this.updater.applyTopologyInit(msg);
          this.lastDirtyAtomCount = msg.header.nAtoms;
          this.lastPacketBytes = buffer.byteLength;
          this.needsTopologyCommit = true;
          this.needsFrameCommit = true;
          if (!this.useNativeStructure) {
            void this.overlayManager.updateTopology(msg, this.updater.coords);
          }
          this.requestRender();
        }
        break;
      }

      case MsgType.FRAME_FULL: {
        if (this.pendingRender && this.needsFrameCommit) {
          this.droppedFrames++;
        }
        this.updater.applyFrameFullBinary(buffer, header);
        this.lastDirtyAtomCount = this.updater.lastDirtyAtomCount;
        this.lastPacketBytes = buffer.byteLength;
        this.needsFrameCommit = true;
        this.requestRender();
        break;
      }

      case MsgType.FRAME_TILE_PATCH: {
        if (this.pendingRender && this.needsFrameCommit) {
          this.droppedFrames++;
        }
        this.updater.applyTilePatchBinary(buffer, header);
        this.lastDirtyAtomCount = this.updater.lastDirtyAtomCount;
        this.lastPacketBytes = buffer.byteLength;
        this.needsFrameCommit = true;
        this.requestRender();
        break;
      }

      case MsgType.OVERLAY_VECTORS: {
        const msg = decodeTriumMessage(buffer);
        if (msg.kind === 'overlays') {
          this.pendingOverlayMsg = msg;
          this.requestRender();
        }
        break;
      }

      case MsgType.VOLUME_GRID3D: {
        const msg = decodeTriumMessage(buffer);
        if (msg.kind === 'volume-grid3d') {
          this.pendingVolumeMsg = msg;
          this.requestRender();
        }
        break;
      }

      case MsgType.GCMC_SLOT_DELTA: {
        const msg = decodeTriumMessage(buffer);
        if (msg.kind === 'gcmc-delta') {
          this.updater.applyGcmcDelta(msg);
          this.needsTopologyCommit = true;
          this.needsFrameCommit = true;
          this.requestRender();
        }
        break;
      }
    }
  }

  async activateSessionStructure(
    cifText: string,
    label: string,
    nAtoms: number,
    defaultColorMode: 'default' | PhysicsChannelName = 'default',
    options?: { readonly keepExternalVisible?: boolean; readonly storageMode?: string },
  ): Promise<void> {
    if (this.disposed) return;

    const preserveExternal =
      Boolean(options?.keepExternalVisible) ||
      options?.storageMode === 'symmetry-compact' ||
      nAtoms > BiologyMmStreamController.COMPACT_SCAFFOLD_ATOM_THRESHOLD;
    this.preserveExternalScaffold = preserveExternal;

    // Hide external static Mol* structures for standard sessions so they do not overlap with
    // the live simulation stream, but preserve external assemblies (e.g. 40M viral capsids)
    // as rigid-scaffold visual context during symmetry-compact / >250k multiscale sessions.
    if (!preserveExternal) {
      const hierarchy = this.plugin.managers?.structure?.hierarchy;
      if (hierarchy) {
        const currentStructures = hierarchy.current.structures;
        const external = currentStructures.filter((s) => s.cell.transform.ref !== this.structureBridge.liveStructureRef);
        if (external.length > 0) {
          const isBusy = this.plugin.state.data.behaviors.isUpdating.value || Boolean((this.plugin.state.data as any).inTransaction);
          if (!isBusy) {
            hierarchy.toggleVisibility(external, 'hide');
          } else {
            queueMicrotask(() => {
              const stillBusy = this.plugin.state.data.behaviors.isUpdating.value || Boolean((this.plugin.state.data as any).inTransaction);
              if (!this.disposed && !stillBusy) {
                hierarchy.toggleVisibility(external, 'hide');
              }
            });
          }
        }
      }
    }

    this.structureColorMode = defaultColorMode;
    if (defaultColorMode !== 'default') {
      this.updater.setActiveChannel(defaultColorMode);
      if (this._activeChannel$.value !== defaultColorMode) {
        this._activeChannel$.next(defaultColorMode);
      }
    }

    if (nAtoms > 0 && nAtoms <= BiologyMmStreamController.NATIVE_STRUCTURE_ATOM_LIMIT) {
      this.useNativeStructure = true;
      await this.clearParticleNodes();
      await this.overlayManager.setVisibility({
        showBox: false,
        showBonds: false,
        showForces: this.desiredOverlays.showForces,
      });
      this.structureBridge.setPendingAppearance(
        defaultColorMode,
        this._activePalette$.value,
        this.desiredOverlays.showBox,
      );
      const loaded = await this.structureBridge.loadSessionStructure(
        cifText,
        label,
        this.buildScalarSnapshot(),
      );
      if (loaded) {
        if (this.updater.count > 0) {
          this.needsFrameCommit = true;
          this.requestRender();
        }
        return;
      }
      // Fall back to GPU particle stream if structure loading failed
    }

    this.useNativeStructure = false;
    await this.structureBridge.dispose();
    await this.overlayManager.setVisibility(this.desiredOverlays);
    if (this.updater.count > 0) {
      this.needsTopologyCommit = true;
      this.needsFrameCommit = true;
      this.requestRender();
    }
  }

  async setStructureColorMode(mode: 'default' | PhysicsChannelName): Promise<void> {
    this.structureColorMode = mode;
    if (mode !== 'default') {
      this.updater.setActiveChannel(mode);
      if (this._activeChannel$.value !== mode) {
        this._activeChannel$.next(mode);
      }
    }

    if (this.useNativeStructure) {
      await this.structureBridge.setColorMode(mode, this._activePalette$.value);
    } else if (mode !== 'default') {
      this.needsThemeCommit = true;
      this.requestRender();
    }
  }

  setChannel(channel: PhysicsChannelName): void {
    if (this._activeChannel$.value === channel && this.structureColorMode === channel) return;
    this.structureColorMode = channel;
    this.updater.setActiveChannel(channel);
    if (this._activeChannel$.value !== channel) {
      this._activeChannel$.next(channel);
    }
    if (this.useNativeStructure) {
      void this.structureBridge.setColorMode(channel, this._activePalette$.value);
    } else {
      this.needsThemeCommit = true;
      this.requestRender();
    }
  }

  setPalette(palette: PaletteName): void {
    if (this._activePalette$.value === palette) return;
    this._activePalette$.next(palette);
    if (this.useNativeStructure) {
      void this.structureBridge.setColorMode(this.structureColorMode, palette);
    } else {
      this.needsThemeCommit = true;
      this.requestRender();
    }
  }

  async setOverlayVisibility(vis: Partial<OverlayVisibility>): Promise<void> {
    this.desiredOverlays = { ...this.desiredOverlays, ...vis };
    if (this.useNativeStructure) {
      if (vis.showBox !== undefined) {
        this.structureBridge.setUnitCellVisible(vis.showBox);
      }
      await this.overlayManager.setVisibility({
        showBox: false,
        showBonds: false,
        showForces: this.desiredOverlays.showForces,
      });
    } else {
      await this.overlayManager.setVisibility(this.desiredOverlays);
    }
  }

  setBoxOrigin(origin: [number, number, number]): void {
    if (this.disposed) return;
    void this.overlayManager.setBoxOrigin(origin);
  }

  setVolumeVisible(visible: boolean): void {
    this.volumeBridge.setVisible(visible);
  }

  resetCamera(): void {
    this.plugin.canvas3d?.requestCameraReset();
  }

  private buildScalarSnapshot(): BiologyMmStructureScalarMap {
    const map = new Map<PhysicsChannelName, BiologyMmStructureScalarEntry>();
    const count = this.updater.count;
    if (count <= 0) return map;

    for (const ch of PHYSICS_CHANNELS) {
      const buf = this.updater.scalarChannels.get(ch);
      if (!buf || buf.length < count) continue;
      const range = this.updater.getScalarRange(ch);
      map.set(ch, {
        values: this.updater.getScalarStagingView(ch),
        min: range.min,
        max: range.max,
      });
    }
    return map;
  }

  private async clearParticleNodes(): Promise<void> {
    const state = this.plugin.state?.data;
    const ref = this.listRef;
    this.listRef = undefined;
    this.reprRef = undefined;
    if (ref && state && state.cells.has(ref)) {
      await state.build().delete(ref).commit();
    }
  }

  private requestRender(): void {
    if (this.disposed || this.pendingRender) return;
    this.pendingRender = true;
    this.rafId = this.scheduleFrame(() => {
      void this.flushRender();
    });
  }

  async flushRender(): Promise<void> {
    this.rafId = undefined;
    this.pendingRender = false;
    if (this.disposed) return;

    if (this.isCommitting) {
      this.requestRender();
      return;
    }

    this.isCommitting = true;
    try {
      const channel = this._activeChannel$.value;
      const palette = this._activePalette$.value;

      if ((this.needsFrameCommit || this.needsTopologyCommit || this.needsThemeCommit) && this.updater.count > 0) {
        const wasTopology = this.needsTopologyCommit;
        const wasTheme = this.needsThemeCommit;
        const wasScalarsDirty = this.updater.scalarsDirty;
        const forceNew = this.needsTopologyCommit;
        this.needsFrameCommit = false;
        this.needsTopologyCommit = false;
        this.needsThemeCommit = false;
        const dirtyRange = this.updater.consumeDirtyRange();

        if (this.useNativeStructure) {
          if (this.structureBridge.isLoaded && (dirtyRange.isDirty || wasTopology || wasTheme)) {
            await this.structureBridge.updateLiveFrame(
              this.updater.coords,
              this.updater.count,
              this.updater.boxH,
              this.buildScalarSnapshot(),
              this.updater.simTimePs,
            );
          }
        } else if (dirtyRange.isDirty || wasTopology || wasTheme || !this.listRef) {
          const list = this.updater.getParticleList(forceNew);
          await this.commitParticleState(list, channel, palette, wasTopology || wasTheme || wasScalarsDirty);
          await this.overlayManager.updateCoords(this.updater.coords, this.updater.boxH);
          if (wasTopology && !this.preserveExternalScaffold) {
            this.plugin.canvas3d?.requestCameraReset();
          }
        }
      }

      if (this.pendingOverlayMsg) {
        const msg = this.pendingOverlayMsg;
        this.pendingOverlayMsg = undefined;
        await this.overlayManager.updateOverlays(msg, this.updater.coords, this.updater.boxH);
      }

      if (this.pendingVolumeMsg) {
        const msg = this.pendingVolumeMsg;
        this.pendingVolumeMsg = undefined;
        await this.volumeBridge.updateFromMessage(msg);
      }

      const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
      if (this.lastRenderTimeMs > 0) {
        const dtMs = Math.max(1, now - this.lastRenderTimeMs);
        const instantFps = 1000 / dtMs;
        this.smoothedFps = this.smoothedFps === 0 ? instantFps : this.smoothedFps * 0.8 + instantFps * 0.2;
      }
      this.lastRenderTimeMs = now;

      const range = this.updater.getScalarRange(channel);
      this._frameStats$.next({
        frameId: this.updater.frameId,
        simTimePs: this.updater.simTimePs,
        nAtoms: this.updater.count,
        activeCount: this.updater.activeCount,
        droppedFrames: this.droppedFrames,
        fps: Math.round(this.smoothedFps * 10) / 10,
        scalarMin: range.min,
        scalarMax: range.max,
        reversed: this.updater.reversed,
        activeChannel: channel,
        activePalette: palette,
        lastDirtyAtomCount: this.lastDirtyAtomCount,
        lastPacketBytes: this.lastPacketBytes,
      });
    } finally {
      this.isCommitting = false;
      if (
        !this.disposed &&
        (this.needsFrameCommit ||
          this.needsTopologyCommit ||
          this.needsThemeCommit ||
          this.pendingOverlayMsg ||
          this.pendingVolumeMsg)
      ) {
        this.requestRender();
      }
    }
  }

  private async commitParticleState(
    list: ParticleList,
    channel: PhysicsChannelName,
    palette: PaletteName,
    updateRepr: boolean,
  ): Promise<void> {
    const state = this.plugin.state?.data;
    if (!state) return;

    const existingList = this.listRef ? state.cells.get(this.listRef) : undefined;
    if (!existingList) {
      if (!this.defaultSpacefillParams) {
        this.defaultSpacefillParams = PD.getDefaultValues(SpacefillParticlesParams);
      }
      const themeName = PHYSICS_CHANNEL_TO_THEME[channel];
      const reprParams = {
        type: { name: 'spacefill', params: this.defaultSpacefillParams },
        colorTheme: {
          name: themeName,
          params: {
            palette,
            domain: { name: 'auto', params: {} },
          },
        },
        sizeTheme: {
          name: 'particle-size',
          params: { scale: 1.0, defaultSize: 0 },
        },
      };
      const builder = state.build();
      const listNode = builder.toRoot().apply(BiologyMmParticleListTransform, { list });
      const reprNode = listNode.apply(ParticlesRepresentation3D, reprParams);
      await builder.commit();
      this.listRef = listNode.ref;
      this.reprRef = reprNode.ref;
    } else {
      const builder = state.build();
      builder.to(this.listRef!).update({ list });
      if (updateRepr && this.reprRef && state.cells.get(this.reprRef)) {
        if (!this.defaultSpacefillParams) {
          this.defaultSpacefillParams = PD.getDefaultValues(SpacefillParticlesParams);
        }
        const themeName = PHYSICS_CHANNEL_TO_THEME[channel];
        builder.to(this.reprRef).update({
          type: { name: 'spacefill', params: this.defaultSpacefillParams },
          colorTheme: {
            name: themeName,
            params: {
              palette,
              domain: { name: 'auto', params: {} },
            },
          },
          sizeTheme: {
            name: 'particle-size',
            params: { scale: 1.0, defaultSize: 0 },
          },
        });
      }
      await builder.commit();
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;

    if (this.rafId !== undefined) {
      this.cancelFrame(this.rafId);
      this.rafId = undefined;
    }
    this.pendingRender = false;

    this.clickSub?.unsubscribe();
    this.clickSub = undefined;

    this.unregisterThemes?.();
    this.unregisterThemes = undefined;

    this.structureBridge.destroy();
    void this.overlayManager.dispose();
    void this.volumeBridge.dispose();

    this._frameStats$.complete();
    this._atomPicked$.complete();
    this._activeChannel$.complete();
    this._activePalette$.complete();
  }
}

export const BiologyMmLiveBehavior = PluginBehavior.create<{}>({
  name: 'biology-mm-live-stream',
  category: 'misc',
  display: {
    name: 'Biology-MM Live Stream',
    description: 'Real-time GPU particle streaming, physics colormaps, and volumetric fields',
  },
  ctor: class implements PluginBehavior<{}> {
    private controller: BiologyMmStreamController | undefined;

    constructor(private readonly ctx: PluginContext) {}

    register(): void {
      if (!controllerByPlugin.has(this.ctx)) {
        this.controller = new BiologyMmStreamController(this.ctx);
        controllerByPlugin.set(this.ctx, this.controller);
      } else {
        this.controller = controllerByPlugin.get(this.ctx);
      }
    }

    unregister(): void {
      if (this.controller) {
        this.controller.dispose();
        controllerByPlugin.delete(this.ctx);
        this.controller = undefined;
      }
    }

    dispose(): void {
      this.unregister();
    }
  },
});

export function getBiologyMmStreamController(plugin: PluginContext): BiologyMmStreamController | undefined {
  return controllerByPlugin.get(plugin);
}
