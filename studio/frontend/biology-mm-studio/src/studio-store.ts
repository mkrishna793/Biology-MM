import type { PluginContext } from '@molstar/plugin/context';
import { PluginCommands } from '@molstar/plugin/commands';
import {
  BiologyMmStreamController,
  extractActiveMolstarStructure,
  extractActiveMolstarStructureBinary,
  extractActiveMolstarSymmetryBinary,
  getBiologyMmStreamController,
} from '@molstar/biology-mm-extension';
import type { Subscription } from 'rxjs';
import type {
  AtomInspection,
  CodeRunResult,
  CustomPhysicsSpec,
  FrameStats,
  MultiscaleZoneConfig,
  PaletteName,
  PhysicsChannelName,
  PresetInfo,
  RelaxationReport,
  SimulationMetrics,
  SimulationMode,
  SteerAtomState,
  StorageMode,
  SteerRequest,
  StudioCapabilities,
} from './types.js';

export type StudioColorMode = 'default' | PhysicsChannelName;

export interface StudioOverlaysState {
  showBox: boolean;
  showBonds: boolean;
  showForces: boolean;
  showVolume: boolean;
}

export interface StudioStoreState {
  capabilities: StudioCapabilities | null;
  presets: PresetInfo[];
  activePresetId: string | null;
  sessionId: string | null;
  sessionTitle: string;
  isPlaying: boolean;
  isLoading: boolean;
  isBusy: boolean;
  compilingPhysics: boolean;
  codeRunning: boolean;
  error: string | null;
  metrics: SimulationMetrics | null;
  metricHistory: SimulationMetrics[];
  frameStats: FrameStats;
  pickedAtom: AtomInspection | null;
  colorMode: StudioColorMode;
  activeChannel: PhysicsChannelName;
  activePalette: PaletteName;
  overlays: StudioOverlaysState;
  isReversed: boolean;
  codeText: string;
  codeRunResult: CodeRunResult | null;
  wideCodeEditorOpen: boolean;
  multiscale: MultiscaleZoneConfig | null;
  relaxationReport: RelaxationReport | null;
  isRelaxing: boolean;
  steerAtomState: SteerAtomState;
  deltaThresholdAng: number;
  storageMode: StorageMode;
}

const MAX_TELEMETRY_POINTS = 300;
const MAX_PENDING_PACKETS = 32;

const INITIAL_FRAME_STATS: FrameStats = {
  fps: 0,
  frameId: 0,
  simTimePs: 0,
  nAtoms: 0,
  scalarMin: 0,
  scalarMax: 1,
  activeChannel: 'force-magnitude',
  activePalette: 'plasma',
  reversed: false,
  wsConnected: false,
  lastDirtyAtomCount: 0,
  lastPacketBytes: 0,
};

const INITIAL_STEER_ATOM_STATE: SteerAtomState = {
  active: false,
  atom_index: null,
  target_ang: null,
  k_spring: 50.0,
};

const INITIAL_OVERLAYS: StudioOverlaysState = {
  showBox: true,
  showBonds: true,
  showForces: false,
  showVolume: false,
};

const ACTIVE_STRUCTURE_SNIPPET = `# Simulate or modify the structure currently loaded in Mol*
from trium_studio.engine_core import SimulationSession

print("Loaded:", active_structure_metadata.get("title") if active_structure_metadata else "Mol* Structure")
print("Atoms:", active_structure.n_atoms, "Bonds:", len(active_structure.bonds))

session = SimulationSession(
    active_structure,
    mode="metal_verlet",
    dt=0.001,
    temperature=310.0,
)
session.step(10)
`;

function buildWsUrl(sid: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/ws/stream/${encodeURIComponent(sid)}`;
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(body || `HTTP ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export class BiologyMmStudioStore {
  readonly plugin: PluginContext;

  private state: StudioStoreState = {
    capabilities: null,
    presets: [],
    activePresetId: null,
    sessionId: null,
    sessionTitle: 'Biology-MM Studio',
    isPlaying: false,
    isLoading: false,
    isBusy: false,
    compilingPhysics: false,
    codeRunning: false,
    error: null,
    metrics: null,
    metricHistory: [],
    frameStats: { ...INITIAL_FRAME_STATS },
    pickedAtom: null,
    colorMode: 'default',
    activeChannel: 'force-magnitude',
    activePalette: 'plasma',
    overlays: { ...INITIAL_OVERLAYS },
    isReversed: false,
    codeText: '',
    codeRunResult: null,
    wideCodeEditorOpen: false,
    multiscale: null,
    relaxationReport: null,
    isRelaxing: false,
    steerAtomState: { ...INITIAL_STEER_ATOM_STATE },
    deltaThresholdAng: 0.02,
    storageMode: 'auto',
  };

  private readonly listeners = new Set<() => void>();
  private controller: BiologyMmStreamController | null = null;
  private controllerSubs: Subscription[] = [];
  private ws: WebSocket | null = null;
  private pendingPackets: ArrayBuffer[] = [];
  private bootPromise: Promise<void> | null = null;
  private disposed = false;

  constructor(plugin: PluginContext) {
    this.plugin = plugin;
  }

  getSnapshot = (): StudioStoreState => this.state;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  private setState(partial: Partial<StudioStoreState>): void {
    if (this.disposed) return;
    this.state = { ...this.state, ...partial };
    for (const listener of this.listeners) {
      listener();
    }
  }

  private resolveController(): BiologyMmStreamController | null {
    if (!this.controller) {
      const ctrl = getBiologyMmStreamController(this.plugin) ?? null;
      if (ctrl) {
        this.attachController(ctrl);
      }
    }
    return this.controller;
  }

  attachController(controller: BiologyMmStreamController | null): void {
    for (const sub of this.controllerSubs) {
      sub.unsubscribe();
    }
    this.controllerSubs = [];
    this.controller = controller;
    if (!controller || this.disposed) return;

    controller.setChannel(this.state.activeChannel);
    controller.setPalette(this.state.activePalette);
    void controller.setOverlayVisibility({
      showBox: this.state.overlays.showBox,
      showBonds: this.state.overlays.showBonds,
      showForces: this.state.overlays.showForces,
    });
    controller.setVolumeVisible(this.state.overlays.showVolume);

    const subStats = controller.frameStats$.subscribe((stats) => {
      const metrics = this.state.metrics;
      const reportedAtoms =
        metrics?.storage_mode === 'symmetry-compact' && metrics.n_atoms > stats.nAtoms
          ? metrics.n_atoms
          : stats.nAtoms;
      this.setState({
        isReversed: stats.reversed,
        frameStats: {
          ...this.state.frameStats,
          fps: stats.fps,
          frameId: stats.frameId,
          simTimePs: stats.simTimePs,
          nAtoms: reportedAtoms,
          scalarMin: stats.scalarMin,
          scalarMax: stats.scalarMax,
          activeChannel: stats.activeChannel,
          activePalette: stats.activePalette,
          reversed: stats.reversed,
          lastDirtyAtomCount: stats.lastDirtyAtomCount,
          lastPacketBytes: stats.lastPacketBytes,
        },
      });
    });

    const subPick = controller.atomPicked$.subscribe((ev) => {
      void this.inspectAtom(ev.index);
    });

    this.controllerSubs = [subStats, subPick];

    if (this.pendingPackets.length > 0) {
      const queued = this.pendingPackets;
      this.pendingPackets = [];
      for (const pkt of queued) {
        controller.ingestPacket(pkt);
      }
    }
  }

  async initializeAndBoot(): Promise<void> {
    if (this.bootPromise) return this.bootPromise;
    this.bootPromise = (async () => {
      this.attachController(getBiologyMmStreamController(this.plugin) ?? null);
      this.setState({ isLoading: true, error: null });
      try {
        const [capabilities, presetsRes] = await Promise.all([
          fetchJson<StudioCapabilities>('/api/capabilities'),
          fetchJson<PresetInfo[] | { presets: PresetInfo[] }>('/api/presets'),
        ]);
        if (this.disposed) return;
        const presets = Array.isArray(presetsRes) ? presetsRes : (presetsRes.presets ?? []);
        const firstPreset = presets[0];
        this.setState({
          capabilities,
          presets,
          codeText: this.state.codeText || firstPreset?.code_template || '',
        });

        if (firstPreset && !this.state.sessionId) {
          await this.loadPreset(firstPreset.id);
        }
      } catch (err) {
        if (!this.disposed) {
          this.setState({
            error: err instanceof Error ? err.message : 'Failed to connect to Biology-MM backend',
          });
        }
      } finally {
        if (!this.disposed) {
          this.setState({ isLoading: false });
        }
      }
    })();
    return this.bootPromise;
  }

  private pushMetrics(m: SimulationMetrics): void {
    const history = this.state.metricHistory;
    const nextHistory = [...history, m];
    if (nextHistory.length > MAX_TELEMETRY_POINTS) {
      nextHistory.splice(0, nextHistory.length - MAX_TELEMETRY_POINTS);
    }
    this.setState({
      metrics: m,
      metricHistory: nextHistory,
      multiscale: m.multiscale ?? this.state.multiscale,
      frameStats: {
        ...this.state.frameStats,
        nAtoms: m.n_atoms,
        simTimePs: m.time_ps,
      },
    });
  }

  private closeWebSocket(): void {
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        // Ignore close errors
      }
      this.ws = null;
    }
  }

  private connectWebSocket(sid: string): void {
    this.closeWebSocket();
    if (this.disposed || typeof WebSocket === 'undefined') return;

    const ws = new WebSocket(buildWsUrl(sid));
    ws.binaryType = 'arraybuffer';
    this.ws = ws;

    ws.onopen = () => {
      if (this.ws !== ws || this.disposed) return;
      this.setState({
        frameStats: { ...this.state.frameStats, wsConnected: true },
      });
    };

    ws.onclose = () => {
      if (this.ws !== ws || this.disposed) return;
      this.setState({
        frameStats: { ...this.state.frameStats, wsConnected: false },
      });
    };

    ws.onerror = () => {
      if (this.ws !== ws || this.disposed) return;
      this.setState({ error: 'WebSocket stream connection error' });
    };

    ws.onmessage = (ev: MessageEvent) => {
      if (this.ws !== ws || this.disposed) return;
      if (ev.data instanceof ArrayBuffer) {
        const ctrl = this.resolveController();
        if (ctrl) {
          ctrl.ingestPacket(ev.data);
        } else {
          this.pendingPackets.push(ev.data);
          if (this.pendingPackets.length > MAX_PENDING_PACKETS) {
            this.pendingPackets.splice(1, 1);
          }
        }
        return;
      }

      if (typeof ev.data === 'string') {
        try {
          const msg = JSON.parse(ev.data) as {
            type?: string;
            metrics?: SimulationMetrics;
            error?: string;
            message?: string;
          } & Partial<SimulationMetrics>;
          if (msg.type === 'error') {
            this.setState({ error: msg.error || msg.message || 'Simulation stream error' });
          } else if (msg.metrics) {
            this.pushMetrics(msg.metrics);
          } else if (typeof msg.step === 'number' && typeof msg.time_ps === 'number') {
            this.pushMetrics(msg as SimulationMetrics);
          }
        } catch {
          // Ignore malformed JSON frames
        }
      }
    };
  }

  private sendWsCommand(payload: Record<string, unknown>): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
    }
  }

  private async activateSessionScene(
    sid: string,
    title: string,
    nAtoms: number,
    channel: PhysicsChannelName,
    palette: PaletteName,
    colorMode: StudioColorMode,
    storageMode?: string,
  ): Promise<void> {
    const ctrl = this.resolveController();
    this.pendingPackets = [];
    this.closeWebSocket();

    const preserveExternalScaffold =
      storageMode === 'symmetry-compact' ||
      nAtoms > BiologyMmStreamController.COMPACT_SCAFFOLD_ATOM_THRESHOLD;
    const activateOpts = { keepExternalVisible: preserveExternalScaffold, storageMode };

    if (ctrl) {
      ctrl.setPalette(palette);
      if (!preserveExternalScaffold && nAtoms > 0 && nAtoms <= BiologyMmStreamController.NATIVE_STRUCTURE_ATOM_LIMIT) {
        try {
          const res = await fetch(
            `/api/sessions/${encodeURIComponent(sid)}/structure.cif?channel=${encodeURIComponent(channel)}`,
          );
          if (res.ok) {
            const cifText = await res.text();
            await ctrl.activateSessionStructure(cifText, title, nAtoms, colorMode, activateOpts);
          } else {
            await ctrl.activateSessionStructure(
              '',
              title,
              nAtoms + BiologyMmStreamController.NATIVE_STRUCTURE_ATOM_LIMIT + 1,
              colorMode,
              activateOpts,
            );
          }
        } catch {
          // Fall back silently to binary WS particle stream if CIF fetch fails
        }
      } else {
        await ctrl.activateSessionStructure('', title, nAtoms, colorMode, activateOpts);
      }
    }

    this.connectWebSocket(sid);
  }

  async loadPreset(presetId: string): Promise<void> {
    this.setState({
      isBusy: true,
      error: null,
      isPlaying: false,
      pickedAtom: null,
      isReversed: false,
    });

    try {
      const res = await fetchJson<{ session_id: string; metrics?: SimulationMetrics }>('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preset_id: presetId }),
      });
      if (this.disposed) return;

      const preset = this.state.presets.find((p) => p.id === presetId);
      const nextChannel = preset?.default_channel ?? this.state.activeChannel;
      const nextPalette = preset?.default_palette ?? this.state.activePalette;
      const nextColorMode: StudioColorMode =
        presetId === 'biomolecular-complex' ? 'default' : nextChannel;
      const nAtoms = res.metrics?.n_atoms ?? preset?.n_atoms ?? 0;
      const title = preset?.title ?? presetId;

      this.setState({
        activePresetId: presetId,
        sessionId: res.session_id,
        sessionTitle: title,
        activeChannel: nextChannel,
        activePalette: nextPalette,
        colorMode: nextColorMode,
        codeText: preset?.code_template ?? this.state.codeText,
        metrics: res.metrics ?? null,
        metricHistory: res.metrics ? [res.metrics] : [],
        multiscale: res.metrics?.multiscale ?? null,
        relaxationReport: null,
        steerAtomState: { ...INITIAL_STEER_ATOM_STATE },
        frameStats: {
          ...this.state.frameStats,
          nAtoms,
          simTimePs: res.metrics?.time_ps ?? 0,
          activeChannel: nextChannel,
          activePalette: nextPalette,
          wsConnected: false,
        },
      });

      await this.activateSessionScene(res.session_id, title, nAtoms, nextChannel, nextPalette, nextColorMode, res.metrics?.storage_mode);
    } catch (err) {
      this.setState({ error: err instanceof Error ? err.message : 'Failed to load preset' });
    } finally {
      this.setState({ isBusy: false });
    }
  }

  async simulateActiveMolstarStructure(options?: {
    mode?: SimulationMode;
    temperature?: number;
    dt?: number;
  }): Promise<boolean> {
    const ctrl = this.resolveController();
    const extractOpts = {
      mode: options?.mode ?? 'metal_verlet',
      temperature: options?.temperature ?? 300.0,
      dt: options?.dt ?? 0.001,
    };

    const symExtracted = extractActiveMolstarSymmetryBinary(
      this.plugin,
      ctrl?.structureBridge.liveStructureRef,
      extractOpts,
    );
    const useSymmetry = Boolean(
      symExtracted && symExtracted.nTransforms > 1 && symExtracted.totalAtoms > 25_000,
    );

    const binExtracted = useSymmetry
      ? null
      : extractActiveMolstarStructureBinary(
          this.plugin,
          ctrl?.structureBridge.liveStructureRef,
          extractOpts,
        );

    if (!useSymmetry && (!binExtracted || binExtracted.nAtoms === 0)) {
      this.setState({
        error: 'No structure loaded in Mol*. Use Download Structure or Open Files in the Home panel first.',
      });
      return false;
    }

    this.setState({
      isBusy: true,
      error: null,
      isPlaying: false,
      pickedAtom: null,
      isReversed: false,
      relaxationReport: null,
      steerAtomState: { ...INITIAL_STEER_ATOM_STATE },
    });

    try {
      const endpoint = useSymmetry
        ? `/api/sessions/create-from-symmetry?storage_mode=${encodeURIComponent(this.state.storageMode)}`
        : '/api/sessions/create-from-binary';
      const payloadBuffer = useSymmetry ? symExtracted!.buffer : binExtracted!.buffer;

      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: payloadBuffer,
      });
      if (!resp.ok) {
        const body = await resp.text().catch(() => '');
        throw new Error(body || `Failed to create simulation: ${resp.statusText}`);
      }
      const res = (await resp.json()) as { session_id: string; title?: string; metrics?: SimulationMetrics };
      if (this.disposed) return false;

      const fallbackTitle = useSymmetry ? symExtracted!.title : binExtracted!.title;
      const fallbackAtoms = useSymmetry ? symExtracted!.totalAtoms : binExtracted!.nAtoms;
      const title = res.title || fallbackTitle;
      const nAtoms = res.metrics?.n_atoms ?? fallbackAtoms;
      const channel = this.state.activeChannel;
      const palette = this.state.activePalette;

      this.setState({
        activePresetId: null,
        sessionId: res.session_id,
        sessionTitle: title,
        colorMode: 'default',
        metrics: res.metrics ?? null,
        metricHistory: res.metrics ? [res.metrics] : [],
        multiscale: res.metrics?.multiscale ?? null,
        frameStats: {
          ...this.state.frameStats,
          nAtoms,
          simTimePs: res.metrics?.time_ps ?? 0,
          wsConnected: false,
        },
      });

      await this.activateSessionScene(res.session_id, title, nAtoms, channel, palette, 'default', res.metrics?.storage_mode);

      if (useSymmetry && symExtracted) {
        PluginCommands.Toast.Show(this.plugin, {
          title: 'Symmetry-Native Upload',
          message: `Uploaded ${symExtracted.nAsuAtoms.toLocaleString()} ASU atoms × ${symExtracted.nTransforms} transforms (${symExtracted.totalAtoms.toLocaleString()} total atoms).`,
          timeoutMs: 4500,
          key: 'bmm-symmetry-upload',
        });
      }

      return true;
    } catch (err) {
      this.setState({
        error: err instanceof Error ? err.message : 'Failed to create simulation from active Mol* structure',
      });
      return false;
    } finally {
      this.setState({ isBusy: false });
    }
  }

  async uploadStructureFile(
    file: File,
    options?: {
      mode?: SimulationMode;
      temperature?: number;
      dt?: number;
    },
  ): Promise<boolean> {
    this.setState({
      isBusy: true,
      error: null,
      isPlaying: false,
      pickedAtom: null,
      isReversed: false,
    });

    try {
      const mode = options?.mode ?? 'metal_verlet';
      const temp = options?.temperature ?? 300.0;
      const dt = options?.dt ?? 0.002;
      const query = new URLSearchParams({
        filename: file.name,
        mode,
        temperature: String(temp),
        dt: String(dt),
      });

      const buffer = await file.arrayBuffer();
      const res = await fetchJson<{
        session_id: string;
        title?: string;
        num_atoms: number;
        num_bonds: number;
        metrics?: SimulationMetrics;
      }>(`/api/sessions/upload?${query.toString()}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: buffer,
      });

      const sid = res.session_id;
      const title = res.title ?? file.name;
      const nAtoms = res.num_atoms ?? 0;
      const channel = this.state.activeChannel;
      const palette = this.state.activePalette;

      this.setState({
        sessionId: sid,
        sessionTitle: title,
        activePresetId: null,
        colorMode: 'default',
        metrics: res.metrics ?? null,
        metricHistory: res.metrics ? [res.metrics] : [],
        multiscale: res.metrics?.multiscale ?? null,
        relaxationReport: null,
        steerAtomState: { ...INITIAL_STEER_ATOM_STATE },
        frameStats: {
          ...this.state.frameStats,
          nAtoms,
          simTimePs: res.metrics?.time_ps ?? 0,
          wsConnected: false,
        },
      });

      await this.activateSessionScene(sid, title, nAtoms, channel, palette, 'default', res.metrics?.storage_mode);
      return true;
    } catch (err) {
      this.setState({
        error: err instanceof Error ? err.message : 'Failed to upload structure file to engine',
      });
      return false;
    } finally {
      this.setState({ isBusy: false });
    }
  }

  async runCode(codeOverride?: string): Promise<CodeRunResult> {
    const code = codeOverride ?? this.state.codeText;
    if (codeOverride !== undefined) {
      this.setState({ codeText: codeOverride });
    }
    this.setState({ codeRunning: true, error: null });

    try {
      const ctrl = this.resolveController();
      const activeStructure = extractActiveMolstarStructure(
        this.plugin,
        ctrl?.structureBridge.liveStructureRef,
      );
      const res = await fetchJson<CodeRunResult>('/api/code/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code,
          active_structure: activeStructure ?? undefined,
        }),
      });
      if (this.disposed) return res;

      this.setState({ codeRunResult: res });
      if (res.ok && res.session_id) {
        const nAtoms = res.metrics?.n_atoms ?? 0;
        const nextMode: StudioColorMode =
          this.state.colorMode === 'default' ? this.state.activeChannel : this.state.colorMode;

        this.setState({
          sessionId: res.session_id,
          sessionTitle: 'Python Code Session',
          isPlaying: false,
          pickedAtom: null,
          colorMode: nextMode,
          metrics: res.metrics ?? null,
          metricHistory: res.metrics ? [res.metrics] : [],
          frameStats: {
            ...this.state.frameStats,
            nAtoms,
            simTimePs: res.metrics?.time_ps ?? 0,
            wsConnected: false,
          },
        });

        await this.activateSessionScene(
          res.session_id,
          'Python Code Session',
          nAtoms,
          this.state.activeChannel,
          this.state.activePalette,
          nextMode,
          res.metrics?.storage_mode,
        );
      }
      return res;
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Code execution failed';
      const fallback: CodeRunResult = {
        ok: false,
        stdout: '',
        stderr: msg,
        error: msg,
        elapsed_ms: 0,
      };
      this.setState({ codeRunResult: fallback, error: msg });
      return fallback;
    } finally {
      this.setState({ codeRunning: false });
    }
  }

  togglePlay(): void {
    const next = !this.state.isPlaying;
    this.setState({ isPlaying: next });
    this.sendWsCommand({ action: next ? 'play' : 'pause' });
  }

  async step(n = 1): Promise<void> {
    const sid = this.state.sessionId;
    if (!sid) return;
    this.setState({ isBusy: true });
    try {
      const m = await fetchJson<SimulationMetrics>(`/api/sessions/${encodeURIComponent(sid)}/step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_steps: n }),
      });
      this.setState({ isReversed: false });
      this.pushMetrics(m);
    } catch (err) {
      this.setState({ error: err instanceof Error ? err.message : 'Step failed' });
    } finally {
      this.setState({ isBusy: false });
    }
  }

  async scrub(targetStep: number): Promise<void> {
    const sid = this.state.sessionId;
    if (!sid) return;
    this.setState({ isPlaying: false });
    this.sendWsCommand({ action: 'pause' });
    try {
      const res = await fetchJson<SimulationMetrics | { metrics: SimulationMetrics; reversed?: boolean }>(
        `/api/sessions/${encodeURIComponent(sid)}/scrub`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ target_step: targetStep }),
        },
      );
      const m = 'metrics' in res ? res.metrics : res;
      const steps = m.history_steps ?? [];
      const maxHistory = steps.length > 0 ? steps[steps.length - 1] : targetStep;
      const rev =
        'reversed' in res && typeof res.reversed === 'boolean' ? res.reversed : targetStep < maxHistory;
      this.setState({ isReversed: rev });
      this.pushMetrics(m);
    } catch (err) {
      this.setState({ error: err instanceof Error ? err.message : 'Scrub failed' });
    }
  }

  async steer(req: SteerRequest): Promise<void> {
    const sid = this.state.sessionId;
    if (!sid) return;
    try {
      const m = await fetchJson<SimulationMetrics>(`/api/sessions/${encodeURIComponent(sid)}/steer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
      });
      this.pushMetrics(m);
    } catch (err) {
      this.setState({ error: err instanceof Error ? err.message : 'Steer failed' });
    }
  }

  async relaxClashes(): Promise<RelaxationReport | null> {
    const sid = this.state.sessionId;
    if (!sid) return null;
    this.setState({ isRelaxing: true, error: null });
    try {
      const report = await fetchJson<RelaxationReport>(
        `/api/sessions/${encodeURIComponent(sid)}/relax`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
        },
      );
      if (this.disposed) return report;
      this.setState({ relaxationReport: report });
      if (report.metrics) {
        this.pushMetrics(report.metrics);
      }
      return report;
    } catch (err) {
      this.setState({ error: err instanceof Error ? err.message : 'Clash relaxation failed' });
      return null;
    } finally {
      if (!this.disposed) {
        this.setState({ isRelaxing: false });
      }
    }
  }

  async setFocusZone(
    centerAng: [number, number, number],
    aaRadiusAng: number,
    cgRadiusAng: number,
  ): Promise<MultiscaleZoneConfig | null> {
    const sid = this.state.sessionId;
    if (!sid) return null;
    const payload = {
      enabled: true,
      center_ang: centerAng,
      aa_radius_ang: Math.max(1, aaRadiusAng),
      cg_radius_ang: Math.max(Math.max(1, aaRadiusAng), cgRadiusAng),
    };
    this.sendWsCommand({
      action: 'set_focus_zone',
      center_ang: payload.center_ang,
      aa_radius_ang: payload.aa_radius_ang,
      cg_radius_ang: payload.cg_radius_ang,
    });
    try {
      const res = await fetchJson<
        (MultiscaleZoneConfig & { metrics?: SimulationMetrics }) | { multiscale?: MultiscaleZoneConfig; metrics?: SimulationMetrics }
      >(`/api/sessions/${encodeURIComponent(sid)}/multiscale`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (this.disposed) return null;
      const ms: MultiscaleZoneConfig =
        'multiscale' in res && res.multiscale
          ? res.multiscale
          : 'counts' in res
            ? res
            : {
                enabled: true,
                center_ang: payload.center_ang,
                aa_radius_ang: payload.aa_radius_ang,
                cg_radius_ang: payload.cg_radius_ang,
                counts: this.state.multiscale?.counts ?? {
                  all_atom: 0,
                  coarse_grained: 0,
                  rigid_scaffold: 0,
                },
              };
      this.setState({ multiscale: ms });
      if ('metrics' in res && res.metrics) {
        this.pushMetrics(res.metrics);
      }
      return ms;
    } catch (err) {
      this.setState({ error: err instanceof Error ? err.message : 'Failed to update multiscale focus zone' });
      return null;
    }
  }

  async steerAtom(
    atomIndex: number,
    targetAng: [number, number, number],
    kSpring = 50.0,
  ): Promise<boolean> {
    const sid = this.state.sessionId;
    if (!sid) return false;
    try {
      const res = await fetchJson<{ ok?: boolean; metrics?: SimulationMetrics }>(
        `/api/sessions/${encodeURIComponent(sid)}/steer-atom`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            enabled: true,
            atom_index: atomIndex,
            target_ang: targetAng,
            k_spring: kSpring,
          }),
        },
      );
      if (this.disposed) return false;
      this.setState({
        steerAtomState: {
          active: true,
          atom_index: atomIndex,
          target_ang: targetAng,
          k_spring: kSpring,
        },
      });
      if (res.metrics) {
        this.pushMetrics(res.metrics);
      }
      return res.ok !== false;
    } catch (err) {
      this.setState({ error: err instanceof Error ? err.message : 'Failed to apply steered MD spring' });
      return false;
    }
  }

  async clearSteerAtom(): Promise<void> {
    const sid = this.state.sessionId;
    this.setState({ steerAtomState: { ...INITIAL_STEER_ATOM_STATE } });
    if (!sid) return;
    try {
      const res = await fetchJson<{ ok?: boolean; metrics?: SimulationMetrics }>(
        `/api/sessions/${encodeURIComponent(sid)}/steer-atom`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            enabled: false,
            atom_index: null,
            target_ang: null,
            k_spring: 0,
          }),
        },
      );
      if (res.metrics) {
        this.pushMetrics(res.metrics);
      }
    } catch (err) {
      this.setState({ error: err instanceof Error ? err.message : 'Failed to clear steered MD spring' });
    }
  }

  setStorageMode(mode: StorageMode): void {
    this.setState({ storageMode: mode });
  }

  async focusZoneAtCameraTarget(
    aaRadiusAng?: number,
    cgRadiusAng?: number,
  ): Promise<MultiscaleZoneConfig | null> {
    const target = this.plugin.canvas3d?.camera.getSnapshot().target;
    if (
      !target ||
      target.length < 3 ||
      !Number.isFinite(target[0]) ||
      !Number.isFinite(target[1]) ||
      !Number.isFinite(target[2])
    ) {
      this.setState({ error: '3D camera target is unavailable.' });
      return null;
    }
    const aaR = aaRadiusAng ?? this.state.multiscale?.aa_radius_ang ?? 15.0;
    const cgR = cgRadiusAng ?? this.state.multiscale?.cg_radius_ang ?? Math.max(aaR * 2, 30.0);
    return this.setFocusZone([target[0], target[1], target[2]], aaR, cgR);
  }

  setDeltaThreshold(thresholdAng: number): void {
    const clamped = Math.max(0, Math.min(0.2, Number.isFinite(thresholdAng) ? thresholdAng : 0.02));
    this.setState({ deltaThresholdAng: clamped });
    this.sendWsCommand({
      action: 'set_delta_threshold',
      threshold_ang: clamped,
    });
  }

  async setColorMode(mode: StudioColorMode): Promise<void> {
    const nextChannel = mode === 'default' ? this.state.activeChannel : mode;
    this.setState({
      colorMode: mode,
      activeChannel: nextChannel,
      frameStats: { ...this.state.frameStats, activeChannel: nextChannel },
    });

    const ctrl = this.resolveController();
    if (ctrl) {
      await ctrl.setStructureColorMode(mode);
    }

    if (mode !== 'default') {
      this.sendWsCommand({ action: 'set_channel', channel: mode });
      await this.steer({ active_channel: mode });
    }
  }

  async setPalette(palette: PaletteName): Promise<void> {
    this.setState({
      activePalette: palette,
      frameStats: { ...this.state.frameStats, activePalette: palette },
    });
    this.resolveController()?.setPalette(palette);
    this.sendWsCommand({ action: 'set_palette', palette });
    await this.steer({ palette });
  }

  toggleOverlay(key: keyof StudioOverlaysState): void {
    const next = {
      ...this.state.overlays,
      [key]: !this.state.overlays[key],
    };
    this.setState({ overlays: next });

    const ctrl = this.resolveController();
    if (ctrl) {
      void ctrl.setOverlayVisibility({
        showBox: next.showBox,
        showBonds: next.showBonds,
        showForces: next.showForces,
      });
      ctrl.setVolumeVisible(next.showVolume);
    }

    if (key === 'showVolume' && next.showVolume) {
      this.sendWsCommand({ action: 'request_volume' });
    }
    if ((key === 'showForces' || key === 'showBonds') && next[key]) {
      this.sendWsCommand({ action: 'request_overlays' });
    }
  }

  async compileCustomPhysics(spec: CustomPhysicsSpec): Promise<boolean> {
    const sid = this.state.sessionId;
    if (!sid) return false;
    this.setState({ compilingPhysics: true, error: null });
    try {
      const res = await fetchJson<{ ok?: boolean; metrics?: SimulationMetrics }>(
        `/api/sessions/${encodeURIComponent(sid)}/custom-physics`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(spec),
        },
      );
      if (res.metrics) {
        this.pushMetrics(res.metrics);
      }
      return res.ok !== false;
    } catch (err) {
      this.setState({ error: err instanceof Error ? err.message : 'Failed to compile custom physics' });
      return false;
    } finally {
      this.setState({ compilingPhysics: false });
    }
  }

  async exportMvsx(): Promise<void> {
    const sid = this.state.sessionId;
    if (!sid) return;
    this.setState({ isBusy: true });
    try {
      const url = `/api/sessions/${encodeURIComponent(sid)}/export/mvsx?channel=${encodeURIComponent(
        this.state.activeChannel,
      )}&palette=${encodeURIComponent(this.state.activePalette)}`;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Export failed (${res.status})`);
      }
      const blob = await res.blob();
      const objUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = objUrl;
      a.download = `biology-mm-${sid}.mvsx`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(objUrl);
    } catch (err) {
      this.setState({ error: err instanceof Error ? err.message : 'Failed to export .mvsx' });
    } finally {
      this.setState({ isBusy: false });
    }
  }

  async inspectAtom(index: number): Promise<void> {
    const sid = this.state.sessionId;
    if (!sid) return;
    try {
      const atom = await fetchJson<AtomInspection>(
        `/api/sessions/${encodeURIComponent(sid)}/atom/${index}`,
      );
      this.setState({ pickedAtom: atom });
    } catch (err) {
      this.setState({ error: err instanceof Error ? err.message : 'Failed to inspect atom' });
    }
  }

  clearPickedAtom(): void {
    this.setState({ pickedAtom: null });
  }

  resetCamera(): void {
    this.resolveController()?.resetCamera();
    this.plugin.canvas3d?.requestCameraReset();
  }

  setCodeText(codeText: string): void {
    this.setState({ codeText });
  }

  insertActiveStructureSnippet(): void {
    const cur = this.state.codeText.trim();
    const next = cur.length > 0 ? `${ACTIVE_STRUCTURE_SNIPPET}\n${cur}` : ACTIVE_STRUCTURE_SNIPPET;
    this.setState({ codeText: next });
  }

  setWideCodeEditorOpen(wideCodeEditorOpen: boolean): void {
    this.setState({ wideCodeEditorOpen });
  }

  clearError(): void {
    this.setState({ error: null });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.closeWebSocket();
    for (const sub of this.controllerSubs) {
      sub.unsubscribe();
    }
    this.controllerSubs = [];
    this.controller = null;
    this.listeners.clear();
  }
}

const storeByPlugin = new WeakMap<PluginContext, BiologyMmStudioStore>();

export function getOrCreateBiologyMmStudioStore(plugin: PluginContext): BiologyMmStudioStore {
  let store = storeByPlugin.get(plugin);
  if (!store) {
    store = new BiologyMmStudioStore(plugin);
    storeByPlugin.set(plugin, store);
  }
  return store;
}

export function getBiologyMmStudioStore(plugin: PluginContext): BiologyMmStudioStore | undefined {
  return storeByPlugin.get(plugin);
}
