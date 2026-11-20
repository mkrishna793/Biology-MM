import { useCallback, useEffect, useRef, useState } from 'react';
import type { PluginContext } from '@molstar/plugin/context';
import { loadStructureFromUrl } from '@molstar/plugin-extension/loaders';
import {
  getBiologyMmStreamController,
  type BiologyMmStreamController,
} from '@molstar/biology-mm-extension';
import type {
  AtomInspection,
  CodeRunResult,
  CustomPhysicsSpec,
  FrameStats,
  PaletteName,
  PhysicsChannelName,
  PresetInfo,
  SimulationMetrics,
  SteerRequest,
  StudioCapabilities,
} from '../types.js';
import type { StudioOverlaysState } from '../components/Toolbar.js';

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
};

const INITIAL_OVERLAYS: StudioOverlaysState = {
  showBox: true,
  showBonds: true,
  showForces: false,
  showVolume: false,
};

export interface UseStudioStreamResult {
  capabilities: StudioCapabilities | null;
  presets: PresetInfo[];
  activePresetId: string | null;
  sessionId: string | null;
  playing: boolean;
  loading: boolean;
  busy: boolean;
  error: string | null;
  metrics: SimulationMetrics | null;
  telemetryHistory: SimulationMetrics[];
  frameStats: FrameStats;
  pickedAtom: AtomInspection | null;
  overlays: StudioOverlaysState;
  activeChannel: PhysicsChannelName;
  activePalette: PaletteName;
  isReversed: boolean;
  codeRunResult: CodeRunResult | null;
  codeRunning: boolean;
  compilingPhysics: boolean;
  onPluginReady: (plugin: PluginContext | null) => void;
  loadPreset: (presetId: string) => Promise<void>;
  togglePlay: () => void;
  step: (n?: number) => Promise<void>;
  scrub: (targetStep: number) => Promise<void>;
  steer: (req: SteerRequest) => Promise<void>;
  setChannel: (channel: PhysicsChannelName) => Promise<void>;
  setPalette: (palette: PaletteName) => Promise<void>;
  toggleOverlay: (key: keyof StudioOverlaysState) => void;
  compileCustomPhysics: (spec: CustomPhysicsSpec) => Promise<boolean>;
  runCode: (code: string) => Promise<CodeRunResult>;
  loadStructureInMolstar: () => Promise<void>;
  exportMvsx: () => Promise<void>;
  inspectAtom: (index: number) => Promise<void>;
  clearPickedAtom: () => void;
  resetCamera: () => void;
  clearError: () => void;
}

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

export function useStudioStream(): UseStudioStreamResult {
  const [capabilities, setCapabilities] = useState<StudioCapabilities | null>(null);
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [activePresetId, setActivePresetId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<SimulationMetrics | null>(null);
  const [telemetryHistory, setTelemetryHistory] = useState<SimulationMetrics[]>([]);
  const [frameStats, setFrameStats] = useState<FrameStats>(INITIAL_FRAME_STATS);
  const [pickedAtom, setPickedAtom] = useState<AtomInspection | null>(null);
  const [overlays, setOverlays] = useState<StudioOverlaysState>(INITIAL_OVERLAYS);
  const [activeChannel, setActiveChannel] = useState<PhysicsChannelName>('force-magnitude');
  const [activePalette, setActivePalette] = useState<PaletteName>('plasma');
  const [isReversed, setIsReversed] = useState(false);
  const [codeRunResult, setCodeRunResult] = useState<CodeRunResult | null>(null);
  const [codeRunning, setCodeRunning] = useState(false);
  const [compilingPhysics, setCompilingPhysics] = useState(false);

  const pluginRef = useRef<PluginContext | null>(null);
  const controllerRef = useRef<BiologyMmStreamController | null>(null);
  const controllerSubsRef = useRef<Array<{ unsubscribe: () => void }>>([]);
  const pendingPacketsRef = useRef<ArrayBuffer[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  const metricsRef = useRef<SimulationMetrics | null>(null);
  const activeChannelRef = useRef<PhysicsChannelName>('force-magnitude');
  const activePaletteRef = useRef<PaletteName>('plasma');
  const overlaysRef = useRef<StudioOverlaysState>(INITIAL_OVERLAYS);

  sessionIdRef.current = sessionId;
  metricsRef.current = metrics;
  activeChannelRef.current = activeChannel;
  activePaletteRef.current = activePalette;
  overlaysRef.current = overlays;

  const pushMetrics = useCallback((m: SimulationMetrics) => {
    setMetrics(m);
    setTelemetryHistory((prev) => {
      const next = [...prev, m];
      return next.length > MAX_TELEMETRY_POINTS
        ? next.slice(next.length - MAX_TELEMETRY_POINTS)
        : next;
    });
    setFrameStats((prev) => ({
      ...prev,
      nAtoms: m.n_atoms,
      simTimePs: m.time_ps,
    }));
  }, []);

  const inspectAtom = useCallback(async (index: number) => {
    const sid = sessionIdRef.current;
    if (!sid) return;
    try {
      const atom = await fetchJson<AtomInspection>(
        `/api/sessions/${encodeURIComponent(sid)}/atom/${index}`,
      );
      setPickedAtom(atom);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to inspect atom');
    }
  }, []);

  const onPluginReady = useCallback(
    (plugin: PluginContext | null) => {
      for (const sub of controllerSubsRef.current) {
        sub.unsubscribe();
      }
      controllerSubsRef.current = [];
      pluginRef.current = plugin;

      if (!plugin) {
        controllerRef.current = null;
        return;
      }

      const ctrl = getBiologyMmStreamController(plugin) ?? null;
      controllerRef.current = ctrl;
      if (!ctrl) return;

      ctrl.setChannel(activeChannelRef.current);
      ctrl.setPalette(activePaletteRef.current);
      void ctrl.setOverlayVisibility({
        showBox: overlaysRef.current.showBox,
        showBonds: overlaysRef.current.showBonds,
        showForces: overlaysRef.current.showForces,
      });
      ctrl.setVolumeVisible(overlaysRef.current.showVolume);

      const subStats = ctrl.frameStats$.subscribe((stats) => {
        setFrameStats((prev) => ({
          ...prev,
          fps: stats.fps,
          frameId: stats.frameId,
          simTimePs: stats.simTimePs,
          nAtoms: stats.nAtoms,
          scalarMin: stats.scalarMin,
          scalarMax: stats.scalarMax,
          activeChannel: stats.activeChannel,
          activePalette: stats.activePalette,
          reversed: stats.reversed,
        }));
        setIsReversed(stats.reversed);
      });
      const subPick = ctrl.atomPicked$.subscribe((ev) => {
        void inspectAtom(ev.index);
      });
      controllerSubsRef.current = [subStats, subPick];

      // Replay any binary WebSocket frames received while Mol* was initializing
      if (pendingPacketsRef.current.length > 0) {
        const queued = pendingPacketsRef.current;
        pendingPacketsRef.current = [];
        for (const pkt of queued) {
          ctrl.ingestPacket(pkt);
        }
      }
    },
    [inspectAtom],
  );

  useEffect(() => {
    return () => {
      for (const sub of controllerSubsRef.current) {
        sub.unsubscribe();
      }
      controllerSubsRef.current = [];
    };
  }, []);

  // Boot capabilities + presets
  useEffect(() => {
    let cancelled = false;
    const boot = async () => {
      setLoading(true);
      try {
        const [caps, presetsRes] = await Promise.all([
          fetchJson<StudioCapabilities>('/api/capabilities'),
          fetchJson<PresetInfo[] | { presets: PresetInfo[] }>('/api/presets'),
        ]);
        if (cancelled) return;
        setCapabilities(caps);
        const list = Array.isArray(presetsRes) ? presetsRes : presetsRes.presets ?? [];
        setPresets(list);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to connect to Biology-MM backend');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };
    void boot();
    return () => {
      cancelled = true;
    };
  }, []);

  // Manage WebSocket lifecycle for active session
  useEffect(() => {
    if (!sessionId) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setFrameStats((prev) => ({ ...prev, wsConnected: false }));
      return;
    }

    const ws = new WebSocket(buildWsUrl(sessionId));
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      setFrameStats((prev) => ({ ...prev, wsConnected: true }));
    };

    ws.onclose = () => {
      setFrameStats((prev) => ({ ...prev, wsConnected: false }));
    };

    ws.onerror = () => {
      setError('WebSocket stream connection error');
    };

    ws.onmessage = (ev: MessageEvent) => {
      if (ev.data instanceof ArrayBuffer) {
        const ctrl = controllerRef.current;
        if (ctrl) {
          ctrl.ingestPacket(ev.data);
        } else {
          const q = pendingPacketsRef.current;
          q.push(ev.data);
          // Preserve initial TOPOLOGY_INIT packet at index 0 while bounding buffer
          if (q.length > MAX_PENDING_PACKETS) {
            q.splice(1, 1);
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
            setError(msg.error || msg.message || 'Simulation stream error');
          } else if (msg.metrics) {
            pushMetrics(msg.metrics);
          } else if (typeof msg.step === 'number' && typeof msg.time_ps === 'number') {
            pushMetrics(msg as SimulationMetrics);
          }
        } catch {
          // Ignore malformed JSON frames
        }
      }
    };

    return () => {
      ws.close();
      if (wsRef.current === ws) {
        wsRef.current = null;
      }
    };
  }, [sessionId, pushMetrics]);

  const sendWsCommand = useCallback((payload: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
    }
  }, []);

  const loadPreset = useCallback(
    async (presetId: string) => {
      setBusy(true);
      setError(null);
      setPlaying(false);
      setPickedAtom(null);
      setIsReversed(false);
      pendingPacketsRef.current = [];
      try {
        const res = await fetchJson<{ session_id: string; metrics?: SimulationMetrics }>(
          '/api/sessions',
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ preset_id: presetId }),
          },
        );
        setActivePresetId(presetId);
        setSessionId(res.session_id);
        setTelemetryHistory(res.metrics ? [res.metrics] : []);
        if (res.metrics) {
          setMetrics(res.metrics);
          setFrameStats((prev) => ({
            ...prev,
            nAtoms: res.metrics!.n_atoms,
            simTimePs: res.metrics!.time_ps,
          }));
        }
        const preset = presets.find((p) => p.id === presetId);
        if (preset) {
          setActiveChannel(preset.default_channel);
          setActivePalette(preset.default_palette);
          controllerRef.current?.setChannel(preset.default_channel);
          controllerRef.current?.setPalette(preset.default_palette);
          setFrameStats((prev) => ({
            ...prev,
            activeChannel: preset.default_channel,
            activePalette: preset.default_palette,
          }));
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load preset');
      } finally {
        setBusy(false);
      }
    },
    [presets],
  );

  const togglePlay = useCallback(() => {
    setPlaying((prev) => {
      const next = !prev;
      sendWsCommand({ action: next ? 'play' : 'pause' });
      return next;
    });
  }, [sendWsCommand]);

  const step = useCallback(
    async (n = 1) => {
      const sid = sessionIdRef.current;
      if (!sid) return;
      setBusy(true);
      try {
        const m = await fetchJson<SimulationMetrics>(
          `/api/sessions/${encodeURIComponent(sid)}/step`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ n_steps: n }),
          },
        );
        setIsReversed(false);
        pushMetrics(m);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Step failed');
      } finally {
        setBusy(false);
      }
    },
    [pushMetrics],
  );

  const scrub = useCallback(
    async (targetStep: number) => {
      const sid = sessionIdRef.current;
      if (!sid) return;
      setPlaying(false);
      sendWsCommand({ action: 'pause' });
      try {
        const res = await fetchJson<
          SimulationMetrics | { metrics: SimulationMetrics; reversed?: boolean }
        >(`/api/sessions/${encodeURIComponent(sid)}/scrub`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ target_step: targetStep }),
        });
        const m = 'metrics' in res ? res.metrics : res;
        const steps = m.history_steps ?? [];
        const maxHistory = steps.length > 0 ? steps[steps.length - 1] : targetStep;
        const rev =
          'reversed' in res && typeof res.reversed === 'boolean'
            ? res.reversed
            : targetStep < maxHistory;
        setIsReversed(rev);
        pushMetrics(m);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Scrub failed');
      }
    },
    [pushMetrics, sendWsCommand],
  );

  const steer = useCallback(
    async (req: SteerRequest) => {
      const sid = sessionIdRef.current;
      if (!sid) return;
      try {
        const m = await fetchJson<SimulationMetrics>(
          `/api/sessions/${encodeURIComponent(sid)}/steer`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req),
          },
        );
        pushMetrics(m);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Steer failed');
      }
    },
    [pushMetrics],
  );

  const setChannel = useCallback(
    async (channel: PhysicsChannelName) => {
      setActiveChannel(channel);
      controllerRef.current?.setChannel(channel);
      setFrameStats((prev) => ({ ...prev, activeChannel: channel }));
      sendWsCommand({ action: 'set_channel', channel });
      await steer({ active_channel: channel });
    },
    [sendWsCommand, steer],
  );

  const setPalette = useCallback(
    async (palette: PaletteName) => {
      setActivePalette(palette);
      controllerRef.current?.setPalette(palette);
      setFrameStats((prev) => ({ ...prev, activePalette: palette }));
      sendWsCommand({ action: 'set_palette', palette });
      await steer({ palette });
    },
    [sendWsCommand, steer],
  );

  const toggleOverlay = useCallback(
    (key: keyof StudioOverlaysState) => {
      setOverlays((prev) => {
        const next = { ...prev, [key]: !prev[key] };
        const ctrl = controllerRef.current;
        if (ctrl) {
          ctrl.setOverlayVisibility({
            showBox: next.showBox,
            showBonds: next.showBonds,
            showForces: next.showForces,
          });
          ctrl.setVolumeVisible(next.showVolume);
        }
        if (key === 'showVolume' && next.showVolume) {
          sendWsCommand({ action: 'request_volume' });
        }
        if ((key === 'showForces' || key === 'showBonds') && next[key]) {
          sendWsCommand({ action: 'request_overlays' });
        }
        return next;
      });
    },
    [sendWsCommand],
  );

  const compileCustomPhysics = useCallback(
    async (spec: CustomPhysicsSpec): Promise<boolean> => {
      const sid = sessionIdRef.current;
      if (!sid) return false;
      setCompilingPhysics(true);
      setError(null);
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
          pushMetrics(res.metrics);
        }
        return res.ok !== false;
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to compile custom physics');
        return false;
      } finally {
        setCompilingPhysics(false);
      }
    },
    [pushMetrics],
  );

  const runCode = useCallback(
    async (code: string): Promise<CodeRunResult> => {
      setCodeRunning(true);
      setError(null);
      try {
        const res = await fetchJson<CodeRunResult>('/api/code/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code }),
        });
        setCodeRunResult(res);
        if (res.ok && res.session_id) {
          pendingPacketsRef.current = [];
          setSessionId(res.session_id);
          setPlaying(false);
          setPickedAtom(null);
          setTelemetryHistory(res.metrics ? [res.metrics] : []);
          if (res.metrics) {
            setMetrics(res.metrics);
          }
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
        setCodeRunResult(fallback);
        setError(msg);
        return fallback;
      } finally {
        setCodeRunning(false);
      }
    },
    [],
  );

  const loadStructureInMolstar = useCallback(async () => {
    const sid = sessionIdRef.current;
    const plugin = pluginRef.current;
    if (!sid || !plugin) return;
    setBusy(true);
    setError(null);
    try {
      const url = `/api/sessions/${encodeURIComponent(sid)}/structure.cif`;
      await loadStructureFromUrl(plugin, url, 'mmcif', false, {
        label: `Biology-MM Step ${metricsRef.current?.step ?? 0}`,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load mmCIF snapshot into Mol*');
    } finally {
      setBusy(false);
    }
  }, []);

  const exportMvsx = useCallback(async () => {
    const sid = sessionIdRef.current;
    if (!sid) return;
    setBusy(true);
    try {
      const url = `/api/sessions/${encodeURIComponent(sid)}/export/mvsx?channel=${encodeURIComponent(
        activeChannel,
      )}&palette=${encodeURIComponent(activePalette)}`;
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
      setError(err instanceof Error ? err.message : 'Failed to export .mvsx');
    } finally {
      setBusy(false);
    }
  }, [activeChannel, activePalette]);

  const clearPickedAtom = useCallback(() => setPickedAtom(null), []);
  const resetCamera = useCallback(() => {
    controllerRef.current?.resetCamera();
  }, []);
  const clearError = useCallback(() => setError(null), []);

  return {
    capabilities,
    presets,
    activePresetId,
    sessionId,
    playing,
    loading,
    busy,
    error,
    metrics,
    telemetryHistory,
    frameStats,
    pickedAtom,
    overlays,
    activeChannel,
    activePalette,
    isReversed,
    codeRunResult,
    codeRunning,
    compilingPhysics,
    onPluginReady,
    loadPreset,
    togglePlay,
    step,
    scrub,
    steer,
    setChannel,
    setPalette,
    toggleOverlay,
    compileCustomPhysics,
    runCode,
    loadStructureInMolstar,
    exportMvsx,
    inspectAtom,
    clearPickedAtom,
    resetCamera,
    clearError,
  };
}
