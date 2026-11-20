// @vitest-environment jsdom
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Unit } from '@molstar/model/model/structure';
import { PluginReactContext } from '@molstar/plugin-ui/base';
import { App } from '../App.js';
import { BiologyMmCodeLabControls } from '../molstar-ui/BiologyMmCodeLabControls.js';
import { BiologyMmEngineControls } from '../molstar-ui/BiologyMmEngineControls.js';
import { BiologyMmTelemetryControls } from '../molstar-ui/BiologyMmTelemetryControls.js';
import { getOrCreateBiologyMmStudioStore } from '../studio-store.js';
import type {
  CodeRunResult,
  PresetInfo,
  SimulationMetrics,
  StudioCapabilities,
} from '../types.js';

(globalThis as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT = true;

const MOCK_CAPABILITIES: StudioCapabilities = {
  studio_version: '1.0.0',
  hardware: {
    has_gpu: true,
    backend: 'Metal',
    device_name: 'Apple M3 Max',
    unified_memory: true,
    simd_width: 32,
  },
  channels: [
    'force-magnitude',
    'kinetic-temp',
    'charge-potential',
    'bond-order',
    'atomic-stress',
    'ternary-state',
  ],
  palettes: [
    'plasma',
    'thermal-div',
    'electrostatic-bwr',
    'bond-viridis',
    'stress-inferno',
    'ternary-triad',
  ],
  presets_count: 2,
  max_interactive_atoms: 1_000_000,
  supports_mvsx: true,
};

const MOCK_PRESETS: PresetInfo[] = [
  {
    id: 'biomolecular-complex',
    title: 'Biomolecular Complex (Protein + Ligand + Ions + Water)',
    category: 'Biomolecular',
    description: '16-residue alpha-helix in water + NaCl + ligand.',
    n_atoms: 169,
    capacity: 169,
    mode: 'metal_verlet',
    units: 'md',
    default_channel: 'charge-potential',
    default_palette: 'electrostatic-bwr',
    temperature: 310,
    code_template: 'from trium_studio.presets import build_biomolecular_complex_system\n',
    tags: ['Protein', 'Ligand'],
  },
  {
    id: 'cnt-tersoff-tensile',
    title: 'CNT Tersoff Tensile',
    category: 'Nanomechanics',
    description: 'Carbon nanotube axial strain simulation.',
    n_atoms: 320,
    capacity: 320,
    mode: 'cnt_tersoff',
    units: 'md',
    default_channel: 'atomic-stress',
    default_palette: 'stress-inferno',
    temperature: 300,
    code_template: 'from biology_mm import NanostructureBuilder\n',
    tags: ['CNT', 'Tersoff'],
  },
];

function makeMetrics(
  presetId: string,
  step: number,
  historySteps: number[],
): SimulationMetrics {
  return {
    step,
    time_ps: step * 0.002,
    n_atoms: presetId === 'biomolecular-complex' ? 169 : 320,
    capacity: presetId === 'biomolecular-complex' ? 169 : 320,
    potential_energy: -1250.4 + step * 0.1,
    kinetic_energy: 185.2,
    total_energy: -1065.2 + step * 0.1,
    temperature_K: 310.0,
    max_force: 14.2,
    ms_per_step: 1.8,
    ns_per_day_2fs: 96.0,
    box_lengths_nm: [4.2, 4.2, 4.2],
    volume_nm3: 74.088,
    stress_zz: 1.25,
    mode: presetId === 'cnt-tersoff-tensile' ? 'cnt_tersoff' : 'metal_verlet',
    backend: 'Metal',
    history_steps: historySteps,
    session_id: `sess-${presetId}`,
  };
}

class MockWebSocket {
  static OPEN = 1;
  readyState = MockWebSocket.OPEN;
  binaryType = 'blob';
  onopen: ((ev: Event) => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onerror: ((ev: Event) => void) | null = null;
  onclose: ((ev: CloseEvent) => void) | null = null;
  sent: string[] = [];

  constructor(public url: string) {
    queueMicrotask(() => {
      this.onopen?.(new Event('open'));
    });
  }

  send(data: string): void {
    this.sent.push(data);
  }

  close(): void {
    this.readyState = 3;
    this.onclose?.(new Event('close') as CloseEvent);
  }
}

function createMockPluginWithStructure() {
  const coords = [
    [0.0, 0.0, 0.0],
    [1.54, 0.0, 0.0],
  ];
  const mockUnit = {
    id: 1,
    kind: Unit.Kind.Atomic,
    elements: [0, 1],
    conformation: {
      position: (el: number, out: number[]) => {
        out[0] = coords[el][0];
        out[1] = coords[el][1];
        out[2] = coords[el][2];
        return out;
      },
    },
    bonds: {
      edgeCount: 2,
      a: [0, 1],
      b: [1, 0],
    },
    model: {
      sourceData: { kind: 'custom', name: 'test-crambin', data: {} },
      _staticPropertyData: {},
      _dynamicPropertyData: {},
      atomicHierarchy: {
        atoms: {
          type_symbol: { value: (i: number) => (i === 0 ? 'C' : 'O') },
          label_atom_id: { value: (i: number) => (i === 0 ? 'CA' : 'O') },
          label_comp_id: { value: () => 'ALA' },
          auth_comp_id: { value: () => 'ALA' },
          pdbx_formal_charge: { isDefined: true, value: () => 0 },
        },
        residues: {
          group_PDB: { value: () => 'ATOM' },
          label_seq_id: { value: () => 1 },
          auth_seq_id: { value: () => 1 },
        },
        chains: {
          label_asym_id: { value: () => 'A' },
          auth_asym_id: { value: () => 'A' },
        },
        residueAtomSegments: { index: [0, 0] },
        chainAtomSegments: { index: [0, 0] },
        index: { getEntityFromChain: () => 0 },
      },
      entities: {
        data: {
          id: { value: () => '1' },
          type: { value: () => 'polymer' },
          pdbx_description: { isDefined: true, value: () => 'Test Peptide' },
        },
      },
    },
  };

  return {
    managers: {
      structure: {
        hierarchy: {
          current: {
            structures: [
              {
                cell: {
                  transform: { ref: 'external-struct-1' },
                  obj: {
                    label: '1CRN Crambin',
                    data: {
                      label: '1CRN Crambin',
                      units: [mockUnit],
                      interUnitBonds: { edgeCount: 0, edges: [] },
                    },
                  },
                },
                components: [],
              },
            ],
          },
        },
      },
    },
    canvas3d: {
      requestCameraReset: vi.fn(),
    },
    state: {
      data: {
        cells: new Map(),
      },
    },
  };
}

describe('Biology-MM Studio Native Mol* UI & Store', () => {
  let container: HTMLDivElement;
  let root: Root;
  let fetchCalls: Array<{ url: string; method: string; body?: string }>;
  let uncaughtErrors: unknown[];

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
    fetchCalls = [];
    uncaughtErrors = [];

    vi.spyOn(console, 'error').mockImplementation(() => {});
    window.addEventListener('error', (e) => uncaughtErrors.push(e.error ?? e.message));

    (globalThis as unknown as { WebSocket: typeof WebSocket }).WebSocket =
      MockWebSocket as unknown as typeof WebSocket;

    let currentPreset = 'biomolecular-complex';

    globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString();
      const method = init?.method ?? 'GET';
      const bodyStr = typeof init?.body === 'string' ? init.body : undefined;
      fetchCalls.push({ url, method, body: bodyStr });

      const jsonResponse = (data: unknown, status = 200) =>
        new Response(JSON.stringify(data), {
          status,
          headers: { 'Content-Type': 'application/json' },
        });

      if (url === '/api/capabilities') return jsonResponse(MOCK_CAPABILITIES);
      if (url === '/api/presets') return jsonResponse(MOCK_PRESETS);

      if (url === '/api/sessions' && method === 'POST') {
        const parsed = bodyStr ? (JSON.parse(bodyStr) as { preset_id: string }) : { preset_id: currentPreset };
        currentPreset = parsed.preset_id;
        return jsonResponse({
          session_id: `sess-${currentPreset}`,
          metrics: makeMetrics(currentPreset, 0, [0]),
        });
      }

      if ((url.startsWith('/api/sessions/create-from-binary') || url.startsWith('/api/sessions/create-from-symmetry') || url.startsWith('/api/sessions/from-structure')) && method === 'POST') {
        const parsed = bodyStr ? (JSON.parse(bodyStr) as { structure: { title: string; elements: string[] } }) : null;
        return jsonResponse({
          session_id: 'sess-imported',
          title: parsed?.structure.title ?? '1CRN Crambin',
          metrics: {
            ...makeMetrics('biomolecular-complex', 0, [0]),
            n_atoms: parsed?.structure.elements.length ?? 2,
            session_id: 'sess-imported',
          },
        });
      }

      if (url.includes('/structure.cif')) {
        return new Response('data_biology_mm\n_cell.length_a 42.0\n', {
          status: 200,
          headers: { 'Content-Type': 'chemical/x-cif' },
        });
      }

      if (url.endsWith('/step') && method === 'POST') {
        return jsonResponse(makeMetrics(currentPreset, 1, [0, 1]));
      }

      if (url.endsWith('/steer') && method === 'POST') {
        return jsonResponse(makeMetrics(currentPreset, 0, [0]));
      }

      if (url === '/api/code/run' && method === 'POST') {
        const result: CodeRunResult = {
          ok: true,
          session_id: 'sess-code',
          stdout: 'Extracted atoms=2\n',
          stderr: '',
          elapsed_ms: 8.4,
          metrics: makeMetrics(currentPreset, 0, [0]),
        };
        return jsonResponse(result);
      }

      return jsonResponse({ ok: true });
    }) as typeof fetch;
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
    vi.restoreAllMocks();
  });

  it('mounts cleanly in jsdom without WebGL2 and shows clean Mol* viewport fallback', async () => {
    await act(async () => {
      root.render(<App />);
    });

    expect(uncaughtErrors).toHaveLength(0);
    expect(container.querySelector('[data-testid="molstar-viewport"]')).not.toBeNull();
    expect(container.textContent).toContain('WebGL2 Unavailable');
  });

  it('boots biomolecular-complex preset, extracts active Mol* structure, and forwards it to simulation and Code Lab', async () => {
    const mockPlugin = createMockPluginWithStructure();
    const store = getOrCreateBiologyMmStudioStore(mockPlugin as any);

    await act(async () => {
      await store.initializeAndBoot();
    });

    let snap = store.getSnapshot();
    expect(snap.activePresetId).toBe('biomolecular-complex');
    expect(snap.colorMode).toBe('default');
    expect(snap.sessionId).toBe('sess-biomolecular-complex');

    const ok = await act(async () => {
      return await store.simulateActiveMolstarStructure({ mode: 'metal_verlet', temperature: 310 });
    });
    expect(ok).toBe(true);
    const fromStructCall = fetchCalls.find(
      (c) =>
        c.url.startsWith('/api/sessions/create-from-binary') ||
        c.url.startsWith('/api/sessions/create-from-symmetry') ||
        c.url.startsWith('/api/sessions/from-structure'),
    );
    expect(fromStructCall).toBeDefined();

    store.insertActiveStructureSnippet();
    const codeRes = await act(async () => {
      return await store.runCode();
    });
    expect(codeRes.ok).toBe(true);
    expect(codeRes.stdout).toContain('Extracted atoms=2');

    const codeCall = fetchCalls.find((c) => c.url === '/api/code/run');
    expect(codeCall).toBeDefined();
    const codeBody = JSON.parse(codeCall!.body!) as { active_structure?: { elements: string[] } };
    expect(codeBody.active_structure?.elements).toEqual(['C', 'O']);

    await act(async () => {
      await store.setColorMode('force-magnitude');
      store.setWideCodeEditorOpen(true);
    });
    snap = store.getSnapshot();
    expect(snap.colorMode).toBe('force-magnitude');
    expect(snap.wideCodeEditorOpen).toBe(true);

    store.dispose();
  });

  it('renders native Mol* CollapsableControls panels cleanly via PluginReactContext', async () => {
    const mockPlugin = createMockPluginWithStructure();
    const store = getOrCreateBiologyMmStudioStore(mockPlugin as any);

    await act(async () => {
      await store.initializeAndBoot();
      root.render(
        <PluginReactContext.Provider value={mockPlugin as any}>
          <div>
            <BiologyMmEngineControls />
            <BiologyMmCodeLabControls />
            <BiologyMmTelemetryControls />
          </div>
        </PluginReactContext.Provider>,
      );
    });

    expect(uncaughtErrors).toHaveLength(0);
    expect(container.textContent).toContain('Biology-MM Engine');
    expect(container.textContent).toContain('Simulate Active Mol* Structure');
    expect(container.textContent).toContain('Biology-MM Code Lab');
    expect(container.textContent).toContain('Expand Wide Editor');
    expect(container.textContent).toContain('Biology-MM Telemetry');
    expect(container.textContent).toContain('Atom Inspector');

    store.dispose();
  });
});
