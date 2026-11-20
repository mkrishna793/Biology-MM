import type { ChangeEvent } from 'react';
import { getPaletteStopsHex } from '@molstar/biology-mm-extension';
import { CollapsableControls, type CollapsableProps, type CollapsableState } from '@molstar/plugin-ui/controls/collapsable';
import { Button, ControlRow, ExpandGroup } from '@molstar/plugin-ui/controls/common';
import {
  getOrCreateBiologyMmStudioStore,
  type BiologyMmStudioStore,
  type StudioColorMode,
  type StudioOverlaysState,
} from '../studio-store.js';
import type { PaletteName, SimulationMode } from '../types.js';

interface EngineControlsState extends CollapsableState {
  selectedPresetId: string;
  importMode: SimulationMode;
  tempInput: string;
  efieldInput: string;
  customExpr: string;
  customRMin: string;
  customRMax: string;
  lastSyncedSessionId: string | null;
}

const COLOR_MODE_OPTIONS: ReadonlyArray<{ value: StudioColorMode; label: string }> = [
  { value: 'default', label: 'Native Mol* (Chain / Element)' },
  { value: 'force-magnitude', label: 'Force Magnitude |F|' },
  { value: 'kinetic-temp', label: 'Kinetic Temp (K)' },
  { value: 'charge-potential', label: 'Charge / Potential' },
  { value: 'bond-order', label: 'Bond Order' },
  { value: 'atomic-stress', label: 'Atomic Stress' },
  { value: 'ternary-state', label: 'Ternary State (-1/0/+1)' },
];

const PALETTE_OPTIONS: ReadonlyArray<{ value: PaletteName; label: string }> = [
  { value: 'plasma', label: 'Plasma' },
  { value: 'thermal-div', label: 'Thermal Diverging' },
  { value: 'electrostatic-bwr', label: 'Electrostatic BWR' },
  { value: 'bond-viridis', label: 'Bond Viridis' },
  { value: 'stress-inferno', label: 'Stress Inferno' },
  { value: 'ternary-triad', label: 'Ternary Triad (-1/0/+1)' },
];

const SIM_MODES: ReadonlyArray<{ value: SimulationMode; label: string }> = [
  { value: 'metal_verlet', label: 'Biomolecular / Metal Verlet' },
  { value: 'cnt_tersoff', label: 'Reactive Carbon (Tersoff)' },
  { value: 'custom_spline', label: 'Custom Pair Potential' },
  { value: 'nanopore_gcmc_nemd', label: 'Nanopore GCMC + NEMD' },
];

export class BiologyMmEngineControls extends CollapsableControls<CollapsableProps, EngineControlsState> {
  private unsubStore: (() => void) | undefined;

  protected defaultState(): EngineControlsState {
    return {
      isCollapsed: false,
      header: 'Biology-MM Engine',
      description: 'Physics & Layers',
      selectedPresetId: 'biomolecular-complex',
      importMode: 'metal_verlet',
      tempInput: '300',
      efieldInput: '0',
      customExpr: '4 * eps * ((sig / r)**12 - (sig / r)**6)',
      customRMin: '0.22',
      customRMax: '1.1',
      lastSyncedSessionId: null,
    };
  }

  private get store(): BiologyMmStudioStore | undefined {
    return this.plugin ? getOrCreateBiologyMmStudioStore(this.plugin) : undefined;
  }

  componentDidMount(): void {
    const store = this.store;
    if (!store) return;

    this.syncSessionDefaults(store);
    this.unsubStore = store.subscribe(() => {
      this.syncSessionDefaults(store);
      this.forceUpdate();
    });
  }

  componentWillUnmount(): void {
    super.componentWillUnmount();
    this.unsubStore?.();
    this.unsubStore = undefined;
  }

  private syncSessionDefaults(store: BiologyMmStudioStore): void {
    const snap = store.getSnapshot();
    const updates: Partial<EngineControlsState> = {};
    if (snap.activePresetId && snap.activePresetId !== this.state.selectedPresetId && snap.sessionId !== this.state.lastSyncedSessionId) {
      updates.selectedPresetId = snap.activePresetId;
    }
    if (snap.sessionId && snap.sessionId !== this.state.lastSyncedSessionId) {
      updates.lastSyncedSessionId = snap.sessionId;
      const activePreset = snap.presets.find((p) => p.id === snap.activePresetId);
      if (activePreset) {
        updates.tempInput = String(activePreset.temperature ?? 300);
        updates.efieldInput = String(activePreset.efield_z ?? 0);
      } else if (snap.metrics) {
        updates.tempInput = String(Math.round(snap.metrics.temperature_K || 300));
      }
    }
    if (Object.keys(updates).length > 0) {
      this.setState(updates as any);
    }
  }

  protected renderControls() {
    const store = this.store;
    if (!store) return null;
    const snap = store.getSnapshot();

    const hardwareBackend = snap.capabilities?.hardware?.backend ?? snap.metrics?.backend ?? 'CPU';
    const currentStep = snap.metrics?.step ?? 0;
    const historySteps = snap.metrics?.history_steps ?? [];
    const minStep = historySteps.length > 0 ? historySteps[0] : 0;
    const maxStep = historySteps.length > 0 ? historySteps[historySteps.length - 1] : 0;

    const stops = getPaletteStopsHex(snap.activePalette);
    const gradientCss = `linear-gradient(to right, ${stops.join(', ')})`;

    const overlayItems: ReadonlyArray<{ key: keyof StudioOverlaysState; label: string }> = [
      { key: 'showBox', label: 'Unit Cell' },
      { key: 'showBonds', label: 'Bonds' },
      { key: 'showForces', label: 'Forces' },
      { key: 'showVolume', label: '3D Field' },
    ];

    return (
      <div className="bmm-msp-panel">
        <div className="bmm-msp-status-bar">
          <span className="bmm-msp-badge">
            <span className={`bmm-msp-dot ${snap.frameStats.wsConnected ? 'is-live' : ''}`} />
            {hardwareBackend.toUpperCase()}
          </span>
          <span className="bmm-msp-stat">{snap.frameStats.nAtoms.toLocaleString()} atoms</span>
          <span className="bmm-msp-stat">step {currentStep}</span>
          <span className="bmm-msp-stat">{snap.frameStats.simTimePs.toFixed(3)} ps</span>
        </div>

        {snap.error && (
          <div className="bmm-msp-error" role="alert">
            <span>{snap.error}</span>
            <button
              type="button"
              className="bmm-msp-close-x"
              onClick={() => store.clearError()}
              aria-label="Dismiss error"
            >
              ×
            </button>
          </div>
        )}

        <ControlRow
          label="Preset"
          control={
            <div className="bmm-msp-inline-group">
              <select
                className="msp-form-control"
                aria-label="Simulation Preset"
                value={this.state.selectedPresetId}
                onChange={(e: ChangeEvent<HTMLSelectElement>) =>
                  this.setState({ selectedPresetId: e.target.value })
                }
                disabled={snap.presets.length === 0 || snap.isBusy}
              >
                {snap.presets.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.title} ({p.n_atoms})
                  </option>
                ))}
              </select>
              <Button
                inline
                onClick={() => void store.loadPreset(this.state.selectedPresetId)}
                disabled={snap.isBusy || snap.presets.length === 0}
                style={{ width: 'auto', padding: '0 8px' }}
                title="Load selected Biology-MM preset into Mol*"
              >
                Load
              </Button>
            </div>
          }
        />

        <ControlRow
          label="Engine"
          control={
            <select
              className="msp-form-control"
              aria-label="Physics Mode for Imported Structure"
              value={this.state.importMode}
              onChange={(e: ChangeEvent<HTMLSelectElement>) =>
                this.setState({ importMode: e.target.value as SimulationMode })
              }
            >
              {SIM_MODES.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.label}
                </option>
              ))}
            </select>
          }
        />

        <div className="msp-btn-row-textbox" style={{ padding: '2px 0' }}>
          <Button
            commit="on"
            onClick={() =>
              void store.simulateActiveMolstarStructure({
                mode: this.state.importMode,
                temperature: Math.max(0.1, Number(this.state.tempInput) || 300),
              })
            }
            disabled={snap.isBusy}
            title="Extract the currently loaded PDB/mmCIF structure from Mol* and simulate it live in Biology-MM"
          >
            Simulate Active Mol* Structure
          </Button>
        </div>

        <div className="msp-btn-row-textbox" style={{ padding: '2px 0' }}>
          <label
            className={`msp-btn msp-btn-action${snap.isBusy ? ' msp-btn-disabled' : ''}`}
            style={{ width: '100%', boxSizing: 'border-box', textAlign: 'center', cursor: snap.isBusy ? 'default' : 'pointer', margin: 0, padding: '4px 6px', fontSize: '11px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            title="Directly upload high-scale raw structure files (.pdb, .cif, .gro, .xyz, .data) up to 1M+ atoms to the simulation engine"
          >
            Upload Structure File to Engine
            <input
              type="file"
              accept=".pdb,.cif,.mmcif,.gro,.xyz,.data,.lammps"
              style={{ display: 'none' }}
              disabled={snap.isBusy}
              onChange={(e: ChangeEvent<HTMLInputElement>) => {
                const file = e.target.files?.[0];
                if (file) {
                  void store.uploadStructureFile(file, {
                    mode: this.state.importMode,
                    temperature: Math.max(0.1, Number(this.state.tempInput) || 300),
                  });
                  e.target.value = '';
                }
              }}
            />
          </label>
        </div>

        <ControlRow
          label="Playback"
          control={
            <div className="bmm-msp-btn-grid-3">
              <Button
                inline
                commit={snap.isPlaying ? 'on' : undefined}
                onClick={() => store.togglePlay()}
                disabled={!snap.sessionId}
              >
                {snap.isPlaying ? 'Pause' : 'Play'}
              </Button>
              <Button
                inline
                onClick={() => void store.step(1)}
                disabled={!snap.sessionId || snap.isBusy}
              >
                +1
              </Button>
              <Button
                inline
                onClick={() => void store.step(10)}
                disabled={!snap.sessionId || snap.isBusy}
              >
                +10
              </Button>
            </div>
          }
        />

        <ControlRow
          label={snap.isReversed ? 'Scrub (REV)' : 'History'}
          control={
            <div className="bmm-msp-scrubber">
              <input
                type="range"
                aria-label="Trajectory History Scrubber"
                min={minStep}
                max={Math.max(minStep, maxStep)}
                value={Math.min(Math.max(minStep, currentStep), Math.max(minStep, maxStep))}
                disabled={!snap.sessionId || historySteps.length <= 1}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  void store.scrub(Number(e.target.value))
                }
              />
              <span className="bmm-msp-stat">{currentStep}/{maxStep}</span>
            </div>
          }
        />

        <ControlRow
          label="Temp (K)"
          control={
            <div className="bmm-msp-inline-group">
              <input
                type="number"
                className="msp-form-control"
                aria-label="Target Temperature (K)"
                value={this.state.tempInput}
                min={1}
                step={10}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  this.setState({ tempInput: e.target.value })
                }
              />
              <input
                type="number"
                className="msp-form-control"
                aria-label="Electric Field Z (V/nm)"
                title="E-field Z (V/nm)"
                value={this.state.efieldInput}
                step={0.05}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  this.setState({ efieldInput: e.target.value })
                }
              />
              <Button
                inline
                style={{ width: 'auto', padding: '0 8px' }}
                disabled={!snap.sessionId}
                onClick={() =>
                  void store.steer({
                    temperature: Math.max(1, Number(this.state.tempInput) || 300),
                    efield_z: Number(this.state.efieldInput) || 0,
                  })
                }
              >
                Steer
              </Button>
            </div>
          }
        />

        <ControlRow
          label="Coloring"
          control={
            <select
              className="msp-form-control"
              aria-label="Color Mode"
              value={snap.colorMode}
              onChange={(e: ChangeEvent<HTMLSelectElement>) =>
                void store.setColorMode(e.target.value as StudioColorMode)
              }
            >
              {COLOR_MODE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          }
        />

        <ControlRow
          label="Palette"
          control={
            <select
              className="msp-form-control"
              aria-label="Color Palette"
              value={snap.activePalette}
              onChange={(e: ChangeEvent<HTMLSelectElement>) =>
                void store.setPalette(e.target.value as PaletteName)
              }
            >
              {PALETTE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          }
        />

        {snap.colorMode !== 'default' && (
          <div className="bmm-msp-colorbar-wrap">
            <span>{snap.frameStats.scalarMin.toFixed(2)}</span>
            <div className="bmm-msp-colorbar" style={{ background: gradientCss }} />
            <span>{snap.frameStats.scalarMax.toFixed(2)}</span>
          </div>
        )}

        <ControlRow
          label="Layers"
          control={
            <div className="bmm-msp-btn-grid-2">
              {overlayItems.map((item) => (
                <Button
                  key={item.key}
                  inline
                  commit={snap.overlays[item.key] ? 'on' : undefined}
                  onClick={() => store.toggleOverlay(item.key)}
                >
                  {item.label}
                </Button>
              ))}
            </div>
          }
        />

        <ControlRow
          label="Actions"
          control={
            <div className="bmm-msp-btn-grid-2">
              <Button
                inline
                onClick={() => void store.exportMvsx()}
                disabled={!snap.sessionId || snap.isBusy}
              >
                Export .mvsx
              </Button>
              <Button inline onClick={() => store.resetCamera()}>
                Reset Cam
              </Button>
            </div>
          }
        />

        <ExpandGroup header="Custom Physics Law (Spline)" initiallyExpanded={false}>
          <ControlRow
            label="U(r)"
            control={
              <input
                type="text"
                className="msp-form-control"
                aria-label="Custom Radial Potential Formula"
                value={this.state.customExpr}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  this.setState({ customExpr: e.target.value })
                }
              />
            }
          />
          <ControlRow
            label="r [nm]"
            control={
              <div className="bmm-msp-inline-group">
                <input
                  type="number"
                  step={0.02}
                  className="msp-form-control"
                  value={this.state.customRMin}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    this.setState({ customRMin: e.target.value })
                  }
                />
                <input
                  type="number"
                  step={0.05}
                  className="msp-form-control"
                  value={this.state.customRMax}
                  onChange={(e: ChangeEvent<HTMLInputElement>) =>
                    this.setState({ customRMax: e.target.value })
                  }
                />
                <Button
                  inline
                  style={{ width: 'auto', padding: '0 8px' }}
                  disabled={!snap.sessionId || snap.compilingPhysics}
                  onClick={() =>
                    void store.compileCustomPhysics({
                      potential_expr: this.state.customExpr,
                      r_min: Number(this.state.customRMin) || 0.22,
                      r_max: Number(this.state.customRMax) || 1.1,
                      temperature: Number(this.state.tempInput) || 300,
                      efield_z: Number(this.state.efieldInput) || 0,
                      gcmc_mu: -2.0,
                      strain_rate: 0,
                    })
                  }
                >
                  {snap.compilingPhysics ? '...' : 'Compile'}
                </Button>
              </div>
            }
          />
        </ExpandGroup>
      </div>
    );
  }
}
