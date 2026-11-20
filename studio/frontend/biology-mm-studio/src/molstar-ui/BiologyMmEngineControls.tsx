import type { ChangeEvent } from 'react';
import { getPaletteStopsHex } from '@molstar/biology-mm-extension';
import { CollapsableControls, type CollapsableProps, type CollapsableState } from '@molstar/plugin-ui/controls/collapsable';
import { Button, ControlRow } from '@molstar/plugin-ui/controls/common';
import {
  getOrCreateBiologyMmStudioStore,
  type BiologyMmStudioStore,
  type StudioColorMode,
  type StudioOverlaysState,
} from '../studio-store.js';
import type { PaletteName, SimulationMode } from '../types.js';

type EngineTabId = 'prep' | 'scale' | 'run' | 'lab';

interface EngineControlsState extends CollapsableState {
  activeTab: EngineTabId;
  selectedPresetId: string;
  importMode: SimulationMode;
  tempInput: string;
  efieldInput: string;
  focusCenterX: string;
  focusCenterY: string;
  focusCenterZ: string;
  aaRadiusInput: string;
  cgRadiusInput: string;
  lastSyncedFocusKey: string;
  customExpr: string;
  customRMin: string;
  customRMax: string;
  lastSyncedSessionId: string | null;
}

const ENGINE_TABS: ReadonlyArray<{ id: EngineTabId; label: string; title: string }> = [
  { id: 'prep', label: 'Prep', title: 'Load presets, import structures, and relax steric clashes' },
  { id: 'scale', label: 'Scale', title: '3-Zone multiscale focus sphere and dirty-tile delta streaming' },
  { id: 'run', label: 'Run', title: 'Playback, steering, physics coloring, layers, and export' },
  { id: 'lab', label: 'Lab', title: 'Custom radial pair potential U(r) spline compiler' },
];

const COLOR_MODE_OPTIONS: ReadonlyArray<{ value: StudioColorMode; label: string }> = [
  { value: 'default', label: 'Native Mol* (Chain / Element)' },
  { value: 'force-magnitude', label: 'Force Magnitude |F|' },
  { value: 'kinetic-temp', label: 'Kinetic Temp (K)' },
  { value: 'charge-potential', label: 'Charge / Potential' },
  { value: 'bond-order', label: 'Bond Order' },
  { value: 'atomic-stress', label: 'Atomic Stress' },
  { value: 'ternary-state', label: 'Ternary State (-1/0/+1)' },
  { value: 'multiscale-zone', label: 'Multiscale Zone (AA / CG / Rigid)' },
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
      description: 'Prep · Scale · Run · Lab',
      activeTab: 'prep',
      selectedPresetId: 'biomolecular-complex',
      importMode: 'metal_verlet',
      tempInput: '300',
      efieldInput: '0',
      focusCenterX: '0.0',
      focusCenterY: '0.0',
      focusCenterZ: '0.0',
      aaRadiusInput: '15.0',
      cgRadiusInput: '30.0',
      lastSyncedFocusKey: '',
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

    if (
      snap.activePresetId &&
      snap.activePresetId !== this.state.selectedPresetId &&
      snap.sessionId !== this.state.lastSyncedSessionId
    ) {
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

    const ms = snap.multiscale ?? snap.metrics?.multiscale;
    if (ms && ms.center_ang) {
      const key = `${ms.center_ang[0].toFixed(2)},${ms.center_ang[1].toFixed(2)},${ms.center_ang[2].toFixed(2)}:${ms.aa_radius_ang.toFixed(1)}:${ms.cg_radius_ang.toFixed(1)}`;
      if (key !== this.state.lastSyncedFocusKey) {
        updates.lastSyncedFocusKey = key;
        updates.focusCenterX = ms.center_ang[0].toFixed(1);
        updates.focusCenterY = ms.center_ang[1].toFixed(1);
        updates.focusCenterZ = ms.center_ang[2].toFixed(1);
        updates.aaRadiusInput = ms.aa_radius_ang.toFixed(1);
        updates.cgRadiusInput = ms.cg_radius_ang.toFixed(1);
      }
    }

    if (Object.keys(updates).length > 0) {
      this.setState(updates as any);
    }
  }

  private renderPrepTab(store: BiologyMmStudioStore) {
    const snap = store.getSnapshot();
    const report = snap.relaxationReport;
    const maxForce = snap.metrics?.max_force;
    const health = snap.metrics?.health_status;

    let clashStatusText = 'Ready';
    let clashStatusClass = 'is-ok';
    if (snap.isRelaxing) {
      clashStatusText = 'Relaxing soft-core contacts...';
      clashStatusClass = 'is-warn';
    } else if (report) {
      const fFinal =
        report.final_max_force !== undefined ? `max |F| ${report.final_max_force.toFixed(1)}` : '';
      const stepsStr = report.steps !== undefined ? `${report.steps} steps` : '';
      const resolvedStr =
        report.clashes_resolved !== undefined ? `${report.clashes_resolved} resolved` : '';
      const summary = [resolvedStr, fFinal, stepsStr].filter(Boolean).join(' · ');
      clashStatusText = report.message || summary || 'Relaxation complete';
      clashStatusClass = (report.remaining_clashes ?? 0) > 0 ? 'is-warn' : 'is-ok';
    } else if (health === 'clash' || (maxForce !== undefined && maxForce > 5000)) {
      clashStatusText = `Steric clash detected (max |F| ${maxForce !== undefined ? maxForce.toFixed(0) : 'high'})`;
      clashStatusClass = 'is-clash';
    } else if (health === 'warning') {
      clashStatusText = `High force warning (max |F| ${maxForce !== undefined ? maxForce.toFixed(1) : '—'})`;
      clashStatusClass = 'is-warn';
    } else if (maxForce !== undefined) {
      clashStatusText = `No clashes · max |F| ${maxForce.toFixed(1)}`;
      clashStatusClass = 'is-ok';
    }

    return (
      <div className="bmm-tab-body">
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
                className="bmm-btn-compact"
                onClick={() => void store.loadPreset(this.state.selectedPresetId)}
                disabled={snap.isBusy || snap.presets.length === 0}
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

        <div className="bmm-action-stack">
          <Button
            commit="on"
            onClick={() =>
              void store.simulateActiveMolstarStructure({
                mode: this.state.importMode,
                temperature: Math.max(0.1, Number(this.state.tempInput) || 300),
              })
            }
            disabled={snap.isBusy}
            title="Extract the currently loaded PDB/mmCIF structure from Mol* (auto symmetry-native for >100k capsid assemblies) and simulate live"
          >
            Simulate Active Mol* Structure
          </Button>

          <label
            className={`msp-btn msp-btn-action bmm-file-upload-btn${snap.isBusy ? ' msp-btn-disabled' : ''}`}
            title="Directly upload structure files (.pdb, .cif, .mmcif, .gro, .xyz, .data) up to 1M+ atoms to the simulation engine"
          >
            Upload Structure File
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

          <Button
            onClick={() => void store.relaxClashes()}
            disabled={!snap.sessionId || snap.isBusy || snap.isRelaxing}
            title="Run soft-core steepest-descent relaxation to resolve steric clashes before MD integration"
          >
            {snap.isRelaxing ? 'Relaxing Clashes...' : 'Relax Clashes (Soft-Core)'}
          </Button>

          <div className={`bmm-relax-status ${clashStatusClass}`}>
            <span className="bmm-relax-dot" />
            <span>{clashStatusText}</span>
          </div>
        </div>
      </div>
    );
  }

  private renderScaleTab(store: BiologyMmStudioStore) {
    const snap = store.getSnapshot();
    const ms = snap.multiscale ?? snap.metrics?.multiscale;
    const counts = ms?.counts ?? {
      all_atom: snap.frameStats.nAtoms,
      coarse_grained: 0,
      rigid_scaffold: 0,
    };

    return (
      <div className="bmm-tab-body">
        <div className="bmm-section-subhdr">3-Zone Multiscale Focus Sphere</div>

        <ControlRow
          label="Center (Å)"
          control={
            <div className="bmm-msp-xyz-grid">
              <input
                type="number"
                step={1}
                className="msp-form-control"
                aria-label="Focus Center X (Å)"
                title="Focus Center X (Å)"
                placeholder="X"
                value={this.state.focusCenterX}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  this.setState({ focusCenterX: e.target.value })
                }
              />
              <input
                type="number"
                step={1}
                className="msp-form-control"
                aria-label="Focus Center Y (Å)"
                title="Focus Center Y (Å)"
                placeholder="Y"
                value={this.state.focusCenterY}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  this.setState({ focusCenterY: e.target.value })
                }
              />
              <input
                type="number"
                step={1}
                className="msp-form-control"
                aria-label="Focus Center Z (Å)"
                title="Focus Center Z (Å)"
                placeholder="Z"
                value={this.state.focusCenterZ}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  this.setState({ focusCenterZ: e.target.value })
                }
              />
            </div>
          }
        />

        <ControlRow
          label="AA Rad (Å)"
          control={
            <input
              type="number"
              min={1}
              step={1}
              className="msp-form-control"
              aria-label="All-Atom Radius (Å)"
              value={this.state.aaRadiusInput}
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                this.setState({ aaRadiusInput: e.target.value })
              }
            />
          }
        />

        <ControlRow
          label="CG Rad (Å)"
          control={
            <input
              type="number"
              min={1}
              step={1}
              className="msp-form-control"
              aria-label="Coarse-Grained Buffer Radius (Å)"
              value={this.state.cgRadiusInput}
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                this.setState({ cgRadiusInput: e.target.value })
              }
            />
          }
        />

        <div className="bmm-action-stack">
          <Button
            commit="on"
            disabled={!snap.sessionId || snap.isBusy}
            onClick={() => {
              const cx = Number(this.state.focusCenterX) || 0;
              const cy = Number(this.state.focusCenterY) || 0;
              const cz = Number(this.state.focusCenterZ) || 0;
              const aaR = Math.max(1, Number(this.state.aaRadiusInput) || 15);
              const cgR = Math.max(aaR, Number(this.state.cgRadiusInput) || 30);
              void store.setFocusZone([cx, cy, cz], aaR, cgR);
            }}
          >
            Apply Focus Zone
          </Button>
        </div>

        <div className="bmm-ms-counts">
          <div className="bmm-ms-chip is-aa">
            <span>All-Atom</span>
            <strong>{counts.all_atom.toLocaleString()}</strong>
          </div>
          <div className="bmm-ms-chip is-cg">
            <span>CG Buffer</span>
            <strong>{counts.coarse_grained.toLocaleString()}</strong>
          </div>
          <div className="bmm-ms-chip is-rigid">
            <span>Rigid</span>
            <strong>{counts.rigid_scaffold.toLocaleString()}</strong>
          </div>
        </div>

        <div className="bmm-section-subhdr">Dirty-Tile Delta Streaming</div>

        <ControlRow
          label="Delta (Å)"
          control={
            <div className="bmm-msp-scrubber">
              <input
                type="range"
                aria-label="Dirty-Tile Delta Threshold (Å)"
                min={0}
                max={0.2}
                step={0.01}
                value={snap.deltaThresholdAng}
                disabled={!snap.sessionId}
                onChange={(e: ChangeEvent<HTMLInputElement>) =>
                  store.setDeltaThreshold(Number(e.target.value))
                }
              />
              <span className="bmm-msp-stat">{snap.deltaThresholdAng.toFixed(2)} Å</span>
            </div>
          }
        />
      </div>
    );
  }

  private renderRunTab(store: BiologyMmStudioStore) {
    const snap = store.getSnapshot();
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
      <div className="bmm-tab-body">
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
              <span className="bmm-msp-stat">
                {currentStep}/{maxStep}
              </span>
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
              <Button
                inline
                className="bmm-btn-compact"
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
          label="E-Field"
          control={
            <input
              type="number"
              className="msp-form-control"
              aria-label="Electric Field Z (V/nm)"
              title="Electric Field Z (V/nm)"
              value={this.state.efieldInput}
              step={0.05}
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                this.setState({ efieldInput: e.target.value })
              }
            />
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
      </div>
    );
  }

  private renderLabTab(store: BiologyMmStudioStore) {
    const snap = store.getSnapshot();

    return (
      <div className="bmm-tab-body">
        <div className="bmm-section-subhdr">Custom Physics Law (U(r) Spline)</div>
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
          label="r_min (nm)"
          control={
            <input
              type="number"
              step={0.02}
              className="msp-form-control"
              aria-label="Spline Minimum Radius (nm)"
              value={this.state.customRMin}
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                this.setState({ customRMin: e.target.value })
              }
            />
          }
        />
        <ControlRow
          label="r_max (nm)"
          control={
            <input
              type="number"
              step={0.05}
              className="msp-form-control"
              aria-label="Spline Maximum Radius (nm)"
              value={this.state.customRMax}
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                this.setState({ customRMax: e.target.value })
              }
            />
          }
        />
        <div className="bmm-action-stack">
          <Button
            commit="on"
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
            {snap.compilingPhysics ? 'Compiling Spline...' : 'Compile U(r) Spline'}
          </Button>
          <div className="bmm-msp-hint">
            Variables: <code>r</code>, <code>eps</code>, <code>sig</code>. Tabulated into GPU spline LUT on compile.
          </div>
        </div>
      </div>
    );
  }

  protected renderControls() {
    const store = this.store;
    if (!store) return null;
    const snap = store.getSnapshot();

    const hardwareBackend = snap.capabilities?.hardware?.backend ?? snap.metrics?.backend ?? 'CPU';
    const currentStep = snap.metrics?.step ?? 0;

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

        <div className="bmm-msp-tabs" role="tablist" aria-label="Engine workflow stages">
          {ENGINE_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={this.state.activeTab === tab.id}
              className={`bmm-msp-tab${this.state.activeTab === tab.id ? ' is-active' : ''}`}
              title={tab.title}
              onClick={() => this.setState({ activeTab: tab.id })}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {this.state.activeTab === 'prep' && this.renderPrepTab(store)}
        {this.state.activeTab === 'scale' && this.renderScaleTab(store)}
        {this.state.activeTab === 'run' && this.renderRunTab(store)}
        {this.state.activeTab === 'lab' && this.renderLabTab(store)}
      </div>
    );
  }
}
