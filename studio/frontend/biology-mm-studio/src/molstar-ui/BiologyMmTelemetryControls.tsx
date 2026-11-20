import { CollapsableControls, type CollapsableProps, type CollapsableState } from '@molstar/plugin-ui/controls/collapsable';
import { Button } from '@molstar/plugin-ui/controls/common';
import { getOrCreateBiologyMmStudioStore, type BiologyMmStudioStore } from '../studio-store.js';
import type { PhysicsHealthStatus, SimulationMetrics } from '../types.js';

function buildSparklinePoints(values: number[], width: number, height: number): string {
  if (values.length === 0) return '';
  if (values.length === 1) {
    return `0,${height / 2} ${width},${height / 2}`;
  }
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const span = Math.max(1e-6, max - min);
  const pad = 2;
  const innerH = Math.max(2, height - pad * 2);

  return values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const norm = (v - min) / span;
      const y = height - pad - norm * innerH;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

function resolveHealthStatus(m: SimulationMetrics | null): { status: PhysicsHealthStatus; label: string } {
  if (!m) return { status: 'bounded', label: 'Bounded' };
  const raw = (m.health_status ?? '').toLowerCase();
  if (raw === 'clash' || raw === 'critical' || raw === 'unstable') {
    return { status: 'clash', label: 'Clash' };
  }
  if (raw === 'warning' || raw === 'warn' || raw === 'drift') {
    return { status: 'warning', label: 'Warning' };
  }
  if (raw === 'bounded' || raw === 'ok' || raw === 'healthy') {
    return { status: 'bounded', label: 'Bounded' };
  }
  if (m.max_force > 5000) return { status: 'clash', label: 'Clash' };
  if ((m.drift_fraction !== undefined && Math.abs(m.drift_fraction) > 0.01) || m.max_force > 1000) {
    return { status: 'warning', label: 'Warning' };
  }
  return { status: 'bounded', label: 'Bounded' };
}

function formatDriftFraction(drift: number | undefined): string {
  if (drift === undefined || !Number.isFinite(drift)) return '—';
  const abs = Math.abs(drift);
  if (abs === 0) return '0.00e+0';
  if (abs < 1e-3 || abs >= 1) return drift.toExponential(2);
  return `${(drift * 100).toFixed(3)}%`;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export class BiologyMmTelemetryControls extends CollapsableControls<CollapsableProps, CollapsableState> {
  private unsubStore: (() => void) | undefined;

  protected defaultState(): CollapsableState {
    return {
      isCollapsed: false,
      header: 'Biology-MM Telemetry',
      description: 'Health · Scale · Atom',
    };
  }

  private get store(): BiologyMmStudioStore | undefined {
    return this.plugin ? getOrCreateBiologyMmStudioStore(this.plugin) : undefined;
  }

  componentDidMount(): void {
    const store = this.store;
    if (!store) return;
    this.unsubStore = store.subscribe(() => this.forceUpdate());
  }

  componentWillUnmount(): void {
    super.componentWillUnmount();
    this.unsubStore?.();
    this.unsubStore = undefined;
  }

  protected renderControls() {
    const store = this.store;
    if (!store) return null;
    const snap = store.getSnapshot();
    const m = snap.metrics;
    const atom = snap.pickedAtom;

    const energies = snap.metricHistory.map((h) => h.total_energy);
    const temps = snap.metricHistory.map((h) => h.temperature_K);

    const health = resolveHealthStatus(m);
    const ms = snap.multiscale ?? m?.multiscale;
    const aaCount = ms?.counts?.all_atom ?? (m?.n_atoms ?? snap.frameStats.nAtoms);
    const cgCount = ms?.counts?.coarse_grained ?? 0;
    const rigidCount = ms?.counts?.rigid_scaffold ?? 0;
    const totalScaleCount = Math.max(1, aaCount + cgCount + rigidCount);
    const aaPct = (aaCount / totalScaleCount) * 100;
    const cgPct = (cgCount / totalScaleCount) * 100;
    const rigidPct = Math.max(0, 100 - aaPct - cgPct);

    const dirtyAtoms = m?.delta_stats?.dirty_atoms ?? snap.frameStats.lastDirtyAtomCount;
    const totalAtoms = m?.delta_stats?.total_atoms ?? snap.frameStats.nAtoms;
    const packetBytes = m?.delta_stats?.packet_bytes ?? snap.frameStats.lastPacketBytes;

    return (
      <div className="bmm-msp-panel">
        <div className="bmm-health-row">
          <span className={`bmm-health-badge is-${health.status}`}>
            <span className="bmm-health-dot" />
            {health.label}
          </span>
          <span className="bmm-health-drift" title="Shadow-Hamiltonian energy drift fraction">
            Shadow-H ΔH/H: <strong>{formatDriftFraction(m?.drift_fraction)}</strong>
          </span>
        </div>

        <div className="bmm-msp-metrics-grid">
          <div className="bmm-msp-metric-cell">
            <span>Total E</span>
            <strong>{m ? m.total_energy.toFixed(1) : '—'}</strong>
          </div>
          <div className="bmm-msp-metric-cell">
            <span>Temp (K)</span>
            <strong>{m ? m.temperature_K.toFixed(1) : '—'}</strong>
          </div>
          <div className="bmm-msp-metric-cell">
            <span>Potential E</span>
            <strong>{m ? m.potential_energy.toFixed(1) : '—'}</strong>
          </div>
          <div className="bmm-msp-metric-cell">
            <span>Kinetic E</span>
            <strong>{m ? m.kinetic_energy.toFixed(1) : '—'}</strong>
          </div>
          <div className="bmm-msp-metric-cell">
            <span>Max |F|</span>
            <strong>{m ? m.max_force.toFixed(2) : '—'}</strong>
          </div>
          <div className="bmm-msp-metric-cell">
            <span>Speed</span>
            <strong>{m ? `${m.ms_per_step.toFixed(1)} ms` : '—'}</strong>
          </div>
        </div>

        <div className="bmm-ms-telemetry">
          <div className="bmm-ms-telemetry-hdr">
            <span>Multiscale Zones</span>
            <span>
              AA {aaCount.toLocaleString()} · CG {cgCount.toLocaleString()} · Rigid {rigidCount.toLocaleString()}
            </span>
          </div>
          <div className="bmm-ms-bar" aria-label="Multiscale Zone Distribution">
            <div className="bmm-ms-seg-aa" style={{ width: `${aaPct}%` }} title={`All-Atom: ${aaCount}`} />
            <div className="bmm-ms-seg-cg" style={{ width: `${cgPct}%` }} title={`Coarse-Grained: ${cgCount}`} />
            <div className="bmm-ms-seg-rigid" style={{ width: `${rigidPct}%` }} title={`Rigid Scaffold: ${rigidCount}`} />
          </div>
        </div>

        <div className="bmm-delta-bar">
          <span>Delta Stream</span>
          <strong>
            {dirtyAtoms.toLocaleString()}/{totalAtoms.toLocaleString()} dirty · {formatBytes(packetBytes)}
          </strong>
        </div>

        <div className="bmm-msp-sparks">
          <div className="bmm-msp-spark-box">
            <span>Total Energy</span>
            <svg viewBox="0 0 120 28" className="bmm-msp-spark-svg" aria-label="Total Energy Sparkline">
              {energies.length > 0 && (
                <polyline
                  fill="none"
                  stroke="#9a6b1f"
                  strokeWidth="1.5"
                  points={buildSparklinePoints(energies, 120, 28)}
                />
              )}
            </svg>
          </div>
          <div className="bmm-msp-spark-box">
            <span>Temperature (K)</span>
            <svg viewBox="0 0 120 28" className="bmm-msp-spark-svg" aria-label="Temperature Sparkline">
              {temps.length > 0 && (
                <polyline
                  fill="none"
                  stroke="#2f6c9a"
                  strokeWidth="1.5"
                  points={buildSparklinePoints(temps, 120, 28)}
                />
              )}
            </svg>
          </div>
        </div>

        <div className="bmm-msp-atom-box">
          <div className="bmm-msp-atom-hdr">
            <span>Atom Inspector</span>
            {atom && (
              <Button inline className="bmm-btn-compact" onClick={() => store.clearPickedAtom()}>
                Clear
              </Button>
            )}
          </div>
          {atom ? (
            <>
              <div className="bmm-msp-atom-grid">
                <div>
                  <span>Atom</span>
                  <strong>
                    {atom.element} #{atom.index}
                  </strong>
                </div>
                <div>
                  <span>|F|</span>
                  <strong>{atom.force_magnitude.toFixed(3)}</strong>
                </div>
                <div>
                  <span>Local T</span>
                  <strong>{atom.local_temp_K.toFixed(1)} K</strong>
                </div>
                <div>
                  <span>Charge</span>
                  <strong>{atom.charge.toFixed(3)}</strong>
                </div>
                <div>
                  <span>Bond Ord</span>
                  <strong>{atom.bond_order.toFixed(2)}</strong>
                </div>
                <div>
                  <span>Ternary</span>
                  <strong>{atom.ternary_state.toFixed(1)}</strong>
                </div>
                <div className="bmm-msp-atom-wide">
                  <span>Pos (Å)</span>
                  <strong>
                    {atom.position_ang[0].toFixed(2)}, {atom.position_ang[1].toFixed(2)},{' '}
                    {atom.position_ang[2].toFixed(2)}
                  </strong>
                </div>
              </div>

              <div className="bmm-atom-actions">
                <Button
                  inline
                  onClick={() => {
                    const aaR = ms?.aa_radius_ang ?? 15.0;
                    const cgR = ms?.cg_radius_ang ?? 30.0;
                    void store.setFocusZone(
                      [atom.position_ang[0], atom.position_ang[1], atom.position_ang[2]],
                      aaR,
                      cgR,
                    );
                  }}
                  title="Center the 3-zone All-Atom multiscale sphere on this atom"
                >
                  Focus Zone Here
                </Button>
                <Button
                  inline
                  commit={
                    snap.steerAtomState.active && snap.steerAtomState.atom_index === atom.index
                      ? 'on'
                      : undefined
                  }
                  onClick={() => {
                    const target: [number, number, number] = [
                      atom.position_ang[0],
                      atom.position_ang[1],
                      atom.position_ang[2] + 5.0,
                    ];
                    void store.steerAtom(atom.index, target, 50.0);
                  }}
                  title="Apply interactive harmonic steered MD spring (+5 Å along Z) to this atom"
                >
                  Pull +5 Å (SMD)
                </Button>
              </div>

              {snap.steerAtomState.active && (
                <div className="bmm-smd-active-row">
                  <span>SMD active on #{snap.steerAtomState.atom_index}</span>
                  <Button inline className="bmm-btn-compact" onClick={() => void store.clearSteerAtom()}>
                    Release
                  </Button>
                </div>
              )}
            </>
          ) : (
            <div className="bmm-msp-hint">Click any atom in 3D to inspect live physics state.</div>
          )}
        </div>
      </div>
    );
  }
}
