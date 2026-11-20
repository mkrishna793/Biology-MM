import { CollapsableControls, type CollapsableProps, type CollapsableState } from '@molstar/plugin-ui/controls/collapsable';
import { Button } from '@molstar/plugin-ui/controls/common';
import { getOrCreateBiologyMmStudioStore, type BiologyMmStudioStore } from '../studio-store.js';
import type { PhysicsHealthStatus, SimulationMetrics } from '../types.js';

const RAM_BUDGET_MB = 24 * 1024; // 24 GB Mac unified memory budget

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

function formatCompactCount(n: number): string {
  if (!Number.isFinite(n) || n < 0) return '0';
  if (n >= 1_000_000) {
    const val = n / 1_000_000;
    return `${val >= 100 ? val.toFixed(0) : val >= 10 ? val.toFixed(1) : val.toFixed(2)}M`;
  }
  if (n >= 10_000) {
    const val = n / 1_000;
    return `${val >= 100 ? val.toFixed(0) : val.toFixed(1)}k`;
  }
  return Math.round(n).toLocaleString();
}

function formatRamMb(ramMb: number): string {
  if (!Number.isFinite(ramMb) || ramMb < 0) return '—';
  if (ramMb >= 1024) return `${(ramMb / 1024).toFixed(2)} GB`;
  if (ramMb >= 10) return `${Math.round(ramMb)} MB`;
  return `${ramMb.toFixed(1)} MB`;
}

function formatStorageModeBadge(mode: string | undefined): string {
  if (!mode) return 'FLOAT64';
  const norm = mode.toLowerCase();
  if (norm === 'symmetry-compact') return 'SYM-COMPACT';
  if (norm === 'tryte-all-atom') return 'TRYTE-27+5';
  if (norm === 'ternary-compact') return 'TERNARY-COMPACT';
  if (norm === 'full' || norm === 'float64') return 'FLOAT64';
  return mode.toUpperCase();
}

function getRamSeverityClass(ramMb: number): 'is-ok' | 'is-warn' | 'is-high' {
  if (ramMb < 4096) return 'is-ok';
  if (ramMb <= 12288) return 'is-warn';
  return 'is-high';
}

export class BiologyMmTelemetryControls extends CollapsableControls<CollapsableProps, CollapsableState> {
  private unsubStore: (() => void) | undefined;

  protected defaultState(): CollapsableState {
    return {
      isCollapsed: false,
      header: 'Biology-MM Telemetry',
      description: 'Health · RAM · Scale · Atom',
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
    const isCompactMode =
      m?.storage_mode === 'symmetry-compact' ||
      m?.storage_mode === 'tryte-all-atom' ||
      m?.storage_mode === 'ternary-compact';

    const ms = m?.multiscale ?? snap.multiscale;
    const msCounts = ms?.counts;
    const hasMsCounts =
      msCounts !== undefined &&
      msCounts.all_atom + msCounts.coarse_grained + msCounts.rigid_scaffold > 0;

    const sessionTotalAtoms = m?.n_atoms ?? snap.frameStats.nAtoms;
    const aaCount = hasMsCounts
      ? msCounts.all_atom
      : isCompactMode && m?.active_all_atom_count !== undefined
        ? m.active_all_atom_count
        : sessionTotalAtoms;
    const cgCount = hasMsCounts ? msCounts.coarse_grained : 0;
    const rigidCount = hasMsCounts
      ? msCounts.rigid_scaffold
      : isCompactMode && m?.active_all_atom_count !== undefined
        ? Math.max(0, sessionTotalAtoms - m.active_all_atom_count)
        : 0;
    const totalScaleCount = Math.max(1, aaCount + cgCount + rigidCount);
    const aaPct = (aaCount / totalScaleCount) * 100;
    const cgPct = (cgCount / totalScaleCount) * 100;
    const rigidPct = Math.max(0, 100 - aaPct - cgPct);

    const dirtyAtoms = m?.delta_stats?.dirty_atoms ?? snap.frameStats.lastDirtyAtomCount;
    const totalAtoms = m?.delta_stats?.total_atoms ?? sessionTotalAtoms;
    const packetBytes = m?.delta_stats?.packet_bytes ?? snap.frameStats.lastPacketBytes;

    const hasRamMetrics =
      (m?.ram_mb !== undefined && Number.isFinite(m.ram_mb)) || Boolean(m?.storage_mode);
    const ramMb = m?.ram_mb !== undefined && Number.isFinite(m.ram_mb) ? Math.max(0, m.ram_mb) : 0;
    const ramPct = ramMb > 0 ? Math.min(100, Math.max(2, (ramMb / RAM_BUDGET_MB) * 100)) : 0;
    const ramSeverity = getRamSeverityClass(ramMb);
    const storageBadge = formatStorageModeBadge(m?.storage_mode);
    const savingsRatio =
      m?.memory_savings_ratio !== undefined &&
      Number.isFinite(m.memory_savings_ratio) &&
      m.memory_savings_ratio > 1.05
        ? m.memory_savings_ratio
        : undefined;
    const savingsLabel = savingsRatio
      ? `${savingsRatio >= 10 ? Math.round(savingsRatio) : savingsRatio.toFixed(1)}× saved`
      : undefined;

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

        {hasRamMetrics && (
          <div className="bmm-msp-ram-box">
            <div className="bmm-msp-ram-hdr">
              <span>Engine RAM</span>
              <strong>
                {m?.ram_mb !== undefined && Number.isFinite(m.ram_mb) ? formatRamMb(ramMb) : '—'} / 24 GB
              </strong>
            </div>
            <div
              className="bmm-msp-ram-bar"
              aria-label="Live Engine RAM Budget"
              title={
                m?.naive_float64_ram_mb !== undefined && Number.isFinite(m.naive_float64_ram_mb)
                  ? `Using ${formatRamMb(ramMb)} vs ${formatRamMb(m.naive_float64_ram_mb)} naive float64`
                  : `Engine RAM: ${formatRamMb(ramMb)} / 24 GB`
              }
            >
              <div className={`bmm-msp-ram-fill ${ramSeverity}`} style={{ width: `${ramPct}%` }} />
            </div>
            <div className="bmm-msp-ram-meta">
              <span
                className={`bmm-msp-ram-badge ${isCompactMode ? 'is-compact' : ''}`}
                title={m?.storage_mode ?? 'float64'}
              >
                {storageBadge}
              </span>
              <span className="bmm-msp-ram-extra">
                {savingsLabel && <strong className="bmm-msp-ram-savings">{savingsLabel}</strong>}
                {isCompactMode && m?.active_all_atom_count !== undefined && (
                  <span title={`Active All-Atom count: ${m.active_all_atom_count.toLocaleString()}`}>
                    {savingsLabel ? ' · ' : ''}AA {formatCompactCount(m.active_all_atom_count)}
                  </span>
                )}
              </span>
            </div>
          </div>
        )}

        <div className="bmm-ms-telemetry">
          <div
            className="bmm-ms-telemetry-hdr"
            title={`All-Atom: ${aaCount.toLocaleString()} · Coarse-Grained: ${cgCount.toLocaleString()} · Rigid Scaffold: ${rigidCount.toLocaleString()}`}
          >
            <span>Multiscale Zones</span>
            <span className="bmm-ms-telemetry-counts">
              AA {formatCompactCount(aaCount)} · CG {formatCompactCount(cgCount)} · Rigid {formatCompactCount(rigidCount)}
            </span>
          </div>
          <div className="bmm-ms-bar" aria-label="Multiscale Zone Distribution">
            <div
              className="bmm-ms-seg-aa"
              style={{ width: `${aaPct}%`, minWidth: aaCount > 0 ? '3px' : undefined }}
              title={`All-Atom: ${aaCount.toLocaleString()}`}
            />
            <div
              className="bmm-ms-seg-cg"
              style={{ width: `${cgPct}%`, minWidth: cgCount > 0 ? '3px' : undefined }}
              title={`Coarse-Grained: ${cgCount.toLocaleString()}`}
            />
            <div
              className="bmm-ms-seg-rigid"
              style={{ width: `${rigidPct}%`, minWidth: rigidCount > 0 ? '3px' : undefined }}
              title={`Rigid Scaffold: ${rigidCount.toLocaleString()}`}
            />
          </div>
        </div>

        <div
          className="bmm-delta-bar"
          title={`${dirtyAtoms.toLocaleString()} / ${totalAtoms.toLocaleString()} dirty atoms · ${formatBytes(packetBytes)}`}
        >
          <span>Delta Stream</span>
          <strong>
            {formatCompactCount(dirtyAtoms)}/{formatCompactCount(totalAtoms)} dirty · {formatBytes(packetBytes)}
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
