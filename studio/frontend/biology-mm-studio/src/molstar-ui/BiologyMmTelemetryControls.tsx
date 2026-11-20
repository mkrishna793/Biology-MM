import { CollapsableControls, type CollapsableProps, type CollapsableState } from '@molstar/plugin-ui/controls/collapsable';
import { Button } from '@molstar/plugin-ui/controls/common';
import { getOrCreateBiologyMmStudioStore, type BiologyMmStudioStore } from '../studio-store.js';

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

export class BiologyMmTelemetryControls extends CollapsableControls<CollapsableProps, CollapsableState> {
  private unsubStore: (() => void) | undefined;

  protected defaultState(): CollapsableState {
    return {
      isCollapsed: false,
      header: 'Biology-MM Telemetry',
      description: 'Metrics & Atom',
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

    return (
      <div className="bmm-msp-panel">
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
              <Button inline style={{ width: 'auto', padding: '0 6px' }} onClick={() => store.clearPickedAtom()}>
                Clear
              </Button>
            )}
          </div>
          {atom ? (
            <div className="bmm-msp-atom-grid">
              <div>
                <span>Atom</span>
                <strong>{atom.element} #{atom.index}</strong>
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
                  {atom.position_ang[0].toFixed(2)}, {atom.position_ang[1].toFixed(2)}, {atom.position_ang[2].toFixed(2)}
                </strong>
              </div>
            </div>
          ) : (
            <div className="bmm-msp-hint">Click any atom in 3D to inspect live physics state.</div>
          )}
        </div>
      </div>
    );
  }
}
