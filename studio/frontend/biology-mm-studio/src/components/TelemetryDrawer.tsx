import { useMemo, useState } from 'react';
import type { SimulationMetrics } from '../types.js';

export interface TelemetryDrawerProps {
  history: readonly SimulationMetrics[];
  latest: SimulationMetrics | null;
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
}

const SPARK_W = 160;
const SPARK_H = 38;
const SPARK_PAD = 3;

function buildSparkPath(values: readonly number[], minVal: number, maxVal: number): string {
  if (values.length === 0) return '';
  if (values.length === 1) {
    const y = SPARK_H / 2;
    return `M${SPARK_PAD},${y} L${SPARK_W - SPARK_PAD},${y}`;
  }
  const span = Math.max(1e-9, maxVal - minVal);
  const w = SPARK_W - 2 * SPARK_PAD;
  const h = SPARK_H - 2 * SPARK_PAD;

  return values
    .map((v, i) => {
      const x = SPARK_PAD + (i / (values.length - 1)) * w;
      const norm = Math.max(0, Math.min(1, (v - minVal) / span));
      const y = SPARK_PAD + (1 - norm) * h;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

function getBounds(series: readonly (readonly number[])[]): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const arr of series) {
    for (const v of arr) {
      if (Number.isFinite(v)) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) return [min - 1, max + 1];
  return [min, max];
}

interface SingleSparkCardProps {
  label: string;
  valueText: string;
  values: readonly number[];
  strokeClass: string;
}

function SingleSparkCard({
  label,
  valueText,
  values,
  strokeClass,
}: SingleSparkCardProps) {
  const path = useMemo(() => {
    const [min, max] = getBounds([values]);
    return buildSparkPath(values, min, max);
  }, [values]);

  return (
    <div className="bmm-spark-card">
      <div className="bmm-spark-head">
        <span className="bmm-spark-label">{label}</span>
        <span className="bmm-spark-val tabular-nums">{valueText}</span>
      </div>
      <svg
        viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
        className="bmm-spark-svg"
        role="img"
        aria-label={`${label} sparkline`}
      >
        {path && <path d={path} className={strokeClass} fill="none" />}
      </svg>
    </div>
  );
}

export function TelemetryDrawer({
  history,
  latest,
  collapsed: controlledCollapsed,
  onToggleCollapsed,
}: TelemetryDrawerProps) {
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  const isCollapsed = controlledCollapsed ?? internalCollapsed;
  const toggleCollapsed =
    onToggleCollapsed ?? (() => setInternalCollapsed((prev) => !prev));

  const series = useMemo(() => {
    const totalE: number[] = [];
    const potE: number[] = [];
    const kinE: number[] = [];
    const temp: number[] = [];
    const maxForce: number[] = [];
    const throughput: number[] = [];
    const stressZZ: number[] = [];

    for (const m of history) {
      totalE.push(m.total_energy);
      potE.push(m.potential_energy);
      kinE.push(m.kinetic_energy);
      temp.push(m.temperature_K);
      maxForce.push(m.max_force);
      throughput.push(m.ns_per_day_2fs);
      stressZZ.push(m.stress_zz);
    }

    const [eMin, eMax] = getBounds([totalE, potE, kinE]);
    return {
      totalEPath: buildSparkPath(totalE, eMin, eMax),
      potEPath: buildSparkPath(potE, eMin, eMax),
      kinEPath: buildSparkPath(kinE, eMin, eMax),
      temp,
      maxForce,
      throughput,
      stressZZ,
    };
  }, [history]);

  return (
    <section
      className={`bmm-telemetry-drawer${isCollapsed ? ' is-collapsed' : ''}`}
      aria-label="Simulation Telemetry"
    >
      <header className="bmm-telemetry-header">
        <div className="bmm-telemetry-summary tabular-nums">
          <span className="bmm-telem-pill">
            Step <strong>{latest ? latest.step.toLocaleString() : '—'}</strong>
          </span>
          <span className="bmm-telem-pill">
            Time <strong>{latest ? latest.time_ps.toFixed(3) : '—'}</strong> ps
          </span>
          <span className="bmm-telem-pill">
            Temp <strong>{latest ? latest.temperature_K.toFixed(1) : '—'}</strong> K
          </span>
          <span className="bmm-telem-pill">
            Total E <strong>{latest ? latest.total_energy.toFixed(2) : '—'}</strong>
          </span>
          <span className="bmm-telem-pill">
            Throughput{' '}
            <strong>{latest ? latest.ns_per_day_2fs.toFixed(2) : '—'}</strong> ns/day
          </span>
          <span className="bmm-telem-pill">
            Stress zz <strong>{latest ? latest.stress_zz.toFixed(3) : '—'}</strong>
          </span>
        </div>

        <button
          type="button"
          className="bmm-btn bmm-btn-ghost bmm-telemetry-toggle"
          onClick={toggleCollapsed}
          aria-expanded={!isCollapsed}
        >
          {isCollapsed ? 'Expand Telemetry' : 'Collapse Telemetry'}
        </button>
      </header>

      {!isCollapsed && (
        <div className="bmm-telemetry-grid">
          <div className="bmm-spark-card bmm-spark-energy">
            <div className="bmm-spark-head">
              <span className="bmm-spark-label">Total / Potential / Kinetic E</span>
              <span className="bmm-spark-val tabular-nums">
                {latest
                  ? `${latest.total_energy.toFixed(1)} / ${latest.potential_energy.toFixed(1)} / ${latest.kinetic_energy.toFixed(1)}`
                  : '—'}
              </span>
            </div>
            <svg
              viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
              className="bmm-spark-svg"
              role="img"
              aria-label="Energy sparkline"
            >
              {series.potEPath && (
                <path d={series.potEPath} className="bmm-spark-line-pot" fill="none" />
              )}
              {series.kinEPath && (
                <path d={series.kinEPath} className="bmm-spark-line-kin" fill="none" />
              )}
              {series.totalEPath && (
                <path d={series.totalEPath} className="bmm-spark-line-total" fill="none" />
              )}
            </svg>
          </div>

          <SingleSparkCard
            label="Temp (K)"
            valueText={latest ? `${latest.temperature_K.toFixed(1)} K` : '—'}
            values={series.temp}
            strokeClass="bmm-spark-line-temp"
          />

          <SingleSparkCard
            label="Max Force"
            valueText={latest ? latest.max_force.toFixed(2) : '—'}
            values={series.maxForce}
            strokeClass="bmm-spark-line-force"
          />

          <SingleSparkCard
            label="Throughput (ns/day)"
            valueText={latest ? `${latest.ns_per_day_2fs.toFixed(2)}` : '—'}
            values={series.throughput}
            strokeClass="bmm-spark-line-perf"
          />

          <SingleSparkCard
            label="Axial Stress zz"
            valueText={latest ? latest.stress_zz.toFixed(3) : '—'}
            values={series.stressZZ}
            strokeClass="bmm-spark-line-stress"
          />
        </div>
      )}
    </section>
  );
}
