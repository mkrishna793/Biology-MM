import { useCallback, useEffect, useMemo, useState } from 'react';
import type { CustomPhysicsSpec, SteerRequest } from '../types.js';

export interface PhysicsBuilderPaneProps {
  initialSpec?: Partial<CustomPhysicsSpec>;
  activePresetId?: string | null;
  compiling?: boolean;
  onCompileAndSwap: (spec: CustomPhysicsSpec) => Promise<void> | void;
  onSteer?: (req: SteerRequest) => Promise<void> | void;
}

interface PotentialChip {
  id: string;
  label: string;
  expr: string;
  rMin: number;
  rMax: number;
}

const POTENTIAL_CHIPS: readonly PotentialChip[] = [
  {
    id: 'lj-12-6',
    label: 'Lennard-Jones 12-6',
    expr: '4 * 0.996 * ((0.34 / r)**12 - (0.34 / r)**6)',
    rMin: 0.28,
    rMax: 1.0,
  },
  {
    id: 'morse-cc',
    label: 'Morse C-C',
    expr: '340.0 * (1 - exp(-21.0 * (r - 0.142)))**2 - 340.0',
    rMin: 0.1,
    rMax: 0.45,
  },
  {
    id: 'yukawa',
    label: 'Screened Yukawa',
    expr: '120.0 * exp(-2.5 * r) / r',
    rMin: 0.15,
    rMax: 1.2,
  },
  {
    id: 'soft-core',
    label: 'Soft-Core Repulsion',
    expr: '50.0 * max(0, 1.0 - r / 0.4)**2',
    rMin: 0.05,
    rMax: 0.6,
  },
];

// Safe recursive-descent evaluator for radial potential U(r) expressions.
function evaluatePotentialExpr(expr: string, rVal: number): number | null {
  let pos = 0;
  const src = expr.trim();
  if (!src) return null;

  const skipWs = () => {
    while (pos < src.length && /\s/.test(src[pos])) pos++;
  };

  const parseExpr = (): number => parseAddSub();

  const parseAddSub = (): number => {
    let left = parseMulDiv();
    while (true) {
      skipWs();
      const ch = src[pos];
      if (ch === '+') {
        pos++;
        left += parseMulDiv();
      } else if (ch === '-') {
        pos++;
        left -= parseMulDiv();
      } else {
        break;
      }
    }
    return left;
  };

  const parseMulDiv = (): number => {
    let left = parsePower();
    while (true) {
      skipWs();
      const ch = src[pos];
      if (ch === '*') {
        if (src[pos + 1] === '*') break;
        pos++;
        left *= parsePower();
      } else if (ch === '/') {
        pos++;
        left /= parsePower();
      } else {
        break;
      }
    }
    return left;
  };

  const parsePower = (): number => {
    const base = parseUnary();
    skipWs();
    if (src[pos] === '^') {
      pos++;
      return Math.pow(base, parsePower());
    }
    if (src[pos] === '*' && src[pos + 1] === '*') {
      pos += 2;
      return Math.pow(base, parsePower());
    }
    return base;
  };

  const parseUnary = (): number => {
    skipWs();
    if (src[pos] === '+') {
      pos++;
      return parseUnary();
    }
    if (src[pos] === '-') {
      pos++;
      return -parseUnary();
    }
    return parsePrimary();
  };

  const parsePrimary = (): number => {
    skipWs();
    if (src[pos] === '(') {
      pos++;
      const v = parseExpr();
      skipWs();
      if (src[pos] !== ')') throw new Error('Expected )');
      pos++;
      return v;
    }

    const numMatch = /^(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/.exec(src.slice(pos));
    if (numMatch) {
      pos += numMatch[0].length;
      return Number(numMatch[0]);
    }

    const idMatch = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(pos));
    if (idMatch) {
      const name = idMatch[0];
      pos += name.length;
      skipWs();
      if (src[pos] === '(') {
        pos++;
        const args: number[] = [];
        skipWs();
        if (src[pos] !== ')') {
          args.push(parseExpr());
          while (true) {
            skipWs();
            if (src[pos] === ',') {
              pos++;
              args.push(parseExpr());
            } else {
              break;
            }
          }
        }
        skipWs();
        if (src[pos] !== ')') throw new Error('Expected )');
        pos++;

        switch (name) {
          case 'exp':
            return Math.exp(args[0]);
          case 'sin':
            return Math.sin(args[0]);
          case 'cos':
            return Math.cos(args[0]);
          case 'tan':
            return Math.tan(args[0]);
          case 'tanh':
            return Math.tanh(args[0]);
          case 'sqrt':
            return Math.sqrt(args[0]);
          case 'abs':
            return Math.abs(args[0]);
          case 'log':
            return Math.log(args[0]);
          case 'min':
            return Math.min(...args);
          case 'max':
            return Math.max(...args);
          case 'pow':
            return Math.pow(args[0], args[1]);
          default:
            throw new Error(`Unknown function: ${name}`);
        }
      }

      if (name === 'r') return rVal;
      if (name === 'pi') return Math.PI;
      if (name === 'e') return Math.E;
      throw new Error(`Unknown identifier: ${name}`);
    }

    throw new Error('Unexpected token');
  };

  try {
    const val = parseExpr();
    skipWs();
    if (pos !== src.length || !Number.isFinite(val)) return null;
    return val;
  } catch {
    return null;
  }
}

interface CurveData {
  valid: boolean;
  uPath: string;
  fPath: string;
  zeroY: number | null;
  uMin: number;
  uMax: number;
}

const SVG_W = 320;
const SVG_H = 150;
const PAD_X = 10;
const PAD_Y = 12;

function buildPotentialCurve(expr: string, rMin: number, rMax: number): CurveData {
  if (!(rMin > 0 && rMax > rMin)) {
    return { valid: false, uPath: '', fPath: '', zeroY: null, uMin: 0, uMax: 0 };
  }

  const samples = 96;
  const rs: number[] = [];
  const us: number[] = [];
  const fs: number[] = [];
  const dr = (rMax - rMin) / (samples - 1);
  const h = Math.max(1e-5, dr * 0.25);

  for (let i = 0; i < samples; i++) {
    const r = rMin + i * dr;
    const u = evaluatePotentialExpr(expr, r);
    const uPlus = evaluatePotentialExpr(expr, r + h);
    const uMinus = evaluatePotentialExpr(expr, Math.max(1e-6, r - h));
    if (u === null || uPlus === null || uMinus === null) {
      return { valid: false, uPath: '', fPath: '', zeroY: null, uMin: 0, uMax: 0 };
    }
    const force = -(uPlus - uMinus) / (2 * h);
    rs.push(r);
    us.push(u);
    fs.push(Number.isFinite(force) ? force : 0);
  }

  // Winsorize extreme repulsive wall values so the potential well remains legible
  const sortedU = [...us].sort((a, b) => a - b);
  const loU = sortedU[Math.floor(sortedU.length * 0.02)];
  const hiU = sortedU[Math.floor(sortedU.length * 0.92)];
  const spanU = Math.max(1e-6, hiU - loU);
  const clampLoU = loU - spanU * 0.1;
  const clampHiU = hiU + spanU * 0.25;

  const sortedF = [...fs].sort((a, b) => a - b);
  const loF = sortedF[Math.floor(sortedF.length * 0.05)];
  const hiF = sortedF[Math.floor(sortedF.length * 0.95)];
  const spanF = Math.max(1e-6, hiF - loF);

  const plotW = SVG_W - 2 * PAD_X;
  const plotH = SVG_H - 2 * PAD_Y;

  const toX = (i: number) => PAD_X + (i / (samples - 1)) * plotW;
  const toYU = (v: number) => {
    const norm = Math.max(0, Math.min(1, (v - clampLoU) / (clampHiU - clampLoU)));
    return PAD_Y + (1 - norm) * plotH;
  };
  const toYF = (v: number) => {
    const norm = Math.max(0, Math.min(1, (v - loF) / spanF));
    return PAD_Y + (1 - norm) * plotH;
  };

  const uPath = rs
    .map((_, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toYU(us[i]).toFixed(1)}`)
    .join(' ');
  const fPath = rs
    .map((_, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toYF(fs[i]).toFixed(1)}`)
    .join(' ');

  const zeroY =
    clampLoU <= 0 && clampHiU >= 0 ? toYU(0) : null;

  return {
    valid: true,
    uPath,
    fPath,
    zeroY,
    uMin: clampLoU,
    uMax: clampHiU,
  };
}

export function PhysicsBuilderPane({
  initialSpec,
  activePresetId = null,
  compiling = false,
  onCompileAndSwap,
  onSteer,
}: PhysicsBuilderPaneProps) {
  const [potentialExpr, setPotentialExpr] = useState(
    initialSpec?.potential_expr ?? POTENTIAL_CHIPS[0].expr
  );
  const [rMin, setRMin] = useState(initialSpec?.r_min ?? POTENTIAL_CHIPS[0].rMin);
  const [rMax, setRMax] = useState(initialSpec?.r_max ?? POTENTIAL_CHIPS[0].rMax);
  const [temperature, setTemperature] = useState(initialSpec?.temperature ?? 300);
  const [efieldZ, setEfieldZ] = useState(initialSpec?.efield_z ?? 0);
  const [gcmcMu, setGcmcMu] = useState(initialSpec?.gcmc_mu ?? -0.2);
  const [strainRate, setStrainRate] = useState(initialSpec?.strain_rate ?? 0);

  useEffect(() => {
    if (initialSpec?.temperature !== undefined) {
      setTemperature(initialSpec.temperature);
    }
    if (initialSpec?.efield_z !== undefined) {
      setEfieldZ(initialSpec.efield_z);
    }
    if (initialSpec?.gcmc_mu !== undefined) {
      setGcmcMu(initialSpec.gcmc_mu);
    }
  }, [initialSpec?.temperature, initialSpec?.efield_z, initialSpec?.gcmc_mu]);

  const curve = useMemo(
    () => buildPotentialCurve(potentialExpr, rMin, rMax),
    [potentialExpr, rMin, rMax]
  );

  const handleSelectChip = useCallback((chip: PotentialChip) => {
    setPotentialExpr(chip.expr);
    setRMin(chip.rMin);
    setRMax(chip.rMax);
  }, []);

  const handleCompile = useCallback(() => {
    if (!curve.valid || compiling) return;
    void onCompileAndSwap({
      potential_expr: potentialExpr,
      r_min: rMin,
      r_max: rMax,
      n_points: 256,
      temperature,
      efield_z: efieldZ,
      gcmc_mu: gcmcMu,
      strain_rate: strainRate,
      seed_preset_id: activePresetId ?? undefined,
    });
  }, [
    activePresetId,
    compiling,
    curve.valid,
    efieldZ,
    gcmcMu,
    onCompileAndSwap,
    potentialExpr,
    rMax,
    rMin,
    strainRate,
    temperature,
  ]);

  const emitSteer = useCallback(
    (patch: SteerRequest) => {
      if (onSteer) {
        void onSteer(patch);
      }
    },
    [onSteer]
  );

  return (
    <div className="bmm-pane bmm-physics-builder">
      <section className="bmm-pb-section">
        <div className="bmm-pb-section-title">Radial Potential U(r)</div>
        <div className="bmm-chip-row" role="group" aria-label="Potential Presets">
          {POTENTIAL_CHIPS.map((chip) => (
            <button
              key={chip.id}
              type="button"
              className={`bmm-chip${potentialExpr === chip.expr ? ' is-selected' : ''}`}
              onClick={() => handleSelectChip(chip)}
            >
              {chip.label}
            </button>
          ))}
        </div>

        <label className="bmm-field">
          <span className="bmm-field-label">Expression U(r)</span>
          <input
            type="text"
            className="bmm-input bmm-input-mono"
            aria-label="Potential Expression U(r)"
            value={potentialExpr}
            onChange={(e) => setPotentialExpr(e.target.value)}
            spellCheck={false}
          />
        </label>

        <div className="bmm-pb-bounds">
          <label className="bmm-field">
            <span className="bmm-field-label">r_min (nm)</span>
            <input
              type="number"
              step="0.01"
              min="0.01"
              className="bmm-input tabular-nums"
              value={rMin}
              onChange={(e) => setRMin(Number(e.target.value))}
            />
          </label>
          <label className="bmm-field">
            <span className="bmm-field-label">r_max (nm)</span>
            <input
              type="number"
              step="0.05"
              min="0.05"
              className="bmm-input tabular-nums"
              value={rMax}
              onChange={(e) => setRMax(Number(e.target.value))}
            />
          </label>
        </div>
      </section>

      <section className="bmm-pb-section">
        <div className="bmm-pb-curve-header">
          <span className="bmm-pb-section-title">Potential &amp; Force Curve</span>
          <div className="bmm-pb-legend">
            <span className="bmm-legend-u">U(r)</span>
            <span className="bmm-legend-f">-dU/dr</span>
          </div>
        </div>

        <div className="bmm-pb-svg-wrap">
          {curve.valid ? (
            <svg
              viewBox={`0 0 ${SVG_W} ${SVG_H}`}
              className="bmm-pb-svg"
              role="img"
              aria-label="Radial Potential U(r) and Force -dU/dr plot"
            >
              {curve.zeroY !== null && (
                <line
                  x1={PAD_X}
                  x2={SVG_W - PAD_X}
                  y1={curve.zeroY}
                  y2={curve.zeroY}
                  className="bmm-svg-zero"
                />
              )}
              <path d={curve.fPath} className="bmm-svg-force" fill="none" />
              <path d={curve.uPath} className="bmm-svg-potential" fill="none" />
            </svg>
          ) : (
            <div className="bmm-pb-error">
              Invalid U(r) expression or bounds (r_min &gt; 0, r_max &gt; r_min).
            </div>
          )}
        </div>
      </section>

      <section className="bmm-pb-section">
        <div className="bmm-pb-section-title">Live Simulation Steering</div>

        <label className="bmm-slider-field">
          <div className="bmm-slider-head">
            <span>Temperature</span>
            <span className="tabular-nums">{temperature.toFixed(0)} K</span>
          </div>
          <input
            type="range"
            min={10}
            max={2000}
            step={10}
            value={temperature}
            onChange={(e) => {
              const v = Number(e.target.value);
              setTemperature(v);
              emitSteer({ temperature: v });
            }}
          />
        </label>

        <label className="bmm-slider-field">
          <div className="bmm-slider-head">
            <span>E-Field Z</span>
            <span className="tabular-nums">{efieldZ.toFixed(2)} V/nm</span>
          </div>
          <input
            type="range"
            min={-5}
            max={5}
            step={0.1}
            value={efieldZ}
            onChange={(e) => {
              const v = Number(e.target.value);
              setEfieldZ(v);
              emitSteer({ efield_z: v });
            }}
          />
        </label>

        <label className="bmm-slider-field">
          <div className="bmm-slider-head">
            <span>GCMC μ</span>
            <span className="tabular-nums">{gcmcMu.toFixed(2)}</span>
          </div>
          <input
            type="range"
            min={-3}
            max={3}
            step={0.05}
            value={gcmcMu}
            onChange={(e) => {
              const v = Number(e.target.value);
              setGcmcMu(v);
              emitSteer({ gcmc_mu: v });
            }}
          />
        </label>

        <label className="bmm-slider-field">
          <div className="bmm-slider-head">
            <span>Strain Rate</span>
            <span className="tabular-nums">{strainRate.toFixed(3)} /ps</span>
          </div>
          <input
            type="range"
            min={-0.05}
            max={0.1}
            step={0.005}
            value={strainRate}
            onChange={(e) => {
              const v = Number(e.target.value);
              setStrainRate(v);
              emitSteer({ strain_rate: v });
            }}
          />
        </label>
      </section>

      <button
        type="button"
        className="bmm-btn bmm-btn-primary bmm-pb-compile"
        disabled={!curve.valid || compiling}
        onClick={handleCompile}
      >
        {compiling ? 'Compiling Spline…' : 'Compile & Hot-Swap Spline'}
      </button>
    </div>
  );
}
