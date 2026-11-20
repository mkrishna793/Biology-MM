import type { AtomInspection } from '../types.js';

export interface AtomInspectorProps {
  atom: AtomInspection | null;
  onClose: () => void;
}

function formatVec3(vec: readonly [number, number, number], digits: number): string {
  return `(${vec[0].toFixed(digits)}, ${vec[1].toFixed(digits)}, ${vec[2].toFixed(digits)})`;
}

function formatTernary(val: number): { label: string; cls: string } {
  if (val < -0.33) return { label: '-1', cls: 'ternary-neg' };
  if (val > 0.33) return { label: '+1', cls: 'ternary-pos' };
  return { label: '0', cls: 'ternary-zero' };
}

export function AtomInspector({ atom, onClose }: AtomInspectorProps) {
  if (!atom) return null;

  const ternary = formatTernary(atom.ternary_state);

  return (
    <aside className="atom-inspector glass-panel" aria-label="Atom Inspector">
      <div className="atom-inspector-header">
        <div className="atom-inspector-title">
          <span className="atom-element-badge">{atom.element}</span>
          <span className="atom-index tabular-nums">#{atom.index}</span>
          <span className={`ternary-badge ${ternary.cls} tabular-nums`} title="Ternary State">
            {ternary.label}
          </span>
        </div>
        <button
          type="button"
          className="icon-btn"
          onClick={onClose}
          aria-label="Close atom inspector"
          title="Close"
        >
          ✕
        </button>
      </div>

      <div className="atom-inspector-grid tabular-nums">
        <div className="inspector-row">
          <span className="inspector-label">Position (nm)</span>
          <span className="inspector-value">{formatVec3(atom.position_nm, 3)}</span>
        </div>
        <div className="inspector-row">
          <span className="inspector-label">Position (Å)</span>
          <span className="inspector-value">{formatVec3(atom.position_ang, 2)}</span>
        </div>
        <div className="inspector-row">
          <span className="inspector-label">Velocity (nm/ps)</span>
          <span className="inspector-value">{formatVec3(atom.velocity_nm_ps, 3)}</span>
        </div>
        <div className="inspector-row">
          <span className="inspector-label">Force</span>
          <span className="inspector-value">{formatVec3(atom.force, 2)}</span>
        </div>
        <div className="inspector-row">
          <span className="inspector-label">|Force|</span>
          <span className="inspector-value">{atom.force_magnitude.toFixed(3)}</span>
        </div>
        <div className="inspector-row">
          <span className="inspector-label">Local Temp</span>
          <span className="inspector-value">{atom.local_temp_K.toFixed(1)} K</span>
        </div>
        <div className="inspector-row">
          <span className="inspector-label">Charge</span>
          <span className="inspector-value">{atom.charge.toFixed(3)} e</span>
        </div>
        <div className="inspector-row">
          <span className="inspector-label">Bond Order</span>
          <span className="inspector-value">{atom.bond_order.toFixed(3)}</span>
        </div>
        <div className="inspector-row">
          <span className="inspector-label">Atomic Stress</span>
          <span className="inspector-value">{atom.atomic_stress.toFixed(3)}</span>
        </div>
      </div>
    </aside>
  );
}
