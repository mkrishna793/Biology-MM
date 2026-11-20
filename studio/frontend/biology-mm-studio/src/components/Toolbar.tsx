import type { PaletteName, PhysicsChannelName } from '../types.js';
import { STUDIO_PHYSICS_CHANNELS } from '../types.js';

export interface StudioOverlaysState {
  showBox: boolean;
  showBonds: boolean;
  showForces: boolean;
  showVolume: boolean;
}

export interface ToolbarProps {
  sessionId: string | null;
  playing: boolean;
  onTogglePlay: () => void;
  onStep: (n: number) => void;
  historySteps: number[];
  currentStep: number;
  isReversed: boolean;
  onScrub: (targetStep: number) => void;
  activeChannel: PhysicsChannelName;
  onChangeChannel: (channel: PhysicsChannelName) => void;
  activePalette: PaletteName;
  onChangePalette: (palette: PaletteName) => void;
  overlays: StudioOverlaysState;
  onToggleOverlay: (key: keyof StudioOverlaysState) => void;
  onExportMvsx: () => void;
  sidebarOpen?: boolean;
  onToggleSidebar?: () => void;
  busy?: boolean;
}

const CHANNEL_LABELS: Record<PhysicsChannelName, string> = {
  'force-magnitude': 'Force |F|',
  'kinetic-temp': 'Kinetic Temp (K)',
  'charge-potential': 'Charge Potential',
  'bond-order': 'Bond Order',
  'atomic-stress': 'Atomic Stress',
  'ternary-state': 'Ternary State',
};

const PALETTE_OPTIONS: ReadonlyArray<{ value: PaletteName; label: string }> = [
  { value: 'plasma', label: 'Plasma' },
  { value: 'thermal-div', label: 'Thermal Div' },
  { value: 'electrostatic-bwr', label: 'Electrostatic BWR' },
  { value: 'bond-viridis', label: 'Bond Viridis' },
  { value: 'stress-inferno', label: 'Stress Inferno' },
  { value: 'ternary-triad', label: 'Ternary Triad' },
];

export function Toolbar({
  sessionId,
  playing,
  onTogglePlay,
  onStep,
  historySteps,
  currentStep,
  isReversed,
  onScrub,
  activeChannel,
  onChangeChannel,
  activePalette,
  onChangePalette,
  overlays,
  onToggleOverlay,
  onExportMvsx,
  sidebarOpen,
  onToggleSidebar,
  busy = false,
}: ToolbarProps) {
  const hasSession = sessionId !== null;
  const maxIndex = Math.max(0, historySteps.length - 1);
  const currentIndex = (() => {
    if (historySteps.length === 0) return 0;
    const exact = historySteps.indexOf(currentStep);
    if (exact >= 0) return exact;
    return maxIndex;
  })();

  return (
    <header className="studio-toolbar" role="toolbar" aria-label="Simulation Controls">
      <div className="toolbar-group toolbar-brand">
        {onToggleSidebar && (
          <button
            type="button"
            className="toolbar-btn icon-only"
            onClick={onToggleSidebar}
            aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
            title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            {sidebarOpen ? '◧' : '☰'}
          </button>
        )}
        <span className="brand-title">Biology-MM Studio</span>
      </div>

      <div className="toolbar-group toolbar-transport">
        <button
          type="button"
          className={`toolbar-btn primary ${playing ? 'is-playing' : ''}`}
          onClick={onTogglePlay}
          disabled={!hasSession || busy}
          aria-label={playing ? 'Pause simulation' : 'Play simulation'}
        >
          {playing ? '⏸ Pause' : '▶ Play'}
        </button>
        <button
          type="button"
          className="toolbar-btn"
          onClick={() => onStep(1)}
          disabled={!hasSession || playing || busy}
          title="Advance 1 step"
        >
          +1
        </button>
        <button
          type="button"
          className="toolbar-btn"
          onClick={() => onStep(10)}
          disabled={!hasSession || playing || busy}
          title="Advance 10 steps"
        >
          +10
        </button>
      </div>

      <div className="toolbar-group toolbar-scrubber">
        <label className="scrubber-label" htmlFor="time-scrubber">
          <span>Time</span>
          <span className="tabular-nums">Step {currentStep}</span>
          {isReversed && (
            <span className="rev-badge" title="Scrubbed backward in reversible history">
              REV
            </span>
          )}
        </label>
        <input
          id="time-scrubber"
          type="range"
          className="scrubber-slider"
          min={0}
          max={maxIndex}
          step={1}
          value={currentIndex}
          disabled={!hasSession || historySteps.length <= 1}
          onChange={(e) => {
            const idx = Number(e.target.value);
            const target = historySteps[idx];
            if (target !== undefined) {
              onScrub(target);
            }
          }}
          aria-label="Reversible Time Scrubber"
        />
      </div>

      <div className="toolbar-group toolbar-selectors">
        <label className="toolbar-select-wrap">
          <span className="select-label">Channel</span>
          <select
            aria-label="Physics Channel"
            value={activeChannel}
            onChange={(e) => onChangeChannel(e.target.value as PhysicsChannelName)}
            disabled={!hasSession}
          >
            {STUDIO_PHYSICS_CHANNELS.map((ch) => (
              <option key={ch} value={ch}>
                {CHANNEL_LABELS[ch]}
              </option>
            ))}
          </select>
        </label>

        <label className="toolbar-select-wrap">
          <span className="select-label">Palette</span>
          <select
            aria-label="Color Palette"
            value={activePalette}
            onChange={(e) => onChangePalette(e.target.value as PaletteName)}
            disabled={!hasSession}
          >
            {PALETTE_OPTIONS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="toolbar-group toolbar-overlays" role="group" aria-label="Viewport Overlays">
        <button
          type="button"
          className={`toolbar-btn toggle ${overlays.showBox ? 'active' : ''}`}
          onClick={() => onToggleOverlay('showBox')}
          aria-pressed={overlays.showBox}
        >
          Box
        </button>
        <button
          type="button"
          className={`toolbar-btn toggle ${overlays.showBonds ? 'active' : ''}`}
          onClick={() => onToggleOverlay('showBonds')}
          aria-pressed={overlays.showBonds}
        >
          Bonds
        </button>
        <button
          type="button"
          className={`toolbar-btn toggle ${overlays.showForces ? 'active' : ''}`}
          onClick={() => onToggleOverlay('showForces')}
          aria-pressed={overlays.showForces}
        >
          Forces
        </button>
        <button
          type="button"
          className={`toolbar-btn toggle ${overlays.showVolume ? 'active' : ''}`}
          onClick={() => onToggleOverlay('showVolume')}
          aria-pressed={overlays.showVolume}
        >
          3D Field
        </button>
      </div>

      <div className="toolbar-group toolbar-export">
        <button
          type="button"
          className="toolbar-btn export-btn"
          onClick={onExportMvsx}
          disabled={!hasSession || busy}
          title="Export MolViewSpec .mvsx archive"
        >
          ↓ .mvsx
        </button>
      </div>
    </header>
  );
}
