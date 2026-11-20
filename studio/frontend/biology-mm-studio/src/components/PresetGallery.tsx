import type { PresetInfo } from '../types.js';

export interface PresetGalleryProps {
  presets: readonly PresetInfo[];
  activePresetId: string | null;
  loading?: boolean;
  onLaunchPreset: (preset: PresetInfo) => void;
  onEditInCodeLab: (preset: PresetInfo) => void;
}

const MODE_LABELS: Record<PresetInfo['mode'], string> = {
  metal_verlet: 'Metal Verlet',
  cnt_tersoff: 'CNT Tersoff',
  custom_spline: 'Custom Spline',
  nanopore_gcmc_nemd: 'GCMC + NEMD',
};

export function PresetGallery({
  presets,
  activePresetId,
  loading = false,
  onLaunchPreset,
  onEditInCodeLab,
}: PresetGalleryProps) {
  if (loading && presets.length === 0) {
    return (
      <div className="bmm-pane bmm-preset-gallery">
        <div className="bmm-pane-empty">Loading scientific presets…</div>
      </div>
    );
  }

  if (presets.length === 0) {
    return (
      <div className="bmm-pane bmm-preset-gallery">
        <div className="bmm-pane-empty">
          No presets available. Connect to the studio backend to load simulations.
        </div>
      </div>
    );
  }

  return (
    <div className="bmm-pane bmm-preset-gallery" role="list" aria-label="Simulation Presets">
      {presets.map((preset) => {
        const isActive = preset.id === activePresetId;
        return (
          <article
            key={preset.id}
            role="listitem"
            className={`bmm-preset-card${isActive ? ' is-active' : ''}`}
            data-preset-id={preset.id}
          >
            <header className="bmm-preset-header">
              <span className="bmm-badge bmm-badge-category">{preset.category}</span>
              <span className="bmm-badge bmm-badge-mode">
                {MODE_LABELS[preset.mode] ?? preset.mode}
              </span>
            </header>

            <h3 className="bmm-preset-title">{preset.title}</h3>
            <p className="bmm-preset-desc">{preset.description}</p>

            <div className="bmm-preset-meta tabular-nums">
              <span className="bmm-meta-item">
                <strong>{preset.n_atoms.toLocaleString()}</strong> atoms
              </span>
              <span className="bmm-meta-item">
                <strong>{preset.temperature.toFixed(0)}</strong> K
              </span>
              <span className="bmm-meta-item">{preset.units}</span>
            </div>

            {preset.tags.length > 0 && (
              <div className="bmm-preset-tags">
                {preset.tags.map((tag) => (
                  <span key={tag} className="bmm-tag">
                    {tag}
                  </span>
                ))}
              </div>
            )}

            <footer className="bmm-preset-actions">
              <button
                type="button"
                className="bmm-btn bmm-btn-primary"
                onClick={() => onLaunchPreset(preset)}
              >
                {isActive ? 'Relaunch Simulation' : 'Launch Simulation'}
              </button>
              <button
                type="button"
                className="bmm-btn bmm-btn-secondary"
                onClick={() => onEditInCodeLab(preset)}
              >
                Edit in Code Lab
              </button>
            </footer>
          </article>
        );
      })}
    </div>
  );
}
