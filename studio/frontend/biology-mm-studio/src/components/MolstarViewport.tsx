import { useEffect, useMemo } from 'react';
import '@molstar/plugin-ui/skin/light.scss';
import { Plugin } from '@molstar/plugin-ui/plugin';
import { useCreatePluginUIViewModel } from '@molstar/plugin-extension/hooks/use-ui-view-model';
import { createViewerSpec } from '@molstar/viewer/plugin-spec';
import { ViewerAutoPreset } from '@molstar/viewer/presets';
import { PluginSpec } from '@molstar/plugin/spec';
import { PluginConfig } from '@molstar/plugin/config';
import { BiologyMmLiveBehavior } from '@molstar/biology-mm-extension';
import { BiologyMmUIBehavior } from '../molstar-ui/behavior.js';
import { getOrCreateBiologyMmStudioStore, type BiologyMmStudioStore } from '../studio-store.js';

export interface MolstarViewportProps {
  onStoreReady?: (store: BiologyMmStudioStore | null) => void;
}

function hasWebGL2(): boolean {
  if (typeof document === 'undefined') return false;
  try {
    const canvas = document.createElement('canvas');
    return Boolean(canvas.getContext('webgl2'));
  } catch {
    return false;
  }
}

function createStudioViewerSpec() {
  const base = createViewerSpec({
    layoutIsExpanded: true,
    layoutShowControls: true,
    layoutShowSequence: true,
    layoutShowLog: true,
    layoutShowLeftPanel: true,
    layoutShowRemoteState: true,
    collapseLeftPanel: false,
    collapseRightPanel: false,
    volumeStreamingServer: 'https://www.ebi.ac.uk/pdbe/densities',
    volumeStreamingDisabled: false,
  });

  return {
    ...base,
    config: [
      ...(base.config ?? []),
      [PluginConfig.Download.DefaultPdbProvider, 'rcsb'],
    ],
    behaviors: [
      ...base.behaviors,
      PluginSpec.Behavior(BiologyMmLiveBehavior),
      PluginSpec.Behavior(BiologyMmUIBehavior),
    ],
  };
}

function MolstarHost({
  onStoreReady,
}: {
  onStoreReady?: (store: BiologyMmStudioStore | null) => void;
}) {
  const spec = useMemo(() => createStudioViewerSpec(), []);
  const model = useCreatePluginUIViewModel({ spec });

  useEffect(() => {
    let cancelled = false;
    const plugin = model.plugin;
    const store = getOrCreateBiologyMmStudioStore(plugin);

    const attach = async () => {
      try {
        await plugin.initialized;
      } catch {
        // Ignore initialization errors in degraded environments
      }
      if (cancelled) return;
      try {
        plugin.builders.structure.representation.registerPreset(ViewerAutoPreset);
      } catch {
        // Ignore if already registered
      }
      void store.initializeAndBoot();
      onStoreReady?.(store);
    };

    void attach();

    return () => {
      cancelled = true;
      onStoreReady?.(null);
      store.dispose();
    };
  }, [model, onStoreReady]);

  return <Plugin plugin={model.plugin} />;
}

export function MolstarViewport({ onStoreReady }: MolstarViewportProps) {
  const webglSupported = useMemo(() => hasWebGL2(), []);

  useEffect(() => {
    if (!webglSupported) {
      onStoreReady?.(null);
    }
  }, [webglSupported, onStoreReady]);

  return (
    <div className="bmm-molstar-host" data-testid="molstar-viewport">
      {webglSupported ? (
        <MolstarHost onStoreReady={onStoreReady} />
      ) : (
        <div className="bmm-molstar-fallback" role="status">
          <div className="fallback-title">Mol* 3D Viewport (WebGL2 Unavailable)</div>
          <div className="fallback-sub">
            Biology-MM Studio requires WebGL2 for interactive 3D rendering in the browser.
          </div>
        </div>
      )}
    </div>
  );
}
