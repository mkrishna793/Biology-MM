import { PluginBehavior } from '@molstar/plugin/behavior/behavior';
import { BiologyMmCodeLabControls } from './BiologyMmCodeLabControls.js';
import { BiologyMmEngineControls } from './BiologyMmEngineControls.js';
import { BiologyMmTelemetryControls } from './BiologyMmTelemetryControls.js';

export const BiologyMmUIBehavior = PluginBehavior.create<{}>({
  name: 'biology-mm-native-ui',
  category: 'misc',
  display: {
    name: 'Biology-MM Native Mol* UI',
    description: 'Registers Biology-MM Engine, Code Lab, and Telemetry inside Mol* Home and Structure Tools',
  },
  ctor: class extends PluginBehavior.Handler<{}> {
    register(): void {
      this.ctx.customImportControls.set('biology-mm-engine', BiologyMmEngineControls as any);
      this.ctx.customImportControls.set('biology-mm-code-lab', BiologyMmCodeLabControls as any);
      this.ctx.customStructureControls.set('biology-mm-telemetry', BiologyMmTelemetryControls as any);
    }

    update(): boolean {
      return false;
    }

    unregister(): void {
      this.ctx.customImportControls.delete('biology-mm-engine');
      this.ctx.customImportControls.delete('biology-mm-code-lab');
      this.ctx.customStructureControls.delete('biology-mm-telemetry');
    }
  },
  params: () => ({}),
});
