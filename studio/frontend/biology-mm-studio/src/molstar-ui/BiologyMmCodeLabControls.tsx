import type { ChangeEvent, KeyboardEvent } from 'react';
import { CollapsableControls, type CollapsableProps, type CollapsableState } from '@molstar/plugin-ui/controls/collapsable';
import { Button, ControlRow } from '@molstar/plugin-ui/controls/common';
import { getOrCreateBiologyMmStudioStore, type BiologyMmStudioStore } from '../studio-store.js';

interface CodeLabControlsState extends CollapsableState {
  selectedTemplateId: string;
}

export class BiologyMmCodeLabControls extends CollapsableControls<CollapsableProps, CodeLabControlsState> {
  private unsubStore: (() => void) | undefined;

  protected defaultState(): CodeLabControlsState {
    return {
      isCollapsed: false,
      header: 'Biology-MM Code Lab',
      description: 'Python SDK',
      selectedTemplateId: 'biomolecular-complex',
    };
  }

  private get store(): BiologyMmStudioStore | undefined {
    return this.plugin ? getOrCreateBiologyMmStudioStore(this.plugin) : undefined;
  }

  componentDidMount(): void {
    const store = this.store;
    if (!store) return;

    this.unsubStore = store.subscribe(() => {
      const snap = store.getSnapshot();
      if (snap.activePresetId && snap.activePresetId !== this.state.selectedTemplateId) {
        this.setState({ selectedTemplateId: snap.activePresetId });
      } else {
        this.forceUpdate();
      }
    });
  }

  componentWillUnmount(): void {
    super.componentWillUnmount();
    this.unsubStore?.();
    this.unsubStore = undefined;
  }

  private handleTemplateChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const store = this.store;
    if (!store) return;
    const id = e.target.value;
    this.setState({ selectedTemplateId: id });
    const preset = store.getSnapshot().presets.find((p) => p.id === id);
    if (preset) {
      store.setCodeText(preset.code_template);
    }
  };

  private handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    const store = this.store;
    if (!store) return;

    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      void store.runCode();
      return;
    }

    if (e.key === 'Tab') {
      e.preventDefault();
      const target = e.currentTarget;
      const start = target.selectionStart;
      const end = target.selectionEnd;
      const value = target.value;
      const updated = `${value.slice(0, start)}  ${value.slice(end)}`;
      store.setCodeText(updated);
      queueMicrotask(() => {
        target.selectionStart = target.selectionEnd = start + 2;
      });
    }
  };

  protected renderControls() {
    const store = this.store;
    if (!store) return null;
    const snap = store.getSnapshot();

    return (
      <div className="bmm-msp-panel">
        <ControlRow
          label="Template"
          control={
            <select
              className="msp-form-control"
              aria-label="Python Script Template"
              value={this.state.selectedTemplateId}
              onChange={this.handleTemplateChange}
              disabled={snap.presets.length === 0}
            >
              {snap.presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.title}
                </option>
              ))}
            </select>
          }
        />

        <div className="bmm-msp-btn-grid-2">
          <Button
            inline
            onClick={() => store.insertActiveStructureSnippet()}
            title="Insert active_structure snippet targeting the structure loaded in Mol*"
          >
            + active_structure
          </Button>
          <Button
            inline
            onClick={() => store.setWideCodeEditorOpen(!snap.wideCodeEditorOpen)}
            title="Open wide side-by-side Python editor alongside Mol* 3D canvas"
          >
            {snap.wideCodeEditorOpen ? 'Close Wide Editor' : 'Expand Wide Editor'}
          </Button>
        </div>

        <textarea
          className="bmm-msp-code-textarea"
          aria-label="Biology-MM Python Simulation Code"
          spellCheck={false}
          rows={9}
          value={snap.codeText}
          onChange={(e: ChangeEvent<HTMLTextAreaElement>) => store.setCodeText(e.target.value)}
          onKeyDown={this.handleKeyDown}
          placeholder="Write Biology-MM Python code here (Cmd/Ctrl+Enter to run)..."
        />

        <Button
          commit="on"
          onClick={() => void store.runCode()}
          disabled={snap.codeRunning || !snap.codeText.trim()}
        >
          {snap.codeRunning ? 'Running Python...' : 'Run Python Simulation (⌘/Ctrl+Enter)'}
        </Button>

        {snap.codeRunResult && (
          <div className={`bmm-msp-console ${snap.codeRunResult.ok ? 'is-ok' : 'is-err'}`}>
            <div className="bmm-msp-console-hdr">
              <span>{snap.codeRunResult.ok ? 'OK' : 'ERROR'}</span>
              <span>{snap.codeRunResult.elapsed_ms.toFixed(1)} ms</span>
            </div>
            {snap.codeRunResult.stdout && (
              <pre className="bmm-msp-console-pre">{snap.codeRunResult.stdout}</pre>
            )}
            {snap.codeRunResult.stderr && (
              <pre className="bmm-msp-console-pre is-stderr">{snap.codeRunResult.stderr}</pre>
            )}
            {snap.codeRunResult.error && (
              <pre className="bmm-msp-console-pre is-stderr">{snap.codeRunResult.error}</pre>
            )}
          </div>
        )}
      </div>
    );
  }
}
