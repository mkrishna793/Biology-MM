import {
  Component,
  useCallback,
  useEffect,
  useState,
  useSyncExternalStore,
  type ChangeEvent,
  type ErrorInfo,
  type KeyboardEvent,
  type ReactNode,
} from 'react';
import { MolstarViewport } from './components/MolstarViewport.js';
import type { BiologyMmStudioStore } from './studio-store.js';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  message: string;
}

export class StudioErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  override state: ErrorBoundaryState = {
    hasError: false,
    message: '',
  };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      message: error.message || 'Unexpected rendering error in Biology-MM Studio.',
    };
  }

  override componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('[Biology-MM Studio] Uncaught UI error:', error, info);
  }

  private handleReset = (): void => {
    this.setState({ hasError: false, message: '' });
  };

  override render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="bmm-studio-root" role="alert">
          <div className="bmm-error-banner">
            <span>Studio Error: {this.state.message}</span>
            <button type="button" onClick={this.handleReset}>
              Recover Studio
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function WideCodeEditorDrawer({ store }: { store: BiologyMmStudioStore }) {
  const snap = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);

  useEffect(() => {
    if (!snap.wideCodeEditorOpen) return;
    const onKeyDown = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') {
        store.setWideCodeEditorOpen(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [snap.wideCodeEditorOpen, store]);

  if (!snap.wideCodeEditorOpen) return null;

  const handleTextareaKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
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
      const next = `${target.value.slice(0, start)}  ${target.value.slice(end)}`;
      store.setCodeText(next);
      queueMicrotask(() => {
        target.selectionStart = target.selectionEnd = start + 2;
      });
    }
  };

  return (
    <aside className="bmm-wide-editor-drawer" aria-label="Wide Python Simulation Editor">
      <div className="bmm-wide-editor-hdr">
        <div className="bmm-wide-editor-title">
          <strong>Biology-MM Python Lab</strong>
          <span>{snap.sessionTitle}</span>
        </div>
        <div className="bmm-wide-editor-actions">
          <select
            className="msp-form-control"
            aria-label="Wide Editor Template"
            value={snap.activePresetId ?? ''}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => {
              const preset = snap.presets.find((p) => p.id === e.target.value);
              if (preset) store.setCodeText(preset.code_template);
            }}
          >
            <option value="" disabled>
              Select Template...
            </option>
            {snap.presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.title}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="bmm-wide-btn"
            onClick={() => store.insertActiveStructureSnippet()}
          >
            + active_structure
          </button>
          <button
            type="button"
            className="bmm-wide-btn is-primary"
            disabled={snap.codeRunning || !snap.codeText.trim()}
            onClick={() => void store.runCode()}
          >
            {snap.codeRunning ? 'Running...' : 'Run (⌘/Ctrl+Enter)'}
          </button>
          <button
            type="button"
            className="bmm-wide-btn"
            onClick={() => store.setWideCodeEditorOpen(false)}
          >
            Close (Esc)
          </button>
        </div>
      </div>

      <textarea
        className="bmm-wide-textarea"
        aria-label="Wide Python Editor"
        spellCheck={false}
        value={snap.codeText}
        onChange={(e: ChangeEvent<HTMLTextAreaElement>) => store.setCodeText(e.target.value)}
        onKeyDown={handleTextareaKeyDown}
        placeholder="Write Biology-MM Python code here (Cmd/Ctrl+Enter to execute)..."
      />

      {snap.codeRunResult && (
        <div className={`bmm-wide-console ${snap.codeRunResult.ok ? 'is-ok' : 'is-err'}`}>
          <div className="bmm-wide-console-hdr">
            <span>{snap.codeRunResult.ok ? 'OK' : 'ERROR'}</span>
            <span>{snap.codeRunResult.elapsed_ms.toFixed(1)} ms</span>
          </div>
          {snap.codeRunResult.stdout && <pre>{snap.codeRunResult.stdout}</pre>}
          {snap.codeRunResult.stderr && <pre className="is-stderr">{snap.codeRunResult.stderr}</pre>}
          {snap.codeRunResult.error && <pre className="is-stderr">{snap.codeRunResult.error}</pre>}
        </div>
      )}
    </aside>
  );
}

export function App() {
  const [store, setStore] = useState<BiologyMmStudioStore | null>(null);
  const handleStoreReady = useCallback((s: BiologyMmStudioStore | null) => {
    setStore(s);
  }, []);

  return (
    <StudioErrorBoundary>
      <div className="bmm-studio-root">
        <MolstarViewport onStoreReady={handleStoreReady} />
        {store && <WideCodeEditorDrawer store={store} />}
      </div>
    </StudioErrorBoundary>
  );
}

export default App;
