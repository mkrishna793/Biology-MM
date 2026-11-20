import { useCallback, type KeyboardEvent } from 'react';
import type { CodeRunResult, PresetInfo } from '../types.js';

export interface CodeLabPaneProps {
  code: string;
  onCodeChange: (code: string) => void;
  activePreset: PresetInfo | null;
  running: boolean;
  lastResult: CodeRunResult | null;
  onRunCode: (code: string) => Promise<void> | void;
}

const DEFAULT_TEMPLATE = `import biology_mm as bmm

# Build a custom molecular mechanics system
system = bmm.System.argon_fcc(nx=4, ny=4, nz=4, temperature=120.0)
sim = bmm.Simulation(system, dt=0.002)
sim.step(10)
`;

export function CodeLabPane({
  code,
  onCodeChange,
  activePreset,
  running,
  lastResult,
  onRunCode,
}: CodeLabPaneProps) {
  const handleRun = useCallback(() => {
    if (!running && code.trim().length > 0) {
      void onRunCode(code);
    }
  }, [code, onRunCode, running]);

  const handleReset = useCallback(() => {
    const template = activePreset?.code_template?.trim()
      ? activePreset.code_template
      : DEFAULT_TEMPLATE;
    onCodeChange(template);
  }, [activePreset, onCodeChange]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        handleRun();
        return;
      }

      if (e.key === 'Tab') {
        e.preventDefault();
        const target = e.currentTarget;
        const start = target.selectionStart;
        const end = target.selectionEnd;
        const updated = `${code.slice(0, start)}  ${code.slice(end)}`;
        onCodeChange(updated);
        requestAnimationFrame(() => {
          target.selectionStart = start + 2;
          target.selectionEnd = start + 2;
        });
      }
    },
    [code, handleRun, onCodeChange]
  );

  return (
    <div className="bmm-pane bmm-code-lab">
      <div className="bmm-codelab-header">
        <div className="bmm-codelab-sync">
          <span className="bmm-label">Template:</span>
          <span className="bmm-badge bmm-badge-preset">
            {activePreset ? activePreset.title : 'Custom Python'}
          </span>
        </div>
        <div className="bmm-codelab-controls">
          <button
            type="button"
            className="bmm-btn bmm-btn-ghost"
            onClick={handleReset}
            disabled={running}
            title="Reset editor to active preset template"
          >
            Reset Template
          </button>
          <button
            type="button"
            className="bmm-btn bmm-btn-primary"
            onClick={handleRun}
            disabled={running || code.trim().length === 0}
            title="Run Python simulation (Cmd/Ctrl + Enter)"
          >
            {running ? 'Running…' : 'Run'}
          </button>
        </div>
      </div>

      <div className="bmm-codelab-editor-wrap">
        <textarea
          className="bmm-code-editor"
          aria-label="Python Simulation Editor"
          spellCheck={false}
          value={code}
          onChange={(e) => onCodeChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Write biology_mm Python code here… (Cmd/Ctrl+Enter to run)"
        />
        <div className="bmm-codelab-hint">
          <span>Tab: 2 spaces</span>
          <span>Cmd/Ctrl + Enter: Run</span>
        </div>
      </div>

      <section className="bmm-codelab-console" aria-label="Execution Output">
        <header className="bmm-console-header">
          <span>Console Output</span>
          {lastResult && (
            <span
              className={`bmm-pill tabular-nums ${
                lastResult.ok ? 'is-ok' : 'is-error'
              }`}
            >
              {lastResult.ok ? 'OK' : 'ERR'} · {lastResult.elapsed_ms.toFixed(1)} ms
            </span>
          )}
        </header>

        {!lastResult ? (
          <div className="bmm-console-empty">
            Run a script to execute in the sandboxed biology_mm runtime.
          </div>
        ) : (
          <div className="bmm-console-body">
            {lastResult.stdout && (
              <pre className="bmm-console-stdout">{lastResult.stdout}</pre>
            )}
            {lastResult.stderr && (
              <pre className="bmm-console-stderr">{lastResult.stderr}</pre>
            )}
            {lastResult.error && (
              <pre className="bmm-console-error">{lastResult.error}</pre>
            )}
            {!lastResult.stdout && !lastResult.stderr && !lastResult.error && (
              <div className="bmm-console-empty">
                Execution completed with no output.
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
