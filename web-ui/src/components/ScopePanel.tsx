import { useCallback, useEffect, useRef, useState } from 'react';

const SCOPES = ['scope_histogram', 'scope_waveform', 'scope_vectorscope', 'scope_parade'] as const;
type ScopeName = (typeof SCOPES)[number];

const SCOPE_LABELS: Record<ScopeName, string> = {
  scope_histogram: 'Histogram',
  scope_waveform: 'Waveform',
  scope_vectorscope: 'Vectorscope',
  scope_parade: 'Parade',
};

const LS_KEY = 'rasmcore-scope-selected';
const LS_COLLAPSED_KEY = 'rasmcore-scope-collapsed';

function loadSelected(): ScopeName {
  try {
    const v = localStorage.getItem(LS_KEY);
    if (v && SCOPES.includes(v as ScopeName)) return v as ScopeName;
  } catch { /* ignore */ }
  return 'scope_histogram';
}

function loadCollapsed(): boolean {
  try { return localStorage.getItem(LS_COLLAPSED_KEY) === '1'; } catch { return false; }
}

interface Props {
  scopeCanvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  hasImage: boolean;
  onScopeChange: (scope: ScopeName) => void;
}

export default function ScopePanel({ scopeCanvasRef, hasImage, onScopeChange }: Props) {
  const [selected, setSelected] = useState<ScopeName>(loadSelected);
  const [collapsed, setCollapsed] = useState(loadCollapsed);

  const handleSelect = useCallback((scope: ScopeName) => {
    setSelected(scope);
    try { localStorage.setItem(LS_KEY, scope); } catch { /* ignore */ }
    onScopeChange(scope);
  }, [onScopeChange]);

  const toggleCollapsed = useCallback(() => {
    setCollapsed((c) => {
      const next = !c;
      try { localStorage.setItem(LS_COLLAPSED_KEY, next ? '1' : '0'); } catch { /* ignore */ }
      return next;
    });
  }, []);

  // Notify parent of initial scope on mount
  useEffect(() => {
    onScopeChange(selected);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="scope-panel">
      <div className="scope-header" onClick={toggleCollapsed}>
        <span className="scope-header-label">
          {collapsed ? '\u25B6' : '\u25BC'} {SCOPE_LABELS[selected]}
        </span>
      </div>
      {!collapsed && (
        <>
          <div className="scope-tabs">
            {SCOPES.map((s) => (
              <button
                key={s}
                className={'scope-tab' + (selected === s ? ' active' : '')}
                onClick={() => handleSelect(s)}
              >
                {SCOPE_LABELS[s]}
              </button>
            ))}
          </div>
          <div className="scope-canvas-wrap">
            {hasImage ? (
              <canvas ref={scopeCanvasRef} className="scope-canvas" />
            ) : (
              <div className="scope-empty">Load an image to view scopes</div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
