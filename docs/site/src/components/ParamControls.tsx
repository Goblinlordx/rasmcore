'use client';

import type { ParamDescriptor } from '@/lib/types';

interface ParamControlsProps {
  params: ParamDescriptor[];
  values: Record<string, number | boolean>;
  onChange: (values: Record<string, number | boolean>) => void;
}

function snakeToTitle(s: string) {
  return s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

export function ParamControls({ params, values, onChange }: ParamControlsProps) {
  if (!params || params.length === 0) return null;

  const update = (name: string, value: number | boolean) => {
    onChange({ ...values, [name]: value });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', margin: '1rem 0' }}>
      {params.map(p => {
        const val = values[p.name];

        if (p.type === 'bool') {
          return (
            <label key={p.name} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem' }}>
              <input
                type="checkbox"
                checked={val as boolean ?? false}
                onChange={e => update(p.name, e.target.checked)}
              />
              {p.description || snakeToTitle(p.name)}
            </label>
          );
        }

        const numVal = (val as number) ?? p.default ?? 0;
        const min = p.min ?? 0;
        const max = p.max ?? 1;
        const step = p.step ?? (max - min) / 100;

        return (
          <div key={p.name} style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', fontSize: '0.9rem' }}>
            <span style={{ minWidth: 120, color: 'var(--text-muted)' }}>
              {p.description || snakeToTitle(p.name)}
            </span>
            <input
              type="range"
              min={min}
              max={max}
              step={step}
              value={numVal}
              onChange={e => update(p.name, parseFloat(e.target.value))}
              style={{ flex: 1, accentColor: 'var(--link)' }}
            />
            <span style={{ minWidth: 50, textAlign: 'right', fontFamily: 'monospace', fontSize: '0.85rem' }}>
              {typeof numVal === 'number' ? numVal.toFixed(2) : String(numVal)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
