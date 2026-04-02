import type { UiParam } from '../types';

interface Props {
  param: UiParam;
  value: number | string | boolean;
  onChange: (value: number | string | boolean) => void;
}

function guardDrag(e: React.PointerEvent | React.MouseEvent) {
  e.stopPropagation();
}

export default function ParamControl({ param, value, onChange }: Props) {
  const p = param;

  if (p.type === 'color') {
    return (
      <div className="param-row">
        <label>{p.label}</label>
        <input
          type="color"
          value={(value as string) || '#808080'}
          style={{ width: '100%', height: 28, border: 'none', cursor: 'pointer' }}
          onPointerDown={guardDrag}
          onMouseDown={guardDrag}
          onInput={(e) => onChange(e.currentTarget.value)}
        />
      </div>
    );
  }

  if (p.type === 'toggle') {
    return (
      <div className="param-row">
        <label className="toggle-label">{p.label}</label>
        <input
          type="checkbox"
          className="toggle-switch"
          checked={!!value}
          onPointerDown={guardDrag}
          onMouseDown={guardDrag}
          onChange={(e) => onChange(e.target.checked)}
        />
      </div>
    );
  }

  if (p.type === 'text') {
    return (
      <div className="param-row">
        <label>{p.label}</label>
        <input
          type="text"
          className="text-input"
          value={(value as string) || ''}
          onPointerDown={guardDrag}
          onMouseDown={guardDrag}
          onChange={(e) => onChange(e.target.value)}
        />
      </div>
    );
  }

  if (p.type === 'spinner') {
    return (
      <div className="param-row">
        <label>
          {p.label} <span className="param-value">{String(value)}</span>
        </label>
        <input
          type="number"
          className="spinner-input"
          min={p.min}
          max={p.max}
          step={p.step}
          value={value as number}
          onPointerDown={guardDrag}
          onMouseDown={guardDrag}
          onInput={(e) => onChange(parseFloat(e.currentTarget.value))}
        />
      </div>
    );
  }

  if (p.type === 'log_slider') {
    const pMin = p.min || 0;
    const pMax = p.max || 100;
    const toSlider = (v: number) =>
      Math.round((Math.log(v - pMin + 1) / Math.log(pMax - pMin + 1)) * 1000);
    const fromSlider = (s: number) => Math.pow(pMax - pMin + 1, s / 1000) + pMin - 1;
    return (
      <div className="param-row">
        <label>
          {p.label} <span className="param-value">{String(value)}</span>
        </label>
        <input
          type="range"
          className="log-slider"
          min={0}
          max={1000}
          step={1}
          value={toSlider(value as number)}
          onPointerDown={guardDrag}
          onMouseDown={guardDrag}
          onInput={(e) => {
            const realVal = Math.round(fromSlider(parseFloat(e.currentTarget.value)) * 100) / 100;
            onChange(realVal);
          }}
        />
      </div>
    );
  }

  if (p.type === 'signed_slider') {
    const v = value as number;
    return (
      <div className="param-row">
        <label>
          {p.label} <span className="param-value">{(v >= 0 ? '+' : '') + v}</span>
        </label>
        <input
          type="range"
          className="signed-slider"
          min={p.min}
          max={p.max}
          step={p.step}
          value={v}
          onPointerDown={guardDrag}
          onMouseDown={guardDrag}
          onInput={(e) => onChange(parseFloat(e.currentTarget.value))}
        />
      </div>
    );
  }

  if (p.type === 'opacity') {
    return (
      <div className="param-row">
        <label>
          {p.label} <span className="param-value">{String(value)}</span>
        </label>
        <div className="opacity-wrap">
          <input
            type="range"
            className="opacity-slider"
            min={p.min || 0}
            max={p.max || 1}
            step={p.step || 0.01}
            value={value as number}
            onPointerDown={guardDrag}
            onMouseDown={guardDrag}
            onInput={(e) => onChange(parseFloat(e.currentTarget.value))}
          />
          <input
            type="number"
            className="opacity-number"
            min={p.min || 0}
            max={p.max || 1}
            step={p.step || 0.01}
            value={value as number}
            onPointerDown={guardDrag}
            onMouseDown={guardDrag}
            onInput={(e) => onChange(parseFloat(e.currentTarget.value))}
          />
        </div>
      </div>
    );
  }

  if (p.type === 'temperature') {
    return (
      <div className="param-row">
        <label>
          {p.label} <span className="param-value">{value}K</span>
        </label>
        <input
          type="range"
          className="temperature-slider"
          min={p.min}
          max={p.max}
          step={p.step}
          value={value as number}
          onPointerDown={guardDrag}
          onMouseDown={guardDrag}
          onInput={(e) => onChange(parseFloat(e.currentTarget.value))}
        />
      </div>
    );
  }

  if (p.type === 'enum') {
    return (
      <div className="param-row">
        <label>{p.label}</label>
        <select
          value={value as string}
          onPointerDown={guardDrag}
          onMouseDown={guardDrag}
          onChange={(e) => onChange(e.target.value)}
        >
          {p.options?.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>
    );
  }

  if (p.type === 'point') {
    return (
      <div className="param-row">
        <label>
          {p.label} <span className="param-value">{String(value)}</span>
        </label>
        <div className="canvas-control point-control">
          <input
            type="number"
            className="spinner-input"
            min={p.min}
            max={p.max}
            step={p.step}
            value={value as number}
            onPointerDown={guardDrag}
            onMouseDown={guardDrag}
            onInput={(e) => onChange(parseFloat(e.currentTarget.value))}
          />
          <span className="canvas-hint">click canvas to set</span>
        </div>
      </div>
    );
  }

  if (p.type === 'path') {
    return (
      <div className="param-row">
        <label>
          {p.label} <span className="param-value">{String(value)}</span>
        </label>
        <div className="canvas-control path-control">
          <input
            type="number"
            className="spinner-input"
            min={p.min}
            max={p.max}
            step={p.step}
            value={value as number}
            onPointerDown={guardDrag}
            onMouseDown={guardDrag}
            onInput={(e) => onChange(parseFloat(e.currentTarget.value))}
          />
          <span className="canvas-hint">draw on canvas</span>
        </div>
      </div>
    );
  }

  if (p.type === 'box_select') {
    return (
      <div className="param-row">
        <label>
          {p.label} <span className="param-value">{String(value)}</span>
        </label>
        <div className="canvas-control box-control">
          <input
            type="number"
            className="spinner-input"
            min={p.min}
            max={p.max}
            step={p.step}
            value={value as number}
            onPointerDown={guardDrag}
            onMouseDown={guardDrag}
            onInput={(e) => onChange(parseFloat(e.currentTarget.value))}
          />
          <span className="canvas-hint">drag on canvas</span>
        </div>
      </div>
    );
  }

  // Default: linear range slider
  return (
    <div className="param-row">
      <label>
        {p.label} <span className="param-value">{String(value)}</span>
      </label>
      <input
        type="range"
        min={p.min}
        max={p.max}
        step={p.step}
        value={value as number}
        onPointerDown={guardDrag}
        onMouseDown={guardDrag}
        onInput={(e) => onChange(parseFloat(e.currentTarget.value))}
      />
    </div>
  );
}
