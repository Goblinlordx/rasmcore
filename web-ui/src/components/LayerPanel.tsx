import { useRef } from 'react';
import type { LayerState } from '../hooks/useLayers';

const BLEND_MODES = [
  'over',
  'multiply',
  'screen',
  'overlay',
  'darken',
  'lighten',
  'soft-light',
  'hard-light',
  'difference',
  'exclusion',
];

interface Props {
  layers: LayerState[];
  activeLayerId: number | null;
  onSelectLayer: (id: number) => void;
  onToggleVisibility: (id: number) => void;
  onRemoveLayer: (id: number) => void;
  onUpdateLayer: (id: number, updates: Partial<LayerState>) => void;
  onAddLayer: (file: File) => void;
  onRequestComposite: () => void;
}

export default function LayerPanel({
  layers,
  activeLayerId,
  onSelectLayer,
  onToggleVisibility,
  onRemoveLayer,
  onUpdateLayer,
  onAddLayer,
  onRequestComposite,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <div className="panel-section" id="layers-section">
      <div className="panel-section-header">
        <h3>Layers</h3>
      </div>
      <div className="panel-section-body">
        <div id="layers">
          {layers.map((layer, idx) => (
            <div
              key={layer.id}
              className={'layer-card' + (layer.id === activeLayerId ? ' active' : '')}
              onClick={() => onSelectLayer(layer.id)}
            >
              <div className="layer-card-header">
                {layer.thumbUrl && (
                  <img className="layer-thumb" src={layer.thumbUrl} alt={layer.name} />
                )}
                <span className="layer-name">
                  {idx === 0 ? 'BG' : `L${idx}`}: {layer.name}
                </span>
                <span
                  className={'layer-vis' + (layer.visible ? '' : ' hidden')}
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleVisibility(layer.id);
                    onRequestComposite();
                  }}
                >
                  {layer.visible ? '\uD83D\uDC41' : '\uD83D\uDC41\u200D\uD83D\uDDE8'}
                </span>
                {layers.length > 1 && (
                  <span
                    className="layer-delete"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemoveLayer(layer.id);
                    }}
                  >
                    {'\u2715'}
                  </span>
                )}
              </div>
              {idx > 0 && (
                <div className="layer-controls">
                  <select
                    value={layer.blendMode}
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => {
                      e.stopPropagation();
                      onUpdateLayer(layer.id, { blendMode: e.target.value });
                      onRequestComposite();
                    }}
                  >
                    {BLEND_MODES.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                  <input
                    type="number"
                    value={layer.x}
                    placeholder="X"
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => {
                      e.stopPropagation();
                      onUpdateLayer(layer.id, { x: parseInt(e.target.value) || 0 });
                      onRequestComposite();
                    }}
                  />
                  <input
                    type="number"
                    value={layer.y}
                    placeholder="Y"
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => {
                      e.stopPropagation();
                      onUpdateLayer(layer.id, { y: parseInt(e.target.value) || 0 });
                      onRequestComposite();
                    }}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
        <button id="add-layer-btn" onClick={() => fileInputRef.current?.click()}>
          + Add Layer
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onAddLayer(file);
            e.target.value = '';
          }}
        />
      </div>
    </div>
  );
}
