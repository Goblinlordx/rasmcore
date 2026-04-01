import { useRef } from 'react';
import type { ChainNode as ChainNodeType } from '../types';
import ParamControl from './ParamControl';

interface Props {
  node: ChainNodeType;
  index: number;
  isEditing: boolean;
  onEdit: () => void;
  onRemove: () => void;
  onApply: () => void;
  onCancelEdit: () => void;
  onParamChange: (paramName: string, value: number | string | boolean) => void;
  onDragStart: (idx: number, el: HTMLElement) => void;
  onDragEnd: () => void;
  onDragOver: (e: React.DragEvent) => void;
  onDrop: (targetIdx: number) => void;
  onSchedulePreview: () => void;
}

export default function ChainNode({
  node,
  index,
  isEditing,
  onEdit,
  onRemove,
  onApply,
  onCancelEdit,
  onParamChange,
  onDragStart,
  onDragEnd,
  onDragOver,
  onDrop,
  onSchedulePreview,
}: Props) {
  const cardRef = useRef<HTMLDivElement>(null);

  const className = 'node-card' + (isEditing ? ' editing' : '') + (!node.applied ? ' pending' : '');

  return (
    <div
      ref={cardRef}
      className={className}
      data-id={node.id}
      data-idx={index}
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        onDragOver(e);
        if (cardRef.current) cardRef.current.style.borderTop = '2px solid #3b82f6';
      }}
      onDragLeave={() => {
        if (cardRef.current) cardRef.current.style.borderTop = '';
      }}
      onDrop={(e) => {
        e.preventDefault();
        if (cardRef.current) cardRef.current.style.borderTop = '';
        onDrop(index);
      }}
    >
      <div className="node-header">
        <span
          className="drag-handle"
          draggable
          onDragStart={(e) => {
            onDragStart(index, cardRef.current!);
            e.dataTransfer.effectAllowed = 'move';
            if (cardRef.current) {
              e.dataTransfer.setDragImage(cardRef.current, 0, 0);
              cardRef.current.style.opacity = '0.4';
            }
          }}
          onDragEnd={() => {
            if (cardRef.current) cardRef.current.style.opacity = '1';
            onDragEnd();
          }}
        >
          {'\u2807'}
        </span>
        <span className="node-name">
          {index + 1}. {node.op.name}
          {!node.applied ? ' *' : ''}
        </span>
        <span className="node-actions">
          <span className="node-timing">{node.timingMs > 0 ? `${node.timingMs}ms` : ''}</span>
          {!isEditing && node.op.params.length > 0 && (
            <span
              style={{ cursor: 'pointer', color: '#60a5fa', fontSize: '0.85rem' }}
              title="Edit parameters"
              onClick={onEdit}
            >
              {'\u270E'}
            </span>
          )}
          <span className="node-remove" onClick={onRemove}>
            {'\u2715'}
          </span>
        </span>
      </div>

      {!isEditing && node.op.params.length > 0 && (
        <div style={{ fontSize: '0.65rem', color: '#666', padding: '0 8px 4px' }}>
          {node.op.params.map((p) => `${p.name}=${node.paramValues[p.name]}`).join(', ')}
        </div>
      )}

      {isEditing && (
        <div className="node-body">
          {node.op.params.map((p) => (
            <ParamControl
              key={p.name}
              param={p}
              value={node.paramValues[p.name]}
              onChange={(v) => {
                onParamChange(p.name, v);
                onSchedulePreview();
              }}
            />
          ))}
          <div style={{ marginTop: 6, display: 'flex', gap: 6 }}>
            <button onClick={onApply}>Apply</button>
            <button className="secondary" style={{ background: '#333' }} onClick={onCancelEdit}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
