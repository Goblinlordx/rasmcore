import { useRef } from 'react';
import type { ChainNode as ChainNodeType } from '../types';
import ParamControl from './ParamControl';

interface Props {
  node: ChainNodeType;
  index: number;
  isSelected: boolean;
  onToggleSelect: () => void;
  onRemove: () => void;
  onParamChange: (paramName: string, value: number | string | boolean) => void;
  onDragStart: (idx: number, el: HTMLElement) => void;
  onDragEnd: () => void;
  onDragOver: (e: React.DragEvent) => void;
  onDrop: (targetIdx: number) => void;
}

export default function ChainNode({
  node,
  index,
  isSelected,
  onToggleSelect,
  onRemove,
  onParamChange,
  onDragStart,
  onDragEnd,
  onDragOver,
  onDrop,
}: Props) {
  const cardRef = useRef<HTMLDivElement>(null);

  const className = 'node-card' + (isSelected ? ' editing' : '');

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
      <div className="node-header" onClick={onToggleSelect} style={{ cursor: 'pointer' }}>
        <span
          className="drag-handle"
          draggable
          onDragStart={(e) => {
            e.stopPropagation();
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
          onClick={(e) => e.stopPropagation()}
        >
          {'\u2807'}
        </span>
        <span className="node-name">
          {index + 1}. {node.op.name}
        </span>
        <span className="node-actions">
          <span className="node-timing">{node.timingMs > 0 ? `${node.timingMs}ms` : ''}</span>
          <span className="node-remove" onClick={(e) => { e.stopPropagation(); onRemove(); }}>
            {'\u2715'}
          </span>
        </span>
      </div>

      {/* Collapsed: show param summary */}
      {!isSelected && node.op.params.length > 0 && (
        <div
          style={{ fontSize: '0.65rem', color: '#666', padding: '0 8px 4px', cursor: 'pointer' }}
          onClick={onToggleSelect}
        >
          {node.op.params.map((p) => `${p.name}=${node.paramValues[p.name]}`).join(', ')}
        </div>
      )}

      {/* Expanded: show param controls */}
      {isSelected && node.op.params.length > 0 && (
        <div className="node-body">
          {node.op.params.map((p) => (
            <ParamControl
              key={p.name}
              param={p}
              value={node.paramValues[p.name]}
              onChange={(v) => onParamChange(p.name, v)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
