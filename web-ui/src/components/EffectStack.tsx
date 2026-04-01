import { useCallback, useRef } from 'react';
import type { ChainNode as ChainNodeType } from '../types';
import ChainNodeComponent from './ChainNode';

interface Props {
  chain: ChainNodeType[];
  editingNodeId: number | null;
  activeLayerName: string;
  onSetEditing: (id: number | null) => void;
  onRemoveNode: (id: number) => void;
  onMoveNode: (from: number, to: number) => void;
  onApplyNode: (id: number) => void;
  onParamChange: (nodeId: number, paramName: string, value: number | string | boolean) => void;
  onApplyFullChain: () => void;
}

export default function EffectStack({
  chain,
  editingNodeId,
  activeLayerName,
  onSetEditing,
  onRemoveNode,
  onMoveNode,
  onApplyNode,
  onParamChange,
  onApplyFullChain,
}: Props) {
  const dragSrcIdx = useRef<number | null>(null);

  const handleDragStart = useCallback((idx: number, _el: HTMLElement) => {
    dragSrcIdx.current = idx;
  }, []);

  const handleDragEnd = useCallback(() => {
    dragSrcIdx.current = null;
  }, []);

  const handleDrop = useCallback(
    (targetIdx: number) => {
      if (dragSrcIdx.current !== null && dragSrcIdx.current !== targetIdx) {
        onMoveNode(dragSrcIdx.current, targetIdx);
      }
      dragSrcIdx.current = null;
    },
    [onMoveNode],
  );

  return (
    <div className="panel-section" id="effects-section">
      <div className="panel-section-header">
        <h3>
          Effect Stack{' '}
          <span style={{ color: '#8b5cf6', fontSize: '0.55rem' }}>
            {activeLayerName ? `(${activeLayerName})` : ''}
          </span>
        </h3>
      </div>
      <div className="panel-section-body">
        <div id="chain">
          {chain.map((node, idx) => (
            <ChainNodeComponent
              key={node.id}
              node={node}
              index={idx}
              isEditing={editingNodeId === node.id}
              onEdit={() => onSetEditing(node.id)}
              onRemove={() => onRemoveNode(node.id)}
              onApply={() => {
                onApplyNode(node.id);
                onApplyFullChain();
              }}
              onCancelEdit={() => onSetEditing(null)}
              onParamChange={(name, val) => onParamChange(node.id, name, val)}
              onDragStart={handleDragStart}
              onDragEnd={handleDragEnd}
              onDragOver={() => {}}
              onDrop={handleDrop}
            />
          ))}
        </div>
        {chain.length === 0 && (
          <div style={{ padding: 12, textAlign: 'center', color: '#555', fontSize: '0.7rem' }}>
            Select an operation from the toolbar
          </div>
        )}
      </div>
    </div>
  );
}
