import { useCallback, useMemo, useState } from 'react';
import type { ChainNode, Operation } from '../types';
import type { LayerState } from './useLayers';

let nextNodeId = 0;

export function useChain(
  activeLayer: LayerState | null,
  updateLayer: (id: number, updates: Partial<LayerState>) => void,
) {
  const [editingNodeId, setEditingNodeId] = useState<number | null>(null);

  const chain = useMemo(() => activeLayer?.chain ?? [], [activeLayer?.chain]);

  const setChain = useCallback(
    (newChain: ChainNode[]) => {
      if (activeLayer) updateLayer(activeLayer.id, { chain: newChain });
    },
    [activeLayer, updateLayer],
  );

  const addNode = useCallback(
    (op: Operation) => {
      if (!activeLayer) return;
      const node: ChainNode = {
        id: nextNodeId++,
        op,
        paramValues: Object.fromEntries(op.params.map((p) => [p.name, p.default])),
        applied: false,
        timingMs: 0,
      };
      updateLayer(activeLayer.id, { chain: [...chain, node] });
      setEditingNodeId(node.id);
    },
    [activeLayer, chain, updateLayer],
  );

  const removeNode = useCallback(
    (id: number) => {
      setChain(chain.filter((n) => n.id !== id));
      if (editingNodeId === id) setEditingNodeId(null);
    },
    [chain, editingNodeId, setChain],
  );

  const moveNode = useCallback(
    (fromIdx: number, toIdx: number) => {
      if (fromIdx === toIdx) return;
      const next = [...chain];
      const [node] = next.splice(fromIdx, 1);
      next.splice(toIdx, 0, node);
      setChain(next);
    },
    [chain, setChain],
  );

  const updateParam = useCallback(
    (nodeId: number, paramName: string, value: number | string | boolean) => {
      setChain(
        chain.map((n) =>
          n.id === nodeId ? { ...n, paramValues: { ...n.paramValues, [paramName]: value } } : n,
        ),
      );
    },
    [chain, setChain],
  );

  const applyNode = useCallback(
    (nodeId: number) => {
      setChain(chain.map((n) => (n.id === nodeId ? { ...n, applied: true } : n)));
      setEditingNodeId(null);
    },
    [chain, setChain],
  );

  const serializeChain = useCallback(() => {
    return chain.map((n) => ({
      name: n.op.name,
      params: n.op.params,
      paramValues: { ...n.paramValues },
    }));
  }, [chain]);

  return {
    chain,
    editingNodeId,
    setEditingNodeId,
    addNode,
    removeNode,
    moveNode,
    updateParam,
    applyNode,
    serializeChain,
  };
}
