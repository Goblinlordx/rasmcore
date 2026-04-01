import { useCallback, useMemo, useState } from 'react';
import type { ChainNode, Operation } from '../types';
import type { LayerState } from './useLayers';

let nextNodeId = 0;

export function useChain(
  activeLayer: LayerState | null,
  updateLayerChain: (id: number, updater: (chain: ChainNode[]) => ChainNode[]) => void,
) {
  const [editingNodeId, setEditingNodeId] = useState<number | null>(null);

  const chain = useMemo(() => activeLayer?.chain ?? [], [activeLayer?.chain]);
  const layerId = activeLayer?.id ?? null;

  const addNode = useCallback(
    (op: Operation) => {
      if (layerId === null) return;
      const node: ChainNode = {
        id: nextNodeId++,
        op,
        paramValues: Object.fromEntries(op.params.map((p) => [p.name, p.default])),
        applied: false,
        timingMs: 0,
      };
      updateLayerChain(layerId, (prev) => [...prev, node]);
      setEditingNodeId(node.id);
    },
    [layerId, updateLayerChain],
  );

  const removeNode = useCallback(
    (id: number) => {
      if (layerId === null) return;
      updateLayerChain(layerId, (prev) => prev.filter((n) => n.id !== id));
      if (editingNodeId === id) setEditingNodeId(null);
    },
    [layerId, editingNodeId, updateLayerChain],
  );

  const moveNode = useCallback(
    (fromIdx: number, toIdx: number) => {
      if (layerId === null || fromIdx === toIdx) return;
      updateLayerChain(layerId, (prev) => {
        const next = [...prev];
        const [node] = next.splice(fromIdx, 1);
        next.splice(toIdx, 0, node);
        return next;
      });
    },
    [layerId, updateLayerChain],
  );

  const updateParam = useCallback(
    (nodeId: number, paramName: string, value: number | string | boolean) => {
      if (layerId === null) return;
      updateLayerChain(layerId, (prev) =>
        prev.map((n) =>
          n.id === nodeId ? { ...n, paramValues: { ...n.paramValues, [paramName]: value } } : n,
        ),
      );
    },
    [layerId, updateLayerChain],
  );

  const applyNode = useCallback(
    (nodeId: number) => {
      if (layerId === null) return;
      updateLayerChain(layerId, (prev) =>
        prev.map((n) => (n.id === nodeId ? { ...n, applied: true } : n)),
      );
      setEditingNodeId(null);
    },
    [layerId, updateLayerChain],
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
