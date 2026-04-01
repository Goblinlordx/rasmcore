import { useCallback, useState } from 'react';
import type { ChainNode } from '../types';

export interface LayerState {
  id: number;
  name: string;
  visible: boolean;
  blendMode: string;
  x: number;
  y: number;
  imageBytes: Uint8Array;
  chain: ChainNode[];
  thumbUrl: string | null;
}

let nextLayerId = 0;

export function useLayers() {
  const [layers, setLayers] = useState<LayerState[]>([]);
  const [activeLayerId, setActiveLayerId] = useState<number | null>(null);

  const addLayer = useCallback(
    async (file: File) => {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const id = nextLayerId++;
      let thumbUrl: string | null = null;
      try {
        const blob = new Blob([bytes], { type: file.type || 'image/png' });
        thumbUrl = URL.createObjectURL(blob);
      } catch {
        /* ignore */
      }

      const layer: LayerState = {
        id,
        name: file.name.replace(/\.[^.]+$/, ''),
        imageBytes: bytes,
        chain: [],
        blendMode: 'over',
        x: 0,
        y: 0,
        visible: true,
        thumbUrl,
      };

      setLayers((prev) => {
        const next = [...prev, layer];
        if (next.length === 1) setActiveLayerId(id);
        return next;
      });

      return { layer, isFirst: layers.length === 0 };
    },
    [layers.length],
  );

  const removeLayer = useCallback(
    (id: number) => {
      setLayers((prev) => {
        const next = prev.filter((l) => l.id !== id);
        if (activeLayerId === id) {
          setActiveLayerId(next.length > 0 ? next[0].id : null);
        }
        return next;
      });
    },
    [activeLayerId],
  );

  const updateLayer = useCallback((id: number, updates: Partial<LayerState>) => {
    setLayers((prev) => prev.map((l) => (l.id === id ? { ...l, ...updates } : l)));
  }, []);

  // Functional updater — reads current layer state to avoid stale closures
  const updateLayerChain = useCallback(
    (id: number, updater: (currentChain: ChainNode[]) => ChainNode[]) => {
      setLayers((prev) => prev.map((l) => (l.id === id ? { ...l, chain: updater(l.chain) } : l)));
    },
    [],
  );

  const activeLayer = layers.find((l) => l.id === activeLayerId) ?? null;

  return {
    layers,
    setLayers,
    activeLayerId,
    setActiveLayerId,
    activeLayer,
    addLayer,
    removeLayer,
    updateLayer,
    updateLayerChain,
  };
}
