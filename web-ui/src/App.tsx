import { useCallback, useEffect, useRef, useState } from 'react';
import { useAppContext } from './context/AppContext';
import { useWorker } from './hooks/useWorker';
import { useLayers } from './hooks/useLayers';
import { useChain } from './hooks/useChain';
import { generateCode } from './utils/codeGeneration';
import Toolbar from './components/Toolbar';
import Canvas from './components/Canvas';
import RightPanel from './components/RightPanel';
import LayerPanel from './components/LayerPanel';
import EffectStack from './components/EffectStack';
import StatusBar from './components/StatusBar';
import CodeModal from './components/CodeModal';

const PREVIEW_DEBOUNCE_MS = 600;

export default function App() {
  const { operations, groups, writeFormats, loading } = useAppContext();
  const worker = useWorker();
  const {
    layers,
    activeLayerId,
    activeLayer,
    addLayer,
    removeLayer,
    updateLayer,
    updateLayerChain,
    setActiveLayerId,
  } = useLayers();
  const {
    chain,
    editingNodeId,
    setEditingNodeId,
    addNode,
    removeNode,
    moveNode,
    updateParam,
    applyNode,
    serializeChain,
  } = useChain(activeLayer, updateLayerChain);

  const [codeModalOpen, setCodeModalOpen] = useState(false);
  const [exportFormat, setExportFormat] = useState('jpeg');
  const [exportQuality, setExportQuality] = useState(85);
  const previewTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Stable ref to current serializeChain so onLoaded always has the latest
  const serializeChainRef = useRef(serializeChain);
  serializeChainRef.current = serializeChain;

  // Set up onLoaded callback to trigger initial render
  useEffect(() => {
    // eslint-disable-next-line react-hooks/immutability
    worker.onLoadedRef.current = () => {
      worker.sendMessage({
        type: 'process',
        chain: serializeChainRef.current(),
        mode: 'full',
      });
    };
  }, [worker]);

  const handleAddLayer = useCallback(
    async (file: File) => {
      const { layer, isFirst } = await addLayer(file);
      if (isFirst) {
        try {
          const blob = new Blob([layer.imageBytes.buffer as ArrayBuffer], {
            type: file.type || 'image/png',
          });
          const url = URL.createObjectURL(blob);
          const img = new Image();
          img.onload = () => {
            const oc = worker.originalCanvasRef.current;
            const pc = worker.previewCanvasRef.current;
            if (oc) {
              oc.width = img.width;
              oc.height = img.height;
              oc.getContext('2d')?.drawImage(img, 0, 0);
            }
            if (pc) {
              pc.width = img.width;
              pc.height = img.height;
              pc.getContext('2d')?.drawImage(img, 0, 0);
            }
            URL.revokeObjectURL(url);
          };
          img.src = url;
        } catch {
          /* ignore */
        }
      }
      const copy = layer.imageBytes.buffer.slice(0);
      worker.sendMessage({ type: 'load', imageBytes: copy });
    },
    [addLayer, worker],
  );

  const applyFullChain = useCallback(() => {
    if (!activeLayer?.imageBytes) return;
    worker.sendMessage({ type: 'process', chain: serializeChainRef.current(), mode: 'full' });
  }, [activeLayer, worker]);

  const requestCompositeProcess = useCallback(() => {
    if (layers.length === 0) return;
    if (layers.length === 1) {
      // Re-load then process — ensures worker has the image
      const copy = layers[0].imageBytes.buffer.slice(0);
      worker.sendMessage({ type: 'load', imageBytes: copy });
      return;
    }
    const layerData = layers
      .filter((l) => l.visible)
      .map((l, idx) => ({
        id: l.id,
        imageBytes: l.imageBytes.buffer.slice(0),
        chain: l.chain.map((n) => ({
          name: n.op.name,
          params: n.op.params,
          paramValues: { ...n.paramValues },
        })),
        blendMode: idx === 0 ? null : l.blendMode === 'over' ? null : l.blendMode,
        x: l.x,
        y: l.y,
      }));
    worker.sendMessage({ type: 'composite', layers: layerData });
  }, [layers, worker]);

  // Remove node AND re-process
  const handleRemoveNode = useCallback(
    (id: number) => {
      removeNode(id);
      // Schedule re-process after state update
      setTimeout(() => applyFullChain(), 0);
    },
    [removeNode, applyFullChain],
  );

  // Move node AND re-process
  const handleMoveNode = useCallback(
    (from: number, to: number) => {
      moveNode(from, to);
      setTimeout(() => applyFullChain(), 0);
    },
    [moveNode, applyFullChain],
  );

  const schedulePreview = useCallback(() => {
    if (previewTimerRef.current) clearTimeout(previewTimerRef.current);
    previewTimerRef.current = setTimeout(() => {
      if (!activeLayer?.imageBytes || editingNodeId === null) return;
      worker.sendMessage({ type: 'process', chain: serializeChainRef.current(), mode: 'thumb' });
    }, PREVIEW_DEBOUNCE_MS);
  }, [activeLayer, editingNodeId, worker]);

  const handleDownload = useCallback(
    (format: string, quality: number) => {
      if (layers.length === 0) return;
      setExportFormat(format);
      setExportQuality(quality);
      worker.sendMessage({ type: 'export', chain: serializeChainRef.current(), format, quality });
    },
    [layers.length, worker],
  );

  if (loading) {
    return <div style={{ color: '#888', padding: 20 }}>Loading SDK...</div>;
  }

  return (
    <>
      <Toolbar
        operations={operations}
        groups={groups}
        writeFormats={writeFormats}
        onAddNode={addNode}
        onDownload={handleDownload}
        onShowCode={() => setCodeModalOpen(true)}
        exportFormat={exportFormat}
        onExportFormatChange={setExportFormat}
        exportQuality={exportQuality}
        onExportQualityChange={setExportQuality}
      />
      <div className="main-layout">
        <Canvas
          previewCanvasRef={worker.previewCanvasRef}
          originalCanvasRef={worker.originalCanvasRef}
          hasImage={layers.length > 0}
          onAddLayer={handleAddLayer}
        />
        <RightPanel>
          <LayerPanel
            layers={layers}
            activeLayerId={activeLayerId}
            onSelectLayer={setActiveLayerId}
            onToggleVisibility={(id) => {
              const layer = layers.find((l) => l.id === id);
              if (layer) updateLayer(id, { visible: !layer.visible });
            }}
            onRemoveLayer={removeLayer}
            onUpdateLayer={updateLayer}
            onAddLayer={handleAddLayer}
            onRequestComposite={requestCompositeProcess}
          />
          <EffectStack
            chain={chain}
            editingNodeId={editingNodeId}
            activeLayerName={activeLayer?.name || ''}
            onSetEditing={setEditingNodeId}
            onRemoveNode={handleRemoveNode}
            onMoveNode={handleMoveNode}
            onApplyNode={applyNode}
            onParamChange={updateParam}
            onSchedulePreview={schedulePreview}
            onApplyFullChain={applyFullChain}
          />
        </RightPanel>
      </div>
      <StatusBar
        dims={worker.imageInfo}
        timings={worker.timings}
        processing={worker.processing}
        error={worker.error}
      />
      <CodeModal
        open={codeModalOpen}
        code={generateCode(layers, chain, exportFormat, exportQuality)}
        onClose={() => setCodeModalOpen(false)}
      />
    </>
  );
}
