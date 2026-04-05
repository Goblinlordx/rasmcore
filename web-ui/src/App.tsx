import { useCallback, useEffect, useRef, useState } from 'react';
import { useAppContext } from './context/AppContext';
import { useWorker } from './hooks/useWorker';
import { usePreviewWorker } from './hooks/usePreviewWorker';
import { useLayers } from './hooks/useLayers';
import { useChain } from './hooks/useChain';
import { generateCode } from './utils/codeGeneration';
import { getWideGamutContext } from './utils/canvasColorSpace';
import { isWebGpuAvailable, isHdrDisplay } from './utils/webgpuDetect';
import Toolbar from './components/Toolbar';
import Canvas from './components/Canvas';
import RightPanel from './components/RightPanel';
import LayerPanel from './components/LayerPanel';
import EffectStack from './components/EffectStack';
import StatusBar from './components/StatusBar';
import CodeModal from './components/CodeModal';

// No debounce — send immediately, single-slot queue handles backpressure

export default function App() {
  const { operations, groups, writeFormats, loading } = useAppContext();
  const worker = useWorker();
  const preview = usePreviewWorker();
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

  /** Whether the last viewport update came from proxy (true) or full-res (false) */
  const [showingProxy, setShowingProxy] = useState(false);
  /** Whether background warm is running */
  const [warming, setWarming] = useState(false);

  // Stable refs so callbacks always have the latest
  const serializeChainRef = useRef(serializeChain);
  serializeChainRef.current = serializeChain;
  const requestCompositeRef = useRef<(() => void) | null>(null);

  // Connect preview worker's viewport canvas to the main Canvas component
  useEffect(() => {
    preview.viewportCanvasRef.current = worker.previewCanvasRef.current;
  });

  // Transfer OffscreenCanvases to preview worker for WebGPU direct display
  const webgpuPreviewInitRef = useRef(false);
  const webgpuOriginalInitRef = useRef(false);
  useEffect(() => {
    if (!isWebGpuAvailable()) return;
    const hdr = isHdrDisplay();
    // Transfer preview canvas (once)
    if (!webgpuPreviewInitRef.current) {
      const canvas = worker.previewCanvasRef.current;
      if (canvas) {
        try {
          preview.setDisplayCanvas(canvas.transferControlToOffscreen(), hdr);
          webgpuPreviewInitRef.current = true;
        } catch { /* stay in 2D mode */ }
      }
    }
    // Transfer original canvas (once) — may not exist on first render
    if (!webgpuOriginalInitRef.current) {
      const canvas = worker.originalCanvasRef.current;
      if (canvas) {
        try {
          preview.setOriginalDisplayCanvas(canvas.transferControlToOffscreen(), hdr);
          webgpuOriginalInitRef.current = true;
        } catch { /* stay in 2D mode */ }
      }
    }
  });

  // After proxy render completes — background warm disabled for now (perf)
  useEffect(() => {
    // eslint-disable-next-line react-hooks/immutability
    preview.onProxyCompleteRef.current = () => {
      setShowingProxy(true);
    };
  }, [preview]);

  // When background warm completes, mark viewport as full-res
  useEffect(() => {
    // eslint-disable-next-line react-hooks/immutability
    worker.onWarmCompleteRef.current = () => {
      setWarming(false);
      setShowingProxy(false);
    };
  }, [worker]);

  // Set up onLoaded callback to trigger initial render via proxy
  useEffect(() => {
    // eslint-disable-next-line react-hooks/immutability
    worker.onLoadedRef.current = () => {
      // Trigger proxy render first (fast), then background warm follows via onProxyComplete
      preview.processChain(serializeChainRef.current());
    };
  }, [worker, preview]);

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
            // In display mode, worker owns both canvases — it renders via GPU on load
            if (!preview.displayMode) {
              const oc = worker.originalCanvasRef.current;
              const pc = worker.previewCanvasRef.current;
              if (oc) {
                oc.width = img.width;
                oc.height = img.height;
                getWideGamutContext(oc)?.drawImage(img, 0, 0);
              }
              if (pc) {
                pc.width = img.width;
                pc.height = img.height;
                getWideGamutContext(pc)?.drawImage(img, 0, 0);
              }
            }
            URL.revokeObjectURL(url);
          };
          img.src = url;
        } catch {
          /* ignore */
        }
      }
      // Send to both workers
      const copy1 = layer.imageBytes.buffer.slice(0);
      const copy2 = layer.imageBytes.buffer.slice(0) as ArrayBuffer;
      worker.sendMessage({ type: 'load', imageBytes: copy1 });
      preview.loadImage(copy2);

      // Multi-layer: trigger composite after load settles
      if (!isFirst) {
        setTimeout(() => requestCompositeRef.current?.(), 100);
      }
    },
    [addLayer, worker, preview],
  );

  // Apply chain via proxy first (fast ~1080p), then background warm follows automatically
  const applyFullChain = useCallback(() => {
    if (!activeLayer?.imageBytes) return;
    preview.processChain(serializeChainRef.current());
  }, [activeLayer, preview]);

  const requestCompositeProcess = useCallback(() => {
    if (layers.length === 0) return;
    if (layers.length === 1) {
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
  requestCompositeRef.current = requestCompositeProcess;

  // Remove node AND re-process
  const handleRemoveNode = useCallback(
    (id: number) => {
      removeNode(id);
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

  // Send preview immediately — single-slot queue in usePreviewWorker drops
  // stale requests automatically (latest params always win).
  const schedulePreview = useCallback(() => {
    preview.processChain(serializeChainRef.current());
  }, [preview]);

  const handleDownload = useCallback(
    (format: string, quality: number) => {
      if (layers.length === 0) return;
      setExportFormat(format);
      setExportQuality(quality);
      worker.sendMessage({ type: 'export', chain: serializeChainRef.current(), format, quality });
    },
    [layers.length, worker],
  );

  const handleViewportChange = useCallback(
    (panX: number, panY: number, zoom: number, cw: number, ch: number) => {
      // Send preview dimensions (what the pixel buffer actually contains),
      // not full-res dimensions — the shader needs the buffer stride to match.
      const imgW = preview.previewWidth || worker.imageInfo?.width || 0;
      const imgH = preview.previewHeight || worker.imageInfo?.height || 0;
      preview.sendViewport(
        panX, panY, zoom, cw, ch,
        imgW, imgH,
        isHdrDisplay() ? 1 : 0,
      );
    },
    [preview, worker.imageInfo?.width, worker.imageInfo?.height],
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
        onAddNode={(op) => { addNode(op); setTimeout(() => schedulePreview(), 0); }}
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
          imageWidth={worker.imageInfo?.width ?? 0}
          imageHeight={worker.imageInfo?.height ?? 0}
          onAddLayer={handleAddLayer}
          displayMode={preview.displayMode}
          onViewportChange={handleViewportChange}
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
            selectedNodeId={editingNodeId}
            activeLayerName={activeLayer?.name || ''}
            onSetSelected={setEditingNodeId}
            onRemoveNode={handleRemoveNode}
            onMoveNode={handleMoveNode}
            onParamChange={updateParam}
            onSchedulePreview={schedulePreview}
          />
        </RightPanel>
      </div>
      <StatusBar
        dims={worker.imageInfo}
        timings={worker.timings}
        processing={worker.processing || preview.processing}
        error={worker.error}
        showingProxy={showingProxy}
        warming={warming}
        proxyMs={preview.proxyMs}
      />
      <CodeModal
        open={codeModalOpen}
        code={generateCode(layers, chain, exportFormat, exportQuality)}
        onClose={() => setCodeModalOpen(false)}
      />
    </>
  );
}
