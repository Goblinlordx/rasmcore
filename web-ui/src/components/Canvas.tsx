import { useCallback, useEffect, useRef, useState, type RefCallback } from 'react';
import { useContainerSize } from '../hooks/useContainerSize';
import { useCanvasTransform, computeTransformCSS, formatZoom } from '../hooks/useCanvasTransform';

type ViewMode = 'current' | 'original' | 'split';

interface Props {
  previewCanvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  originalCanvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  hasImage: boolean;
  imageWidth: number;
  imageHeight: number;
  onAddLayer: (file: File) => void;
  /** Whether WebGPU display mode is active (shader handles pan/zoom). */
  displayMode?: boolean;
  /** Forward viewport changes to worker for shader-based pan/zoom. */
  onViewportChange?: (panX: number, panY: number, zoom: number, canvasWidth: number, canvasHeight: number) => void;
}

export default function Canvas({
  previewCanvasRef,
  originalCanvasRef,
  hasImage,
  imageWidth,
  imageHeight,
  onAddLayer,
  displayMode = false,
  onViewportChange,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const splitContainerRef = useRef<HTMLDivElement>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('current');
  const [splitPos, setSplitPos] = useState(50);
  const isDragging = useRef(false);
  // Saved zoom/pan state before entering split mode
  const savedTransformRef = useRef<{ zoom: number; panX: number; panY: number } | null>(null);

  // Measure the container's available space
  const [containerRef, containerSize] = useContainerSize();

  // Pan/zoom state
  const transform = useCanvasTransform(containerSize, imageWidth, imageHeight);

  // Combine containerRef (ResizeObserver) + gestureRef (Safari gestures) on same element
  const viewportRef = useCallback((el: HTMLDivElement | null) => {
    (containerRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
    transform.gestureRef(el);
  }, [containerRef, transform.gestureRef]);
  // Display mode: preview canvas fills viewport (shader handles pan/zoom).
  // Original canvas always uses CSS transform (never stretched).
  const displayCanvasStyle = hasImage
    ? { position: 'absolute' as const, left: 0, top: 0, width: containerSize.width, height: containerSize.height }
    : undefined;
  const transformCanvasStyle = hasImage
    ? computeTransformCSS(transform.state, containerSize, imageWidth, imageHeight)
    : undefined;
  const showPreview = viewMode === 'current';
  const showOriginal = viewMode === 'original';
  const showSplit = viewMode === 'split';

  // Split mode: always fit-to-viewport with no pan
  const fitState = { zoom: transform.fitZoom, panX: 0, panY: 0 };
  const fitCanvasStyle = hasImage
    ? computeTransformCSS(fitState, containerSize, imageWidth, imageHeight)
    : undefined;
  // Preview uses display mode style when GPU owns it, original always uses CSS transform
  const previewStyle = showSplit ? fitCanvasStyle : (displayMode ? displayCanvasStyle : transformCanvasStyle);
  const originalStyle = showSplit ? fitCanvasStyle : transformCanvasStyle;

  // Forward viewport changes to worker for shader-based pan/zoom
  useEffect(() => {
    if (!displayMode || !onViewportChange || !hasImage) return;
    onViewportChange(
      transform.state.panX, transform.state.panY, transform.state.zoom,
      containerSize.width, containerSize.height,
    );
  }, [displayMode, onViewportChange, hasImage, transform.state.panX, transform.state.panY, transform.state.zoom, containerSize.width, containerSize.height]);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) onAddLayer(file);
    },
    [onAddLayer],
  );

  // Split divider drag handlers
  const startDrag = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    e.stopPropagation();
    isDragging.current = true;
  }, []);

  useEffect(() => {
    const handleMove = (clientX: number) => {
      if (!isDragging.current || !splitContainerRef.current) return;
      const rect = splitContainerRef.current.getBoundingClientRect();
      const pct = ((clientX - rect.left) / rect.width) * 100;
      setSplitPos(Math.max(5, Math.min(95, pct)));
    };

    const onMouseMove = (e: MouseEvent) => handleMove(e.clientX);
    const onTouchMove = (e: TouchEvent) => {
      if (e.touches[0]) handleMove(e.touches[0].clientX);
    };
    const onEnd = () => { isDragging.current = false; };

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onEnd);
    document.addEventListener('touchmove', onTouchMove);
    document.addEventListener('touchend', onEnd);
    return () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onEnd);
      document.removeEventListener('touchmove', onTouchMove);
      document.removeEventListener('touchend', onEnd);
    };
  }, []);

  return (
    <div
      className="canvas-area"
      id="canvas-area"
      onClick={() => !hasImage && fileInputRef.current?.click()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      {/* Tab bar */}
      {hasImage && (
        <div className="canvas-tabs">
          {(['current', 'original', 'split'] as const).map((mode) => (
            <button
              key={mode}
              className={'canvas-tab' + (viewMode === mode ? ' active' : '')}
              onClick={(e) => {
                e.stopPropagation();
                if (mode === 'split' && viewMode !== 'split') {
                  // Save current zoom/pan before entering split
                  savedTransformRef.current = { ...transform.state };
                  transform.resetToFit();
                } else if (mode !== 'split' && viewMode === 'split' && savedTransformRef.current) {
                  // Restore zoom/pan when leaving split
                  transform.restore(savedTransformRef.current);
                  savedTransformRef.current = null;
                }
                setViewMode(mode);
              }}
            >
              {mode.charAt(0).toUpperCase() + mode.slice(1)}
            </button>
          ))}
        </div>
      )}

      {/* Zoom indicator */}
      {hasImage && (
        <div className="canvas-zoom-indicator" onClick={transform.resetToFit}>
          {formatZoom(transform.state.zoom, transform.fitZoom)}
        </div>
      )}

      {!hasImage && (
        <div id="drop-zone">
          <div className="drop-prompt">
            Drop an image to start
            <br />
            <span style={{ fontSize: '0.7rem', color: '#444' }}>or click to browse</span>
          </div>
        </div>
      )}

      {/* Viewport container — fills available space, handles pan/zoom events */}
      {/* Split mode: no pan/zoom — fit-to-viewport only, divider drag handles interaction */}
      <div
        ref={viewportRef}
        className="canvas-viewport"
        {...(hasImage && !showSplit ? transform.handlers : {})}
      >
        {/* Both canvases always rendered (never unmount) — visibility controlled by CSS */}
        <canvas
          ref={previewCanvasRef}
          id="preview-canvas"
          style={{
            ...previewStyle,
            display: hasImage && (showPreview || showSplit) ? 'block' : 'none',
            clipPath: showSplit ? `inset(0 0 0 ${splitPos}%)` : undefined,
          }}
        />
        <canvas
          ref={originalCanvasRef}
          id="original-canvas"
          style={{
            ...originalStyle,
            display: hasImage && (showOriginal || showSplit) ? 'block' : 'none',
            clipPath: showSplit ? `inset(0 ${100 - splitPos}% 0 0)` : undefined,
          }}
        />

        {/* Split view divider + labels */}
        {showSplit && hasImage && (
          <div
            ref={splitContainerRef}
            style={{
              ...originalStyle,
              pointerEvents: 'none',
            }}
          >
            <div
              className="split-divider"
              style={{ left: `${splitPos}%`, pointerEvents: 'auto' }}
              onMouseDown={startDrag}
              onTouchStart={startDrag}
            >
              <div className="split-divider-handle" />
            </div>
            <div className="split-labels">
              <span className="split-label-before">Before</span>
              <span className="split-label-after">After</span>
            </div>
          </div>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onAddLayer(file);
        }}
      />
    </div>
  );
}
