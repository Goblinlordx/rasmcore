import { useCallback, useEffect, useRef, useState } from 'react';
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
}

export default function Canvas({
  previewCanvasRef,
  originalCanvasRef,
  hasImage,
  imageWidth,
  imageHeight,
  onAddLayer,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const splitContainerRef = useRef<HTMLDivElement>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('current');
  const [splitPos, setSplitPos] = useState(50);
  const isDragging = useRef(false);

  // Measure the container's available space
  const [containerRef, containerSize] = useContainerSize();

  // Pan/zoom state
  const transform = useCanvasTransform(containerSize, imageWidth, imageHeight);
  const canvasStyle = hasImage
    ? computeTransformCSS(transform.state, containerSize, imageWidth, imageHeight)
    : undefined;

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

  const showPreview = viewMode === 'current';
  const showOriginal = viewMode === 'original';
  const showSplit = viewMode === 'split';

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
      <div
        ref={containerRef}
        className="canvas-viewport"
        {...(hasImage ? transform.handlers : {})}
      >
        {/* Normal view: show one canvas at a time */}
        {!showSplit && (
          <>
            <canvas
              ref={previewCanvasRef}
              id="preview-canvas"
              style={{
                ...canvasStyle,
                display: hasImage && showPreview ? 'block' : 'none',
              }}
            />
            <canvas
              ref={originalCanvasRef}
              id="original-canvas"
              style={{
                ...canvasStyle,
                display: hasImage && showOriginal ? 'block' : 'none',
              }}
            />
          </>
        )}

        {/* Split view: both canvases with shared transform, clip-path divider */}
        {showSplit && hasImage && (
          <div
            className="split-container"
            ref={splitContainerRef}
            style={{
              ...canvasStyle,
              // Override: split container matches image dimensions, transform handles positioning
            }}
          >
            <canvas
              ref={previewCanvasRef}
              id="preview-canvas"
              className="split-canvas"
              style={{ display: 'block', width: '100%', height: '100%' }}
            />
            <canvas
              ref={originalCanvasRef}
              id="original-canvas"
              className="split-canvas split-before"
              style={{
                display: 'block',
                width: '100%',
                height: '100%',
                clipPath: `inset(0 ${100 - splitPos}% 0 0)`,
              }}
            />
            <div
              className="split-divider"
              style={{ left: `${splitPos}%` }}
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
