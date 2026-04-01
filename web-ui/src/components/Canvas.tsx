import { useCallback, useEffect, useRef, useState } from 'react';

type ViewMode = 'current' | 'original' | 'split';

interface Props {
  previewCanvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  originalCanvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  hasImage: boolean;
  onAddLayer: (file: File) => void;
}

export default function Canvas({
  previewCanvasRef,
  originalCanvasRef,
  hasImage,
  onAddLayer,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const splitContainerRef = useRef<HTMLDivElement>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('current');
  const [splitPos, setSplitPos] = useState(50); // percentage
  const isDragging = useRef(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) onAddLayer(file);
    },
    [onAddLayer],
  );

  // Divider drag handlers
  const startDrag = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
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
    const onEnd = () => {
      isDragging.current = false;
    };

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

      {!hasImage && (
        <div id="drop-zone">
          <div className="drop-prompt">
            Drop an image to start
            <br />
            <span style={{ fontSize: '0.7rem', color: '#444' }}>or click to browse</span>
          </div>
        </div>
      )}

      {/* Normal view: show one canvas at a time */}
      {!showSplit && (
        <>
          <canvas
            ref={previewCanvasRef}
            id="preview-canvas"
            style={{ display: hasImage && showPreview ? 'block' : 'none' }}
          />
          <canvas
            ref={originalCanvasRef}
            id="original-canvas"
            style={{ display: hasImage && showOriginal ? 'block' : 'none' }}
          />
        </>
      )}

      {/* Split view: both canvases overlaid with clip-path divider */}
      {showSplit && hasImage && (
        <div className="split-container" ref={splitContainerRef}>
          <canvas
            ref={previewCanvasRef}
            id="preview-canvas"
            className="split-canvas"
            style={{ display: 'block' }}
          />
          <canvas
            ref={originalCanvasRef}
            id="original-canvas"
            className="split-canvas split-before"
            style={{
              display: 'block',
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
