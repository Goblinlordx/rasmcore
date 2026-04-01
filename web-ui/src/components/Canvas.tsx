import { useCallback, useRef, useState } from 'react';

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
  const [showingOriginal, setShowingOriginal] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) onAddLayer(file);
    },
    [onAddLayer],
  );

  return (
    <div
      className="canvas-area"
      id="canvas-area"
      onClick={() => !hasImage && fileInputRef.current?.click()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      {!hasImage && (
        <div id="drop-zone">
          <div className="drop-prompt">
            Drop an image to start
            <br />
            <span style={{ fontSize: '0.7rem', color: '#444' }}>or click to browse</span>
          </div>
        </div>
      )}
      <canvas
        ref={previewCanvasRef}
        id="preview-canvas"
        style={{ display: hasImage && !showingOriginal ? 'block' : 'none' }}
      />
      <canvas
        ref={originalCanvasRef}
        id="original-canvas"
        style={{ display: hasImage && showingOriginal ? 'block' : 'none' }}
      />
      {hasImage && (
        <button
          className="secondary"
          style={{ position: 'absolute', top: 8, right: 8, fontSize: '0.65rem' }}
          onClick={(e) => {
            e.stopPropagation();
            setShowingOriginal((s) => !s);
          }}
        >
          {showingOriginal ? 'Show Result' : 'Before/After'}
        </button>
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
