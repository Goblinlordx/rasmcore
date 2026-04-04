'use client';

import { useRef, useCallback, useState } from 'react';

interface SplitViewProps {
  beforeSrc: string;
  afterSrc: string;
}

export function SplitView({ beforeSrc, afterSrc }: SplitViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState(50);
  const dragging = useRef(false);

  const update = useCallback((clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    setPos(x * 100);
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
  }, []);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (dragging.current) { e.preventDefault(); update(e.clientX); }
  }, [update]);

  const onMouseUp = useCallback(() => { dragging.current = false; }, []);

  return (
    <div className="split-view">
      <div className="split-container" ref={containerRef}
        onMouseMove={onMouseMove} onMouseUp={onMouseUp} onMouseLeave={onMouseUp}
        onClick={(e) => update(e.clientX)}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className="split-before" src={beforeSrc} alt="Before" draggable={false} />
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className="split-after" src={afterSrc} alt="After" draggable={false}
          style={{ clipPath: `inset(0 0 0 ${pos}%)` }} />
        <div className="split-divider" style={{ left: `${pos}%` }} onMouseDown={onMouseDown}>
          <div className="split-handle" />
        </div>
        <div className="split-labels">
          <span className="split-label">Before</span>
          <span className="split-label">After</span>
        </div>
      </div>
    </div>
  );
}
