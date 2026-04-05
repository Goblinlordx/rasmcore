'use client';

import { useState, useRef, useCallback } from 'react';
import type { ParamDescriptor } from '@/lib/types';
import { renderFilterToCanvas, isLoaded, loadWasm } from '@/lib/wasm-loader';
import { ParamControls } from './ParamControls';
import { LiveCodeExample } from './LiveCodeExample';

interface PlaygroundProps {
  filterName: string;
  params: ParamDescriptor[];
  referenceImageUrl: string;
  staticAfterUrl?: string;
}

type Status = 'idle' | 'loading-wasm' | 'rendering' | 'ready' | 'error';

export function Playground({ filterName, params, referenceImageUrl, staticAfterUrl }: PlaygroundProps) {
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string>('');
  const [hasResult, setHasResult] = useState(!!staticAfterUrl);
  const [staticUrl, setStaticUrl] = useState(staticAfterUrl || '');
  const [values, setValues] = useState<Record<string, number | boolean>>(() => {
    const initial: Record<string, number | boolean> = {};
    for (const p of params) {
      if (p.type === 'bool') {
        initial[p.name] = (p.default ?? 0) > 0.5;
      } else {
        let val = p.default ?? 0;
        if (Math.abs(val) < 1e-6 && p.min != null && p.max != null) {
          val = p.min + (p.max - p.min) * 0.3;
        }
        initial[p.name] = val;
      }
    }
    return initial;
  });

  const refImageBytes = useRef<Uint8Array | null>(null);
  const afterCanvasRef = useRef<HTMLCanvasElement>(null);
  const renderSeq = useRef(0);

  const loadRefImage = useCallback(async () => {
    if (refImageBytes.current) return refImageBytes.current;
    const resp = await fetch(referenceImageUrl);
    const ab = await resp.arrayBuffer();
    refImageBytes.current = new Uint8Array(ab);
    return refImageBytes.current;
  }, [referenceImageUrl]);

  // Render directly to canvas — GPU when available, 2D fallback
  const doRender = useCallback(async (vals: Record<string, number | boolean>) => {
    const seq = ++renderSeq.current;
    try {
      if (!isLoaded()) {
        setStatus('loading-wasm');
        await loadWasm();
      }

      setStatus('rendering');
      const imgBytes = await loadRefImage();

      if (seq !== renderSeq.current) return;

      if (afterCanvasRef.current) {
        await renderFilterToCanvas(
          afterCanvasRef.current,
          imgBytes,
          filterName,
          vals,
          referenceImageUrl,
        );
      }

      setStaticUrl('');
      setHasResult(true);
      setStatus('ready');
      setError('');
    } catch (e) {
      if (seq !== renderSeq.current) return;
      setStatus('error');
      console.error('[playground]', e);
      const payload = (e as any)?.payload;
      const msg = payload ? JSON.stringify(payload, null, 2) : (e instanceof Error ? e.message : JSON.stringify(e, null, 2));
      setError(msg);
    }
  }, [filterName, loadRefImage, referenceImageUrl]);

  const onParamChange = useCallback((newValues: Record<string, number | boolean>) => {
    const t0 = performance.now();
    setValues(newValues);
    // No debounce — renderSeq handles stale results. Fire immediately.
    doRender(newValues).then(() => {
      console.log(`[playground] e2e: slider→visible=${(performance.now() - t0).toFixed(0)}ms`);
    });
  }, [doRender]);

  const [activated, setActivated] = useState(false);
  const activate = useCallback(() => {
    if (activated) return;
    setActivated(true);
    doRender(values);
  }, [activated, doRender, values]);

  // Split view state
  const containerRef = useRef<HTMLDivElement>(null);
  const [splitPos, setSplitPos] = useState(50);
  const dragging = useRef(false);

  const updateSplit = useCallback((clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    setSplitPos(x * 100);
  }, []);

  return (
    <div style={{ margin: '1.5rem 0', border: '1px solid var(--border)', borderRadius: 8, padding: '1rem', background: 'var(--bg-sidebar)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.8rem' }}>
        <h3 style={{ margin: 0, fontSize: '1rem', color: 'var(--heading)' }}>Interactive Playground</h3>
        {status === 'loading-wasm' && <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Loading WASM...</span>}
        {status === 'rendering' && <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Rendering...</span>}
        {status === 'error' && <span style={{ fontSize: '0.8rem', color: '#f85149' }}>Error: {error}</span>}
      </div>

      {hasResult ? (
        <div
          ref={containerRef}
          style={{
            position: 'relative', display: 'inline-block', maxWidth: '100%',
            cursor: 'col-resize', borderRadius: 6, overflow: 'hidden',
            border: '1px solid var(--border)', userSelect: 'none',
          }}
          onMouseMove={e => { if (dragging.current) { e.preventDefault(); updateSplit(e.clientX); } }}
          onMouseUp={() => { dragging.current = false; }}
          onMouseLeave={() => { dragging.current = false; }}
          onClick={e => { if (!activated) activate(); updateSplit(e.clientX); }}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={referenceImageUrl} alt="Before" draggable={false} style={{ display: 'block', maxWidth: '100%', pointerEvents: 'none' }} />

          {/* Canvas always mounted — GPU binds once, survives across renders */}
          <canvas ref={afterCanvasRef} style={{
            position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
            objectFit: 'cover', pointerEvents: 'none',
            clipPath: `inset(0 0 0 ${splitPos}%)`,
            display: staticUrl ? 'none' : 'block',
          }} />
          {/* Static SSG image shown until first live render */}
          {staticUrl && (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img src={staticUrl} alt="After" draggable={false} style={{
              position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
              objectFit: 'cover', pointerEvents: 'none',
              clipPath: `inset(0 0 0 ${splitPos}%)`,
            }} />
          )}

          <div
            style={{
              position: 'absolute', top: 0, bottom: 0, left: `${splitPos}%`,
              width: 3, background: 'var(--link)', transform: 'translateX(-50%)', zIndex: 2,
            }}
            onMouseDown={e => { e.preventDefault(); dragging.current = true; }}
          >
            <div style={{
              position: 'absolute', top: '50%', left: '50%',
              transform: 'translate(-50%,-50%)', width: 28, height: 28,
              background: 'var(--link)', borderRadius: '50%', border: '2px solid var(--bg)',
            }} />
          </div>
          <div style={{
            position: 'absolute', bottom: 8, left: 0, right: 0,
            display: 'flex', justifyContent: 'space-between', padding: '0 12px',
            pointerEvents: 'none', zIndex: 3,
          }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#fff', background: 'rgba(0,0,0,0.6)', padding: '2px 8px', borderRadius: 3 }}>Before</span>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#fff', background: 'rgba(0,0,0,0.6)', padding: '2px 8px', borderRadius: 3 }}>After</span>
          </div>
        </div>
      ) : (
        <div
          style={{
            padding: '2rem', textAlign: 'center', border: '1px dashed var(--border)',
            borderRadius: 6, cursor: 'pointer', color: 'var(--text-muted)',
          }}
          onClick={activate}
        >
          Click to load interactive preview
        </div>
      )}

      {params.length > 0 && (
        <ParamControls params={params} values={values} onChange={onParamChange} />
      )}

      <LiveCodeExample name={filterName} values={values} />
    </div>
  );
}
