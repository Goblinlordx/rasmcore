import { useCallback, useEffect, useRef, useState } from 'react';
import { getWideGamutContext } from '../utils/canvasColorSpace';

export interface WorkerTimings {
  totalMs: number;
  ops: { name: string; ms: number }[];
}

interface WorkerState {
  ready: boolean;
  processing: boolean;
  imageInfo: { width: number; height: number } | null;
  timings: WorkerTimings | null;
  error: string | null;
}

export function useWorker() {
  const workerRef = useRef<Worker | null>(null);
  const [state, setState] = useState<WorkerState>({
    ready: false,
    processing: false,
    imageInfo: null,
    timings: null,
    error: null,
  });
  // Use refs for mutable flags so sendMessage doesn't need to re-create on every state change
  const readyRef = useRef(false);
  const processingRef = useRef(false);
  const queueRef = useRef<unknown>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const originalCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const onLoadedRef = useRef<(() => void) | null>(null);
  /** Called when a 'warm' mode render completes (cache populated) */
  const onWarmCompleteRef = useRef<(() => void) | null>(null);

  const setProcessing = useCallback((active: boolean) => {
    processingRef.current = active;
    setState((s) => ({ ...s, processing: active }));
  }, []);

  const setReady = useCallback((ready: boolean) => {
    readyRef.current = ready;
    setState((s) => ({ ...s, ready }));
  }, []);

  useEffect(() => {
    const w = new Worker(new URL('../v2-pipeline-worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = w;
    w.postMessage({ type: 'init' });

    w.onmessage = (e: MessageEvent) => {
      const { type } = e.data;

      if (type === 'ready') {
        setReady(true);
        return;
      }

      if (type === 'loaded') {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        processingRef.current = false;
        setState((s) => ({
          ...s,
          processing: false,
          imageInfo: e.data.info,
          error: null,
        }));
        if (onLoadedRef.current) onLoadedRef.current();
        return;
      }

      if (type === 'result') {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);

        const isWarm = e.data.mode === 'warm';

        // Always draw to canvas — warm mode upgrades proxy to full-res silently
        const blob = new Blob([e.data.png], { type: 'image/png' });
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          const canvas = previewCanvasRef.current;
          if (canvas) {
            canvas.width = img.width;
            canvas.height = img.height;
            getWideGamutContext(canvas)?.drawImage(img, 0, 0);
          }
          URL.revokeObjectURL(url);
        };
        img.src = url;

        processingRef.current = false;
        if (e.data.timings) {
          setState((s) => ({
            ...s,
            processing: false,
            error: null,
            timings: { totalMs: e.data.totalMs, ops: e.data.timings },
          }));
        } else {
          setState((s) => ({ ...s, processing: false, error: null }));
        }

        // Notify warm completion
        if (isWarm && onWarmCompleteRef.current) onWarmCompleteRef.current();

        // Drain queue
        if (queueRef.current) {
          const next = queueRef.current as { imageBytes?: ArrayBuffer };
          queueRef.current = null;
          processingRef.current = true;
          setState((s) => ({ ...s, processing: true }));
          w.postMessage(next, next.imageBytes ? [next.imageBytes] : []);
        }
        return;
      }

      if (type === 'exported') {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        processingRef.current = false;
        setState((s) => ({ ...s, processing: false }));
        const blob = new Blob([e.data.data], { type: e.data.mime });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `rasmcore-pipeline.${e.data.format}`;
        a.click();
        return;
      }

      if (type === 'error') {
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
        processingRef.current = false;
        setState((s) => ({
          ...s,
          processing: false,
          error: e.data.message,
        }));
        if (queueRef.current) {
          const next = queueRef.current as { imageBytes?: ArrayBuffer };
          queueRef.current = null;
          processingRef.current = true;
          setState((s) => ({ ...s, processing: true }));
          w.postMessage(next, next.imageBytes ? [next.imageBytes] : []);
        }
      }
    };

    return () => {
      w.terminate();
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [setProcessing, setReady]);

  // Stable sendMessage — uses refs, never stale
  const sendMessage = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (msg: any) => {
      if (!workerRef.current || !readyRef.current) return;
      if (processingRef.current) {
        queueRef.current = msg;
        return;
      }
      processingRef.current = true;
      setState((s) => ({ ...s, processing: true, error: null }));
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = setTimeout(() => {
        if (processingRef.current) {
          processingRef.current = false;
          setState((s) => ({ ...s, processing: false, error: 'Processing timed out' }));
        }
      }, 15000);
      workerRef.current.postMessage(msg, msg.imageBytes ? [msg.imageBytes] : []);
    },
    [], // stable — no deps, uses refs only
  );

  return {
    ...state,
    sendMessage,
    previewCanvasRef,
    originalCanvasRef,
    onLoadedRef,
    onWarmCompleteRef,
  };
}
