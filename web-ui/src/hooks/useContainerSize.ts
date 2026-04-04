import { useEffect, useRef, useState } from 'react';

export interface ContainerSize {
  width: number;
  height: number;
}

/**
 * Measures a container element's available space via ResizeObserver.
 * Returns { width, height } that update when the container resizes.
 */
export function useContainerSize(): [React.RefObject<HTMLDivElement | null>, ContainerSize] {
  const ref = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState<ContainerSize>({ width: 0, height: 0 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const { width, height } = entry.contentRect;
      setSize((prev) => {
        if (prev.width === Math.round(width) && prev.height === Math.round(height)) return prev;
        return { width: Math.round(width), height: Math.round(height) };
      });
    });

    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return [ref, size];
}
