import { describe, it, expect } from 'vitest';
import { fitZoom, computeTransformCSS, formatZoom } from '../useCanvasTransform';

describe('fitZoom', () => {
  it('fits landscape image in container', () => {
    const zoom = fitZoom(800, 600, 1600, 1200);
    expect(zoom).toBe(0.5); // min(800/1600, 600/1200) = 0.5
  });

  it('fits portrait image in landscape container', () => {
    const zoom = fitZoom(800, 600, 400, 800);
    expect(zoom).toBe(0.75); // min(800/400=2, 600/800=0.75) = 0.75
  });

  it('fits square image in landscape container', () => {
    const zoom = fitZoom(800, 600, 500, 500);
    expect(zoom).toBe(1.2); // min(800/500=1.6, 600/500=1.2) = 1.2
  });

  it('returns 1 for zero-sized inputs', () => {
    expect(fitZoom(0, 600, 800, 600)).toBe(1);
    expect(fitZoom(800, 0, 800, 600)).toBe(1);
    expect(fitZoom(800, 600, 0, 600)).toBe(1);
  });

  it('large image scales down', () => {
    const zoom = fitZoom(1920, 1080, 8000, 6000);
    expect(zoom).toBeCloseTo(0.18, 1);
  });
});

describe('computeTransformCSS', () => {
  it('centers image at fit zoom', () => {
    const style = computeTransformCSS(
      { zoom: 0.5, panX: 0, panY: 0 },
      { width: 800, height: 600 },
      1600,
      1200,
    );
    expect(style.width).toBe(1600);
    expect(style.height).toBe(1200);
    expect(style.transformOrigin).toBe('0 0');
    // scaledW = 1600*0.5 = 800, offsetX = (800-800)/2 = 0
    // scaledH = 1200*0.5 = 600, offsetY = (600-600)/2 = 0
    expect(style.transform).toBe('translate(0px, 0px) scale(0.5)');
  });

  it('offsets for pan', () => {
    const style = computeTransformCSS(
      { zoom: 1.0, panX: 50, panY: -30 },
      { width: 800, height: 600 },
      800,
      600,
    );
    // scaledW = 800, offsetX = (800-800)/2 + 50*1 = 50
    // scaledH = 600, offsetY = (600-600)/2 + (-30)*1 = -30
    expect(style.transform).toBe('translate(50px, -30px) scale(1)');
  });

  it('enables pixelated rendering at high zoom', () => {
    const style = computeTransformCSS(
      { zoom: 5.0, panX: 0, panY: 0 },
      { width: 800, height: 600 },
      100,
      100,
    );
    expect(style.imageRendering).toBe('pixelated');
  });
});

describe('formatZoom', () => {
  it('shows Fit when at fit zoom', () => {
    expect(formatZoom(0.5, 0.5)).toBe('Fit');
  });

  it('shows percentage when zoomed', () => {
    expect(formatZoom(1.0, 0.5)).toBe('100%');
    expect(formatZoom(2.0, 0.5)).toBe('200%');
    expect(formatZoom(0.25, 0.5)).toBe('25%');
  });
});
