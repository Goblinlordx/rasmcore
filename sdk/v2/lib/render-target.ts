/**
 * RenderTarget — stateful canvas render target for the V2 SDK.
 *
 * Created once per canvas element. Handles WebGPU initialization,
 * shader cache, and GPU-to-canvas blit. Falls back to 2D canvas
 * when WebGPU is unavailable.
 *
 * Usage:
 *   const target = createRenderTarget(canvas, { hdr: true });
 *   pipe.read(source).brightness({ amount: 0.3 }).writeRenderTarget(target);
 *   target.destroy();
 */

import { GpuHandlerV2, type GpuShader } from './gpu-handler';

export interface RenderTargetOptions {
  /** Enable HDR extended tone mapping on HDR-capable displays. */
  hdr?: boolean;
}

/**
 * A persistent render target bound to a canvas element.
 *
 * Owns the GPU device, shader cache, and blit pipeline.
 * Reuses GPU state across multiple writeRenderTarget() calls.
 */
export class RenderTarget {
  private canvas: HTMLCanvasElement | OffscreenCanvas;
  private gpuHandler: GpuHandlerV2 | null = null;
  private ctx2d: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;
  private hdr: boolean;
  private initialized = false;

  constructor(canvas: HTMLCanvasElement | OffscreenCanvas, options?: RenderTargetOptions) {
    this.canvas = canvas;
    this.hdr = options?.hdr ?? false;
  }

  /** Lazy-initialize GPU or 2D fallback on first use. */
  private async ensureInit(): Promise<void> {
    if (this.initialized) return;
    this.initialized = true;

    if (GpuHandlerV2.isAvailable()) {
      try {
        this.gpuHandler = new GpuHandlerV2();
        await this.gpuHandler.setDisplayCanvas(this.canvas as OffscreenCanvas, this.hdr);
        return;
      } catch {
        // WebGPU init failed — fall through to 2D
        this.gpuHandler = null;
      }
    }

    // 2D canvas fallback
    this.ctx2d = (this.canvas as HTMLCanvasElement).getContext('2d');
  }

  /**
   * Write GPU shader results directly to the canvas.
   * Used when the pipeline has a GPU plan (shader chain + input pixels).
   */
  async writeGpu(
    ops: GpuShader[],
    inputPixels: Float32Array,
    width: number,
    height: number,
  ): Promise<boolean> {
    await this.ensureInit();

    if (!this.gpuHandler) return false;

    // Set canvas dimensions
    this.canvas.width = width;
    this.canvas.height = height;

    // Update viewport to fit (no pan/zoom — just fill canvas)
    this.gpuHandler.updateViewport(0, 0, 1.0, width, height, width, height, this.hdr ? 1 : 0);

    // Pre-compile and execute + blit
    await this.gpuHandler.prepare(ops);
    const err = await this.gpuHandler.executeAndDisplay(ops, inputPixels, width, height);
    return err === null;
  }

  /**
   * Write CPU f32 pixels to the canvas.
   * Used when no GPU plan exists or GPU dispatch failed.
   */
  writeCpu(pixels: Float32Array, width: number, height: number): void {
    // Set canvas dimensions
    this.canvas.width = width;
    this.canvas.height = height;

    if (this.gpuHandler?.hasDisplay) {
      // Use GPU blit path even for CPU-rendered pixels (uploads to GPU)
      this.gpuHandler.updateViewport(0, 0, 1.0, width, height, width, height, this.hdr ? 1 : 0);
      this.gpuHandler.displayFromCpu(pixels, width, height);
      return;
    }

    // Pure 2D fallback: quantize f32 RGBA to Uint8ClampedArray
    if (!this.ctx2d) {
      this.ctx2d = (this.canvas as HTMLCanvasElement).getContext('2d');
    }
    if (!this.ctx2d) return;

    const pixelCount = width * height;
    const u8 = new Uint8ClampedArray(pixelCount * 4);
    for (let i = 0; i < pixelCount; i++) {
      const si = i * 4;
      u8[si] = Math.round(Math.max(0, Math.min(1, pixels[si])) * 255);
      u8[si + 1] = Math.round(Math.max(0, Math.min(1, pixels[si + 1])) * 255);
      u8[si + 2] = Math.round(Math.max(0, Math.min(1, pixels[si + 2])) * 255);
      u8[si + 3] = Math.round(Math.max(0, Math.min(1, pixels[si + 3])) * 255);
    }
    const imageData = new ImageData(u8, width, height);
    this.ctx2d.putImageData(imageData, 0, 0);
  }

  /** Clean up GPU resources. */
  destroy(): void {
    if (this.gpuHandler) {
      this.gpuHandler.destroy();
      this.gpuHandler = null;
    }
    this.ctx2d = null;
    this.initialized = false;
  }
}

/**
 * Create a persistent render target bound to a canvas element.
 *
 * The render target handles WebGPU initialization, shader caching,
 * and GPU-to-canvas blit internally. Falls back to 2D canvas when
 * WebGPU is unavailable.
 */
export function createRenderTarget(
  canvas: HTMLCanvasElement | OffscreenCanvas,
  options?: RenderTargetOptions,
): RenderTarget {
  return new RenderTarget(canvas, options);
}
