/**
 * WebGPU handler for pipeline gpu-execute calls.
 *
 * Matches the WIT gpu-execute signature:
 *   gpu-execute(ops: list<gpu-op>, input: list<u8>, width: u32, height: u32) -> result<list<u8>, gpu-error>
 *
 * Manages:
 * - Lazy WebGPU device initialization (first call)
 * - Shader compilation cache (keyed by source hash)
 * - Ping-pong storage buffers for multi-op batches
 * - Input upload via queue.writeBuffer, output readback via mapAsync
 */

// ─── Types matching WIT gpu-op record ──────────────────────────────────────

/** Buffer element format for GPU storage buffers. */
export type BufferFormat = 'u32-packed' | 'f32-vec4';

export interface GpuOp {
  shader: string;
  entryPoint: string;
  workgroupX: number;
  workgroupY: number;
  workgroupZ: number;
  params: Uint8Array;
  extraBuffers: Uint8Array[];
  bufferFormat: BufferFormat;
}

export type GpuError =
  | { tag: 'not-available'; val: string }
  | { tag: 'shader-error'; val: string }
  | { tag: 'execution-error'; val: string };

export type GpuResult = { ok: Uint8Array } | { err: GpuError };

/** Bytes per pixel for a given buffer format. */
function bytesPerPixel(format: BufferFormat): number {
  return format === 'f32-vec4' ? 16 : 4;
}

// ─── Shader Cache ──────────────────────────────────────────────────────────

/** FNV-1a hash of shader source for cache keying. */
function hashShaderSource(source: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < source.length; i++) {
    hash ^= source.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16);
}

interface CachedPipeline {
  module: GPUShaderModule;
  pipelines: Map<string, GPUComputePipeline>; // keyed by entry point
}

// ─── GPU Handler ───────────────────────────────────────────────────────────

export class GpuHandler {
  private device: GPUDevice | null = null;
  private adapter: GPUAdapter | null = null;
  private shaderCache: Map<string, CachedPipeline> = new Map();
  private initPromise: Promise<void> | null = null;
  private available = true;

  /** Check if WebGPU is available without initializing. */
  static isAvailable(): boolean {
    return typeof navigator !== 'undefined' && 'gpu' in navigator;
  }

  /** Lazy-initialize the WebGPU device on first use. */
  private async ensureDevice(): Promise<void> {
    if (this.device) return;
    if (!this.available) throw new GpuNotAvailableError('WebGPU previously failed to initialize');

    if (this.initPromise) {
      await this.initPromise;
      return;
    }

    this.initPromise = this.initDevice();
    await this.initPromise;
  }

  private async initDevice(): Promise<void> {
    if (!GpuHandler.isAvailable()) {
      this.available = false;
      throw new GpuNotAvailableError('WebGPU not supported in this browser');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      this.available = false;
      throw new GpuNotAvailableError('No WebGPU adapter found');
    }

    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });

    device.lost.then((info) => {
      console.warn('[gpu-handler] device lost:', info.message);
      this.device = null;
      this.adapter = null;
      this.shaderCache.clear();
      this.initPromise = null;
      // Don't set available=false — device can be re-acquired
    });

    this.adapter = adapter;
    this.device = device;
  }

  /** Get or compile a shader module + compute pipeline for a given source + entry point + format. */
  private getOrCreatePipeline(device: GPUDevice, shader: string, entryPoint: string, extraBufferCount: number, bufferFormat: BufferFormat = 'u32-packed'): GPUComputePipeline {
    // Include buffer format in cache key to avoid collisions between u32/f32 variants
    const hash = hashShaderSource(shader + ':' + bufferFormat);
    let cached = this.shaderCache.get(hash);

    if (!cached) {
      const module = device.createShaderModule({ code: shader });
      cached = { module, pipelines: new Map() };
      this.shaderCache.set(hash, cached);
    }

    // Pipeline key includes entry point and extra buffer count (affects bind group layout)
    const pipelineKey = `${entryPoint}:${extraBufferCount}`;
    let pipeline = cached.pipelines.get(pipelineKey);

    if (!pipeline) {
      // Build bind group layout entries:
      //   @binding(0) input: storage<read>
      //   @binding(1) output: storage<read_write>
      //   @binding(2) params: uniform
      //   @binding(3..N) extra: storage<read>
      const entries: GPUBindGroupLayoutEntry[] = [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ];
      for (let i = 0; i < extraBufferCount; i++) {
        entries.push({
          binding: 3 + i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' },
        });
      }

      const bindGroupLayout = device.createBindGroupLayout({ entries });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

      pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module: cached.module, entryPoint },
      });
      cached.pipelines.set(pipelineKey, pipeline);
    }

    return pipeline;
  }

  /**
   * Execute a batch of GPU compute operations on pixel data.
   * Matches the WIT gpu-execute signature.
   *
   * Operations are chained: output of op[i] = input of op[i+1].
   * All intermediate buffers stay in GPU memory via ping-pong.
   * Only the final output is read back to CPU.
   */
  async execute(ops: GpuOp[], input: Uint8Array, width: number, height: number, bufferFormat: BufferFormat = 'u32-packed'): Promise<GpuResult> {
    if (ops.length === 0) {
      return { ok: input };
    }

    try {
      await this.ensureDevice();
    } catch (e) {
      return { err: { tag: 'not-available', val: (e as Error).message } };
    }

    const device = this.device!;
    const bpp = bytesPerPixel(bufferFormat);
    const pixelBytes = width * height * bpp;

    // Validate input size
    if (input.byteLength !== pixelBytes) {
      return {
        err: {
          tag: 'execution-error',
          val: `Input size ${input.byteLength} != expected ${pixelBytes} (${width}x${height}x${bpp})`,
        },
      };
    }

    try {
      // Create ping-pong storage buffers (A and B)
      const bufferA = device.createBuffer({
        size: pixelBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const bufferB = device.createBuffer({
        size: pixelBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

      // Upload input pixels to buffer A
      device.queue.writeBuffer(bufferA, 0, input);

      // Process each op, ping-ponging between A and B
      let readBuffer = bufferA;
      let writeBuffer = bufferB;

      for (let i = 0; i < ops.length; i++) {
        const op = ops[i];

        // Get or create pipeline
        let pipeline: GPUComputePipeline;
        try {
          pipeline = this.getOrCreatePipeline(device, op.shader, op.entryPoint, op.extraBuffers.length, bufferFormat);
        } catch (e) {
          bufferA.destroy();
          bufferB.destroy();
          return { err: { tag: 'shader-error', val: `Op ${i} (${op.entryPoint}): ${(e as Error).message}` } };
        }

        // Create uniform buffer for params (16-byte aligned)
        const alignedParamSize = Math.max(16, Math.ceil(op.params.byteLength / 16) * 16);
        const paramBuffer = device.createBuffer({
          size: alignedParamSize,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(paramBuffer, 0, op.params);

        // Create extra storage buffers
        const extraGpuBuffers: GPUBuffer[] = [];
        for (const extra of op.extraBuffers) {
          const buf = device.createBuffer({
            size: Math.max(4, extra.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          device.queue.writeBuffer(buf, 0, extra);
          extraGpuBuffers.push(buf);
        }

        // Build bind group
        const entries: GPUBindGroupEntry[] = [
          { binding: 0, resource: { buffer: readBuffer } },
          { binding: 1, resource: { buffer: writeBuffer } },
          { binding: 2, resource: { buffer: paramBuffer } },
        ];
        for (let j = 0; j < extraGpuBuffers.length; j++) {
          entries.push({ binding: 3 + j, resource: { buffer: extraGpuBuffers[j] } });
        }

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries,
        });

        // Dispatch compute shader
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);

        const dispatchX = Math.ceil(width / op.workgroupX);
        const dispatchY = Math.ceil(height / op.workgroupY);
        const dispatchZ = Math.ceil(1 / op.workgroupZ);
        pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        pass.end();

        device.queue.submit([encoder.finish()]);

        // Cleanup per-op buffers
        paramBuffer.destroy();
        for (const buf of extraGpuBuffers) buf.destroy();

        // Ping-pong: swap read and write buffers
        [readBuffer, writeBuffer] = [writeBuffer, readBuffer];
      }

      // Read back the final output (now in readBuffer after last swap)
      const stagingBuffer = device.createBuffer({
        size: pixelBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const copyEncoder = device.createCommandEncoder();
      copyEncoder.copyBufferToBuffer(readBuffer, 0, stagingBuffer, 0, pixelBytes);
      device.queue.submit([copyEncoder.finish()]);

      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const outputData = new Uint8Array(stagingBuffer.getMappedRange().slice(0));
      stagingBuffer.unmap();

      // Cleanup
      bufferA.destroy();
      bufferB.destroy();
      stagingBuffer.destroy();

      return { ok: outputData };
    } catch (e) {
      return { err: { tag: 'execution-error', val: (e as Error).message } };
    }
  }

  /** Release all GPU resources. */
  destroy(): void {
    this.shaderCache.clear();
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }
    this.adapter = null;
    this.initPromise = null;
  }

  /** Returns true if the handler has a live device. */
  get isInitialized(): boolean {
    return this.device !== null;
  }

  /** Number of cached shader modules. */
  get cacheSize(): number {
    return this.shaderCache.size;
  }
}

class GpuNotAvailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GpuNotAvailableError';
  }
}

// ─── WIT Import Provider ───────────────────────────────────────────────────

/**
 * Create the gpu-execute function matching the WIT import signature.
 * This is the function that gets wired into the WASM component imports.
 *
 * Returns null if WebGPU is not available (graceful fallback — pipeline uses CPU).
 */
export function createGpuExecuteImport(): ((ops: GpuOp[], input: Uint8Array, width: number, height: number, bufferFormat: BufferFormat) => Promise<GpuResult>) | null {
  if (!GpuHandler.isAvailable()) {
    return null;
  }

  const handler = new GpuHandler();

  return async (ops: GpuOp[], input: Uint8Array, width: number, height: number, bufferFormat: BufferFormat = 'u32-packed'): Promise<GpuResult> => {
    return handler.execute(ops, input, width, height, bufferFormat);
  };
}
