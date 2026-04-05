/**
 * V2 WebGPU handler — f32-only, no format dispatch.
 *
 * All pixel data is Float32Array (vec4<f32> per pixel).
 * No BufferFormat enum. No U32Packed. No pack/unpack.
 * Always array<vec4<f32>> storage buffers.
 *
 * Shader bodies use load_pixel/store_pixel from io_f32 fragment
 * (auto-composed by the pipeline).
 */

// @ts-ignore — bundler resolves raw WGSL import
import BLIT_SHADER from './shaders/display-blit.wgsl?raw';

export interface GpuShader {
  /** Complete WGSL source (io_f32 + body, already composed). */
  source: string;
  entryPoint: string;
  workgroupX: number;
  workgroupY: number;
  workgroupZ: number;
  params: Uint8Array;
  extraBuffers: Uint8Array[];
}

export type GpuError =
  | { tag: 'not-available'; val: string }
  | { tag: 'shader-error'; val: string }
  | { tag: 'execution-error'; val: string };

export type GpuResult = { ok: Float32Array } | { err: GpuError };

/** Viewport uniform layout — must match Viewport struct in display-blit.wgsl. */
const VIEWPORT_BYTE_SIZE = 32; // 7 f32 + 1 u32 = 32 bytes

/** FNV-1a hash of shader source for cache keying. */
function hashSource(source: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < source.length; i++) {
    hash ^= source.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16);
}

/**
 * V2 GPU handler — f32-only.
 *
 * All buffers are array<vec4<f32>>. No BufferFormat. No format dispatch.
 * Bytes per pixel = 16 (4 channels * 4 bytes). Always.
 */
export class GpuHandlerV2 {
  private device: GPUDevice | null = null;
  private shaderCache: Map<string, GPUComputePipeline> = new Map();
  private initPromise: Promise<void> | null = null;
  private available = true;

  // Display surface state (preview / processed view)
  private displayCanvas: OffscreenCanvas | null = null;
  private canvasCtx: GPUCanvasContext | null = null;
  private blitPipeline: GPURenderPipeline | null = null;
  private blitBindGroupLayout: GPUBindGroupLayout | null = null;
  private viewportBuf: GPUBuffer | null = null;
  private lastOutputBuf: GPUBuffer | null = null;
  private lastImageWidth = 0;
  private lastImageHeight = 0;

  // Original view display surface (second canvas, same device + blit pipeline)
  private origCanvas: OffscreenCanvas | null = null;
  private origCtx: GPUCanvasContext | null = null;
  private origViewportBuf: GPUBuffer | null = null;
  private sourcePixelBuf: GPUBuffer | null = null;
  private sourceImageWidth = 0;
  private sourceImageHeight = 0;
  /** Buffers queued for deferred destruction — destroyed on next submit, not immediately. */
  private pendingDestroy: GPUBuffer[] = [];

  /** Queue a buffer for destruction on the next submit (not immediately). */
  private deferDestroy(buf: GPUBuffer): void {
    this.pendingDestroy.push(buf);
  }

  /** Flush deferred destroys — safe to call before a new submit. */
  private flushDeferred(): void {
    for (const buf of this.pendingDestroy) buf.destroy();
    this.pendingDestroy.length = 0;
  }

  static isAvailable(): boolean {
    return typeof navigator !== 'undefined' && 'gpu' in navigator;
  }

  private async ensureDevice(): Promise<void> {
    if (this.device) return;
    if (!this.available) throw new Error('WebGPU not available');
    if (this.initPromise) { await this.initPromise; return; }
    this.initPromise = this.initDevice();
    await this.initPromise;
  }

  private async initDevice(): Promise<void> {
    if (!GpuHandlerV2.isAvailable()) {
      this.available = false;
      throw new Error('WebGPU not supported');
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) { this.available = false; throw new Error('No adapter'); }
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    this.device.lost.then(() => {
      this.device = null;
      this.shaderCache.clear();
      this.initPromise = null;
    });
  }

  /**
   * Execute GPU shaders on f32 pixel data.
   *
   * Always vec4<f32>. Always 16 bytes per pixel. No format variants.
   */
  async execute(
    ops: GpuShader[],
    input: Float32Array,
    width: number,
    height: number,
  ): Promise<GpuResult> {
    if (ops.length === 0) return { ok: input };

    try { await this.ensureDevice(); }
    catch (e) { return { err: { tag: 'not-available', val: (e as Error).message } }; }

    const device = this.device!;
    const pixelCount = width * height;
    const floatCount = pixelCount * 4;
    const byteCount = floatCount * 4; // f32 = 4 bytes

    if (input.length !== floatCount) {
      return { err: { tag: 'execution-error',
        val: `Input ${input.length} floats != expected ${floatCount} (${width}x${height}x4)` } };
    }

    try {
      // Ping-pong f32 buffers
      const bufA = device.createBuffer({
        size: byteCount,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const bufB = device.createBuffer({
        size: byteCount,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

      device.queue.writeBuffer(bufA, 0, input);
      let readBuf = bufA;
      let writeBuf = bufB;

      for (let i = 0; i < ops.length; i++) {
        const op = ops[i];
        const hash = hashSource(op.source) + ':' + op.entryPoint;
        let pipeline = this.shaderCache.get(hash);

        if (!pipeline) {
          const module = device.createShaderModule({ code: op.source });
          const entries: GPUBindGroupLayoutEntry[] = [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          ];
          for (let j = 0; j < op.extraBuffers.length; j++) {
            entries.push({ binding: 3 + j, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
          }
          const layout = device.createBindGroupLayout({ entries });
          pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module, entryPoint: op.entryPoint },
          });
          this.shaderCache.set(hash, pipeline);
        }

        // Uniform buffer
        const paramSize = Math.max(16, Math.ceil(op.params.byteLength / 16) * 16);
        const paramBuf = device.createBuffer({
          size: paramSize,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(paramBuf, 0, op.params);

        // Extra buffers
        const extras: GPUBuffer[] = op.extraBuffers.map(data => {
          const buf = device.createBuffer({
            size: Math.max(4, data.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          device.queue.writeBuffer(buf, 0, data);
          return buf;
        });

        // Bind group
        const bgEntries: GPUBindGroupEntry[] = [
          { binding: 0, resource: { buffer: readBuf } },
          { binding: 1, resource: { buffer: writeBuf } },
          { binding: 2, resource: { buffer: paramBuf } },
        ];
        extras.forEach((buf, j) => bgEntries.push({ binding: 3 + j, resource: { buffer: buf } }));

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: bgEntries,
        });

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
          Math.ceil(width / op.workgroupX),
          Math.ceil(height / op.workgroupY),
          1,
        );
        pass.end();
        this.flushDeferred();
        device.queue.submit([encoder.finish()]);

        // paramBuf and extras GC'd when unreferenced
        [readBuf, writeBuf] = [writeBuf, readBuf];
      }

      // Readback
      const staging = device.createBuffer({
        size: byteCount,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const copyEnc = device.createCommandEncoder();
      copyEnc.copyBufferToBuffer(readBuf, 0, staging, 0, byteCount);
      this.flushDeferred();
      device.queue.submit([copyEnc.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      const output = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();

      bufA.destroy();
      bufB.destroy();
      staging.destroy();

      return { ok: output };
    } catch (e) {
      return { err: { tag: 'execution-error', val: (e as Error).message } };
    }
  }

  /**
   * Pre-compile shader sources into GPUComputePipelines.
   * Warms the cache so subsequent execute()/executeAndDisplay() calls
   * get O(1) pipeline lookups with no shader compilation stalls.
   */
  async prepare(shaders: GpuShader[]): Promise<void> {
    if (shaders.length === 0) return;

    try { await this.ensureDevice(); }
    catch { return; }

    const device = this.device!;
    for (const op of shaders) {
      const hash = hashSource(op.source) + ':' + op.entryPoint;
      if (this.shaderCache.has(hash)) continue;

      const module = device.createShaderModule({ code: op.source });
      const entries: GPUBindGroupLayoutEntry[] = [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ];
      for (let j = 0; j < op.extraBuffers.length; j++) {
        entries.push({ binding: 3 + j, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
      }
      const layout = device.createBindGroupLayout({ entries });
      const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module, entryPoint: op.entryPoint },
      });
      this.shaderCache.set(hash, pipeline);
    }
  }

  /** Number of cached shader pipelines. */
  get cacheSize(): number {
    return this.shaderCache.size;
  }

  // ─── Display Surface ────────────────────────────────────────────────────

  /**
   * Configure a WebGPU canvas for direct display.
   * Call once after receiving an OffscreenCanvas from the main thread.
   */
  async setDisplayCanvas(canvas: OffscreenCanvas, hdr: boolean): Promise<void> {
    await this.ensureDevice();
    const device = this.device!;
    this.displayCanvas = canvas;

    const ctx = canvas.getContext('webgpu') as GPUCanvasContext;
    const format: GPUTextureFormat = 'rgba16float';
    const toneMapping: GPUCanvasToneMapping = { mode: hdr ? 'extended' : 'standard' };
    ctx.configure({
      device,
      format,
      alphaMode: 'premultiplied',
      toneMapping,
    });
    this.canvasCtx = ctx;

    // Blit bind group layout: storage buffer (pixels) + uniform (viewport)
    this.blitBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });

    const shaderModule = device.createShaderModule({ code: BLIT_SHADER });
    this.blitPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.blitBindGroupLayout] }),
      vertex: { module: shaderModule, entryPoint: 'vs_main' },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{
          format,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });

    // Persistent viewport uniform buffer
    this.viewportBuf = device.createBuffer({
      size: VIEWPORT_BYTE_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  // ─── Original View Display ──────────────────────────────────────────────

  /** Configure a second WebGPU canvas for the original/before view. */
  async setOriginalCanvas(canvas: OffscreenCanvas, hdr: boolean): Promise<void> {
    await this.ensureDevice();
    const device = this.device!;
    this.origCanvas = canvas;
    const ctx = canvas.getContext('webgpu') as GPUCanvasContext;
    ctx.configure({
      device,
      format: 'rgba16float',
      alphaMode: 'premultiplied',
      toneMapping: { mode: hdr ? 'extended' : 'standard' },
    });
    this.origCtx = ctx;

    // Ensure blit pipeline exists (may already be created by setDisplayCanvas)
    if (!this.blitBindGroupLayout) {
      this.blitBindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        ],
      });
      const shaderModule = device.createShaderModule({ code: BLIT_SHADER });
      this.blitPipeline = device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.blitBindGroupLayout] }),
        vertex: { module: shaderModule, entryPoint: 'vs_main' },
        fragment: {
          module: shaderModule, entryPoint: 'fs_main',
          targets: [{ format: 'rgba16float' as GPUTextureFormat, blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          }}],
        },
        primitive: { topology: 'triangle-list' },
      });
    }
    this.origViewportBuf = device.createBuffer({
      size: VIEWPORT_BYTE_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  get hasOriginalDisplay(): boolean {
    return this.origCtx !== null && this.blitPipeline !== null;
  }

  /** Upload source pixels for original view. Blit happens on next viewport update. */
  storeSourcePixels(pixels: Float32Array, width: number, height: number): void {
    if (!this.device) return;
    if (this.sourcePixelBuf) this.sourcePixelBuf.destroy();
    this.sourcePixelBuf = this.device.createBuffer({
      size: pixels.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.sourcePixelBuf, 0, pixels);
    this.sourceImageWidth = width;
    this.sourceImageHeight = height;

    // If original canvas is already sized by a viewport message, update viewport
    // with correct image dimensions and blit immediately
    if (this.origCtx && this.origCanvas && this.origCanvas.width > 1 && this.origViewportBuf) {
      this.updateOriginalViewport(0, 0, 1, this.origCanvas.width, this.origCanvas.height, 0);
      this.blitOriginal();
    }
  }

  updateOriginalViewport(panX: number, panY: number, zoom: number, canvasWidth: number, canvasHeight: number, toneMode: number): void {
    if (!this.device || !this.origViewportBuf) return;
    const data = new ArrayBuffer(VIEWPORT_BYTE_SIZE);
    const f = new Float32Array(data, 0, 7);
    const u = new Uint32Array(data, 28, 1);
    f[0] = canvasWidth; f[1] = canvasHeight;
    f[2] = this.sourceImageWidth; f[3] = this.sourceImageHeight;
    f[4] = panX; f[5] = panY; f[6] = zoom;
    u[0] = toneMode;
    this.device.queue.writeBuffer(this.origViewportBuf, 0, data);
  }

  resizeOriginalDisplay(width: number, height: number): void {
    if (!this.origCanvas) return;
    if (this.origCanvas.width !== width) this.origCanvas.width = width;
    if (this.origCanvas.height !== height) this.origCanvas.height = height;
  }

  blitOriginal(): void {
    if (!this.device || !this.origCtx || !this.blitPipeline || !this.blitBindGroupLayout || !this.origViewportBuf || !this.sourcePixelBuf) return;
    const bindGroup = this.device.createBindGroup({
      layout: this.blitBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.sourcePixelBuf } },
        { binding: 1, resource: { buffer: this.origViewportBuf } },
      ],
    });
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.origCtx.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear', storeOp: 'store',
      }],
    });
    pass.setPipeline(this.blitPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Whether a display canvas has been configured. */
  get hasDisplay(): boolean {
    return this.canvasCtx !== null && this.blitPipeline !== null;
  }

  /** Width of the last rendered image (preview resolution). */
  get imageWidth(): number { return this.lastImageWidth; }

  /** Height of the last rendered image (preview resolution). */
  get imageHeight(): number { return this.lastImageHeight; }

  /** Resize the OffscreenCanvas from the worker thread (skip if unchanged — resize clears the canvas). */
  resizeDisplay(width: number, height: number): void {
    if (!this.displayCanvas) return;
    if (this.displayCanvas.width !== width) this.displayCanvas.width = width;
    if (this.displayCanvas.height !== height) this.displayCanvas.height = height;
  }

  /**
   * Update viewport uniform for pan/zoom.
   * Call on every pan/zoom change — cheap (just a writeBuffer).
   */
  updateViewport(
    panX: number, panY: number, zoom: number,
    canvasWidth: number, canvasHeight: number,
    imageWidth: number, imageHeight: number,
    toneMode: number,
  ): void {
    if (!this.device || !this.viewportBuf) return;
    const data = new ArrayBuffer(VIEWPORT_BYTE_SIZE);
    const f = new Float32Array(data, 0, 7);
    const u = new Uint32Array(data, 28, 1);
    f[0] = canvasWidth;
    f[1] = canvasHeight;
    f[2] = imageWidth;
    f[3] = imageHeight;
    f[4] = panX;
    f[5] = panY;
    f[6] = zoom;
    u[0] = toneMode;
    this.device.queue.writeBuffer(this.viewportBuf, 0, data);
  }

  /**
   * Execute GPU shaders and blit the result directly to the canvas.
   * No CPU readback — the result stays on the GPU.
   */
  async executeAndDisplay(
    ops: GpuShader[],
    input: Float32Array,
    width: number,
    height: number,
  ): Promise<GpuError | null> {
    if (!this.canvasCtx || !this.blitPipeline || !this.blitBindGroupLayout || !this.viewportBuf) {
      return { tag: 'not-available', val: 'Display canvas not configured' };
    }

    try { await this.ensureDevice(); }
    catch (e) { return { tag: 'not-available', val: (e as Error).message }; }

    const device = this.device!;

    // Viewport is managed by handleViewport from the main thread.
    // But if no viewport has been set yet (first render), set a default.
    if (this.lastImageWidth === 0 && this.viewportBuf) {
      const cw = this.displayCanvas?.width || width;
      const ch = this.displayCanvas?.height || height;
      this.updateViewport(0, 0, 1, cw, ch, width, height, 0);
    }

    const pixelCount = width * height;
    const floatCount = pixelCount * 4;
    const byteCount = floatCount * 4;

    if (input.length !== floatCount) {
      return { tag: 'execution-error',
        val: `Input ${input.length} floats != expected ${floatCount} (${width}x${height}x4)` };
    }

    try {
      // Ping-pong f32 buffers
      const bufA = device.createBuffer({
        size: byteCount,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      const bufB = device.createBuffer({
        size: byteCount,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

      device.queue.writeBuffer(bufA, 0, input);
      let readBuf = bufA;
      let writeBuf = bufB;

      const encoder = device.createCommandEncoder();

      // Compute passes
      for (let i = 0; i < ops.length; i++) {
        const op = ops[i];
        const hash = hashSource(op.source) + ':' + op.entryPoint;
        let pipeline = this.shaderCache.get(hash);

        if (!pipeline) {
          const module = device.createShaderModule({ code: op.source });
          const entries: GPUBindGroupLayoutEntry[] = [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          ];
          for (let j = 0; j < op.extraBuffers.length; j++) {
            entries.push({ binding: 3 + j, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } });
          }
          const layout = device.createBindGroupLayout({ entries });
          pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module, entryPoint: op.entryPoint },
          });
          this.shaderCache.set(hash, pipeline);
        }

        const paramSize = Math.max(16, Math.ceil(op.params.byteLength / 16) * 16);
        const paramBuf = device.createBuffer({
          size: paramSize,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(paramBuf, 0, op.params);

        const extras: GPUBuffer[] = op.extraBuffers.map(data => {
          const buf = device.createBuffer({
            size: Math.max(4, data.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          device.queue.writeBuffer(buf, 0, data);
          return buf;
        });

        const bgEntries: GPUBindGroupEntry[] = [
          { binding: 0, resource: { buffer: readBuf } },
          { binding: 1, resource: { buffer: writeBuf } },
          { binding: 2, resource: { buffer: paramBuf } },
        ];
        extras.forEach((buf, j) => bgEntries.push({ binding: 3 + j, resource: { buffer: buf } }));

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: bgEntries,
        });

        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(
          Math.ceil(width / op.workgroupX),
          Math.ceil(height / op.workgroupY),
          1,
        );
        pass.end();

        // paramBuf and extras GC'd when unreferenced
        [readBuf, writeBuf] = [writeBuf, readBuf];
      }

      // Blit render pass — reads compute output, writes to canvas
      this.appendBlitPass(encoder, readBuf);

      // Flush previous frame's deferred destroys before submitting new work
      device.queue.submit([encoder.finish()]);

      // Defer destruction of previous frame's output buffer
      // Previous output buffer GC'd when lastOutputBuf reference changes
      this.lastOutputBuf = readBuf;
      this.lastImageWidth = width;
      this.lastImageHeight = height;

      // Defer destruction of the other ping-pong buffer
      // writeBuf GC'd when unreferenced

      return null; // success
    } catch (e) {
      return { tag: 'execution-error', val: (e as Error).message };
    }
  }

  /**
   * Re-blit the last compute output with current viewport.
   * For viewport-only updates (pan/zoom) — no recompute.
   */
  displayOnly(): void {
    if (!this.device || !this.canvasCtx || !this.blitPipeline || !this.lastOutputBuf) return;

    const encoder = this.device.createCommandEncoder();
    this.appendBlitPass(encoder, this.lastOutputBuf);
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Upload CPU f32 pixels to GPU and display via blit.
   * For CPU-only filter chains that still need the WebGPU display path.
   */
  displayFromCpu(pixels: Float32Array, width: number, height: number): void {
    if (!this.device || !this.canvasCtx || !this.blitPipeline || !this.blitBindGroupLayout || !this.viewportBuf) return;

    if (this.lastImageWidth === 0) {
      const cw = this.displayCanvas?.width || width;
      const ch = this.displayCanvas?.height || height;
      this.updateViewport(0, 0, 1, cw, ch, width, height, 0);
    }

    const byteCount = pixels.byteLength;
    const buf = this.device.createBuffer({
      size: byteCount,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, pixels);

    const encoder = this.device.createCommandEncoder();
    this.appendBlitPass(encoder, buf);

    this.device.queue.submit([encoder.finish()]);

    // Defer destruction of previous frame's buffer
    // Previous output buffer GC'd when lastOutputBuf reference changes
    this.lastOutputBuf = buf;
    this.lastImageWidth = width;
    this.lastImageHeight = height;
  }

  /** Append a blit render pass to an existing command encoder. */
  private appendBlitPass(encoder: GPUCommandEncoder, pixelBuf: GPUBuffer): void {
    if (!this.canvasCtx || !this.blitPipeline || !this.blitBindGroupLayout || !this.viewportBuf) return;

    const canvasTexture = this.canvasCtx.getCurrentTexture();
    const bindGroup = this.device!.createBindGroup({
      layout: this.blitBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: pixelBuf } },
        { binding: 1, resource: { buffer: this.viewportBuf } },
      ],
    });

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: canvasTexture.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
      }],
    });
    pass.setPipeline(this.blitPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3); // fullscreen triangle
    pass.end();
  }

  destroy(): void {
    this.shaderCache.clear();
    this.flushDeferred();
    if (this.lastOutputBuf) { this.lastOutputBuf.destroy(); this.lastOutputBuf = null; }
    if (this.viewportBuf) { this.viewportBuf.destroy(); this.viewportBuf = null; }
    this.displayCanvas = null;
    this.canvasCtx = null;
    this.blitPipeline = null;
    this.blitBindGroupLayout = null;
    if (this.device) { this.device.destroy(); this.device = null; }
    this.initPromise = null;
  }
}
