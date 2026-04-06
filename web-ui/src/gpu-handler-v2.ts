/**
 * V2 WebGPU handler — f32-only, no format dispatch.
 *
 * All pixel data is Float32Array (vec4<f32> per pixel).
 * No BufferFormat enum. No U32Packed. No pack/unpack.
 * Always array<vec4<f32>> storage buffers.
 *
 * Supports N named display targets via addDisplay(name, canvas, hdr).
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

/** A single stage in a multi-target GPU execution plan. */
export interface MultiStage {
  /** Display target name to blit result to. */
  name: string;
  /** Compute shaders to run (empty = passthrough — just upload + blit). */
  shaders: GpuShader[];
  /** Input data: fresh pixels or reuse a prior stage's output buffer. */
  input: { tag: 'pixels'; data: Float32Array } | { tag: 'prior'; name: string };
  width: number;
  height: number;
}

/** Registered display target — an OffscreenCanvas with WebGPU context. */
interface DisplayRegistration {
  name: string;
  canvas: OffscreenCanvas;
  ctx: GPUCanvasContext;
  viewportBuf: GPUBuffer;
  lastPixelBuf: GPUBuffer | null;
  imageWidth: number;
  imageHeight: number;
}

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
 * V2 GPU handler — f32-only, multi-display.
 *
 * All buffers are array<vec4<f32>>. No BufferFormat. No format dispatch.
 * Bytes per pixel = 16 (4 channels * 4 bytes). Always.
 */
export class GpuHandlerV2 {
  private device: GPUDevice | null = null;
  private shaderCache: Map<string, GPUComputePipeline> = new Map();
  private initPromise: Promise<void> | null = null;
  private available = true;

  // Shared blit pipeline (same shader for all displays)
  private blitPipeline: GPURenderPipeline | null = null;
  private blitBindGroupLayout: GPUBindGroupLayout | null = null;

  // Named display targets
  private displays: Map<string, DisplayRegistration> = new Map();

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

  private ensureBlitPipeline(device: GPUDevice): void {
    if (this.blitPipeline) return;

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
          format: 'rgba16float' as GPUTextureFormat,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  // ─── Display Registry ──────────────────────────────────────────────────

  /** Register a named display target from an OffscreenCanvas. */
  async addDisplay(name: string, canvas: OffscreenCanvas, hdr: boolean): Promise<void> {
    await this.ensureDevice();
    const device = this.device!;
    this.ensureBlitPipeline(device);

    const ctx = canvas.getContext('webgpu') as GPUCanvasContext;
    ctx.configure({
      device,
      format: 'rgba16float',
      alphaMode: 'premultiplied',
      toneMapping: { mode: hdr ? 'extended' : 'standard' },
    });

    const viewportBuf = device.createBuffer({
      size: VIEWPORT_BYTE_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.displays.set(name, {
      name,
      canvas,
      ctx,
      viewportBuf,
      lastPixelBuf: null,
      imageWidth: 0,
      imageHeight: 0,
    });
  }

  /** Remove a named display target and clean up its resources. */
  removeDisplay(name: string): void {
    const d = this.displays.get(name);
    if (!d) return;
    d.viewportBuf.destroy();
    this.displays.delete(name);
  }

  /** Check if a named display target exists. */
  hasDisplayTarget(name: string): boolean {
    return this.displays.has(name);
  }

  /** Check if any display targets are registered. */
  get hasDisplay(): boolean {
    return this.displays.size > 0;
  }

  /** Width of the last rendered image for a display. */
  imageWidthFor(name: string): number {
    return this.displays.get(name)?.imageWidth ?? 0;
  }

  /** Height of the last rendered image for a display. */
  imageHeightFor(name: string): number {
    return this.displays.get(name)?.imageHeight ?? 0;
  }

  // Legacy getters for backward compatibility
  get imageWidth(): number { return this.imageWidthFor('viewport'); }
  get imageHeight(): number { return this.imageHeightFor('viewport'); }

  /** Resize a named display canvas (skip if unchanged — resize clears the canvas). */
  resizeDisplay(name: string, width: number, height: number): void {
    const d = this.displays.get(name);
    if (!d) return;
    if (d.canvas.width !== width) d.canvas.width = width;
    if (d.canvas.height !== height) d.canvas.height = height;
  }

  /** Update viewport uniform for a named display. */
  updateViewportFor(
    name: string,
    panX: number, panY: number, zoom: number,
    canvasWidth: number, canvasHeight: number,
    imageWidth: number, imageHeight: number,
    toneMode: number,
  ): void {
    if (!this.device) return;
    const d = this.displays.get(name);
    if (!d) return;
    const data = new ArrayBuffer(VIEWPORT_BYTE_SIZE);
    const f = new Float32Array(data, 0, 7);
    const u = new Uint32Array(data, 28, 1);
    f[0] = canvasWidth; f[1] = canvasHeight;
    f[2] = imageWidth; f[3] = imageHeight;
    f[4] = panX; f[5] = panY; f[6] = zoom;
    u[0] = toneMode;
    this.device.queue.writeBuffer(d.viewportBuf, 0, data);
    d.imageWidth = imageWidth;
    d.imageHeight = imageHeight;
  }

  // Legacy updateViewport — delegates to 'viewport' display
  updateViewport(
    panX: number, panY: number, zoom: number,
    canvasWidth: number, canvasHeight: number,
    imageWidth: number, imageHeight: number,
    toneMode: number,
  ): void {
    this.updateViewportFor('viewport', panX, panY, zoom, canvasWidth, canvasHeight, imageWidth, imageHeight, toneMode);
  }

  /** Blit a pixel buffer to a named display target. */
  blitTo(name: string, pixelBuf: GPUBuffer): void {
    if (!this.device || !this.blitPipeline || !this.blitBindGroupLayout) return;
    const d = this.displays.get(name);
    if (!d) return;

    const canvasTexture = d.ctx.getCurrentTexture();
    const bindGroup = this.device.createBindGroup({
      layout: this.blitBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: pixelBuf } },
        { binding: 1, resource: { buffer: d.viewportBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
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
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    d.lastPixelBuf = pixelBuf;
  }

  /** Append a blit render pass to an existing command encoder for a named display. */
  private appendBlitPassTo(encoder: GPUCommandEncoder, name: string, pixelBuf: GPUBuffer): void {
    if (!this.blitPipeline || !this.blitBindGroupLayout) return;
    const d = this.displays.get(name);
    if (!d) return;

    const canvasTexture = d.ctx.getCurrentTexture();
    const bindGroup = this.device!.createBindGroup({
      layout: this.blitBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: pixelBuf } },
        { binding: 1, resource: { buffer: d.viewportBuf } },
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
    pass.draw(3);
    pass.end();
    d.lastPixelBuf = pixelBuf;
  }

  /** Re-blit without recompute for a named display. */
  displayOnly(name: string): void {
    const d = this.displays.get(name);
    if (!d?.lastPixelBuf) return;
    this.blitTo(name, d.lastPixelBuf);
  }

  // ─── Compute ───────────────────────────────────────────────────────────

  /**
   * Execute GPU shaders on f32 pixel data with CPU readback.
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
    const byteCount = floatCount * 4;

    if (input.length !== floatCount) {
      return { err: { tag: 'execution-error',
        val: `Input ${input.length} floats != expected ${floatCount} (${width}x${height}x4)` } };
    }

    try {
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
        const pipeline = this.getOrCreatePipeline(device, op);

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

        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: bgEntries,
        }));
        pass.dispatchWorkgroups(
          Math.ceil(width / op.workgroupX),
          Math.ceil(height / op.workgroupY),
          1,
        );
        pass.end();
        device.queue.submit([encoder.finish()]);
        [readBuf, writeBuf] = [writeBuf, readBuf];
      }

      // Readback
      const staging = device.createBuffer({
        size: byteCount,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const copyEnc = device.createCommandEncoder();
      copyEnc.copyBufferToBuffer(readBuf, 0, staging, 0, byteCount);
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
   */
  async prepare(shaders: GpuShader[]): Promise<void> {
    if (shaders.length === 0) return;
    try { await this.ensureDevice(); }
    catch { return; }
    const device = this.device!;
    for (const op of shaders) {
      this.getOrCreatePipeline(device, op);
    }
  }

  get cacheSize(): number {
    return this.shaderCache.size;
  }

  /**
   * Execute GPU shaders and blit the result to the 'viewport' display.
   * No CPU readback — the result stays on the GPU.
   */
  async executeAndDisplay(
    ops: GpuShader[],
    input: Float32Array,
    width: number,
    height: number,
  ): Promise<GpuError | null> {
    return this.executeAndBlitTo('viewport', ops, input, width, height);
  }

  /**
   * Execute GPU shaders and blit the result to a named display target.
   */
  async executeAndBlitTo(
    displayName: string,
    ops: GpuShader[],
    input: Float32Array,
    width: number,
    height: number,
  ): Promise<GpuError | null> {
    const d = this.displays.get(displayName);
    if (!d || !this.blitPipeline || !this.blitBindGroupLayout) {
      return { tag: 'not-available', val: `Display '${displayName}' not configured` };
    }

    try { await this.ensureDevice(); }
    catch (e) { return { tag: 'not-available', val: (e as Error).message }; }

    const device = this.device!;

    // Default viewport on first render
    if (d.imageWidth === 0) {
      const cw = d.canvas.width || width;
      const ch = d.canvas.height || height;
      this.updateViewportFor(displayName, 0, 0, 1, cw, ch, width, height, 0);
    }

    const pixelCount = width * height;
    const floatCount = pixelCount * 4;
    const byteCount = floatCount * 4;

    if (input.length !== floatCount) {
      return { tag: 'execution-error',
        val: `Input ${input.length} floats != expected ${floatCount} (${width}x${height}x4)` };
    }

    try {
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

      for (let i = 0; i < ops.length; i++) {
        const op = ops[i];
        const pipeline = this.getOrCreatePipeline(device, op);

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

        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: bgEntries,
        }));
        pass.dispatchWorkgroups(
          Math.ceil(width / op.workgroupX),
          Math.ceil(height / op.workgroupY),
          1,
        );
        pass.end();
        [readBuf, writeBuf] = [writeBuf, readBuf];
      }

      // Blit to named display
      this.appendBlitPassTo(encoder, displayName, readBuf);
      device.queue.submit([encoder.finish()]);

      d.lastPixelBuf = readBuf;
      d.imageWidth = width;
      d.imageHeight = height;

      return null;
    } catch (e) {
      return { tag: 'execution-error', val: (e as Error).message };
    }
  }

  /**
   * Execute ordered stages, each targeting a named display.
   * Stages can reuse prior stage output buffers via { tag: 'prior', name }.
   */
  async executeMulti(stages: MultiStage[]): Promise<GpuError | null> {
    if (stages.length === 0) return null;

    try { await this.ensureDevice(); }
    catch (e) { return { tag: 'not-available', val: (e as Error).message }; }

    const device = this.device!;
    const encoder = device.createCommandEncoder();
    const stageOutputs: Map<string, GPUBuffer> = new Map();

    for (const stage of stages) {
      const d = this.displays.get(stage.name);
      if (!d) continue;

      const byteCount = stage.width * stage.height * 4 * 4;

      // Resolve input
      let inputBuf: GPUBuffer;
      if (stage.input.tag === 'prior') {
        const prior = stageOutputs.get(stage.input.name);
        if (!prior) {
          return { tag: 'execution-error', val: `Prior stage '${stage.input.name}' not found` };
        }
        inputBuf = prior;
      } else {
        inputBuf = device.createBuffer({
          size: byteCount,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        device.queue.writeBuffer(inputBuf, 0, stage.input.data);
      }

      let readBuf = inputBuf;

      if (stage.shaders.length > 0) {
        // Compute passes
        const writeBufInit = device.createBuffer({
          size: byteCount,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        let writeBuf = writeBufInit;

        for (const op of stage.shaders) {
          const pipeline = this.getOrCreatePipeline(device, op);

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

          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: bgEntries,
          }));
          pass.dispatchWorkgroups(
            Math.ceil(stage.width / op.workgroupX),
            Math.ceil(stage.height / op.workgroupY),
            1,
          );
          pass.end();
          [readBuf, writeBuf] = [writeBuf, readBuf];
        }
      }

      // Blit to display
      this.appendBlitPassTo(encoder, stage.name, readBuf);
      d.lastPixelBuf = readBuf;
      d.imageWidth = stage.width;
      d.imageHeight = stage.height;
      stageOutputs.set(stage.name, readBuf);
    }

    device.queue.submit([encoder.finish()]);
    return null;
  }

  /**
   * Upload CPU f32 pixels and blit to a named display.
   */
  displayFromCpu(name: string, pixels: Float32Array, width: number, height: number): void {
    if (!this.device || !this.blitPipeline || !this.blitBindGroupLayout) return;
    const d = this.displays.get(name);
    if (!d) return;

    if (d.imageWidth === 0) {
      const cw = d.canvas.width || width;
      const ch = d.canvas.height || height;
      this.updateViewportFor(name, 0, 0, 1, cw, ch, width, height, 0);
    }

    const buf = this.device.createBuffer({
      size: pixels.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, pixels);
    this.blitTo(name, buf);
    d.imageWidth = width;
    d.imageHeight = height;
  }

  // ─── Legacy Compatibility ──────────────────────────────────────────────

  // These delegate to the named display registry for backward compatibility
  // with existing worker code that uses the old hardcoded API.

  async setDisplayCanvas(canvas: OffscreenCanvas, hdr: boolean): Promise<void> {
    return this.addDisplay('viewport', canvas, hdr);
  }

  async setOriginalCanvas(canvas: OffscreenCanvas, hdr: boolean): Promise<void> {
    return this.addDisplay('original', canvas, hdr);
  }

  get hasOriginalDisplay(): boolean {
    return this.hasDisplayTarget('original');
  }

  storeSourcePixels(pixels: Float32Array, width: number, height: number): void {
    this.displayFromCpu('original', pixels, width, height);
  }

  updateOriginalViewport(panX: number, panY: number, zoom: number, canvasWidth: number, canvasHeight: number, toneMode: number): void {
    const d = this.displays.get('original');
    if (!d) return;
    this.updateViewportFor('original', panX, panY, zoom, canvasWidth, canvasHeight, d.imageWidth, d.imageHeight, toneMode);
  }

  resizeOriginalDisplay(width: number, height: number): void {
    this.resizeDisplay('original', width, height);
  }

  blitOriginal(): void {
    this.displayOnly('original');
  }

  // ─── Helpers ───────────────────────────────────────────────────────────

  private getOrCreatePipeline(device: GPUDevice, op: GpuShader): GPUComputePipeline {
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
    return pipeline;
  }

  destroy(): void {
    this.shaderCache.clear();
    for (const d of this.displays.values()) {
      d.viewportBuf.destroy();
    }
    this.displays.clear();
    this.blitPipeline = null;
    this.blitBindGroupLayout = null;
    if (this.device) { this.device.destroy(); this.device = null; }
    this.initPromise = null;
  }
}
