import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { GpuHandler, createGpuExecuteImport } from './gpu-handler';
import type { GpuOp } from './gpu-handler';

// ─── WebGPU constants (not available in jsdom) ─────────────────────────────

const GPU_BUFFER_USAGE = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
};

beforeEach(() => {
  // Provide WebGPU constants used by gpu-handler.ts
  (globalThis as Record<string, unknown>).GPUBufferUsage = GPU_BUFFER_USAGE;
  (globalThis as Record<string, unknown>).GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
  (globalThis as Record<string, unknown>).GPUMapMode = { READ: 1, WRITE: 2 };
});

afterEach(() => {
  // Delete navigator.gpu so 'gpu' in navigator returns false
  delete (navigator as Record<string, unknown>).gpu;
});

// ─── Mock helpers ──────────────────────────────────────────────────────────

function setNavigatorGpu(device: Record<string, unknown>) {
  const adapter = {
    limits: {
      maxStorageBufferBindingSize: 256 * 1024 * 1024,
      maxBufferSize: 256 * 1024 * 1024,
    },
    requestDevice: vi.fn().mockResolvedValue(device),
  };
  Object.defineProperty(navigator, 'gpu', {
    value: { requestAdapter: vi.fn().mockResolvedValue(adapter) },
    configurable: true,
  });
}

function mockGpuDevice() {
  const outputData = new ArrayBuffer(16);
  const stagingBuffer = {
    mapAsync: vi.fn().mockResolvedValue(undefined),
    getMappedRange: vi.fn().mockReturnValue(outputData),
    unmap: vi.fn(),
    destroy: vi.fn(),
  };
  const storageBuffer = { destroy: vi.fn() };
  const paramBuffer = { destroy: vi.fn() };

  const pass = {
    setPipeline: vi.fn(),
    setBindGroup: vi.fn(),
    dispatchWorkgroups: vi.fn(),
    end: vi.fn(),
  };

  const encoder = {
    beginComputePass: vi.fn().mockReturnValue(pass),
    copyBufferToBuffer: vi.fn(),
    finish: vi.fn().mockReturnValue('commands'),
  };

  const bindGroupLayout = {};
  const pipeline = {
    getBindGroupLayout: vi.fn().mockReturnValue(bindGroupLayout),
  };

  const device: Record<string, unknown> = {
    createShaderModule: vi.fn().mockReturnValue({}),
    createComputePipeline: vi.fn().mockReturnValue(pipeline),
    createBindGroupLayout: vi.fn().mockReturnValue(bindGroupLayout),
    createPipelineLayout: vi.fn().mockReturnValue({}),
    createBindGroup: vi.fn().mockReturnValue('bindGroup'),
    createCommandEncoder: vi.fn().mockReturnValue(encoder),
    createBuffer: vi.fn().mockImplementation((desc: { usage: number }) => {
      if (desc.usage & GPU_BUFFER_USAGE.MAP_READ) return stagingBuffer;
      if (desc.usage & GPU_BUFFER_USAGE.UNIFORM) return paramBuffer;
      return storageBuffer;
    }),
    queue: { writeBuffer: vi.fn(), submit: vi.fn() },
    lost: new Promise<never>(() => {}),
    destroy: vi.fn(),
  };

  return { device, encoder, pass, stagingBuffer, storageBuffer, pipeline };
}

function makeOp(overrides: Partial<GpuOp> = {}): GpuOp {
  return {
    shader: '@compute @workgroup_size(1) fn main() {}',
    entryPoint: 'main',
    workgroupX: 1,
    workgroupY: 1,
    workgroupZ: 1,
    params: new Uint8Array(16),
    extraBuffers: [],
    ...overrides,
  };
}

// ─── Tests ─────────────────────────────────────────────────────────────────

describe('GpuHandler', () => {
  describe('isAvailable', () => {
    it('returns false when navigator.gpu is absent', () => {
      expect(GpuHandler.isAvailable()).toBe(false);
    });

    it('returns true when navigator.gpu is present', () => {
      Object.defineProperty(navigator, 'gpu', { value: {}, configurable: true });
      expect(GpuHandler.isAvailable()).toBe(true);
    });
  });

  describe('execute', () => {
    it('returns input unchanged for empty ops list', async () => {
      const handler = new GpuHandler();
      const input = new Uint8Array([1, 2, 3, 4]);
      const result = await handler.execute([], input, 1, 1);
      expect(result).toHaveProperty('ok');
      if ('ok' in result) expect(result.ok).toEqual(input);
    });

    it('returns not-available error when WebGPU is absent', async () => {
      const handler = new GpuHandler();
      const input = new Uint8Array(16);
      const result = await handler.execute([makeOp()], input, 2, 2);
      expect(result).toHaveProperty('err');
      if ('err' in result) expect(result.err.tag).toBe('not-available');
    });

    it('validates input size matches width * height * 4', async () => {
      const { device } = mockGpuDevice();
      setNavigatorGpu(device);

      const handler = new GpuHandler();
      const result = await handler.execute([makeOp()], new Uint8Array(10), 2, 2);
      expect(result).toHaveProperty('err');
      if ('err' in result) {
        expect(result.err.tag).toBe('execution-error');
        expect(result.err.val).toContain('10');
      }
      handler.destroy();
    });

    it('executes single op with correct dispatch dimensions', async () => {
      const { device, pass } = mockGpuDevice();
      setNavigatorGpu(device);

      const handler = new GpuHandler();
      const input = new Uint8Array(16); // 2x2 RGBA8
      const ops = [makeOp({ workgroupX: 16, workgroupY: 16, workgroupZ: 1 })];

      const result = await handler.execute(ops, input, 2, 2);
      expect(result).toHaveProperty('ok');
      expect(pass.dispatchWorkgroups).toHaveBeenCalledWith(1, 1, 1);
      expect((device.createShaderModule as ReturnType<typeof vi.fn>)).toHaveBeenCalledTimes(1);
      handler.destroy();
    });

    it('caches shader module across multiple executions', async () => {
      const { device } = mockGpuDevice();
      setNavigatorGpu(device);

      const handler = new GpuHandler();
      const input = new Uint8Array(16);
      const ops = [makeOp()];

      await handler.execute(ops, input, 2, 2);
      await handler.execute(ops, input, 2, 2);

      expect((device.createShaderModule as ReturnType<typeof vi.fn>)).toHaveBeenCalledTimes(1);
      expect(handler.cacheSize).toBe(1);
      handler.destroy();
    });

    it('ping-pongs buffers for multi-op batches', async () => {
      const { device, pass } = mockGpuDevice();
      setNavigatorGpu(device);

      const handler = new GpuHandler();
      const input = new Uint8Array(16);
      const ops = [
        makeOp({ shader: 'fn pass1() {}', entryPoint: 'pass1' }),
        makeOp({ shader: 'fn pass2() {}', entryPoint: 'pass2' }),
      ];

      const result = await handler.execute(ops, input, 2, 2);
      expect(result).toHaveProperty('ok');
      expect(pass.dispatchWorkgroups).toHaveBeenCalledTimes(2);
      // 2 compute submits + 1 copy submit = 3
      expect((device.queue as { submit: ReturnType<typeof vi.fn> }).submit).toHaveBeenCalledTimes(3);
      handler.destroy();
    });

    it('creates extra storage buffers for ops with extraBuffers', async () => {
      const { device } = mockGpuDevice();
      setNavigatorGpu(device);

      const handler = new GpuHandler();
      const input = new Uint8Array(16);
      const kernel = new Uint8Array(new Float32Array([0.1, 0.2, 0.4, 0.2, 0.1]).buffer);
      const ops = [makeOp({ extraBuffers: [kernel] })];

      await handler.execute(ops, input, 2, 2);

      const layoutCall = (device.createBindGroupLayout as ReturnType<typeof vi.fn>).mock.calls[0][0];
      expect(layoutCall.entries).toHaveLength(4); // input, output, params, kernel
      handler.destroy();
    });
  });

  describe('destroy', () => {
    it('clears cache and device', () => {
      const handler = new GpuHandler();
      expect(handler.isInitialized).toBe(false);
      handler.destroy();
      expect(handler.cacheSize).toBe(0);
    });
  });
});

describe('createGpuExecuteImport', () => {
  it('returns null when WebGPU is not available', () => {
    // navigator.gpu is undefined (afterEach cleanup)
    const fn = createGpuExecuteImport();
    expect(fn).toBeNull();
  });

  it('returns a function when WebGPU is available', () => {
    Object.defineProperty(navigator, 'gpu', { value: {}, configurable: true });
    const fn = createGpuExecuteImport();
    expect(fn).toBeInstanceOf(Function);
  });
});
