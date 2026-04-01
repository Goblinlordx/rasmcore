"""GPU execution host using wgpu-py.

Provides the gpu-execute implementation for the WASM pipeline.
Compiles WGSL shaders, manages buffers, dispatches compute.
"""

from __future__ import annotations

import hashlib
from typing import Optional

try:
    import wgpu
    HAS_WGPU = True
except ImportError:
    HAS_WGPU = False


class GpuOp:
    """Native representation of a GPU operation."""
    __slots__ = ("shader", "entry_point", "workgroup_size", "params", "extra_buffers")

    def __init__(self, shader: str, entry_point: str, workgroup_size: tuple[int, int, int],
                 params: bytes, extra_buffers: list[bytes]):
        self.shader = shader
        self.entry_point = entry_point
        self.workgroup_size = workgroup_size
        self.params = params
        self.extra_buffers = extra_buffers


def translate_ops(wit_ops) -> list[GpuOp]:
    """Convert WIT gpu-op records to native GpuOp objects."""
    result = []
    for op in wit_ops:
        result.append(GpuOp(
            shader=op["shader"],
            entry_point=op["entry-point"],
            workgroup_size=(op["workgroup-x"], op["workgroup-y"], op["workgroup-z"]),
            params=bytes(op["params"]),
            extra_buffers=[bytes(b) for b in op.get("extra-buffers", [])],
        ))
    return result


class WgpuExecutor:
    """GPU compute executor using wgpu-py.

    Compiles WGSL shaders, manages storage buffers, dispatches compute.
    Caches compiled pipelines by shader source hash.
    """

    def __init__(self, max_buffer_size: int = 256 * 1024 * 1024):
        if not HAS_WGPU:
            raise RuntimeError("wgpu not installed: pip install wgpu")

        self._adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if not self._adapter:
            raise RuntimeError("No GPU adapter found")

        self._device = self._adapter.request_device_sync()
        self._pipeline_cache: dict[str, wgpu.GPUComputePipeline] = {}
        self._max_buffer_size = max_buffer_size

        info = self._adapter.info
        print(f"[rasmcore GPU] {info.get('description', 'unknown GPU')}")

    def _get_pipeline(self, shader_source: str, entry_point: str) -> wgpu.GPUComputePipeline:
        """Get or compile a compute pipeline (cached by source hash)."""
        key = hashlib.sha256(f"{shader_source}:{entry_point}".encode()).hexdigest()[:16]
        if key not in self._pipeline_cache:
            shader_module = self._device.create_shader_module(code=shader_source)
            self._pipeline_cache[key] = self._device.create_compute_pipeline(
                layout="auto",
                compute={"module": shader_module, "entry_point": entry_point},
            )
        return self._pipeline_cache[key]

    def execute(self, ops: list[GpuOp], input_data: bytes, width: int, height: int) -> bytes:
        """Execute a batch of GPU operations.

        Chains ops via ping-pong buffers. Returns output pixel data.
        """
        byte_len = width * height * 4
        if byte_len > self._max_buffer_size:
            raise RuntimeError(f"Image too large for GPU: {byte_len} bytes (max {self._max_buffer_size})")

        # Create ping-pong storage buffers
        buf_a = self._device.create_buffer(
            size=byte_len,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )
        buf_b = self._device.create_buffer(
            size=byte_len,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )

        # Upload input to buf_a
        self._device.queue.write_buffer(buf_a, 0, input_data)

        current_input = buf_a
        current_output = buf_b

        for op in ops:
            pipeline = self._get_pipeline(op.shader, op.entry_point)

            # Create uniform buffer for params
            param_buf = None
            if op.params:
                # Pad to 16-byte alignment
                padded = op.params + b'\x00' * ((-len(op.params)) % 16)
                param_buf = self._device.create_buffer_with_data(
                    data=padded,
                    usage=wgpu.BufferUsage.UNIFORM,
                )

            # Create extra storage buffers
            extra_bufs = []
            for extra_data in op.extra_buffers:
                eb = self._device.create_buffer_with_data(
                    data=extra_data,
                    usage=wgpu.BufferUsage.STORAGE,
                )
                extra_bufs.append(eb)

            # Build bind group entries
            entries = [
                {"binding": 0, "resource": {"buffer": current_input, "offset": 0, "size": byte_len}},
                {"binding": 1, "resource": {"buffer": current_output, "offset": 0, "size": byte_len}},
            ]
            if param_buf:
                entries.append({"binding": 2, "resource": {"buffer": param_buf, "offset": 0, "size": param_buf.size}})
            for i, eb in enumerate(extra_bufs):
                entries.append({"binding": 3 + i, "resource": {"buffer": eb, "offset": 0, "size": eb.size}})

            bind_group = self._device.create_bind_group(
                layout=pipeline.get_bind_group_layout(0),
                entries=entries,
            )

            # Dispatch
            wx, wy, wz = op.workgroup_size
            dispatch_x = (width + wx - 1) // wx
            dispatch_y = (height + wy - 1) // wy

            encoder = self._device.create_command_encoder()
            pass_ = encoder.begin_compute_pass()
            pass_.set_pipeline(pipeline)
            pass_.set_bind_group(0, bind_group)
            pass_.dispatch_workgroups(dispatch_x, dispatch_y, 1)
            pass_.end()
            self._device.queue.submit([encoder.finish()])

            # Swap ping-pong
            current_input, current_output = current_output, current_input

            # Clean up per-op buffers
            if param_buf:
                param_buf.destroy()
            for eb in extra_bufs:
                eb.destroy()

        # Read back result (current_input has the last output after swap)
        result = self._device.queue.read_buffer(current_input, 0, byte_len)

        # Cleanup
        buf_a.destroy()
        buf_b.destroy()

        return bytes(result)
