"""Basic pipeline tests — requires WASM component to be built."""

import os
import pytest

# Skip all tests if WASM not available
WASM_PATH = os.environ.get("RASMCORE_WASM_PATH")
HAS_WASM = WASM_PATH and os.path.exists(WASM_PATH)

pytestmark = pytest.mark.skipif(not HAS_WASM, reason="RASMCORE_WASM_PATH not set or file not found")


def test_pipeline_import():
    """Pipeline class can be imported."""
    from rasmcore import Pipeline
    assert Pipeline is not None


def test_pipeline_create():
    """Pipeline instance can be created."""
    from rasmcore import Pipeline
    pipe = Pipeline(use_gpu=False)
    assert pipe is not None


def test_gpu_host_import():
    """GPU host module can be imported (even without GPU hardware)."""
    try:
        from rasmcore.gpu_host import WgpuExecutor
        # May fail if no GPU — that's OK
    except ImportError:
        pytest.skip("wgpu not installed")
    except RuntimeError:
        pass  # No GPU adapter — expected in CI


def test_gpu_host_unavailable_fallback():
    """Pipeline works without GPU."""
    from rasmcore import Pipeline
    pipe = Pipeline(use_gpu=False)
    assert pipe._gpu_executor is None
