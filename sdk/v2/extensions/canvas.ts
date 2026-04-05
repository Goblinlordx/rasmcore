// ─── Canvas Render Extension ────────────────────────────────────────────────
// Adds writeRenderTarget() to the Pipeline class for direct GPU-to-canvas render.
// This file is merged into the generated fluent SDK by generate-v2-fluent-sdk.mjs.

import { RenderTarget, createRenderTarget, type RenderTargetOptions } from '../lib/render-target';
export { RenderTarget, createRenderTarget, type RenderTargetOptions };

// Extend the Pipeline class with writeRenderTarget()
// (Pipeline is defined above in the generated code)

declare module './index' {
  interface Pipeline {
    writeRenderTarget(target: RenderTarget): Promise<void>;
  }
}

/**
 * Render the pipeline result directly to a RenderTarget canvas.
 *
 * If the pipeline has a GPU-acceleratable chain, shaders execute on the GPU
 * and blit directly to the canvas — zero CPU readback. Otherwise, falls back
 * to CPU render + 2D canvas putImageData.
 *
 * @param target - A RenderTarget created via createRenderTarget(canvas)
 */
Pipeline.prototype.writeRenderTarget = async function (target: RenderTarget): Promise<void> {
  const raw = this._pipe;
  const node = this._node;

  // Try GPU path: get shader chain + input pixels
  if (typeof raw.renderGpuPlan === 'function') {
    try {
      const plan = raw.renderGpuPlan(node);
      if (plan) {
        const ops = plan.shaders.map((s: any) => ({
          source: s.source,
          entryPoint: s.entryPoint,
          workgroupX: s.workgroupX,
          workgroupY: s.workgroupY,
          workgroupZ: s.workgroupZ,
          params: new Uint8Array(s.params),
          extraBuffers: s.extraBuffers.map((b: any) => new Uint8Array(b)),
        }));
        const success = await target.writeGpu(
          ops,
          new Float32Array(plan.inputPixels),
          plan.width,
          plan.height,
        );
        if (success) return;
      }
    } catch {
      // GPU plan failed — fall through to CPU
    }
  }

  // CPU fallback: render f32 pixels, write to canvas
  if (typeof raw.render === 'function') {
    const pixels = raw.render(node);
    const info = raw.nodeInfo(node);
    target.writeCpu(new Float32Array(pixels), info.width, info.height);
  }
};
