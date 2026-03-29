/**
 * @rasmcore/sdk — Modular media processing powered by WebAssembly.
 *
 * ```typescript
 * import { rcimage } from '@rasmcore/sdk';
 *
 * const jpeg = rcimage.load(pngBytes)
 *   .blur(3.0)
 *   .resize(800, 600, 'lanczos3')
 *   .brightness(0.1)
 *   .toJpeg({ quality: 85 });
 * ```
 *
 * Future: { rcimage, rcaudio, rcvideo }
 */

export { RcImage, rcimage } from './rcimage.js';

// Re-export types for consumers
export type {
  ImageInfo,
  ResizeFilter,
  Rotation,
  FlipDirection,
  PixelFormat,
  JpegWriteConfig,
  PngWriteConfig,
  WebpWriteConfig,
  AvifWriteConfig,
  TiffWriteConfig,
  GifWriteConfig,
  BmpWriteConfig,
  IcoWriteConfig,
  QoiWriteConfig,
  MetadataSet,
  ExifOrientation,
} from '../../demo/sdk/interfaces/rasmcore-image-pipeline.js';
