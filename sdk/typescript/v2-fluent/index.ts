// Auto-generated V2 fluent SDK — do not edit.
// Generated from 73 filters, 12 encoders, 14 decoders.
// Regenerate: node scripts/generate-v2-fluent-sdk.mjs

import type { ImagePipelineV2 as RawPipeline } from '../v2-generated/interfaces/rasmcore-v2-image-pipeline-v2.js';

// ─── Param serialization helpers ────────────────────────────────────────────

function pushF32(buf: number[], name: string, value: number) {
  buf.push(name.length);
  for (let i = 0; i < name.length; i++) buf.push(name.charCodeAt(i));
  buf.push(0); // f32 type
  const ab = new ArrayBuffer(4);
  new DataView(ab).setFloat32(0, value, true);
  buf.push(...new Uint8Array(ab));
}

function pushU32(buf: number[], name: string, value: number) {
  buf.push(name.length);
  for (let i = 0; i < name.length; i++) buf.push(name.charCodeAt(i));
  buf.push(1); // u32 type
  const ab = new ArrayBuffer(4);
  new DataView(ab).setUint32(0, value, true);
  buf.push(...new Uint8Array(ab));
}

function pushBool(buf: number[], name: string, value: boolean) {
  buf.push(name.length);
  for (let i = 0; i < name.length; i++) buf.push(name.charCodeAt(i));
  buf.push(2); // bool type
  buf.push(value ? 1 : 0);
}

// ─── Config interfaces ──────────────────────────────────────────────────────

export interface FrequencyLowConfig {
  sigma?: number;
}

export interface ClaheConfig {
  tileGrid?: number;
  clipLimit?: number;
}

export interface VignettePowerlawConfig {
  strength?: number;
  falloff?: number;
}

export interface ClarityConfig {
  amount?: number;
  radius?: number;
}

export interface ShadowHighlightConfig {
  shadows?: number;
  highlights?: number;
  whitepoint?: number;
  radius?: number;
  compress?: number;
  shadowsCcorrect?: number;
  highlightsCcorrect?: number;
}

export interface FrequencyHighConfig {
  sigma?: number;
}

export interface VignetteConfig {
  sigma?: number;
  xInset?: number;
  yInset?: number;
}

export interface RetinexMsrConfig {
  sigmaSmall?: number;
  sigmaMedium?: number;
  sigmaLarge?: number;
}

export interface DehazeConfig {
  patchRadius?: number;
  omega?: number;
  tMin?: number;
}

export interface NormalizeConfig {
  blackClip?: number;
  whiteClip?: number;
}

export interface PyramidDetailRemapConfig {
  sigma?: number;
  levels?: number;
}

export interface NlmDenoiseConfig {
  h?: number;
  patchRadius?: number;
  searchRadius?: number;
}

export interface RetinexSsrConfig {
  sigma?: number;
}

export interface RetinexMsrcrConfig {
  sigmaSmall?: number;
  sigmaMedium?: number;
  sigmaLarge?: number;
  alpha?: number;
  beta?: number;
}

export interface LabSharpenConfig {
  amount?: number;
  radius?: number;
}

export interface WhiteBalanceTemperatureConfig {
  temperature?: number;
  tint?: number;
}

export interface DitherFloydSteinbergConfig {
  maxColors?: number;
}

export interface ModulateConfig {
  brightness?: number;
  saturation?: number;
  hue?: number;
}

export interface ReplaceColorConfig {
  centerHue?: number;
  hueRange?: number;
  satMin?: number;
  satMax?: number;
  lumMin?: number;
  lumMax?: number;
  hueShift?: number;
  satShift?: number;
  lumShift?: number;
}

export interface QuantizeConfig {
  maxColors?: number;
}

export interface DitherOrderedConfig {
  maxColors?: number;
  mapSize?: number;
}

export interface ColorizeConfig {
  targetR?: number;
  targetG?: number;
  targetB?: number;
  amount?: number;
}

export interface PhotoFilterConfig {
  colorR?: number;
  colorG?: number;
  colorB?: number;
  density?: number;
  preserveLuminosity?: boolean;
}

export interface KmeansQuantizeConfig {
  k?: number;
  maxIterations?: number;
  seed?: number;
}

export interface HueRotateConfig {
  degrees?: number;
}

export interface LabAdjustConfig {
  aOffset?: number;
  bOffset?: number;
}

export interface VibranceConfig {
  amount?: number;
}

export interface SaturateConfig {
  factor?: number;
}

export interface SepiaConfig {
  intensity?: number;
}

export interface SelectiveColorConfig {
  targetHue?: number;
  hueRange?: number;
  hueShift?: number;
  saturation?: number;
  lightness?: number;
}

export interface TonemapDragoConfig {
  lMax?: number;
  bias?: number;
}

export interface FilmGrainGradingConfig {
  amount?: number;
  size?: number;
  color?: boolean;
  seed?: number;
}

export interface TonemapFilmicConfig {
  a?: number;
  b?: number;
  c?: number;
  d?: number;
  e?: number;
}

export interface GlitchConfig {
  shiftAmount?: number;
  channelOffset?: number;
  intensity?: number;
  bandHeight?: number;
  seed?: number;
}

export interface UniformNoiseConfig {
  range?: number;
  seed?: number;
}

export interface MirrorKaleidoscopeConfig {
  segments?: number;
  angle?: number;
  mode?: number;
}

export interface ChromaticSplitConfig {
  redDx?: number;
  redDy?: number;
  greenDx?: number;
  greenDy?: number;
  blueDx?: number;
  blueDy?: number;
}

export interface SaltPepperNoiseConfig {
  density?: number;
  seed?: number;
}

export interface PixelateConfig {
  blockSize?: number;
}

export interface OilPaintConfig {
  radius?: number;
}

export interface ChromaticAberrationConfig {
  strength?: number;
}

export interface FilmGrainConfig {
  amount?: number;
  size?: number;
  seed?: number;
}

export interface LightLeakConfig {
  intensity?: number;
  positionX?: number;
  positionY?: number;
  radius?: number;
  warmth?: number;
}

export interface PoissonNoiseConfig {
  scale?: number;
  seed?: number;
}

export interface CharcoalConfig {
  radius?: number;
  sigma?: number;
}

export interface GaussianNoiseConfig {
  amount?: number;
  mean?: number;
  sigma?: number;
  seed?: number;
}

export interface HalftoneConfig {
  dotSize?: number;
  angleOffset?: number;
}

export interface LevelsConfig {
  black?: number;
  white?: number;
  gamma?: number;
}

export interface SolarizeConfig {
  threshold?: number;
}

export interface BrightnessConfig {
  amount?: number;
}

export interface GammaConfig {
  gamma?: number;
}

export interface BurnConfig {
  amount?: number;
}

export interface SigmoidalContrastConfig {
  strength?: number;
  midpoint?: number;
  sharpen?: boolean;
}

export interface ContrastConfig {
  amount?: number;
}

export interface DodgeConfig {
  amount?: number;
}

export interface PosterizeConfig {
  levels?: number;
}

export interface ExposureConfig {
  ev?: number;
  offset?: number;
  gammaCorrection?: number;
}

export interface GaussianBlurConfig {
  radius?: number;
}

export interface MotionBlurConfig {
  angle?: number;
  length?: number;
}

export interface SharpenConfig {
  radius?: number;
  amount?: number;
}

export interface BilateralConfig {
  diameter?: number;
  sigmaColor?: number;
  sigmaSpace?: number;
}

export interface HighPassConfig {
  radius?: number;
}

export interface BoxBlurConfig {
  radius?: number;
}

export interface MedianConfig {
  radius?: number;
}

// ─── Pipeline class ──────────────────────────────────────────────────────────

export interface ReadConfig {
  hint?: string;
}

/**
 * V2 image processing pipeline with fluent API.
 *
 * Usage:
 *   const result = Pipeline.open(imageBytes)
 *     .brightness({ amount: 0.5 })
 *     .blur({ radius: 3 })
 *     .writePng();
 */
export class Pipeline {
  private _pipe: RawPipeline;
  private _node: number;

  private constructor(pipe: RawPipeline, node: number) {
    this._pipe = pipe;
    this._node = node;
  }

  /** Create a pipeline from raw image bytes (auto-loads WASM module). */
  static open(data: Uint8Array, config?: ReadConfig): Pipeline {
    const { pipelineV2 } = require('../v2-generated/rasmcore-v2-image.js');
    const pipe = new pipelineV2.ImagePipelineV2();
    const readConfig = config ? { formatHint: config.hint } : undefined;
    const node = pipe.read(data, readConfig);
    return new Pipeline(pipe, node);
  }

  /** Create a pipeline from a pre-loaded pipeline class (for web workers).
   *  Optionally accepts a LayerCache for cross-pipeline result reuse. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  static fromRaw(PipelineClass: any, data: Uint8Array, config?: ReadConfig, layerCache?: any): Pipeline {
    const pipe = new PipelineClass();
    if (layerCache && pipe.setLayerCache) {
      pipe.setLayerCache(layerCache);
    }
    const readConfig = config ? { formatHint: config.hint } : undefined;
    const node = pipe.read(data, readConfig);
    return new Pipeline(pipe, node);
  }

  /** Get image dimensions and color space. */
  get info() {
    return this._pipe.nodeInfo(this._node);
  }

  /** Get raw f32 RGBA pixel data. */
  render(): Float32Array {
    return this._pipe.render(this._node);
  }

  // ─── Filter methods (generated) ─────────────────────────────────────────

  /** Frequency Low */
  frequencyLow(config: FrequencyLowConfig): Pipeline {
    const serialize = (config: FrequencyLowConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'sigma', config.sigma ?? 3);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'frequency_low', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Clahe */
  clahe(config: ClaheConfig): Pipeline {
    const serialize = (config: ClaheConfig): Uint8Array => {
    const buf: number[] = [];
  pushU32(buf, 'tile_grid', config.tileGrid ?? 8);
  pushF32(buf, 'clip_limit', config.clipLimit ?? 2);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'clahe', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Vignette Powerlaw */
  vignettePowerlaw(config: VignettePowerlawConfig): Pipeline {
    const serialize = (config: VignettePowerlawConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'strength', config.strength ?? 0.5);
  pushF32(buf, 'falloff', config.falloff ?? 2);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'vignette_powerlaw', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Clarity */
  clarity(config: ClarityConfig): Pipeline {
    const serialize = (config: ClarityConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'amount', config.amount ?? 0);
  pushF32(buf, 'radius', config.radius ?? 20);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'clarity', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Shadow Highlight */
  shadowHighlight(config: ShadowHighlightConfig): Pipeline {
    const serialize = (config: ShadowHighlightConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'shadows', config.shadows ?? 0);
  pushF32(buf, 'highlights', config.highlights ?? 0);
  pushF32(buf, 'whitepoint', config.whitepoint ?? 0);
  pushF32(buf, 'radius', config.radius ?? 30);
  pushF32(buf, 'compress', config.compress ?? 50);
  pushF32(buf, 'shadows_ccorrect', config.shadowsCcorrect ?? 50);
  pushF32(buf, 'highlights_ccorrect', config.highlightsCcorrect ?? 50);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'shadow_highlight', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Frequency High */
  frequencyHigh(config: FrequencyHighConfig): Pipeline {
    const serialize = (config: FrequencyHighConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'sigma', config.sigma ?? 3);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'frequency_high', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Vignette */
  vignette(config: VignetteConfig): Pipeline {
    const serialize = (config: VignetteConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'sigma', config.sigma ?? 10);
  pushU32(buf, 'x_inset', config.xInset ?? 0);
  pushU32(buf, 'y_inset', config.yInset ?? 0);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'vignette', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Equalize */
  equalize(): Pipeline {
    const node = this._pipe.applyFilter(this._node, 'equalize', new Uint8Array(0));
    return new Pipeline(this._pipe, node);
  }

  /** Retinex Msr */
  retinexMsr(config: RetinexMsrConfig): Pipeline {
    const serialize = (config: RetinexMsrConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'sigma_small', config.sigmaSmall ?? 15);
  pushF32(buf, 'sigma_medium', config.sigmaMedium ?? 80);
  pushF32(buf, 'sigma_large', config.sigmaLarge ?? 250);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'retinex_msr', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Auto Level */
  autoLevel(): Pipeline {
    const node = this._pipe.applyFilter(this._node, 'auto_level', new Uint8Array(0));
    return new Pipeline(this._pipe, node);
  }

  /** Dehaze */
  dehaze(config: DehazeConfig): Pipeline {
    const serialize = (config: DehazeConfig): Uint8Array => {
    const buf: number[] = [];
  pushU32(buf, 'patch_radius', config.patchRadius ?? 7);
  pushF32(buf, 'omega', config.omega ?? 0.95);
  pushF32(buf, 't_min', config.tMin ?? 0.1);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'dehaze', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Normalize */
  normalize(config: NormalizeConfig): Pipeline {
    const serialize = (config: NormalizeConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'black_clip', config.blackClip ?? 0.02);
  pushF32(buf, 'white_clip', config.whiteClip ?? 0.01);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'normalize', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Pyramid Detail Remap */
  pyramidDetailRemap(config: PyramidDetailRemapConfig): Pipeline {
    const serialize = (config: PyramidDetailRemapConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'sigma', config.sigma ?? 0.5);
  pushU32(buf, 'levels', config.levels ?? 0);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'pyramid_detail_remap', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Nlm Denoise */
  nlmDenoise(config: NlmDenoiseConfig): Pipeline {
    const serialize = (config: NlmDenoiseConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'h', config.h ?? 0.1);
  pushU32(buf, 'patch_radius', config.patchRadius ?? 3);
  pushU32(buf, 'search_radius', config.searchRadius ?? 10);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'nlm_denoise', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Retinex Ssr */
  retinexSsr(config: RetinexSsrConfig): Pipeline {
    const serialize = (config: RetinexSsrConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'sigma', config.sigma ?? 80);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'retinex_ssr', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Retinex Msrcr */
  retinexMsrcr(config: RetinexMsrcrConfig): Pipeline {
    const serialize = (config: RetinexMsrcrConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'sigma_small', config.sigmaSmall ?? 15);
  pushF32(buf, 'sigma_medium', config.sigmaMedium ?? 80);
  pushF32(buf, 'sigma_large', config.sigmaLarge ?? 250);
  pushF32(buf, 'alpha', config.alpha ?? 125);
  pushF32(buf, 'beta', config.beta ?? 46);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'retinex_msrcr', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Lab Sharpen */
  labSharpen(config: LabSharpenConfig): Pipeline {
    const serialize = (config: LabSharpenConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'amount', config.amount ?? 1);
  pushF32(buf, 'radius', config.radius ?? 2);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'lab_sharpen', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** White Balance Temperature */
  whiteBalanceTemperature(config: WhiteBalanceTemperatureConfig): Pipeline {
    const serialize = (config: WhiteBalanceTemperatureConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'temperature', config.temperature ?? 6500);
  pushF32(buf, 'tint', config.tint ?? 0);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'white_balance_temperature', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Dither Floyd Steinberg */
  ditherFloydSteinberg(config: DitherFloydSteinbergConfig): Pipeline {
    const serialize = (config: DitherFloydSteinbergConfig): Uint8Array => {
    const buf: number[] = [];
  pushU32(buf, 'max_colors', config.maxColors ?? 16);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'dither_floyd_steinberg', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Modulate */
  modulate(config: ModulateConfig): Pipeline {
    const serialize = (config: ModulateConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'brightness', config.brightness ?? 1);
  pushF32(buf, 'saturation', config.saturation ?? 1);
  pushF32(buf, 'hue', config.hue ?? 0);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'modulate', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Replace Color */
  replaceColor(config: ReplaceColorConfig): Pipeline {
    const serialize = (config: ReplaceColorConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'center_hue', config.centerHue ?? 0);
  pushF32(buf, 'hue_range', config.hueRange ?? 30);
  pushF32(buf, 'sat_min', config.satMin ?? 0);
  pushF32(buf, 'sat_max', config.satMax ?? 1);
  pushF32(buf, 'lum_min', config.lumMin ?? 0);
  pushF32(buf, 'lum_max', config.lumMax ?? 1);
  pushF32(buf, 'hue_shift', config.hueShift ?? 0);
  pushF32(buf, 'sat_shift', config.satShift ?? 0);
  pushF32(buf, 'lum_shift', config.lumShift ?? 0);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'replace_color', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Quantize */
  quantize(config: QuantizeConfig): Pipeline {
    const serialize = (config: QuantizeConfig): Uint8Array => {
    const buf: number[] = [];
  pushU32(buf, 'max_colors', config.maxColors ?? 16);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'quantize', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** White Balance Gray World */
  whiteBalanceGrayWorld(): Pipeline {
    const node = this._pipe.applyFilter(this._node, 'white_balance_gray_world', new Uint8Array(0));
    return new Pipeline(this._pipe, node);
  }

  /** Dither Ordered */
  ditherOrdered(config: DitherOrderedConfig): Pipeline {
    const serialize = (config: DitherOrderedConfig): Uint8Array => {
    const buf: number[] = [];
  pushU32(buf, 'max_colors', config.maxColors ?? 16);
  pushU32(buf, 'map_size', config.mapSize ?? 4);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'dither_ordered', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Colorize */
  colorize(config: ColorizeConfig): Pipeline {
    const serialize = (config: ColorizeConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'target_r', config.targetR ?? 1);
  pushF32(buf, 'target_g', config.targetG ?? 0.8);
  pushF32(buf, 'target_b', config.targetB ?? 0.6);
  pushF32(buf, 'amount', config.amount ?? 0.5);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'colorize', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Photo Filter */
  photoFilter(config: PhotoFilterConfig): Pipeline {
    const serialize = (config: PhotoFilterConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'color_r', config.colorR ?? 1);
  pushF32(buf, 'color_g', config.colorG ?? 0.6);
  pushF32(buf, 'color_b', config.colorB ?? 0.2);
  pushF32(buf, 'density', config.density ?? 0.25);
  pushBool(buf, 'preserve_luminosity', config.preserveLuminosity ?? false);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'photo_filter', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Kmeans Quantize */
  kmeansQuantize(config: KmeansQuantizeConfig): Pipeline {
    const serialize = (config: KmeansQuantizeConfig): Uint8Array => {
    const buf: number[] = [];
  pushU32(buf, 'k', config.k ?? 16);
  pushU32(buf, 'max_iterations', config.maxIterations ?? 20);
  pushU32(buf, 'seed', config.seed ?? 42);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'kmeans_quantize', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Hue Rotate */
  hueRotate(config: HueRotateConfig): Pipeline {
    const serialize = (config: HueRotateConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'degrees', config.degrees ?? 0);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'hue_rotate', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Lab Adjust */
  labAdjust(config: LabAdjustConfig): Pipeline {
    const serialize = (config: LabAdjustConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'a_offset', config.aOffset ?? 0);
  pushF32(buf, 'b_offset', config.bOffset ?? 0);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'lab_adjust', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Vibrance */
  vibrance(config: VibranceConfig): Pipeline {
    const serialize = (config: VibranceConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'amount', config.amount ?? 0);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'vibrance', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Saturate */
  saturate(config: SaturateConfig): Pipeline {
    const serialize = (config: SaturateConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'factor', config.factor ?? 1);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'saturate', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Sepia */
  sepia(config: SepiaConfig): Pipeline {
    const serialize = (config: SepiaConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'intensity', config.intensity ?? 1);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'sepia', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Selective Color */
  selectiveColor(config: SelectiveColorConfig): Pipeline {
    const serialize = (config: SelectiveColorConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'target_hue', config.targetHue ?? 0);
  pushF32(buf, 'hue_range', config.hueRange ?? 30);
  pushF32(buf, 'hue_shift', config.hueShift ?? 0);
  pushF32(buf, 'saturation', config.saturation ?? 1);
  pushF32(buf, 'lightness', config.lightness ?? 0);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'selective_color', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /**  */
  splitToning(): Pipeline {
    const node = this._pipe.applyFilter(this._node, 'split_toning', new Uint8Array(0));
    return new Pipeline(this._pipe, node);
  }

  /** Tonemap Drago */
  tonemapDrago(config: TonemapDragoConfig): Pipeline {
    const serialize = (config: TonemapDragoConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'l_max', config.lMax ?? 1);
  pushF32(buf, 'bias', config.bias ?? 0.85);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'tonemap_drago', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /**  */
  liftGammaGain(): Pipeline {
    const node = this._pipe.applyFilter(this._node, 'lift_gamma_gain', new Uint8Array(0));
    return new Pipeline(this._pipe, node);
  }

  /**  */
  ascCdl(): Pipeline {
    const node = this._pipe.applyFilter(this._node, 'asc_cdl', new Uint8Array(0));
    return new Pipeline(this._pipe, node);
  }

  /** Film Grain Grading */
  filmGrainGrading(config: FilmGrainGradingConfig): Pipeline {
    const serialize = (config: FilmGrainGradingConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'amount', config.amount ?? 0.1);
  pushF32(buf, 'size', config.size ?? 1);
  pushBool(buf, 'color', config.color ?? false);
  pushU32(buf, 'seed', config.seed ?? 42);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'film_grain_grading', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Tonemap Filmic */
  tonemapFilmic(config: TonemapFilmicConfig): Pipeline {
    const serialize = (config: TonemapFilmicConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'a', config.a ?? 2.51);
  pushF32(buf, 'b', config.b ?? 0.03);
  pushF32(buf, 'c', config.c ?? 2.43);
  pushF32(buf, 'd', config.d ?? 0.59);
  pushF32(buf, 'e', config.e ?? 0.14);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'tonemap_filmic', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Tonemap Reinhard */
  tonemapReinhard(): Pipeline {
    const node = this._pipe.applyFilter(this._node, 'tonemap_reinhard', new Uint8Array(0));
    return new Pipeline(this._pipe, node);
  }

  /** Glitch */
  glitch(config: GlitchConfig): Pipeline {
    const serialize = (config: GlitchConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'shift_amount', config.shiftAmount ?? 20);
  pushF32(buf, 'channel_offset', config.channelOffset ?? 10);
  pushF32(buf, 'intensity', config.intensity ?? 0.5);
  pushU32(buf, 'band_height', config.bandHeight ?? 8);
  pushU32(buf, 'seed', config.seed ?? 42);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'glitch', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Uniform Noise */
  uniformNoise(config: UniformNoiseConfig): Pipeline {
    const serialize = (config: UniformNoiseConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'range', config.range ?? 25);
  pushU32(buf, 'seed', config.seed ?? 42);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'uniform_noise', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Mirror Kaleidoscope */
  mirrorKaleidoscope(config: MirrorKaleidoscopeConfig): Pipeline {
    const serialize = (config: MirrorKaleidoscopeConfig): Uint8Array => {
    const buf: number[] = [];
  pushU32(buf, 'segments', config.segments ?? 4);
  pushF32(buf, 'angle', config.angle ?? 0);
  pushU32(buf, 'mode', config.mode ?? 0);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'mirror_kaleidoscope', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Chromatic Split */
  chromaticSplit(config: ChromaticSplitConfig): Pipeline {
    const serialize = (config: ChromaticSplitConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'red_dx', config.redDx ?? 0);
  pushF32(buf, 'red_dy', config.redDy ?? 0);
  pushF32(buf, 'green_dx', config.greenDx ?? 0);
  pushF32(buf, 'green_dy', config.greenDy ?? 0);
  pushF32(buf, 'blue_dx', config.blueDx ?? 0);
  pushF32(buf, 'blue_dy', config.blueDy ?? 0);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'chromatic_split', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Salt Pepper Noise */
  saltPepperNoise(config: SaltPepperNoiseConfig): Pipeline {
    const serialize = (config: SaltPepperNoiseConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'density', config.density ?? 0.05);
  pushU32(buf, 'seed', config.seed ?? 42);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'salt_pepper_noise', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Pixelate */
  pixelate(config: PixelateConfig): Pipeline {
    const serialize = (config: PixelateConfig): Uint8Array => {
    const buf: number[] = [];
  pushU32(buf, 'block_size', config.blockSize ?? 8);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'pixelate', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Oil Paint */
  oilPaint(config: OilPaintConfig): Pipeline {
    const serialize = (config: OilPaintConfig): Uint8Array => {
    const buf: number[] = [];
  pushU32(buf, 'radius', config.radius ?? 4);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'oil_paint', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Emboss */
  emboss(): Pipeline {
    const node = this._pipe.applyFilter(this._node, 'emboss', new Uint8Array(0));
    return new Pipeline(this._pipe, node);
  }

  /** Chromatic Aberration */
  chromaticAberration(config: ChromaticAberrationConfig): Pipeline {
    const serialize = (config: ChromaticAberrationConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'strength', config.strength ?? 5);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'chromatic_aberration', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Film Grain */
  filmGrain(config: FilmGrainConfig): Pipeline {
    const serialize = (config: FilmGrainConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'amount', config.amount ?? 0.1);
  pushF32(buf, 'size', config.size ?? 1);
  pushU32(buf, 'seed', config.seed ?? 42);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'film_grain', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Light Leak */
  lightLeak(config: LightLeakConfig): Pipeline {
    const serialize = (config: LightLeakConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'intensity', config.intensity ?? 0.5);
  pushF32(buf, 'position_x', config.positionX ?? 0.5);
  pushF32(buf, 'position_y', config.positionY ?? 0.5);
  pushF32(buf, 'radius', config.radius ?? 0.5);
  pushF32(buf, 'warmth', config.warmth ?? 0.8);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'light_leak', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Poisson Noise */
  poissonNoise(config: PoissonNoiseConfig): Pipeline {
    const serialize = (config: PoissonNoiseConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'scale', config.scale ?? 100);
  pushU32(buf, 'seed', config.seed ?? 42);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'poisson_noise', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Charcoal */
  charcoal(config: CharcoalConfig): Pipeline {
    const serialize = (config: CharcoalConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'radius', config.radius ?? 1);
  pushF32(buf, 'sigma', config.sigma ?? 1);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'charcoal', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Gaussian Noise */
  gaussianNoise(config: GaussianNoiseConfig): Pipeline {
    const serialize = (config: GaussianNoiseConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'amount', config.amount ?? 10);
  pushF32(buf, 'mean', config.mean ?? 0);
  pushF32(buf, 'sigma', config.sigma ?? 25);
  pushU32(buf, 'seed', config.seed ?? 42);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'gaussian_noise', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Halftone */
  halftone(config: HalftoneConfig): Pipeline {
    const serialize = (config: HalftoneConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'dot_size', config.dotSize ?? 8);
  pushF32(buf, 'angle_offset', config.angleOffset ?? 0);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'halftone', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Levels */
  levels(config: LevelsConfig): Pipeline {
    const serialize = (config: LevelsConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'black', config.black ?? 0);
  pushF32(buf, 'white', config.white ?? 1);
  pushF32(buf, 'gamma', config.gamma ?? 1);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'levels', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Solarize */
  solarize(config: SolarizeConfig): Pipeline {
    const serialize = (config: SolarizeConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'threshold', config.threshold ?? 0.5);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'solarize', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Brightness */
  brightness(config: BrightnessConfig): Pipeline {
    const serialize = (config: BrightnessConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'amount', config.amount ?? 0);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'brightness', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Gamma */
  gamma(config: GammaConfig): Pipeline {
    const serialize = (config: GammaConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'gamma', config.gamma ?? 1);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'gamma', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Burn */
  burn(config: BurnConfig): Pipeline {
    const serialize = (config: BurnConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'amount', config.amount ?? 0.5);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'burn', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Invert */
  invert(): Pipeline {
    const node = this._pipe.applyFilter(this._node, 'invert', new Uint8Array(0));
    return new Pipeline(this._pipe, node);
  }

  /** Sigmoidal Contrast */
  sigmoidalContrast(config: SigmoidalContrastConfig): Pipeline {
    const serialize = (config: SigmoidalContrastConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'strength', config.strength ?? 3);
  pushF32(buf, 'midpoint', config.midpoint ?? 0.5);
  pushBool(buf, 'sharpen', config.sharpen ?? true);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'sigmoidal_contrast', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Contrast */
  contrast(config: ContrastConfig): Pipeline {
    const serialize = (config: ContrastConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'amount', config.amount ?? 0);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'contrast', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Dodge */
  dodge(config: DodgeConfig): Pipeline {
    const serialize = (config: DodgeConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'amount', config.amount ?? 0.5);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'dodge', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Posterize */
  posterize(config: PosterizeConfig): Pipeline {
    const serialize = (config: PosterizeConfig): Uint8Array => {
    const buf: number[] = [];
  pushU32(buf, 'levels', config.levels ?? 4);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'posterize', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Exposure */
  exposure(config: ExposureConfig): Pipeline {
    const serialize = (config: ExposureConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'ev', config.ev ?? 0);
  pushF32(buf, 'offset', config.offset ?? 0);
  pushF32(buf, 'gamma_correction', config.gammaCorrection ?? 1);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'exposure', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Gaussian Blur */
  gaussianBlur(config: GaussianBlurConfig): Pipeline {
    const serialize = (config: GaussianBlurConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'radius', config.radius ?? 1);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'gaussian_blur', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Motion Blur */
  motionBlur(config: MotionBlurConfig): Pipeline {
    const serialize = (config: MotionBlurConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'angle', config.angle ?? 0);
  pushF32(buf, 'length', config.length ?? 10);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'motion_blur', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Sharpen */
  sharpen(config: SharpenConfig): Pipeline {
    const serialize = (config: SharpenConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'radius', config.radius ?? 1);
  pushF32(buf, 'amount', config.amount ?? 1);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'sharpen', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Bilateral */
  bilateral(config: BilateralConfig): Pipeline {
    const serialize = (config: BilateralConfig): Uint8Array => {
    const buf: number[] = [];
  pushU32(buf, 'diameter', config.diameter ?? 5);
  pushF32(buf, 'sigma_color', config.sigmaColor ?? 0.1);
  pushF32(buf, 'sigma_space', config.sigmaSpace ?? 10);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'bilateral', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** High Pass */
  highPass(config: HighPassConfig): Pipeline {
    const serialize = (config: HighPassConfig): Uint8Array => {
    const buf: number[] = [];
  pushF32(buf, 'radius', config.radius ?? 3);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'high_pass', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Box Blur */
  boxBlur(config: BoxBlurConfig): Pipeline {
    const serialize = (config: BoxBlurConfig): Uint8Array => {
    const buf: number[] = [];
  pushU32(buf, 'radius', config.radius ?? 1);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'box_blur', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  /** Median */
  median(config: MedianConfig): Pipeline {
    const serialize = (config: MedianConfig): Uint8Array => {
    const buf: number[] = [];
  pushU32(buf, 'radius', config.radius ?? 1);
  return new Uint8Array(buf);
    };
    const node = this._pipe.applyFilter(this._node, 'median', serialize(config));
    return new Pipeline(this._pipe, node);
  }

  // ─── Write methods ──────────────────────────────────────────────────────

  /** Encode to format by name (generic). */
  write(format: string, quality?: number): Uint8Array {
    return this._pipe.write(this._node, format, quality);
  }

  /** Encode as PNG. */
  writePng(quality?: number): Uint8Array {
    return this._pipe.write(this._node, 'png', quality);
  }

  /** Encode as JPEG. */
  writeJpeg(quality?: number): Uint8Array {
    return this._pipe.write(this._node, 'jpeg', quality);
  }

  /** Encode as WebP. */
  writeWebp(quality?: number): Uint8Array {
    return this._pipe.write(this._node, 'webp', quality);
  }

  /** Encode as GIF. */
  writeGif(quality?: number): Uint8Array {
    return this._pipe.write(this._node, 'gif', quality);
  }

  /** Encode as BMP. */
  writeBmp(quality?: number): Uint8Array {
    return this._pipe.write(this._node, 'bmp', quality);
  }

  /** Encode as TIFF. */
  writeTiff(quality?: number): Uint8Array {
    return this._pipe.write(this._node, 'tiff', quality);
  }

  /** Encode as QOI. */
  writeQoi(quality?: number): Uint8Array {
    return this._pipe.write(this._node, 'qoi', quality);
  }

  /** Encode as ICO. */
  writeIco(quality?: number): Uint8Array {
    return this._pipe.write(this._node, 'ico', quality);
  }

  /** Encode as TGA. */
  writeTga(quality?: number): Uint8Array {
    return this._pipe.write(this._node, 'tga', quality);
  }

  /** Encode as PNM. */
  writePnm(quality?: number): Uint8Array {
    return this._pipe.write(this._node, 'pnm', quality);
  }

  /** Encode as EXR. */
  writeExr(quality?: number): Uint8Array {
    return this._pipe.write(this._node, 'exr', quality);
  }

  /** Encode as HDR. */
  writeHdr(quality?: number): Uint8Array {
    return this._pipe.write(this._node, 'hdr', quality);
  }

  // ─── Discovery ──────────────────────────────────────────────────────────

  /** List all available operations. */
  /** List all available operations (auto-loads WASM module). */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  static listOperations(): any[] {
    const { pipelineV2 } = require('../v2-generated/rasmcore-v2-image.js');
    const pipe = new pipelineV2.ImagePipelineV2();
    return pipe.listOperations();
  }

  /** List all available operations from a pre-loaded pipeline class. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  static listOperationsFromRaw(PipelineClass: any): any[] {
    const pipe = new PipelineClass();
    return pipe.listOperations();
  }
}
