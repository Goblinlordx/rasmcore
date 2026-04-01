import type { Operation, UiParam } from '../types';

declare const __SDK_PATH__: string;
const SDK_PATH = typeof __SDK_PATH__ !== 'undefined' ? __SDK_PATH__ : './sdk';

function snakeToCamel(s: string) {
  return s.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());
}

interface ManifestRawParam {
  name: string;
  type: string;
  hint?: string;
  min?: string | number | null;
  max?: string | number | null;
  step?: string | number | null;
  default?: string | number | null;
  label?: string;
  options?: string[];
}

interface ManifestRawFilter {
  name: string;
  category: string;
  group?: string;
  variant?: string;
  reference?: string;
  params: ManifestRawParam[];
}

function mapParam(p: ManifestRawParam): UiParam | null {
  const t = p.type;
  const hasHint = p.hint && p.hint.startsWith('rc.');
  if (
    !(
      t === 'f32' ||
      t === 'f64' ||
      t === 'u32' ||
      t === 'u16' ||
      t === 'u8' ||
      t === 'i32' ||
      t === 'bool' ||
      hasHint
    )
  )
    return null;

  if (p.hint === 'rc.color_rgb' || p.hint === 'rc.color_rgba')
    return {
      name: p.name,
      type: 'color',
      hint: p.hint,
      default: '#808080',
      label: p.label || p.name,
    };
  if (p.hint === 'rc.angle_deg')
    return {
      name: p.name,
      type: 'number',
      hint: p.hint,
      min: 0,
      max: 360,
      step: 1,
      default: 0,
      label: p.label || p.name,
    };
  if (p.hint === 'rc.percentage')
    return {
      name: p.name,
      type: 'number',
      hint: p.hint,
      min: 0,
      max: 100,
      step: 1,
      default: 50,
      label: p.label || p.name,
    };
  if (p.hint === 'rc.toggle')
    return {
      name: p.name,
      type: 'toggle',
      hint: p.hint,
      default: p.default === 'true' || p.default === '1',
      label: p.label || p.name,
    };
  if (p.hint === 'rc.text')
    return {
      name: p.name,
      type: 'text',
      hint: p.hint,
      default: p.default || '',
      label: p.label || p.name,
    };

  const numMin = p.min != null ? parseFloat(String(p.min)) : undefined;
  const numMax = p.max != null ? parseFloat(String(p.max)) : undefined;
  const numStep = p.step != null ? parseFloat(String(p.step)) : undefined;
  const numDefault =
    p.default != null && p.default !== ''
      ? isNaN(Number(p.default))
        ? p.default
        : Number(p.default)
      : 0;

  const hintMap: Record<string, UiParam['type']> = {
    'rc.pixels': 'spinner',
    'rc.spinner': 'spinner',
    'rc.seed': 'spinner',
    'rc.log_slider': 'log_slider',
    'rc.signed_slider': 'signed_slider',
    'rc.opacity': 'opacity',
    'rc.temperature_k': 'temperature',
  };
  const mapped = p.hint ? hintMap[p.hint] : undefined;
  return {
    name: p.name,
    type: mapped || 'number',
    hint: p.hint || '',
    min: numMin,
    max: numMax,
    step: numStep,
    default: numDefault as number | string | boolean,
    label: p.label || p.name,
  };
}

const PIPELINE_EXTRAS: Record<string, { category: string; params: UiParam[] }> = {
  resize: {
    category: 'transform',
    params: [
      {
        name: 'width',
        type: 'number',
        hint: '',
        min: 1,
        max: 8192,
        step: 1,
        default: 256,
        label: 'Width in pixels',
      },
      {
        name: 'height',
        type: 'number',
        hint: '',
        min: 1,
        max: 8192,
        step: 1,
        default: 256,
        label: 'Height in pixels',
      },
      {
        name: 'filter',
        type: 'enum',
        hint: '',
        options: ['nearest', 'bilinear', 'bicubic', 'lanczos3'],
        default: 'lanczos3',
        label: 'Resize filter',
      },
    ],
  },
  crop: {
    category: 'transform',
    params: [
      {
        name: 'x',
        type: 'number',
        hint: '',
        min: 0,
        max: 8192,
        step: 1,
        default: 0,
        label: 'X offset',
      },
      {
        name: 'y',
        type: 'number',
        hint: '',
        min: 0,
        max: 8192,
        step: 1,
        default: 0,
        label: 'Y offset',
      },
      {
        name: 'width',
        type: 'number',
        hint: '',
        min: 1,
        max: 8192,
        step: 1,
        default: 256,
        label: 'Crop width',
      },
      {
        name: 'height',
        type: 'number',
        hint: '',
        min: 1,
        max: 8192,
        step: 1,
        default: 256,
        label: 'Crop height',
      },
    ],
  },
  rotate: {
    category: 'transform',
    params: [
      {
        name: 'angle',
        type: 'enum',
        hint: '',
        options: ['r90', 'r180', 'r270'],
        default: 'r90',
        label: 'Rotation angle',
      },
    ],
  },
  flip: {
    category: 'transform',
    params: [
      {
        name: 'direction',
        type: 'enum',
        hint: '',
        options: ['horizontal', 'vertical'],
        default: 'horizontal',
        label: 'Flip direction',
      },
    ],
  },
  grayscale: { category: 'color', params: [] },
  convertFormat: {
    category: 'color',
    params: [
      {
        name: 'target',
        type: 'enum',
        hint: '',
        options: ['rgb8', 'rgba8', 'gray8'],
        default: 'rgb8',
        label: 'Target pixel format',
      },
    ],
  },
};

const CATEGORY_LABELS: Record<string, string> = {
  spatial: 'Filters',
  adjustment: 'Adjustment',
  color: 'Color',
  edge: 'Edge',
  transform: 'Transform',
  alpha: 'Alpha',
  other: 'Other',
  enhancement: 'Enhancement',
  morphology: 'Morphology',
  threshold: 'Threshold',
  grading: 'Grading',
  tonemapping: 'Tonemapping',
  effect: 'Effects',
  distortion: 'Distortion',
  draw: 'Draw',
  generator: 'Generator',
  tool: 'Tool',
  advanced: 'Advanced',
};

const SKIP = new Set(['constructor', 'read', 'nodeInfo', 'composite', 'iccToSrgb', 'autoOrient']);

export interface ManifestGroupMeta {
  group: string;
  variant: string;
  reference: string;
}

export interface LoadedManifest {
  operations: Operation[];
  groups: Record<string, ManifestGroupMeta>;
  writeFormats: string[];
}

export async function loadManifest(): Promise<LoadedManifest> {
  const paramMeta: Record<string, UiParam[]> = {};
  const categories: Record<string, string> = {};
  const groups: Record<string, ManifestGroupMeta> = {};

  try {
    const manifest = await fetch(`${SDK_PATH}/param-manifest.json`).then((r) => r.json());
    for (const filter of manifest.filters as ManifestRawFilter[]) {
      const camelName = snakeToCamel(filter.name);
      categories[camelName] = filter.category;
      groups[camelName] = {
        group: filter.group || '',
        variant: filter.variant || '',
        reference: filter.reference || '',
      };
      paramMeta[camelName] = filter.params.map(mapParam).filter((p): p is UiParam => p !== null);
    }
  } catch (e) {
    console.warn('Could not load param-manifest.json:', e);
  }

  // Merge pipeline extras
  for (const [name, extra] of Object.entries(PIPELINE_EXTRAS)) {
    if (!paramMeta[name]) {
      paramMeta[name] = extra.params;
      categories[name] = extra.category;
    }
  }

  // Discover operations from SDK
  const operations: Operation[] = [];
  try {
    const sdk = await import(/* @vite-ignore */ `${SDK_PATH}/rasmcore-image.js`);
    const PipelineClass = sdk.pipeline.ImagePipeline;
    const tempPipe = new PipelineClass();
    const proto = Object.getPrototypeOf(tempPipe);
    const methodNames = Object.getOwnPropertyNames(proto).filter(
      (n: string) => typeof proto[n] === 'function' && !SKIP.has(n) && !n.startsWith('write'),
    );
    for (const name of methodNames) {
      if (!paramMeta[name]) continue;
      const rawCat = categories[name] || 'other';
      const category = CATEGORY_LABELS[rawCat] || rawCat.charAt(0).toUpperCase() + rawCat.slice(1);
      operations.push({ name, category, params: paramMeta[name] });
    }
  } catch (e) {
    console.warn('Could not discover operations from SDK:', e);
  }

  // Discover write formats
  let writeFormats: string[] = ['jpeg', 'png', 'webp'];
  try {
    const sdk = await import(/* @vite-ignore */ `${SDK_PATH}/rasmcore-image.js`);
    const fmts = sdk.pipeline.supportedWriteFormats();
    if (fmts && fmts.length > 0) writeFormats = fmts;
  } catch {
    /* fallback */
  }

  return { operations, groups, writeFormats };
}

export const MENU_MAP: Record<string, string> = {
  Filters: 'filters',
  Enhancement: 'filters',
  Edge: 'filters',
  Morphology: 'filters',
  Transform: 'transforms',
  Distortion: 'transforms',
  Effects: 'effects',
  Effect: 'effects',
  Draw: 'effects',
  Generator: 'effects',
  Grading: 'grading',
  Tonemapping: 'grading',
  Color: 'grading',
  Adjustment: 'grading',
};
