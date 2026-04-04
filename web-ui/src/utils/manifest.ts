import type { Operation, UiParam } from '../types';

declare const __SDK_PATH__: string;
const SDK_PATH = typeof __SDK_PATH__ !== 'undefined' ? __SDK_PATH__ : '/sdk';
const V2_SDK_PATH = `${typeof __SDK_PATH__ !== 'undefined' ? __SDK_PATH__ : '/sdk'}/v2`;

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
      t === 'u64' ||
      t === 'i64' ||
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
    'rc.point': 'point',
    'rc.path': 'path',
    'rc.box_select': 'box_select',
  };
  const mapped = p.hint ? hintMap[p.hint] : undefined;
  // u64/i64 types default to spinner (large integer fields like seeds)
  const typeDefault = t === 'u64' || t === 'i64' ? 'spinner' : 'number';
  return {
    name: p.name,
    type: mapped || typeDefault,
    hint: p.hint || '',
    witType: t,
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

  // Discover operations from V2 registry (authoritative source of truth)
  const operations: Operation[] = [];
  try {
    const sdk = await import(/* @vite-ignore */ `${V2_SDK_PATH}/rasmcore-v2-image.js`);
    const PipelineClass = sdk.pipelineV2.ImagePipelineV2;
    const tempPipe = new PipelineClass();
    const v2Ops = tempPipe.listOperations();

    for (const op of v2Ops) {
      if (op.kind !== 'filter' && op.kind !== 'transform' && op.kind !== 'color-conversion') continue;
      const camelName = snakeToCamel(op.name);

      // Build params from V2 registry if not already in param-manifest
      if (!paramMeta[camelName] && op.params && op.params.length > 0) {
        paramMeta[camelName] = op.params.map((p: { name: string; valueType: string; min?: number; max?: number; step?: number; defaultVal?: number; hint?: string }) =>
          mapParam({
            name: p.name,
            type: p.valueType.replace('-val', ''),
            min: p.min ?? null,
            max: p.max ?? null,
            step: p.step ?? null,
            default: p.defaultVal ?? null,
            hint: p.hint || undefined,
          }),
        ).filter((p: UiParam | null): p is UiParam => p !== null);
      }

      const rawCat = op.category || categories[camelName] || 'other';
      const category = CATEGORY_LABELS[rawCat] || rawCat.charAt(0).toUpperCase() + rawCat.slice(1);
      operations.push({
        name: camelName,
        category,
        params: paramMeta[camelName] || [],
      });
    }
  } catch (e) {
    console.warn('Could not discover V2 operations:', e);
  }

  // Discover write formats from V2 registry
  let writeFormats: string[] = ['jpeg', 'png', 'webp'];
  try {
    const sdk = await import(/* @vite-ignore */ `${V2_SDK_PATH}/rasmcore-v2-image.js`);
    const PipelineClass = sdk.pipelineV2.ImagePipelineV2;
    const tempPipe = new PipelineClass();
    const v2Ops = tempPipe.listOperations();
    const encoders = v2Ops
      .filter((op: { kind: string; name: string }) => op.kind === 'encoder')
      .map((op: { name: string }) => op.name.replace('_encode', ''));
    if (encoders.length > 0) writeFormats = encoders;
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
