/** Shared types for the pipeline builder UI. */

/** Parameter metadata from param-manifest.json. */
export interface ManifestParam {
  name: string;
  type: string;
  min: number | null;
  max: number | null;
  step: number | null;
  default: number | string | null;
  label: string;
  hint: string;
}

/** Filter entry from param-manifest.json. */
export interface ManifestFilter {
  name: string;
  category: string;
  group: string;
  variant: string;
  reference: string;
  params: ManifestParam[];
}

/** Generator/compositor/mapper entry from manifest. */
export interface ManifestEntry {
  name: string;
  category: string;
  group: string;
  variant: string;
  reference: string;
  kind: string;
}

/** Full manifest structure. */
export interface Manifest {
  filters: ManifestFilter[];
  generators?: ManifestEntry[];
  compositors?: ManifestEntry[];
  mappers?: ManifestEntry[];
}

/** UI control type derived from hints. */
export type ControlType =
  | 'number' // default linear slider
  | 'color' // color picker
  | 'toggle' // boolean switch
  | 'text' // text input
  | 'spinner' // number input with arrows (no slider)
  | 'log_slider' // logarithmic slider
  | 'signed_slider' // bipolar slider centered at 0
  | 'opacity' // thin slider + text input
  | 'temperature' // slider with gradient track
  | 'enum' // dropdown select
  | 'point' // canvas point selector (click to place)
  | 'path' // canvas path drawer (freehand stroke)
  | 'box_select'; // canvas rectangle selector (drag to define)

/** Resolved parameter for UI rendering. */
export interface UiParam {
  name: string;
  type: ControlType;
  hint: string;
  witType?: string;
  min?: number;
  max?: number;
  step?: number;
  default: number | string | boolean;
  label: string;
  options?: string[];
}

/** Operation definition for the palette/toolbar. */
export interface Operation {
  name: string;
  category: string;
  params: UiParam[];
}

/** A node in the processing chain. */
export interface ChainNode {
  id: number;
  op: Operation;
  paramValues: Record<string, number | string | boolean>;
  applied: boolean;
  timingMs: number;
}

/** Layer state. */
export interface LayerState {
  id: number;
  name: string;
  visible: boolean;
  blendMode: string;
  opacity: number;
  imageBytes: ArrayBuffer | null;
  chain: ChainNode[];
  thumbnail: string | null;
}

/** Worker message types. */
export type WorkerRequest =
  | { type: 'init'; sdkPath: string }
  | { type: 'load'; imageBytes: ArrayBuffer }
  | { type: 'process'; chain: WorkerChainOp[]; mode: 'preview' | 'full' }
  | { type: 'export'; chain: WorkerChainOp[]; format: string; quality: number }
  | { type: 'composite'; layers: WorkerLayerData[] };

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'loaded'; width: number; height: number; thumbnail: string }
  | {
      type: 'result';
      imageData: ArrayBuffer;
      width: number;
      height: number;
      timings: WorkerTiming[];
    }
  | { type: 'exported'; blob: ArrayBuffer; mimeType: string }
  | { type: 'error'; message: string };

export interface WorkerChainOp {
  name: string;
  args: (number | string | boolean)[];
}

export interface WorkerLayerData {
  imageBytes: ArrayBuffer;
  blendMode: string;
  opacity: number;
  visible: boolean;
}

export interface WorkerTiming {
  op: string;
  us: number;
}
