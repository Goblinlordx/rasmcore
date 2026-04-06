export interface ParamDescriptor {
  name: string;
  type: string;
  min: number | null;
  max: number | null;
  step: number | null;
  default: number | null;
  hint: string | null;
  description: string;
}

export interface FilterInfo {
  name: string;
  displayName: string;
  category: string;
  docPath: string;
  params: ParamDescriptor[];
  /** Algorithmic cost relative to pixel count (n). e.g. "O(n)", "O(n * r) separable" */
  cost?: string;
  /** GPU-specific cost if different from CPU. */
  gpuCost?: string;
}

export interface EncoderInfo {
  name: string;
  displayName: string;
  mime: string;
  docPath: string;
  extensions: string[];
}

export interface DecoderInfo {
  name: string;
  displayName: string;
  docPath: string;
  extensions: string[];
}

export interface Registry {
  filters: FilterInfo[];
  encoders: EncoderInfo[];
  decoders: DecoderInfo[];
}

export interface ManualPage {
  title: string;
  section: string;
  order: number;
  sectionOrder: number;
  slug: string;
  content: string;
}
