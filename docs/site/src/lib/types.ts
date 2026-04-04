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
  slug: string;
  content: string;
}
