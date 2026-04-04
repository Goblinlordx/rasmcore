import { readFileSync, existsSync } from 'fs';
import type { Registry } from './types';

const REGISTRY_PATH = '/tmp/v2_registry_docs.json';

let cached: Registry | null = null;

export async function getRegistry(): Promise<Registry> {
  if (cached) return cached;

  if (!existsSync(REGISTRY_PATH)) {
    console.warn(`Registry not found at ${REGISTRY_PATH} — run dump_registry first`);
    cached = { filters: [], encoders: [], decoders: [] };
    return cached;
  }

  const raw = JSON.parse(readFileSync(REGISTRY_PATH, 'utf8'));
  cached = raw as Registry;
  return cached;
}

export function snakeToKebab(s: string): string {
  return s.replace(/_/g, '-');
}

export function snakeToTitle(s: string): string {
  return s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
