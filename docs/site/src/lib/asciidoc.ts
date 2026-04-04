import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

// eslint-disable-next-line @typescript-eslint/no-require-imports
const Asciidoctor = require('@asciidoctor/core');
const asciidoctor = Asciidoctor();

const PROJECT_ROOT = join(process.cwd(), '..', '..');

export function renderAdocFile(docPath: string): string | null {
  if (!docPath) return null;
  const fullPath = join(PROJECT_ROOT, docPath);
  if (!existsSync(fullPath)) return null;
  const content = readFileSync(fullPath, 'utf8');
  return asciidoctor.convert(content, { safe: 'safe', standalone: false });
}

export function renderAdocString(content: string): string {
  return asciidoctor.convert(content, { safe: 'safe', standalone: false });
}
