import { readFileSync, readdirSync, existsSync } from 'fs';
import { join, basename } from 'path';
import type { ManualPage } from './types';

const PAGES_DIR = join(process.cwd(), '..', 'pages');

let cached: ManualPage[] | null = null;

export async function getManualPages(): Promise<ManualPage[]> {
  if (cached) return cached;

  if (!existsSync(PAGES_DIR)) {
    cached = [];
    return cached;
  }

  const pages: ManualPage[] = [];
  for (const file of readdirSync(PAGES_DIR)) {
    if (!file.endsWith('.adoc')) continue;
    const raw = readFileSync(join(PAGES_DIR, file), 'utf8');

    let title = basename(file, '.adoc');
    let section = 'Guides';
    let order = 99;

    const titleMatch = raw.match(/^= (.+)$/m);
    if (titleMatch) title = titleMatch[1];

    const sectionMatch = raw.match(/^:section:\s*(.+)$/m);
    if (sectionMatch) section = sectionMatch[1].trim();

    const orderMatch = raw.match(/^:order:\s*(\d+)$/m);
    if (orderMatch) order = parseInt(orderMatch[1]);

    pages.push({ title, section, order, slug: basename(file, '.adoc'), content: raw });
  }

  cached = pages;
  return cached;
}
