import type { ChainNode } from '../types';
import type { LayerState } from '../hooks/useLayers';

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

export function generateCode(
  layers: LayerState[],
  chain: ChainNode[],
  format: string,
  quality: number,
): string {
  const lines: string[] = [];
  lines.push(`import { rcimage } from '@rasmcore/sdk';`);
  lines.push('');

  const visibleLayers = layers.filter((l) => l.visible);
  if (visibleLayers.length <= 1) {
    lines.push('const result = rcimage.load(imageBytes)');
  } else {
    lines.push(`const bg = rcimage.load(layer0Bytes)`);
    for (const node of visibleLayers[0]?.chain || []) {
      const args = node.op.params
        .map((p) => {
          const val = node.paramValues[p.name];
          return typeof val === 'string' ? `'${val}'` : val;
        })
        .join(', ');
      lines.push(`  .${node.op.name}(${args})`);
    }
    lines.push('');
    for (let i = 1; i < visibleLayers.length; i++) {
      const l = visibleLayers[i];
      lines.push(`const layer${i} = bg.loadLayer(layer${i}Bytes)`);
      for (const node of l.chain) {
        const args = node.op.params
          .map((p) => {
            const val = node.paramValues[p.name];
            return typeof val === 'string' ? `'${val}'` : val;
          })
          .join(', ');
        lines.push(`  .${node.op.name}(${args})`);
      }
      const opts: string[] = [];
      if (l.x) opts.push(`x: ${l.x}`);
      if (l.y) opts.push(`y: ${l.y}`);
      if (l.blendMode && l.blendMode !== 'over') opts.push(`blend: '${l.blendMode}'`);
      const optsStr = opts.length ? `{ ${opts.join(', ')} }` : '';
      lines.push(`bg.composite(layer${i}${optsStr ? ', ' + optsStr : ''});`);
      lines.push('');
    }
    lines.push('const result = bg');
  }

  for (const node of visibleLayers.length <= 1 ? chain : []) {
    if (node.op.params.length === 0) {
      lines.push(`  .${node.op.name}()`);
    } else {
      const args: (string | number | boolean)[] = [];
      for (const p of node.op.params) {
        const val = node.paramValues[p.name];
        if (p.type === 'color') {
          const [r, g, b] = hexToRgb((val as string) || '#808080');
          args.push(r, g, b);
        } else if (p.type === 'enum' || typeof val === 'string') {
          args.push(`'${val}'`);
        } else {
          args.push(val);
        }
      }
      lines.push(`  .${node.op.name}(${args.join(', ')})`);
    }
  }

  const formatMap: Record<string, { method: string; config: string }> = {
    jpeg: { method: 'toJpeg', config: `{ quality: ${quality} }` },
    png: { method: 'toPng', config: '{}' },
    webp: { method: 'toWebp', config: `{ quality: ${quality} }` },
  };
  const fmt = formatMap[format] || formatMap.jpeg;
  lines.push(`  .${fmt.method}(${fmt.config});`);

  return lines.join('\n');
}

export function highlightCode(code: string): string {
  return code
    .replace(/\b(import|from|const)\b/g, '<span class="kw">$1</span>')
    .replace(/'([^']+)'/g, '\'<span class="str">$1</span>\'')
    .replace(/\b(\d+\.?\d*)\b/g, '<span class="num">$1</span>')
    .replace(/\.(\w+)\(/g, '.<span class="fn">$1</span>(');
}
