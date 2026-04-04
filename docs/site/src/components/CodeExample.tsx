import type { ParamDescriptor } from '@/lib/types';

function snakeToCamel(s: string) {
  return s.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());
}

export function CodeExample({ name, params }: { name: string; params: ParamDescriptor[] }) {
  const method = snakeToCamel(name);

  if (!params || params.length === 0) {
    const code = `Pipeline.open(imageBytes)\n  .${method}()\n  .writePng();`;
    return <><h2>Usage</h2><pre><code>{code}</code></pre></>;
  }

  const args = params.map(p => {
    const camel = snakeToCamel(p.name);
    let val = p.default != null && Math.abs(p.default) > 1e-6 ? p.default : null;
    if (val == null) {
      if (p.min != null && p.max != null) val = p.min + (p.max - p.min) * 0.3;
      else val = 0.5;
    }
    if (p.type === 'bool') return `    ${camel}: true`;
    if (Number.isInteger(val) || p.type === 'u32' || p.type === 'i32') return `    ${camel}: ${Math.round(val)}`;
    return `    ${camel}: ${Number(val.toFixed(2))}`;
  }).join(',\n');

  const code = [
    'Pipeline.open(imageBytes)',
    `  .${method}({`,
    args,
    '  })',
    '  .writePng();',
  ].join('\n');

  return <><h2>Usage</h2><pre><code>{code}</code></pre></>;
}
