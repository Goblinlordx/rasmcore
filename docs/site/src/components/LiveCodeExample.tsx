'use client';

function snakeToCamel(s: string) {
  return s.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());
}

interface LiveCodeExampleProps {
  name: string;
  values: Record<string, number | boolean>;
}

export function LiveCodeExample({ name, values }: LiveCodeExampleProps) {
  const method = snakeToCamel(name);
  const entries = Object.entries(values);

  let code: string;
  if (entries.length === 0) {
    code = [
      'Pipeline.open(imageBytes)',
      `  .${method}()`,
      '  .writePng();',
    ].join('\n');
  } else {
    const args = entries.map(([k, v]) => {
      const camel = snakeToCamel(k);
      if (typeof v === 'boolean') return `    ${camel}: ${v}`;
      if (Number.isInteger(v)) return `    ${camel}: ${v}`;
      return `    ${camel}: ${Number(v).toFixed(2)}`;
    }).join(',\n');

    code = [
      'Pipeline.open(imageBytes)',
      `  .${method}({`,
      args,
      '  })',
      '  .writePng();',
    ].join('\n');
  }

  return (
    <div style={{ margin: '1rem 0' }}>
      <h3 style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '0.3rem' }}>Live Code</h3>
      <pre style={{ margin: 0 }}><code>{code}</code></pre>
    </div>
  );
}
