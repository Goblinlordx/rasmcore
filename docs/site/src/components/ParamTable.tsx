import type { ParamDescriptor } from '@/lib/types';

function snakeToTitle(s: string) {
  return s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

export function ParamTable({ params }: { params: ParamDescriptor[] }) {
  if (!params || params.length === 0) {
    return <p><em>No parameters.</em></p>;
  }

  return (
    <table className="param-table">
      <thead>
        <tr>
          <th>Parameter</th>
          <th>Type</th>
          <th>Range</th>
          <th>Default</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        {params.map(p => {
          const range = p.min != null && p.max != null
            ? `${p.min} – ${p.max}${p.step != null ? ` (step ${p.step})` : ''}`
            : '—';
          return (
            <tr key={p.name}>
              <td><code>{p.name}</code></td>
              <td><code>{p.type}</code></td>
              <td>{range}</td>
              <td>{p.default != null ? String(p.default) : '—'}</td>
              <td>{p.description || ''}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
