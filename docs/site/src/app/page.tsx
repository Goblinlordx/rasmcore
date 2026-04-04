import Link from 'next/link';
import { getRegistry, snakeToKebab, snakeToTitle } from '@/lib/registry';

export default async function IndexPage() {
  const registry = await getRegistry();

  const categories: Record<string, typeof registry.filters> = {};
  for (const f of registry.filters) {
    const cat = f.category || 'uncategorized';
    if (!categories[cat]) categories[cat] = [];
    categories[cat].push(f);
  }

  return (
    <>
      <h1>Operations</h1>
      <div className="op-index">
        {Object.keys(categories).sort().map(cat => (
          <div key={cat}>
            <h2>{snakeToTitle(cat)}</h2>
            <ul>
              {categories[cat].sort((a, b) => a.name.localeCompare(b.name)).map(op => (
                <li key={op.name}>
                  <Link href={`/operations/${snakeToKebab(op.name)}`}>
                    {op.displayName || snakeToTitle(op.name)}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        ))}

        {registry.encoders.length > 0 && (
          <div>
            <h2>Encoders</h2>
            <ul>
              {registry.encoders.map(e => (
                <li key={e.name}>
                  <Link href={`/codecs/${snakeToKebab(e.name)}-encoder`}>{e.displayName}</Link>
                </li>
              ))}
            </ul>
          </div>
        )}

        {registry.decoders.length > 0 && (
          <div>
            <h2>Decoders</h2>
            <ul>
              {registry.decoders.map(d => (
                <li key={d.name}>
                  <Link href={`/codecs/${snakeToKebab(d.name)}-decoder`}>{d.displayName}</Link>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </>
  );
}
