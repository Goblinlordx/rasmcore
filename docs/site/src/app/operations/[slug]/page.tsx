import { getRegistry, snakeToKebab, snakeToTitle } from '@/lib/registry';
import { renderAdocFile } from '@/lib/asciidoc';
import { ParamTable } from '@/components/ParamTable';
import { Playground } from '@/components/Playground';
import { existsSync } from 'fs';
import { join } from 'path';
import Link from 'next/link';

export async function generateStaticParams() {
  const registry = await getRegistry();
  return registry.filters.map(f => ({ slug: snakeToKebab(f.name) }));
}

export default async function OperationPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const registry = await getRegistry();
  const op = registry.filters.find(f => snakeToKebab(f.name) === slug);

  if (!op) {
    return <><h1>Not Found</h1><p>Operation &quot;{slug}&quot; not found.</p></>;
  }

  const name = op.displayName || snakeToTitle(op.name);
  const adocHtml = renderAdocFile(op.docPath);

  // Check if static example image exists (used as initial preview before WASM loads)
  const examplesDir = join(process.cwd(), 'public', 'assets', 'examples');
  const hasExample = existsSync(join(examplesDir, `${op.name}-after.png`));

  return (
    <>
      <div className="breadcrumb">
        <Link href="/">Operations</Link> &rsaquo; {snakeToTitle(op.category || '')} &rsaquo; {name}
      </div>
      <h1>{name}</h1>

      {/* Interactive playground — shows static image initially, loads WASM on interaction */}
      <Playground
        filterName={op.name}
        params={op.params}
        referenceImageUrl="/assets/examples/reference.png"
        staticAfterUrl={hasExample ? `/assets/examples/${op.name}-after.png` : undefined}
      />

      {adocHtml && (
        <div className="op-description" dangerouslySetInnerHTML={{ __html: adocHtml }} />
      )}

      <h2>Parameters</h2>
      <ParamTable params={op.params} />

      <div className="op-meta">
        <p><strong>Category:</strong> {snakeToTitle(op.category || 'uncategorized')}</p>
        <p><strong>Registry name:</strong> <code>{op.name}</code></p>
      </div>
    </>
  );
}
