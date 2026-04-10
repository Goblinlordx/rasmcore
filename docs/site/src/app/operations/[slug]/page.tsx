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

  // Load showcase params (matches prerendered image) so UI defaults = prerender values
  let showcaseParams: Record<string, number | boolean> | undefined;
  try {
    const showcasePath = join(examplesDir, 'showcase-params.json');
    if (existsSync(showcasePath)) {
      const all = JSON.parse(require('fs').readFileSync(showcasePath, 'utf8'));
      showcaseParams = all[op.name];
    }
  } catch { /* ignore */ }

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
        showcaseParams={showcaseParams}
      />

      {adocHtml && (
        <div className="op-description" dangerouslySetInnerHTML={{ __html: adocHtml }} />
      )}

      <h2>Parameters</h2>
      <ParamTable params={op.params} />

      <div className="op-meta">
        <p><strong>Category:</strong> {snakeToTitle(op.category || 'uncategorized')}</p>
        <p><strong>Registry name:</strong> <code>{op.name}</code></p>
        {op.cost && (
          <p>
            <strong>Cost:</strong>{' '}
            <code>{op.cost}</code>
            <span className="cost-legend"> where n = pixels{
              op.cost.includes(' r') ? ', r = radius' : ''
            }{
              op.cost.includes(' d') ? ', d = diameter' : ''
            }{
              op.cost.includes('sigma') ? ', σ = sigma' : ''
            }{
              op.cost.includes('sr') ? ', sr = search radius' : ''
            }{
              op.cost.includes('pr') ? ', pr = patch radius' : ''
            }{
              op.cost.includes('length') ? ', length = blur length' : ''
            }{
              op.cost.includes('levels') ? ', levels = pyramid levels' : ''
            }</span>
          </p>
        )}
        {op.gpuCost && (
          <p><strong>GPU Cost:</strong> <code>{op.gpuCost}</code></p>
        )}
      </div>
    </>
  );
}
