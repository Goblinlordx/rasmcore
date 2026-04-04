import { getRegistry, snakeToKebab, snakeToTitle } from '@/lib/registry';
import { renderAdocFile } from '@/lib/asciidoc';
import { ParamTable } from '@/components/ParamTable';
import { CodeExample } from '@/components/CodeExample';
import { SplitView } from '@/components/SplitView';
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

  // Check if example images exist
  const examplesDir = join(process.cwd(), 'public', 'assets', 'examples');
  const hasExample = existsSync(join(examplesDir, `${op.name}-after.png`));

  return (
    <>
      <div className="breadcrumb">
        <Link href="/">Operations</Link> &rsaquo; {snakeToTitle(op.category || '')} &rsaquo; {name}
      </div>
      <h1>{name}</h1>

      {hasExample && (
        <SplitView
          beforeSrc="/assets/examples/reference.png"
          afterSrc={`/assets/examples/${op.name}-after.png`}
        />
      )}

      {adocHtml && (
        <div className="op-description" dangerouslySetInnerHTML={{ __html: adocHtml }} />
      )}

      <CodeExample name={op.name} params={op.params} />

      <h2>Parameters</h2>
      <ParamTable params={op.params} />

      <div className="op-meta">
        <p><strong>Category:</strong> {snakeToTitle(op.category || 'uncategorized')}</p>
        <p><strong>Registry name:</strong> <code>{op.name}</code></p>
      </div>
    </>
  );
}
