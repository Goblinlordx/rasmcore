import { getRegistry, snakeToKebab, snakeToTitle } from '@/lib/registry';
import { renderAdocFile } from '@/lib/asciidoc';
import Link from 'next/link';

export async function generateStaticParams() {
  const registry = await getRegistry();
  const params = [
    ...registry.encoders.map(e => ({ slug: `${snakeToKebab(e.name)}-encoder` })),
    ...registry.decoders.map(d => ({ slug: `${snakeToKebab(d.name)}-decoder` })),
  ];
  return params;
}

export default async function CodecPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const registry = await getRegistry();

  const encoder = registry.encoders.find(e => `${snakeToKebab(e.name)}-encoder` === slug);
  const decoder = registry.decoders.find(d => `${snakeToKebab(d.name)}-decoder` === slug);
  const codec = encoder || decoder;

  if (!codec) {
    return <><h1>Not Found</h1><p>Codec &quot;{slug}&quot; not found.</p></>;
  }

  const isEncoder = !!encoder;
  const name = `${codec.displayName} ${isEncoder ? 'Encoder' : 'Decoder'}`;
  const adocHtml = renderAdocFile(codec.docPath);

  return (
    <>
      <div className="breadcrumb">
        <Link href="/">Operations</Link> &rsaquo; {isEncoder ? 'Encoders' : 'Decoders'} &rsaquo; {codec.displayName}
      </div>
      <h1>{name}</h1>

      {adocHtml && (
        <div className="op-description" dangerouslySetInnerHTML={{ __html: adocHtml }} />
      )}

      <div className="op-meta">
        {isEncoder && encoder && <p><strong>MIME type:</strong> <code>{encoder.mime}</code></p>}
        <p><strong>Extensions:</strong> {codec.extensions.map(x => <code key={x}>.{x}</code>).reduce<React.ReactNode[]>((acc, el, i) => i === 0 ? [el] : [...acc, ', ', el], [])}</p>
      </div>
    </>
  );
}
