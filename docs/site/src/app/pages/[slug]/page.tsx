import { getManualPages } from '@/lib/pages';
import { renderAdocString } from '@/lib/asciidoc';
import Link from 'next/link';

export async function generateStaticParams() {
  const pages = await getManualPages();
  return pages.map(p => ({ slug: p.slug }));
}

export default async function GuidePage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const pages = await getManualPages();
  const page = pages.find(p => p.slug === slug);

  if (!page) {
    return <><h1>Not Found</h1><p>Page &quot;{slug}&quot; not found.</p></>;
  }

  const html = renderAdocString(page.content);

  return (
    <>
      <div className="breadcrumb">
        <Link href="/">Home</Link> &rsaquo; {page.section} &rsaquo; {page.title}
      </div>
      <h1>{page.title}</h1>
      <div dangerouslySetInnerHTML={{ __html: html }} />
    </>
  );
}
