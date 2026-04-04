import type { Metadata } from 'next';
import './globals.css';
import { Sidebar } from '@/components/Sidebar';
import { getRegistry } from '@/lib/registry';
import { getManualPages } from '@/lib/pages';

export const metadata: Metadata = {
  title: 'rasmcore docs',
  description: 'rasmcore image processing pipeline documentation',
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const registry = await getRegistry();
  const manualPages = await getManualPages();

  return (
    <html lang="en">
      <body>
        <div className="layout">
          <Sidebar registry={registry} manualPages={manualPages} />
          <main className="content">{children}</main>
        </div>
      </body>
    </html>
  );
}
