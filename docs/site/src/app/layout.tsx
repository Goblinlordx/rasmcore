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
      <head>
        <script type="importmap" dangerouslySetInnerHTML={{ __html: JSON.stringify({
          imports: {
            "@bytecodealliance/preview2-shim/cli": "/sdk/v2/preview2-shim/cli.js",
            "@bytecodealliance/preview2-shim/clocks": "/sdk/v2/preview2-shim/clocks.js",
            "@bytecodealliance/preview2-shim/filesystem": "/sdk/v2/preview2-shim/filesystem.js",
            "@bytecodealliance/preview2-shim/io": "/sdk/v2/preview2-shim/io.js",
            "@bytecodealliance/preview2-shim/random": "/sdk/v2/preview2-shim/random.js",
          }
        })}} />
      </head>
      <body>
        <div className="layout">
          <Sidebar registry={registry} manualPages={manualPages} />
          <main className="content">{children}</main>
        </div>
      </body>
    </html>
  );
}
