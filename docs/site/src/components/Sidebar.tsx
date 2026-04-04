'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useRef } from 'react';
import type { Registry, ManualPage } from '@/lib/types';

function snakeToTitle(s: string) {
  return s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function snakeToKebab(s: string) {
  return s.replace(/_/g, '-');
}

interface SidebarProps {
  registry: Registry;
  manualPages: ManualPage[];
}

export function Sidebar({ registry, manualPages }: SidebarProps) {
  const pathname = usePathname();
  const activeRef = useRef<HTMLAnchorElement>(null);

  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: 'center' });
  }, [pathname]);

  // Group filters by category
  const categories: Record<string, typeof registry.filters> = {};
  for (const f of registry.filters) {
    const cat = f.category || 'uncategorized';
    if (!categories[cat]) categories[cat] = [];
    categories[cat].push(f);
  }

  // Group manual pages by section
  const sections: Record<string, ManualPage[]> = {};
  for (const p of manualPages) {
    if (!sections[p.section]) sections[p.section] = [];
    sections[p.section].push(p);
  }

  const isActive = (href: string) => pathname === href || pathname === href + '/';

  return (
    <nav style={{
      width: 280, minWidth: 280, background: 'var(--bg-sidebar)',
      borderRight: '1px solid var(--border)', padding: '1rem 0',
      overflowY: 'auto', position: 'sticky', top: 0, height: '100vh',
    }}>
      <Link href="/" style={{
        display: 'block', padding: '0.5rem 1rem', fontWeight: 700,
        fontSize: '1.1rem', color: 'var(--heading)', textDecoration: 'none',
      }}>rasmcore</Link>

      {Object.keys(categories).sort().map(cat => (
        <div key={cat} style={{ marginBottom: '0.5rem' }}>
          <div style={{
            padding: '0.3rem 1rem', fontSize: '0.75rem', fontWeight: 600,
            textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)',
          }}>{snakeToTitle(cat)}</div>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {categories[cat].sort((a, b) => a.name.localeCompare(b.name)).map(op => {
              const href = `/operations/${snakeToKebab(op.name)}`;
              const active = isActive(href);
              return (
                <li key={op.name}>
                  <Link href={href}
                    ref={active ? activeRef : undefined}
                    style={{
                      display: 'block', padding: '0.2rem 1rem 0.2rem 1.5rem',
                      color: active ? 'var(--link)' : 'var(--text)',
                      fontSize: '0.85rem', textDecoration: 'none',
                      fontWeight: active ? 600 : 400,
                      borderLeft: active ? '2px solid var(--link)' : '2px solid transparent',
                      background: active ? 'rgba(88,166,255,0.08)' : 'transparent',
                    }}>{op.displayName || snakeToTitle(op.name)}</Link>
                </li>
              );
            })}
          </ul>
        </div>
      ))}

      {registry.encoders.length > 0 && (
        <div style={{ marginBottom: '0.5rem' }}>
          <div style={{ padding: '0.3rem 1rem', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>Encoders</div>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {registry.encoders.map(e => {
              const href = `/codecs/${snakeToKebab(e.name)}-encoder`;
              const active = isActive(href);
              return (
                <li key={e.name}>
                  <Link href={href} ref={active ? activeRef : undefined}
                    style={{ display: 'block', padding: '0.2rem 1rem 0.2rem 1.5rem', color: active ? 'var(--link)' : 'var(--text)', fontSize: '0.85rem', textDecoration: 'none', fontWeight: active ? 600 : 400, borderLeft: active ? '2px solid var(--link)' : '2px solid transparent', background: active ? 'rgba(88,166,255,0.08)' : 'transparent' }}>{e.displayName}</Link>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {Object.keys(sections).sort().map(section => (
        <div key={section} style={{ marginBottom: '0.5rem' }}>
          <div style={{ padding: '0.3rem 1rem', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)' }}>{section}</div>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {sections[section].sort((a, b) => a.order - b.order).map(p => {
              const href = `/pages/${p.slug}`;
              const active = isActive(href);
              return (
                <li key={p.slug}>
                  <Link href={href} ref={active ? activeRef : undefined}
                    style={{ display: 'block', padding: '0.2rem 1rem 0.2rem 1.5rem', color: active ? 'var(--link)' : 'var(--text)', fontSize: '0.85rem', textDecoration: 'none', fontWeight: active ? 600 : 400, borderLeft: active ? '2px solid var(--link)' : '2px solid transparent', background: active ? 'rgba(88,166,255,0.08)' : 'transparent' }}>{p.title}</Link>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </nav>
  );
}
