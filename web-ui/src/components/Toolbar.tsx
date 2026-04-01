import { useCallback, useEffect, useRef, useState } from 'react';
import type { Operation } from '../types';
import { MENU_MAP, type ManifestGroupMeta } from '../utils/manifest';

interface FormatParams {
  quality: number;
  progressive?: boolean;
  compression?: number;
  lossless?: boolean;
}

interface Props {
  operations: Operation[];
  groups: Record<string, ManifestGroupMeta>;
  writeFormats: string[];
  onAddNode: (op: Operation) => void;
  onDownload: (format: string, quality: number, params: FormatParams) => void;
  onShowCode: () => void;
  exportFormat: string;
  onExportFormatChange: (format: string) => void;
  exportQuality: number;
  onExportQualityChange: (quality: number) => void;
}

interface MenuDef {
  id: string;
  label: string;
}

const MENUS: MenuDef[] = [
  { id: 'filters', label: 'Filters' },
  { id: 'transforms', label: 'Transform' },
  { id: 'effects', label: 'Effects' },
  { id: 'grading', label: 'Grading' },
];

function opLabel(op: Operation, groups: Record<string, ManifestGroupMeta>) {
  const meta = groups[op.name];
  if (meta?.variant) return meta.variant.replace(/_/g, ' ');
  return op.name;
}

export default function Toolbar({
  operations,
  groups,
  writeFormats,
  onAddNode,
  onDownload,
  onShowCode,
  exportFormat,
  onExportFormatChange,
  exportQuality,
  onExportQualityChange,
}: Props) {
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const [progressive, setProgressive] = useState(false);
  const [compression, setCompression] = useState(6);
  const [lossless, setLossless] = useState(false);
  const toolbarRef = useRef<HTMLDivElement>(null);

  // Group operations by menu
  const menuOps: Record<string, Record<string, Operation[]>> = {};
  for (const menu of MENUS) menuOps[menu.id] = {};

  for (const op of operations) {
    const menuId = MENU_MAP[op.category] || 'filters';
    if (!menuOps[menuId]) menuOps[menuId] = {};
    if (!menuOps[menuId][op.category]) menuOps[menuId][op.category] = [];
    menuOps[menuId][op.category].push(op);
  }

  // Close on outside click
  useEffect(() => {
    const handler = () => setOpenMenu(null);
    document.addEventListener('click', handler);
    return () => document.removeEventListener('click', handler);
  }, []);

  const toggleMenu = useCallback((menuId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setOpenMenu((prev) => (prev === menuId ? null : menuId));
  }, []);

  const handleDownload = useCallback(() => {
    onDownload(exportFormat, exportQuality, {
      quality: exportQuality,
      progressive: exportFormat === 'jpeg' ? progressive : undefined,
      compression: exportFormat === 'png' ? compression : undefined,
      lossless: exportFormat === 'webp' ? lossless : undefined,
    });
    setOpenMenu(null);
  }, [exportFormat, exportQuality, progressive, compression, lossless, onDownload]);

  return (
    <div className="toolbar" ref={toolbarRef}>
      <span className="brand">rasmcore</span>
      <span className="badge">Pipeline Builder</span>

      {/* File menu — export + code */}
      <div
        className={'menu-btn' + (openMenu === 'file' ? ' active' : '')}
        style={{ position: 'relative' }}
        onClick={(e) => toggleMenu('file', e)}
      >
        File &#9662;
        <div
          className={'dropdown' + (openMenu === 'file' ? ' open' : '')}
          style={{ minWidth: 240 }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="dropdown-header">Export</div>
          <div style={{ padding: '4px 12px' }}>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 6 }}>
              <select
                value={exportFormat}
                onChange={(e) => onExportFormatChange(e.target.value)}
                style={{
                  fontSize: '0.7rem',
                  padding: '3px 6px',
                  background: '#1a1a2e',
                  color: '#e0e0e0',
                  border: '1px solid #333',
                  borderRadius: 3,
                }}
              >
                {writeFormats.map((f) => (
                  <option key={f} value={f}>
                    {f.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            {/* Per-format params */}
            {exportFormat === 'jpeg' && (
              <>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ fontSize: '0.65rem', color: '#888', width: 50 }}>Quality</span>
                  <input
                    type="range"
                    min={1}
                    max={100}
                    value={exportQuality}
                    style={{ flex: 1, accentColor: '#3b82f6' }}
                    onInput={(e) => onExportQualityChange(parseInt(e.currentTarget.value))}
                  />
                  <span style={{ fontSize: '0.6rem', color: '#888', width: 24 }}>
                    {exportQuality}
                  </span>
                </div>
                <label
                  style={{
                    display: 'flex',
                    gap: 6,
                    alignItems: 'center',
                    fontSize: '0.65rem',
                    color: '#888',
                    marginBottom: 4,
                  }}
                >
                  <input
                    type="checkbox"
                    checked={progressive}
                    onChange={(e) => setProgressive(e.target.checked)}
                  />
                  Progressive
                </label>
              </>
            )}

            {exportFormat === 'png' && (
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: '0.65rem', color: '#888', width: 80 }}>Compression</span>
                <input
                  type="range"
                  min={0}
                  max={9}
                  value={compression}
                  style={{ flex: 1, accentColor: '#3b82f6' }}
                  onInput={(e) => setCompression(parseInt(e.currentTarget.value))}
                />
                <span style={{ fontSize: '0.6rem', color: '#888', width: 16 }}>{compression}</span>
              </div>
            )}

            {exportFormat === 'webp' && (
              <>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4 }}>
                  <span style={{ fontSize: '0.65rem', color: '#888', width: 50 }}>Quality</span>
                  <input
                    type="range"
                    min={1}
                    max={100}
                    value={exportQuality}
                    style={{ flex: 1, accentColor: '#3b82f6' }}
                    onInput={(e) => onExportQualityChange(parseInt(e.currentTarget.value))}
                  />
                  <span style={{ fontSize: '0.6rem', color: '#888', width: 24 }}>
                    {exportQuality}
                  </span>
                </div>
                <label
                  style={{
                    display: 'flex',
                    gap: 6,
                    alignItems: 'center',
                    fontSize: '0.65rem',
                    color: '#888',
                    marginBottom: 4,
                  }}
                >
                  <input
                    type="checkbox"
                    checked={lossless}
                    onChange={(e) => setLossless(e.target.checked)}
                  />
                  Lossless
                </label>
              </>
            )}

            {/* Fallback quality for other formats */}
            {exportFormat !== 'jpeg' && exportFormat !== 'png' && exportFormat !== 'webp' && (
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: '0.65rem', color: '#888', width: 50 }}>Quality</span>
                <input
                  type="range"
                  min={1}
                  max={100}
                  value={exportQuality}
                  style={{ flex: 1, accentColor: '#3b82f6' }}
                  onInput={(e) => onExportQualityChange(parseInt(e.currentTarget.value))}
                />
                <span style={{ fontSize: '0.6rem', color: '#888', width: 24 }}>
                  {exportQuality}
                </span>
              </div>
            )}

            <button className="sm" style={{ width: '100%', marginTop: 4 }} onClick={handleDownload}>
              Download
            </button>
          </div>
          <div className="dropdown-divider" />
          <div
            className="dropdown-item"
            onClick={() => {
              onShowCode();
              setOpenMenu(null);
            }}
          >
            {'{ }'} Get the Code
          </div>
        </div>
      </div>

      {/* Operation menus */}
      {MENUS.map((menu) => (
        <div
          key={menu.id}
          className={'menu-btn' + (openMenu === menu.id ? ' active' : '')}
          style={{ position: 'relative' }}
          onClick={(e) => toggleMenu(menu.id, e)}
        >
          {menu.label} &#9662;
          <div className={'dropdown' + (openMenu === menu.id ? ' open' : '')}>
            {Object.entries(menuOps[menu.id] || {}).map(([cat, ops]) => (
              <div key={cat}>
                <div className="dropdown-header">{cat}</div>
                {ops.map((op) => (
                  <div
                    key={op.name}
                    className="dropdown-item"
                    title={groups[op.name]?.reference || ''}
                    onClick={(e) => {
                      e.stopPropagation();
                      onAddNode(op);
                      setOpenMenu(null);
                    }}
                  >
                    {opLabel(op, groups)}
                  </div>
                ))}
                <div className="dropdown-divider" />
              </div>
            ))}
          </div>
        </div>
      ))}

      <div className="toolbar-spacer" />
    </div>
  );
}
