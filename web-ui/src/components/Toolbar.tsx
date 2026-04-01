import { useCallback, useEffect, useRef, useState } from 'react';
import type { Operation } from '../types';
import { MENU_MAP, type ManifestGroupMeta } from '../utils/manifest';

interface Props {
  operations: Operation[];
  groups: Record<string, ManifestGroupMeta>;
  onAddNode: (op: Operation) => void;
  onShowCode: () => void;
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

export default function Toolbar({ operations, groups, onAddNode, onShowCode }: Props) {
  const [openMenu, setOpenMenu] = useState<string | null>(null);
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

  return (
    <div className="toolbar" ref={toolbarRef}>
      <span className="brand">rasmcore</span>
      <span className="badge">Pipeline Builder</span>

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
      <button className="secondary" onClick={onShowCode}>
        {'{ }'} Code
      </button>
    </div>
  );
}
