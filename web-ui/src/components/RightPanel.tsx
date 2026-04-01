import { useState, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

export default function RightPanel({ children }: Props) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <>
      <div className={'right-panel' + (collapsed ? ' collapsed' : '')} id="right-panel">
        <div className="panel-scroll">{children}</div>
      </div>
      <button
        className="panel-toggle"
        style={{ right: collapsed ? 0 : 320 }}
        onClick={() => setCollapsed((c) => !c)}
        dangerouslySetInnerHTML={{ __html: collapsed ? '&#9654;' : '&#9664;' }}
      />
    </>
  );
}
