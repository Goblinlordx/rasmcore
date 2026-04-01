import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

// Mock the Worker constructor since jsdom doesn't support Web Workers
vi.stubGlobal(
  'Worker',
  class {
    postMessage = vi.fn();
    terminate = vi.fn();
    onmessage: ((e: MessageEvent) => void) | null = null;
    addEventListener = vi.fn();
    removeEventListener = vi.fn();
  },
);

// Mock dynamic imports for SDK
// Dynamic SDK imports are handled by the context — no mock needed for component smoke tests

import ParamControl from '../ParamControl';
import StatusBar from '../StatusBar';
import CodeModal from '../CodeModal';
import ExportSection from '../ExportSection';
import RightPanel from '../RightPanel';
import Canvas from '../Canvas';
import LayerPanel from '../LayerPanel';
import EffectStack from '../EffectStack';
import ChainNode from '../ChainNode';
import Toolbar from '../Toolbar';

describe('Component smoke tests', () => {
  it('ParamControl renders a number slider', () => {
    render(
      <ParamControl
        param={{
          name: 'amount',
          type: 'number',
          hint: '',
          min: 0,
          max: 100,
          step: 1,
          default: 50,
          label: 'Amount',
        }}
        value={50}
        onChange={() => {}}
      />,
    );
    expect(screen.getByText('Amount')).toBeInTheDocument();
  });

  it('ParamControl renders a toggle', () => {
    render(
      <ParamControl
        param={{ name: 'enabled', type: 'toggle', hint: '', default: false, label: 'Enabled' }}
        value={false}
        onChange={() => {}}
      />,
    );
    expect(screen.getByText('Enabled')).toBeInTheDocument();
  });

  it('ParamControl renders an enum select', () => {
    render(
      <ParamControl
        param={{
          name: 'mode',
          type: 'enum',
          hint: '',
          options: ['a', 'b'],
          default: 'a',
          label: 'Mode',
        }}
        value="a"
        onChange={() => {}}
      />,
    );
    expect(screen.getByText('Mode')).toBeInTheDocument();
  });

  it('StatusBar renders dimensions and timing', () => {
    render(
      <StatusBar
        dims={{ width: 800, height: 600 }}
        timings={{ totalMs: 42, ops: [{ name: 'blur', ms: 42 }] }}
        processing={false}
        error={null}
      />,
    );
    expect(screen.getByText('800\u00d7600')).toBeInTheDocument();
    expect(screen.getByText('42ms')).toBeInTheDocument();
    expect(screen.getByText('1 ops')).toBeInTheDocument();
  });

  it('StatusBar shows processing state', () => {
    render(<StatusBar dims={null} timings={null} processing={true} error={null} />);
    expect(screen.getByText('Processing...')).toBeInTheDocument();
  });

  it('StatusBar shows error', () => {
    render(<StatusBar dims={null} timings={null} processing={false} error="Something broke" />);
    expect(screen.getByText('Something broke')).toBeInTheDocument();
  });

  it('CodeModal renders when open', () => {
    render(<CodeModal open={true} code="const x = 1;" onClose={() => {}} />);
    expect(screen.getByText('SDK Code')).toBeInTheDocument();
    expect(screen.getByText('Copy to Clipboard')).toBeInTheDocument();
  });

  it('CodeModal does not render when closed', () => {
    const { container } = render(<CodeModal open={false} code="" onClose={() => {}} />);
    expect(container.innerHTML).toBe('');
  });

  it('ExportSection renders format and download', () => {
    render(<ExportSection formats={['jpeg', 'png']} onDownload={() => {}} />);
    expect(screen.getByText('Download')).toBeInTheDocument();
  });

  it('RightPanel renders children and toggle', () => {
    render(
      <RightPanel>
        <div>Panel Content</div>
      </RightPanel>,
    );
    expect(screen.getByText('Panel Content')).toBeInTheDocument();
  });

  it('Canvas renders drop zone when no image', () => {
    const ref1 = { current: null };
    const ref2 = { current: null };
    render(
      <Canvas
        previewCanvasRef={ref1}
        originalCanvasRef={ref2}
        hasImage={false}
        onAddLayer={() => {}}
      />,
    );
    expect(screen.getByText(/Drop an image/)).toBeInTheDocument();
  });

  it('Canvas renders tabs when image is loaded', () => {
    const ref1 = { current: null };
    const ref2 = { current: null };
    render(
      <Canvas
        previewCanvasRef={ref1}
        originalCanvasRef={ref2}
        hasImage={true}
        onAddLayer={() => {}}
      />,
    );
    expect(screen.getByText('Current')).toBeInTheDocument();
    expect(screen.getByText('Original')).toBeInTheDocument();
    expect(screen.getByText('Split')).toBeInTheDocument();
  });

  it('LayerPanel renders add layer button', () => {
    render(
      <LayerPanel
        layers={[]}
        activeLayerId={null}
        onSelectLayer={() => {}}
        onToggleVisibility={() => {}}
        onRemoveLayer={() => {}}
        onUpdateLayer={() => {}}
        onAddLayer={() => {}}
        onRequestComposite={() => {}}
      />,
    );
    expect(screen.getByText('+ Add Layer')).toBeInTheDocument();
  });

  it('EffectStack renders empty hint', () => {
    render(
      <EffectStack
        chain={[]}
        editingNodeId={null}
        activeLayerName=""
        onSetEditing={() => {}}
        onRemoveNode={() => {}}
        onMoveNode={() => {}}
        onApplyNode={() => {}}
        onParamChange={() => {}}
        onSchedulePreview={() => {}}
        onApplyFullChain={() => {}}
      />,
    );
    expect(screen.getByText('Select an operation from the toolbar')).toBeInTheDocument();
  });

  it('ChainNode renders node name', () => {
    const node = {
      id: 1,
      op: { name: 'blur', category: 'Filters', params: [] },
      paramValues: {},
      applied: true,
      timingMs: 0,
    };
    render(
      <ChainNode
        node={node}
        index={0}
        isEditing={false}
        onEdit={() => {}}
        onRemove={() => {}}
        onApply={() => {}}
        onCancelEdit={() => {}}
        onParamChange={() => {}}
        onDragStart={() => {}}
        onDragEnd={() => {}}
        onDragOver={() => {}}
        onDrop={() => {}}
        onSchedulePreview={() => {}}
      />,
    );
    expect(screen.getByText(/blur/)).toBeInTheDocument();
  });

  it('Toolbar renders brand and menus', () => {
    render(<Toolbar operations={[]} groups={{}} onAddNode={() => {}} onShowCode={() => {}} />);
    expect(screen.getByText('rasmcore')).toBeInTheDocument();
    expect(screen.getByText('Pipeline Builder')).toBeInTheDocument();
  });
});
