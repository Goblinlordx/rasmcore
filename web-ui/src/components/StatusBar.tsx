import type { WorkerTimings } from '../hooks/useWorker';

interface Props {
  dims: { width: number; height: number } | null;
  timings: WorkerTimings | null;
  processing: boolean;
  error: string | null;
}

export default function StatusBar({ dims, timings, processing, error }: Props) {
  return (
    <div className="status-bar" id="status-bar">
      <span className="status-dims">{dims ? `${dims.width}\u00d7${dims.height}` : ''}</span>
      <span className="status-time">
        {processing ? 'Processing...' : timings ? `${timings.totalMs}ms` : ''}
      </span>
      <span className="status-ops">{timings ? `${timings.ops.length} ops` : ''}</span>
      <span className="status-spacer" />
      <span className="status-error">{error || ''}</span>
    </div>
  );
}
