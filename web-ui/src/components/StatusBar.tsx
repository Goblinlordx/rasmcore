import type { WorkerTimings } from '../hooks/useWorker';

interface Props {
  dims: { width: number; height: number } | null;
  timings: WorkerTimings | null;
  processing: boolean;
  error: string | null;
  /** true when viewport shows proxy (lower-res) result */
  showingProxy?: boolean;
  /** true when background full-res warm is running */
  warming?: boolean;
  /** Proxy render time in ms */
  proxyMs?: number | null;
}

export default function StatusBar({
  dims,
  timings,
  processing,
  error,
  showingProxy,
  warming,
  proxyMs,
}: Props) {
  return (
    <div className="status-bar" id="status-bar">
      <span className="status-dims">{dims ? `${dims.width}\u00d7${dims.height}` : ''}</span>
      <span className="status-time">
        {processing ? 'Processing...' : proxyMs != null ? `${proxyMs}ms` : timings ? `${timings.totalMs}ms` : ''}
      </span>
      <span className="status-ops">{timings ? `${timings.ops.length} ops` : ''}</span>
      {showingProxy != null && dims && (
        <span className="status-proxy-indicator">
          {showingProxy ? (warming ? 'Proxy \u00b7 warming\u2026' : 'Proxy') : 'Full'}
        </span>
      )}
      <span className="status-spacer" />
      <span className="status-error">{error || ''}</span>
    </div>
  );
}
