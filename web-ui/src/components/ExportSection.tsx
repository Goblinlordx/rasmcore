import { useState } from 'react';

interface Props {
  formats: string[];
  onDownload: (format: string, quality: number) => void;
}

export default function ExportSection({ formats, onDownload }: Props) {
  const [format, setFormat] = useState(formats[0] || 'jpeg');
  const [quality, setQuality] = useState(85);

  return (
    <div className="export-section">
      <div className="export-row">
        <select value={format} onChange={(e) => setFormat(e.target.value)}>
          {formats.map((f) => (
            <option key={f} value={f}>
              {f.toUpperCase()}
            </option>
          ))}
        </select>
        <input
          type="range"
          min={1}
          max={100}
          value={quality}
          title="Quality"
          onInput={(e) => setQuality(parseInt(e.currentTarget.value))}
        />
        <span style={{ fontSize: '0.6rem', color: '#888', width: 24 }}>{quality}</span>
        <button className="sm" onClick={() => onDownload(format, quality)}>
          Download
        </button>
      </div>
    </div>
  );
}

export function useExportState() {
  const [format, setFormat] = useState('jpeg');
  const [quality, setQuality] = useState(85);
  return { format, setFormat, quality, setQuality };
}
