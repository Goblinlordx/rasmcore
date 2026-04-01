import { highlightCode } from '../utils/codeGeneration';

interface Props {
  open: boolean;
  code: string;
  onClose: () => void;
}

export default function CodeModal({ open, code, onClose }: Props) {
  if (!open) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      const btn = document.getElementById('copy-code-btn');
      if (btn) {
        btn.textContent = 'Copied!';
        setTimeout(() => {
          btn.textContent = 'Copy to Clipboard';
        }, 1500);
      }
    });
  };

  return (
    <div className="modal-overlay open" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal">
        <h3>SDK Code</h3>
        <pre dangerouslySetInnerHTML={{ __html: highlightCode(code) }} />
        <div className="modal-actions">
          <button id="copy-code-btn" onClick={handleCopy}>
            Copy to Clipboard
          </button>
          <button style={{ background: '#333' }} onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
