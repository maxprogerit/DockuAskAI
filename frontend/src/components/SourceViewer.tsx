import { useEffect } from "react";
import { api } from "../api";
import { splitByQuote } from "../highlight";
import type { Citation, RetrievedItem } from "../types";

interface Props {
  citation: Citation;
  retrieved: RetrievedItem[];
  onClose: () => void;
}

/**
 * Side drawer showing where an answer came from.
 * - the retrieved chunk with the verified quote highlighted (works for PDF and DOCX);
 * - for PDFs, the original file opened at the cited page via the browser's built-in viewer.
 *   DOCX cannot be rendered by a browser, so it gets a download link instead.
 */
export function SourceViewer({ citation, retrieved, onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const chunk = retrieved.find((r) => r.chunk_id === citation.chunk_id);
  const isPdf = citation.filename.toLowerCase().endsWith(".pdf");

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" role="dialog" aria-label="Source" onClick={(e) => e.stopPropagation()}>
        <header>
          <div>
            <h3>{citation.filename}</h3>
            <p className="muted small">
              {citation.page ? `Page ${citation.page}` : citation.section ?? "Word document (no page numbers)"}
              {citation.para_index !== null ? ` · paragraph ${citation.para_index}` : ""}
            </p>
          </div>
          <button type="button" className="icon" aria-label="Close" onClick={onClose}>
            ✕
          </button>
        </header>

        <h4>Passage</h4>
        {chunk ? (
          <p className="chunk-text passage">
            {splitByQuote(chunk.text, citation.quote).map((s, i) =>
              s.match ? <mark key={i}>{s.text}</mark> : <span key={i}>{s.text}</span>,
            )}
          </p>
        ) : (
          <blockquote>“{citation.quote}”</blockquote>
        )}

        {isPdf ? (
          <>
            <h4>Original document</h4>
            <iframe title="Original PDF" className="pdf" src={api.fileUrl(citation.document_id, citation.page)} />
          </>
        ) : (
          <a className="button-link" href={api.fileUrl(citation.document_id)} download={citation.filename}>
            Download the original .docx
          </a>
        )}
      </aside>
    </div>
  );
}
