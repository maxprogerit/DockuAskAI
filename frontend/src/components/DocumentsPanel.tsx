import { useRef, useState } from "react";
import type { DragEvent } from "react";
import { api } from "../api";
import type { DocumentOut } from "../types";

interface Props {
  documents: DocumentOut[];
  loadError: string | null;
  selected: Set<string>;
  onToggle: (id: string) => void;
  onClearSelection: () => void;
  onChanged: () => Promise<void>;
}

const ACCEPT = ".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document";

export function DocumentsPanel({ documents, loadError, selected, onToggle, onClearSelection, onChanged }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState<string | null>(null); // filename being indexed
  const [notice, setNotice] = useState<{ kind: "ok" | "error"; text: string } | null>(null);

  // Sequential on purpose: ingestion is synchronous on the server and embeds in batches.
  async function upload(files: FileList | File[]) {
    setNotice(null);
    const messages: string[] = [];
    let failed = false;
    for (const file of Array.from(files)) {
      setBusy(file.name);
      try {
        const doc = await api.uploadDocument(file);
        messages.push(doc.already_processed ? `${file.name}: already indexed, nothing re-embedded.` : `${file.name}: indexed (${doc.chunk_count} chunks).`);
      } catch (e) {
        failed = true;
        messages.push(`${file.name}: ${e instanceof Error ? e.message : "upload failed."}`);
      }
    }
    setBusy(null);
    setNotice({ kind: failed ? "error" : "ok", text: messages.join("\n") });
    await onChanged();
  }

  async function remove(doc: DocumentOut) {
    if (!window.confirm(`Delete "${doc.filename}"? Its vectors and stored file are removed too.`)) return;
    try {
      await api.deleteDocument(doc.id);
      setNotice(null);
    } catch (e) {
      setNotice({ kind: "error", text: e instanceof Error ? e.message : "Delete failed." });
    }
    await onChanged();
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    if (busy === null && e.dataTransfer.files.length > 0) void upload(e.dataTransfer.files);
  }

  return (
    <aside className="panel documents">
      <h2>Documents</h2>

      <div
        className={`dropzone${dragging ? " dragging" : ""}${busy ? " disabled" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        {busy ? (
          <p>
            <span className="spinner" /> Indexing <strong>{busy}</strong>…
            <br />
            <small>Parsing and embedding can take a while for long files.</small>
          </p>
        ) : (
          <>
            <p>Drop PDF or DOCX here</p>
            <button type="button" className="secondary" onClick={() => input.current?.click()}>
              Choose files
            </button>
          </>
        )}
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) void upload(e.target.files);
            e.target.value = ""; // allow re-selecting the same file
          }}
        />
      </div>

      {notice && <pre className={`notice notice-${notice.kind}`}>{notice.text}</pre>}
      {loadError && <p className="notice notice-error">{loadError}</p>}

      {documents.length === 0 && !loadError ? (
        <p className="muted">No documents yet.</p>
      ) : (
        <>
          <div className="list-head">
            <span className="muted">
              {selected.size === 0 ? "Searching all documents" : `Searching ${selected.size} selected`}
            </span>
            {selected.size > 0 && (
              <button type="button" className="link" onClick={onClearSelection}>
                Clear
              </button>
            )}
          </div>
          <ul className="doc-list">
            {documents.map((d) => (
              <li key={d.id}>
                <label>
                  <input type="checkbox" checked={selected.has(d.id)} onChange={() => onToggle(d.id)} />
                  <span className="doc-main">
                    <span className="doc-name" title={d.filename}>
                      {d.filename}
                    </span>
                    <span className="muted small">
                      {d.file_type.toUpperCase()}
                      {d.page_count ? ` · ${d.page_count} p.` : ""} · {d.chunk_count} chunks
                      {d.status !== "ready" ? ` · ${d.status}` : ""}
                    </span>
                  </span>
                </label>
                <button type="button" className="icon" aria-label={`Delete ${d.filename}`} title="Delete" onClick={() => void remove(d)}>
                  ✕
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </aside>
  );
}
