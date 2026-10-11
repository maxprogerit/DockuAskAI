import { useRef, useState } from "react";
import type { DragEvent } from "react";
import { api } from "../api";
import type { DocumentOut } from "../types";
import { HealthBadge } from "./HealthBadge";

interface Props {
  documents: DocumentOut[];
  loadError: string | null;
  selected: Set<string>;
  onToggle: (id: string) => void;
  onChanged: () => Promise<void>;
}

const ACCEPT = ".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document";

export function DocumentsPanel({ documents, loadError, selected, onToggle, onChanged }: Props) {
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
        messages.push(
          doc.already_processed
            ? `${file.name}: already indexed, nothing re-embedded.`
            : `${file.name}: indexed (${doc.chunk_count} chunks).`,
        );
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
    <aside
      className={`sidebar${dragging ? " dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (e.currentTarget === e.target) setDragging(false);
      }}
      onDrop={onDrop}
    >
      <div className="brand">
        Doku <span>Ask AI</span>
      </div>

      <button type="button" className="btn-primary upload" disabled={busy !== null} onClick={() => input.current?.click()}>
        {busy ? "Indexing…" : "Upload document"}
      </button>
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
      {busy && (
        <p className="busy">
          <span className="spinner" /> {busy}
          <small>Parsing and embedding can take a while for long files.</small>
        </p>
      )}
      {notice && <pre className={`notice notice-${notice.kind}`}>{notice.text}</pre>}
      {loadError && <p className="notice notice-error">{loadError}</p>}

      <h2 className="label">Documents</h2>
      {documents.length === 0 && !loadError ? (
        <p className="muted small">No documents yet. Drop a PDF or DOCX anywhere on this panel.</p>
      ) : (
        <ul className="doc-list">
          {documents.map((d) => (
            <li key={d.id}>
              <button
                type="button"
                className={`doc${selected.has(d.id) ? " active" : ""}`}
                aria-pressed={selected.has(d.id)}
                title={`${d.filename} · ${d.file_type.toUpperCase()}${d.page_count ? ` · ${d.page_count} p.` : ""} · ${d.chunk_count} chunks`}
                onClick={() => onToggle(d.id)}
              >
                <span className="doc-name">{d.filename}</span>
                {d.status !== "ready" && <span className="doc-status">{d.status}</span>}
              </button>
              <button
                type="button"
                className="doc-delete"
                aria-label={`Delete ${d.filename}`}
                title="Delete"
                onClick={() => void remove(d)}
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
      {documents.length > 0 && (
        <p className="hint">
          {selected.size === 0 ? "Searching all documents. Click one to narrow down." : "Click again to deselect."}
        </p>
      )}

      <div className="sidebar-foot">
        <HealthBadge />
      </div>
    </aside>
  );
}
