import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { Chat } from "./components/Chat";
import { DocumentsPanel } from "./components/DocumentsPanel";
import type { DocumentOut } from "./types";

export function App() {
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const refresh = useCallback(async () => {
    try {
      const docs = await api.listDocuments();
      setDocuments(docs);
      setLoadError(null);
      // Drop selections of documents that no longer exist (deleted in another tab, etc.).
      setSelected((prev) => new Set([...prev].filter((id) => docs.some((d) => d.id === id))));
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "Could not load documents.");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const selectedDocs = documents.filter((d) => selected.has(d.id));
  const scopeChips =
    selectedDocs.length === 0
      ? [documents.length === 0 ? "No documents" : `All documents (${documents.length})`]
      : selectedDocs.length <= 2
        ? selectedDocs.map((d) => d.filename)
        : [selectedDocs[0].filename, `+${selectedDocs.length - 1} more`];

  return (
    <div className="app">
      <DocumentsPanel
        documents={documents}
        loadError={loadError}
        selected={selected}
        onToggle={toggle}
        onChanged={refresh}
      />
      <main className="main">
        <header className="page-head">
          <div>
            <h1>Ask your documents</h1>
            <p className="subtitle">
              Answers come only from {selected.size === 0 ? "your documents" : "the selected documents"}, with sources.
            </p>
          </div>
          <div className="scope">
            {scopeChips.map((c) => (
              <span key={c} className="chip chip-outline" title={c}>
                {c}
              </span>
            ))}
          </div>
        </header>
        <Chat documentCount={documents.length} selectedIds={[...selected]} />
      </main>
    </div>
  );
}
