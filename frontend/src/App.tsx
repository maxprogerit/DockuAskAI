import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { AskPanel } from "./components/AskPanel";
import { DocumentsPanel } from "./components/DocumentsPanel";
import { HealthBadge } from "./components/HealthBadge";
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

  return (
    <div className="app">
      <header className="topbar">
        <h1>DocuAsk</h1>
        <p className="tagline">Answers grounded only in your documents, with verified citations.</p>
        <HealthBadge />
      </header>
      <main className="layout">
        <DocumentsPanel
          documents={documents}
          loadError={loadError}
          selected={selected}
          onToggle={toggle}
          onClearSelection={() => setSelected(new Set())}
          onChanged={refresh}
        />
        <AskPanel documents={documents} selectedIds={[...selected]} />
      </main>
    </div>
  );
}
