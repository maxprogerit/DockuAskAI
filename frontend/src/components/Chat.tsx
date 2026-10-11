import { useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { api } from "../api";
import type { AskResponse, Citation } from "../types";
import { Message } from "./Message";
import { SourceViewer } from "./SourceViewer";
import { SourcesPanel } from "./SourcesPanel";

interface Turn {
  id: number;
  question: string;
  status: "pending" | "done" | "error";
  response?: AskResponse;
  error?: string;
}

interface Props {
  documentCount: number;
  selectedIds: string[];
}

const MAX_QUESTION = 2000; // AskRequest.question max_length

export function Chat({ documentCount, selectedIds }: Props) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [active, setActive] = useState<{ turnId: number; index: number } | null>(null);
  const [viewing, setViewing] = useState<{ citation: Citation; turn: Turn } | null>(null);
  const nextId = useRef(1);
  const bottom = useRef<HTMLDivElement>(null);
  const loading = turns.some((t) => t.status === "pending");

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [turns]);

  async function send(e?: FormEvent) {
    e?.preventDefault();
    const q = question.trim();
    if (!q || loading) return;
    const id = nextId.current++;
    setQuestion("");
    setTurns((prev) => [...prev, { id, question: q, status: "pending" }]);
    try {
      const response = await api.ask(q, selectedIds);
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, status: "done", response } : t)));
      setActive(response.citations.length > 0 ? { turnId: id, index: 0 } : { turnId: id, index: -1 });
    } catch (err) {
      const message = err instanceof Error ? err.message : "Request failed.";
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, status: "error", error: message } : t)));
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  // The Sources panel follows the clicked chip, otherwise the latest answer that has sources (a
  // refusal has none, and should not wipe the evidence of the previous answer).
  const withSources = (t: Turn) => (t.response?.citations.length ?? 0) > 0;
  const sourceTurn =
    turns.find((t) => t.id === active?.turnId && withSources(t)) ??
    [...turns].reverse().find(withSources) ??
    [...turns].reverse().find((t) => t.response);
  const activeIndex = active && sourceTurn && active.turnId === sourceTurn.id ? Math.max(active.index, 0) : 0;

  const scope =
    selectedIds.length === 0
      ? `all ${documentCount} document${documentCount === 1 ? "" : "s"}`
      : `${selectedIds.length} selected document${selectedIds.length === 1 ? "" : "s"}`;

  return (
    <div className="chat-layout">
      <section className="chat panel" aria-label="Conversation">
        <div className="messages" aria-live="polite">
          {turns.length === 0 && (
            <p className="empty">
              Ask something about {documentCount === 0 ? "your documents once you have uploaded one" : scope}.
              <br />
              If the documents do not contain the answer, you get an explicit refusal instead of a guess.
            </p>
          )}
          {turns.map((t) => (
            <div key={t.id} className="turn">
              <div className="bubble user">{t.question}</div>
              <Message
                turn={t}
                activeIndex={active?.turnId === t.id ? active.index : -1}
                onPick={(index) => setActive({ turnId: t.id, index })}
              />
            </div>
          ))}
          <div ref={bottom} />
        </div>

        <form className="composer" onSubmit={send}>
          <label htmlFor="question" className="visually-hidden">
            Your question
          </label>
          <textarea
            id="question"
            value={question}
            maxLength={MAX_QUESTION}
            rows={1}
            placeholder="Ask your documents…"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={onKeyDown}
          />
          <button type="submit" className="btn-primary" disabled={loading || question.trim().length === 0}>
            {loading ? "…" : "Send"}
          </button>
        </form>
      </section>

      <SourcesPanel
        response={sourceTurn?.response}
        question={sourceTurn?.question}
        activeIndex={activeIndex}
        onPick={(index) => sourceTurn && setActive({ turnId: sourceTurn.id, index })}
        onOpen={(citation) => sourceTurn && setViewing({ citation, turn: sourceTurn })}
      />

      {viewing?.turn.response && (
        <SourceViewer
          citation={viewing.citation}
          retrieved={viewing.turn.response.retrieved}
          onClose={() => setViewing(null)}
        />
      )}
    </div>
  );
}
