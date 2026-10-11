import { useState } from "react";
import type { FormEvent } from "react";
import { api } from "../api";
import type { AskResponse, DocumentOut } from "../types";
import { AnswerView } from "./AnswerView";

interface Props {
  documents: DocumentOut[];
  selectedIds: string[];
}

const MAX_QUESTION = 2000; // AskRequest.question max_length

export function AskPanel({ documents, selectedIds }: Props) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ question: string; response: AskResponse } | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || loading) return;
    setLoading(true);
    setError(null);
    try {
      const response = await api.ask(q, selectedIds);
      setResult({ question: q, response });
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Request failed.");
    } finally {
      setLoading(false);
    }
  }

  const scope =
    selectedIds.length === 0
      ? `all ${documents.length} document${documents.length === 1 ? "" : "s"}`
      : `${selectedIds.length} selected document${selectedIds.length === 1 ? "" : "s"}`;

  return (
    <section className="panel ask">
      <form onSubmit={submit}>
        <label htmlFor="question" className="visually-hidden">
          Your question
        </label>
        <textarea
          id="question"
          value={question}
          maxLength={MAX_QUESTION}
          rows={3}
          placeholder="Ask a question about your documents…"
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) void submit(e);
          }}
        />
        <div className="form-row">
          <span className="muted small">Searching {scope} · Ctrl+Enter to send</span>
          <button type="submit" disabled={loading || question.trim().length === 0}>
            {loading ? "Thinking…" : "Ask"}
          </button>
        </div>
      </form>

      {loading && (
        <p className="muted">
          <span className="spinner" /> Retrieving passages and verifying citations…
        </p>
      )}
      {error && <p className="notice notice-error">{error}</p>}
      {result && !loading && <AnswerView question={result.question} response={result.response} />}
      {!result && !error && !loading && (
        <p className="muted empty">
          Answers come only from the uploaded documents. If they do not contain the answer, you will get an explicit
          refusal instead of a guess.
        </p>
      )}
    </section>
  );
}
