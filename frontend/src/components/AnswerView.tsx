import { useState } from "react";
import { REASON_TEXT } from "../reasons";
import type { AskResponse, Citation } from "../types";
import { DebugPanel } from "./DebugPanel";
import { SourceViewer } from "./SourceViewer";

function location(c: Citation): string {
  if (c.page) return `p. ${c.page}`;
  const parts: string[] = [];
  if (c.section) parts.push(c.section);
  if (c.para_index !== null) parts.push(`¶ ${c.para_index}`);
  return parts.join(" · ");
}

export function AnswerView({ question, response }: { question: string; response: AskResponse }) {
  const [open, setOpen] = useState<Citation | null>(null);
  const reason = response.reason ? REASON_TEXT[response.reason] : null;

  return (
    <div className="result">
      <p className="asked">{question}</p>

      {response.answerable ? (
        <div className="answer card">
          <p className="answer-text">{response.answer}</p>
        </div>
      ) : (
        <div className="refusal card">
          <h3>{reason?.title ?? "No answer"}</h3>
          <p>{reason?.detail ?? response.answer}</p>
          {/* Without an LLM call (llm_calls = 0) `answer` is only a canned sentence, not a model note. */}
          {response.answer && reason && response.usage.llm_calls > 0 && (
            <p className="muted small">Model note: {response.answer}</p>
          )}
        </div>
      )}

      {response.citations.length > 0 && (
        <div className="citations">
          <h3>Sources</h3>
          <ol>
            {response.citations.map((c, i) => (
              <li key={`${c.chunk_id}-${i}`} className="card">
                <div className="cite-head">
                  <strong>{c.filename}</strong>
                  <span className="muted small">{location(c)}</span>
                </div>
                <blockquote>“{c.quote}”</blockquote>
                <button type="button" className="secondary small" onClick={() => setOpen(c)}>
                  View in source
                </button>
              </li>
            ))}
          </ol>
        </div>
      )}

      <DebugPanel response={response} />

      {open && <SourceViewer citation={open} retrieved={response.retrieved} onClose={() => setOpen(null)} />}
    </div>
  );
}
