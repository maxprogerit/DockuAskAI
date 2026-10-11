import type { AskResponse } from "../types";
import { REASON_TEXT } from "../reasons";
import { citationLabel } from "../util";
import { DebugPanel } from "./DebugPanel";

interface Props {
  turn: { status: "pending" | "done" | "error"; response?: AskResponse; error?: string };
  activeIndex: number;
  onPick: (index: number) => void;
}

/** The assistant side of one turn: thinking indicator, error, answer, or a refusal. */
export function Message({ turn, activeIndex, onPick }: Props) {
  if (turn.status === "pending") {
    return (
      <div className="bubble assistant muted">
        <span className="spinner" /> Retrieving passages and verifying citations…
      </div>
    );
  }
  if (turn.status === "error" || !turn.response) {
    return <div className="bubble assistant error">{turn.error ?? "Request failed."}</div>;
  }

  const { response } = turn;
  const reason = response.reason ? REASON_TEXT[response.reason] : null;

  return (
    <>
      {response.answerable ? (
        <div className="bubble assistant">{response.answer}</div>
      ) : (
        <div className="bubble assistant refusal">
          <strong>{reason?.title ?? "No answer"}</strong>
          <span>{reason?.detail ?? response.answer}</span>
          {/* Without an LLM call (llm_calls = 0) `answer` is only a canned sentence, not a model note. */}
          {response.answer && reason && response.usage.llm_calls > 0 && (
            <small>Model note: {response.answer}</small>
          )}
        </div>
      )}

      {response.citations.length > 0 && (
        <div className="chips" role="group" aria-label="Sources of this answer">
          {response.citations.map((c, i) => (
            <button
              key={`${c.chunk_id}-${i}`}
              type="button"
              className={`chip chip-source${i === activeIndex ? " active" : ""}`}
              aria-pressed={i === activeIndex}
              onClick={() => onPick(i)}
            >
              {citationLabel(c, i)}
            </button>
          ))}
        </div>
      )}

      <DebugPanel response={response} />
    </>
  );
}
