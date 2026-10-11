import type { AskResponse, Citation } from "../types";
import { citationLabel } from "../util";

interface Props {
  response?: AskResponse;
  question?: string; // the question these sources belong to
  activeIndex: number;
  onPick: (index: number) => void;
  onOpen: (citation: Citation) => void;
}

/** Right-hand column: the verified quotes behind the selected answer. */
export function SourcesPanel({ response, question, activeIndex, onPick, onOpen }: Props) {
  return (
    <aside className="sources panel" aria-label="Sources">
      <h2 className="label">Sources</h2>
      {response && question && (
        <p className="muted small for-question" title={question}>
          For: {question}
        </p>
      )}
      {!response && <p className="muted small">Sources appear here once you ask a question.</p>}
      {response && response.citations.length === 0 && (
        <p className="muted small">This answer has no sources: nothing in the documents supports one.</p>
      )}
      {response?.citations.map((c, i) => (
        <article
          key={`${c.chunk_id}-${i}`}
          className={`source${i === activeIndex ? " active" : ""}`}
          onClick={() => onPick(i)}
        >
          <header>{citationLabel(c, i)}</header>
          <p>{c.quote}</p>
          <button
            type="button"
            className="link"
            onClick={(e) => {
              e.stopPropagation();
              onOpen(c);
            }}
          >
            View in source →
          </button>
        </article>
      ))}
    </aside>
  );
}
