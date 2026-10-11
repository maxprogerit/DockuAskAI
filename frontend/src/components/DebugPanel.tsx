import type { AskResponse } from "../types";

/** Collapsible "what did the system actually do" view: usage plus every retrieved chunk with its score. */
export function DebugPanel({ response }: { response: AskResponse }) {
  const { usage, retrieved, citations } = response;
  const cited = new Set(citations.map((c) => c.chunk_id));

  return (
    <details className="debug">
      <summary>Debug: retrieval and usage</summary>

      <dl className="usage">
        <div>
          <dt>Latency</dt>
          <dd>{usage.latency_ms} ms</dd>
        </div>
        <div>
          <dt>LLM calls</dt>
          <dd>
            {usage.llm_calls}
            {usage.llm_calls === 0 ? " (refused before the model)" : usage.llm_calls > 1 ? " (citation retry)" : ""}
          </dd>
        </div>
        <div>
          <dt>Tokens</dt>
          <dd>
            {usage.prompt_tokens} in / {usage.completion_tokens} out
          </dd>
        </div>
        <div>
          <dt>Embedding tokens</dt>
          <dd>{usage.embedding_tokens}</dd>
        </div>
        <div>
          <dt>Model</dt>
          <dd>{usage.model ?? "–"}</dd>
        </div>
      </dl>

      <h4>Retrieved chunks ({retrieved.length})</h4>
      {retrieved.length === 0 && <p className="muted small">Nothing was retrieved.</p>}
      <ol className="chunks">
        {retrieved.map((r) => (
          <li key={r.chunk_id} className={cited.has(r.chunk_id) ? "cited" : undefined}>
            <div className="chunk-head">
              <span className="score" title="Cosine similarity">
                {r.score.toFixed(3)}
              </span>
              <span className="small">
                {r.filename}
                {r.page ? ` · p. ${r.page}` : r.section ? ` · ${r.section}` : ""}
              </span>
              {cited.has(r.chunk_id) && <span className="tag">cited</span>}
            </div>
            <p className="chunk-text">{r.text}</p>
          </li>
        ))}
      </ol>
    </details>
  );
}
