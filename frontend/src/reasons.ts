import type { Reason } from "./types";

// Plain-language versions of the backend's refusal reasons (see app/schemas/ask.py).
export const REASON_TEXT: Record<Reason, { title: string; detail: string }> = {
  no_documents: {
    title: "No documents to search",
    detail: "The library (or the selected subset) is empty. Upload a document first.",
  },
  below_score_threshold: {
    title: "Nothing relevant found",
    detail:
      "Even the best matching passage was not similar enough to the question, so the language model was not called.",
  },
  no_answer_in_documents: {
    title: "The documents do not answer this",
    detail: "The model read the retrieved passages and found no answer in them.",
  },
  citation_validation_failed: {
    title: "Answer withheld: citations could not be verified",
    detail:
      "The model's quotes did not match the retrieved text, even after one retry. Unverified answers are never shown.",
  },
  llm_refused: {
    title: "The model declined to answer",
    detail: "The model returned no usable output for this question.",
  },
};
