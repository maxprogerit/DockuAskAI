// Mirrors app/schemas/*.py. Kept in sync by hand (the API is small; /openapi.json is the source of truth).

export interface DocumentOut {
  id: string;
  filename: string;
  file_type: "pdf" | "docx";
  page_count: number | null;
  chunk_count: number;
  status: string;
  created_at: string;
}

export interface UploadResponse extends DocumentOut {
  already_processed: boolean;
}

export interface Citation {
  document_id: string;
  filename: string;
  page: number | null;
  section: string | null;
  para_index: number | null;
  chunk_id: string;
  quote: string;
}

export interface RetrievedItem {
  chunk_id: string;
  score: number;
  document_id: string;
  filename: string;
  page: number | null;
  section: string | null;
  text: string;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  embedding_tokens: number;
  llm_calls: number;
  model: string | null;
}

export type Reason =
  | "no_documents"
  | "below_score_threshold"
  | "no_answer_in_documents"
  | "citation_validation_failed"
  | "llm_refused";

export interface AskResponse {
  answer: string;
  answerable: boolean;
  reason: Reason | null;
  citations: Citation[];
  retrieved: RetrievedItem[];
  usage: Usage;
}

export interface HealthResponse {
  status: string;
  database: string;
  vector_store: string;
}
