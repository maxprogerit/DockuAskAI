import type { AskResponse, DocumentOut, HealthResponse, UploadResponse } from "./types";

// nginx (production) and the Vite dev server both forward /api/* to the backend and add the API key.
export const API_BASE = "/api";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId?: string;

  constructor(message: string, status: number, code: string, requestId?: string) {
    super(message);
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

/** The backend always answers errors as {"error": {"code", "message"}, "request_id"}. */
async function toApiError(res: Response): Promise<ApiError> {
  try {
    const body = await res.json();
    const err = body?.error;
    if (err?.message) {
      return new ApiError(err.message, res.status, err.code ?? "http_error", body.request_id);
    }
  } catch {
    // not JSON (e.g. an nginx 413/502 page): fall through
  }
  const fallback: Record<number, string> = {
    413: "The file is too large.",
    502: "The backend is not reachable.",
    504: "The backend took too long to answer.",
  };
  return new ApiError(fallback[res.status] ?? `Request failed (${res.status}).`, res.status, "http_error");
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_BASE + path, init);
  } catch {
    throw new ApiError("Cannot reach the server. Check your connection.", 0, "network_error");
  }
  if (!res.ok) throw await toApiError(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  /** /health answers 503 with a normal HealthResponse body when degraded, so a 503 is not an error here. */
  async health(): Promise<HealthResponse> {
    let res: Response;
    try {
      res = await fetch(API_BASE + "/health");
    } catch {
      throw new ApiError("Cannot reach the server.", 0, "network_error");
    }
    if (res.ok || res.status === 503) {
      try {
        const body = await res.json();
        if (typeof body?.status === "string") return body as HealthResponse;
      } catch {
        // proxy error page: fall through
      }
    }
    throw await toApiError(res);
  },
  listDocuments: () => request<DocumentOut[]>("/documents"),
  uploadDocument(file: File) {
    const form = new FormData();
    form.append("file", file);
    return request<UploadResponse>("/documents", { method: "POST", body: form });
  },
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),
  ask(question: string, documentIds: string[], topK?: number) {
    return request<AskResponse>("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        document_ids: documentIds.length > 0 ? documentIds : null,
        top_k: topK ?? null,
      }),
    });
  },
  /** URL of the original upload; the browser loads it in an <iframe>, nginx adds the key. */
  fileUrl: (id: string, page?: number | null) => `${API_BASE}/documents/${id}/file${page ? `#page=${page}` : ""}`,
};
