from __future__ import annotations


class DocuAskError(Exception):
    """Base for errors that map to a clean HTTP response (never a raw stack trace)."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class FileTooLargeError(DocuAskError):
    status_code = 413
    code = "file_too_large"


class UnsupportedFileTypeError(DocuAskError):
    status_code = 422
    code = "unsupported_file_type"


class UnreadableDocumentError(DocuAskError):
    status_code = 422
    code = "unreadable_document"


class NoExtractableTextError(DocuAskError):
    status_code = 422
    code = "no_extractable_text"


class TooManyPagesError(DocuAskError):
    status_code = 422
    code = "too_many_pages"


class DocumentNotFoundError(DocuAskError):
    status_code = 404
    code = "document_not_found"


class ServiceNotConfiguredError(DocuAskError):
    status_code = 503
    code = "service_not_configured"


class UpstreamError(DocuAskError):
    """OpenAI (embeddings or LLM) failed after retries."""

    status_code = 502
    code = "upstream_error"
