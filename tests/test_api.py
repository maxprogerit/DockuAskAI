from __future__ import annotations

import io
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
import httpx
import openai
import pytest
from sqlalchemy import select

import app.main as main_module
from app.models.document import Document
from app.models.query import Query
from app.schemas.llm import LLMAnswer, LLMCitation
from tests.conftest import Env
from tests.fakes import FakeEmbeddingsClient

SAMPLES = Path(__file__).parent / "samples"
HANDBOOK = (SAMPLES / "employee_handbook.pdf").read_bytes()
AGREEMENT = (SAMPLES / "cloud_service_agreement.docx").read_bytes()
PDF = "application/pdf"


def upload(env: Env, name: str, data: bytes, content_type: str = PDF, headers: Optional[Dict[str, str]] = None) -> httpx.Response:
    return env.client.post(
        "/documents", files={"file": (name, io.BytesIO(data), content_type)}, headers=env.headers if headers is None else headers
    )


def stored_files(env: Env) -> List[Path]:
    return [p for p in env.settings.upload_dir.glob("*") if p.is_file()] if env.settings.upload_dir.exists() else []


def error_code(response: httpx.Response) -> str:
    body = response.json()
    assert set(body) == {"error", "request_id"} and body["request_id"]
    return str(body["error"]["code"])


# ---------------------------------------------------------------- auth


def test_endpoints_require_api_key(env: Env) -> None:
    for method, url in [("get", "/documents"), ("post", "/ask"), ("get", "/documents/%s" % uuid.uuid4())]:
        response = getattr(env.client, method)(url)
        assert response.status_code == 401 and error_code(response) == "unauthorized"
    wrong = env.client.get("/documents", headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401
    assert env.client.get("/documents", headers=env.headers).status_code == 200


def test_docs_page_is_available(env: Env) -> None:
    assert env.client.get("/docs").status_code == 200
    assert env.client.get("/openapi.json").status_code == 200


# ---------------------------------------------------------------- upload


def test_upload_pdf_indexes_chunks_and_stores_original(env: Env) -> None:
    response = upload(env, "employee_handbook.pdf", HANDBOOK)
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "employee_handbook.pdf" and body["file_type"] == "pdf"
    assert body["page_count"] == 6 and body["chunk_count"] > 0
    assert body["status"] == "ready" and body["already_processed"] is False
    assert env.store.count() == body["chunk_count"]
    assert [p.name for p in stored_files(env)] == ["%s.pdf" % body["id"]]  # server-generated name
    assert sum(len(call) for call in env.embeddings.calls) == body["chunk_count"]


def test_upload_docx_has_no_page_count(env: Env) -> None:
    body = upload(env, "cloud_service_agreement.docx", AGREEMENT, "application/octet-stream").json()
    assert body["file_type"] == "docx" and body["page_count"] is None and body["chunk_count"] > 0


def test_duplicate_upload_returns_existing_document_without_embedding_again(env: Env) -> None:
    first = upload(env, "employee_handbook.pdf", HANDBOOK).json()
    calls_after_first = list(env.embeddings.calls)
    count_after_first = env.store.count()

    second = upload(env, "renamed_copy.pdf", HANDBOOK)  # same bytes, different name
    assert second.status_code == 200
    body = second.json()
    assert body["id"] == first["id"] and body["filename"] == "employee_handbook.pdf"
    assert body["already_processed"] is True
    assert env.embeddings.calls == calls_after_first  # embeddings API NOT called again
    assert env.store.count() == count_after_first
    assert len(env.client.get("/documents", headers=env.headers).json()) == 1
    assert len(stored_files(env)) == 1


def test_unsupported_and_spoofed_files_are_rejected_with_422(env: Env) -> None:
    for name, data in [("notes.txt", b"plain text"), ("fake.pdf", b"<html>nope</html>"), ("renamed.docx", HANDBOOK)]:
        response = upload(env, name, data)
        assert response.status_code == 422 and error_code(response) == "unsupported_file_type", name
    assert stored_files(env) == [] and env.embeddings.calls == []


def test_file_over_limit_returns_413(env: Env) -> None:
    response = upload(env, "big.pdf", b"%PDF-1.7\n" + b"x" * (1024 * 1024))  # limit is 1 MB in tests
    assert response.status_code == 413 and error_code(response) == "file_too_large"


def test_content_length_precheck_returns_413_before_parsing(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "get_settings", lambda: env.settings)
    response = upload(env, "huge.pdf", b"%PDF-1.7\n" + b"x" * (2 * 1024 * 1024))
    assert response.status_code == 413 and error_code(response) == "file_too_large"


def test_scanned_pdf_without_text_returns_422_and_leaves_no_trace(env: Env) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(fitz.Rect(10, 10, 200, 200), pixmap=fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False))
    response = upload(env, "scan.pdf", doc.tobytes())
    assert response.status_code == 422 and error_code(response) == "no_extractable_text"
    assert env.client.get("/documents", headers=env.headers).json() == []
    assert stored_files(env) == [] and env.store.count() == 0


def test_corrupt_pdf_returns_422(env: Env) -> None:
    response = upload(env, "broken.pdf", b"%PDF-1.7\ngarbage that is not a pdf")
    assert response.status_code == 422 and error_code(response) == "unreadable_document"


def test_embeddings_outage_returns_502_and_rolls_everything_back(env: Env) -> None:
    class Down(FakeEmbeddingsClient):
        def _create(self, **_: Any) -> Any:  # type: ignore[override]
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))

    from app.api.deps import get_embedder
    from app.main import app
    from app.services.embeddings import Embedder

    down = Down()
    down.embeddings.create = down._create  # type: ignore[assignment]
    app.dependency_overrides[get_embedder] = lambda: Embedder(down, "m", 8)

    response = upload(env, "employee_handbook.pdf", HANDBOOK)
    assert response.status_code == 502 and error_code(response) == "upstream_error"
    assert env.client.get("/documents", headers=env.headers).json() == []
    assert stored_files(env) == [] and env.store.count() == 0
    # ...and a retry after recovery is not blocked by a leftover row with the same hash
    del app.dependency_overrides[get_embedder]


# ---------------------------------------------------------------- read / file / delete


def test_list_get_and_404(env: Env) -> None:
    created = upload(env, "employee_handbook.pdf", HANDBOOK).json()
    upload(env, "cloud_service_agreement.docx", AGREEMENT)
    listing = env.client.get("/documents", headers=env.headers).json()
    assert len(listing) == 2 and {d["filename"] for d in listing} == {"employee_handbook.pdf", "cloud_service_agreement.docx"}
    assert env.client.get("/documents/%s" % created["id"], headers=env.headers).json()["page_count"] == 6
    missing = env.client.get("/documents/%s" % uuid.uuid4(), headers=env.headers)
    assert missing.status_code == 404 and error_code(missing) == "document_not_found"
    assert env.client.get("/documents/not-a-uuid", headers=env.headers).status_code == 422


def test_file_endpoint_serves_original_inline_with_correct_type(env: Env) -> None:
    pdf_id = upload(env, "employee_handbook.pdf", HANDBOOK).json()["id"]
    docx_id = upload(env, "cloud_service_agreement.docx", AGREEMENT).json()["id"]
    pdf = env.client.get("/documents/%s/file" % pdf_id, headers=env.headers)
    assert pdf.status_code == 200 and pdf.content == HANDBOOK
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.headers["content-disposition"].startswith("inline") and pdf.headers["x-content-type-options"] == "nosniff"
    docx = env.client.get("/documents/%s/file" % docx_id, headers=env.headers)
    assert docx.headers["content-type"].startswith("application/vnd.openxmlformats-officedocument.wordprocessingml")


def test_file_endpoint_refuses_paths_outside_upload_dir(env: Env) -> None:
    secret = env.settings.upload_dir.parent / "secret.txt"
    secret.write_text("top secret")
    doc_id = upload(env, "employee_handbook.pdf", HANDBOOK).json()["id"]
    with env.sessions() as session:  # simulate a tampered DB row
        row = session.get(Document, uuid.UUID(doc_id))
        row.file_path = "../secret.txt"
        session.commit()
    response = env.client.get("/documents/%s/file" % doc_id, headers=env.headers)
    assert response.status_code == 404 and "top secret" not in response.text


def test_delete_removes_row_file_and_vectors(env: Env) -> None:
    keep = upload(env, "cloud_service_agreement.docx", AGREEMENT).json()
    gone = upload(env, "employee_handbook.pdf", HANDBOOK).json()
    assert env.client.delete("/documents/%s" % gone["id"], headers=env.headers).status_code == 204

    assert env.client.get("/documents/%s" % gone["id"], headers=env.headers).status_code == 404
    assert env.store.count() == keep["chunk_count"]  # only the other document's chunks remain
    assert [p.name for p in stored_files(env)] == ["%s.docx" % keep["id"]]
    assert env.client.delete("/documents/%s" % gone["id"], headers=env.headers).status_code == 404
    # the same file can be uploaded again after deletion
    assert upload(env, "employee_handbook.pdf", HANDBOOK).status_code == 201


# ---------------------------------------------------------------- ask


LEAVE_Q = "How many days of paid annual leave does every full-time employee receive?"


def _answer_citing(text: str) -> Any:
    def build(kwargs: Dict[str, Any]) -> LLMAnswer:
        prompt = kwargs["input"][1]["content"]
        alias = next(m.group(1) for m in re.finditer(r'<chunk id="(C\d+)"[^>]*>\n(.*?)\n</chunk>', prompt, re.S) if text in m.group(2))
        return LLMAnswer(answerable=True, citations=[LLMCitation(chunk_id=alias, quote=text)], answer="25 days.")

    return build


def test_ask_answers_with_verified_citation_and_logs_metrics_only(env: Env) -> None:
    doc = upload(env, "employee_handbook.pdf", HANDBOOK).json()
    env.llm.script(_answer_citing("25 days of paid annual leave per calendar year"))

    response = env.client.post("/ask", json={"question": LEAVE_Q}, headers=env.headers)
    assert response.status_code == 200
    body = response.json()
    assert body["answerable"] is True and body["reason"] is None
    citation = body["citations"][0]
    assert (citation["document_id"], citation["filename"], citation["page"]) == (doc["id"], "employee_handbook.pdf", 2)
    assert body["retrieved"][0]["chunk_id"] == citation["chunk_id"] and body["retrieved"][0]["score"] > 0.2
    assert set(body["usage"]) >= {"prompt_tokens", "completion_tokens", "latency_ms"}

    with env.sessions() as session:
        row = session.scalars(select(Query)).one()
    assert row.question == LEAVE_Q and row.answerable is True and row.model_name == "gpt-test"
    assert (row.prompt_tokens, row.completion_tokens) == (1000, 100) and row.retrieved_count == len(body["retrieved"])
    assert not hasattr(row, "answer")  # no answer / document text persisted


def test_ask_refusal_below_threshold_does_not_call_llm_and_logs_null_model(env: Env) -> None:
    upload(env, "employee_handbook.pdf", HANDBOOK)
    response = env.client.post("/ask", json={"question": "zebra quantum saxophone"}, headers=env.headers)
    body = response.json()
    assert response.status_code == 200 and body["answerable"] is False and body["reason"] == "below_score_threshold"
    assert env.llm.calls == [] and body["usage"]["llm_calls"] == 0
    with env.sessions() as session:
        row = session.scalars(select(Query)).one()
    assert row.model_name is None and row.prompt_tokens == 0 and row.answerable is False


def test_ask_can_be_restricted_to_selected_documents(env: Env) -> None:
    handbook = upload(env, "employee_handbook.pdf", HANDBOOK).json()
    agreement = upload(env, "cloud_service_agreement.docx", AGREEMENT).json()
    env.llm.script(LLMAnswer(answerable=False, citations=[], answer="No."))
    body = env.client.post(
        "/ask", json={"question": "uptime guarantee per month", "document_ids": [handbook["id"]]}, headers=env.headers
    ).json()
    assert {r["document_id"] for r in body["retrieved"]} <= {handbook["id"]}
    assert agreement["id"] not in {r["document_id"] for r in body["retrieved"]}


def test_ask_unknown_document_id_is_404(env: Env) -> None:
    response = env.client.post("/ask", json={"question": "x?", "document_ids": [str(uuid.uuid4())]}, headers=env.headers)
    assert response.status_code == 404 and error_code(response) == "document_not_found"


def test_ask_validation_errors_use_the_envelope_and_do_not_echo_input(env: Env) -> None:
    for payload in [{"question": ""}, {"question": "ok", "top_k": 0}, {"question": "x" * 2001}, {}]:
        response = env.client.post("/ask", json=payload, headers=env.headers)
        assert response.status_code == 422 and error_code(response) == "validation_error"
    assert "xxxxxxxx" not in env.client.post("/ask", json={"question": "x" * 2001}, headers=env.headers).text


def test_llm_outage_is_502_not_a_stack_trace(env: Env) -> None:
    upload(env, "employee_handbook.pdf", HANDBOOK)
    env.llm.script(openai.APIConnectionError(request=httpx.Request("POST", "http://x")))
    response = env.client.post("/ask", json={"question": LEAVE_Q}, headers=env.headers)
    assert response.status_code == 502 and error_code(response) == "upstream_error"
    assert "Traceback" not in response.text


def test_unexpected_exception_returns_generic_500(env: Env) -> None:
    upload(env, "employee_handbook.pdf", HANDBOOK)
    env.llm.script(lambda _k: (_ for _ in ()).throw(RuntimeError("secret internal detail")))
    response = env.client.post("/ask", json={"question": LEAVE_Q}, headers=env.headers)
    assert response.status_code == 500 and error_code(response) == "internal_error"
    assert "secret internal detail" not in response.text and "Traceback" not in response.text


def test_missing_openai_key_gives_clean_503() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    app.dependency_overrides.clear()
    # Real get_embedder -> get_openai_client with OPENAI_API_KEY absent from the test env.
    from app.core.config import Settings, get_settings
    from app.core.openai_client import get_openai_client

    get_openai_client.cache_clear()
    keyless = Settings(api_key="test-api-key-0123456789", database_url="sqlite://", openai_api_key=None)  # type: ignore[arg-type]
    app.dependency_overrides[get_settings] = lambda: keyless
    import app.core.openai_client as module

    original = module.get_settings
    module.get_settings = lambda: keyless  # type: ignore[assignment]
    try:
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/ask", json={"question": "hi"}, headers={"X-API-Key": "test-api-key-0123456789"})
        assert response.status_code == 503 and response.json()["error"]["code"] == "service_not_configured"
    finally:
        module.get_settings = original  # type: ignore[assignment]
        app.dependency_overrides.clear()
        get_openai_client.cache_clear()
