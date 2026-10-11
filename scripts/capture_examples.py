"""Captures real API request/response pairs into docs/examples/*.json for the README.

    python -m scripts.capture_examples --base-url http://localhost:8000 --api-key "$API_KEY"

Run it against a deployment that has a REAL OPENAI_API_KEY: the answerable / refusal responses are
only meaningful then. The script overwrites the example files on every run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

OUT = Path(__file__).resolve().parent.parent / "docs" / "examples"
SAMPLES = Path(__file__).resolve().parent.parent / "tests" / "samples"


def save(name: str, request: Dict[str, Any], response: httpx.Response) -> Dict[str, Any]:
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text[:500]
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"request": request, "status": response.status_code, "response": body}
    (OUT / ("%s.json" % name)).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("%-28s -> HTTP %d" % (name, response.status_code))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()
    auth = {"X-API-Key": args.api_key}

    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        save("01_health", {"method": "GET", "path": "/health"}, client.get("/health"))
        save("02_unauthorized", {"method": "GET", "path": "/documents", "headers": "no X-API-Key"}, client.get("/documents"))
        save("03_unsupported_file", {"method": "POST", "path": "/documents", "file": "notes.txt (text/plain)"},
             client.post("/documents", headers=auth, files={"file": ("notes.txt", b"hello", "text/plain")}))

        handbook = SAMPLES / "employee_handbook.pdf"
        upload: Optional[httpx.Response] = None
        for name in ("04_upload_new", "05_upload_duplicate"):
            upload = client.post("/documents", headers=auth, files={"file": (handbook.name, handbook.read_bytes(), "application/pdf")})
            save(name, {"method": "POST", "path": "/documents", "file": handbook.name}, upload)

        questions = [
            ("06_ask_answerable", "How many days of paid annual leave does a full-time employee get per year?"),
            ("07_ask_refusal_no_answer", "What is the dress code at Northwind Labs?"),
            ("08_ask_refusal_off_topic", "What is the capital of France?"),
        ]
        for name, question in questions:
            payload = {"question": question}
            save(name, {"method": "POST", "path": "/ask", "json": payload}, client.post("/ask", headers=auth, json=payload))
    print("examples written to", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
