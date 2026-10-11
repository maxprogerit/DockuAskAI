from __future__ import annotations

import enum
import io
import os
import re
import zipfile

from app.core.errors import FileTooLargeError, UnsupportedFileTypeError

# Zip-bomb guard: a DOCX is a zip, and we never want to inflate gigabytes from a small upload.
MAX_DOCX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024


class FileType(str, enum.Enum):
    PDF = "pdf"
    DOCX = "docx"


_EXTENSIONS = {".pdf": FileType.PDF, ".docx": FileType.DOCX}
_CONTROL_CHARS = re.compile(r'[\x00-\x1f\x7f"]')  # quotes would break Content-Disposition headers


def sanitize_filename(raw: str) -> str:
    """Display name only. The file is stored as <uuid>.<ext>, so this never touches the filesystem."""
    name = (raw or "").replace("\\", "/").split("/")[-1]  # drop any client-side path
    name = _CONTROL_CHARS.sub("", name).strip()
    return name[:255] or "document"


def check_size(size_bytes: int, max_bytes: int) -> None:
    if size_bytes > max_bytes:
        raise FileTooLargeError("File is larger than the limit of %d MB." % (max_bytes // (1024 * 1024)))


def detect_file_type(data: bytes, filename: str) -> FileType:
    """Decide the type from CONTENT (magic bytes / zip structure) and require the extension to agree.

    The extension alone is trivially spoofable; the content alone would let a renamed .xlsx through
    (it is a zip too), so for DOCX we look for the parts Word always writes.
    """
    if not data:
        raise UnsupportedFileTypeError("The uploaded file is empty.")

    extension = os.path.splitext(filename)[1].lower()
    declared = _EXTENSIONS.get(extension)
    if declared is None:
        raise UnsupportedFileTypeError("Only .pdf and .docx files are supported.")

    # PDF spec allows junk before the header, but only within the first 1024 bytes.
    if b"%PDF-" in data[:1024]:
        detected = FileType.PDF
    elif data[:4] == b"PK\x03\x04" and _is_docx(data):
        detected = FileType.DOCX
    else:
        raise UnsupportedFileTypeError("File content is not a valid PDF or DOCX document.")

    if detected != declared:
        raise UnsupportedFileTypeError(
            "File extension .%s does not match its content (%s)." % (declared.value, detected.value)
        )
    return detected


def _is_docx(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                return False
            total = sum(info.file_size for info in archive.infolist())
            if total > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise UnsupportedFileTypeError("DOCX expands to an unreasonable size.")
            return True
    except zipfile.BadZipFile:
        return False
