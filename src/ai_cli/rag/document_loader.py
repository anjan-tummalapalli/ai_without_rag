DISALLOWED_PATTERNS = [
    "ignore previous instructions",
    "system prompt",
    "reveal secrets",
]

from __future__ import annotations
from pathlib import Path
from pypdf import PdfReader
from ai_cli.rag.models import Document


class DocumentLoader:
    """
    Loads TXT, Markdown and PDF documents.
    """

    ALLOWED_EXTENSIONS = {
        ".txt",
        ".md",
        ".pdf",
    }

    def load(
        self,
        file_path: str,
    ) -> Document:
        """
        Load document content.
        """

        path = Path(file_path)

        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}"
            )

        if path.suffix.lower() == ".pdf":
            content = self._load_pdf(path)
        else:
            content = path.read_text(
                encoding="utf-8"
            )

        return Document(
            content=content,
            source=str(path),
        )

    def _load_pdf(
        self,
        path: Path,
    ) -> str:
        """
        Extract PDF text.
        """

        reader = PdfReader(str(path))

        text = []

        for page in reader.pages:
            extracted = page.extract_text()

            if extracted:
                text.append(extracted)

        return "\n".join(text)