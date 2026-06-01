from __future__ import annotations
from pathlib import Path
import importlib
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

        # Dynamically import a PDF reader backend to avoid hard top-level import
        PdfReader = None
        for module_name in ("pypdf", "PyPDF2"):
            try:
                module = importlib.import_module(module_name)
                PdfReader = getattr(module, "PdfReader")
                break
            except ModuleNotFoundError:
                PdfReader = None

        if PdfReader is None:
            raise RuntimeError(
                "No PDF backend found. Install 'pypdf' or 'PyPDF2'."
            )

        reader = PdfReader(str(path))

        text = []

        # Support both pypdf and PyPDF2 page APIs and older method names
        for page in getattr(reader, "pages", []):
            if hasattr(page, "extract_text"):
                extracted = page.extract_text()
            elif hasattr(page, "extractText"):
                extracted = page.extractText()
            else:
                extracted = None

            if extracted:
                text.append(extracted)

        return "\n".join(text)