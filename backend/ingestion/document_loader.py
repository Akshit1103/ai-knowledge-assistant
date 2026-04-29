"""
Document loaders for PDF, TXT, and Markdown files.
"""

import os
from typing import Dict, Any
from PyPDF2 import PdfReader


class DocumentLoader:
    """Loads documents from various file formats into a unified structure."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    @staticmethod
    def load(file_path: str) -> Dict[str, Any]:
        """
        Load a document and return a dict with keys:
          name, pages (list of {text, page_number}), full_text, num_pages
        """
        ext = os.path.splitext(file_path)[1].lower()
        name = os.path.basename(file_path)

        if ext == ".pdf":
            return DocumentLoader._load_pdf(file_path, name)
        elif ext in {".txt", ".md", ".markdown"}:
            return DocumentLoader._load_text(file_path, name)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def _load_pdf(file_path: str, name: str) -> Dict[str, Any]:
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"text": text, "page_number": i + 1})
        return {
            "name": name,
            "pages": pages,
            "full_text": "\n\n".join(p["text"] for p in pages),
            "num_pages": len(reader.pages),
        }

    @staticmethod
    def _load_text(file_path: str, name: str) -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return {
            "name": name,
            "pages": [{"text": text, "page_number": 1}],
            "full_text": text,
            "num_pages": 1,
        }
