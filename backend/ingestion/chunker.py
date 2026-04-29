"""
Recursive text chunker with configurable size and overlap.
"""

from typing import List, Dict, Any
from config import settings


class TextChunker:
    """Splits documents into overlapping chunks using a recursive strategy."""

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a loaded document into metadata-enriched chunks."""
        chunks: List[Dict[str, Any]] = []
        chunk_index = 0

        for page in document["pages"]:
            page_chunks = self._split_text(page["text"])
            for chunk_text in page_chunks:
                if chunk_text.strip():
                    chunks.append(
                        {
                            "text": chunk_text.strip(),
                            "document_name": document["name"],
                            "page_number": page.get("page_number"),
                            "chunk_index": chunk_index,
                        }
                    )
                    chunk_index += 1
        return chunks

    # ------------------------------------------------------------------
    # Splitting helpers
    # ------------------------------------------------------------------

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        return self._recursive_split(text, 0)

    def _recursive_split(self, text: str, sep_idx: int) -> List[str]:
        if sep_idx >= len(self.SEPARATORS):
            return self._force_split(text)

        separator = self.SEPARATORS[sep_idx]
        splits = text.split(separator)

        chunks: List[str] = []
        current = ""

        for part in splits:
            candidate = (current + separator + part) if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > self.chunk_size:
                    chunks.extend(self._recursive_split(part, sep_idx + 1))
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _force_split(self, text: str) -> List[str]:
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks = []
        for i in range(0, len(text), step):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """Re-build chunks so each one carries a tail from the previous."""
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self.chunk_overlap :]
            merged = prev_tail + chunks[i]
            if len(merged) <= self.chunk_size:
                result.append(merged)
            else:
                result.append(chunks[i])
        return result
