from __future__ import annotations


class RAGPipeline:
    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim
        self.documents: list[str] = []
        self.doc_ids: list[str] = []

    def upsert_documents(
        self,
        documents: list[str],
        doc_ids: list[str] | None = None,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> None:
        self.documents.extend(documents)

        if doc_ids:
            self.doc_ids.extend(doc_ids)
        else:
            self.doc_ids.extend(
                str(i) for i in range(len(self.documents))
            )

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        if not self.documents:
            return ""

        matches = [
            doc for doc in self.documents
            if query.lower() in doc.lower()
        ]

        if not matches:
            matches = self.documents[:top_k]

        return "\n\n".join(matches[:top_k])

    def __len__(self) -> int:
        return len(self.documents)
