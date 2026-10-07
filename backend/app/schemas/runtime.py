from typing import Literal
from pydantic import BaseModel
from app.schemas.generation import RAGResponse

RetrievalStrategy = Literal["dense", "bm25", "hybrid", "llm_only"]


class DocumentInfo(BaseModel):
    document_id: str
    name: str
    extracted_unit_count: int
    chunk_count: int
    index_status: Literal["ready"] = "ready"


class QueryResponse(RAGResponse):
    retrieval_strategy: RetrievalStrategy
