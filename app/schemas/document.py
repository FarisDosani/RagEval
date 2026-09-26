from pydantic import BaseModel, Field


class ExtractedUnit(BaseModel):
    """Text from one PDF page, TXT file, or DOCX paragraph."""

    document_id: str
    source: str
    page_number: int | None = Field(default=None, ge=1)
    text: str
