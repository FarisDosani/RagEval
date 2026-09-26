from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints, model_validator

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
BenchmarkCategory = Literal[
    "factual", "definition", "comparison", "multi_hop", "reasoning", "unanswerable"
]
BenchmarkDifficulty = Literal["easy", "medium", "hard"]


class BenchmarkItem(BaseModel):
    """A controlled QA example; chunk IDs must refer to the intended corpus."""

    model_config = ConfigDict(extra="forbid")

    question_id: NonEmptyText
    question: NonEmptyText
    ground_truth: NonEmptyText | None = None
    answerable: StrictBool
    relevant_chunk_ids: list[NonEmptyText] = Field(default_factory=list)
    category: BenchmarkCategory
    difficulty: BenchmarkDifficulty

    @model_validator(mode="after")
    def require_answer(self) -> Self:
        if self.answerable and self.ground_truth is None:
            raise ValueError("Answerable questions require non-empty ground_truth")
        return self
