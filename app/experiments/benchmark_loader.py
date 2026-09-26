import json
from pathlib import Path

from pydantic import ValidationError

from app.schemas.benchmark import BenchmarkItem


def load_benchmark(path: str | Path) -> list[BenchmarkItem]:
    """Load a UTF-8 JSON array in file order, validating all items and IDs.

    An empty array is valid. Invalid JSON, item schemas, and duplicate IDs
    raise ValueError with file/item context. File access errors propagate.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON benchmark in {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"Benchmark {path} must contain a JSON array of items")

    items: list[BenchmarkItem] = []
    seen: set[str] = set()
    for position, raw_item in enumerate(payload, start=1):
        try:
            item = BenchmarkItem.model_validate(raw_item)
        except ValidationError as exc:
            raise ValueError(f"Invalid benchmark item {position} in {path}: {exc}") from exc
        if item.question_id in seen:
            raise ValueError(f"Duplicate question_id {item.question_id!r} at item {position} in {path}")
        seen.add(item.question_id)
        items.append(item)
    return items
