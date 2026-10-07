"""Bounded raw-file upload; filenames never select a filesystem path."""
import os
import re
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from app.schemas.runtime import DocumentInfo

router = APIRouter()
UPLOAD_ROOT = Path(__file__).resolve().parents[2] / "data/uploads"


def runtime_service(request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(503, "Enable RAGEVAL_AUTO_INIT to manage the session corpus")
    return runtime


def safe_filename(filename):
    if any(c in filename for c in ('/', '\\', ':', '\x00')) or filename in {'.', '..'}:
        raise HTTPException(400, "Filename must not contain a path")
    name = re.sub(r'[^\w.() #\-]', '_', filename).strip(' .')
    stem = Path(name).stem.upper()
    if stem in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        raise HTTPException(400, "Invalid filename")
    if Path(name).suffix.lower() not in {'.pdf', '.txt', '.docx'}:
        raise HTTPException(400, "Supported formats: PDF, TXT, DOCX")
    return name


@router.get('/documents', response_model=list[DocumentInfo])
def documents(request: Request):
    return runtime_service(request).snapshot.documents


@router.post('/documents/upload', response_model=DocumentInfo)
async def upload(request: Request, filename: str = Query(min_length=1, max_length=120)):
    """Send file bytes as application/octet-stream and filename as a query parameter."""
    runtime = runtime_service(request)
    name = safe_filename(filename)
    limit = int(os.environ.get('RAGEVAL_MAX_UPLOAD_BYTES', str(10 * 1024 * 1024)))
    if limit <= 0:
        raise RuntimeError('RAGEVAL_MAX_UPLOAD_BYTES must be positive')
    folder = UPLOAD_ROOT / str(uuid4())
    folder.mkdir(parents=True)
    path = folder / name
    completed = False
    try:
        size = 0
        with path.open('xb') as target:
            async for block in request.stream():
                size += len(block)
                if size > limit:
                    raise HTTPException(413, 'Upload exceeds the configured size limit')
                target.write(block)
        if size == 0:
            raise HTTPException(400, 'File must not be empty')
        result = await run_in_threadpool(runtime.add_document, path)
        completed = True
        return result
    finally:
        if not completed:
            path.unlink(missing_ok=True)
            folder.rmdir()
