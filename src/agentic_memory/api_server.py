"""REST API server for agentic-memory.

Usage:
    am-server --repo /path/to/repo [--host 0.0.0.0] [--port 8080]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False

    # Stubs so class definitions don't crash at import time
    class BaseModel:  # type: ignore[no-redef]
        pass

    def Field(**kw):  # type: ignore[no-redef]
        return None

    class HTTPException(Exception):  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            pass

    class FastAPI:  # type: ignore[no-redef]
        """Stub so @app.post()/@app.get() decorators don't crash at import time."""

        def __init__(self, **kw):
            pass

        def _noop_decorator(self, *a, **kw):
            def decorator(fn):
                return fn
            return decorator

        post = get = delete = put = patch = _noop_decorator

from agentic_memory.evidence import FileRef, GitCommitRef, ManualRef, URLRef
from agentic_memory.memory import Memory

if _API_AVAILABLE:
    app = FastAPI(
        title="agentic-memory",
        description="Repo memory for AI agents — every memory has a source, every source gets verified.",
        version="0.7.2",
    )
else:
    app = FastAPI()  # type: ignore[assignment]

_memory: Memory | None = None


def _get_memory() -> Memory:
    global _memory
    if _memory is None:
        repo_path = os.environ.get("AGENTIC_MEMORY_REPO", ".")
        _memory = Memory(repo_path)
    return _memory


# --- Request / Response models ---


class EvidenceRequest(BaseModel):
    type: str = Field(description="One of: file, url, git_commit, manual")
    file_path: str | None = None
    lines_start: int | None = None
    lines_end: int | None = None
    url: str | None = None
    commit_sha: str | None = None
    commit_file: str | None = None
    note: str | None = None


class AddRequest(BaseModel):
    content: str
    evidence: EvidenceRequest
    tags: list[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    query: str
    limit: int = 5
    validate_citations: bool = Field(default=True, alias="validate")
    include_stale: bool = True
    fts_weight: float = 0.65
    vector_weight: float = 0.35


class MemoryResponse(BaseModel):
    id: str
    content: str
    evidence_label: str
    confidence: float
    validation_status: str
    validation_message: str
    tags: list[str]


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    memories: list[MemoryResponse]


class StatusResponse(BaseModel):
    total: int
    valid: int
    stale: int
    invalid: int
    unchecked: int


class ValidateResponse(BaseModel):
    status: StatusResponse
    problematic: list[MemoryResponse]


class MessageResponse(BaseModel):
    message: str


# --- Helpers ---


def _build_evidence(ev: EvidenceRequest) -> FileRef | GitCommitRef | URLRef | ManualRef:
    if ev.type == "file":
        if not ev.file_path:
            raise HTTPException(400, "file_path is required for file evidence")
        lines = (ev.lines_start, ev.lines_end) if ev.lines_start and ev.lines_end else None
        return FileRef(path=ev.file_path, lines=lines)
    elif ev.type == "url":
        if not ev.url:
            raise HTTPException(400, "url is required for url evidence")
        return URLRef(url=ev.url)
    elif ev.type == "git_commit":
        if not ev.commit_sha:
            raise HTTPException(400, "commit_sha is required for git_commit evidence")
        return GitCommitRef(sha=ev.commit_sha, file_path=ev.commit_file)
    elif ev.type == "manual":
        return ManualRef(note=ev.note or "manually added")
    else:
        raise HTTPException(400, f"Unknown evidence type: {ev.type}")


def _record_to_response(record: Any) -> MemoryResponse:
    return MemoryResponse(
        id=record.id,
        content=record.content,
        evidence_label=record.evidence_label,
        confidence=record.confidence,
        validation_status=record.validation_status.value,
        validation_message=record.validation_message,
        tags=record.tags,
    )


# --- Endpoints ---


@app.post("/memories", response_model=MemoryResponse, status_code=201)
def add_memory(req: AddRequest):
    """Add a memory with required evidence citation."""
    mem = _get_memory()
    evidence = _build_evidence(req.evidence)
    try:
        record = mem.add(req.content, evidence=evidence, tags=req.tags)
        return _record_to_response(record)
    except TypeError as e:
        raise HTTPException(400, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/memories/query", response_model=QueryResponse)
def query_memories(req: QueryRequest):
    """Query memories with hybrid search and citation re-validation."""
    mem = _get_memory()
    result = mem.query(
        req.query,
        limit=req.limit,
        validate=req.validate_citations,
        include_stale=req.include_stale,
        fts_weight=req.fts_weight,
        vector_weight=req.vector_weight,
    )
    return QueryResponse(
        answer=result.answer,
        confidence=result.confidence,
        memories=[_record_to_response(r) for r in result.memories],
    )


@app.get("/memories", response_model=list[MemoryResponse])
def list_memories(limit: int = 50):
    """List all stored memories."""
    mem = _get_memory()
    records = mem.list_all(limit=limit)
    return [_record_to_response(r) for r in records]


@app.get("/memories/{memory_id}", response_model=MemoryResponse)
def get_memory(memory_id: str):
    """Get a specific memory by ID."""
    mem = _get_memory()
    record = mem.get(memory_id)
    if record is None:
        raise HTTPException(404, f"Memory {memory_id} not found")
    return _record_to_response(record)


@app.delete("/memories/{memory_id}", response_model=MessageResponse)
def delete_memory(memory_id: str):
    """Delete a memory by ID."""
    mem = _get_memory()
    if not mem.delete(memory_id):
        raise HTTPException(404, f"Memory {memory_id} not found")
    return MessageResponse(message=f"Deleted memory {memory_id}")


@app.post("/memories/validate", response_model=ValidateResponse)
def validate_memories():
    """Validate all memories by re-checking evidence citations."""
    mem = _get_memory()
    problematic = mem.validate()
    s = mem.status()
    return ValidateResponse(
        status=StatusResponse(**s),
        problematic=[_record_to_response(r) for r in problematic],
    )


@app.get("/status", response_model=StatusResponse)
def memory_status():
    """Get summary status of all memories."""
    mem = _get_memory()
    return StatusResponse(**mem.status())


def main():
    if not _API_AVAILABLE:
        print("Error: API dependencies not installed. Run: pip install memcite[api]", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="agentic-memory REST API server")
    parser.add_argument(
        "--repo", default=None, help="Repository path (default: cwd or AGENTIC_MEMORY_REPO env)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    args = parser.parse_args()

    if args.repo:
        os.environ["AGENTIC_MEMORY_REPO"] = args.repo

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
