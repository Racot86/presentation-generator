#!/usr/bin/env python3
"""
Simple API server for listing and downloading generated presentations.

Endpoints:
 - GET /get_list_of_presentations
     Returns JSON with a list of available presentation files (relative paths),
     discovered under generated_presentations/ recursively. Only .pdf files are listed.

 - GET /download_presentation?name=<relative_path>
     Sends the requested PDF for download. The `name` must be a path relative to
     the generated_presentations/ directory (e.g., `biology/10/Topic_Name.pdf`).

Run locally:
  uvicorn server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel
import threading
import time
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).parent
PRESENTATIONS_ROOT = PROJECT_ROOT / "generated_presentations"

app = FastAPI(
    title="Presentation Generator API",
    description=(
        "Browse and download generated presentation PDFs. "
        "Interactive API documentation is available via Swagger UI at /docs and ReDoc at /redoc."
    ),
    version="0.1.0",
    openapi_tags=[
        {
            "name": "Presentations",
            "description": "Operations for listing and downloading presentation PDFs.",
        }
    ],
)


class PresentationListResponse(BaseModel):
    presentations: List[str]


class ErrorResponse(BaseModel):
    detail: str


def _safe_resolve_within(base: Path, candidate: Path) -> Path:
    """Resolve candidate and ensure it stays within base to prevent path traversal."""
    base_resolved = base.resolve()
    cand_resolved = candidate.resolve()
    try:
        cand_resolved.relative_to(base_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")
    return cand_resolved


@app.get(
    "/get_list_of_presentations",
    response_model=PresentationListResponse,
    tags=["Presentations"],
    summary="List available presentations",
    description=(
        "Recursively scans the generated_presentations/ directory and returns "
        "relative POSIX-style paths to available .pdf files."
    ),
)
def get_list_of_presentations() -> PresentationListResponse:
    if not PRESENTATIONS_ROOT.exists():
        return PresentationListResponse(presentations=[])

    presentations: List[str] = []
    for path in PRESENTATIONS_ROOT.rglob("*.pdf"):
        # produce POSIX-style relative path for API stability
        rel = path.relative_to(PRESENTATIONS_ROOT).as_posix()
        presentations.append(rel)

    # Optional: stable sort for deterministic output
    presentations.sort()
    return PresentationListResponse(presentations=presentations)


@app.get(
    "/download_presentation",
    tags=["Presentations"],
    summary="Download a presentation PDF",
    description=(
        "Provide the relative path (e.g., biology/10/Topic_Name.pdf) under "
        "generated_presentations/ to download the file."
    ),
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "The requested PDF file.",
        },
        400: {"model": ErrorResponse, "description": "Invalid path or parameters."},
        404: {"model": ErrorResponse, "description": "File not found."},
    },
)
def download_presentation(
    name: str = Query(
        ..., description="Relative path to the PDF under generated_presentations/"
    )
):
    if not name or name.strip() == "":
        raise HTTPException(status_code=400, detail="Parameter 'name' is required")

    # Only allow relative paths, no absolute
    if name.startswith("/"):
        raise HTTPException(status_code=400, detail="Path must be relative")

    requested_path = PRESENTATIONS_ROOT / name
    try:
        safe_path = _safe_resolve_within(PRESENTATIONS_ROOT, requested_path)
    except HTTPException:
        # Re-raise as 400 for clarity
        raise

    if not safe_path.exists() or not safe_path.is_file():
        raise HTTPException(status_code=404, detail="Presentation not found")

    return FileResponse(
        path=str(safe_path),
        media_type="application/pdf",
        filename=safe_path.name,
    )


# -----------------------------
# Generation job state handling
# -----------------------------

class JobStatus(BaseModel):
    running: bool
    started_at: Optional[str] = None
    elapsed_sec: Optional[int] = None
    generated: int = 0
    total: int = 0
    finished: bool = False
    message: Optional[str] = None


class _JobState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.running: bool = False
        self.started_at: Optional[float] = None  # epoch seconds
        self.generated: int = 0
        self.total: int = 0
        self.finished: bool = False
        self.message: Optional[str] = None

    def snapshot(self) -> JobStatus:
        with self._lock:
            started_iso = (
                datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat()
                if self.started_at
                else None
            )
            elapsed = int(time.time() - self.started_at) if self.started_at else None
            return JobStatus(
                running=self.running,
                started_at=started_iso,
                elapsed_sec=elapsed,
                generated=self.generated,
                total=self.total,
                finished=self.finished,
                message=self.message,
            )

    def reset_and_start(self, total: int) -> None:
        with self._lock:
            self.running = True
            self.started_at = time.time()
            self.generated = 0
            self.total = total
            self.finished = False
            self.message = "in-progress"

    def increment_generated(self, n: int = 1) -> None:
        with self._lock:
            self.generated += n

    def finish(self, message: str = "done") -> None:
        with self._lock:
            self.running = False
            self.finished = True
            self.message = message

    def fail(self, message: str) -> None:
        with self._lock:
            self.running = False
            self.finished = True
            self.message = message


job_state = _JobState()


def _compute_total_topics() -> int:
    """Compute total number of topics that will be attempted.

    Mirrors the generator's JSON parsing and topic extraction as closely as possible.
    We count non-empty titles; greeting/intro skips are applied to approximate the
    generator's filtering.
    """
    toc_root = PROJECT_ROOT / "toc_openai_filtered"
    if not toc_root.exists():
        return 0

    total = 0
    import json

    # Same skip patterns as in generate_presentations_gemini.process_single_topic
    skip_patterns = {
        "зміст",
        "вступ",
        "від автора",
        "передмова",
        "додатки",
        "список літератури",
        "шановні",
        "дорогі",
        "читачі",
        "друзі",
        "шановні дев'ятикласники",
        "шановні дев'ятикласниці",
        "дорогі учні",
        "дорогі студенти",
        "шановні учні",
        "шановні студенти",
    }
    greeting_starters = ("шановні", "дорогі", "читачі", "друзі")

    for json_path in toc_root.rglob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        items = data.get("toc", [])
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            title = str(it.get("title", "")).strip()
            if not title:
                continue
            low = title.lower()
            if any(word in low for word in skip_patterns):
                continue
            if low.startswith(greeting_starters):
                continue
            total += 1
    return total


def _run_generation_job() -> None:
    """Background job wrapper that updates job_state while running the generator."""
    try:
        # Import inside to avoid startup overhead
        import importlib

        gen_mod = importlib.import_module("generate_presentations_gemini")

        # Wrap process_single_topic to increment the counter on success
        orig_process_single_topic = getattr(gen_mod, "process_single_topic", None)

        if callable(orig_process_single_topic):
            def wrapped_process_single_topic(*args: Any, **kwargs: Any):
                res = orig_process_single_topic(*args, **kwargs)
                try:
                    # res is (topic_title, success, message)
                    if isinstance(res, tuple) and len(res) >= 2 and bool(res[1]):
                        job_state.increment_generated(1)
                except Exception:
                    pass
                return res

            setattr(gen_mod, "process_single_topic", wrapped_process_single_topic)

        # Run the main generator
        gen_main = getattr(gen_mod, "main")
        gen_main()
        job_state.finish("done")
    except SystemExit:
        # The generator may call sys.exit; treat as finished
        job_state.finish("done")
    except Exception as e:
        job_state.fail(f"failed: {e}")


@app.post(
    "/generate_all",
    tags=["Presentations"],
    summary="Generate all presentations (background)",
    description=(
        "Starts generation of all presentations as a background task and returns immediately. "
        "Watch server logs for progress."
    ),
)
def generate_all_presentations(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Trigger generation of all presentations in the background.

    Uses the main() entrypoint from generate_presentations_gemini.py.
    """
    # If already running, return current status without starting a new one
    status = job_state.snapshot()
    if status.running:
        return {
            "status": "already-running",
            "running": status.running,
            "started_at": status.started_at,
            "elapsed_sec": status.elapsed_sec,
            "generated": status.generated,
            "total": status.total,
            "finished": status.finished,
            "message": status.message,
        }

    # Not running — compute total and start a new background job
    total = _compute_total_topics()
    job_state.reset_and_start(total)
    background_tasks.add_task(_run_generation_job)
    status = job_state.snapshot()
    return {
        "status": "scheduled",
        "running": status.running,
        "started_at": status.started_at,
        "elapsed_sec": status.elapsed_sec,
        "generated": status.generated,
        "total": status.total,
        "finished": status.finished,
        "message": status.message,
    }


@app.get(
    "/generate_status",
    tags=["Presentations"],
    summary="Get generation status",
    description="Returns whether generation is running and progress (generated/total).",
    response_model=JobStatus,
)
def generate_status() -> JobStatus:
    return job_state.snapshot()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect to interactive API docs (Swagger UI)."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    # Allow running via `python server.py` for convenience
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
