"""
api.py — Transcription Segmentation REST API
=============================================
Direct endpoints (no file upload, reads from server):
  POST /transcript/process            — full result in one call
  POST /transcript/process/titles     — only titles in one call
  POST /transcript/process/segments   — only segments in one call
  POST /transcript/process/summaries  — only summaries in one call

Job-based endpoints (manual job_id):
  POST /transcript                    — submit transcript, returns job_id
  GET  /transcript/{job_id}           — full result
  GET  /transcript/{job_id}/titles    — only titles
  GET  /transcript/{job_id}/segments  — only segments
  GET  /transcript/{job_id}/summaries — only summaries

Run:
  uvicorn api:app --reload
"""

import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core import run_pipeline, TopicSegment

load_dotenv(Path(__file__).parent / ".env")


# ─────────────────────────────────────────────
# CONFIG — edit these values directly
# ─────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"

# Path to the transcript file written by your extraction tool
# Can be absolute or relative to this file
TRANSCRIPT_FILE = Path(__file__).parent / "transcript.txt"

# ─────────────────────────────────────────────
# In-memory job store  { job_id: dict }
# ─────────────────────────────────────────────

store: dict[str, dict] = {}


# ─────────────────────────────────────────────
# Pydantic response models
# ─────────────────────────────────────────────

class SegmentOut(BaseModel):
    index: int
    title: str
    summary: str
    text: str
    start_line: int
    end_line: int


class TranscriptResult(BaseModel):
    job_id: str
    status: str
    segments: list[SegmentOut] = []
    error: Optional[str] = None


class TitlesResult(BaseModel):
    job_id: str
    titles: list[str]


class SegmentsResult(BaseModel):
    job_id: str
    segments: list[dict]


class SummariesResult(BaseModel):
    job_id: str
    summaries: list[dict]


class SubmitResponse(BaseModel):
    job_id: str
    message: str


# Direct response models (no job_id)
class DirectTranscriptResult(BaseModel):
    segments: list[SegmentOut] = []


class DirectTitlesResult(BaseModel):
    titles: list[str]


class DirectSegmentsResult(BaseModel):
    segments: list[dict]


class DirectSummariesResult(BaseModel):
    summaries: list[dict]


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Transcription Segmentation API",
    description="Segment transcripts by topic using Groq LLM.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _read_transcript() -> str:
    """Read transcript from the fixed file path set in CONFIG."""
    if not TRANSCRIPT_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Transcript file not found at: {TRANSCRIPT_FILE}"
        )
    return TRANSCRIPT_FILE.read_text(encoding="utf-8")


def _get_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in .env file.")
    return key


def _segments_to_dicts(segments: list[TopicSegment]) -> list[dict]:
    return [
        {
            "index": seg.index,
            "title": seg.title,
            "summary": seg.summary,
            "text": seg.text,
            "start_line": seg.start_line,
            "end_line": seg.end_line,
        }
        for seg in segments
    ]


def _get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job.get("error", "Pipeline failed."))
    return job


# ─────────────────────────────────────────────
# Direct endpoints — reads transcript from server file
# ─────────────────────────────────────────────

@app.post("/transcript/process", response_model=DirectTranscriptResult)
def process_transcript():
    """Read transcript from server and return full result (titles + summaries + segments)."""
    segments = run_pipeline(_read_transcript(), _get_api_key(), GROQ_MODEL)
    return DirectTranscriptResult(segments=[SegmentOut(**s) for s in _segments_to_dicts(segments)])


@app.post("/transcript/process/titles", response_model=DirectTitlesResult)
def process_titles():
    """Read transcript from server and return only the topic titles."""
    segments = run_pipeline(_read_transcript(), _get_api_key(), GROQ_MODEL)
    return DirectTitlesResult(titles=[seg.title for seg in segments])


@app.post("/transcript/process/segments", response_model=DirectSegmentsResult)
def process_segments():
    """Read transcript from server and return only the segment texts."""
    segments = run_pipeline(_read_transcript(), _get_api_key(), GROQ_MODEL)
    return DirectSegmentsResult(
        segments=[
            {"index": seg.index, "title": seg.title, "text": seg.text}
            for seg in segments
        ]
    )


@app.post("/transcript/process/summaries", response_model=DirectSummariesResult)
def process_summaries():
    """Read transcript from server and return only the summaries."""
    segments = run_pipeline(_read_transcript(), _get_api_key(), GROQ_MODEL)
    return DirectSummariesResult(
        summaries=[
            {"index": seg.index, "title": seg.title, "summary": seg.summary}
            for seg in segments
        ]
    )


# ─────────────────────────────────────────────
# Job-based endpoints — returns job_id
# ─────────────────────────────────────────────

@app.post("/transcript", response_model=SubmitResponse, status_code=202)
def submit_transcript():
    """Read transcript from server, process it, and return a job_id for the GET endpoints."""
    job_id = str(uuid.uuid4())
    store[job_id] = {"status": "processing", "segments": []}

    try:
        segments = run_pipeline(_read_transcript(), _get_api_key(), GROQ_MODEL)
        store[job_id] = {"status": "done", "segments": _segments_to_dicts(segments)}
    except Exception as e:
        store[job_id] = {"status": "error", "segments": [], "error": str(e)}
        raise HTTPException(status_code=500, detail=str(e))

    return SubmitResponse(job_id=job_id, message="Processing complete.")


@app.get("/transcript/{job_id}", response_model=TranscriptResult)
def get_transcript(job_id: str):
    """Full result — titles, summaries, and segment texts."""
    job = _get_job(job_id)
    return TranscriptResult(
        job_id=job_id,
        status=job["status"],
        segments=[SegmentOut(**s) for s in job["segments"]],
        error=job.get("error"),
    )


@app.get("/transcript/{job_id}/titles", response_model=TitlesResult)
def get_titles(job_id: str):
    """Only the topic titles, in order."""
    job = _get_job(job_id)
    return TitlesResult(job_id=job_id, titles=[s["title"] for s in job["segments"]])


@app.get("/transcript/{job_id}/segments", response_model=SegmentsResult)
def get_segments(job_id: str):
    """Only the segment texts."""
    job = _get_job(job_id)
    return SegmentsResult(
        job_id=job_id,
        segments=[
            {"index": s["index"], "title": s["title"], "text": s["text"]}
            for s in job["segments"]
        ],
    )


@app.get("/transcript/{job_id}/summaries", response_model=SummariesResult)
def get_summaries(job_id: str):
    """Only the summaries."""
    job = _get_job(job_id)
    return SummariesResult(
        job_id=job_id,
        summaries=[
            {"index": s["index"], "title": s["title"], "summary": s["summary"]}
            for s in job["segments"]
        ],
    )