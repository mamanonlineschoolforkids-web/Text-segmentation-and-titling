"""
api.py — Transcription Segmentation REST API
=============================================
Endpoints:
  POST /transcript                    — submit transcript, returns job_id
  GET  /transcript/{job_id}           — full result (titles + segments + summaries)
  GET  /transcript/{job_id}/titles    — only topic titles
  GET  /transcript/{job_id}/segments  — only segment texts
  GET  /transcript/{job_id}/summaries — only summaries

Run:
  uvicorn api:app --reload
"""

import os
import uuid
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core import run_pipeline

load_dotenv(Path(__file__).parent / ".env")


# ─────────────────────────────────────────────
# CONFIG — edit these values directly
# ─────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"

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


# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Transcription Segmentation API",
    description="Segment transcripts by topic using Groq LLM.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post("/transcript", response_model=SubmitResponse, status_code=202)
async def submit_transcript(
    text: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
):
    """
    Submit a transcript for processing.
    Send either a `text` form field or a `file` (.txt) upload.
    Returns a `job_id` to use with the GET endpoints.
    """
    if file:
        content    = await file.read()
        transcript = content.decode("utf-8")
    elif text:
        transcript = text
    else:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'file'.")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in .env file.")

    job_id = str(uuid.uuid4())
    store[job_id] = {"status": "processing", "segments": []}

    try:
        segments = run_pipeline(
            transcript=transcript,
            groq_api_key=groq_api_key,
            groq_model=GROQ_MODEL,
        )
        store[job_id] = {
            "status": "done",
            "segments": [
                {
                    "index": seg.index,
                    "title": seg.title,
                    "summary": seg.summary,
                    "text": seg.text,
                    "start_line": seg.start_line,
                    "end_line": seg.end_line,
                }
                for seg in segments
            ],
        }
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
    """Only the topic title for each segment, in order."""
    job = _get_job(job_id)
    return TitlesResult(
        job_id=job_id,
        titles=[s["title"] for s in job["segments"]],
    )


@app.get("/transcript/{job_id}/segments", response_model=SegmentsResult)
def get_segments(job_id: str):
    """Only the text content of each segment."""
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
    """Only the summary for each segment."""
    job = _get_job(job_id)
    return SummariesResult(
        job_id=job_id,
        summaries=[
            {"index": s["index"], "title": s["title"], "summary": s["summary"]}
            for s in job["segments"]
        ],
    )


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def _get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job.get("error", "Pipeline failed."))
    return job