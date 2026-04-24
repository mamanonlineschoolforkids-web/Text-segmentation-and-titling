"""
core.py — Transcription Segmentation Logic
============================================
Groq only returns segment boundaries (start/end line numbers) + title + summary.
Python slices the original transcript lines to produce the segment text.
Nothing is dropped or hallucinated.

Imported by both pipeline.py and api.py.
"""

import json
from dataclasses import dataclass
from groq import Groq


# ─────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────

@dataclass
class TopicSegment:
    """A single topic segment."""
    index: int
    title: str
    summary: str
    text: str           # sliced directly from the original transcript
    start_line: int     # inclusive, 0-based
    end_line: int       # inclusive, 0-based


# ─────────────────────────────────────────────
# Groq prompt
# ─────────────────────────────────────────────

GROQ_SYSTEM_PROMPT = """\
You are an expert transcript analyst. You will receive a transcript where every line is prefixed with its line number, like:

0: Hello everyone, welcome to the meeting.
1: Today we will discuss the budget.
2: ...

Your task is to:
1. SEGMENT the transcript into coherent topic sections, in chronological order.
2. LABEL each segment with a concise topic title (8 words or less).
3. SUMMARIZE each segment in 1-3 sentences.
4. Return the START and END line numbers of each segment.

Return ONLY a valid JSON array (no markdown fences, no commentary) like:
[
  {
    "title": "...",
    "summary": "...",
    "start_line": 0,
    "end_line": 5
  }
]

Rules:
- Every line must be covered. Segments must be contiguous with no gaps.
- start_line and end_line are both inclusive.
- Produce as many segments as needed — do not over-merge unrelated topics.
- The transcript may be in Arabic. Respond with titles and summaries in the same language as the transcript.
"""


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _number_lines(transcript: str) -> tuple[list[str], str]:
    """
    Split transcript into lines and return:
    - lines: the original lines (no prefix)
    - numbered: the same lines with "N: " prefix for Groq to read
    """
    lines = [l for l in transcript.splitlines() if l.strip()]
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(lines))
    return lines, numbered


def _parse_groq_response(raw_json: str) -> list[dict]:
    """Strip markdown fences if present and parse JSON."""
    raw_json = raw_json.strip()
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
        raw_json = raw_json.strip()
    return json.loads(raw_json)


def _slice_lines(lines: list[str], start: int, end: int) -> str:
    """Slice original lines by index range and join back into text."""
    start = max(0, start)
    end   = min(len(lines) - 1, end)
    return "\n".join(lines[start : end + 1])


# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    transcript: str,
    groq_api_key: str,
    groq_model: str,
) -> list[TopicSegment]:
    """
    1. Number each line of the transcript.
    2. Send to Groq — Groq returns only boundaries (start/end line), title, summary.
    3. Slice the original lines in Python to build segment text.
    Returns a list of TopicSegments in chronological order.
    """
    lines, numbered_transcript = _number_lines(transcript)

    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model=groq_model,
        messages=[
            {"role": "system", "content": GROQ_SYSTEM_PROMPT},
            {"role": "user",   "content": numbered_transcript},
        ],
        temperature=0.2,
        max_tokens=4096,
    )

    data = _parse_groq_response(response.choices[0].message.content)

    segments = []
    for i, item in enumerate(data):
        start = item.get("start_line", 0)
        end   = item.get("end_line", start)
        segments.append(TopicSegment(
            index=i,
            title=item.get("title", f"Topic {i + 1}"),
            summary=item.get("summary", ""),
            text=_slice_lines(lines, start, end),
            start_line=start,
            end_line=end,
        ))

    return segments