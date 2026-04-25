"""
core.py — Transcription Segmentation Logic
============================================
Groq only returns segment boundaries (start/end line numbers) + title + summary.
Python slices the original transcript lines to produce the segment text.

If Groq leaves gaps:
- Each gap is sent back to Groq WITH context (lines before and after)
- Groq decides if the gap belongs to the previous segment, next segment, or is a new topic
- All gaps are resolved in a single second request
- Maximum 2 Groq calls total

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
# Groq prompts
# ─────────────────────────────────────────────

SEGMENTATION_PROMPT = """\
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
- Every single line must be covered. Segments must be contiguous with no gaps.
- The first segment must start at line 0.
- The last segment must end at the last line of the transcript.
- start_line and end_line are both inclusive.
- Produce as many segments as needed — do not over-merge unrelated topics.
- The transcript may be in Arabic. Respond with titles and summaries in the same language as the transcript.
"""


GAP_RESOLUTION_PROMPT = """\
You are an expert transcript analyst. Several gaps were found in a transcript segmentation.
For each gap below, you are given:
- The last few lines of the PREVIOUS segment
- The GAP lines that were not assigned to any segment
- The first few lines of the NEXT segment

For each gap decide:
- If the gap continues the PREVIOUS segment → set "belongs_to": "previous"
- If the gap starts the NEXT segment → set "belongs_to": "next"
- If the gap is a separate topic → set "belongs_to": "new" and provide a title and summary

Return ONLY a valid JSON array (no markdown fences, no commentary), one entry per gap:
[
  {
    "gap_id": 0,
    "belongs_to": "previous" | "next" | "new",
    "title": "..." ,
    "summary": "..."
  }
]

Rules:
- "title" and "summary" are only required when belongs_to is "new". Set them to "" otherwise.
- The transcript may be in Arabic. Respond with titles and summaries in the same language as the transcript.
"""


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _number_lines(transcript: str) -> tuple[list[str], str]:
    lines = [l for l in transcript.splitlines() if l.strip()]
    numbered = "\n".join(f"{i}: {line}" for i, line in enumerate(lines))
    return lines, numbered


def _parse_json(raw: str) -> list[dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def _slice_lines(lines: list[str], start: int, end: int) -> str:
    start = max(0, start)
    end   = min(len(lines) - 1, end)
    return "\n".join(lines[start : end + 1])


def _find_gaps(data: list[dict], total_lines: int) -> list[dict]:
    """
    Return a list of gaps — line ranges not covered by any segment.
    Each gap: { gap_id, start_line, end_line }
    """
    data   = sorted(data, key=lambda x: x["start_line"])
    gaps   = []
    gap_id = 0

    if data[0]["start_line"] > 0:
        gaps.append({"gap_id": gap_id, "start_line": 0, "end_line": data[0]["start_line"] - 1})
        gap_id += 1

    for i in range(len(data) - 1):
        current_end = data[i]["end_line"]
        next_start  = data[i + 1]["start_line"]
        if next_start > current_end + 1:
            gaps.append({"gap_id": gap_id, "start_line": current_end + 1, "end_line": next_start - 1})
            gap_id += 1

    if data[-1]["end_line"] < total_lines - 1:
        gaps.append({"gap_id": gap_id, "start_line": data[-1]["end_line"] + 1, "end_line": total_lines - 1})
        gap_id += 1

    return gaps


def _build_gap_context_message(
    gaps: list[dict],
    data: list[dict],
    lines: list[str],
    context_lines: int = 3,
) -> str:
    """
    Build the user message for the gap resolution request.
    Each gap is shown with the last N lines of the previous segment
    and the first N lines of the next segment.
    """
    data   = sorted(data, key=lambda x: x["start_line"])
    parts  = []

    for gap in gaps:
        gap_start = gap["start_line"]
        gap_end   = gap["end_line"]

        # Find previous segment (segment that ends just before this gap)
        prev_seg = next(
            (s for s in reversed(data) if s["end_line"] < gap_start), None
        )
        # Find next segment (segment that starts just after this gap)
        next_seg = next(
            (s for s in data if s["start_line"] > gap_end), None
        )

        block = [f"--- GAP {gap['gap_id']} (lines {gap_start}–{gap_end}) ---"]

        if prev_seg:
            prev_lines = _slice_lines(
                lines,
                max(prev_seg["start_line"], prev_seg["end_line"] - context_lines + 1),
                prev_seg["end_line"],
            )
            block.append(f"[PREVIOUS SEGMENT — last {context_lines} lines]\n{prev_lines}")

        gap_text = _slice_lines(lines, gap_start, gap_end)
        block.append(f"[GAP LINES]\n{gap_text}")

        if next_seg:
            next_lines = _slice_lines(
                lines,
                next_seg["start_line"],
                min(next_seg["end_line"], next_seg["start_line"] + context_lines - 1),
            )
            block.append(f"[NEXT SEGMENT — first {context_lines} lines]\n{next_lines}")

        parts.append("\n\n".join(block))

    return "\n\n" + ("=" * 60) + "\n\n".join(parts)


def _apply_gap_resolutions(
    data: list[dict],
    gaps: list[dict],
    resolutions: list[dict],
    lines: list[str],
) -> list[dict]:
    """
    Merge gap resolutions back into the segment list.
    - belongs_to "previous" → extend the previous segment's end_line
    - belongs_to "next"     → extend the next segment's start_line
    - belongs_to "new"      → insert as a new segment
    """
    data = sorted(data, key=lambda x: x["start_line"])

    # Index resolutions by gap_id
    res_by_id = {r["gap_id"]: r for r in resolutions}

    for gap in gaps:
        gap_id    = gap["gap_id"]
        gap_start = gap["start_line"]
        gap_end   = gap["end_line"]
        res       = res_by_id.get(gap_id, {"belongs_to": "new", "title": "Uncategorized", "summary": ""})
        belongs   = res.get("belongs_to", "new")

        if belongs == "previous":
            # Extend the previous segment to include the gap
            for seg in reversed(data):
                if seg["end_line"] < gap_start:
                    seg["end_line"] = gap_end
                    break

        elif belongs == "next":
            # Extend the next segment to include the gap
            for seg in data:
                if seg["start_line"] > gap_end:
                    seg["start_line"] = gap_start
                    break

        else:
            # Insert as a new standalone segment
            data.append({
                "title":      res.get("title", "Uncategorized"),
                "summary":    res.get("summary", ""),
                "start_line": gap_start,
                "end_line":   gap_end,
            })

    return sorted(data, key=lambda x: x["start_line"])


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
    2. Send to Groq — returns boundaries, titles, summaries (call 1).
    3. Detect any gaps in coverage.
    4. If gaps exist, send them back with context for resolution (call 2).
    5. Merge resolutions back and slice original lines for all segment texts.
    """
    lines, numbered_transcript = _number_lines(transcript)
    total_lines = len(lines)

    client = Groq(api_key=groq_api_key)

    # ── Call 1: full segmentation
    response = client.chat.completions.create(
        model=groq_model,
        messages=[
            {"role": "system", "content": SEGMENTATION_PROMPT},
            {"role": "user",   "content": numbered_transcript},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    data = _parse_json(response.choices[0].message.content)
    data = sorted(data, key=lambda x: x["start_line"])

    # ── Detect gaps
    gaps = _find_gaps(data, total_lines)

    # ── Call 2: resolve gaps (only if gaps exist)
    if gaps:
        gap_message = _build_gap_context_message(gaps, data, lines)
        gap_response = client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": GAP_RESOLUTION_PROMPT},
                {"role": "user",   "content": gap_message},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        resolutions = _parse_json(gap_response.choices[0].message.content)
        data = _apply_gap_resolutions(data, gaps, resolutions, lines)

    # ── Build final segments by slicing original lines
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