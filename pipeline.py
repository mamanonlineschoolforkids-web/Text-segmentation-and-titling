"""
pipeline.py — Run the segmentation pipeline directly
======================================================
HOW TO USE:
  1. Place your transcript .txt file in this folder
  2. Set GROQ_API_KEY in the .env file
  3. Edit the CONFIG block below if needed
  4. Run: python pipeline.py
"""

import os
import json
import textwrap
from pathlib import Path
from dotenv import load_dotenv

from core import run_pipeline, TopicSegment

load_dotenv(Path(__file__).parent / ".env")


# ─────────────────────────────────────────────
# CONFIG — edit these values directly
# ─────────────────────────────────────────────

TRANSCRIPT_FILE = "transcript.txt"

# Groq model to use
# Options: "llama-3.3-70b-versatile" | "llama-3.1-8b-instant" | "mixtral-8x7b-32768"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Where to save the results JSON (set to None to skip saving)
OUTPUT_FILE = "results.json"

# ─────────────────────────────────────────────


def print_results(segments: list[TopicSegment]) -> None:
    width = 80
    print("\n" + "=" * width)
    print("  SEGMENTATION RESULTS")
    print("=" * width)
    for seg in segments:
        print(f"\n{'-' * width}")
        print(f"  [{seg.index + 1}]  {seg.title.upper()}")
        print(f"{'-' * width}")
        print(f"  SUMMARY: {seg.summary}")
        print()
        wrapped = textwrap.fill(
            seg.text, width=width - 4,
            initial_indent="  ", subsequent_indent="  "
        )
        print(wrapped)
    print("\n" + "=" * width)


def save_results(segments: list[TopicSegment], output_path: str) -> None:
    data = [
        {
            "index": seg.index,
            "title": seg.title,
            "summary": seg.summary,
            "text": seg.text,
        }
        for seg in segments
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved -> {output_path}")


def main() -> None:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Add it to your .env file:\n"
            "  GROQ_API_KEY=your_key_here"
        )

    script_dir      = Path(__file__).parent
    transcript_path = script_dir / TRANSCRIPT_FILE
    if not transcript_path.exists():
        raise FileNotFoundError(
            f"Transcript not found: {transcript_path}\n"
            f"Place your .txt file in the same folder and update TRANSCRIPT_FILE."
        )

    print(f"\nReading transcript: {transcript_path}")
    transcript = transcript_path.read_text(encoding="utf-8")

    print(f"Sending to Groq ({GROQ_MODEL})...")

    segments = run_pipeline(
        transcript=transcript,
        groq_api_key=groq_api_key,
        groq_model=GROQ_MODEL,
    )

    print(f"Done. {len(segments)} topic segments produced.")
    print_results(segments)

    if OUTPUT_FILE:
        save_results(segments, str(script_dir / OUTPUT_FILE))


if __name__ == "__main__":
    main()