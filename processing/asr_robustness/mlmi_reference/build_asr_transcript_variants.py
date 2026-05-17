"""Build Table 5.8 ASR transcript variants from Whisper JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import string

import pandas as pd

from mlmi_thesis.paths import PATHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=str(PATHS["asr_whisper_large_v3_root"]))
    parser.add_argument("--medium-threshold", type=float, default=0.5)
    parser.add_argument("--long-threshold", type=float, default=2.0)
    parser.add_argument("--summary-path")
    parser.add_argument(
        "--text-style",
        choices=("liu", "katherine"),
        default="liu",
        help="Render ASR text using Liu-style normalization or Katherine-style raw ASR text.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=(
            "both_speakers_no_pause",
            "both_speakers_pause_encoded",
            "single_speaker_no_pause",
            "single_speaker_pause_encoded",
        ),
        default=(
            "both_speakers_no_pause",
            "both_speakers_pause_encoded",
            "single_speaker_no_pause",
            "single_speaker_pause_encoded",
        ),
        help="Subset of transcript variants to build.",
    )
    return parser.parse_args()


def clean_spacing(text: str) -> str:
    return " ".join(str(text).split()).strip()


PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def normalize_liu_text(text: str) -> str:
    return clean_spacing(str(text).lower().translate(PUNCT_TRANSLATION))


def normalize_katherine_text(text: str) -> str:
    return clean_spacing(str(text))


def normalize_text(text: str, text_style: str) -> str:
    if text_style == "liu":
        return normalize_liu_text(text)
    if text_style == "katherine":
        return normalize_katherine_text(text)
    raise ValueError(f"Unknown text style: {text_style}")


def pause_token_for_gap(gap_seconds: float, medium_threshold: float, long_threshold: float) -> str | None:
    if gap_seconds >= long_threshold:
        return "..."
    if gap_seconds >= medium_threshold:
        return "."
    return None


def render_no_pause_text(payload: dict, text_style: str) -> str:
    transcription = payload.get("transcription", {})
    return normalize_text(transcription.get("text", ""), text_style)


def render_from_chunks(
    chunks: list[dict],
    medium_threshold: float,
    long_threshold: float,
    encode_pauses: bool,
    text_style: str,
) -> str:
    if text_style == "katherine":
        return render_from_chunks_katherine(chunks, medium_threshold, long_threshold, encode_pauses)

    parts: list[str] = []
    previous_end: float | None = None
    for chunk in chunks:
        text = normalize_text(chunk.get("text", ""), text_style)
        if not text:
            continue
        start = chunk.get("start")
        if encode_pauses and previous_end is not None and start is not None:
            token = pause_token_for_gap(float(start) - previous_end, medium_threshold, long_threshold)
            if token:
                parts.append(token)
        parts.append(text)
        end = chunk.get("end")
        previous_end = float(end) if end is not None else previous_end
    return clean_spacing(" ".join(parts))


def render_from_chunks_katherine(
    chunks: list[dict],
    medium_threshold: float,
    long_threshold: float,
    encode_pauses: bool,
) -> str:
    final_text = ""
    last_end = 0.0
    for chunk in chunks:
        text = normalize_katherine_text(chunk.get("text", ""))
        if not text:
            continue
        start = chunk.get("start")
        if encode_pauses and start is not None:
            gap = float(start) - last_end
            if gap >= long_threshold:
                final_text += " ... "
            elif gap >= medium_threshold:
                final_text += " . "
        final_text += text
        end = chunk.get("end")
        if end is not None:
            last_end = float(end)
    return final_text.strip()


def count_medium(text: str) -> int:
    return text.count(" . ")


def process_json_dir(
    input_dir: Path,
    output_dir: Path,
    transcript_condition: str,
    medium_threshold: float,
    long_threshold: float,
    encode_pauses: bool,
    text_style: str,
) -> list[dict[str, str | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    for json_path in sorted(input_dir.glob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        subject_id = payload["subject_id"]
        if encode_pauses:
            chunks = payload.get("transcription", {}).get("chunks", [])
            rendered = render_from_chunks(chunks, medium_threshold, long_threshold, encode_pauses, text_style)
        else:
            rendered = render_no_pause_text(payload, text_style)
        (output_dir / f"{subject_id}.txt").write_text(rendered + "\n", encoding="utf-8")
        rows.append(
            {
                "split": payload["split"],
                "subject_id": subject_id,
                "transcript_condition": transcript_condition,
                "tokens": len(rendered.split()),
                "short": rendered.count(","),
                "medium": count_medium(rendered),
                "long": rendered.count("..."),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    rows: list[dict[str, str | int]] = []
    summary_path = Path(args.summary_path) if args.summary_path else root / "variant_summary.csv"
    conditions = set(args.conditions)

    both_root = root / "raw_both_speakers"
    single_root = root / "raw_single_speaker"

    for split in ("train", "test"):
        if "both_speakers_no_pause" in conditions:
            rows.extend(
                process_json_dir(
                    both_root / split,
                    root / "both_speakers_no_pause" / split,
                    "asr_both_speakers_no_pause",
                    args.medium_threshold,
                    args.long_threshold,
                    encode_pauses=False,
                    text_style=args.text_style,
                )
            )
        if "both_speakers_pause_encoded" in conditions:
            rows.extend(
                process_json_dir(
                    both_root / split,
                    root / "both_speakers_pause_encoded" / split,
                    "asr_both_speakers_pause_encoded",
                    args.medium_threshold,
                    args.long_threshold,
                    encode_pauses=True,
                    text_style=args.text_style,
                )
            )
        if "single_speaker_no_pause" in conditions:
            rows.extend(
                process_json_dir(
                    single_root / split,
                    root / "single_speaker_no_pause" / split,
                    "asr_single_speaker_no_pause",
                    args.medium_threshold,
                    args.long_threshold,
                    encode_pauses=False,
                    text_style=args.text_style,
                )
            )
        if "single_speaker_pause_encoded" in conditions:
            rows.extend(
                process_json_dir(
                    single_root / split,
                    root / "single_speaker_pause_encoded" / split,
                    "asr_single_speaker_pause_encoded",
                    args.medium_threshold,
                    args.long_threshold,
                    encode_pauses=True,
                    text_style=args.text_style,
                )
            )

    summary = pd.DataFrame(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote ASR transcript variants to {root}")
    print(f"Wrote summary CSV to {summary_path}")


if __name__ == "__main__":
    main()
