"""Generate pause-encoded participant transcripts from ADReSS CHAT files.

This reconstructs the pause-encoding described in Katherine Jackson's thesis:

- short pause: under 0.5 seconds, encoded as `,`
- medium pause: 0.5 to 2 seconds, encoded as `.`
- long pause: over 2 seconds, encoded as `...`

The script reads raw `.cha` files, keeps only participant (`PAR`) main-tier
utterances, saves their CHAT timestamps, strips CHAT annotations and ordinary
punctuation, then injects pause tokens from timestamp gaps between consecutive
participant segments. It can also preserve explicit CHAT pause markers such as
``(.)``, ``(..)``, ``(...)`` and ``+...`` for hybrid transcript-only encoding.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mlmi_thesis.paths import PATHS, repo_path


SHORT_PAUSE = "__SHORT_PAUSE__"
MEDIUM_PAUSE = "__MEDIUM_PAUSE__"
LONG_PAUSE = "__LONG_PAUSE__"


@dataclass
class Utterance:
    speaker: str
    text: str
    start_ms: int | None
    end_ms: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medium-threshold", type=float, default=0.5)
    parser.add_argument("--long-threshold", type=float, default=2.0)
    parser.add_argument(
        "--gap-scope",
        choices=["participant-only", "adjacent-main"],
        default="participant-only",
        help=(
            "participant-only computes gaps between consecutive PAR segments after "
            "interviewer removal; adjacent-main only counts gaps where the previous "
            "raw main tier was also PAR."
        ),
    )
    parser.add_argument(
        "--include-chat-pause-markers",
        action="store_true",
        help="Also convert visible CHAT markers like (.), (..), (...) into pause tokens.",
    )
    parser.add_argument(
        "--keep-full-stops",
        action="store_true",
        help="Preserve ordinary full stops in cleaned text; these will remain countable as medium pauses.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root containing train/ and test/ folders. Defaults to PATHS pause folders.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional summary CSV path. Defaults under figures/pause_analysis/timestamp_reference.",
    )
    parser.add_argument("--plot", action="store_true", default=True)
    return parser.parse_args()


def extract_timestamp(text: str) -> tuple[int | None, int | None]:
    match = re.search(r"\x15(\d+)_(\d+)\x15", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def parse_chat_main_tiers(path: Path) -> list[Utterance]:
    utterances: list[Utterance] = []
    current_speaker: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_speaker, current_lines
        if current_speaker is None:
            return
        text = " ".join(line.strip() for line in current_lines)
        start_ms, end_ms = extract_timestamp(text)
        utterances.append(Utterance(current_speaker, text, start_ms, end_ms))
        current_speaker = None
        current_lines = []

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw_line.startswith("*"):
            flush()
            match = re.match(r"\*(\w+):\s*(.*)", raw_line)
            if match:
                current_speaker = match.group(1)
                current_lines = [match.group(2)]
            continue

        if current_speaker is not None:
            if raw_line.startswith("%") or raw_line.startswith("@"):
                flush()
            elif raw_line.startswith("\t") or raw_line.startswith(" "):
                current_lines.append(raw_line)

    flush()
    return utterances


def pause_token_for_gap(gap_seconds: float, medium_threshold: float, long_threshold: float) -> str | None:
    if gap_seconds >= long_threshold:
        return LONG_PAUSE
    if gap_seconds >= medium_threshold:
        return MEDIUM_PAUSE
    if gap_seconds > 0:
        return SHORT_PAUSE
    return None


def protect_explicit_pauses(text: str) -> str:
    text = re.sub(r"\+\s*\.\s*\.\s*\.", f" {LONG_PAUSE} ", text)
    text = re.sub(r"\(\s*\.\s*\.\s*\.\s*\)", f" {LONG_PAUSE} ", text)
    text = re.sub(r"\(\s*\.\s*\.\s*\)", f" {MEDIUM_PAUSE} ", text)
    text = re.sub(r"\(\s*\.\s*\)", f" {SHORT_PAUSE} ", text)
    return text


def expand_repetitions(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        return " ".join([match.group(1)] * int(match.group(2)))

    return re.sub(r"\b([A-Za-z][\w'_-]*)\s+\[x\s+(\d+)\]", repl, text)


def clean_participant_text(
    text: str,
    include_chat_pause_markers: bool = False,
    keep_full_stops: bool = False,
) -> str:
    if include_chat_pause_markers:
        text = protect_explicit_pauses(text)
    text = re.sub(r"\x15\d+_\d+\x15", " ", text)
    text = expand_repetitions(text)

    # Keep replacement text in constructs like "word [: replacement]".
    protected = {
        SHORT_PAUSE: " shortpauseplaceholder ",
        MEDIUM_PAUSE: " mediumpauseplaceholder ",
        LONG_PAUSE: " longpauseplaceholder ",
    }
    for marker, placeholder in protected.items():
        text = text.replace(marker, placeholder)

    text = re.sub(r"\[:\s*([^\]]+)\]", r" \1 ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\b([A-Za-z]+)\(([A-Za-z]+)\)", r"\1\2", text)
    text = re.sub(r"&=\w+", " ", text)
    text = re.sub(r"&[A-Za-z]+", " ", text)
    text = re.sub(r"\bPOSTCLITIC\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bxxx\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[<>/@&]", " ", text)

    # Remove ordinary punctuation after pause markers have been protected.
    punctuation_pattern = r"[,!?;:\"“”‘’()\[\]{}]" if keep_full_stops else r"[,\.!?;:\"“”‘’()\[\]{}]"
    text = re.sub(punctuation_pattern, " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = text.replace("shortpauseplaceholder", ",")
    text = text.replace("mediumpauseplaceholder", ".")
    text = text.replace("longpauseplaceholder", "...")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_chat_file(
    path: Path,
    medium_threshold: float,
    long_threshold: float,
    gap_scope: str,
    include_chat_pause_markers: bool,
    keep_full_stops: bool,
) -> str:
    utterances = parse_chat_main_tiers(path)
    output_parts: list[str] = []
    previous_main: Utterance | None = None
    previous_participant: Utterance | None = None

    for utterance in utterances:
        if utterance.speaker == "PAR":
            previous_for_gap = previous_participant if gap_scope == "participant-only" else previous_main
            if (
                previous_for_gap is not None
                and previous_for_gap.speaker == "PAR"
                and previous_for_gap.end_ms is not None
                and utterance.start_ms is not None
            ):
                gap_seconds = (utterance.start_ms - previous_for_gap.end_ms) / 1000.0
                token = pause_token_for_gap(gap_seconds, medium_threshold, long_threshold)
                if token:
                    output_parts.append(token)

            cleaned = clean_participant_text(
                utterance.text,
                include_chat_pause_markers=include_chat_pause_markers,
                keep_full_stops=keep_full_stops,
            )
            if cleaned:
                output_parts.append(cleaned)
            previous_participant = utterance

        previous_main = utterance

    text = " ".join(output_parts)
    text = text.replace(SHORT_PAUSE, ",").replace(MEDIUM_PAUSE, ".").replace(LONG_PAUSE, "...")
    return re.sub(r"\s+", " ", text).strip()


def process_folder(
    input_dir: Path,
    output_dir: Path,
    medium_threshold: float,
    long_threshold: float,
    gap_scope: str,
    include_chat_pause_markers: bool,
    keep_full_stops: bool,
) -> list[dict[str, int | str | bool]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | str | bool]] = []
    for cha_path in sorted(input_dir.glob("*.cha")):
        encoded = encode_chat_file(
            cha_path,
            medium_threshold,
            long_threshold,
            gap_scope,
            include_chat_pause_markers,
            keep_full_stops,
        )
        out_path = output_dir / f"{cha_path.stem.upper()}.txt"
        out_path.write_text(encoded + "\n", encoding="utf-8")
        rows.append(
            {
                "id": cha_path.stem.upper(),
                "medium": len(re.findall(r"(?<!\.)\.(?!\.)", encoded)),
                "long": encoded.count("..."),
                "short": len(re.findall(r"(?<!\S),(?!\S)|,", encoded)),
                "tokens": len(encoded.split()),
                "gap_scope": gap_scope,
                "include_chat_pause_markers": include_chat_pause_markers,
                "keep_full_stops": keep_full_stops,
            }
        )
    return rows


def plot_pause_distribution(summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for split, group in summary.groupby("split"):
        ax.scatter(group["medium"], group["long"], label=split, alpha=0.8)
    ax.set_xlabel("Medium pauses")
    ax.set_ylabel("Long pauses")
    ax.set_title("Reconstructed pause encoding counts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "pause_encoding_distribution.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    train_root = PATHS["adress_train_chat"]
    test_root = PATHS["adress_test_chat"]
    if args.output_root:
        output_root = Path(args.output_root)
        train_output = output_root / "train"
        test_output = output_root / "test"
    else:
        train_output = PATHS["pause_train_transcripts"]
        test_output = PATHS["pause_test_transcripts"]

    rows: list[dict[str, int | str]] = []
    for class_dir in ("cc", "cd"):
        for row in process_folder(
            train_root / class_dir,
            train_output,
            args.medium_threshold,
            args.long_threshold,
            args.gap_scope,
            args.include_chat_pause_markers,
            args.keep_full_stops,
        ):
            row["split"] = "train"
            row["class_dir"] = class_dir
            rows.append(row)

    for row in process_folder(
        test_root,
        test_output,
        args.medium_threshold,
        args.long_threshold,
        args.gap_scope,
        args.include_chat_pause_markers,
        args.keep_full_stops,
    ):
        row["split"] = "test"
        row["class_dir"] = "test"
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path
        else repo_path("figures", "pause_analysis", "timestamp_reference", "csv", "pause_encoding_summary.csv")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    if args.plot:
        plot_pause_distribution(summary, summary_path.parent)

    print(f"Wrote {len(summary[summary.split == 'train'])} train transcripts")
    print(f"Wrote {len(summary[summary.split == 'test'])} test transcripts")
    print(f"Wrote summary to {summary_path}")
    print(summary.groupby("split")[["short", "medium", "long", "tokens"]].mean().round(2))


if __name__ == "__main__":
    main()
