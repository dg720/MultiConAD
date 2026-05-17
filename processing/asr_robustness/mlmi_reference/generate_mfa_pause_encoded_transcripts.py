"""Generate pause-encoded transcripts from MFA utterance-level TextGrids.

This consumes the segmented MFA corpus/alignment artifacts and reconstructs one
participant transcript per ADReSS subject. Pause punctuation comes from MFA word
tier silence intervals inside participant utterance clips, plus optional CHAT
timestamp gaps between adjacent participant utterances.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from mlmi_thesis.paths import repo_path


SILENCE_LABELS = {"", "sil", "sp", "<eps>"}


def bucket_duration(duration_seconds: float, medium_threshold: float, long_threshold: float) -> str | None:
    if duration_seconds >= long_threshold:
        return "long"
    if duration_seconds >= medium_threshold:
        return "medium"
    if duration_seconds > 0:
        return "short"
    return None


def parse_textgrid_intervals(path: Path) -> list[tuple[float, float, str]]:
    intervals: list[tuple[float, float, str]] = []
    current_start: float | None = None
    current_end: float | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("xmin ="):
            current_start = float(line.split("=", 1)[1].strip())
        elif line.startswith("xmax ="):
            current_end = float(line.split("=", 1)[1].strip())
        elif line.startswith("text =") and current_start is not None and current_end is not None:
            label = line.split("=", 1)[1].strip().strip('"')
            intervals.append((current_start, current_end, label))
            current_start = None
            current_end = None
    return intervals


def build_adjacent_segment_pairs(manifest: pd.DataFrame) -> set[tuple[str, str]]:
    """Return adjacent participant segment pairs from manifest ordering.

    If the manifest contains a main-tier adjacency flag from the segmentation
    step, use it. Otherwise fall back to adjacent participant segments sorted by
    subject and start time.
    """

    for column in ("is_adjacent_main", "adjacent_main"):
        if column in manifest.columns:
            rows = manifest.sort_values(["id", "start_ms"])
            pairs: set[tuple[str, str]] = set()
            previous_by_id: dict[str, str] = {}
            for _, row in rows.iterrows():
                sid = str(row["id"]).upper()
                segment_id = str(row["segment_id"])
                if bool(row[column]) and sid in previous_by_id:
                    pairs.add((previous_by_id[sid], segment_id))
                previous_by_id[sid] = segment_id
            return pairs

    pairs = set()
    for _, group in manifest.sort_values(["id", "start_ms"]).groupby("id"):
        segment_ids = [str(value) for value in group["segment_id"].tolist()]
        pairs.update(zip(segment_ids, segment_ids[1:]))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(repo_path("figures", "pause_analysis", "mfa_utterance_segments", "csv", "mfa_segment_manifest.csv")),
    )
    parser.add_argument(
        "--train-textgrid-root",
        default=str(repo_path("data", "derived", "mfa_utterance_segments", "aligned", "train")),
    )
    parser.add_argument(
        "--test-textgrid-root",
        default=str(repo_path("data", "derived", "mfa_utterance_segments", "aligned", "test")),
    )
    parser.add_argument(
        "--output-root",
        default=str(repo_path("data", "derived", "pause_encoding_mfa_utterance_cleaned")),
    )
    parser.add_argument(
        "--summary-path",
        default=str(
            repo_path(
                "figures",
                "pause_analysis",
                "mfa_utterance_segments_cleaned",
                "csv",
                "pause_encoding_transcript_summary.csv",
            )
        ),
    )
    parser.add_argument("--medium-threshold", type=float, default=0.5)
    parser.add_argument("--long-threshold", type=float, default=2.0)
    parser.add_argument(
        "--gap-scope",
        choices=("adjacent-main", "participant-only", "none"),
        default="adjacent-main",
        help="Which between-utterance timestamp gaps to add after MFA within-utterance silences.",
    )
    parser.add_argument(
        "--include-edge-silences",
        action="store_true",
        help="Keep leading/trailing MFA silences inside each utterance clip. Default removes them for cleaner text.",
    )
    return parser.parse_args()


def pause_token(duration_seconds: float, medium_threshold: float, long_threshold: float) -> str | None:
    bucket = bucket_duration(duration_seconds, medium_threshold, long_threshold)
    if bucket == "short":
        return ","
    if bucket == "medium":
        return "."
    if bucket == "long":
        return "..."
    return None


def segment_id_from_textgrid(path: Path) -> str:
    return re.sub(r"\.TextGrid$", "", path.name, flags=re.IGNORECASE)


def load_textgrid_map(root: Path) -> dict[str, Path]:
    return {segment_id_from_textgrid(path): path for path in root.rglob("*.TextGrid")}


def lab_tokens(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").split()


def is_pause_marker(token: str) -> bool:
    return token in {",", ".", "..."}


def stronger_pause(left: str, right: str) -> str:
    rank = {",": 1, ".": 2, "...": 3}
    return left if rank[left] >= rank[right] else right


def append_token(tokens: list[str], token: str) -> None:
    if not token or token == "<unk>":
        return
    if tokens and is_pause_marker(tokens[-1]) and is_pause_marker(token):
        tokens[-1] = stronger_pause(tokens[-1], token)
        return
    tokens.append(token)


def textgrid_tokens(
    path: Path,
    medium_threshold: float,
    long_threshold: float,
    include_edge_silences: bool,
) -> tuple[list[str], dict[str, int]]:
    tokens: list[str] = []
    counts = {"short": 0, "medium": 0, "long": 0}
    intervals = parse_textgrid_intervals(path)
    if not include_edge_silences:
        word_indices = [idx for idx, (_, _, label) in enumerate(intervals) if label.strip().lower() not in SILENCE_LABELS]
        if not word_indices:
            return [], counts
        first_word = min(word_indices)
        last_word = max(word_indices)
    else:
        first_word = 0
        last_word = len(intervals) - 1

    for index, (start, end, label) in enumerate(intervals):
        label = label.strip().lower()
        duration = end - start
        if label in SILENCE_LABELS:
            if index < first_word or index > last_word:
                continue
            token = pause_token(duration, medium_threshold, long_threshold)
            if token:
                before_len = len(tokens)
                append_token(tokens, token)
                if len(tokens) == before_len:
                    continue
                if token == ",":
                    counts["short"] += 1
                elif token == ".":
                    counts["medium"] += 1
                else:
                    counts["long"] += 1
            continue
        append_token(tokens, label)
    return tokens, counts


def count_encoded_pauses(text: str) -> dict[str, int]:
    return {
        "short": text.count(","),
        "medium": len(re.findall(r"(?<!\.)\.(?!\.)", text)),
        "long": text.count("..."),
    }


def trim_edge_pause_markers(tokens: list[str]) -> list[str]:
    while tokens and is_pause_marker(tokens[0]):
        tokens = tokens[1:]
    while tokens and is_pause_marker(tokens[-1]):
        tokens = tokens[:-1]
    return tokens


def build_transcripts_for_split(
    manifest: pd.DataFrame,
    split: str,
    textgrid_root: Path,
    output_dir: Path,
    medium_threshold: float,
    long_threshold: float,
    gap_scope: str,
    include_edge_silences: bool,
) -> list[dict[str, int | str | bool]]:
    split_manifest = manifest[manifest["split"] == split].copy()
    split_manifest["start_ms"] = pd.to_numeric(split_manifest["start_ms"], errors="coerce")
    split_manifest["end_ms"] = pd.to_numeric(split_manifest["end_ms"], errors="coerce")
    split_manifest = split_manifest.dropna(subset=["start_ms", "end_ms"])
    textgrid_map = load_textgrid_map(textgrid_root)

    adjacent_pairs: set[tuple[str, str]] = set()
    if gap_scope == "adjacent-main":
        adjacent_pairs = build_adjacent_segment_pairs(split_manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | str | bool]] = []
    for sid, group in split_manifest.sort_values(["id", "start_ms"]).groupby("id"):
        sid = str(sid).upper()
        transcript_tokens: list[str] = []
        aligned_segments = 0
        fallback_segments = 0
        previous_end_ms: float | None = None
        previous_segment_id: str | None = None
        group_name = str(group.iloc[0]["group"])
        class_dir = str(group.iloc[0]["class_dir"])

        for _, segment in group.iterrows():
            segment_id = str(segment["segment_id"])
            start_ms = float(segment["start_ms"])
            end_ms = float(segment["end_ms"])

            add_gap = False
            if previous_end_ms is not None and gap_scope != "none":
                add_gap = gap_scope == "participant-only"
                if gap_scope == "adjacent-main" and previous_segment_id is not None:
                    add_gap = (previous_segment_id, segment_id) in adjacent_pairs
            if add_gap and previous_end_ms is not None:
                token = pause_token((start_ms - previous_end_ms) / 1000.0, medium_threshold, long_threshold)
                if token:
                    append_token(transcript_tokens, token)

            textgrid_path = textgrid_map.get(segment_id)
            if textgrid_path:
                segment_tokens, _ = textgrid_tokens(
                    textgrid_path,
                    medium_threshold,
                    long_threshold,
                    include_edge_silences,
                )
                aligned_segments += 1
            else:
                segment_tokens = [token for token in lab_tokens(Path(segment["lab_path"])) if token != "<unk>"]
                fallback_segments += 1
            for token in segment_tokens:
                append_token(transcript_tokens, token)
            previous_end_ms = end_ms
            previous_segment_id = segment_id

        transcript_tokens = trim_edge_pause_markers(transcript_tokens)
        text = " ".join(transcript_tokens)
        text = re.sub(r"\s+", " ", text).strip()
        (output_dir / f"{sid}.txt").write_text(text + "\n", encoding="utf-8")
        counts = count_encoded_pauses(text)
        rows.append(
            {
                "id": sid,
                "split": split,
                "class_dir": class_dir,
                "group": group_name,
                "tokens": len(text.split()),
                "short": counts["short"],
                "medium": counts["medium"],
                "long": counts["long"],
                "dot_count": counts["medium"] + (3 * counts["long"]),
                "segments": len(group),
                "aligned_segments": aligned_segments,
                "fallback_segments": fallback_segments,
                "gap_scope": gap_scope,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    output_root = Path(args.output_root)
    rows = []
    rows.extend(
        build_transcripts_for_split(
            manifest,
            "train",
            Path(args.train_textgrid_root),
            output_root / "train",
            args.medium_threshold,
            args.long_threshold,
            args.gap_scope,
            args.include_edge_silences,
        )
    )
    rows.extend(
        build_transcripts_for_split(
            manifest,
            "test",
            Path(args.test_textgrid_root),
            output_root / "test",
            args.medium_threshold,
            args.long_threshold,
            args.gap_scope,
            args.include_edge_silences,
        )
    )

    summary = pd.DataFrame(rows)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {len(summary[summary['split'] == 'train'])} train transcripts to {output_root / 'train'}")
    print(f"Wrote {len(summary[summary['split'] == 'test'])} test transcripts to {output_root / 'test'}")
    print(f"Wrote summary to {summary_path}")
    print(summary.groupby(["split", "group"])[["medium", "long", "tokens", "fallback_segments"]].agg(["mean", "max"]).round(2))


if __name__ == "__main__":
    main()
