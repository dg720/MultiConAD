"""Build WhisperX prompt-scrub ablations with no-pause and pause-encoded outputs.

This uses the raw WhisperX single-speaker chunk timelines so post-processing
can happen before pause encoding is rendered.
"""

from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import pandas as pd

from mlmi_thesis.paths import PATHS


PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
CONTROL_IDS = {
    "S001", "S002", "S003", "S004", "S005", "S006", "S007", "S009", "S011",
    "S012", "S013", "S015", "S016", "S017", "S018", "S019", "S020", "S021",
    "S024", "S025", "S027", "S028", "S029", "S030", "S032", "S033", "S034",
    "S035", "S036", "S038", "S039", "S040", "S041", "S043", "S048", "S049",
    "S051", "S052", "S055", "S056", "S058", "S059", "S061", "S062", "S063",
    "S064", "S067", "S068", "S070", "S071", "S072", "S073", "S076", "S077",
}
PROMPT_PHRASES = [
    "tell me what you see",
    "tell me what you see happening",
    "tell me what you see going on",
    "tell me everything that you see",
    "tell me everything that you see happening",
    "tell me everything that you see going on",
    "what do you see",
    "what do you see happening",
    "what do you see going on",
    "what do you see going on in that picture",
    "id like you to tell me",
    "i want you to tell me",
    "just tell me",
    "anything else",
    "anything else going on",
    "anything else going on in the picture",
    "anything else that you see happening",
    "can you tell me more",
    "can you see anything else",
    "is there anything else",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, help="raw_single_speaker root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-ref-dir", required=True)
    parser.add_argument("--test-ref-dir", required=True)
    parser.add_argument("--note-path", required=True)
    parser.add_argument("--medium-threshold", type=float, default=0.5)
    parser.add_argument("--long-threshold", type=float, default=2.0)
    parser.add_argument("--utterance-gap-s", type=float, default=1.0)
    parser.add_argument("--test-labels-path", default=str(PATHS["adress_test_labels"]))
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().translate(PUNCT_TRANSLATION).split()).strip()


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


PROMPT_TOKEN_PHRASES = [tokenize(phrase) for phrase in PROMPT_PHRASES]


def load_references(ref_dir: Path) -> dict[str, list[str]]:
    return {path.stem.upper(): tokenize(path.read_text(encoding="utf-8")) for path in sorted(ref_dir.glob("*.txt"))}


def load_test_label_map(path: Path) -> dict[str, int]:
    labels = pd.read_csv(path, sep=";")
    labels.columns = labels.columns.str.strip().str.lower()
    labels["id"] = labels["id"].str.strip().str.upper()
    labels["label"] = labels["label"].astype(int)
    return dict(zip(labels["id"], labels["label"]))


def load_raw_chunks(raw_root: Path, split: str, subject_id: str) -> list[dict[str, object]]:
    path = raw_root / split / f"{subject_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[dict[str, object]] = []
    for chunk in payload.get("transcription", {}).get("chunks", []):
        text = normalize_text(chunk.get("text", ""))
        if not text:
            continue
        chunks.append(
            {
                "text": text,
                "tokens": text.split(),
                "start": chunk.get("start"),
                "end": chunk.get("end"),
            }
        )
    return chunks


def levenshtein(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
    i = len(reference)
    j = len(hypothesis)
    substitutions = deletions = insertions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1]:
            i -= 1
            j -= 1
            continue
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
            continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
            continue
        insertions += 1
        j -= 1
    return substitutions, deletions, insertions


def aggregate_wer(rows: list[dict[str, object]]) -> dict[str, float]:
    def compute(subset: list[dict[str, object]]) -> float:
        subs = sum(int(r["substitutions"]) for r in subset)
        dels = sum(int(r["deletions"]) for r in subset)
        ins = sum(int(r["insertions"]) for r in subset)
        ref_words = sum(int(r["reference_words"]) for r in subset)
        return ((subs + dels + ins) / ref_words) * 100 if ref_words else 0.0

    return {
        "All": round(compute(rows), 2),
        "Healthy": round(compute([r for r in rows if r["label_name"] == "Healthy"]), 2),
        "Alzheimer": round(compute([r for r in rows if r["label_name"] == "Alzheimer"]), 2),
        "Train": round(compute([r for r in rows if r["split"] == "train"]), 2),
        "Test": round(compute([r for r in rows if r["split"] == "test"]), 2),
    }


def build_utterances(chunks: list[dict[str, object]], gap_threshold: float) -> list[list[int]]:
    if not chunks:
        return []
    utterances: list[list[int]] = []
    current = [0]
    previous_end = chunks[0].get("end")
    for idx, chunk in enumerate(chunks[1:], start=1):
        start = chunk.get("start")
        split_here = False
        if previous_end is not None and start is not None:
            split_here = float(start) - float(previous_end) >= gap_threshold
        if split_here:
            utterances.append(current)
            current = [idx]
        else:
            current.append(idx)
        previous_end = chunk.get("end")
    if current:
        utterances.append(current)
    return utterances


def flatten_chunk_tokens(chunks: list[dict[str, object]]) -> tuple[list[str], list[int]]:
    tokens: list[str] = []
    token_to_chunk: list[int] = []
    for chunk_idx, chunk in enumerate(chunks):
        for token in chunk["tokens"]:
            tokens.append(token)
            token_to_chunk.append(chunk_idx)
    return tokens, token_to_chunk


def match_phrase_spans(tokens: list[str], leading_only: bool) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    if leading_only:
        pos = 0
        changed = True
        while changed:
            changed = False
            for phrase in PROMPT_TOKEN_PHRASES:
                end = pos + len(phrase)
                if end <= len(tokens) and tokens[pos:end] == phrase:
                    spans.append((pos, end))
                    pos = end
                    changed = True
                    break
        return spans

    i = 0
    while i < len(tokens):
        matched = False
        for phrase in PROMPT_TOKEN_PHRASES:
            end = i + len(phrase)
            if end <= len(tokens) and tokens[i:end] == phrase:
                spans.append((i, end))
                i = end
                matched = True
                break
        if not matched:
            i += 1
    return spans


def chunks_after_phrase_drop(chunks: list[dict[str, object]], leading_only: bool) -> list[dict[str, object]]:
    flat_tokens, token_to_chunk = flatten_chunk_tokens(chunks)
    spans = match_phrase_spans(flat_tokens, leading_only=leading_only)
    if not spans:
        return list(chunks)
    drop_chunk_indices: set[int] = set()
    for start, end in spans:
        for token_pos in range(start, end):
            drop_chunk_indices.add(token_to_chunk[token_pos])
    return [chunk for idx, chunk in enumerate(chunks) if idx not in drop_chunk_indices]


def utterance_contains_prompt(chunks: list[dict[str, object]], utterance: list[int]) -> bool:
    tokens: list[str] = []
    for chunk_idx in utterance:
        tokens.extend(chunks[chunk_idx]["tokens"])
    for phrase in PROMPT_TOKEN_PHRASES:
        max_i = len(tokens) - len(phrase) + 1
        for i in range(max(0, max_i)):
            if tokens[i : i + len(phrase)] == phrase:
                return True
    return False


def chunks_after_aggressive_utterance_drop(
    chunks: list[dict[str, object]],
    gap_threshold: float,
) -> list[dict[str, object]]:
    keep_indices: list[int] = []
    for utterance in build_utterances(chunks, gap_threshold):
        if utterance_contains_prompt(chunks, utterance):
            continue
        keep_indices.extend(utterance)
    return [chunks[idx] for idx in keep_indices]


def pause_token_for_gap(gap_seconds: float, medium_threshold: float, long_threshold: float) -> str | None:
    if gap_seconds >= long_threshold:
        return "..."
    if gap_seconds >= medium_threshold:
        return "."
    return None


def render_no_pause(chunks: list[dict[str, object]]) -> str:
    return " ".join(chunk["text"] for chunk in chunks if chunk["text"]).strip()


def render_pause_encoded(
    chunks: list[dict[str, object]],
    medium_threshold: float,
    long_threshold: float,
) -> str:
    parts: list[str] = []
    previous_end: float | None = None
    for chunk in chunks:
        text = chunk["text"]
        if not text:
            continue
        start = chunk.get("start")
        if previous_end is not None and start is not None:
            token = pause_token_for_gap(float(start) - previous_end, medium_threshold, long_threshold)
            if token:
                parts.append(token)
        parts.append(text)
        end = chunk.get("end")
        previous_end = float(end) if end is not None else previous_end
    return " ".join(parts).strip()


def pause_counts(text: str) -> tuple[int, int, int]:
    short = text.count(",")
    long = text.count("...")
    medium = text.replace("...", "").count(".")
    return short, medium, long


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    train_refs = load_references(Path(args.train_ref_dir))
    test_refs = load_references(Path(args.test_ref_dir))
    test_label_map = load_test_label_map(Path(args.test_labels_path))

    variants = {
        "baseline_existing": lambda chunks: list(chunks),
        "keyword_leading": lambda chunks: chunks_after_phrase_drop(chunks, leading_only=True),
        "keyword_anywhere": lambda chunks: chunks_after_phrase_drop(chunks, leading_only=False),
        "aggressive_utterance_drop": lambda chunks: chunks_after_aggressive_utterance_drop(chunks, args.utterance_gap_s),
    }

    for variant in variants:
        for suffix in ("no_pause", "pause_encoded"):
            for split in ("train", "test"):
                (output_root / f"{variant}_{suffix}" / split).mkdir(parents=True, exist_ok=True)

    wer_rows: list[dict[str, object]] = []
    pause_rows: list[dict[str, object]] = []

    for split in ("train", "test"):
        refs = train_refs if split == "train" else test_refs
        for json_path in sorted((raw_root / split).glob("*.json")):
            subject_id = json_path.stem.upper()
            chunks = load_raw_chunks(raw_root, split, subject_id)
            reference = refs.get(subject_id)
            if reference is None:
                continue
            if split == "train":
                label_name = "Healthy" if subject_id in CONTROL_IDS else "Alzheimer"
            else:
                label_name = "Healthy" if test_label_map.get(subject_id) == 0 else "Alzheimer"
            for variant, transform in variants.items():
                kept_chunks = transform(chunks)
                no_pause_text = render_no_pause(kept_chunks)
                pause_text = render_pause_encoded(kept_chunks, args.medium_threshold, args.long_threshold)

                (output_root / f"{variant}_no_pause" / split / f"{subject_id}.txt").write_text(
                    no_pause_text + "\n",
                    encoding="utf-8",
                )
                (output_root / f"{variant}_pause_encoded" / split / f"{subject_id}.txt").write_text(
                    pause_text + "\n",
                    encoding="utf-8",
                )

                hypothesis = no_pause_text.split()
                subs, dels, ins = levenshtein(reference, hypothesis)
                wer_rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "subject_id": subject_id,
                        "label_name": label_name,
                        "reference_words": len(reference),
                        "substitutions": subs,
                        "deletions": dels,
                        "insertions": ins,
                    }
                )

                short, medium, long = pause_counts(pause_text)
                pause_rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "subject_id": subject_id,
                        "label_name": label_name,
                        "short": short,
                        "medium": medium,
                        "long": long,
                    }
                )

    wer_summary = pd.DataFrame(
        [{"Variant": variant, **aggregate_wer([r for r in wer_rows if r["variant"] == variant])} for variant in variants]
    )
    pause_df = pd.DataFrame(pause_rows)
    pause_summary = (
        pause_df.groupby(["variant", "label_name"], as_index=False)
        .agg(
            transcripts=("subject_id", "count"),
            mean_short=("short", "mean"),
            mean_medium=("medium", "mean"),
            mean_long=("long", "mean"),
            median_short=("short", "median"),
            median_medium=("medium", "median"),
            median_long=("long", "median"),
        )
        .sort_values(["variant", "label_name"])
    )

    wer_summary.to_csv(output_root / "wer_summary.csv", index=False)
    pause_summary.to_csv(output_root / "pause_summary.csv", index=False)
    (output_root / "wer_detail.json").write_text(json.dumps(wer_rows, indent=2), encoding="utf-8")

    lines = [
        "Table: WhisperX prompt-scrub pause ablation",
        "Category: evaluation",
        "Task: compare post-processing methods when scrubbing happens before pause encoding",
        "",
        f"Raw root: `{raw_root}`",
        "Order:",
        "- `scrub -> pause encode`",
        "- this preserves only pauses between retained spans and avoids keeping artificial interviewer-induced pause markers",
        "",
        "Methods:",
        "- `baseline_existing`: keep all retained single-speaker chunks",
        "- `keyword_leading`: drop chunks overlapping leading prompt phrase spans",
        "- `keyword_anywhere`: drop chunks overlapping prompt phrase spans anywhere",
        f"- `aggressive_utterance_drop`: split on gaps >= {args.utterance_gap_s:.1f}s and drop whole prompt-containing utterances",
        "",
        "WER on no-pause renders:",
        "| Variant | All | Healthy | Alzheimer | Train | Test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in wer_summary.to_dict("records"):
        lines.append(
            f"| {record['Variant']} | {record['All']:.2f} | {record['Healthy']:.2f} | {record['Alzheimer']:.2f} | {record['Train']:.2f} | {record['Test']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Pause count summary on pause-encoded renders:",
            "| Variant | Group | Mean Medium | Mean Long | Median Medium | Median Long |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for record in pause_summary.to_dict("records"):
        lines.append(
            f"| {record['variant']} | {record['label_name']} | {record['mean_medium']:.2f} | {record['mean_long']:.2f} | {record['median_medium']:.2f} | {record['median_long']:.2f} |"
        )
    Path(args.note_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(wer_summary.to_string(index=False))
    print(pause_summary.to_string(index=False))


if __name__ == "__main__":
    main()
