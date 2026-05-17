"""Compute transcript-level WER for ASR output roots."""

from __future__ import annotations

import argparse
import csv
import json
import string
from pathlib import Path

from mlmi_thesis.paths import PATHS


PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-root", required=True, help="ASR root with condition subdirectories.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "both_speakers_no_pause",
            "single_speaker_no_pause",
            "single_speaker_pause_encoded",
        ],
    )
    parser.add_argument("--train-ref-dir", default=str(PATHS["manual_train_transcripts"]))
    parser.add_argument("--test-ref-dir", default=str(PATHS["manual_test_transcripts"]))
    parser.add_argument("--normalization", choices=["liu", "raw"], default="liu")
    parser.add_argument("--output-path")
    return parser.parse_args()


def normalize_text(text: str, normalization: str) -> list[str]:
    cleaned = " ".join(str(text).split()).strip()
    if normalization == "liu":
        cleaned = cleaned.lower().translate(PUNCT_TRANSLATION)
    return [token for token in cleaned.split() if token]


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
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + 1,
                )

    i = len(reference)
    j = len(hypothesis)
    substitutions = 0
    deletions = 0
    insertions = 0
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


def score_dir(hyp_dir: Path, ref_dir: Path, normalization: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    totals = {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0, "files": 0}
    for ref_path in sorted(ref_dir.glob("*.txt")):
        hyp_path = hyp_dir / ref_path.name
        if not hyp_path.exists():
            continue
        reference = normalize_text(ref_path.read_text(encoding="utf-8"), normalization)
        hypothesis = normalize_text(hyp_path.read_text(encoding="utf-8"), normalization)
        subs, dels, ins = levenshtein(reference, hypothesis)
        ref_words = len(reference)
        wer = (subs + dels + ins) / ref_words if ref_words else 0.0
        rows.append(
            {
                "subject_id": ref_path.stem.upper(),
                "reference_words": ref_words,
                "hypothesis_words": len(hypothesis),
                "substitutions": subs,
                "deletions": dels,
                "insertions": ins,
                "wer": wer,
            }
        )
        totals["substitutions"] += subs
        totals["deletions"] += dels
        totals["insertions"] += ins
        totals["reference_words"] += ref_words
        totals["files"] += 1
    totals["wer"] = (
        (totals["substitutions"] + totals["deletions"] + totals["insertions"]) / totals["reference_words"]
        if totals["reference_words"]
        else 0.0
    )
    return rows, totals


def main() -> None:
    args = parse_args()
    asr_root = Path(args.asr_root)
    train_ref_dir = Path(args.train_ref_dir)
    test_ref_dir = Path(args.test_ref_dir)

    summary_rows: list[dict[str, object]] = []
    detail: dict[str, dict[str, list[dict[str, object]]]] = {}

    for condition in args.conditions:
        condition_detail: dict[str, list[dict[str, object]]] = {}
        for split, ref_dir in (("train", train_ref_dir), ("test", test_ref_dir)):
            hyp_dir = asr_root / condition / split
            rows, totals = score_dir(hyp_dir, ref_dir, args.normalization)
            condition_detail[split] = rows
            summary_rows.append(
                {
                    "condition": condition,
                    "split": split,
                    **totals,
                }
            )
        detail[condition] = condition_detail

    output = {
        "asr_root": str(asr_root),
        "train_ref_dir": str(train_ref_dir),
        "test_ref_dir": str(test_ref_dir),
        "normalization": args.normalization,
        "summary": summary_rows,
        "detail": detail,
    }

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = asr_root / "wer_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition",
                "split",
                "files",
                "reference_words",
                "substitutions",
                "deletions",
                "insertions",
                "wer",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(json.dumps({"output_path": str(output_path), "csv_path": str(csv_path), "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
