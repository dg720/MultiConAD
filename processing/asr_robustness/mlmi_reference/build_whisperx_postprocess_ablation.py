"""Build participant-only WhisperX transcript ablations from both-speaker outputs.

This script keeps ASR fixed and varies only post-processing:

- diarize full audio and keep the longest speaker
- diarize full audio and keep SPEAKER_00
- optionally strip common interviewer prompt phrases from the resulting text

It writes separate participant-only transcript directories plus a compact WER
summary so the post-processing variants can be compared directly.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import string
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

from mlmi_thesis.paths import PATHS, repo_path


TARGET_SAMPLE_RATE = 16_000
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
    "look at that picture and tell me what you see going on",
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
    parser.add_argument("--asr-root", required=True, help="WhisperX ASR root with raw_both_speakers.")
    parser.add_argument("--output-root", required=True, help="Destination for post-processing variants.")
    parser.add_argument("--pyannote-model-id", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--hf-token-env", default="HUGGINGFACE_HUB_TOKEN")
    parser.add_argument("--cache-dir", default=str(PATHS["hf_cache"]))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-ref-dir", default=str(repo_path("data", "derived", "no_pause_clean", "train")))
    parser.add_argument("--test-ref-dir", default=str(repo_path("data", "derived", "no_pause_clean", "test")))
    parser.add_argument("--summary-path")
    parser.add_argument("--note-path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int)
    return parser.parse_args()


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    waveform = audio.T
    if waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0, keepdims=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = resample_poly(waveform, TARGET_SAMPLE_RATE, sample_rate, axis=1)
        sample_rate = TARGET_SAMPLE_RATE
    return torch.from_numpy(np.ascontiguousarray(waveform)), sample_rate


def resolve_cached_hf_snapshot(cache_dir: str | Path, repo_id: str) -> str:
    cache_root = Path(cache_dir)
    repo_dir = cache_root / "hub" / f"models--{repo_id.replace('/', '--')}"
    ref_path = repo_dir / "refs" / "main"
    if ref_path.exists():
        snapshot_dir = repo_dir / "snapshots" / ref_path.read_text(encoding="utf-8").strip()
        if snapshot_dir.exists():
            config_path = snapshot_dir / "config.yaml"
            if config_path.exists():
                return str(config_path)
            return str(snapshot_dir)
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            config_path = snapshots[-1] / "config.yaml"
            if config_path.exists():
                return str(config_path)
            return str(snapshots[-1])
    return repo_id


def resolve_cached_hf_model_checkpoint(cache_dir: str | Path, repo_id: str) -> str:
    cache_root = Path(cache_dir)
    repo_dir = cache_root / "hub" / f"models--{repo_id.replace('/', '--')}"
    ref_path = repo_dir / "refs" / "main"
    if ref_path.exists():
        snapshot_dir = repo_dir / "snapshots" / ref_path.read_text(encoding="utf-8").strip()
        checkpoint_path = snapshot_dir / "pytorch_model.bin"
        if checkpoint_path.exists():
            return str(checkpoint_path)
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            checkpoint_path = snapshots[-1] / "pytorch_model.bin"
            if checkpoint_path.exists():
                return str(checkpoint_path)
    return repo_id


def prepare_pyannote_pipeline_source(cache_dir: str | Path, repo_id: str) -> str:
    source = resolve_cached_hf_snapshot(cache_dir, repo_id)
    if not source.endswith("config.yaml"):
        return source

    try:
        import yaml
    except ImportError:
        return source

    source_path = Path(source)
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    pipeline_params = config.get("pipeline", {}).get("params", {})
    changed = False
    for key in ("segmentation", "embedding"):
        model_id = pipeline_params.get(key)
        if not isinstance(model_id, str) or "/" not in model_id:
            continue
        checkpoint_path = resolve_cached_hf_model_checkpoint(cache_dir, model_id)
        if checkpoint_path != model_id:
            pipeline_params[key] = checkpoint_path
            changed = True

    if not changed:
        return source

    patched_dir = Path(cache_dir) / "patched_pyannote_configs"
    patched_dir.mkdir(parents=True, exist_ok=True)
    repo_tag = repo_id.replace("/", "--")
    patched_path = patched_dir / f"{repo_tag}.yaml"
    patched_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(patched_path)


def load_diarization_pipeline(args: argparse.Namespace):
    from pyannote.audio import Pipeline

    token = None
    if args.hf_token_env:
        token = __import__("os").environ.get(args.hf_token_env)
    from_pretrained_params = inspect.signature(Pipeline.from_pretrained).parameters
    auth_kwargs: dict[str, Any] = {}
    if token and "token" in from_pretrained_params:
        auth_kwargs["token"] = token
    elif token and "use_auth_token" in from_pretrained_params:
        auth_kwargs["use_auth_token"] = token
    diarization_source = prepare_pyannote_pipeline_source(args.cache_dir, args.pyannote_model_id)
    pipeline_obj = Pipeline.from_pretrained(diarization_source, **auth_kwargs)
    if pipeline_obj is None:
        raise RuntimeError(f"pyannote pipeline '{diarization_source}' did not load successfully.")
    if args.device.startswith("cuda") and torch.cuda.is_available():
        pipeline_obj.to(torch.device(args.device))
    return pipeline_obj


def speaker_for_interval(
    start: float | None,
    end: float | None,
    diarization_segments: list[dict[str, Any]],
) -> str | None:
    if not diarization_segments:
        return None
    if start is None and end is None:
        return None

    interval_start = float(start) if start is not None else float(end)
    interval_end = float(end) if end is not None else float(start)
    if interval_end < interval_start:
        interval_end = interval_start

    overlaps: dict[str, float] = {}
    midpoint = (interval_start + interval_end) / 2.0
    nearest_speaker: str | None = None
    nearest_distance: float | None = None
    for segment in diarization_segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        speaker = str(segment["speaker"])
        overlap = max(0.0, min(interval_end, seg_end) - max(interval_start, seg_start))
        if overlap > 0.0:
            overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap
        seg_midpoint = (seg_start + seg_end) / 2.0
        distance = abs(midpoint - seg_midpoint)
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_speaker = speaker
    if overlaps:
        return max(overlaps.items(), key=lambda item: item[1])[0]
    return nearest_speaker


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().translate(PUNCT_TRANSLATION).split()).strip()


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


PROMPT_TOKEN_PHRASES = [tokenize(phrase) for phrase in PROMPT_PHRASES]


def remove_phrase_spans(tokens: list[str], phrases: list[list[str]], leading_only: bool) -> list[str]:
    if not tokens:
        return tokens

    if leading_only:
        changed = True
        current = tokens
        while changed:
            changed = False
            for phrase in phrases:
                if len(current) >= len(phrase) and current[: len(phrase)] == phrase:
                    current = current[len(phrase) :]
                    changed = True
                    break
        return current

    result: list[str] = []
    i = 0
    while i < len(tokens):
        matched = False
        for phrase in phrases:
            end = i + len(phrase)
            if end <= len(tokens) and tokens[i:end] == phrase:
                i = end
                matched = True
                break
        if matched:
            continue
        result.append(tokens[i])
        i += 1
    return result


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


def iter_reference_rows(train_ref_dir: Path, test_ref_dir: Path) -> dict[str, dict[str, list[str]]]:
    rows: dict[str, dict[str, list[str]]] = {"train": {}, "test": {}}
    for split, ref_dir in (("train", train_ref_dir), ("test", test_ref_dir)):
        for path in sorted(ref_dir.glob("*.txt")):
            rows[split][path.stem.upper()] = tokenize(path.read_text(encoding="utf-8"))
    return rows


def aggregate_wer(detail_rows: list[dict[str, Any]]) -> dict[str, float]:
    def compute(subset: list[dict[str, Any]]) -> float:
        subs = sum(row["substitutions"] for row in subset)
        dels = sum(row["deletions"] for row in subset)
        ins = sum(row["insertions"] for row in subset)
        ref_words = sum(row["reference_words"] for row in subset)
        return ((subs + dels + ins) / ref_words) * 100 if ref_words else 0.0

    return {
        "All": round(compute(detail_rows), 2),
        "Healthy": round(compute([row for row in detail_rows if row["label_name"] == "Healthy"]), 2),
        "Alzheimer": round(compute([row for row in detail_rows if row["label_name"] == "Alzheimer"]), 2),
        "Train": round(compute([row for row in detail_rows if row["split"] == "train"]), 2),
        "Test": round(compute([row for row in detail_rows if row["split"] == "test"]), 2),
    }


def main() -> None:
    args = parse_args()
    asr_root = Path(args.asr_root)
    output_root = Path(args.output_root)
    summary_path = Path(args.summary_path) if args.summary_path else output_root / "wer_summary.csv"
    note_path = Path(args.note_path) if args.note_path else output_root / "notes.txt"
    both_root = asr_root / "raw_both_speakers"
    train_ref_dir = Path(args.train_ref_dir)
    test_ref_dir = Path(args.test_ref_dir)
    ref_tokens = iter_reference_rows(train_ref_dir, test_ref_dir)
    diarization_pipeline = load_diarization_pipeline(args)

    variants = [
        "longest_overlap",
        "speaker_00_overlap",
        "longest_overlap_keyword_leading",
        "longest_overlap_keyword_anywhere",
        "speaker_00_overlap_keyword_leading",
    ]
    for variant in variants:
        for split in ("train", "test"):
            (output_root / variant / split).mkdir(parents=True, exist_ok=True)

    items = []
    for split in ("train", "test"):
        for json_path in sorted((both_root / split).glob("*.json")):
            items.append((split, json_path))
    if args.max_files is not None:
        items = items[: args.max_files]

    detail_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    for split, json_path in items:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        subject_id = payload["subject_id"].upper()
        audio_path = Path(payload["source_audio_path"])
        waveform, sample_rate = load_audio(audio_path)
        diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
        if hasattr(diarization, "exclusive_speaker_diarization"):
            diarization_annotation = diarization.exclusive_speaker_diarization
        elif hasattr(diarization, "speaker_diarization"):
            diarization_annotation = diarization.speaker_diarization
        else:
            diarization_annotation = diarization

        diarization_segments: list[dict[str, Any]] = []
        speaker_durations: dict[str, float] = {}
        for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
            start = float(turn.start)
            end = float(turn.end)
            speaker = str(speaker)
            diarization_segments.append({"start": start, "end": end, "speaker": speaker})
            speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + max(0.0, end - start)

        if not speaker_durations:
            continue

        longest_speaker = max(speaker_durations.items(), key=lambda item: item[1])[0]
        speaker_00_available = "SPEAKER_00" in speaker_durations
        chunks = payload["transcription"]["chunks"]
        assigned_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            assigned_chunks.append(
                {
                    "text": str(chunk.get("text", "")).strip(),
                    "speaker": speaker_for_interval(chunk.get("start"), chunk.get("end"), diarization_segments),
                }
            )

        chunk_texts = {
            "longest_overlap": " ".join(
                chunk["text"] for chunk in assigned_chunks if chunk["speaker"] == longest_speaker and chunk["text"]
            ),
            "speaker_00_overlap": " ".join(
                chunk["text"] for chunk in assigned_chunks if chunk["speaker"] == "SPEAKER_00" and chunk["text"]
            ) if speaker_00_available else "",
        }
        chunk_texts["longest_overlap_keyword_leading"] = " ".join(
            remove_phrase_spans(tokenize(chunk_texts["longest_overlap"]), PROMPT_TOKEN_PHRASES, leading_only=True)
        )
        chunk_texts["longest_overlap_keyword_anywhere"] = " ".join(
            remove_phrase_spans(tokenize(chunk_texts["longest_overlap"]), PROMPT_TOKEN_PHRASES, leading_only=False)
        )
        chunk_texts["speaker_00_overlap_keyword_leading"] = " ".join(
            remove_phrase_spans(tokenize(chunk_texts["speaker_00_overlap"]), PROMPT_TOKEN_PHRASES, leading_only=True)
        )

        metadata_rows.append(
            {
                "split": split,
                "subject_id": subject_id,
                "longest_speaker": longest_speaker,
                "speaker_00_available": speaker_00_available,
                "speaker_durations_seconds": json.dumps(speaker_durations, sort_keys=True),
            }
        )

        reference = ref_tokens[split].get(subject_id)
        if reference is None:
            continue
        label_name = "Healthy" if subject_id in CONTROL_IDS else "Alzheimer"

        for variant, transcript in chunk_texts.items():
            normalized = normalize_text(transcript)
            out_path = output_root / variant / split / f"{subject_id}.txt"
            if args.overwrite or not out_path.exists():
                out_path.write_text(normalized + "\n", encoding="utf-8")
            hypothesis = normalized.split()
            subs, dels, ins = levenshtein(reference, hypothesis)
            detail_rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "subject_id": subject_id,
                    "label_name": label_name,
                    "reference_words": len(reference),
                    "hypothesis_words": len(hypothesis),
                    "substitutions": subs,
                    "deletions": dels,
                    "insertions": ins,
                }
            )

    summary_records = []
    for variant in variants:
        variant_rows = [row for row in detail_rows if row["variant"] == variant]
        summary_records.append({"Variant": variant, **aggregate_wer(variant_rows)})

    summary_df = pd.DataFrame(summary_records)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame(metadata_rows).to_csv(output_root / "metadata.csv", index=False)

    lines = [
        "Table: WhisperX single-speaker post-processing ablation",
        "Category: evaluation",
        "Task: compare participant-only transcript selection methods without rerunning ASR",
        "",
        f"ASR root: `{asr_root}`",
        "Reference protocol:",
        f"- cleaned participant-only transcripts from `{train_ref_dir.parent}`",
        "- scoring uses Liu-style lowercase / punctuation-stripped tokenization",
        "",
        "Post-processing variants:",
        "- `longest_overlap`: diarize full audio, assign ASR chunks by temporal overlap, keep longest speaker",
        "- `speaker_00_overlap`: same, but force `SPEAKER_00`",
        "- `longest_overlap_keyword_leading`: `longest_overlap`, then strip leading prompt phrases only",
        "- `longest_overlap_keyword_anywhere`: `longest_overlap`, then strip any exact prompt phrase spans",
        "- `speaker_00_overlap_keyword_leading`: `speaker_00_overlap`, then strip leading prompt phrases",
        "",
        "| Variant | All | Healthy | Alzheimer | Train | Test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in summary_records:
        lines.append(
            f"| {record['Variant']} | {record['All']:.2f} | {record['Healthy']:.2f} | {record['Alzheimer']:.2f} | {record['Train']:.2f} | {record['Test']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Prompt inventory used for keyword stripping:",
            "- `tell me what you see`",
            "- `what do you see`",
            "- `I'd like you to tell me` / `I want you to tell me`",
            "- `anything else` / `anything else going on`",
            "- `can you tell me more` / `can you see anything else`",
        ]
    )
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(f"Wrote note to {note_path}")


if __name__ == "__main__":
    main()
