"""Build participant-selection ablations from existing ASR both-speaker outputs.

This script does not rerun ASR. It reuses `raw_both_speakers` JSON plus source
audio, runs diarization, assigns existing ASR chunks to speakers by temporal
overlap, and renders participant-only transcript variants.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import string
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import torch

from mlmi_thesis.paths import PATHS


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
    "anything else",
    "anything else going on",
    "anything else going on in the picture",
    "anything else that you see happening",
    "can you tell me more",
    "can you see anything else",
    "is there anything else",
]
METHODS = (
    "speaker_00",
    "speaker_00_aggressive",
    "longest_speaker",
    "longest_speaker_aggressive",
    "keyword_identified_interviewer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pyannote-model-id", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--hf-token-env", default="HUGGINGFACE_HUB_TOKEN")
    parser.add_argument("--cache-dir", default=str(PATHS["hf_cache"]))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-ref-dir", default=str(Path("data/derived/no_pause_clean/train")))
    parser.add_argument("--test-ref-dir", default=str(Path("data/derived/no_pause_clean/test")))
    parser.add_argument("--test-labels-path", default=str(PATHS["adress_test_labels"]))
    parser.add_argument("--medium-threshold", type=float, default=0.5)
    parser.add_argument("--long-threshold", type=float, default=2.0)
    parser.add_argument("--utterance-gap-s", type=float, default=1.0)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--note-path")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().translate(PUNCT_TRANSLATION).split()).strip()


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


PROMPT_TOKEN_PHRASES = [tokenize(phrase) for phrase in PROMPT_PHRASES]


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
    patched_path = patched_dir / f"{repo_id.replace('/', '--')}.yaml"
    patched_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return str(patched_path)


def load_diarization_pipeline(args: argparse.Namespace):
    from pyannote.audio import Pipeline

    token = os.environ.get(args.hf_token_env)
    from_pretrained_params = inspect.signature(Pipeline.from_pretrained).parameters
    auth_kwargs: dict[str, Any] = {}
    if token and "token" in from_pretrained_params:
        auth_kwargs["token"] = token
    elif token and "use_auth_token" in from_pretrained_params:
        auth_kwargs["use_auth_token"] = token
    diarization_source = prepare_pyannote_pipeline_source(args.cache_dir, args.pyannote_model_id)
    pipeline_obj = Pipeline.from_pretrained(diarization_source, **auth_kwargs)
    if hasattr(pipeline_obj, "to") and args.device.startswith("cuda") and torch.cuda.is_available():
        pipeline_obj.to(torch.device(args.device))
    return pipeline_obj


def load_references(ref_dir: Path) -> dict[str, list[str]]:
    return {path.stem.upper(): tokenize(path.read_text(encoding="utf-8")) for path in sorted(ref_dir.glob("*.txt"))}


def load_test_label_map(path: Path) -> dict[str, int]:
    labels = pd.read_csv(path, sep=";")
    labels.columns = labels.columns.str.strip().str.lower()
    labels["id"] = labels["id"].str.strip().str.upper()
    labels["label"] = labels["label"].astype(int)
    return dict(zip(labels["id"], labels["label"]))


def load_raw_chunks(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for chunk in payload.get("transcription", {}).get("chunks", []):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        chunks.append(
            {
                "text": normalize_text(text),
                "raw_text": text,
                "tokens": tokenize(text),
                "start": chunk.get("start"),
                "end": chunk.get("end"),
            }
        )
    return payload, chunks


def extract_diarization_segments(diarization: Any) -> list[dict[str, Any]]:
    if hasattr(diarization, "exclusive_speaker_diarization"):
        diarization = diarization.exclusive_speaker_diarization
    elif hasattr(diarization, "speaker_diarization"):
        diarization = diarization.speaker_diarization
    segments: list[dict[str, Any]] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
    return segments


def speaker_for_interval(start: float | None, end: float | None, diarization_segments: list[dict[str, Any]]) -> str | None:
    if not diarization_segments or (start is None and end is None):
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
        distance = abs(midpoint - ((seg_start + seg_end) / 2.0))
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_speaker = speaker
    if overlaps:
        return max(overlaps.items(), key=lambda item: item[1])[0]
    return nearest_speaker


def speaker_durations(diarization_segments: list[dict[str, Any]]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for segment in diarization_segments:
        durations[segment["speaker"]] = durations.get(segment["speaker"], 0.0) + max(
            0.0, float(segment["end"]) - float(segment["start"])
        )
    return durations


def build_utterances(chunks: list[dict[str, Any]], gap_threshold: float) -> list[list[int]]:
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


def utterance_tokens(chunks: list[dict[str, Any]], utterance: list[int]) -> list[str]:
    tokens: list[str] = []
    for chunk_idx in utterance:
        tokens.extend(chunks[chunk_idx]["tokens"])
    return tokens


def utterance_speaker(chunks: list[dict[str, Any]], utterance: list[int]) -> str | None:
    counts: dict[str, int] = {}
    for chunk_idx in utterance:
        speaker = chunks[chunk_idx].get("speaker")
        if speaker is None:
            continue
        counts[str(speaker)] = counts.get(str(speaker), 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def contains_prompt(tokens: list[str]) -> bool:
    for phrase in PROMPT_TOKEN_PHRASES:
        max_i = len(tokens) - len(phrase) + 1
        for i in range(max(0, max_i)):
            if tokens[i : i + len(phrase)] == phrase:
                return True
    return False


def infer_interviewer_speaker(chunks: list[dict[str, Any]], utterances: list[list[int]]) -> str | None:
    speaker_hits: dict[str, int] = {}
    for utterance in utterances:
        tokens = utterance_tokens(chunks, utterance)
        if not contains_prompt(tokens):
            continue
        speaker = utterance_speaker(chunks, utterance)
        if speaker is None:
            continue
        speaker_hits[speaker] = speaker_hits.get(speaker, 0) + 1
    if not speaker_hits:
        return None
    return max(speaker_hits.items(), key=lambda item: item[1])[0]


def flatten_chunk_tokens(chunks: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    tokens: list[str] = []
    token_to_chunk: list[int] = []
    for chunk_idx, chunk in enumerate(chunks):
        for token in chunk["tokens"]:
            tokens.append(token)
            token_to_chunk.append(chunk_idx)
    return tokens, token_to_chunk


def match_phrase_spans(tokens: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
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


def apply_aggressive_drop(chunks: list[dict[str, Any]], gap_threshold: float) -> list[dict[str, Any]]:
    keep_indices: list[int] = []
    for utterance in build_utterances(chunks, gap_threshold):
        if contains_prompt(utterance_tokens(chunks, utterance)):
            continue
        keep_indices.extend(utterance)
    return [chunks[idx] for idx in keep_indices]


def select_chunks(method: str, chunks: list[dict[str, Any]], diarization_segments: list[dict[str, Any]], gap_threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    durations = speaker_durations(diarization_segments)
    longest = max(durations.items(), key=lambda item: item[1])[0] if durations else None
    utterances = build_utterances(chunks, gap_threshold)
    interviewer_speaker = infer_interviewer_speaker(chunks, utterances)
    metadata = {
        "longest_speaker": longest,
        "interviewer_speaker": interviewer_speaker,
        "speaker_durations_seconds": durations,
    }

    if method.startswith("speaker_00"):
        kept = [chunk for chunk in chunks if chunk.get("speaker") == "SPEAKER_00"]
    elif method.startswith("longest_speaker"):
        kept = [chunk for chunk in chunks if chunk.get("speaker") == longest]
    elif method == "keyword_identified_interviewer":
        if interviewer_speaker is not None:
            kept = [chunk for chunk in chunks if chunk.get("speaker") != interviewer_speaker]
            metadata["fallback"] = None
        else:
            kept = [chunk for chunk in chunks if chunk.get("speaker") == longest]
            metadata["fallback"] = "longest_speaker"
    else:
        raise ValueError(f"Unknown method: {method}")

    if method.endswith("_aggressive"):
        kept = apply_aggressive_drop(kept, gap_threshold)
        metadata["aggressive_prompt_drop"] = True
    else:
        metadata["aggressive_prompt_drop"] = False
    return kept, metadata


def pause_token_for_gap(gap_seconds: float, medium_threshold: float, long_threshold: float) -> str | None:
    if gap_seconds >= long_threshold:
        return "..."
    if gap_seconds >= medium_threshold:
        return "."
    return None


def render_no_pause(chunks: list[dict[str, Any]]) -> str:
    return " ".join(chunk["text"] for chunk in chunks if chunk["text"]).strip()


def render_pause_encoded(chunks: list[dict[str, Any]], medium_threshold: float, long_threshold: float) -> str:
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


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    note_path = Path(args.note_path) if args.note_path else output_root / "notes.txt"

    train_refs = load_references(Path(args.train_ref_dir))
    test_refs = load_references(Path(args.test_ref_dir))
    test_label_map = load_test_label_map(Path(args.test_labels_path))
    diarization_pipeline = load_diarization_pipeline(args)

    for method in METHODS:
        for suffix in ("no_pause", "pause_encoded"):
            for split in ("train", "test"):
                (output_root / method / suffix / split).mkdir(parents=True, exist_ok=True)

    wer_rows: list[dict[str, object]] = []
    pause_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    items = []
    for split in ("train", "test"):
        items.extend(sorted((raw_root / split).glob("*.json")))
    if args.max_files is not None:
        items = items[: args.max_files]

    for json_path in items:
        split = json_path.parent.name
        subject_id = json_path.stem.upper()
        payload, chunks = load_raw_chunks(json_path)
        waveform, sample_rate = load_audio(Path(payload["source_audio_path"]))
        diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
        diarization_segments = extract_diarization_segments(diarization)
        for chunk in chunks:
            chunk["speaker"] = speaker_for_interval(chunk.get("start"), chunk.get("end"), diarization_segments)

        refs = train_refs if split == "train" else test_refs
        reference = refs.get(subject_id)
        if reference is None:
            continue
        if split == "train":
            label_name = "Healthy" if subject_id in CONTROL_IDS else "Alzheimer"
        else:
            label_name = "Healthy" if test_label_map.get(subject_id) == 0 else "Alzheimer"

        for method in METHODS:
            kept_chunks, method_meta = select_chunks(method, chunks, diarization_segments, args.utterance_gap_s)
            no_pause_text = render_no_pause(kept_chunks)
            pause_text = render_pause_encoded(kept_chunks, args.medium_threshold, args.long_threshold)
            (output_root / method / "no_pause" / split / f"{subject_id}.txt").write_text(no_pause_text + "\n", encoding="utf-8")
            (output_root / method / "pause_encoded" / split / f"{subject_id}.txt").write_text(pause_text + "\n", encoding="utf-8")

            hyp = no_pause_text.split()
            subs, dels, ins = levenshtein(reference, hyp)
            wer_rows.append(
                {
                    "method": method,
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
                    "method": method,
                    "split": split,
                    "subject_id": subject_id,
                    "label_name": label_name,
                    "short": short,
                    "medium": medium,
                    "long": long,
                }
            )
            metadata_rows.append(
                {
                    "method": method,
                    "split": split,
                    "subject_id": subject_id,
                    **method_meta,
                }
            )

    wer_summary = pd.DataFrame(
        [{"Method": method, **aggregate_wer([r for r in wer_rows if r["method"] == method])} for method in METHODS]
    )
    pause_summary = (
        pd.DataFrame(pause_rows)
        .groupby(["method", "label_name"], as_index=False)
        .agg(
            transcripts=("subject_id", "count"),
            mean_medium=("medium", "mean"),
            mean_long=("long", "mean"),
            median_medium=("medium", "median"),
            median_long=("long", "median"),
        )
        .sort_values(["method", "label_name"])
    )
    wer_summary.to_csv(output_root / "wer_summary.csv", index=False)
    pause_summary.to_csv(output_root / "pause_summary.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(output_root / "metadata.csv", index=False)

    lines = [
        "Table: Participant Selection Ablation",
        "Category: evaluation",
        "Task: compare post-processing-only participant identification methods",
        "",
        f"ASR root: `{args.asr_root}`",
        f"Raw root: `{args.raw_root}`",
        "",
        "Methods:",
        "- `speaker_00`: keep diarized speaker `SPEAKER_00`",
        "- `speaker_00_aggressive`: `speaker_00` plus whole-utterance prompt drop",
        "- `longest_speaker`: keep the longest diarized speaker",
        "- `longest_speaker_aggressive`: `longest_speaker` plus whole-utterance prompt drop",
        "- `keyword_identified_interviewer`: identify interviewer speaker from prompt-containing utterances and remove that speaker",
        "",
        "| Method | All | Healthy | Alzheimer | Train | Test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in wer_summary.to_dict("records"):
        lines.append(
            f"| {record['Method']} | {record['All']:.2f} | {record['Healthy']:.2f} | {record['Alzheimer']:.2f} | {record['Train']:.2f} | {record['Test']:.2f} |"
        )
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(wer_summary.to_string(index=False))


if __name__ == "__main__":
    main()
