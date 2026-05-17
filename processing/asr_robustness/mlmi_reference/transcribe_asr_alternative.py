"""Transcribe ADReSS audio with WhisperX or CrisperWhisper.

This mirrors the existing Whisper large-v3 pipeline but adds alternatives that
emphasize word-level alignment and, for WhisperX, integrated diarization.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from tqdm import tqdm

from mlmi_thesis.paths import PATHS


TARGET_SAMPLE_RATE = 16_000


@dataclass
class AudioItem:
    split: str
    subject_id: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["whisperx", "crisperwhisper", "hf_whisper"],
        required=True,
    )
    parser.add_argument(
        "--speaker-mode",
        choices=["both-speakers", "single-speaker"],
        required=True,
        help="Whether to transcribe original audio or participant-only audio.",
    )
    parser.add_argument("--model-id")
    parser.add_argument("--language", default="en")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-length-s", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type")
    parser.add_argument("--cache-dir", default=str(PATHS["hf_cache"]))
    parser.add_argument("--train-audio-root", default=str(PATHS["adress_train_audio"]))
    parser.add_argument("--test-audio-root", default=str(PATHS["adress_test_audio"]))
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--participant-speaker-strategy",
        choices=["longest", "speaker_00"],
        default="longest",
    )
    parser.add_argument("--keep-filtered-audio", action="store_true")
    parser.add_argument("--pyannote-model-id", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--hf-token-env", default="HUGGINGFACE_HUB_TOKEN")
    parser.add_argument("--splits", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--log-path")
    parser.add_argument("--whisperx-min-speakers", type=int)
    parser.add_argument("--whisperx-max-speakers", type=int)
    parser.add_argument("--whisperx-align-device", default="cpu")
    parser.add_argument("--whisperx-diarize-device", default="cpu")
    parser.add_argument("--whisperx-align-model-id")
    parser.add_argument("--crisper-pause-split-threshold", type=float, default=0.12)
    args = parser.parse_args()
    if not args.model_id:
        if args.backend == "whisperx":
            args.model_id = "large-v3"
        elif args.backend == "hf_whisper":
            args.model_id = "openai/whisper-large-v3"
        else:
            args.model_id = "nyrahealth/CrisperWhisper"
    return args


def collect_audio_items(train_root: Path, test_root: Path) -> list[AudioItem]:
    items: list[AudioItem] = []
    for wav_path in sorted(train_root.rglob("*.wav")):
        items.append(AudioItem(split="train", subject_id=wav_path.stem.upper(), path=wav_path))
    for wav_path in sorted(test_root.glob("*.wav")):
        items.append(AudioItem(split="test", subject_id=wav_path.stem.upper(), path=wav_path))
    return items


def append_log(log_path: Path | None, event: dict[str, Any]) -> None:
    if log_path is None:
        return
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


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


def resolve_compute_type(args: argparse.Namespace) -> str:
    if args.compute_type:
        return args.compute_type
    if args.device.startswith("cuda") and torch.cuda.is_available():
        return "float16"
    return "int8"


def device_summary(args: argparse.Namespace, compute_type: str) -> dict[str, Any]:
    summary = {
        "backend": args.backend,
        "requested_device": args.device,
        "compute_type": compute_type,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "torch_cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        try:
            summary["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            summary["cuda_device_name_error"] = str(exc)
    if args.backend == "whisperx":
        summary["whisperx_align_device"] = args.whisperx_align_device
        summary["whisperx_diarize_device"] = args.whisperx_diarize_device
    return summary


def speaker_output_root(base_root: Path, speaker_mode: str) -> Path:
    if speaker_mode == "both-speakers":
        return base_root / "raw_both_speakers"
    return base_root / "raw_single_speaker"


def join_word_tokens(tokens: list[str]) -> str:
    return " ".join(token.strip() for token in tokens if str(token).strip()).strip()


def normalize_whisper_chunks(chunks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for chunk in chunks or []:
        timestamp = chunk.get("timestamp") or [None, None]
        start, end = timestamp if len(timestamp) == 2 else (None, None)
        normalized.append(
            {
                "text": str(chunk.get("text", "")).strip(),
                "start": float(start) if start is not None else None,
                "end": float(end) if end is not None else None,
            }
        )
    return normalized


def normalize_whisperx_words(result: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        words = segment.get("words") or []
        if words:
            for word in words:
                text = str(word.get("word", "")).strip()
                if not text:
                    continue
                chunks.append(
                    {
                        "text": text,
                        "start": float(word["start"]) if word.get("start") is not None else None,
                        "end": float(word["end"]) if word.get("end") is not None else None,
                        "speaker": word.get("speaker"),
                    }
                )
            continue
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        chunks.append(
            {
                "text": text,
                "start": float(segment["start"]) if segment.get("start") is not None else None,
                "end": float(segment["end"]) if segment.get("end") is not None else None,
                "speaker": segment.get("speaker"),
            }
        )
    return chunks


def derive_participant_speaker(
    diarization_segments: list[dict[str, Any]],
    strategy: str,
) -> tuple[str, dict[str, float]]:
    speaker_durations: dict[str, float] = {}
    for segment in diarization_segments:
        speaker = str(segment["speaker"])
        start = float(segment["start"])
        end = float(segment["end"])
        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + max(0.0, end - start)
    if not speaker_durations:
        raise RuntimeError("Speaker diarisation produced no speaker turns.")
    if strategy == "speaker_00":
        participant_speaker = "SPEAKER_00"
        if participant_speaker not in speaker_durations:
            raise RuntimeError("SPEAKER_00 was not present in diarisation output.")
    elif strategy == "longest":
        participant_speaker = max(speaker_durations.items(), key=lambda item: item[1])[0]
    else:
        raise RuntimeError(f"Unknown participant speaker strategy: {strategy}")
    return participant_speaker, speaker_durations


def filter_waveform_to_speaker_segments(
    waveform: torch.Tensor,
    sample_rate: int,
    diarization_segments: list[dict[str, Any]],
    participant_speaker: str,
) -> tuple[torch.Tensor, list[dict[str, float | str]]]:
    kept_chunks: list[torch.Tensor] = []
    kept_segments: list[dict[str, float | str]] = []
    for segment in diarization_segments:
        if str(segment.get("speaker")) != participant_speaker:
            continue
        start = float(segment["start"])
        end = float(segment["end"])
        start_idx = max(0, int(start * sample_rate))
        end_idx = min(waveform.shape[1], int(end * sample_rate))
        if end_idx <= start_idx:
            continue
        kept_chunks.append(waveform[:, start_idx:end_idx])
        kept_segments.append({"start": start, "end": end, "speaker": participant_speaker})
    if not kept_chunks:
        raise RuntimeError("No participant diarization segments remained after filtering.")
    return torch.cat(kept_chunks, dim=1), kept_segments


def extract_whisperx_diarization_segments(diarize_segments: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if hasattr(diarize_segments, "iterrows"):
        for _, row in diarize_segments.iterrows():
            segments.append(
                {
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "speaker": str(row["speaker"]),
                }
            )
        return segments
    if isinstance(diarize_segments, list):
        for row in diarize_segments:
            segments.append(
                {
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "speaker": str(row["speaker"]),
                }
            )
        return segments
    raise RuntimeError("Unsupported WhisperX diarization segment format.")


class PyannoteWhisperXDiarizer:
    """Adapter that returns the DataFrame shape expected by whisperx.assign_word_speakers."""

    def __init__(self, pipeline_obj: Any):
        self.pipeline = pipeline_obj

    def __call__(
        self,
        audio: np.ndarray,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> Any:
        import pandas as pd

        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)[None, :])
        diarization = self.pipeline(
            {"waveform": waveform, "sample_rate": TARGET_SAMPLE_RATE},
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        diarize_df = pd.DataFrame(
            diarization.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda segment: float(segment.start))
        diarize_df["end"] = diarize_df["segment"].apply(lambda segment: float(segment.end))
        return diarize_df


def adjust_pauses_for_hf_pipeline_output(
    pipeline_output: dict[str, Any],
    split_threshold: float,
) -> dict[str, Any]:
    adjusted_chunks = [dict(chunk) for chunk in pipeline_output.get("chunks", [])]
    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]
        current_timestamp = current_chunk.get("timestamp")
        next_timestamp = next_chunk.get("timestamp")
        if not current_timestamp or not next_timestamp:
            continue
        current_start, current_end = current_timestamp
        next_start, next_end = next_timestamp
        if None in (current_start, current_end, next_start, next_end):
            continue
        pause_duration = float(next_start) - float(current_end)
        if pause_duration <= 0.0 or pause_duration > split_threshold:
            continue
        distribute = pause_duration / 2.0
        current_chunk["timestamp"] = (float(current_start), float(current_end) + distribute)
        next_chunk["timestamp"] = (float(next_start) - distribute, float(next_end))
    adjusted = dict(pipeline_output)
    adjusted["chunks"] = adjusted_chunks
    return adjusted


def load_whisperx_resources(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import whisperx
        from whisperx.vad import load_vad_model
    except ImportError as exc:
        raise SystemExit("whisperx is required for --backend whisperx. Install it first.") from exc

    compute_type = resolve_compute_type(args)
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    align_device = (
        args.whisperx_align_device
        if args.whisperx_align_device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    diarize_device = (
        args.whisperx_diarize_device
        if args.whisperx_diarize_device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    vad_model = load_vad_model(
        torch.device("cpu"),
        vad_onset=0.500,
        vad_offset=0.363,
        use_auth_token=None,
    )
    model = whisperx.load_model(
        args.model_id,
        device=device,
        compute_type=compute_type,
        download_root=args.cache_dir,
        language=args.language,
        vad_model=vad_model,
    )
    align_model_id = args.whisperx_align_model_id
    if align_model_id is None and args.language == "en":
        # Avoid the torchaudio bundle default for English because it relies on
        # cluster-side CUDA runtime resolution that has been brittle on CSD3.
        align_model_id = "facebook/wav2vec2-base-960h"
    model_a, metadata = whisperx.load_align_model(
        language_code=args.language,
        device=align_device,
        model_name=align_model_id,
        model_dir=args.cache_dir,
    )
    diarize_model = None
    if args.speaker_mode == "single-speaker":
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise SystemExit(
                "pyannote.audio is required for WhisperX single-speaker diarisation."
            ) from exc
        token = os.environ.get(args.hf_token_env)
        from_pretrained_params = inspect.signature(Pipeline.from_pretrained).parameters
        auth_kwargs: dict[str, Any] = {}
        if token and "token" in from_pretrained_params:
            auth_kwargs["token"] = token
        elif token and "use_auth_token" in from_pretrained_params:
            auth_kwargs["use_auth_token"] = token
        diarization_source = prepare_pyannote_pipeline_source(args.cache_dir, args.pyannote_model_id)
        pipeline_obj = Pipeline.from_pretrained(diarization_source, **auth_kwargs)
        if pipeline_obj is None:
            raise RuntimeError(
                f"pyannote pipeline '{diarization_source}' did not load successfully."
            )
        if hasattr(pipeline_obj, "to"):
            pipeline_obj.to(torch.device(diarize_device))
        diarize_model = PyannoteWhisperXDiarizer(pipeline_obj)
    return {
        "whisperx": whisperx,
        "asr_device": device,
        "align_device": align_device,
        "diarize_device": diarize_device,
        "model": model,
        "align_model": model_a,
        "align_metadata": metadata,
        "diarize_model": diarize_model,
    }


def load_diarization_pipeline(args: argparse.Namespace):
    if args.speaker_mode != "single-speaker" or args.backend not in {"crisperwhisper", "hf_whisper"}:
        return None
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise SystemExit(
            "pyannote.audio is required for CrisperWhisper single-speaker mode. "
            "Install it into the repo venv first."
        ) from exc

    token = os.environ.get(args.hf_token_env)
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


def load_hf_whisper_resources(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    cuda_available = args.device.startswith("cuda") and torch.cuda.is_available()
    torch_dtype = torch.float16 if cuda_available else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=False,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if cuda_available:
        model.to(args.device)
        pipe_device: int | str = 0
    else:
        pipe_device = "cpu"
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=args.chunk_length_s,
        batch_size=args.batch_size,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=pipe_device,
    )
    return {"pipeline": asr_pipeline}


def diarize_participant_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    diarization_pipeline,
    keep_filtered_audio: bool,
    participant_speaker_strategy: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})

    if hasattr(diarization, "exclusive_speaker_diarization"):
        diarization_annotation = diarization.exclusive_speaker_diarization
    elif hasattr(diarization, "speaker_diarization"):
        diarization_annotation = diarization.speaker_diarization
    else:
        diarization_annotation = diarization

    speaker_durations: dict[str, float] = {}
    segments: list[tuple[float, float, str]] = []
    for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        duration = max(0.0, end - start)
        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration
        segments.append((start, end, speaker))

    if not speaker_durations:
        raise RuntimeError("Speaker diarisation produced no speaker turns.")

    if participant_speaker_strategy == "speaker_00":
        participant_speaker = "SPEAKER_00"
        if participant_speaker not in speaker_durations:
            raise RuntimeError("SPEAKER_00 was not present in diarisation output.")
    elif participant_speaker_strategy == "longest":
        participant_speaker = max(speaker_durations.items(), key=lambda item: item[1])[0]
    else:
        raise RuntimeError(f"Unknown participant speaker strategy: {participant_speaker_strategy}")
    kept_chunks: list[torch.Tensor] = []
    kept_segments: list[dict[str, float | str]] = []
    for start, end, speaker in segments:
        if speaker != participant_speaker:
            continue
        start_idx = max(0, int(start * sample_rate))
        end_idx = min(waveform.shape[1], int(end * sample_rate))
        if end_idx <= start_idx:
            continue
        kept_chunks.append(waveform[:, start_idx:end_idx])
        kept_segments.append({"start": start, "end": end, "speaker": speaker})

    if not kept_chunks:
        raise RuntimeError("No participant-only audio remained after diarisation.")

    filtered = torch.cat(kept_chunks, dim=1)
    metadata = {
        "participant_speaker": participant_speaker,
        "participant_speaker_strategy": participant_speaker_strategy,
        "speaker_durations_seconds": speaker_durations,
        "kept_segments": kept_segments,
        "filtered_seconds": filtered.shape[1] / sample_rate,
        "saved_filtered_audio": keep_filtered_audio,
    }
    return filtered, metadata


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
    if interval_start is None or interval_end is None:
        return None
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


def transcribe_with_whisperx(
    resources: dict[str, Any],
    waveform: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    whisperx = resources["whisperx"]
    audio = waveform.squeeze(0).cpu().numpy().astype(np.float32)
    result = resources["model"].transcribe(
        audio,
        batch_size=args.batch_size,
        task=args.task,
        language=args.language,
    )
    aligned = whisperx.align(
        result["segments"],
        resources["align_model"],
        resources["align_metadata"],
        audio,
        resources["align_device"],
        return_char_alignments=False,
    )

    diarization_metadata: dict[str, Any] | None = None
    if args.speaker_mode == "single-speaker":
        diarize_kwargs: dict[str, Any] = {}
        if args.whisperx_min_speakers is not None:
            diarize_kwargs["min_speakers"] = args.whisperx_min_speakers
        if args.whisperx_max_speakers is not None:
            diarize_kwargs["max_speakers"] = args.whisperx_max_speakers
        diarize_segments = resources["diarize_model"](audio, **diarize_kwargs)
        assigned = whisperx.assign_word_speakers(diarize_segments, aligned)
        diarization_segments = extract_whisperx_diarization_segments(diarize_segments)
        participant_speaker, speaker_durations = derive_participant_speaker(
            diarization_segments,
            args.participant_speaker_strategy,
        )

        filtered_chunks = [
            chunk
            for chunk in normalize_whisperx_words(assigned)
            if chunk.get("speaker") == participant_speaker
        ]
        fallback_mode = "word_speaker_filter"
        if not filtered_chunks:
            filtered_chunks = []
            for segment in assigned.get("segments", []):
                if segment.get("speaker") != participant_speaker:
                    continue
                text = str(segment.get("text", "")).strip()
                if not text:
                    continue
                filtered_chunks.append(
                    {
                        "text": text,
                        "start": float(segment["start"]) if segment.get("start") is not None else None,
                        "end": float(segment["end"]) if segment.get("end") is not None else None,
                    }
                )
            if filtered_chunks:
                fallback_mode = "segment_speaker_filter"
        if not filtered_chunks:
            filtered_waveform, kept_segments = filter_waveform_to_speaker_segments(
                waveform,
                TARGET_SAMPLE_RATE,
                diarization_segments,
                participant_speaker,
            )
            fallback_mode = "diarization_audio_filter_rerun"
            rerun_transcription, _ = transcribe_with_whisperx(
                resources,
                filtered_waveform,
                argparse.Namespace(**{**vars(args), "speaker_mode": "both-speakers"}),
            )
            diarization_metadata = {
                "participant_speaker": participant_speaker,
                "participant_speaker_strategy": args.participant_speaker_strategy,
                "speaker_durations_seconds": speaker_durations,
                "segment_count": len(diarization_segments),
                "diarization_backend": "whisperx",
                "fallback_mode": fallback_mode,
                "kept_segments": kept_segments,
                "filtered_seconds": filtered_waveform.shape[1] / TARGET_SAMPLE_RATE,
            }
            return rerun_transcription, diarization_metadata
        transcription = {
            "text": join_word_tokens([chunk["text"] for chunk in filtered_chunks]),
            "chunks": [
                {
                    "text": chunk["text"],
                    "start": chunk.get("start"),
                    "end": chunk.get("end"),
                }
                for chunk in filtered_chunks
            ],
        }
        diarization_metadata = {
            "participant_speaker": participant_speaker,
            "participant_speaker_strategy": args.participant_speaker_strategy,
            "speaker_durations_seconds": speaker_durations,
            "segment_count": len(diarization_segments),
            "diarization_backend": "whisperx",
            "fallback_mode": fallback_mode,
        }
        return transcription, diarization_metadata

    chunks = normalize_whisperx_words(aligned)
    transcription = {
        "text": str(aligned.get("text", "")).strip() or join_word_tokens([chunk["text"] for chunk in chunks]),
        "chunks": [
            {
                "text": chunk["text"],
                "start": chunk.get("start"),
                "end": chunk.get("end"),
            }
            for chunk in chunks
        ],
    }
    return transcription, diarization_metadata


def transcribe_with_hf_whisper(
    resources: dict[str, Any],
    waveform: torch.Tensor,
    sample_rate: int,
    args: argparse.Namespace,
    diarization_pipeline,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    audio = waveform.squeeze(0).cpu().numpy().astype(np.float32)
    result = resources["pipeline"](
        {"raw": audio, "sampling_rate": sample_rate},
        generate_kwargs={"language": args.language, "task": args.task},
    )
    chunks = normalize_whisper_chunks(result.get("chunks"))

    if args.speaker_mode != "single-speaker":
        return {
            "text": str(result.get("text", "")).strip(),
            "chunks": chunks,
        }, None

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
        diarization_segments.append({"start": start, "end": end, "speaker": str(speaker)})
        speaker_durations[str(speaker)] = speaker_durations.get(str(speaker), 0.0) + max(0.0, end - start)
    if not speaker_durations:
        raise RuntimeError("Speaker diarisation produced no speaker turns.")

    if args.participant_speaker_strategy == "speaker_00":
        participant_speaker = "SPEAKER_00"
        if participant_speaker not in speaker_durations:
            raise RuntimeError("SPEAKER_00 was not present in diarisation output.")
    elif args.participant_speaker_strategy == "longest":
        participant_speaker = max(speaker_durations.items(), key=lambda item: item[1])[0]
    else:
        raise RuntimeError(f"Unknown participant speaker strategy: {args.participant_speaker_strategy}")

    filtered_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        assigned_speaker = speaker_for_interval(chunk.get("start"), chunk.get("end"), diarization_segments)
        if assigned_speaker != participant_speaker:
            continue
        filtered_chunks.append(
            {
                "text": chunk.get("text", ""),
                "start": chunk.get("start"),
                "end": chunk.get("end"),
            }
        )
    if not filtered_chunks:
        raise RuntimeError("No participant-only Whisper chunks remained after diarisation.")

    transcription = {
        "text": join_word_tokens([chunk["text"] for chunk in filtered_chunks]),
        "chunks": filtered_chunks,
    }
    diarization_metadata = {
        "participant_speaker": participant_speaker,
        "participant_speaker_strategy": args.participant_speaker_strategy,
        "speaker_durations_seconds": speaker_durations,
        "segment_count": len(diarization_segments),
        "diarization_backend": "pyannote",
        "fallback_mode": "chunk_speaker_filter",
    }
    return transcription, diarization_metadata


def load_crisperwhisper_resources(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    cuda_available = args.device.startswith("cuda") and torch.cuda.is_available()
    torch_dtype = torch.float16 if cuda_available else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        cache_dir=args.cache_dir,
        low_cpu_mem_usage=False,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if cuda_available:
        model.to(args.device)
        pipe_device: int | str = 0
    else:
        pipe_device = "cpu"
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=args.chunk_length_s,
        batch_size=args.batch_size,
        return_timestamps="word",
        torch_dtype=torch_dtype,
        device=pipe_device,
    )
    plain_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=0,
        batch_size=1,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=pipe_device,
    )
    return {"pipeline": asr_pipeline, "plain_pipeline": plain_pipeline}


def transcribe_with_crisperwhisper(
    resources: dict[str, Any],
    waveform: torch.Tensor,
    sample_rate: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    audio = waveform.squeeze(0).cpu().numpy().astype(np.float32)
    try:
        result = resources["pipeline"](
            {"raw": audio, "sampling_rate": sample_rate},
            generate_kwargs={"language": args.language, "task": args.task},
        )
    except RuntimeError as exc:
        error_text = str(exc)
        if "Expected 3D or 4D" not in error_text:
            raise
        # Some files trigger a zero-length timestamp-attention slice in the
        # Whisper timestamp path. Retry without chunking so generation runs on
        # the full utterance rather than stitching chunk-level timestamp
        # windows.
        try:
            result = resources["pipeline"](
                {"raw": audio, "sampling_rate": sample_rate},
                chunk_length_s=0,
                batch_size=1,
                generate_kwargs={"language": args.language, "task": args.task},
            )
        except Exception:
            result = resources["plain_pipeline"](
                {"raw": audio, "sampling_rate": sample_rate},
                generate_kwargs={"language": args.language, "task": args.task},
            )
            text = str(result.get("text", "")).strip()
            return {
                "text": text,
                "chunks": [{"text": text, "start": None, "end": None}] if text else [],
            }
    except AttributeError as exc:
        error_text = str(exc)
        if "'tuple' object has no attribute 'cpu'" not in error_text:
            raise
        result = resources["plain_pipeline"](
            {"raw": audio, "sampling_rate": sample_rate},
            generate_kwargs={"language": args.language, "task": args.task},
        )
        text = str(result.get("text", "")).strip()
        return {
            "text": text,
            "chunks": [{"text": text, "start": None, "end": None}] if text else [],
        }
    except Exception as exc:
        error_text = str(exc)
        if "tuple" not in error_text and "timestamp" not in error_text:
            raise
        result = resources["plain_pipeline"](
            {"raw": audio, "sampling_rate": sample_rate},
            generate_kwargs={"language": args.language, "task": args.task},
        )
        text = str(result.get("text", "")).strip()
        return {
            "text": text,
            "chunks": [{"text": text, "start": None, "end": None}] if text else [],
        }
    try:
        result = adjust_pauses_for_hf_pipeline_output(result, args.crisper_pause_split_threshold)
    except Exception:
        text = str(result.get("text", "")).strip()
        return {
            "text": text,
            "chunks": normalize_whisper_chunks(result.get("chunks")) if result.get("chunks") else ([{"text": text, "start": None, "end": None}] if text else []),
        }
    return {
        "text": str(result.get("text", "")).strip(),
        "chunks": normalize_whisper_chunks(result.get("chunks")),
    }


def transcribe_item(
    waveform: torch.Tensor,
    sample_rate: int,
    args: argparse.Namespace,
    backend_resources: dict[str, Any],
    diarization_pipeline,
) -> tuple[dict[str, Any], dict[str, Any] | None, torch.Tensor]:
    diarization_metadata: dict[str, Any] | None = None
    effective_waveform = waveform

    if args.backend == "crisperwhisper" and args.speaker_mode == "single-speaker":
        effective_waveform, diarization_metadata = diarize_participant_audio(
            waveform,
            sample_rate,
            diarization_pipeline,
            args.keep_filtered_audio,
            args.participant_speaker_strategy,
        )
        transcription = transcribe_with_crisperwhisper(
            backend_resources,
            effective_waveform,
            sample_rate,
            args,
        )
        return transcription, diarization_metadata, effective_waveform

    if args.backend == "crisperwhisper":
        transcription = transcribe_with_crisperwhisper(
            backend_resources,
            effective_waveform,
            sample_rate,
            args,
        )
        return transcription, diarization_metadata, effective_waveform

    if args.backend == "hf_whisper":
        transcription, diarization_metadata = transcribe_with_hf_whisper(
            backend_resources,
            effective_waveform,
            sample_rate,
            args,
            diarization_pipeline,
        )
        return transcription, diarization_metadata, effective_waveform

    transcription, diarization_metadata = transcribe_with_whisperx(
        backend_resources,
        effective_waveform,
        args,
    )
    return transcription, diarization_metadata, effective_waveform


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    train_audio_root = Path(args.train_audio_root)
    test_audio_root = Path(args.test_audio_root)
    raw_root = speaker_output_root(output_root, args.speaker_mode)
    raw_root.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path else raw_root / f"progress_{'_'.join(args.splits)}.jsonl"

    items = collect_audio_items(train_audio_root, test_audio_root)
    items = [item for item in items if item.split in set(args.splits)]
    if args.max_files is not None:
        items = items[: args.max_files]

    compute_type = resolve_compute_type(args)
    summary = device_summary(args, compute_type)
    if args.backend == "whisperx":
        backend_resources = load_whisperx_resources(args)
    elif args.backend == "hf_whisper":
        backend_resources = load_hf_whisper_resources(args)
    else:
        backend_resources = load_crisperwhisper_resources(args)
    diarization_pipeline = load_diarization_pipeline(args)

    print("ASR device summary:", json.dumps(summary, sort_keys=True), flush=True)
    append_log(
        log_path,
        {
            "event": "run_started",
            "backend": args.backend,
            "speaker_mode": args.speaker_mode,
            "splits": args.splits,
            "planned_items": len(items),
            "overwrite": args.overwrite,
            "max_files": args.max_files,
            "device_summary": summary,
        },
    )

    for item in tqdm(items, desc=f"{args.backend}-{args.speaker_mode}"):
        out_dir = raw_root / item.split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{item.subject_id}.json"
        if out_path.exists() and not args.overwrite:
            append_log(
                log_path,
                {
                    "event": "skipped_existing",
                    "backend": args.backend,
                    "speaker_mode": args.speaker_mode,
                    "split": item.split,
                    "subject_id": item.subject_id,
                    "output_path": str(out_path),
                },
            )
            continue

        append_log(
            log_path,
            {
                "event": "item_started",
                "backend": args.backend,
                "speaker_mode": args.speaker_mode,
                "split": item.split,
                "subject_id": item.subject_id,
                "source_audio_path": str(item.path),
            },
        )

        try:
            waveform, sample_rate = load_audio(item.path)
            transcription, diarization_metadata, effective_waveform = transcribe_item(
                waveform,
                sample_rate,
                args,
                backend_resources,
                diarization_pipeline,
            )
            payload = {
                "subject_id": item.subject_id,
                "split": item.split,
                "backend": args.backend,
                "speaker_mode": args.speaker_mode,
                "source_audio_path": str(item.path),
                "model_id": args.model_id,
                "sample_rate": sample_rate,
                "source_seconds": waveform.shape[1] / sample_rate,
                "transcription_seconds": effective_waveform.shape[1] / sample_rate,
                "diarization": diarization_metadata,
                "transcription": transcription,
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            if args.keep_filtered_audio and args.backend == "crisperwhisper" and args.speaker_mode == "single-speaker":
                filtered_audio_dir = raw_root / "filtered_audio" / item.split
                filtered_audio_dir.mkdir(parents=True, exist_ok=True)
                sf.write(
                    filtered_audio_dir / f"{item.subject_id}.wav",
                    effective_waveform.squeeze(0).cpu().numpy(),
                    sample_rate,
                )
            append_log(
                log_path,
                {
                    "event": "item_completed",
                    "backend": args.backend,
                    "speaker_mode": args.speaker_mode,
                    "split": item.split,
                    "subject_id": item.subject_id,
                    "output_path": str(out_path),
                    "source_seconds": waveform.shape[1] / sample_rate,
                    "transcription_seconds": effective_waveform.shape[1] / sample_rate,
                    "chunk_count": len(transcription.get("chunks", [])),
                },
            )
        except Exception as exc:
            append_log(
                log_path,
                {
                    "event": "item_failed",
                    "backend": args.backend,
                    "speaker_mode": args.speaker_mode,
                    "split": item.split,
                    "subject_id": item.subject_id,
                    "source_audio_path": str(item.path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            if not args.continue_on_error:
                raise

    append_log(
        log_path,
        {
            "event": "run_finished",
            "backend": args.backend,
            "speaker_mode": args.speaker_mode,
            "splits": args.splits,
        },
    )


if __name__ == "__main__":
    main()
