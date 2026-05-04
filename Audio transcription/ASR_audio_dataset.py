"""
Step 1: Audio Transcription — Whisper Large-v3
Run once per dataset:
    python ASR_audio_dataset.py --dataset taukadial_train
    python ASR_audio_dataset.py --dataset taukadial_test
    python ASR_audio_dataset.py --dataset adress_m_gr
    python ASR_audio_dataset.py --dataset ds3
    python ASR_audio_dataset.py --dataset ds5
    python ASR_audio_dataset.py --dataset ds7

Output: ../transcriptions/<dataset>_transcriptions.json
Format:  [{"file_name": "...", "transcription": "...", "language": "..."}, ...]
"""

import os
import json
import argparse
import whisper
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW = os.path.join(PROJECT_ROOT, "raw_datasets")
OUT = os.path.join(PROJECT_ROOT, "transcriptions")

SAMPLE_RATE = 16000  # Whisper internal sample rate

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
# mode "flat"  – transcribe every .wav in a directory (recursive=True walks subdirs)
# mode "tasks" – segment each patient's full-session .wav by .tasks timestamps
#                before transcribing; only the task_label window is used
# language     – ISO-639-1 code passed to Whisper (None = auto-detect)
# filter_langs – if set, skip results whose detected language is not in this list
# ---------------------------------------------------------------------------
CONFIGS = {
    "taukadial_train": {
        "path":           os.path.join(RAW, "TAUKADIAL", "TAUKADIAL-24-train", "TAUKADIAL-24", "train"),
        "mode":           "flat",
        "recursive":      False,
        "language":       None,           # mixed EN + ZH; let Whisper detect
        "filter_langs":   ["en", "zh"],
        "output":         os.path.join(OUT, "taukadial_train_transcriptions.json"),
    },
    "taukadial_test": {
        "path":           os.path.join(RAW, "TAUKADIAL", "TAUKADIAL-24-test", "TAUKADIAL-24", "test"),
        "mode":           "flat",
        "recursive":      False,
        "language":       None,
        "filter_langs":   ["en", "zh"],
        "output":         os.path.join(OUT, "taukadial_test_transcriptions.json"),
    },
    "adress_m_gr": {
        "path":           os.path.join(RAW, "ADReSS-M", "ADReSS-M-test-gr", "test-gr"),
        "mode":           "flat",
        "recursive":      False,
        "language":       "el",           # Greek
        "filter_langs":   None,
        "output":         os.path.join(OUT, "adress_m_gr_transcriptions.json"),
    },
    "ds3": {
        # DS3 is pre-split per task: day_folder/patient_X/testN/file.wav
        # Recursive walk picks up all of them; file_name preserves the full path
        "path":           os.path.join(RAW, "Greek", "DS3"),
        "mode":           "flat",
        "recursive":      True,
        "language":       "el",
        "filter_langs":   None,
        "output":         os.path.join(OUT, "ds3_transcriptions.json"),
    },
    "ds5": {
        # DS5: one long session .wav per patient; .tasks gives per-task timestamps
        "path":           os.path.join(RAW, "Greek", "DS5"),
        "mode":           "tasks",
        "language":       "el",
        "filter_langs":   None,
        "task_label":     "pict_descr",   # Cookie Theft equivalent
        "output":         os.path.join(OUT, "ds5_transcriptions.json"),
    },
    "ds7": {
        "path":           os.path.join(RAW, "Greek", "DS7"),
        "mode":           "tasks",
        "language":       "el",
        "filter_langs":   None,
        "task_label":     "pict_descr",
        "output":         os.path.join(OUT, "ds7_transcriptions.json"),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def whisper_kwargs(cfg):
    kw = {}
    if cfg.get("language"):
        kw["language"] = cfg["language"]
    return kw


def parse_tasks_file(tasks_path, task_label):
    """Return (start_sec, end_sec) for the first line matching task_label, else None."""
    with open(tasks_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and parts[0] == task_label:
                return float(parts[1]), float(parts[2])
    return None


# ---------------------------------------------------------------------------
# Transcription modes
# ---------------------------------------------------------------------------

def transcribe_flat(model, cfg):
    """Transcribe every .wav in a directory, optionally recursive."""
    root = cfg["path"]
    filter_langs = cfg.get("filter_langs")
    kw = whisper_kwargs(cfg)
    results = []

    if cfg.get("recursive"):
        wav_files = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith(".wav"):
                    wav_files.append(os.path.join(dirpath, fname))
    else:
        wav_files = [
            os.path.join(root, f)
            for f in sorted(os.listdir(root))
            if f.lower().endswith(".wav")
        ]

    for audio_path in tqdm(wav_files, desc=f"Transcribing ({os.path.basename(root)})"):
        try:
            result = model.transcribe(audio_path, **kw)
            lang = result["language"]
            if filter_langs and lang not in filter_langs:
                continue
            # Preserve subpath for DS3 so patient/test context is kept
            rel = os.path.relpath(audio_path, root)
            file_id = os.path.splitext(rel)[0].replace(os.sep, "/")
            results.append({"file_name": file_id, "transcription": result["text"], "language": lang})
        except Exception as e:
            print(f"  Error: {audio_path}: {e}")

    return results


def transcribe_tasks(model, cfg):
    """
    DS5 / DS7: each patient folder contains one full-session .wav and a .tasks
    file with per-task timestamps.  Extract only the task_label segment, then
    transcribe with Whisper.
    """
    root = cfg["path"]
    task_label = cfg.get("task_label", "pict_descr")
    kw = whisper_kwargs(cfg)
    results = []
    skipped = []

    patient_dirs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    for patient_id in tqdm(patient_dirs, desc=f"Transcribing ({os.path.basename(root)})"):
        pdir = os.path.join(root, patient_id)
        wavs   = [f for f in os.listdir(pdir) if f.lower().endswith(".wav")]
        tasks  = [f for f in os.listdir(pdir) if f.endswith(".tasks")]

        if not wavs or not tasks:
            skipped.append(f"{patient_id}: missing .wav or .tasks")
            continue

        segment = parse_tasks_file(os.path.join(pdir, tasks[0]), task_label)
        if segment is None:
            skipped.append(f"{patient_id}: no '{task_label}' entry in {tasks[0]}")
            continue

        start_sec, end_sec = segment
        try:
            audio = whisper.load_audio(os.path.join(pdir, wavs[0]))
            clip  = audio[int(start_sec * SAMPLE_RATE): int(end_sec * SAMPLE_RATE)]
            result = model.transcribe(clip, **kw)
            results.append({
                "file_name":     patient_id,
                "transcription": result["text"],
                "language":      result["language"],
            })
        except Exception as e:
            skipped.append(f"{patient_id}: {e}")

    if skipped:
        print(f"  Skipped {len(skipped)} patients:")
        for s in skipped:
            print(f"    {s}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio datasets with Whisper Large-v3")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(CONFIGS.keys()),
        help="Dataset to transcribe. Run once per dataset.",
    )
    args = parser.parse_args()
    cfg = CONFIGS[args.dataset]

    os.makedirs(OUT, exist_ok=True)

    print(f"Dataset  : {args.dataset}")
    print(f"Input    : {cfg['path']}")
    print(f"Output   : {cfg['output']}")
    print(f"Language : {cfg.get('language') or 'auto-detect'}")
    print()

    print("Loading Whisper large-v3...")
    model = whisper.load_model("large-v3")

    if cfg["mode"] == "tasks":
        results = transcribe_tasks(model, cfg)
    else:
        results = transcribe_flat(model, cfg)

    with open(cfg["output"], "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nDone: {len(results)} transcriptions saved to {cfg['output']}")


if __name__ == "__main__":
    main()
