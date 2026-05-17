from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".wma"}


def media_stem(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("@Media:"):
            value = line.split(":", 1)[1].strip()
            return re.split(r"[,\t ]+", value, maxsplit=1)[0].strip()
    return path.stem


def audio_index(root: Path, min_bytes: int) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS and path.stat().st_size >= min_bytes:
            index.setdefault(path.stem, []).append(path)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CHAT transcript @Media stems against local audio files.")
    parser.add_argument("--root", type=Path, default=DATA_ROOT / "English")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "asr-ablations" / "talkbank_audio_coverage.csv")
    parser.add_argument("--min-bytes", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    index = audio_index(root, args.min_bytes)
    rows = []
    for transcript in sorted(root.rglob("*.cha")):
        stem = media_stem(transcript)
        matches = index.get(stem, [])
        rel = transcript.relative_to(root)
        rows.append(
            {
                "dataset_folder": rel.parts[0] if rel.parts else "",
                "transcript_path": str(transcript),
                "media_stem": stem,
                "audio_match_count": len(matches),
                "audio_path": str(matches[0]) if len(matches) == 1 else "",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    matched = sum(1 for row in rows if int(row["audio_match_count"]) > 0)
    print(f"Wrote {args.out}")
    print(f"Matched {matched}/{total} CHAT transcripts under {root}")


if __name__ == "__main__":
    main()
