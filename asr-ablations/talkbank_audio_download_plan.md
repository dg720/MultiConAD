# TalkBank Audio Download Setup

This setup mirrors authorized DementiaBank/TalkBank media into `data/` so future
WhisperX and acoustic-feature runs can use local audio.

Credentials must not be committed. Use environment variables or the secure prompt:

```powershell
$env:TALKBANK_USER = "your_username"
$env:TALKBANK_PASSWORD = "your_password"
```

Run a small authenticated dry run first:

```powershell
python processing/downloads/download_talkbank_media.py --only english-delaware --dry-run --max-files 10
```

Download a single corpus:

```powershell
python processing/downloads/download_talkbank_media.py --only english-delaware
```

Download every enabled source in `processing/downloads/talkbank_media_sources.json`:

```powershell
python processing/downloads/download_talkbank_media.py
```

After downloading, audit which CHAT transcripts have matching local media:

```powershell
python processing/downloads/audit_chat_audio_coverage.py --root data/English
```

The audit writes:

```text
asr-ablations/talkbank_audio_coverage.csv
```

Notes:

- The downloader preserves the path below each TalkBank media URL inside the matching
  `data/` destination.
- Existing media files are skipped by default. Use `--overwrite` only when replacing
  corrupted or intentionally outdated downloads.
- `english-delaware` follows the confirmed `English/Protocol/Delaware/` media pattern.
- Other TalkBank URLs are editable in `processing/downloads/talkbank_media_sources.json`;
  run `--dry-run` after login and correct any source URL that lists zero files.
- Keep all downloaded media under `data/`, matching the repository data-layout rule.
