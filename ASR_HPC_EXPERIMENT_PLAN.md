# ASR Robustness HPC Experiment Plan

Purpose: generate reusable ASR transcript sets for MultiConAD without overwriting existing transcripts, then rerun downstream transcript and feature experiments with ASR-derived text while keeping acoustic features fixed.

This plan is scoped to the MultiConAD replication datasets, not every corpus listed on TalkBank.

## Prior Reference: `mlmi-thesis-private`

The ADReSS work in `../mlmi-thesis-private` used the following conventions that should carry over here:

- Raw ASR outputs are kept separately from rendered `.txt` transcript variants.
- Speaker conditions are stored separately as `raw_both_speakers` and `raw_single_speaker`.
- Plain Whisper is tested only as a both-speaker/original-audio baseline. Do not run a plain-Whisper single-speaker, diarization, or pause-encoded condition.
- WhisperX single-speaker mode uses diarization, assigns word speakers, then keeps the participant speaker by longest diarized duration.
- Pause encoding is rendered after speaker selection and prompt/interviewer filtering, so interviewer prompts do not create artificial pause markers in participant text.
- The canonical single-speaker cleaning branch is the ADReSS `longest_speaker_aggressive` style:
  - normalize ASR chunks for matching by lowercasing, removing ordinary punctuation, and collapsing whitespace;
  - select the longest diarized speaker, not `SPEAKER_00`;
  - split retained chunks into utterance groups using a `>= 1.0s` inter-chunk gap;
  - remove full retained utterances containing known interviewer prompt phrases;
  - only then render no-pause or pause-encoded transcript text.
- ASR pause encoding uses aligned chunk or word gaps:
  - gap `< 0.5s`: no pause marker in ASR transcript variants
  - gap `>= 0.5s` and `< 2.0s`: `.`
  - gap `>= 2.0s`: `...`
- The ADReSS manual/MFA pipeline can encode short pauses as `,`, but the ADReSS ASR variants did not add commas for short ASR gaps. For MultiConAD ASR comparability, use the ASR convention above.
- The 13 May thesis report reaches the same conclusion: WhisperX single-speaker with prompt cleanup is the strongest ASR methodology, and the strongest rule is to keep the longest diarized speaker and then remove prompt-like interviewer utterances.

Relevant prior files:

- `../mlmi-thesis-private/scripts/preprocessing/transcribe_asr_alternative.py`
- `../mlmi-thesis-private/scripts/preprocessing/build_asr_transcript_variants.py`
- `../mlmi-thesis-private/scripts/preprocessing/build_whisperx_prompt_pause_ablation.py`
- `../mlmi-thesis-private/scripts/hpc/run_whisperx_medium_asr.sh`
- `../mlmi-thesis-private/slurm_commands/whisperx_asr/README.md`

Local reference copies now live under:

```text
processing/asr_robustness/mlmi_reference/
```

These are provenance copies for porting, not the final MultiConAD-specific runners. Adapt new runnable code under `processing/asr_robustness/` so ADReSS-specific imports and paths do not leak into the MultiConAD pipeline.

## Dataset Scope

Run ASR only where local source audio exists.

| Language | Source | ASR Status |
|---|---|---|
| English | Pitt | ready |
| English | Lu | ready |
| English | VAS | ready |
| English | Baycrest | ready |
| English | Kempler | ready |
| English | WLS | ready |
| English | Delaware | ready, but 2 CHAT stems remain unmatched for supervised checks |
| English | Taukdial | ready, audio-only for supervised transcript scoring |
| Spanish | Ivanova | ready, 357/357 CHAT transcripts match local audio |
| Spanish | PerLA | exclude from ASR, no TalkBank media available |
| Chinese | NCMMSC | ready |
| Chinese | iFlyTek | exclude from ASR, no local/source audio available |
| Chinese | Taukdial | ready, audio exists and benchmark transcripts are ASR-derived |
| Greek | DS3 | ready via `data/Greek/Dem@Care` mirror |
| Greek | DS5 | ready via `data/Greek/Dem@Care` mirror |
| Greek | DS7 | ready via `data/Greek/Dem@Care` mirror |
| Greek | ADReSS-M | ready |

Keep `PerLA` and `iFlyTek` in downstream tables as text-only reference rows, but do not queue ASR jobs for them.

## Four Transcript Conditions

All four conditions must be generated into separate roots. No condition should overwrite another condition, and raw model outputs should be retained.

| Condition ID | Name | Backend | Speaker Handling | Pause Encoding |
|---|---|---|---|---|
| `whisper-large` | Whisper large, both speakers | OpenAI Whisper large-v3 | original audio, both speakers retained, no diarization | no |
| `whisperx-medium-both` | WhisperX medium, both speakers | WhisperX medium | original audio, all detected speech retained | no |
| `whisperx-medium-single` | WhisperX medium, single speaker | WhisperX medium + diarization | keep participant speaker only | no |
| `whisperx-medium-single-pe` | WhisperX medium, single speaker + pause encoding | WhisperX medium + diarization | keep participant speaker only | yes |

## Output Layout

Use this root so generated transcripts stay separate from current cleaned benchmark text:

```text
data/processed/asr-transcripts/
  manifest/
    asr_audio_manifest.csv
    asr_audio_manifest.jsonl
    asr_run_index.jsonl

  whisper-large/
    raw_jsonl/
      <dataset>.jsonl
    transcripts/
      <dataset>/<sample_id>.txt
    metadata/
      <dataset>_summary.csv
      <dataset>_errors.jsonl

  whisperx-medium-both/
    raw_jsonl/
      <dataset>.jsonl
    transcripts/
      <dataset>/<sample_id>.txt
    metadata/
      <dataset>_summary.csv
      <dataset>_errors.jsonl

  whisperx-medium-single/
    raw_jsonl/
      <dataset>.jsonl
    diarization/
      <dataset>/<sample_id>.json
    transcripts/
      <dataset>/<sample_id>.txt
    metadata/
      <dataset>_summary.csv
      <dataset>_errors.jsonl

  whisperx-medium-single-pe/
    raw_jsonl/
      <dataset>.jsonl
    diarization/
      <dataset>/<sample_id>.json
    transcripts/
      <dataset>/<sample_id>.txt
    pause_metadata/
      <dataset>/<sample_id>.json
    metadata/
      <dataset>_summary.csv
      <dataset>_errors.jsonl
```

Run logs and downstream model results should go under:

```text
tables/04-ablation-asr-robustness/
  logs/
  job_logs/
  result-tables/
  result-tables/csv/
  summaries/
```

## Manifest Requirements

Before queuing HPC jobs, build one canonical ASR audio manifest with one row per ASR item:

```text
asr_id
language
source
dataset_key
sample_id
diagnosis
split
task_type
audio_path
reference_transcript_path
has_reference_transcript
has_chat_annotation
audio_duration_s
expected_language_code
notes
```

Important manifest rules:

- `asr_id` must be stable and unique, for example `english_pitt__Dementia_cookie_001-0`.
- `sample_id` should preserve the existing dataset identifier used by MultiConAD labels.
- For long-session datasets with `.tasks`, queue only the segment required by the benchmark task or write the task window to a temporary clip cache.
- For CHAT-supervised datasets, link `reference_transcript_path` for later WER and prompt-removal checks.
- For audio-only rows, keep `reference_transcript_path` empty and still generate ASR text.

## HPC Job Plan

### Stage 0: Preflight

1. Clone the repository on HPC.
2. Copy the full local `data/` folder to the HPC clone, including the newly mirrored audio. Media files are intentionally git-ignored, so clone alone is not enough.
3. Confirm that the copied `data/` folder contains the audio-bearing sources listed below.
4. Build `data/processed/asr-transcripts/manifest/asr_audio_manifest.csv`.
5. Validate:
   - all queued rows have `audio_path` present;
   - no queued row points to PerLA or iFlyTek;
   - all output roots are empty or explicitly marked as resumable;
   - no transcript condition writes into `data/processed/cleaned/`.

Audio-bearing sources currently under `data/`:

```text
data/English/Pitt
data/English/Lu
data/English/VAS
data/English/Baycrest
data/English/Kempler
data/English/WLS
data/English/Delaware
data/Spanish/Ivanova
data/NCMMSC2021_AD
data/TAUKADIAL
data/Greek/DS3
data/Greek/DS5
data/Greek/DS7
data/Greek/Dem@Care
data/ADReSS-M
```

Known non-audio table rows:

```text
data/Spanish/PerLA
data/Chinese/iFlytek
```

PerLA and iFlyTek should be excluded from ASR queues and retained only as text rows for non-ASR comparisons.

### Stage 1: `whisper-large`

Use Whisper large-v3 as the both-speaker baseline ASR condition. This is the only plain-Whisper condition.

Expected behavior:

- no diarization;
- no single-speaker filtering;
- no prompt cleanup beyond generic transcript rendering;
- no pause encoding;
- retain all speech in the audio segment;
- store raw JSONL and rendered `.txt` transcripts separately;
- language is passed per row where known: `en`, `es`, `zh`, `el`; use auto-detect only for mixed Taukdial if needed.

Output root:

```text
data/processed/asr-transcripts/whisper-large/
```

### Stage 2: `whisperx-medium-both`

Use WhisperX medium with alignment, but keep both speakers.

Expected behavior:

- run WhisperX ASR and alignment;
- retain all aligned words/chunks;
- keep speaker labels if diarization is available, but do not filter by speaker;
- no pause encoding in rendered transcript.

Output root:

```text
data/processed/asr-transcripts/whisperx-medium-both/
```

### Stage 3: `whisperx-medium-single`

Use WhisperX medium with diarization and participant-speaker filtering.

Expected behavior:

- run diarization;
- assign word speakers;
- infer participant speaker using the ADReSS-compatible default: longest diarized speaker;
- do not use `SPEAKER_00` as the default participant rule;
- apply the ADReSS-compatible cleaning branch before rendering:
  - normalize chunks for prompt matching with lowercase, punctuation removal, and whitespace collapse;
  - group retained chunks into utterances using `utterance_gap_s = 1.0`;
  - drop retained utterances containing source-specific interviewer prompt phrases;
  - preserve cleaned chunk timestamps for downstream pause rendering;
- write diarization metadata for every item;
- render participant-only no-pause transcript.

Output root:

```text
data/processed/asr-transcripts/whisperx-medium-single/
```

Quality checks:

- flag rows where only one speaker is detected;
- flag rows where participant-speaker duration is less than 35 percent of total diarized speech;
- flag rows where prompt-like utterances were removed, with `prompt_chunks_removed`, `prompt_tokens_removed`, and `prompt_utterances_removed`;
- for CHAT-supervised datasets, compare retained text against participant-tier transcript where possible.

### Stage 4: `whisperx-medium-single-pe`

Render pause-encoded transcripts from the same single-speaker WhisperX raw outputs.

Expected behavior:

- do not rerun ASR if `whisperx-medium-single/raw_jsonl/` and diarization metadata already exist;
- apply participant filtering first;
- reuse exactly the same cleaned chunk sequence as `whisperx-medium-single`;
- remove known interviewer/prompt utterances before pause encoding where source-specific prompt rules exist;
- insert pause markers from aligned gaps:
  - `.` for gaps `>= 0.5s` and `< 2.0s`;
  - `...` for gaps `>= 2.0s`;
  - no marker for gaps `< 0.5s`;
- write pause counts and gap summaries to sidecar JSON.

Output root:

```text
data/processed/asr-transcripts/whisperx-medium-single-pe/
```

## Prompt and Interviewer Handling

Do not use one global English-only prompt scrubber. Use source-aware rules:

- CHAT datasets: prefer participant tiers over text-pattern removal when deriving supervised checks.
- Picture-description tasks: remove interviewer prompt phrases before pause encoding if they are present in the retained single-speaker text.
- Free-conversation and narrative tasks: avoid aggressive prompt deletion unless the source has a known repeated prompt.
- For multilingual prompts, store the exact phrase list in:

```text
data/processed/asr-transcripts/manifest/prompt_scrub_rules.json
```

Keep two fields in summaries:

```text
prompt_tokens_removed
prompt_chunks_removed
```

This follows the ADReSS finding that scrub-before-pause-encode is safer than pause-encode-before-scrub.

## Downstream Experiments

Before running classifier experiments, run a systematic ASR quality audit across all completed transcript conditions. The audit is a review gate, not an automatic exclusion step.

### Stage 5: ASR Quality and Robustness Audit

The proposed audit should be implemented as one modular script first:

```text
processing/asr_robustness/audit_asr_quality.py
```

Runnable form:

```bash
python -m processing.asr_robustness.audit_asr_quality \
  --metadata data/processed/asr-transcripts/manifest/asr_audio_manifest.csv \
  --asr-root data/processed/asr-transcripts \
  --out-dir tables/04-ablation-asr-robustness/asr-quality-audit
```

Default output root:

```text
tables/04-ablation-asr-robustness/asr-quality-audit/
```

Required outputs:

```text
file_quality_metrics.csv
dataset_asr_quality_summary.csv
language_asr_quality_summary.csv
dataset_asr_quality_report_table.csv
```

Optional plots:

```text
wer_by_language_asr_condition.png
cer_by_language_asr_condition.png
silence_ratio_by_language.png
prompt_contamination_by_asr_condition.png
```

The main report table should use this schema:

```text
dataset | language | task | asr_condition | n_files | n_refs |
mean_duration_sec | mean_silence_ratio | mean_wer | mean_cer |
pct_high_error | pct_prompt_contamination | pct_warning_or_inspect
```

Inputs:

- `data/processed/asr-transcripts/manifest/asr_audio_manifest.csv`
- the four ASR transcript roots under `data/processed/asr-transcripts/`
- manual/reference transcripts where available
- raw ASR metadata, especially detected language and WhisperX chunk timings

Reuse or mirror current project pieces where possible:

- Manifest fields already exist in spirit in `processing/phase1/build_manifest.py`: dataset/source, language, task, diagnosis, split, audio path, and transcript path.
- Audio loading and RMS-based frame statistics already exist in `processing/phase2/extract_features.py`; reuse the same `librosa`/RMS style rather than introducing a heavy VAD model.
- Existing ASR ingestion lives in `processing/extraction/ASR_collection.py`, but it does not compute quality metrics.
- CHAT/audio pairing audits already exist in `processing/downloads/audit_chat_audio_coverage.py`; use them to populate `has_manual_transcript` and reference availability.
- The ADReSS repo has WER normalization/edit-distance logic in `../mlmi-thesis-private/scripts/evaluation/compute_asr_wer.py` and prompt-contamination ablations in `../mlmi-thesis-private/scripts/preprocessing/build_whisperx_prompt_pause_ablation.py`.

New pieces needed for MultiConAD:

- One row per `file_id` x `asr_condition`.
- CER, especially for Chinese. WER can be reported for Chinese, but CER should be treated as the main quality metric.
- Insertion, deletion, and substitution rates, not only aggregate WER.
- Detected-language mismatch flags from ASR metadata.
- Prompt keyword counts in English, Spanish, Chinese, and Greek.
- Filled-pause counts and unknown/non-speech marker counts.
- Warning flags and an overall file-level quality label.

Recommended file-level columns:

```text
file_id
dataset
language
task
diagnosis
split
asr_condition
has_audio
has_manual_transcript
duration_sec
speech_duration_sec
silence_ratio
sample_rate
channels
detected_language
manual_word_count
asr_word_count
manual_char_count
asr_char_count
wer
cer
deletion_rate
insertion_rate
substitution_rate
prompt_keyword_count
filled_pause_count
unknown_token_count
flag_short_audio
flag_high_silence
flag_language_mismatch
flag_high_error_rate
flag_prompt_contamination
flag_empty_or_tiny_transcript
overall_quality_flag
```

Recommended thresholds:

- `flag_short_audio`: `duration_sec < 10`
- `flag_high_silence`: `silence_ratio > 0.70`
- `flag_language_mismatch`: detected language exists and does not match expected language
- `flag_high_error_rate`: Chinese `CER > 0.40`; non-Chinese `WER > 0.50`
- `flag_prompt_contamination`: `prompt_keyword_count >= 2`
- `flag_empty_or_tiny_transcript`: `asr_word_count < 5` and `asr_char_count < 20`
- `overall_quality_flag`: `pass` for no flags, `warning` for 1-2 flags, `inspect` for 3+ flags

Reference/error-rate rules:

- Missing manual references should leave WER/CER blank. This is not a failure.
- For WER/CER, normalize manual and ASR text the same way: Unicode normalization, lowercase where appropriate, repeated whitespace collapse, and punctuation removal for scoring only.
- Do not mutate or overwrite raw/manual/ASR transcript files during scoring.
- For Chinese, emphasize CER in summaries and use WER only as a secondary diagnostic.

Transcript robustness lists:

- Filled pauses:
  - English: `um`, `uh`, `er`, `erm`
  - Spanish: `eh`, `em`, `este`
  - Greek: `εε`, `εμμ`, `εμ`
  - Chinese: `嗯`, `呃`, `啊`, `哦`
- Unknown/non-speech markers:
  - `xxx`, `unk`, `unknown`, `inaudible`, `unintelligible`, `[noise]`, `[laugh]`, `<unk>`, `【笑】`, `【清嗓子】`
- Prompt keywords:
  - English: `tell me`, `describe`, `what do you see`, `picture`, `everything you see`
  - Spanish: `describe`, `imagen`, `qué ves`, `que ves`, `cuéntame`, `dime`
  - Chinese: `请描述`, `描述`, `你看到`, `图片`, `发生了什么`, `说一说`
  - Greek: start with `describe`, `picture`; TODO: manually expand after exact Greek prompt phrases are verified.

The audit script should be lightweight and must not run ASR models. It should tolerate missing audio, missing manual transcripts, and missing ASR outputs by leaving metrics blank and setting appropriate flags.

For each of the four transcript conditions:

1. Build a cleaned ASR collection without modifying current cleaned JSONL files.
2. Rebuild Phase 1 and Phase 2 manifests using an explicit `transcript_condition` column.
3. Re-extract text-derived and task-specific text features from ASR transcripts.
4. Keep acoustic features fixed from the original audio where possible, so ASR effects are mainly transcript effects.
5. Run the same model families and tuning protocol as the existing Phase 2 task-specific feature experiments.
6. Report:
   - per-language and pooled results;
   - per-source results for large sources;
   - supervised WER where reference transcripts exist;
   - prompt-removal and diarization diagnostics;
   - pause-count distribution by diagnosis for the pause-encoded condition.

Do not compare `whisperx-medium-single-pe` only as a model input. Also compare its pause sidecar statistics against existing pause features to detect double-counting or systematic drift.

## No-Overwrite Rules

Every ASR job must be resumable and must refuse accidental overwrites by default.

Required behavior:

- if an output transcript exists and raw JSON exists, skip unless `--overwrite` is explicitly set;
- write failed rows to `<dataset>_errors.jsonl`;
- append run metadata to `asr_run_index.jsonl`;
- include git commit, condition ID, model ID, device, compute type, language, source dataset, row count, and start/end timestamps;
- never write ASR transcripts into:

```text
data/processed/cleaned/
data/processed/extracted/
```

## Initial Job Matrix

Start with smoke jobs before full queues:

| Smoke Job | Rows | Purpose |
|---|---:|---|
| `whisper-large` | 2 per language | verify language routing and output schema |
| `whisperx-medium-both` | 2 per language | verify WhisperX model/alignment on HPC |
| `whisperx-medium-single` | 2 per language | verify diarization and participant selection |
| `whisperx-medium-single-pe` | same rows as single | verify pause rendering from existing raw outputs |

Then queue by source, prioritizing supervised sources:

1. English CHAT sources: Pitt, Lu, VAS, Baycrest, Kempler, WLS, Delaware.
2. Spanish Ivanova.
3. Greek Dem@Care/DS3/DS5/DS7.
4. Chinese NCMMSC.
5. Audio-only Taukdial and ADReSS-M rows.

## Completion Criteria

The ASR generation stage is complete only when:

- all eligible manifest rows have four transcript-condition entries, except rows intentionally excluded from a condition;
- `PerLA` and `iFlyTek` are listed as excluded, not failed;
- raw JSONL, rendered transcripts, summaries, and errors exist for every condition;
- supervised datasets have WER or alignment diagnostics where reference transcripts exist;
- each downstream experiment records which transcript condition was used;
- current benchmark transcripts remain unchanged.
