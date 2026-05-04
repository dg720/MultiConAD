# MultiConAD Reproduction — Progress Log

Reproducing the pipeline from [arXiv:2502.19208](https://arxiv.org/abs/2502.19208).

---

## Step 1: Audio Transcription — COMPLETE

Transcribed all audio-only datasets using **Whisper large-v3** on a local GTX 1660 Super (CUDA 12.4, PyTorch 2.6.0+cu124).

| Dataset | Records | Language | Notes |
|---|---|---|---|
| ADReSS-M Greek (test-gr) | 46 | el | flat mode |
| DS3 | 250 | el | recursive flat mode |
| DS5 | 93 | el | task-segmented (pict_descr) |
| DS7 | 59 | el | task-segmented (pict_descr); 3 skipped (missing .wav) |
| TAUKADIAL test | 120 | en/zh | auto-detect + language filter |
| TAUKADIAL train | 386 | en/zh | auto-detect + language filter |

Output: `transcriptions/*.json` (one file per dataset).

**Key fixes during Step 1:**
- PyTorch initially installed as CPU-only build; reinstalled with `--index-url https://download.pytorch.org/whl/cu124` to enable CUDA.
- DS5/DS7 use `.tasks` files for audio segmentation — only the `pict_descr` window is transcribed, not the full session.
- Whisper hallucinates "Υπότιτλοι AUTHORWAVE" on silent/noisy clips; handled downstream by `text_cleaning_Greek.py` (`min_length=60` filter).

---

## Step 2: Data Collection & Extraction — COMPLETE

All datasets parsed into normalised JSONL records via `run_step2_collections.py`.  
Output: `Extracting data/jsonl_files/` (17 files).

**What this step does:** Each collection script reads raw source files (.cha, .tsv, or Whisper JSON) and maps them to a standardised `NormalizedDataPoint` schema — capturing PID, diagnosis, age, gender, language, task, transcription text (participant-only, interviewer-only, and combined), and dataset metadata. One JSONL line = one session/recording.

| JSONL file | Source format | Records | Diagnosis distribution |
|---|---|---|---|
| English_Pitt_Control_cookie_output.jsonl | .cha | 243 | Control: 243 |
| English_Pitt_Dementia_cookie_output.jsonl | .cha | 309 | ProbableAD/AD: 309 |
| English_Lu_output.jsonl | .cha | 54 | Alzheimer's/Control |
| English_Baycrest_output.jsonl | .cha | 10 | MCI/Control |
| English_Delaware_output.jsonl | .cha | 407 | MCI/Control |
| English_Kempler_output.jsonl | .cha | 7 | mixed |
| English_VAS_output.jsonl | .cha | 100 | mixed |
| English_WLS_output.jsonl | .cha | 1368 | mixed (population cohort) |
| Spanish_Ivanova_output.jsonl | .cha | 357 | AD: 74, MCI: 87, HC: 196 |
| Spanish_PerLA_output.jsonl | .cha | 27 | mixed |
| Chinese_iFlytek_output.jsonl | .tsv | 401 | AD: 68, MCI: 144, CTRL: 111, Unknown: 78 |
| ASR_adress_m_gr_output.jsonl | Whisper JSON | 46 | Unknown (test set) |
| ASR_ds3_output.jsonl | Whisper JSON | 250 | Unknown |
| ASR_ds5_output.jsonl | Whisper JSON | 93 | Unknown |
| ASR_ds7_output.jsonl | Whisper JSON | 59 | Unknown |
| ASR_taukadial_test_output.jsonl | Whisper JSON | 120 | Unknown (test set) |
| ASR_taukadial_train_output.jsonl | Whisper JSON | 386 | Unknown |
| **Total** | | **4,237** | |

**Label joining (added to `run_step2_collections.py`):**
- DS5 (short protocol) + DS7 (long protocol): joined from `raw_datasets/Greek/Dem@Care/{short,long}/0tasks/{AD,HC,MCI}/` — diagnosis encoded directly in subdirectory. DS5: 93/93 fully labeled; DS7: 58/59 (Patient85 absent from all Dem@Care folders — 1 record dropped).
- DS3 (pilot): `Dem@Care/pilot/control/` → HC, `Dem@Care/pilot/patients/` → AD matched on `patient N` path component; `controlday*` dirs inferred as HC by directory name. 149/250 labeled; remaining 101 from unlabelled day-recording dirs → Unknown (dropped in cleaning).
- TAUKADIAL train: joined from `groundtruth.csv` (all 386 labeled).
- TAUKADIAL test: joined from `testgroundtruth.csv` + `meta_test.csv` (all 120 labeled).
- ADReSS-M test-gr: `test-gr-groundtruth.csv` — 46/46 matched (Control: 24, ProbableAD: 22), all labeled with age/gender/educ/MMSE.
- `greek-groundtruth.csv` retained for MMSE/age/education metadata on DS5 patients where available.

**Bugs fixed during Step 2:**
- `TSV_collection.py`: critical fix — loaded `2_final_list_train.csv` to populate Diagnosis/age/gender/Education per iFlytek record. Without this, all 401 records had `Diagnosis=Unknown` and would be dropped.
- `cha_collection.py`: Ivanova diagnosis and Dataset field extracted from filename (e.g. `AD-M-57-163.cha`) — `@ID:` line uses `Participant` role (not `Target_Adult`) so the standard parser skipped it.

**Datasets excluded from Step 2:**
- `ADReSS_IS2020_train` / `ADReSS_IS2020_test`: overlap with Pitt corpus; not used per paper Table 1.
- Chinese Lu: not yet available.
- NCMMSC2021: not publicly available.

---

## Step 3: Text Preprocessing — COMPLETE

Output: `Preprocessing_text/cleaned/` (train + test JSONL per language).

**Bugs fixed in cleaning scripts:**
- All 4 scripts: placeholder paths replaced with actual JSONL locations; `sys.path` added so `from collection import JSONLCombiner` resolves across directories.
- `text_cleaning_Greek.py`: `Text_length` column name corrected to `length` (was causing a `KeyError`).
- `text_cleaning_English.py`: added `"Conrol": "HC"` to diagnosis rename map (one-record typo in source data).
- `text_cleaning_Chinese.py`: removed NCMMSC2021 input (unavailable); script now uses iFlytek + TAUKADIAL only.
- `text_cleaning_Spanish.py`: `word_limits` filter now correctly matches `"Ivanova"` (Dataset field was `Unknown` due to upstream parsing gap; fixed in `cha_collection.py`).

**Post-cleaning record counts:**

| Language | Train | Test | Total | Diagnosis breakdown |
|---|---|---|---|---|
| English | 1,000 | 251 | 1,251 | HC: 619, MCI: 352, Dementia: 280 |
| Spanish | 303 | 76 | 379 | HC: 195, AD: 97, MCI: 87 |
| Chinese | 304 | 76 | 380 | MCI: 179, HC: 155, AD: 46 |
| Greek | 211 | 53 | 264 | AD: 132, HC: 83, MCI: 49 |
| **Total** | **1,818** | **459** | **2,277** | |

**Greek dataset note:** 264 records survive cleaning (99 → 220 → 264 across three label additions). Remaining losses:
- DS7 Patient85 (1): absent from all Dem@Care folders — dropped
- DS3 (101): unlabelled day-recording dirs (`audiorecday*`) — dropped (see investigation below)
- Short/hallucination transcripts removed by `min_length=60` filter

**DS3 unlabelled day-recordings — investigation (2026-05-04):**

101 DS3 records come from `audiorecday2_a/b/c`, `audiorecday3_a/b`, `audiorecday4` directories. These contain subdirs named `patient N` (patient 40–44, etc.) but carry no diagnosis. Investigated via:

- [demcare.eu/datasets](https://demcare.eu/datasets/) — confirms DS3 is "audio files from microphone"; no per-session labels published; data access requires a formal email request to info@demcare.eu.
- Karakostas et al., *The Dem@Care Experiments and Datasets: a Technical Report*, arXiv:1701.01142 — states the pilot recruited **89 participants "in various levels of severity of the cognitive/behavioral disturbances"**, explicitly a mixed cohort of healthy, MCI, mild dementia, and some full AD cases.

**Conclusion:** The `audiorecday*` participants cannot be safely assumed to belong to any single diagnosis group. The "patient N" subdirectory label is a generic participant identifier, not a diagnostic indicator. No external groundtruth file covering these sessions is publicly available. Dropping all 101 records is the correct approach; recovering labels would require contacting info@demcare.eu for the pilot participant metadata.

---

## Step 4: Translation — PENDING

Translate Greek, Spanish, Chinese → English using GPT-4:
```
python translation_all_language.py --source_language=gr ...
```

---

## Step 5: Classification — PENDING

```
python TF_IDF_classifier.py --test_language=... --task=... --translated=...
python e5_large_classifier.py  --test_language=... --task=... --translated=...
```
