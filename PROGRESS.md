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

## Step 4: Translation — COMPLETE

Translated Greek, Spanish, Chinese → English using **GPT-4o** via `Translation/translation_all_language.py`.  
Output: `Translation/translated/` (6 files — train + test per language).

**Script changes from original:**
- Switched model from `gpt-4` → `gpt-4o` (equivalent quality, ~10× cheaper)
- API key loaded from `OPENAI_API_KEY` environment variable instead of hardcoded
- Added `ThreadPoolExecutor` for parallel requests (3 workers per run)
- Added exponential backoff retry on `RateLimitError` (up to 8 attempts, 2^n second waits)
- Initial run with 10 workers × 6 parallel processes hit 30K TPM limit; reduced to 3 workers sequential per file

**Results (2026-05-04):**

| File | Records | Missing translations |
|---|---|---|
| train_greek_translated.jsonl | 211 | 0 |
| test_greek_translated.jsonl | 53 | 0 |
| train_spanish_translated.jsonl | 303 | 0 |
| test_spanish_translated.jsonl | 76 | 0 |
| train_chinese_translated.jsonl | 304 | 0 |
| test_chinese_translated.jsonl | 76 | 0 |
| **Total** | **1,023** | **0** |

**Quality notes:**
- Greek picture description and narrative tasks translate cleanly
- Spanish PerLA task (Don Quixote passage) translated with appropriate literary register; AD variant with vocabulary errors correctly preserved
- Chinese interviewer/patient speech interleaving preserved in output
- One Greek AD record contains nonsense syllables (`Πα πα κα...`) from a verbal fluency task — translated literally (correct behaviour; classifier sees English equivalent of incoherent speech)
- `translated` field added to each record alongside original `Text_interviewer_participant`

---

## Step 5: Classification — IN PROGRESS

### 5.1 WLS Label Recovery (2026-05-05)

Identified that all 1,368 WLS records had blank `Diagnosis` — dropped in English cleaning, causing English dataset to be ~1,000 records smaller than the paper (1,251 vs ~2,751). Root cause: DementiaBank WLS CHA files contain no demographic metadata.

Recovered using `raw_datasets/English/WLS/WLS-data.xlsx` (provided by user):
- **Primary**: `Data - 2020` sheet — research consensus diagnosis (1=HC, 2=MCI, 3=Dementia): 187 records
- **Fallback**: `Data - 2004, 2011` sheet — category fluency threshold (paper Section 3.1.4): age<60→16, 60–79→14, >79→12 words; 1,111 additional records
- **Join key**: `idtlkbnk % 100000 == int(File_ID)`
- **Result**: 1,298/1,368 WLS records labeled (70 remain unlabeled — no usable age/fluency data)

English dataset after recovery: **2,546 records** (HC:1,658, MCI:427, Dementia:461).

Greek gap (~241 records vs paper) confirmed unrecoverable — DS3 day-recording directories carry no diagnosis labels in any available file.

### 5.2 TF-IDF Sparse Classifier (2026-05-05)

Scripts fixed from original: placeholder paths, filename mismatches, mono/multi overwrite bug, `AD→Dementia` normalisation, added `--training mono|multi` arg, `class_weight='balanced'` on SVC (prevents majority-class collapse on imbalanced combined training sets), `n_jobs=-1` for GridSearchCV.

Full experiment matrix: 32 configs × 5 classifiers = **160 runs**. Results saved to `Experiments/results/tfidf_results.txt`, comparison tables (Tables 5 & 6 format) to `Experiments/results/tfidf_comparison_tables.txt`.

**Effect of WLS addition on English results:** Performance dropped ~0.10–0.16 F1 across most classifiers vs the pre-WLS run. The paper's English SVM monolingual binary (0.77) is now matched (0.75). Pre-WLS performance was artificially high due to a smaller, cleaner dataset. The degradation is attributable to:
- WLS fluency-threshold labels are noisier than clinical diagnoses
- HC class dominates training (65% HC after WLS addition)
- Domain mismatch: labels derived from verbal fluency task, evaluated on picture description

This is expected and aligns with the paper — kept as-is per paper methodology.

**Statistical reliability note:** Non-English test sets are small (Greek: 53, Spanish: 76, Chinese: 76 records), making single-seed F1 estimates unreliable (±0.05–0.10 variance). Averaging across multiple random seeds for the train/test split is planned before final comparison to paper. TF-IDF is fast enough (~2 hrs for 5 seeds); E5-large will use 3 seeds.

### 5.3 E5-Large Dense Classifier — COMPLETE (2026-05-05)

Full 32-config matrix run: 4 test languages × 2 tasks × 2 translated settings × 2 training modes × 4 classifiers = 128 evaluations. Results in `Experiments/results/e5_results.txt`, comparison tables updated in `Experiments/results/tfidf_comparison_tables.txt`.

**GPU memory issue encountered and fixed:** Each subprocess loaded E5-large (~2.2 GB VRAM) and PyTorch/CUDA did not release VRAM when the subprocess exited. After 5 configs, GPU (GTX 1660 Super, 6 GB) was 94% full (5780/6144 MB), causing config 6 to fall back to CPU and stall for 1.5+ hours. Fixed by:
- Making the model a module-level singleton (loaded once per subprocess, not once per `get_embeddings` call)
- Adding explicit `del _model; gc.collect(); torch.cuda.empty_cache()` at script exit

After fix, each config completed in ~3–5 min; full run in ~2.5 hours.

**Embedding caching:** 48 unique `.npy` cache files written to `Experiments/embedding_cache/`. Subsequent runs (e.g., multi-seed) will skip all encoding and go straight to GridSearchCV.

**Best macro F1 — E5-large (Dense), single seed:**

| Task       | Language | Mono   | Multi  | Multi+Trans |
|------------|----------|--------|--------|-------------|
| Binary     | EN       | 0.74   | 0.73   | 0.73        |
|            | GR       | 0.78   | 0.54   | 0.58        |
|            | ZH       | 0.89   | 0.74   | 0.76        |
|            | ES       | 0.73   | 0.67   | 0.67        |
| Multiclass | EN       | 0.63   | 0.62   | 0.61        |
|            | GR       | 0.60   | 0.61   | 0.53        |
|            | ZH       | 0.60   | 0.62   | 0.52        |
|            | ES       | 0.53   | 0.42   | 0.45        |

**Key observations:**

1. **E5 vs TF-IDF (binary):** E5-large substantially outperforms TF-IDF on Chinese monolingual (0.89 vs 0.71 SVM) and is comparable on English (0.74 vs 0.75). Greek monolingual is similar (0.78 vs 0.75). Spanish monolingual is similar (0.73 vs 0.77).

2. **Combined multilingual hurts E5 (binary):** TF-IDF benefits from combined training in some configs (Greek translated: 0.77 RF), but E5-large degrades significantly — Greek multi drops to 0.54, vs mono 0.78. Likely cause: the English-dominated training set (74% HC/Dementia English) pulls embeddings toward English disease-language patterns that do not transfer to Greek AD speech.

3. **Translation mostly unhelpful for E5:** E5-large is natively multilingual and encodes Greek/Chinese/Spanish directly; translating to English before encoding loses language-specific cues. In most configs, translated ≤ original for E5.

4. **Logistic Regression collapses under combined training:** In all multi-lingual binary configs, LR selected C=0.1 and predicted exclusively HC (macro F1 ~0.44). This is a regularisation/class-imbalance interaction: the combined training set is 74% HC, and heavy L2 shrinkage forces LR to the majority class. SVM with `class_weight='balanced'` is more robust here.

5. **Multiclass:** E5-large matches or exceeds TF-IDF across the board. Chinese mono E5 (0.60) vs TF-IDF (0.37) is the largest gain. The combined training helps Greek multiclass for E5 (multi LR: 0.61 vs mono SVM: 0.60) but not Spanish.

6. **Statistical note:** Non-English test sets are small (GR: 53, ZH: 76, ES: 76); single-seed estimates carry ±0.05–0.10 variance. Multi-seed averaging planned.

**Classifier rankings (consistent across languages/tasks):**
- Binary: LR > SVM > RF > DT (mono), SVM > RF > DT > LR (multi)
- Multiclass: LR ≈ SVM > RF > DT (mono+multi)
