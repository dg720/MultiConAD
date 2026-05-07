Dataset handling notes

Pitt
- The current repo matches the paper counts closely.
- English cleaning maps AD-like labels into the benchmark dementia/AD bucket.

Lu
- The core paper counts are present locally.
- The source JSONL also contains a few extra non-paper diagnoses, but the paper-comparable counts still line up.

VAS
- The `.cha` files do not carry usable diagnosis labels in the header metadata.
- We attach labels from `data/English/VAS/demo.xlsx`.
- The local repo has 100 labeled VAS files, one fewer than the paper-equivalent row.

Baycrest
- The current repo matches the paper counts.
- No special handling beyond the standard English CHAT extraction is needed.

Kempler
- The corpus note states all cases are Alzheimer's dementia cases.
- We label all Kempler examples as AD at extraction time.
- `d6cookie` is treated as the same participant metadata as `d6`.

WLS
- WLS is weakly labeled from the 2011 category-fluency threshold, not from an explicit AD diagnosis.
- The default path now follows the paper-style loose rule: `HC` if the threshold is met, `Dementia` otherwise.
- One paper-row equivalent transcript (`fid=2`) is absent locally, so the repo has `262` impaired rather than `263`.

Delaware
- The local Delaware folder is a larger longitudinal/session-level release than the paper row.
- The repo currently counts transcript files, not a paper-compatible participant subset.
- This is why the current counts are much larger than the paper's `61 MCI / 34 HC`.

Taukdial
- Each participant has three valid recordings, and the current benchmark keeps all three.
- The repo therefore counts Taukdial at the file level, not the participant level.
- Current file-level totals are English `246` and Chinese `260`, for `506` rows overall.

Ivanova
- The local repo is close to the paper counts, with small count drift.
- Current handling follows the existing Spanish extraction and cleaning pipeline without special relabeling.

PerLA
- The local repo has slightly more AD-like cases than the paper row.
- Current handling effectively maps `DTA` rows into the AD bucket.

NCMMSE
- Dataset is now fully wired into the pipeline (2026-05-06).
- Raw file counts confirmed: train AD=79 / HC=108 / MCI=93 (280 total); test_have_label AD=35 / HC=45 / MCI=39 (119 total); 399 labeled files in total.
- The paper Table 1 row (AD=79 / MCI=93 / HC=108) is TRAIN-ONLY. This is the one exception in Table 1:
  all other datasets are reported as total source (train+test combined), but for NCMMSC the paper authors
  used only the training split because test labels were unavailable (Section 3.1.4).
- The current repo has test_have_label which the paper did not use; our total labeled pool (399) exceeds
  the paper source count (280) by 119 files.
- test_none_label (119 unlabeled files) is skipped throughout.
- AD_dataset_6s (2468 train + 1153 test clips) is NOT used in the text pipeline.
- Whisper large-v3 transcription is in progress; benchmark counts will be available after transcription + text_cleaning_Chinese.py complete.

iFkyTek
- The labeled train/eval counts match the paper closely.
- The source JSONL still includes additional unlabeled test rows that are not part of the paper-comparable labeled counts.

DS3
- Canonical path: use public `data/Greek/DS3` files as items and attach labels from `data/Greek/Dem@Care/pilot/`.
- Current public-folder count after label join is `149 = AD 107 / HC 42`; the remaining `101` DS3 files stay `Unknown`.
- DS3 is file-level, not participant-level: many participants contribute `test1`..`test4` files, so repeated files for the same patient are expected.
- The current extracted DS3 scope is larger than the paper row because the public release includes multiple labeled files/interviews for the same participant.
- For replication, treat this as the intended file-level benchmark definition rather than a missing-data gap versus the paper.

DS5
- Canonical path: use public `data/Greek/DS5/PatientNNN/` folders as items and attach labels from `data/Greek/Dem@Care/short/0tasks/`.
- Current public-folder count after label join is `93 = AD 26 / MCI 36 / HC 31`.
- DS5 is very close to the paper counts; the current repo differs by only a small MCI count delta.

DS7
- Canonical path: use public `data/Greek/DS7/PatientNNN/` folders as items and attach labels from `data/Greek/Dem@Care/long/0tasks/`.
- Current public-folder count after label join is `58 = AD 27 / MCI 29 / HC 2`.
- DS7 is close on AD and HC, but lower on MCI than the paper.
- Excluded public dirs are `Patient33`, `Patient52`, `Patient58` (missing `.wav`) and `Patient85` (no Dem@Care label), which is why the joined count is lower than the raw folder count.

ADReSS-M
- The current repo matches the paper counts.
- No special handling is needed beyond the standard Greek processing path.
