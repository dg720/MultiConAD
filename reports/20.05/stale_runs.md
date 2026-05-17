# Stale Runs After Greek DS3 Label Recovery

Update applied: 2026-05-16 12:43 +01:00

## Dataset Change

The DS3 label attachment now scans all DS3 path segments for `patient N` / `control N` rather than checking only one fixed path segment.

Recovered source labels:

| Source | AD | MCI | HC | Unknown |
|---|---:|---:|---:|---:|
| DS3 before recovery | 107 | 0 | 42 | 101 |
| DS3 after recovery | 168 | 0 | 82 | 0 |

Recovered rows entering the cleaned benchmark after the existing `length >= 60` filter:

| Label | Added rows |
|---|---:|
| AD | 46 |
| HC | 32 |
| MCI | 0 |

Updated Greek cleaned split:

| Split | AD | MCI | HC | Total |
|---|---:|---:|---:|---:|
| Train | 142 | 39 | 92 | 273 |
| Test | 36 | 10 | 23 | 69 |
| Total | 178 | 49 | 115 | 342 |

## Fresh Artifacts

These files have been updated and should be treated as the current dataset basis:

- `processing/extraction/run_step2_collections.py`
- `data/processed/extracted/ASR_ds3_output.jsonl`
- `data/processed/cleaned/combined_jsonl_Greek_M_ADReSS_D3_5_7.jsonl`
- `data/processed/cleaned/train_greek.jsonl`
- `data/processed/cleaned/test_greek.jsonl`
- `data/processed/phase1/phase1_manifest.jsonl`

## Stale Feature Artifacts

These feature matrices were generated before the DS3 recovery and still reflect the old Greek row set. They must be regenerated before feature-based experiments are rerun or reported:

- `data/processed/phase1/phase1_features.csv`
- `data/processed/phase1/phase1_features_pd_ctp.csv`
- `data/processed/phase1/phase1_feature_metadata.csv`
- `data/processed/phase1/phase1_feature_metadata_pd_ctp.csv`
- `data/processed/phase2/phase2_features.csv`
- `data/processed/phase2/phase2_feature_metadata.csv`

Attempted regeneration note: `python -m processing.phase1.extract_features` failed under system Python because `stanza` was missing. The virtualenv run was stopped after exceeding 30 minutes, before the final feature matrix was written.

## Stale Experiment Outputs

All result artifacts below were produced against the old Greek data and should be rerun before being used in reports:

- `reports/13.05/**`
- `reports/20.05/all_feature_catalogue.txt`
- `reports/20.05/feature-list-working-changes.md`
- `reports/20.05/all_universal_multiseed_grid_anova_welch_report.txt`
- `reports/20.05/extended_excluding_task_specific_multiseed_grid_report.txt`
- `reports/20.05/full_feature_set_multiseed_grid_report.txt`
- `reports/20.05/task_specific_feature_catalogue.txt`
- `reports/20.05/task_specific_only_multiseed_grid_report.txt`
- `tables/01-baselines/embedding-baselines/**`
- `tables/01-baselines/transfer-learning-baselines/**`
- `tables/02-ablation-task-specific-features/**`
- `tables/03-ablation-translingual-language-specific/**`
- `tables/dataset-completion/paper_vs_current_labels.txt`
- `tables/dataset-completion/paper_vs_current_labels_old.txt`
- `tables/dataset-completion/dataset_readme.txt`

Practical rule: any experiment output generated before this update that includes Greek rows, multilingual pooled rows, or feature artifacts derived from `phase1_features.csv` / `phase2_features.csv` is stale.
