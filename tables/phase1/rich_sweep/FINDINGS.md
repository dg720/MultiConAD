# Phase 1 Rich Sweep Findings

This note documents the additive rich feature benchmark run under `tables/phase1/rich_sweep/` and the dataset/task normalization decisions that feed it.

## Reference sources

- Archive code reference: `MultiConAD-archive/Experiments/TF_IDF_classifier.py`, `MultiConAD-archive/Experiments/e5_larg_classifier.py`
- Paper reference: `2024_multiconad_multilingual_alzheimer_shakeri_farmanbar_balog_annotated.pdf`
- DementiaBank / corpus pages used during task review:
  - https://talkbank.org/dementia/access/
  - https://talkbank.org/dementia/access/English/WLS.html
  - https://talkbank.org/dementia/access/English/Protocol/Delaware.html
  - https://talkbank.org/dementia/access/English/Kempler.html
  - https://talkbank.org/dementia/access/English/Lu.html
  - https://talkbank.org/dementia/TAUKADIAL/

## Dataset bucketing decisions

These decisions are reflected in the current manifest builder and were made by checking the paper, official corpus descriptions, and local transcript / CHAT structure.

- `WLS` is treated as `PD_CTP` for the speech task. The corpus page and local `.cha` files indicate Cookie Theft picture descriptions, even though labels in the current repo are weak and partly fluency-derived.
- `Delaware` is treated as `MIXED_PROTOCOL`. Local `.cha` files include multiple tasks within a file: Cookie Theft, Cat Rescue, Rockwell, Cinderella, Sandwich, Tea.
- `Baycrest` is treated as `MIXED_PROTOCOL`. Local `.cha` files include multiple bundled discourse prompts, including Cinderella retelling and other discourse tasks.
- `Kempler` is split file-by-file between `PD_CTP`, `CONVERSATION`, and mixed protocol based on transcript content.
- `Lu` remains `PD_CTP`.
- `VAS` is treated as `COMMAND`, not `PD_CTP`.
- `PerLA` is treated as `CONVERSATION`.
- `DS3` is bucketed from test folders:
  - `test1 -> PD_CTP`
  - `test2 -> PICTURE_RECALL`
  - `test3 -> REPETITION`
  - `test4 -> MOTOR_SPEECH`
- `NCMMSC2021_AD` remains `OTHER`. The current long-recording packaging does not preserve reliable per-task boundaries.

Current task summary in the manifest is therefore less "unknown" than before, but not fully harmonized. Some corpus differences are structural and cannot be normalized away just by relabeling.

## Feature stack

The rich feature table is `data/processed/phase1/phase1_features.csv`.

- Text / discourse / graph features: lexical, discourse repetition, speech-graph, and length features.
- Syntax block: CLAN `%mor` / `%gra` tiers when available, otherwise Stanza fallback.
- Acoustic block: proper `openSMILE` `eGeMAPSv02` plus fallback `librosa` descriptors.
- Pause block: audio-derived silence/pause features, with TSV timing fallback where available.

The run explicitly validated the acoustic block and found `88` `openSMILE` eGeMAPS columns.

## Modeling methodology

The rich sweep runner is `experiments/phase1/run_rich_sweep.py`.

- Split: single-seed grouped 80/20 holdout by `group_id`, stratified at the group label level.
- Eligible labels: binary `AD` vs `HC` only.
- Normalization: train-only hierarchical z-scoring with fallback.
  - pooled benchmark: `language + task_type + dataset_name -> language + task_type -> language -> global`
  - monolingual all-task runs: `task_type + dataset_name -> task_type -> global`
  - task-isolated monolingual runs: `dataset_name -> global`
- Feature selection: train-only `ANOVA f_classif`, with `k in {5, 10, 20, 50, 100, all}`.
- Feature subsets:
  - `all_universal`
  - `text_only`
  - `pause_only`
  - `speech_graph_only`
  - `acoustic_only`
  - `text_plus_pause`
  - `text_plus_pause_plus_graph`
  - `text_plus_pause_plus_acoustic`
- Model family, aligned to the archive setup:
  - `LR`: logistic regression
  - `DT`: decision tree
  - `RF`: random forest
  - `SVM linear`
  - `SVM rbf`

To keep the run single-seed and tractable, the family was mirrored from the archive, but the feature sweep itself is fixed-config rather than nested hyperparameter CV:

- `LR`: `C=1`
- `DT`: `max_depth=20`
- `RF`: `n_estimators=200`
- `SVM linear`: `C=1`
- `SVM rbf`: `C=1`

## Importance ranking methodology

There are three different ranking layers in these outputs.

### 1. ANOVA ranking

Files:

- `*_anova_ranking.csv`
- `*_anova_feature_groups.csv`

This is a train-only univariate screen. It is useful for showing which single features most separate classes before the model is fit, but it is not the final importance claim.

### 2. Held-out permutation importance

Files:

- `*_permutation_importance.csv`
- `*_permutation_importance_feature_groups.csv`

This is the primary importance view for interpretation.

- It is computed on the held-out test set.
- It works for all model classes, including `SVM rbf`.
- It uses the original selected input columns, not the transformed post-imputation design matrix.

This makes it the only ranking that is directly comparable across `LR`, `DT`, `RF`, and both SVM variants.

### 3. Native model importance

Files:

- `*_native_importance.csv`
- `*_native_importance_no_missing_indicators.csv`

This is supplemental only.

- Linear models expose coefficients.
- Tree models expose impurity importances.
- `SVM rbf` has no native coefficient vector, so there is no native importance file when the best model is nonlinear SVM.

### Missingness indicators

The pipeline uses `SimpleImputer(strategy="median", add_indicator=True)`.

That means a feature can influence the model in two ways:

- through its observed numeric value
- through a derived binary `missingindicator_*` flag if the value is absent

Important consequence:

- native importance files can show explicit `missingindicator_*` rows
- permutation importance files do not surface these as separate rows, because permutation runs on the original selected columns

This is why permutation importance is the main biomarker-facing ranking. It is less likely to overstate corpus-format missingness as if it were an independent biological feature.

In the pooled `PD_CTP` linear SVM run, `4` of the top `20` native coefficients were missingness indicators. In the English and Greek monolingual `PD_CTP` winners, the top `20` native features contained `0` missingness indicators.

## Benchmark-wide results

Key outputs:

- `benchmark_wide_rich_summary.json`
- `benchmark_wide_rich_model_results.csv`
- `benchmark_wide_rich_permutation_importance.csv`

Best benchmark-wide model:

- model: `SVM rbf`
- subset: `all_universal`
- top-k: `50`
- balanced accuracy: `0.749`
- macro F1: `0.752`
- AUROC: `0.818`
- AUPRC: `0.736`

Top held-out permutation features:

1. `ac_egemaps_slopeuv0_500_sma3nz_amean`
2. `ac_egemaps_stddevvoicedsegmentlengthsec`
3. `disc_mean_pairwise_utt_similarity`
4. `ac_egemaps_jitterlocal_sma3nz_stddevnorm`
5. `ac_egemaps_slopev0_500_sma3nz_amean`
6. `graph_density`
7. `syn_determiner_noun_ratio`

Top ANOVA features were much more syntax-heavy:

1. `syn_num_ratio`
2. `syn_interjection_ratio`
3. `syn_mean_dependency_length`
4. `syn_propn_ratio`
5. `syn_utterance_count`
6. `len_utterance_count`

Interpretation:

- the broad pooled benchmark still mixes genuine speech differences with language / task / corpus structure
- the richer acoustic block is now materially represented in the pooled winner
- ANOVA and final permutation rankings are not interchangeable; the final nonlinear model is using interactions beyond the top univariate syntax scores

## Primary pooled PD_CTP results

Key outputs:

- `pd_ctp_pooled_rich_summary.json`
- `pd_ctp_pooled_rich_model_results.csv`
- `pd_ctp_pooled_rich_permutation_importance.csv`
- `pd_ctp_pooled_rich_native_importance.csv`

Best pooled `PD_CTP` model:

- model: `SVM linear`
- subset: `all_universal`
- top-k: `100`
- balanced accuracy: `0.751`
- macro F1: `0.752`
- AUROC: `0.813`
- AUPRC: `0.727`

Top held-out permutation features:

1. `len_token_count`
2. `lex_function_word_ratio`
3. `syn_content_function_ratio`
4. `len_tokens_per_utterance`
5. `syn_open_closed_ratio`
6. `syn_mean_dependency_length`
7. `syn_aux_verb_ratio`
8. `syn_noun_verb_ratio`

Interpretation:

- once the analysis is restricted to `PD_CTP`, the feature story shifts away from generic corpus cues and toward lexical/syntactic structure
- acoustics still contribute, but they are not the dominant pooled `PD_CTP` signal
- missingness is still present in the native coefficient table and should be treated as a warning flag, not a biomarker claim

## Monolingual results

These are the main language-isolated outputs.

| Run | Best model | Best subset | BalAcc | AUROC |
| --- | --- | --- | ---: | ---: |
| `language_en_all_rich` | `SVM rbf` | `text_only`, `k=100` | `0.746` | `0.814` |
| `language_es_all_rich` | `RF` | `text_only`, `k=20` | `0.798` | `0.784` |
| `language_zh_all_rich` | `SVM rbf` | `text_plus_pause`, `k=100` | `0.838` | `0.913` |
| `language_el_all_rich` | `SVM rbf` | `acoustic_only`, `k=100` | `0.875` | `0.934` |

Important caveat:

- the language-isolated all-task runs are not equally comparable across languages because the underlying task mix differs
- English is dominated by `PD_CTP`
- Spanish is largely `READING`
- Chinese is a mix of `PD_CTP` and `OTHER`
- Greek is dominated by audio-first datasets and task families with less transcript structure

This is why the primary cross-language comparison should focus on `PD_CTP`, not on the all-task monolingual runs.

## Monolingual PD_CTP results

Key outputs:

- `language_en_pd_ctp_rich_*`
- `language_zh_pd_ctp_rich_*`
- `language_el_pd_ctp_rich_*`

Best monolingual `PD_CTP` results:

| Run | Best model | Best subset | BalAcc | AUROC |
| --- | --- | --- | ---: | ---: |
| `language_en_pd_ctp_rich` | `SVM linear` | `text_only`, `k=50` | `0.756` | `0.825` |
| `language_zh_pd_ctp_rich` | `DT` | `pause_only`, `k=20` | `0.986` | `0.986` |
| `language_el_pd_ctp_rich` | `SVM linear` | `acoustic_only`, `k=50` | `0.925` | `0.944` |

These very high Greek and Chinese `PD_CTP` scores should be treated cautiously.

- They are based on smaller sample sizes.
- They are closer to single-corpus structure than the pooled English case.
- They are still useful for feature ranking, but they are not yet strong evidence of stable cross-language biomarkers.

Top monolingual `PD_CTP` signals by language:

- English:
  - `len_type_count`
  - `syn_num_ratio`
  - `syn_determiner_ratio`
  - `syn_aux_ratio`
  - `lex_lemma_type_token_ratio`
  - `syn_mean_dependency_length`
- Chinese:
  - `pause_median_duration`
  - `len_tokens_per_second`
  - `pause_pause_to_word_ratio`
  - `len_audio_duration`
  - `pause_iqr_duration`
- Greek:
  - `ac_egemaps_jitterlocal_sma3nz_amean`
  - `ac_egemaps_loudness_sma3_stddevnorm`
  - `ac_mfcc_5_std`
  - `ac_egemaps_loudness_sma3_percentile50_0`
  - `ac_egemaps_mfcc4_sma3_stddevnorm`

Interpretation:

- English `PD_CTP` is currently driven mostly by lexical/syntactic structure.
- Chinese `PD_CTP` is currently driven mostly by pause timing and rate structure.
- Greek `PD_CTP` is currently driven mostly by acoustic micro-prosodic features.

That is not the pattern you would expect if a single stable universal feature set were already emerging across languages.

## Spanish task-specific secondary run

Because Spanish does not currently provide a comparable `PD_CTP` slice, a secondary `READING` run was added:

- output prefix: `language_es_reading_rich`
- best model: `SVM linear`
- subset: `text_only`
- top-k: `10`
- balanced accuracy: `0.740`
- AUROC: `0.752`

Top features:

1. `lex_mattr_20`
2. `lex_lemma_type_token_ratio`
3. `syn_determiner_ratio`
4. `lex_mean_token_length`
5. `lex_std_token_length`

This is much more aligned with the paper / Phase 2 guidance than forcing Spanish reading data into `PD_CTP`.

## Cross-lingual stability tables

Key outputs:

- `cross_lingual_all_tasks_feature_stability.csv`
- `cross_lingual_all_tasks_feature_group_stability.csv`
- `cross_lingual_pd_ctp_feature_stability.csv`
- `cross_lingual_pd_ctp_feature_group_stability.csv`

### All-task monolingual overlap

Shared top features across `en`, `es`, and `zh` all-task models:

- `lex_mattr_20`
- `disc_repeated_bigram_ratio`
- `lex_lemma_type_token_ratio`
- `lex_repetition_rate`
- `lex_type_token_ratio`
- `lex_content_word_ratio`
- `disc_repeated_unigram_ratio`

Shared top feature groups across `en`, `es`, and `zh`:

- `len`
- `lex`
- `disc`
- `syn`

This suggests there is some general multilingual overlap at the level of lexical diversity and repetition structure, even when task mix is imperfect.

### PD_CTP overlap

The `PD_CTP` monolingual comparison is much less stable.

- No single top feature appeared across `en`, `zh`, and `el`.
- The only cross-language group overlap at the top was `len` across `en` and `zh`.
- Greek `PD_CTP` was dominated by acoustics.
- Chinese `PD_CTP` was dominated by pause/rate features.
- English `PD_CTP` was dominated by lexical/syntactic features.

This is the most important substantive conclusion from the current run:

> after richer syntax and proper `openSMILE` features are added, cross-language `PD_CTP` feature importance is still language-specific rather than converging on a single stable universal biomarker set.

## Practical interpretation

What is solid now:

- the repo has a working additive rich feature benchmark
- the archive-aligned model family is now covered: `LR`, `DT`, `RF`, `SVM`
- nonlinear SVM is included
- proper `openSMILE` / `eGeMAPS` is included
- held-out permutation importance is available for every best model
- monolingual and task-isolated results are now documented

What is not solved yet:

- benchmark-wide pooled importance is still partly confounded by task and corpus structure
- `PD_CTP` cross-language feature stability is weak
- `NCMMSC2021_AD` and other mixed or weakly labeled sources still limit normalization quality
- Phase 2 prompt-specific semantic modules are still the next major step for interpretation

## Recommended next step

Proceed with the Phase 2 semantic task modules now that the task bucketing is cleaner:

- `PD_CTP`: content-unit dictionaries and prompt-aware semantic coverage
- `READING`: reading-specific lexical/pause/error analysis
- `CONVERSATION` and `STORY_NARRATIVE`: separate task-specific semantic modules, not pooled with `PD_CTP`

The current rich sweep is the correct base layer for that next step, but it is not the final biomarker interpretation by itself.
