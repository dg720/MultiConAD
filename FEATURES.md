# MultiConAD Biomarker Plan

This document defines the thesis-scoped experimental plan for a Python research pipeline on cross-lingual Alzheimer's dementia detection from speech using MultiConAD.

The goal is not only to maximize classification performance, but to identify which speech biomarkers are:

- language-general
- task-general
- language-specific
- task-specific
- unstable

The plan below is intentionally pragmatic. It keeps the analysis scientifically defensible, but avoids unnecessary complexity.

## Core decisions

- Primary label mode remains `AD_vs_HC`.
- In `AD_vs_HC`, bundle `AD`, `ProbableAD`, `PossibleAD`, and generic `Dementia` into the positive class.
- Keep `MCI` in the manifest, but exclude it from the main binary biomarker experiments by default.
- Continue using WLS in the benchmark and in biomarker extraction.
- Treat WLS as a valid but weaker-label fluency-derived source rather than excluding it.
- Primary biomarker claim is restricted to picture description / Cookie Theft style data, referred to as `PD_CTP`.
- Acoustic features should still be extracted even when speaker isolation is imperfect; missingness and instability will be handled downstream rather than used as a hard blocker.
- The pipeline should be implemented in phases:
  - Phase 1: universal task-agnostic feature screening across the whole benchmark
  - Phase 2: task-specific semantic modules
  - Phase 3: final biomarker interpretation with `PD_CTP` as the main claim

## Practical benchmark assumptions

The current local benchmark is heterogeneous and not fully standardized at the sample metadata level. The pipeline should therefore be designed to tolerate:

- mixed transcript formats
- missing audio
- missing participant-only turns
- uneven task coverage across languages
- weak task labels in some subsets
- dataset-specific label quality differences

This means the pipeline must always preserve:

- `dataset_name`
- `language`
- `task_type`
- `diagnosis_raw`
- `diagnosis_mapped`
- `label_source`
- `feature availability`
- `missingness indicators`

## Main scientific claim

The primary thesis claim should be:

```text
Which biomarkers are stable across languages within semantically comparable
picture-description / Cookie Theft tasks?
```

This is the cleanest cross-lingual comparison in the current benchmark.

Secondary analyses can examine:

- broader multilingual pooled classification
- fluency-driven biomarkers
- reading-task biomarkers
- story/narrative biomarkers
- dataset robustness

But these should not be the main biomarker claim.

## Task taxonomy

Use a slightly expanded but still interpretable task taxonomy:

```text
PD_CTP           = picture description / Cookie Theft style prompts
FLUENCY          = semantic or phonemic fluency, including WLS-derived fluency data
STORY_NARRATIVE  = story recall, story retelling, narrative prompts
CONVERSATION     = free conversation / interview-style connected speech
READING          = reading tasks
OTHER            = task not confidently assigned
```

This is intentionally simpler than a very fine-grained ontology, but more useful than a single pooled "speech task" label.

## Label policy

The main experiment mode is:

```text
AD_vs_HC
```

Mapping:

```text
HC -> 0
AD / ProbableAD / PossibleAD / Dementia -> 1
MCI -> excluded by default from primary experiments
OTHER / unknown -> excluded
```

Retain raw labels and add:

```text
diagnosis_raw
diagnosis_mapped
binary_label
label_source
label_quality
```

Suggested `label_quality` values:

```text
clinical
dataset_provided
derived_from_metadata
weak_rule_based
```

For WLS specifically, preserve it in the benchmark and mark its labels accordingly.

## Manifest design

Create one row per sample with at least:

```text
sample_id
participant_id
participant_id_source
dataset_name
language
task_type
task_type_source
task_type_confidence
prompt_id
diagnosis_raw
diagnosis_mapped
binary_label
label_source
label_quality
transcript_path
audio_path
age
sex
education
mmse
moca
has_text
has_audio
has_participant_only_text
```

Important implementation note:

- if a clean participant ID is unavailable, derive a fallback grouping ID from stable metadata such as `PID`, `File_ID`, or path-based patient identifiers
- for Greek specifically, use patient-like identifiers embedded in `File_ID` or path strings until the normalized participant field is repaired

This is a metadata normalization issue, not evidence that Greek lacks subjects.

## Feature strategy

The feature strategy should be:

```text
extract many features
group them cleanly
rank and narrow them using train-fold-only ANOVA and related statistics
```

The thesis should not depend on guessing a tiny hand-picked feature set up front.

The feature pipeline should be explicitly phased.

## Phase 1: Universal benchmark-wide feature screening

This is the pragmatic first pass across the full benchmark.

Goal:

```text
use a task-agnostic universal feature set
screen broad feature groups across all tasks
identify which features are consistently useful enough to keep
before investing time in task-specific semantic modeling
```

Phase 1 should avoid task-specific semantic features such as:

- Cookie Theft content units
- story proposition recall
- semantic fluency validity dictionaries
- fluency clustering / switching
- phonemic letter-validity features
- task-specific keyword dictionaries

These belong to Phase 2.

### High-level feature blocks

Use the following blocks:

```text
A. Acoustic library features
B. Fallback acoustic features
C. Pause and fluency timing features
D. Lexical richness features
E. POS and syntactic features
F. Discourse and coherence features
G. Lightweight speech-graph features
G. Task-specific semantic features
H. Dense embedding baselines
```

For Phase 1, use only:

```text
A. Acoustic library features
B. Fallback acoustic features
C. Pause and fluency timing features
D. Lexical richness features
E. POS and syntactic features
F. Discourse and coherence features
G. Lightweight speech-graph features
H. Dense embedding baselines (optional)
```

Reserve:

```text
Task-specific semantic features
```

for Phase 2.

## Recommended feature-library stack

Use feature libraries where they are mature and helpful, but do not let them replace task-aware feature engineering.

| Purpose | Recommended tool | Use in project |
| --- | --- | --- |
| Acoustic baseline | `opensmile` with `eGeMAPSv02` | main acoustic/paralinguistic block |
| Optional large acoustic baseline | `opensmile` with `ComParE_2016` | optional, high-dimensional, likely overfitting risk |
| Fallback audio features | `librosa` | MFCC, RMS, ZCR, duration, pitch fallback |
| Optional pitch / voice sanity checks | `praat-parselmouth` | optional secondary validation for F0, jitter, shimmer |
| POS / dependency features | `spaCy` or `stanza` | universal POS ratios, dependency features |
| Chinese segmentation | `jieba`, spaCy Chinese, or stanza | required for Chinese lexical features |
| Dense text embeddings | multilingual sentence-transformers / XLM-R family | baseline, non-interpretable comparison |
| Dense audio embeddings | wav2vec2 / XLS-R / Whisper encoder features | optional audio baseline |
| Pause features | custom extraction | should not rely on eGeMAPS alone |

Important distinction:

```text
openSMILE = feature extraction library
eGeMAPS   = predefined acoustic feature set within openSMILE
```

## Acoustic feature block

Use `openSMILE` with `eGeMAPSv02` as the main acoustic baseline.

Why:

- compact
- accepted in clinical speech work
- reproducible
- much safer than very high-dimensional acoustic sets for small multilingual cells

Main acoustic block:

```text
ac_eGeMAPS_*
```

Optional comparison block:

```text
ac_compare_*
```

Fallback if `opensmile` is unavailable:

```text
ac_duration
ac_rms_*
ac_zcr_*
ac_f0_*
ac_mfcc_*
```

Acoustic features should remain one feature group among many, not the whole thesis.

## Universal starter feature set

This is the default Phase 1 feature set for the whole benchmark, regardless of task.

### Length and rate features

```text
len_token_count
len_type_count
len_utterance_count
len_mean_utterance_length
len_std_utterance_length
len_audio_duration
len_speech_duration
len_tokens_per_second
len_tokens_per_utterance
len_syllable_count
len_syllables_per_second
```

### Universal lexical features

```text
lex_type_token_ratio
lex_mattr_10
lex_mattr_20
lex_hapax_ratio
lex_repetition_rate
lex_mean_token_length
lex_std_token_length
lex_function_word_ratio
lex_content_word_ratio
```

## Pause and fluency timing block

Pause features should be custom and task-aware.

Use:

- transcript pause markers if present
- timestamps if present
- ASR timestamps if available
- audio-derived silence estimates where feasible

Target pause features:

```text
pause_filled_count
pause_filled_per_100_words
pause_short_count
pause_medium_count
pause_long_count
pause_total_duration
pause_mean_duration
pause_median_duration
pause_std_duration
pause_iqr_duration
pause_pause_to_word_ratio
pause_long_ratio
pause_to_speech_ratio
pause_silence_ratio
pause_ratio_speaking
pause_ratio_nonspeaking
pause_speaking_to_total_ratio
pause_speech_rate_words_per_second
pause_articulation_rate_words_per_speaking_second
```

Interpret pause features within task type, not globally.

## Additional lexical features

Extract broad lexical richness and repetition features:

```text
lex_token_count
lex_type_count
lex_type_token_ratio
lex_mattr_10
lex_mattr_20
lex_brunet
lex_honore
lex_average_token_length
lex_repetition_rate
lex_function_word_ratio
lex_content_word_ratio
lex_lemma_type_token_ratio
```

Chinese tokenization must be language-specific and should use `jieba` or equivalent when available.

## POS and syntactic features

Use universal POS and dependency-style features where tooling allows:

```text
syn_noun_ratio
syn_verb_ratio
syn_pronoun_ratio
syn_adjective_ratio
syn_adverb_ratio
syn_adposition_ratio
syn_determiner_ratio
syn_auxiliary_ratio
syn_proper_noun_ratio
syn_cconj_ratio
syn_sconj_ratio
syn_interjection_ratio
syn_noun_verb_ratio
syn_pronoun_noun_ratio
syn_determiner_noun_ratio
syn_content_function_ratio
syn_open_closed_ratio
syn_mean_dependency_length
syn_mean_utterance_length_words
syn_utterance_count
```

These are useful but should be treated as semi-interpretable rather than perfect clinical measures.

## Discourse and coherence features

For connected speech, extract:

```text
disc_repeated_unigram_ratio
disc_adjacent_utterance_similarity_mean
disc_adjacent_utterance_similarity_std
disc_embedding_dispersion
disc_repeated_bigram_ratio
disc_repeated_trigram_ratio
disc_immediate_repetition_count
disc_mean_pairwise_utt_similarity
disc_semantic_drift
disc_local_coherence
```

If embedding models are unavailable, use TF-IDF cosine similarity fallbacks.

## Lightweight speech-graph features

Add a small optional speech-graph block for the universal Phase 1 baseline. This is a commonly used family in AD language work and is still task-agnostic enough to be useful as a first-pass screen.

Target features:

```text
graph_node_count
graph_edge_count
graph_density
graph_self_loop_ratio
graph_largest_component_ratio
graph_average_degree
```

These should remain optional if graph construction becomes brittle for some transcript formats, but they are worth including in the plan as a standard universal feature family.

## Phase 2: Task-specific semantic features

This remains the scientific core of the final biomarker interpretation, but it should come after the universal baseline is working.

### PD_CTP features

This is the main biomarker module.

Use concept / content-unit dictionaries by language and prompt.

Target features:

```text
pd_unique_units_count
pd_unique_units_ratio
pd_total_unit_mentions
pd_content_density
pd_object_units_ratio
pd_action_units_ratio
pd_keyword_to_nonkeyword_ratio
pd_repeated_content_unit_ratio
pd_semantic_similarity_to_unit_list
```

This is the primary task-specific semantic block for cross-lingual biomarker claims.

### FLUENCY features

For semantic or phonemic fluency style tasks:

```text
ft_item_count
ft_unique_item_count
ft_repetition_count
ft_intrusion_count
ft_valid_item_count
ft_valid_item_ratio
ft_items_per_second
ft_cluster_count
ft_mean_cluster_size
ft_switch_count
ft_letter_valid_count
ft_letter_violation_count
```

WLS should contribute here.

### STORY_NARRATIVE features

For story recall / retelling style prompts:

```text
sr_propositions_recalled_count
sr_propositions_recalled_ratio
sr_key_event_coverage
sr_event_order_score
sr_intrusion_count
sr_semantic_similarity_to_reference_story
sr_repetition_rate
```

### CONVERSATION features

For interview or free conversation speech:

```text
fc_utterance_count
fc_topic_coherence
fc_topic_switch_rate
fc_embedding_dispersion
fc_named_entity_density
fc_repetition_rate
fc_mean_utterance_similarity
```

### READING features

Reading tasks should not be forced into `PD_CTP`.

Treat them as a separate task family and emphasize:

```text
lexical
pause
acoustic
prosodic
error / repetition
coherence-lite
```

Do not make reading-task semantic coverage features central unless prompt-specific reading references are available.

## Dense embedding baselines

Keep dense models as comparison baselines, not the interpretability core.

Use:

```text
emb_text_*
emb_audio_*
```

Examples:

- multilingual sentence-transformers
- XLM-R-based sentence encoders
- wav2vec2 / XLS-R / Whisper encoder embeddings

## Optional lexical abnormality features

If lexicon support is reliable enough for a language, add:

```text
lex_non_dictionary_ratio
lex_oov_ratio
```

These are common approximations of lexical abnormality, but they should be optional because cross-lingual dictionary coverage is uneven.

## Feature metadata and grouping

Every extracted feature should have metadata:

```text
feature_name
feature_group
task_specific
valid_task_types
requires_text
requires_audio
language_dependent_tooling
description
```

Also add:

```text
feature_group_available
feature_group_num_missing
```

for each block.

## Feature reduction strategy

Start broad, then narrow inside train folds only.

Primary model-facing narrowing methods:

```text
top-k ANOVA F-value
optional mutual information
optional L1 logistic selection
```

Target `k` values:

```text
5, 10, 20, 50, 100, all
```

This should happen only inside the training data of each split.

For the universal Phase 1 baseline, target broad starter subsets such as:

```text
all universal handcrafted features
top-k ANOVA universal features
text-only
pause-only
speech-graph-only
acoustic-only
text + pause
text + pause + speech-graph
text + pause + acoustic
optional embedding PCA
```

## Stability analysis

Stability is central, but it should be scoped properly.

Primary final stability analysis:

```text
within PD_CTP across languages
```

Secondary stability analysis:

```text
within FLUENCY
within STORY_NARRATIVE
within CONVERSATION
```

For each feature, compute:

```text
language_coverage
task_coverage
effect_direction
ANOVA rank
effect size
significance count
rank variance
top-rank count
```

Feature classes:

```text
language_general_within_task
task_general
language_specific
task_specific
unstable
```

Important rule:

```text
stability-derived feature sets used for modeling must be computed from training data only
```

It is acceptable to compute full-dataset descriptive stability tables for reporting, but not to use them for leakage-prone feature selection.

## Normalization and imputation

Normalization should remain task-aware:

```text
1. language x task_type x dataset_name
2. language x task_type
3. language
4. global
```

Only fit normalization and imputation inside the training split.

For the universal benchmark-wide feature screening stage, add this explicit rule:

```text
do not run benchmark-wide ANOVA on raw pooled features
```

Instead use one of:

```text
preferred:
    z-score within language x task_type

optional stronger variant:
    residualise language + task_type + dataset_name
    then run ANOVA on the residualised feature
```

The default should be the simpler `language x task_type` normalization.

For classical models:

- median imputation
- missingness indicators
- standardization

## Main experiments

### Experiment 1: Dataset audit

Always start with:

```text
counts by language x task_type x diagnosis
counts by dataset x task_type x diagnosis
missing text / missing audio
participant ID coverage
participant-only text coverage
task label confidence
```

### Experiment 2: Sanity baselines

Train:

- majority baseline
- demographics-only
- language-only
- task-only
- dataset-only
- duration-only

If dataset-only or task-only performs unexpectedly well, treat biomarker claims cautiously.

### Experiment 3: Universal pooled AD_vs_HC baseline

This is the new Phase 1 benchmark-wide screening experiment.

Use:

- all benchmark samples eligible for `AD_vs_HC`
- universal features only
- no task-specific semantic modules
- normalization by `language x task_type`
- ANOVA ranking
- top-k selection inside training folds only

Compare:

```text
all universal handcrafted features
top-k ANOVA universal features
text-only
pause-only
speech-graph-only
acoustic-only
text + pause
text + pause + speech-graph
text + pause + eGeMAPS
optional embedding PCA
```

This experiment is not the final clinical biomarker claim. It is the efficient first-pass feature narrowing stage.

### Experiment 4: Universal pooled AD_vs_HC residualised baseline

Repeat the pooled universal baseline, but residualise:

```text
language + task_type + dataset_name
```

before ANOVA.

This is the stronger benchmark-wide confound-controlled screening variant.

### Experiment 5: Held-out-language universal transfer

Use only universal features.

Procedure:

- select top-k features using training languages only
- evaluate on held-out target language
- compare which broad feature groups survive transfer best

This gives a fast answer about which universal feature types generalise at all before task-specific modules are added.

### Experiment 6: Primary monolingual PD_CTP modeling

Run monolingual within-task experiments for picture description / Cookie Theft only where data are sufficient.

Feature sets:

```text
all_features
core_task_agnostic
task_specific_semantic
semantic_plus_pause
acoustic_only
pause_only
lexical_only
syntactic_only
discourse_only
dense_text_only
dense_audio_only
```

This is the main biomarker experiment.

### Experiment 7: Primary cross-lingual PD_CTP transfer

Train on one or more source languages and test on a target language within `PD_CTP`.

Examples:

```text
en PD_CTP -> zh PD_CTP
en PD_CTP -> el PD_CTP
multi-source -> held-out target language within PD_CTP
```

This is the cleanest cross-lingual biomarker test.

### Experiment 8: Secondary multilingual pooled training

Train on pooled multilingual data and report:

- aggregate performance
- per-language performance
- per-task performance
- per-dataset performance

This is important, but secondary to the within-task biomarker question.

### Experiment 9: Secondary FLUENCY analysis

Use WLS and other fluency-like sources to study fluency-specific biomarkers.

This should be reported as:

```text
task-specific biomarker analysis
```

not as the main cross-lingual semantic biomarker claim.

### Experiment 10: Secondary READING analysis

Use reading-based Spanish subsets such as Ivanova as a separate task family.

This should be analyzed monolingually or as task-specific evidence, not merged into `PD_CTP`.

### Experiment 11: Exploratory held-out-task and held-out-dataset transfer

Keep these as robustness experiments, not primary thesis evidence.

## Models

Prefer simple and interpretable classifiers first:

```text
logistic regression
linear SVM
RBF SVM
random forest
XGBoost if available
```

Fusion options:

```text
early_concat
late_average
late_weighted
stacking
```

Priority fusion comparisons:

```text
text_core + acoustic
semantic + pause
dense_text + acoustic
all handcrafted
handcrafted + dense
all with vs without task-specific semantic features
```

## Reporting

Report at minimum:

```text
balanced accuracy
macro F1
AUROC
AUPRC
sensitivity
specificity
precision
confidence intervals where feasible
```

Important biomarker reporting outputs:

```text
sample count table by language x task x label
feature availability table by language x task
top universal features across the whole benchmark
top universal features by feature group
top biomarkers per language-task
language-general biomarkers within PD_CTP
task-general biomarkers
unstable biomarkers
model comparison tables
transfer-drop tables
ablation plots
```

## Thesis-scoped implementation order

Implement in this order:

1. Manifest builder and dataset audit
2. Participant ID normalization / fallback grouping strategy
3. Text preprocessing
4. Universal starter features:
   - length/rate
   - lexical
   - POS/syntax
   - repetition/coherence
   - pause
   - speech-graph
5. Acoustic baseline with openSMILE eGeMAPSv02 and librosa fallback
6. Feature matrix export and metadata
7. Leakage-safe classical pipelines with ANOVA top-k selection
8. Universal pooled AD_vs_HC baseline
9. Universal held-out-language baseline
10. PD_CTP task-specific semantic module
11. Feature statistics and stability analysis
12. Monolingual and cross-lingual PD_CTP experiments
13. Secondary pooled and task-specific analyses
14. Reporting and plots

## Bottom line

The thesis version of this project should be:

- broad in feature extraction
- phased in complexity
- selective in biomarker claims
- primary on `AD_vs_HC`
- inclusive of WLS
- simple in task taxonomy
- universal-feature-first for fast screening
- strongest on `PD_CTP` cross-lingual biomarker stability

Use `openSMILE/eGeMAPSv02` as the standard acoustic block, but do not let acoustic libraries replace the main contribution:

```text
task-aware multilingual biomarker design
```
