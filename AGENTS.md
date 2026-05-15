Before making edits in this repository, always review both of these sources first:

1. `MultiConAD-archive/MultiConAD-main/`
   Use this as the original code reference for intended pipeline behavior, dataset handling, and naming.
2. `2024_multiconad_multilingual_alzheimer_shakeri_farmanbar_balog_annotated.pdf`
   Use this as the paper reference for the stated dataset composition, preprocessing assumptions, and experiment setup.

Rules:

- Do not modify anything inside `MultiConAD-archive/`.
- Treat `MultiConAD-archive/MultiConAD-main/` as read-only reference code.
- When current code differs from the original pipeline, check the archive code and the paper before changing behavior.
- For Chinese dataset changes in particular, verify against the original `text_cleaning_Chinese.py` in the archive and the paper's Chinese dataset description before editing.
- Keep all live replication inputs under `data/`. If auxiliary metadata or source assets arrive elsewhere, move or copy the necessary files into the matching location under `data/` before wiring or updating the pipeline.
- Keep processing code under `processing/`, experiment code under `experiments/`, processed artifacts under `data/processed/`, and generated result tables/logs under `tables/`.
- Organise `tables/` outputs by the experimental plan first, then by artifact purpose:
- `01-baselines/embedding-baselines/` for sparse/dense transcript baseline outputs, embedding caches, multiseed-suite artifacts, and pipeline queue logs.
- `01-baselines/feature-baselines/` for general feature-set baseline outputs and Phase 1 manifest/feature audit tables.
- `01-baselines/transfer-learning-baselines/` for BERT/RoBERTa, CLS fine-tuning, and prompt-based transfer-learning baseline outputs.
- `02-ablation-task-specific-features/` for Phase 2 task-specific feature sweeps, clean prompt sweeps, task-feature projection assets, and related logs.
- `03-ablation-translingual-language-specific/` for monolingual ranking, top-k feature experiments, cross-language feature consistency, and saliency-map outputs.
- `04-ablation-asr-robustness/`, `05-ablation-llm-prompting/`, `06-ablation-token-interpretability/`, and `07-ablation-fusion-settings/` for the corresponding later ablations.
- Keep dataset audit/completion material that is not a model result under `tables/dataset-completion/`.
- `result-tables/` for consolidated human-readable `.txt` result reports; combine related tables/analyses into one readable file instead of producing many small text files.
- `result-tables/csv/` for raw sweep CSVs, ranking CSVs, metric CSVs, and machine-readable tabular outputs.
- `summaries/` for JSON run indexes, run states, smoke summaries, manifest summaries, and compact aggregate suite summaries.
- `logs/`, `job_logs/`, or `launcher_logs/` for runtime logs.
- `cache/`, `embedding-cache/`, `raw_results/`, `seed_data/`, and `partial_summaries/` for reusable intermediate run artifacts.
- `report_assets/` for figures, saliency maps, projections, and report-supporting image/data assets.
- `key-results/` for concise markdown findings or narrative result summaries inside a run area.
- Use top-level `tables/highlights/` only for curated pointers or copies of the clearest key results; do not use it as a dumping ground for full sweeps.
- Prefer lowercase, hyphenated directory names for any new top-level or organizational folders.
- Use `reports/13.05/` only for curated `.txt`, result tables, analysis, and figures; keep it lightweight by copying only the clearest results intended for weekly report transfer.
- Treat HC-oriented median + IQR saliency overlays as the canonical saliency-map output format going forward; use fixed pooled and language-specific versions for comparison, and reserve single-sample / paired-sample saliency plots for explicit case-level inspection only.

## Experimental Plan

### 1. Baselines

#### 1.1 Embedding-based baselines
- Train DT, RF, SVM, and LR using sparse and dense transcript embeddings.
- Run over 5 random seeds.
- Use the same hyperparameter search/tuning protocol as MultiConAD.
- Apply the same tuning protocol to all later classical ML experiments.
- Evaluate individual models and simple method ensembling.

#### 1.2 Feature-based baselines
- Train DT, RF, SVM, and LR using the full general feature set (excluding task-specific features).
- Concatenate and normalise all features.
- Use the same feature set across all languages.
- This establishes the main interpretable feature-based baseline.

#### 1.3 Transfer-learning baselines
- Compare three transfer-learning approaches:
- DT, RF, SVM, and LR using BERT/RoBERTa-derived embeddings.
- Fine-tuned BERT/RoBERTa with CLS-token classification.
- Prompt-based fine-tuning on transcripts, with and without pause encoding (Jacksonâ€™s method).

### 2. Ablation: Task-specific features
- MultiConAD contains a wide range of task types, so some clinically meaningful features may only apply to specific tasks.
- Add task-specific features to the general feature set.
- Only apply task-specific features where the relevant task domain is present.
- Repeat DT, RF, SVM, and LR using concatenated and normalised features.
- Compare against:
- embedding-based baselines;
- feature-based baselines excluding task-specific features.

### 3. Ablation: Translingual and language-specific features
- Use feature selection to identify which features are consistently useful across languages and which appear language-specific.
- Keep predictive top-k experiments comparable with earlier full feature-set runs: use the same model families, tuning protocol, train/test grouping, ranking modes, and k grid wherever possible.
- Treat language/task specificity analysis as an interpretability layer on top of predictive ranking results, not as a replacement for ANOVA/Welch top-k methodology unless explicitly running a new effect-size-ranking experiment.

#### 3.1 Monolingual feature ranking
- For each language:
- Train simple monolingual classifiers.
- Rank features using both supported ranking modes:
- ANOVA ranking, to preserve comparability with the original top-k feature sweeps.
- Welch t-test ranking with 5% Bonferroni correction, matching the saliency/feature-interpretability ranking used in later analyses.
- Store raw p-values, corrected p-values, effect sizes, feature family, and selected feature names in `result-tables/csv/`.
- Put the human-readable comparison of ranking methods and strongest features in a consolidated `.txt` file under `result-tables/`.

#### 3.2 Top-k feature experiments
- Re-train DT, RF, SVM, and LR using top-k features for each language:
- Use the standard comparable k grid: `k = 5, 10, 20, 50, 100, all`.
- Run the same k grid for core/general features excluding task-specific features, task-specific features only, and extended/full feature sets including task-specific features.
- Use full tuning for classical ML comparability unless a run is explicitly marked as a fixed-config smoke test.
- Report mean +/- sd over the planned seeds where multiseed results are requested.

#### 3.3 Feature consistency analysis
- Analyse which features:
- appear repeatedly across languages, suggesting possible translingual candidate markers;
- appear mainly in one language after task context is considered, suggesting language-sensitive candidate markers;
- vary by task domain, suggesting task sensitivity or task confounding rather than true language specificity.
- Use the paper-style task labels consistently in reports and figures: PD = Picture Description; FT = Fluency Task; SR = Story Retelling; FC = Free Conversation; NA = Narrative.
- Also retain the live manifest task labels actually used in computation, such as `PD_CTP`, `READING`, `COMMAND`, `CONVERSATION`, `MIXED_PROTOCOL`, `PICTURE_RECALL`, `REPETITION`, `MOTOR_SPEECH`, `PROCEDURAL`, and `OTHER`.
- Distinguish source-level task coverage from sample-level manifest labels: source-level coverage can be broader than the currently exposed live task cells.

#### 3.4 Language-task effect-size interpretation
- Use simplified effect-size mode as the default first implementation for visualising language/task specificity.
- For each feature and each valid language-task cell, split AD vs HC/control and compute signed Cohen's d:
- `d = (mean_AD - mean_HC) / pooled_std`.
- Store signed d, absolute d, direction, p-value where used, corrected p-value where used, and AD/HC group counts.
- Only compute cell-level effects where each class has enough groups; use at least 5 AD and 5 HC for exploratory analysis, and flag stronger evidence at 10 AD and 10 HC.
- Compute feature-level interpretation scores:
- diagnostic strength = mean absolute Cohen's d across valid cells;
- direction consistency = proportion of valid cells sharing the modal AD/HC direction;
- language specificity = standard deviation of signed language-averaged diagnosis effects;
- task specificity = standard deviation of signed task-averaged diagnosis effects.
- Use signed Cohen's d, not absolute d, for language and task specificity so direction flips are treated as instability or specificity.
- Mark features with too few valid cells as sparse/confounded, not language-sensitive.
- Interpret score patterns as:
- low task specificity + low language specificity + strong diagnostic strength = translingual candidate;
- high language specificity + low task specificity = language-sensitive candidate;
- low language specificity + high task specificity = task-sensitive feature;
- high language specificity + high task specificity = language-task confounded;
- low evidence coverage or direction flips = unstable / insufficient evidence.
- Use regression interaction mode only as a robustness check for top selected features:
- `feature_value ~ diagnosis + language + task + diagnosis:language + diagnosis:task`.
- Prefer the simplified effect-size method for the main scatter plot and heatmap because it is more transparent and easier to debug on sparse, partially confounded cells.

#### 3.5 Best feature subset per language
- Identify the best-performing feature subset for each language by pruning top-k features until performance peaks.
- Use the effect-size interpretation layer to support interpretable feature saliency maps and language/task visualisations.
- Frame claims cautiously because task type and language are partially confounded in the available benchmark.

### 4. Ablation: ASR robustness
- Evaluate whether improved ASR, alignment, and diarisation affect downstream feature selection and classification.
- Apply WhisperX in multilingual settings to improve alignment and diarisation.
- Check transcript quality on a per-dataset basis.
- Verify whether interviewer prompts/scripts are currently being removed correctly.
- Repeat Ablation 2 using improved transcripts.
- Keep acoustic features unchanged, so differences can be attributed mainly to transcript/ASR quality.

### 5. Ablation: LLM prompting (Extension)
- Use the best features from Ablation 2 as structured prompt information.
- Encode top-performing features as qualitative labels prepended to transcripts.
- Compare prompting settings:
- target-language few-shot examples only;
- multilingual few-shot examples;
- other-language examples only.
- Test prompt variants using:
- translingual feature summaries;
- language-specific feature summaries;
- combined feature summaries.

### 6. Ablation: Token interpretability (Extension)
- Extend prior MoE-style interpretability to the multilingual setting.
- This is valuable but lower priority than feature-based interpretability, because token-level MoE analysis may not transfer cleanly across languages.
- Train MoE models separately in each monolingual setting.
- Compare token importance by lexical and semantic category.
- Analyse whether important tokens correspond to language-specific lexical artefacts or broader cognitive markers.
- Use BERT/RoBERTa token-level attribution as a simpler backup interpretability method.

### 7. Ablation: Further fusion settings (Extension)
- Explore whether combining feature-based and embedding-based signals improves performance.
- Add selected feature groups to BERT/RoBERTa-based models.
- Add sparse, dense, and BERT/RoBERTa embeddings to feature-based classifiers.
- Compare accuracy gains against interpretability loss.
- Evaluate whether fusion improves held-out-language robustness.
