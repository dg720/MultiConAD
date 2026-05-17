# Transfer-Learning Baselines Plan

This plan implements the `AGENTS.md` objective `1.3 Transfer-learning baselines`:

- DT, RF, SVM, and LR using BERT/RoBERTa-derived embeddings.
- Fine-tuned BERT/RoBERTa with CLS-token classification.
- Prompt-based fine-tuning on transcripts, with and without pause encoding.

## Scope

- Use the existing MultiConAD train/test structure, label normalization, binary/multiclass tasks, and 5-seed protocol used by the TF-IDF and E5 baselines.
- Store all outputs under `tables/01-baselines/transfer-learning-baselines/`.
- Keep reusable embedding/model caches under that run area, not under `data/`.
- Treat `MultiConAD-archive/` and the annotated MultiConAD paper as read-only references for intended split/task behavior.

## Stage 1: Frozen PLM Embeddings

Goal: create the safest comparable baseline before end-to-end fine-tuning.

- Extract transcript embeddings from `bert-base-multilingual-cased`, `xlm-roberta-base`, and translated-English `roberta-base`.
- Compare pooling modes: CLS/pooler embedding and mean-token pooling.
- Train DT, RF, SVM, and LR with the same grid-search protocol as TF-IDF/E5.
- Run monolingual, combined-multilingual, and combined-translated settings for binary and multiclass tasks.
- Report per-seed and mean +/- sd accuracy, plus confusion matrices for multiclass.

## Stage 2: Fine-Tuned CLS Classifiers

Goal: test the standard literature baseline while controlling for small-data instability.

- Fine-tune `AutoModelForSequenceClassification` using CLS-style sequence classification.
- Primary model: `xlm-roberta-base` for original multilingual transcripts.
- Secondary models: `bert-base-multilingual-cased` and translated-English `roberta-base`.
- Use short, conservative hyperparameter grids: learning rate, epochs, batch size, weight decay.
- Report all seeds, not just best runs, because BERT-style fine-tuning is high variance on small AD transcript datasets.

## Stage 3: Length Handling Ablation

Goal: make CLS baselines honest for long English transcripts.

- Baseline mode: first-512-token truncation, for direct comparability with common BERT sequence-classification setups.
- Robust mode: split long transcripts into token windows and average chunk CLS embeddings or logits at transcript level.
- Keep both modes explicit in output filenames and result tables.
- Interpret truncation-heavy English results cautiously; Chinese and Greek mostly fit within 512 tokens, while English has a substantial long-transcript tail.

## Stage 4: Prompt and Pause Extensions

Goal: add literature-inspired transfer-learning variants after plain baselines are stable.

- Add prompt-based fine-tuning only after Stage 1 and Stage 2 outputs are reproducible.
- Compare transcript-only prompts against prompts augmented with available disfluency/pause indicators.
- Mark pause-encoded runs as conditional, because pause availability and alignment quality are inconsistent across source datasets.
- Keep prompt/pause results separate from plain BERT/RoBERTa baselines to avoid overstating comparability.

## Minimum Deliverables

- `experiments/transfer_learning_embeddings.py`: frozen PLM embedding extraction and classical classifier runs.
- `experiments/transfer_learning_finetune.py`: CLS sequence-classification fine-tuning.
- `tables/01-baselines/transfer-learning-baselines/result-tables/transfer_learning_results.txt`: consolidated readable results.
- `tables/01-baselines/transfer-learning-baselines/result-tables/csv/`: raw per-run metrics, best parameters, and confusion matrices.
- `tables/01-baselines/transfer-learning-baselines/summaries/`: compact JSON run indexes and aggregate summaries.
- `tables/01-baselines/transfer-learning-baselines/logs/`: runtime logs.

## Evaluation Rules

- Use accuracy for direct comparability with the original MultiConAD tables.
- Also store macro-F1 and per-class precision/recall, especially for multiclass AD/MCI/HC.
- Preserve language, task, training setting, translation setting, seed, model, pooling mode, and length mode in every result row.
- Compare transfer-learning baselines against TF-IDF and E5, not only against each other.
