Results and run artifacts live under `tables/`, grouped by the experimental plan.

Current convention:

- `dataset-completion/`: dataset audit notes and label-completion checks.
- `01-baselines/embedding-baselines/`: sparse/dense transcript baselines, embedding caches, multiseed-suite artifacts, and pipeline queue logs.
- `01-baselines/feature-baselines/`: general feature-set baselines plus Phase 1 manifest/feature audit outputs.
- `01-baselines/transfer-learning-baselines/`: BERT/RoBERTa, CLS fine-tuning, and prompt-based transfer-learning baselines.
- `02-ablation-task-specific-features/`: Phase 2 task-specific feature sweeps, clean prompt sweeps, projection assets, and related logs.
- `03-ablation-translingual-language-specific/`: monolingual ranking, top-k feature experiments, feature consistency analysis, and saliency-map outputs.
- `04-ablation-asr-robustness/`: ASR robustness results.
- `05-ablation-llm-prompting/`: LLM prompting extension outputs.
- `06-ablation-token-interpretability/`: token-level interpretability outputs.
- `07-ablation-fusion-settings/`: feature/embedding fusion outputs.
- `highlights/`: curated pointers or copies of the clearest result summaries for quick review.

Within each area or run folder:

- `result-tables/`: consolidated human-readable `.txt` result reports.
- `result-tables/csv/`: raw sweep CSVs, ranking CSVs, metric CSVs, and machine-readable tabular outputs.
- `summaries/`: JSON run indexes, states, smoke summaries, manifest summaries, and compact aggregate suite summaries.
- `logs/`, `job_logs/`, `launcher_logs/`: runtime logs.
- `cache/`, `embedding-cache/`, `raw_results/`, `seed_data/`, `partial_summaries/`: reusable intermediate artifacts.
- `report_assets/`: figures, projections, saliency maps, and report-supporting assets.
- `key-results/`: concise markdown findings and narrative result summaries.
