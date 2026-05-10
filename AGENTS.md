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
- Keep processing code under `processing/`, experiment code under `experiments/`, processed artifacts under `data/processed/`, and generated result tables/logs under `tables/experiment-results/`.
- Prefer lowercase, hyphenated directory names for any new top-level or organizational folders.
