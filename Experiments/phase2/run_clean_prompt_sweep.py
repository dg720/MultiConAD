import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.phase1.common import make_logger, write_json
from processing.phase2.common import PHASE2_ROOT, TABLES_PHASE2_ROOT
import experiments.phase2.run_phase2_sweep as sweep


CLEAN_ROOT = TABLES_PHASE2_ROOT / "clean_prompt_sweep"
CLEAN_ROOT.mkdir(parents=True, exist_ok=True)


def clean_run_specs():
    return [
        sweep.RunSpec(
            name="cookie_theft_pooled_clean_phase2",
            data_filter={"task_type": "PD_CTP", "phase2_prompt_family": "cookie_theft"},
            grouping_levels=[["language", "dataset_name"], ["language"], []],
            min_rows=150,
            min_groups=50,
        ),
        sweep.RunSpec(
            name="language_en_cookie_theft_clean_phase2",
            data_filter={"language": "en", "task_type": "PD_CTP", "phase2_prompt_family": "cookie_theft"},
            grouping_levels=[["dataset_name"], []],
            min_rows=120,
            min_groups=40,
        ),
        sweep.RunSpec(
            name="language_zh_cookie_theft_clean_phase2",
            data_filter={"language": "zh", "task_type": "PD_CTP", "phase2_prompt_family": "cookie_theft"},
            grouping_levels=[["dataset_name"], []],
            min_rows=60,
            min_groups=20,
        ),
        sweep.RunSpec(
            name="language_el_lion_scene_clean_phase2",
            data_filter={"language": "el", "task_type": "PD_CTP", "phase2_prompt_family": "lion_scene"},
            grouping_levels=[["dataset_name"], []],
            min_rows=60,
            min_groups=20,
        ),
        sweep.RunSpec(
            name="language_es_reading_clean_phase2",
            data_filter={"language": "es", "task_type": "READING"},
            grouping_levels=[["dataset_name"], []],
            min_rows=100,
            min_groups=20,
        ),
    ]


def main():
    sweep.RUN_ROOT = CLEAN_ROOT
    sweep.RUN_ROOT.mkdir(parents=True, exist_ok=True)
    log = make_logger("phase2_clean_prompt_sweep")
    df = pd.read_csv(PHASE2_ROOT / "phase2_features.csv")
    summaries = []
    skipped = []
    for spec in clean_run_specs():
        try:
            summary = sweep.run_spec(df, spec, log)
            if summary is None:
                skipped.append({"run_name": spec.name, "reason": "min_rows_or_groups"})
            else:
                summaries.append(summary)
        except Exception as exc:
            skipped.append({"run_name": spec.name, "reason": str(exc)})
            log(f"Failed {spec.name}: {exc}")
    write_json(CLEAN_ROOT / "clean_prompt_run_index.json", {"completed_runs": summaries, "skipped_runs": skipped})
    pd.DataFrame(skipped).to_csv(CLEAN_ROOT / "clean_prompt_skipped_runs.csv", index=False)


if __name__ == "__main__":
    main()
