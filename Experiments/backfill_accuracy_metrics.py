from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.phase1.run_rich_sweep as p1
import experiments.phase2.run_clean_prompt_sweep as p2_clean
import experiments.phase2.run_phase2_sweep as p2

from processing.phase1.common import make_logger


PHASE1_RUN_ROOT = p1.RICH_SWEEP_ROOT
PHASE2_RUN_ROOT = p2.RUN_ROOT
PHASE2_CLEAN_ROOT = p2_clean.CLEAN_ROOT


@dataclass(frozen=True)
class BackfillRun:
    run_name: str
    csv_path: Path
    summary_path: Path
    test_pos: int
    test_neg: int
    rerun_callback: callable | None


def compute_accuracy_column(df: pd.DataFrame, pos_count: int, neg_count: int) -> pd.Series:
    total = pos_count + neg_count
    return ((df["sensitivity"] * pos_count) + (df["specificity"] * neg_count)) / total


def best_key_from_row(row: pd.Series) -> tuple[str, str, str, str]:
    subset_col = "subset" if "subset" in row.index else "subset_name"
    return (
        str(row[subset_col]),
        str(row["top_k"]),
        str(row["model_family"]),
        str(row["model_variant"]),
    )


def update_results_file(run: BackfillRun, log) -> bool:
    if not run.csv_path.exists() or not run.summary_path.exists():
        log(f"Skipping {run.run_name}: missing result files")
        return False

    results = pd.read_csv(run.csv_path)
    if "accuracy" not in results.columns:
        results["accuracy"] = compute_accuracy_column(results, run.test_pos, run.test_neg)
    else:
        results["accuracy"] = compute_accuracy_column(results, run.test_pos, run.test_neg)

    results = results.sort_values(["accuracy", "auroc", "balanced_accuracy", "macro_f1"], ascending=False)
    results.to_csv(run.csv_path, index=False)

    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    old_best = summary.get("best_result") or summary.get("best_model") or {}
    old_key = None
    if old_best:
        old_key = best_key_from_row(pd.Series(old_best))

    new_best = results.iloc[0].to_dict()
    new_key = best_key_from_row(pd.Series(new_best))

    if "best_result" in summary:
        summary["best_result"] = new_best
        if "best_config" in summary:
            summary["best_config"].update(
                {
                    "subset": new_key[0],
                    "top_k": new_key[1],
                    "model_family": new_key[2],
                    "model_variant": new_key[3],
                    "num_features": int(results.iloc[0].get("num_features", summary["best_config"].get("num_features", 0))),
                }
            )
    if "best_model" in summary:
        summary["best_model"] = new_best
    summary["test_class_counts"] = {"0": int(run.test_neg), "1": int(run.test_pos)}
    summary["primary_selection_metric"] = "raw accuracy"
    run.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    changed = old_key != new_key
    if changed:
        log(f"{run.run_name}: best config changed under raw accuracy from {old_key} to {new_key}")
    else:
        log(f"{run.run_name}: best config unchanged; accuracy backfilled")
    return changed


def phase1_run_specs() -> list[p1.RunSpec]:
    return [
        p1.RunSpec("benchmark_wide_rich", "pooled", {}, [["language", "task_type", "dataset_name"], ["language", "task_type"], ["language"], []]),
        p1.RunSpec("pd_ctp_pooled_rich", "task_primary", {"task_type": "PD_CTP"}, [["language", "dataset_name"], ["language"], []]),
        p1.RunSpec("language_en_all_rich", "language_all", {"language": "en"}, [["task_type", "dataset_name"], ["task_type"], []]),
        p1.RunSpec("language_es_all_rich", "language_all", {"language": "es"}, [["task_type", "dataset_name"], ["task_type"], []]),
        p1.RunSpec("language_zh_all_rich", "language_all", {"language": "zh"}, [["task_type", "dataset_name"], ["task_type"], []]),
        p1.RunSpec("language_el_all_rich", "language_all", {"language": "el"}, [["task_type", "dataset_name"], ["task_type"], []]),
        p1.RunSpec("language_en_pd_ctp_rich", "language_pd_ctp", {"language": "en", "task_type": "PD_CTP"}, [["dataset_name"], []]),
        p1.RunSpec("language_zh_pd_ctp_rich", "language_pd_ctp", {"language": "zh", "task_type": "PD_CTP"}, [["dataset_name"], []]),
        p1.RunSpec("language_el_pd_ctp_rich", "language_pd_ctp", {"language": "el", "task_type": "PD_CTP"}, [["dataset_name"], []]),
        p1.RunSpec("language_es_reading_rich", "language_task_secondary", {"language": "es", "task_type": "READING"}, [["dataset_name"], []]),
    ]


def load_phase1_df() -> pd.DataFrame:
    manifest = pd.read_json(p1.MANIFEST_PATH, lines=True)
    features = pd.read_csv(p1.FEATURES_PATH)
    df = manifest.merge(
        features,
        on=["sample_id", "group_id", "dataset_name", "language", "task_type", "diagnosis_mapped", "binary_label"],
    )
    return df[df["binary_label"].isin([0, 1])].copy()


def phase1_backfill_runs(log) -> list[BackfillRun]:
    df = load_phase1_df()
    runs: list[BackfillRun] = []
    for spec in phase1_run_specs():
        run_df = p1.dataset_for_spec(df, spec)
        train_df, test_df = p1.grouped_train_test_split(run_df, test_size=0.2, seed=p1.SEED)
        counts = test_df["binary_label"].value_counts().to_dict()
        runs.append(
            BackfillRun(
                run_name=spec.name,
                csv_path=PHASE1_RUN_ROOT / f"{spec.name}_model_results.csv",
                summary_path=PHASE1_RUN_ROOT / f"{spec.name}_summary.json",
                test_pos=int(counts.get(1, 0)),
                test_neg=int(counts.get(0, 0)),
                rerun_callback=lambda spec=spec, run_df=run_df: p1.run_feature_sweep(run_df, spec.name, spec.grouping_levels, log, PHASE1_RUN_ROOT),
            )
        )
    return runs


def phase2_backfill_runs(log) -> list[BackfillRun]:
    df = pd.read_csv(p2.FEATURES_PATH)
    runs: list[BackfillRun] = []
    for spec in p2.run_specs():
        filtered = p2.apply_filters(df, spec.data_filter)
        filtered = filtered[filtered["binary_label"].isin([0, 1])].copy().dropna(subset=["group_id"])
        train_df, test_df = p2.grouped_train_test_split(filtered, test_size=0.2, seed=p2.SEED)
        counts = test_df["binary_label"].value_counts().to_dict()
        runs.append(
            BackfillRun(
                run_name=spec.name,
                csv_path=PHASE2_RUN_ROOT / f"{spec.name}_model_results.csv",
                summary_path=PHASE2_RUN_ROOT / f"{spec.name}_summary.json",
                test_pos=int(counts.get(1, 0)),
                test_neg=int(counts.get(0, 0)),
                rerun_callback=lambda spec=spec, df=df: p2.run_spec(df, spec, log),
            )
        )
    return runs


def phase2_clean_backfill_runs(log) -> list[BackfillRun]:
    df = pd.read_csv(p2_clean.PHASE2_ROOT / "phase2_features.csv")
    runs: list[BackfillRun] = []
    for spec in p2_clean.clean_run_specs():
        filtered = p2.apply_filters(df, spec.data_filter)
        filtered = filtered[filtered["binary_label"].isin([0, 1])].copy().dropna(subset=["group_id"])
        train_df, test_df = p2.grouped_train_test_split(filtered, test_size=0.2, seed=p2.SEED)
        counts = test_df["binary_label"].value_counts().to_dict()
        runs.append(
            BackfillRun(
                run_name=spec.name,
                csv_path=PHASE2_CLEAN_ROOT / f"{spec.name}_model_results.csv",
                summary_path=PHASE2_CLEAN_ROOT / f"{spec.name}_summary.json",
                test_pos=int(counts.get(1, 0)),
                test_neg=int(counts.get(0, 0)),
                rerun_callback=lambda spec=spec, df=df: _rerun_phase2_clean(spec, df, log),
            )
        )
    return runs


def _rerun_phase2_clean(spec, df, log):
    old_root = p2.RUN_ROOT
    try:
        p2.RUN_ROOT = PHASE2_CLEAN_ROOT
        p2.RUN_ROOT.mkdir(parents=True, exist_ok=True)
        return p2.run_spec(df, spec, log)
    finally:
        p2.RUN_ROOT = old_root


def regenerate_reports(log) -> None:
    import experiments.phase1.generate_all_task_feature_comparison as p1_compare
    import experiments.phase1.generate_report_artifacts as p1_report
    import experiments.phase2.generate_language_feature_summary as p2_lang

    p1_compare.main()
    p1_report.main()
    p2_lang.main()
    log("Regenerated Phase 1 and Phase 2 report summaries using raw accuracy as the primary sort metric")


def main() -> None:
    log = make_logger("backfill_accuracy_metrics")
    changed_runs: list[BackfillRun] = []

    for run in phase1_backfill_runs(log) + phase2_backfill_runs(log) + phase2_clean_backfill_runs(log):
        changed = update_results_file(run, log)
        if changed and run.rerun_callback is not None:
            changed_runs.append(run)

    for run in changed_runs:
        log(f"Rerunning {run.run_name} because the best config changed under raw accuracy")
        run.rerun_callback()

    regenerate_reports(log)
    log(f"Backfilled accuracy for all Phase 1/2 runs; reran {len(changed_runs)} changed runs")


if __name__ == "__main__":
    main()
