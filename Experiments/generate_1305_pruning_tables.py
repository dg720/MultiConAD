from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.phase1.run_rich_sweep as p1
import experiments.phase2.run_phase2_sweep as p2
from experiments.phase1.generate_all_task_feature_comparison import parse_best_text_baselines


REPORT_ROOT = PROJECT_ROOT / "13.05-report"
TEXT_SUMMARY_PATH = PROJECT_ROOT / "tables" / "01-baselines" / "embedding-baselines" / "multiseed-suite" / "result-tables" / "paper_vs_ours_3tables.txt"
K_VALUES: list[int | str] = ["all", 100, 50, 25, 10]


PHASE1_SPECS = [
    {
        "label": "Pooled",
        "run_name": "benchmark_wide_rich",
        "filters": {},
        "grouping_levels": [["language", "task_type", "dataset_name"], ["language", "task_type"], ["language"], []],
    },
    {
        "label": "English",
        "run_name": "language_en_all_rich",
        "filters": {"language": "en"},
        "grouping_levels": [["task_type", "dataset_name"], ["task_type"], []],
    },
    {
        "label": "Spanish",
        "run_name": "language_es_all_rich",
        "filters": {"language": "es"},
        "grouping_levels": [["task_type", "dataset_name"], ["task_type"], []],
    },
    {
        "label": "Chinese",
        "run_name": "language_zh_all_rich",
        "filters": {"language": "zh"},
        "grouping_levels": [["task_type", "dataset_name"], ["task_type"], []],
    },
    {
        "label": "Greek",
        "run_name": "language_el_all_rich",
        "filters": {"language": "el"},
        "grouping_levels": [["task_type", "dataset_name"], ["task_type"], []],
    },
]


PHASE2_SPECS = [
    {
        "label": "English Cookie Theft",
        "run_name": "language_en_cookie_theft_clean_phase2",
        "filters": {"language": "en", "task_type": "PD_CTP", "phase2_prompt_family": "cookie_theft"},
        "grouping_levels": [["dataset_name"], []],
    },
    {
        "label": "Chinese Cookie Theft",
        "run_name": "language_zh_cookie_theft_clean_phase2",
        "filters": {"language": "zh", "task_type": "PD_CTP", "phase2_prompt_family": "cookie_theft"},
        "grouping_levels": [["dataset_name"], []],
    },
    {
        "label": "Greek Lion Scene",
        "run_name": "language_el_lion_scene_clean_phase2",
        "filters": {"language": "el", "task_type": "PD_CTP", "phase2_prompt_family": "lion_scene"},
        "grouping_levels": [["dataset_name"], []],
    },
    {
        "label": "Spanish Reading",
        "run_name": "language_es_reading_clean_phase2",
        "filters": {"language": "es", "task_type": "READING"},
        "grouping_levels": [["dataset_name"], []],
    },
]


def load_phase1_df() -> pd.DataFrame:
    manifest = pd.read_json(p1.MANIFEST_PATH, lines=True)
    features = pd.read_csv(p1.FEATURES_PATH)
    df = manifest.merge(
        features,
        on=["sample_id", "group_id", "dataset_name", "language", "task_type", "diagnosis_mapped", "binary_label"],
    )
    return df[df["binary_label"].isin([0, 1])].copy()


def best_model_for_k_phase1(
    df: pd.DataFrame,
    filters: dict[str, object],
    grouping_levels: list[list[str]],
    subset_name: str,
    k: int | str,
) -> dict[str, object]:
    run_df = p1.dataset_for_spec(df, p1.RunSpec("tmp", "tmp", filters, grouping_levels))
    train_df, test_df = p1.grouped_train_test_split(run_df, test_size=0.2, seed=p1.SEED)
    feature_cols = p1.feature_subset_columns(run_df, subset_name)
    train_norm, test_norm = p1.normalize_with_fallback(train_df, test_df, feature_cols, grouping_levels)
    y_train = train_norm["binary_label"].astype(int)
    y_test = test_norm["binary_label"].astype(int)
    selected_cols, _ = p1.select_top_k(train_norm, y_train, feature_cols, k)
    x_train = train_norm[selected_cols]
    x_test = test_norm[selected_cols]

    best = None
    for spec in p1.model_specs():
        estimator = clone(spec["estimator"])
        estimator.fit(x_train, y_train)
        pred = estimator.predict(x_test)
        if hasattr(estimator, "predict_proba"):
            scores = estimator.predict_proba(x_test)[:, 1]
        else:
            scores = estimator.decision_function(x_test)
        acc = float(accuracy_score(y_test, pred))
        try:
            auroc = float(roc_auc_score(y_test, scores))
        except Exception:
            auroc = float("nan")
        row = {
            "accuracy": acc,
            "auroc": auroc,
            "model": f"{spec['model_family']}:{spec['model_variant']}",
            "num_features": len(selected_cols),
        }
        if best is None or row["accuracy"] > best["accuracy"] or (
            row["accuracy"] == best["accuracy"] and row["auroc"] > best["auroc"]
        ):
            best = row
    assert best is not None
    return best


def best_model_for_k_phase2(
    df: pd.DataFrame,
    filters: dict[str, object],
    grouping_levels: list[list[str]],
    subset_name: str,
    k: int | str,
) -> dict[str, object]:
    filtered = p2.apply_filters(df, filters)
    filtered = filtered[filtered["binary_label"].isin([0, 1])].copy().dropna(subset=["group_id"])
    train_df, test_df = p2.grouped_train_test_split(filtered, test_size=0.2, seed=p2.SEED)
    feature_cols = p2.feature_subset_columns(filtered, subset_name)
    train_norm, test_norm = p2.normalize_with_fallback(train_df, test_df, feature_cols, grouping_levels)
    y_train = train_norm["binary_label"].astype(int)
    y_test = test_norm["binary_label"].astype(int)
    selected_cols, _ = p2.select_top_k(train_norm, y_train, feature_cols, k)
    x_train = train_norm[selected_cols]
    x_test = test_norm[selected_cols]

    best = None
    for spec in p2.model_specs():
        estimator = clone(spec["estimator"])
        estimator.fit(x_train, y_train)
        pred = estimator.predict(x_test)
        if hasattr(estimator, "predict_proba"):
            scores = estimator.predict_proba(x_test)[:, 1]
        else:
            scores = estimator.decision_function(x_test)
        acc = float(accuracy_score(y_test, pred))
        try:
            auroc = float(roc_auc_score(y_test, scores))
        except Exception:
            auroc = float("nan")
        row = {
            "accuracy": acc,
            "auroc": auroc,
            "model": f"{spec['model_family']}:{spec['model_variant']}",
            "num_features": len(selected_cols),
        }
        if best is None or row["accuracy"] > best["accuracy"] or (
            row["accuracy"] == best["accuracy"] and row["auroc"] > best["auroc"]
        ):
            best = row
    assert best is not None
    return best


def markdown_like_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    header_line = " ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-" * len(header_line)
    data_lines = [" ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    return "\n".join([header_line, sep_line] + data_lines)


def build_phase1_tables() -> str:
    df = load_phase1_df()
    text = parse_best_text_baselines(TEXT_SUMMARY_PATH)

    summary_rows = []
    sensitivity_rows = []
    model_rows = []

    for spec in PHASE1_SPECS:
        run_df = pd.read_csv(p1.RICH_SWEEP_RESULT_TABLES / f"{spec['run_name']}_model_results.csv")
        best_overall = (
            run_df[run_df["subset"] == "all_universal"]
            .sort_values(["accuracy", "auroc", "balanced_accuracy", "macro_f1"], ascending=False)
            .iloc[0]
        )
        mono = multi = translated = ""
        if spec["label"] != "Pooled":
            mono = f"{float(text[(text['section'] == 'Monolingual') & (text['language'] == spec['label'])].iloc[0]['best_text_accuracy']):.1f}"
            multi = f"{float(text[(text['section'] == 'Multilingual-Combined') & (text['language'] == spec['label'])].iloc[0]['best_text_accuracy']):.1f}"
            translated = f"{float(text[(text['section'] == 'Translated-Combined') & (text['language'] == spec['label'])].iloc[0]['best_text_accuracy']):.1f}"
        summary_rows.append(
            [
                spec["label"],
                mono,
                multi,
                translated,
                f"{best_overall['accuracy'] * 100:.1f}",
                f"{best_overall['model_family']}:{best_overall['model_variant']}",
                str(best_overall["top_k"]),
                str(int(best_overall["num_features"])),
            ]
        )

        sens_acc = [spec["label"]]
        sens_model = [spec["label"]]
        for k in K_VALUES:
            best = best_model_for_k_phase1(df, spec["filters"], spec["grouping_levels"], "all_universal", k)
            sens_acc.append(f"{best['accuracy'] * 100:.1f}")
            sens_model.append(f"{best['model']} ({best['num_features']})")
        sensitivity_rows.append(sens_acc)
        model_rows.append(sens_model)

    parts = []
    parts.append("Phase 1 Combined-Pool Language Summary")
    parts.append("=====================================")
    parts.append("All Phase 1 rows below use the full combined universal feature pool (`all_universal`) within each language or the pooled benchmark.")
    parts.append("k is the number of ANOVA-ranked features kept from that combined pool before the classifier is fit.")
    parts.append("")
    parts.append("Table 1. Best combined-pool setup per language")
    parts.append("")
    parts.append(
        markdown_like_table(
            ["Scope", "Text Mono", "Text Multi", "Text Trans", "Best Acc", "Best Model", "Best k", "#Feat"],
            summary_rows,
        )
    )
    parts.append("")
    parts.append("Table 2. Accuracy sensitivity by pruning level")
    parts.append("")
    parts.append(
        markdown_like_table(
            ["Scope", "k=all", "k=100", "k=50", "k=25", "k=10"],
            sensitivity_rows,
        )
    )
    parts.append("")
    parts.append("Table 3. Best model at each pruning level")
    parts.append("")
    parts.append(
        markdown_like_table(
            ["Scope", "k=all", "k=100", "k=50", "k=25", "k=10"],
            model_rows,
        )
    )
    return "\n".join(parts) + "\n"


def build_phase2_tables() -> str:
    df = pd.read_csv(p2.FEATURES_PATH)
    summary_rows = []
    sensitivity_rows = []
    model_rows = []

    for spec in PHASE2_SPECS:
        summary = json.loads((p2.RUN_ROOT.parent / "phase2-clean-prompt-sweep" / "summaries" / f"{spec['run_name']}_summary.json").read_text(encoding="utf-8"))
        best = summary["best_model"]
        subset_name = best["subset_name"]
        summary_rows.append(
            [
                spec["label"],
                subset_name,
                f"{best['model_family']}:{best['model_variant']}",
                str(best["top_k"]),
                str(int(best["num_selected_features"])),
                f"{best['accuracy'] * 100:.1f}",
            ]
        )
        sens_acc = [spec["label"]]
        sens_model = [spec["label"]]
        for k in K_VALUES:
            run = best_model_for_k_phase2(df, spec["filters"], spec["grouping_levels"], subset_name, k)
            sens_acc.append(f"{run['accuracy'] * 100:.1f}")
            sens_model.append(f"{run['model']} ({run['num_features']})")
        sensitivity_rows.append(sens_acc)
        model_rows.append(sens_model)

    parts = []
    parts.append("Phase 2 Clean-Slice Summary")
    parts.append("===========================")
    parts.append("These rows keep the current best feature subset per clean slice, then show how raw accuracy changes as k varies.")
    parts.append("")
    parts.append("Table 1. Best pruning strategy overall from the stored clean-slice runs")
    parts.append("")
    parts.append(
        markdown_like_table(
            ["Slice", "Best Subset", "Best Model", "Best k", "#Feat", "Best Acc"],
            summary_rows,
        )
    )
    parts.append("")
    parts.append("Table 2. Accuracy sensitivity by pruning level within the winning subset")
    parts.append("")
    parts.append(
        markdown_like_table(
            ["Slice", "k=all", "k=100", "k=50", "k=25", "k=10"],
            sensitivity_rows,
        )
    )
    parts.append("")
    parts.append("Table 3. Best model at each pruning level within the winning subset")
    parts.append("")
    parts.append(
        markdown_like_table(
            ["Slice", "k=all", "k=100", "k=50", "k=25", "k=10"],
            model_rows,
        )
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "01_phase1_combined_pool_language_summary.txt").write_text(build_phase1_tables(), encoding="utf-8")
    (REPORT_ROOT / "03_phase2_clean_slice_summary_table.txt").write_text(build_phase2_tables(), encoding="utf-8")


if __name__ == "__main__":
    main()
