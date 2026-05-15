import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.phase1.common import make_logger
from processing.phase2.common import TABLES_PHASE2_ROOT


CLEAN_ROOT = TABLES_PHASE2_ROOT / "phase2-clean-prompt-sweep"
CLEAN_RESULT_TABLES = CLEAN_ROOT / "result-tables" / "csv"
CLEAN_SUMMARIES = CLEAN_ROOT / "summaries"
REPORT_ROOT = CLEAN_ROOT / "report_assets"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

SLICE_CONFIG = [
    ("en_cookie_theft", "English", "Cookie Theft", "language_en_cookie_theft_clean_phase2"),
    ("zh_cookie_theft", "Chinese", "Cookie Theft", "language_zh_cookie_theft_clean_phase2"),
    ("el_lion_scene", "Greek", "Lion Scene", "language_el_lion_scene_clean_phase2"),
    ("es_reading", "Spanish", "Reading", "language_es_reading_clean_phase2"),
]


def best_row(df: pd.DataFrame) -> pd.Series:
    sort_cols = ["accuracy", "auroc", "balanced_accuracy", "macro_f1", "auprc"]
    return df.sort_values(sort_cols, ascending=[False, False, False, False, False]).iloc[0]


def load_summary(run_name: str) -> dict:
    return json.loads((CLEAN_SUMMARIES / f"{run_name}_summary.json").read_text(encoding="utf-8"))


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    log = make_logger("phase2_language_feature_summary")
    summary_rows = []
    top_feature_rows = []

    for slice_slug, language, task, run_name in SLICE_CONFIG:
        results = pd.read_csv(CLEAN_RESULT_TABLES / f"{run_name}_model_results.csv")
        best = best_row(results)

        same_subset = results[results["subset_name"] == best["subset_name"]].copy()
        same_subset_k100 = same_subset[same_subset["top_k"].astype(str) == "100"].copy()
        same_subset_k100_best = best_row(same_subset_k100) if not same_subset_k100.empty else None

        all_phase2 = results[results["subset_name"] == "all_phase2"].copy()
        all_phase2_best = best_row(all_phase2) if not all_phase2.empty else None

        importance = pd.read_csv(CLEAN_RESULT_TABLES / f"{run_name}_permutation_importance.csv")
        top_features = importance.head(10).copy()
        top_features["slice_slug"] = slice_slug
        top_features["language"] = language
        top_features["task"] = task
        top_feature_rows.append(top_features)

        summary_rows.append(
            {
                "slice_slug": slice_slug,
                "language": language,
                "task": task,
                "run_name": run_name,
                "best_subset": best["subset_name"],
                "best_top_k": int(best["top_k"]),
                "best_num_selected_features": int(best["num_selected_features"]),
                "best_model": f"{best['model_family']}:{best['model_variant']}",
                "best_accuracy": float(best["accuracy"]),
                "best_balanced_accuracy": float(best["balanced_accuracy"]),
                "best_auroc": float(best["auroc"]),
                "same_subset_k100_accuracy": float(same_subset_k100_best["accuracy"]) if same_subset_k100_best is not None else None,
                "same_subset_k100_balanced_accuracy": float(same_subset_k100_best["balanced_accuracy"]) if same_subset_k100_best is not None else None,
                "same_subset_k100_model": f"{same_subset_k100_best['model_family']}:{same_subset_k100_best['model_variant']}" if same_subset_k100_best is not None else None,
                "delta_vs_same_subset_k100_accuracy": float(best["accuracy"] - same_subset_k100_best["accuracy"]) if same_subset_k100_best is not None else None,
                "delta_vs_same_subset_k100_balanced_accuracy": float(best["balanced_accuracy"] - same_subset_k100_best["balanced_accuracy"]) if same_subset_k100_best is not None else None,
                "best_all_phase2_top_k": int(all_phase2_best["top_k"]) if all_phase2_best is not None else None,
                "best_all_phase2_model": f"{all_phase2_best['model_family']}:{all_phase2_best['model_variant']}" if all_phase2_best is not None else None,
                "best_all_phase2_accuracy": float(all_phase2_best["accuracy"]) if all_phase2_best is not None else None,
                "best_all_phase2_balanced_accuracy": float(all_phase2_best["balanced_accuracy"]) if all_phase2_best is not None else None,
                "delta_vs_best_all_phase2_accuracy": float(best["accuracy"] - all_phase2_best["accuracy"]) if all_phase2_best is not None else None,
                "delta_vs_best_all_phase2_balanced_accuracy": float(best["balanced_accuracy"] - all_phase2_best["balanced_accuracy"]) if all_phase2_best is not None else None,
                "note": "No true no-selection full-feature run exists for clean slices; k=100 and best all_phase2 are the nearest available comparisons.",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    top_features_df = pd.concat(top_feature_rows, ignore_index=True)

    summary_df.to_csv(REPORT_ROOT / "language_feature_selection_summary.csv", index=False)
    top_features_df.to_csv(REPORT_ROOT / "language_top_features_summary.csv", index=False)

    md_lines = [
        "# Language Feature Selection Summary",
        "",
        "This summary is based on the clean prompt/task slices.",
        "There is no true no-selection full-feature condition in these clean-slice sweeps; the nearest comparisons are:",
        "- the fuller `k=100` version of the same winning feature family",
        "- the best `all_phase2` result for the same slice",
        "",
        markdown_table(summary_df),
        "",
        "## Top Permutation Features",
        "",
    ]
    for slice_slug, language, task, _ in SLICE_CONFIG:
        subset = top_features_df[top_features_df["slice_slug"] == slice_slug][["feature_name", "feature_group", "importance_mean", "importance_std"]]
        md_lines.append(f"### {language} {task}")
        md_lines.append("")
        md_lines.append(markdown_table(subset))
        md_lines.append("")

    (REPORT_ROOT / "language_feature_selection_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    log("Wrote consolidated language feature summary tables")


if __name__ == "__main__":
    main()
