from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = REPO_ROOT / "tables" / "03-ablation-translingual-language-specific" / "phase1-rich-sweep"
PHASE1_RESULT_TABLES = PHASE1_DIR / "result-tables" / "csv"
PHASE1_SUMMARIES = PHASE1_DIR / "summaries"
REPORT_ASSETS_DIR = PHASE1_DIR / "report_assets"
MULTISEED_DIR = REPO_ROOT / "tables" / "01-baselines" / "embedding-baselines" / "multiseed-suite"
TEXT_SUMMARY_PATH = MULTISEED_DIR / "result-tables" / "paper_vs_ours_3tables.txt"
FEATURE_METADATA_PATH = REPO_ROOT / "data" / "processed" / "phase1" / "phase1_feature_metadata.csv"

LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "zh": "Chinese",
    "el": "Greek",
}
TEXT_SECTION_NAMES = ["Monolingual", "Multilingual-Combined", "Translated-Combined"]


def extract_ours_scores(cell: str) -> list[float]:
    scores = [float(match) for match in re.findall(r"/\s*([0-9]+(?:\.[0-9]+)?)\s*\+/-", cell)]
    if scores:
        return scores
    standalone_matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*\+/-", cell)
    return [float(match) for match in standalone_matches]


def parse_best_text_baselines(path: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    current_section = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.strip() in TEXT_SECTION_NAMES:
                current_section = line.strip()
                continue
            if not current_section or not line.startswith("Binary"):
                continue

            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 9:
                continue

            language = parts[1]
            representation = parts[2]
            candidate_cells = {
                "DT": parts[3],
                "RF": parts[4],
                "SVM": parts[5],
                "LR": parts[6],
                "Best Ensemble": parts[7],
            }

            best_label = None
            best_score = None
            for label, cell in candidate_cells.items():
                scores = extract_ours_scores(cell)
                if not scores:
                    continue
                score = scores[0]
                if best_score is None or score > best_score:
                    best_label = label
                    best_score = score

            if best_score is None:
                continue

            records.append(
                {
                    "section": current_section,
                    "language": language,
                    "representation": representation,
                    "best_text_accuracy": best_score,
                    "best_text_source": best_label,
                }
            )

    frame = pd.DataFrame(records)
    frame = frame.sort_values(["section", "language", "best_text_accuracy"], ascending=[True, True, False])
    return frame.groupby(["section", "language"], as_index=False).first()


def load_summary(summary_path: Path) -> dict[str, object]:
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_model(row: pd.Series) -> str:
    return f"{row['model_family']}:{row['model_variant']}"


def join_top_items(items: list[str], limit: int = 5) -> str:
    return ", ".join(items[:limit])


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}"


def format_signed_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}"


def derive_language_feature_summary() -> pd.DataFrame:
    feature_metadata = pd.read_csv(FEATURE_METADATA_PATH)
    feature_groups = feature_metadata[["feature_name", "feature_group"]].drop_duplicates()
    top_features = pd.read_csv(PHASE1_RESULT_TABLES / "cross_lingual_all_tasks_feature_top30_by_language.csv")
    top_features = top_features.merge(feature_groups, on="feature_name", how="left")
    text_baselines = parse_best_text_baselines(TEXT_SUMMARY_PATH)

    benchmark_summary = load_summary(PHASE1_SUMMARIES / "benchmark_wide_rich_summary.json")
    benchmark_result = benchmark_summary["best_result"]
    benchmark_perm = pd.read_csv(PHASE1_RESULT_TABLES / "benchmark_wide_rich_permutation_importance_no_missing_indicators.csv")
    benchmark_groups = (
        benchmark_perm.merge(feature_groups, on="feature_name", how="left")
        .groupby("feature_group", as_index=False)["permutation_importance_mean"]
        .mean()
        .sort_values("permutation_importance_mean", ascending=False)
    )

    benchmark_top_features = join_top_items(benchmark_perm["feature_name"].tolist(), limit=5)
    benchmark_top_groups = join_top_items(benchmark_groups["feature_group"].tolist(), limit=4)

    rows: list[dict[str, object]] = []
    for code, language_name in LANGUAGES.items():
        model_results = pd.read_csv(PHASE1_RESULT_TABLES / f"language_{code}_all_rich_model_results.csv")
        summary = load_summary(PHASE1_SUMMARIES / f"language_{code}_all_rich_summary.json")
        best_row = model_results.sort_values(["accuracy", "auroc", "balanced_accuracy"], ascending=False).iloc[0]

        same_subset_full = model_results[
            (model_results["subset"] == best_row["subset"])
            & (model_results["top_k"].astype(str) == "all")
        ].sort_values(["accuracy", "auroc", "balanced_accuracy"], ascending=False)
        same_subset_full_row = same_subset_full.iloc[0] if not same_subset_full.empty else None

        full_universal = model_results[
            (model_results["subset"] == "all_universal")
            & (model_results["top_k"].astype(str) == "all")
        ].sort_values(["accuracy", "auroc", "balanced_accuracy"], ascending=False)
        full_universal_row = full_universal.iloc[0] if not full_universal.empty else None

        language_top = top_features[top_features["language"] == code].sort_values("rank")
        group_summary = (
            language_top.groupby("feature_group", as_index=False)["permutation_importance_mean"]
            .mean()
            .sort_values("permutation_importance_mean", ascending=False)
        )

        text_lookup = {}
        for section in TEXT_SECTION_NAMES:
            subset = text_baselines[
                (text_baselines["section"] == section) & (text_baselines["language"] == language_name)
            ]
            if subset.empty:
                continue
            text_lookup[section] = subset.iloc[0]

        selected_delta_same_subset = None
        if same_subset_full_row is not None:
            selected_delta_same_subset = best_row["accuracy"] - same_subset_full_row["accuracy"]

        selected_delta_full_universal = None
        if full_universal_row is not None:
            selected_delta_full_universal = best_row["accuracy"] - full_universal_row["accuracy"]

        delta_vs_global = best_row["accuracy"] - benchmark_result["accuracy"]

        beats_global = "yes" if delta_vs_global > 0 else "no"
        text_mono = text_lookup.get("Monolingual")
        text_combined = text_lookup.get("Multilingual-Combined")
        text_translated = text_lookup.get("Translated-Combined")

        rows.append(
            {
                "language_code": code,
                "language": language_name,
                "global_feature_subset": benchmark_summary["best_config"]["subset"],
                "global_feature_top_k": benchmark_summary["best_config"]["top_k"],
                "global_feature_model": f"{benchmark_summary['best_config']['model_family']}:{benchmark_summary['best_config']['model_variant']}",
                "global_feature_accuracy": benchmark_result["accuracy"],
                "global_feature_balanced_accuracy": benchmark_result["balanced_accuracy"],
                "global_feature_auroc": benchmark_result["auroc"],
                "global_top_feature_groups": benchmark_top_groups,
                "global_top_features": benchmark_top_features,
                "local_feature_subset": best_row["subset"],
                "local_feature_top_k": str(best_row["top_k"]),
                "local_feature_num_features": int(best_row["num_features"]),
                "local_feature_model": format_model(best_row),
                "local_feature_accuracy": best_row["accuracy"],
                "local_feature_balanced_accuracy": best_row["balanced_accuracy"],
                "local_feature_auroc": best_row["auroc"],
                "delta_local_vs_global_feature": delta_vs_global,
                "beats_global_feature_benchmark": beats_global,
                "same_subset_full_accuracy": None if same_subset_full_row is None else same_subset_full_row["accuracy"],
                "same_subset_full_balanced_accuracy": None if same_subset_full_row is None else same_subset_full_row["balanced_accuracy"],
                "delta_local_vs_same_subset_full": selected_delta_same_subset,
                "full_universal_accuracy": None if full_universal_row is None else full_universal_row["accuracy"],
                "full_universal_balanced_accuracy": None if full_universal_row is None else full_universal_row["balanced_accuracy"],
                "delta_local_vs_full_universal": selected_delta_full_universal,
                "text_best_monolingual_accuracy": None if text_mono is None else text_mono["best_text_accuracy"],
                "text_best_monolingual_source": None if text_mono is None else f"{text_mono['representation']} {text_mono['best_text_source']}",
                "text_best_multilingual_combined_accuracy": None if text_combined is None else text_combined["best_text_accuracy"],
                "text_best_multilingual_combined_source": None if text_combined is None else f"{text_combined['representation']} {text_combined['best_text_source']}",
                "text_best_translated_combined_accuracy": None if text_translated is None else text_translated["best_text_accuracy"],
                "text_best_translated_combined_source": None if text_translated is None else f"{text_translated['representation']} {text_translated['best_text_source']}",
                "local_top_feature_groups": join_top_items(group_summary["feature_group"].tolist(), limit=4),
                "local_top_features": join_top_items(language_top["feature_name"].tolist(), limit=5),
                "interpretation": interpret_language_row(code, best_row["subset"], delta_vs_global),
            }
        )

    return pd.DataFrame(rows).sort_values("language")


def interpret_language_row(language_code: str, subset: str, delta_vs_global: float) -> str:
    if language_code == "en":
        return "English improves materially under raw accuracy once the feature fit is allowed to specialize within language; the strongest slice uses a broader text-plus-graph feature mix."
    if language_code == "es":
        return "Spanish improves over the pooled benchmark, but the best raw-accuracy model no longer prefers a tiny monolingual text subset; the full universal feature pool is competitive once accuracy becomes primary."
    if language_code == "zh":
        return "Chinese benefits from a monolingual text+pause fit; pause and rate features become more discriminative once English and Greek are removed."
    if language_code == "el":
        return "Greek benefits the most from monolingual specialization; acoustic-only features dominate and outperform the pooled benchmark clearly."
    return f"Local subset {subset} changes the fit by {delta_vs_global:.4f} balanced-accuracy points versus the pooled benchmark."


def write_language_summary(frame: pd.DataFrame) -> None:
    REPORT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_ASSETS_DIR / "all_task_language_feature_selection_summary.csv"
    frame.to_csv(csv_path, index=False)

    md_path = REPORT_ASSETS_DIR / "all_task_language_feature_selection_summary.md"
    with md_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Phase 1 All-Task Feature Selection Summary\n\n")
        handle.write("Text baseline columns are accuracy percentages from `paper_vs_ours_3tables.txt`. ")
        handle.write("Feature benchmark columns are single-seed raw accuracy values from the Phase 1 rich sweep; balanced accuracy is retained in the CSV for reference.\n\n")
        handle.write("| Language | Best Text Mono Acc | Best Text Combined Acc | Best Feature Local Acc | Local Feature Set | Delta vs Global Feature | Delta vs Full Universal | Top Local Feature Groups |\n")
        handle.write("| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |\n")
        for row in frame.to_dict("records"):
            handle.write(
                f"| {row['language']} | "
                f"{row['text_best_monolingual_accuracy']:.1f} | "
                f"{row['text_best_multilingual_combined_accuracy']:.1f} | "
                f"{format_pct(row['local_feature_accuracy'])} | "
                f"{row['local_feature_subset']} k={row['local_feature_top_k']} {row['local_feature_model']} | "
                f"{format_signed_pct(row['delta_local_vs_global_feature'])} | "
                f"{format_signed_pct(row['delta_local_vs_full_universal'])} | "
                f"{row['local_top_feature_groups']} |\n"
            )
        handle.write("\n")
        for row in frame.to_dict("records"):
            handle.write(f"- **{row['language']}**: {row['interpretation']}\n")


def write_compact_comparison(frame: pd.DataFrame) -> None:
    output_path = MULTISEED_DIR / "paper_vs_feature_phase1_all_tasks.txt"
    csv_path = MULTISEED_DIR / "paper_vs_feature_phase1_all_tasks.csv"
    frame.to_csv(csv_path, index=False)

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("Phase 1 All-Task Feature Comparison\n")
        handle.write("===================================\n")
        handle.write("Text baseline numbers are accuracy (%) from `paper_vs_ours_3tables.txt`.\n")
        handle.write("Feature benchmark numbers are single-seed raw accuracy (%) from the Phase 1 rich sweep.\n\n")

        benchmark_row = frame.iloc[0]
        handle.write("Global Feature Benchmark\n")
        handle.write("------------------------\n")
        handle.write(
            f"All languages pooled: {format_pct(benchmark_row['global_feature_accuracy'])} accuracy, "
            f"{format_pct(benchmark_row['global_feature_balanced_accuracy'])} bal acc, "
            f"{format_pct(benchmark_row['global_feature_auroc'])} AUROC, "
            f"{benchmark_row['global_feature_subset']} k={benchmark_row['global_feature_top_k']} "
            f"{benchmark_row['global_feature_model']}.\n"
        )
        handle.write(f"Top pooled feature groups: {benchmark_row['global_top_feature_groups']}\n")
        handle.write(f"Top pooled features: {benchmark_row['global_top_features']}\n\n")

        handle.write("Language-Level All-Task Comparison\n")
        handle.write("----------------------------------\n")
        header = (
            "Language | Text Mono Acc | Text Multi-Combined Acc | Text Translated Acc | "
            "Feature Local Acc | Feature Global Acc | Delta Local-Global | "
            "Local Feature Set | Local Top Features\n"
        )
        handle.write(header)
        handle.write("-" * (len(header) - 1) + "\n")
        for row in frame.to_dict("records"):
            handle.write(
                f"{row['language']:<8} | "
                f"{row['text_best_monolingual_accuracy']:>13.1f} | "
                f"{row['text_best_multilingual_combined_accuracy']:>24.1f} | "
                f"{row['text_best_translated_combined_accuracy']:>19.1f} | "
                f"{float(format_pct(row['local_feature_accuracy'])):>17.1f} | "
                f"{float(format_pct(row['global_feature_accuracy'])):>18.1f} | "
                f"{float(format_signed_pct(row['delta_local_vs_global_feature'])):>18.1f} | "
                f"{row['local_feature_subset']} k={row['local_feature_top_k']} {row['local_feature_model']} | "
                f"{row['local_top_features']}\n"
            )

        handle.write("\n")
        handle.write("Feature Selection Readout\n")
        handle.write("-------------------------\n")
        for row in frame.to_dict("records"):
            handle.write(
                f"{row['language']}: local best {format_pct(row['local_feature_accuracy'])} "
                f"vs same-subset full {format_pct(row['same_subset_full_accuracy'])} "
                f"({format_signed_pct(row['delta_local_vs_same_subset_full'])}), "
                f"vs full universal {format_pct(row['full_universal_accuracy'])} "
                f"({format_signed_pct(row['delta_local_vs_full_universal'])}).\n"
            )


def write_global_feature_table(frame: pd.DataFrame) -> None:
    output_path = REPORT_ASSETS_DIR / "all_task_language_top_features_summary.csv"
    top_rows = []
    for _, row in frame.iterrows():
        for rank, feature_name in enumerate(row["local_top_features"].split(", "), start=1):
            top_rows.append(
                {
                    "language": row["language"],
                    "rank": rank,
                    "feature_name": feature_name,
                    "local_feature_subset": row["local_feature_subset"],
                    "local_feature_model": row["local_feature_model"],
                    "local_feature_accuracy": row["local_feature_accuracy"],
                    "local_feature_balanced_accuracy": row["local_feature_balanced_accuracy"],
                }
            )
    pd.DataFrame(top_rows).to_csv(output_path, index=False)


def main() -> None:
    frame = derive_language_feature_summary()
    write_language_summary(frame)
    write_compact_comparison(frame)
    write_global_feature_table(frame)
    print(f"Wrote outputs for {len(frame)} languages.")


if __name__ == "__main__":
    main()
