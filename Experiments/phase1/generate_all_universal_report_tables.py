from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase1.generate_all_task_feature_comparison import parse_best_text_baselines
from experiments.phase1.run_rich_sweep import (
    FEATURES_PATH,
    MANIFEST_PATH,
    RICH_SWEEP_ROOT,
    SEED,
    feature_subset_columns,
    grouped_train_test_split,
    model_specs,
    normalize_with_fallback,
    permutation_importance_frame,
    select_top_k,
)


REPORT_ROOT = RICH_SWEEP_ROOT / "report_assets"
TEXT_SUMMARY_PATH = PROJECT_ROOT / "tables" / "experiment-results" / "multiseed-suite" / "paper_vs_ours_3tables.txt"
LANGUAGE_META = {
    "en": {
        "label": "English",
        "filters": {"language": "en"},
        "grouping_levels": [["task_type", "dataset_name"], ["task_type"], []],
    },
    "es": {
        "label": "Spanish",
        "filters": {"language": "es"},
        "grouping_levels": [["task_type", "dataset_name"], ["task_type"], []],
    },
    "zh": {
        "label": "Chinese",
        "filters": {"language": "zh"},
        "grouping_levels": [["task_type", "dataset_name"], ["task_type"], []],
    },
    "el": {
        "label": "Greek",
        "filters": {"language": "el"},
        "grouping_levels": [["task_type", "dataset_name"], ["task_type"], []],
    },
}


def load_merged() -> pd.DataFrame:
    manifest = pd.read_json(MANIFEST_PATH, lines=True)
    features = pd.read_csv(FEATURES_PATH)
    df = manifest.merge(
        features,
        on=["sample_id", "group_id", "dataset_name", "language", "task_type", "diagnosis_mapped", "binary_label"],
    )
    return df[df["binary_label"].isin([0, 1])].copy()


def apply_filters(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for key, value in filters.items():
        out = out[out[key] == value].copy()
    return out


def get_model_pipeline(model_family: str, model_variant: str):
    for spec in model_specs():
        if spec["model_family"] == model_family and spec["model_variant"] == model_variant:
            return clone(spec["estimator"])
    raise KeyError((model_family, model_variant))


def best_all_universal_row(run_name: str) -> pd.Series:
    df = pd.read_csv(RICH_SWEEP_ROOT / f"{run_name}_model_results.csv")
    return (
        df[df["subset"] == "all_universal"]
        .sort_values(["accuracy", "auroc", "balanced_accuracy", "macro_f1"], ascending=False)
        .iloc[0]
    )


def reconstruct_run(df: pd.DataFrame, filters: dict[str, object], grouping_levels: list[list[str]], row: pd.Series):
    run_df = apply_filters(df, filters)
    train_df, test_df = grouped_train_test_split(run_df, test_size=0.2, seed=SEED)
    feature_cols = feature_subset_columns(run_df, "all_universal")
    train_norm, test_norm = normalize_with_fallback(train_df, test_df, feature_cols, grouping_levels)
    y_train = train_norm["binary_label"].astype(int)
    y_test = test_norm["binary_label"].astype(int)
    selected_cols, anova_ranking = select_top_k(train_norm, y_train, feature_cols, row["top_k"])
    x_train = train_norm[selected_cols]
    x_test = test_norm[selected_cols]
    estimator = get_model_pipeline(row["model_family"], row["model_variant"])
    estimator.fit(x_train, y_train)
    if hasattr(estimator, "predict_proba"):
        score = estimator.predict_proba(x_test)[:, 1]
    else:
        score = estimator.decision_function(x_test)
    perm = permutation_importance_frame(estimator, x_test, y_test)
    return {
        "selected_cols": selected_cols,
        "anova_ranking": anova_ranking,
        "permutation_importance": perm,
        "num_features": len(selected_cols),
    }


def write_summary(df: pd.DataFrame) -> None:
    csv_path = REPORT_ROOT / "all_universal_language_summary.csv"
    txt_path = REPORT_ROOT / "all_universal_language_summary.txt"
    df.to_csv(csv_path, index=False)

    lines = []
    lines.append("Phase 1 Combined-Pool Language Summary")
    lines.append("=====================================")
    lines.append("All rows in this table use the full Phase 1 combined feature pool (`all_universal`) within each language.")
    lines.append("k is the number of ANOVA-ranked features kept from that combined pool before fitting the classifier.")
    lines.append("")
    header = f"{'Language':<10} {'Best Text Mono':>14} {'Best Text Multi':>15} {'Combined Feature Acc':>21} {'Model':<12} {'k':>5} {'#Feat':>7} {'Delta vs Global':>16}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in df.sort_values("language").iterrows():
        lines.append(
            f"{row['language']:<10} "
            f"{row['text_best_monolingual_accuracy']:>14.1f} "
            f"{row['text_best_multilingual_combined_accuracy']:>15.1f} "
            f"{row['combined_feature_accuracy'] * 100:>21.1f} "
            f"{row['combined_feature_model']:<12} "
            f"{str(row['combined_feature_top_k']):>5} "
            f"{int(row['combined_feature_num_features']):>7} "
            f"{row['delta_vs_global_combined_feature'] * 100:>16.1f}"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def write_top_features(df: pd.DataFrame) -> None:
    csv_path = REPORT_ROOT / "all_universal_language_top_features.csv"
    txt_path = REPORT_ROOT / "all_universal_language_top_features.txt"
    df.to_csv(csv_path, index=False)

    lines = []
    lines.append("Phase 1 Combined-Pool Top Features By Language")
    lines.append("=============================================")
    lines.append("These are permutation-ranked features from the best `all_universal` model in each language.")
    lines.append("")
    for language in ["Chinese", "English", "Greek", "Spanish"]:
        subset = df[df["language"] == language].sort_values("rank")
        if subset.empty:
            continue
        meta = subset.iloc[0]
        lines.append(language)
        lines.append("-" * len(language))
        lines.append(
            f"Model: {meta['combined_feature_model']} | k={meta['combined_feature_top_k']} | "
            f"acc={meta['combined_feature_accuracy'] * 100:.1f}%"
        )
        for _, row in subset.iterrows():
            lines.append(f"{int(row['rank'])}. {row['feature_name']} [{row['feature_group']}]")
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    merged = load_merged()
    text = parse_best_text_baselines(TEXT_SUMMARY_PATH)

    global_row = best_all_universal_row("benchmark_wide_rich")
    global_accuracy = float(global_row["accuracy"])
    global_top_k = str(global_row["top_k"])
    global_model = f"{global_row['model_family']}:{global_row['model_variant']}"

    summary_rows = []
    top_rows = []

    for code, meta in LANGUAGE_META.items():
        run_name = f"language_{code}_all_rich"
        best_row = best_all_universal_row(run_name)
        reconstructed = reconstruct_run(merged, meta["filters"], meta["grouping_levels"], best_row)

        mono = text[(text["section"] == "Monolingual") & (text["language"] == meta["label"])].iloc[0]
        multi = text[(text["section"] == "Multilingual-Combined") & (text["language"] == meta["label"])].iloc[0]
        translated = text[(text["section"] == "Translated-Combined") & (text["language"] == meta["label"])].iloc[0]

        summary_rows.append(
            {
                "language_code": code,
                "language": meta["label"],
                "global_combined_feature_accuracy": global_accuracy,
                "global_combined_feature_model": global_model,
                "global_combined_feature_top_k": global_top_k,
                "combined_feature_accuracy": float(best_row["accuracy"]),
                "combined_feature_balanced_accuracy": float(best_row["balanced_accuracy"]),
                "combined_feature_auroc": float(best_row["auroc"]),
                "combined_feature_model": f"{best_row['model_family']}:{best_row['model_variant']}",
                "combined_feature_top_k": str(best_row["top_k"]),
                "combined_feature_num_features": int(best_row["num_features"]),
                "delta_vs_global_combined_feature": float(best_row["accuracy"] - global_accuracy),
                "text_best_monolingual_accuracy": float(mono["best_text_accuracy"]),
                "text_best_multilingual_combined_accuracy": float(multi["best_text_accuracy"]),
                "text_best_translated_combined_accuracy": float(translated["best_text_accuracy"]),
            }
        )

        perm = reconstructed["permutation_importance"].copy().head(10)
        perm["language"] = meta["label"]
        perm["combined_feature_model"] = f"{best_row['model_family']}:{best_row['model_variant']}"
        perm["combined_feature_top_k"] = str(best_row["top_k"])
        perm["combined_feature_accuracy"] = float(best_row["accuracy"])
        perm["rank"] = range(1, len(perm) + 1)
        perm["feature_group"] = perm["feature_name"].str.split("_").str[0]
        top_rows.append(
            perm[
                [
                    "language",
                    "rank",
                    "feature_name",
                    "feature_group",
                    "combined_feature_model",
                    "combined_feature_top_k",
                    "combined_feature_accuracy",
                ]
            ]
        )

    summary_df = pd.DataFrame(summary_rows)
    top_df = pd.concat(top_rows, ignore_index=True)
    write_summary(summary_df)
    write_top_features(top_df)


if __name__ == "__main__":
    main()
