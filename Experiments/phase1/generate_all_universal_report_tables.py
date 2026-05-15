from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import kruskal
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase1.generate_all_task_feature_comparison import parse_best_text_baselines
from experiments.phase1.run_rich_sweep import (
    FEATURES_PATH,
    MANIFEST_PATH,
    RICH_SWEEP_RESULT_TABLES,
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
TEXT_SUMMARY_PATH = PROJECT_ROOT / "tables" / "01-baselines" / "embedding-baselines" / "multiseed-suite" / "result-tables" / "paper_vs_ours_3tables.txt"
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
LANGUAGE_ORDER = ["Chinese", "English", "Greek", "Spanish"]


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
    df = pd.read_csv(RICH_SWEEP_RESULT_TABLES / f"{run_name}_model_results.csv")
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
        "feature_cols": feature_cols,
        "selected_cols": selected_cols,
        "anova_ranking": anova_ranking,
        "permutation_importance": perm,
        "num_features": len(selected_cols),
        "train_df": train_df,
    }


def bh_fdr(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    prev = 1.0
    for rev_pos in range(len(indexed) - 1, -1, -1):
        original_idx, p_value = indexed[rev_pos]
        rank = rev_pos + 1
        candidate = min(prev, (p_value * len(p_values)) / rank, 1.0)
        adjusted[original_idx] = candidate
        prev = candidate
    return adjusted


def language_significance_frame(train_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_cols:
        ad_vals = pd.to_numeric(train_df.loc[train_df["binary_label"] == 1, feature], errors="coerce").dropna()
        hc_vals = pd.to_numeric(train_df.loc[train_df["binary_label"] == 0, feature], errors="coerce").dropna()
        if len(ad_vals) < 3 or len(hc_vals) < 3:
            continue
        combined = pd.concat([ad_vals, hc_vals], ignore_index=True)
        if combined.nunique(dropna=True) <= 1:
            continue
        try:
            h_stat, p_value = kruskal(ad_vals, hc_vals, nan_policy="omit")
        except ValueError:
            continue
        rows.append(
            {
                "feature_name": feature,
                "feature_group": feature.split("_", 1)[0],
                "h_statistic": float(h_stat),
                "p_value": float(p_value),
                "median_ad": float(ad_vals.median()),
                "median_hc": float(hc_vals.median()),
                "mean_ad": float(ad_vals.mean()),
                "mean_hc": float(hc_vals.mean()),
                "direction": "AD>HC" if ad_vals.median() > hc_vals.median() else "AD<HC",
            }
        )
    if not rows:
        raise RuntimeError("No language-specific significance rows were generated.")
    df = pd.DataFrame(rows).sort_values(["p_value", "h_statistic"], ascending=[True, False]).reset_index(drop=True)
    df["bonferroni_p"] = (df["p_value"] * len(df)).clip(upper=1.0)
    df["fdr_bh_p"] = bh_fdr(df["p_value"].tolist())
    df["significant_bonferroni"] = df["bonferroni_p"] < 0.05
    df["significant_fdr"] = df["fdr_bh_p"] < 0.05
    df["significance_rank"] = range(1, len(df) + 1)
    return df


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
    for language in LANGUAGE_ORDER:
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


def write_language_significance(significance_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    sig_csv_path = REPORT_ROOT / "all_universal_language_significance.csv"
    cmp_csv_path = REPORT_ROOT / "all_universal_language_significance_vs_permutation.csv"
    txt_path = REPORT_ROOT / "all_universal_language_significance_vs_permutation.txt"

    significance_df.to_csv(sig_csv_path, index=False)
    comparison_df.to_csv(cmp_csv_path, index=False)

    lines = []
    lines.append("Phase 1 Language-Specific Significance vs Permutation Importance")
    lines.append("==============================================================")
    lines.append("Significance uses per-language Kruskal-Wallis tests on raw Phase 1 feature values from the training split of the stored best `all_universal` run.")
    lines.append("Bonferroni and FDR are applied within each language over all tested Phase 1 features.")
    lines.append("Permutation importance is taken from the fitted best `all_universal` model on the held-out test split.")
    lines.append("")

    for language in LANGUAGE_ORDER:
        sig_subset = significance_df[significance_df["language"] == language].sort_values("significance_rank")
        cmp_subset = comparison_df[comparison_df["language"] == language].copy()
        if sig_subset.empty or cmp_subset.empty:
            continue

        summary = sig_subset.iloc[0]
        top_sig = cmp_subset[cmp_subset["list_name"] == "significance_top10"].sort_values("list_rank")
        top_perm = cmp_subset[cmp_subset["list_name"] == "permutation_top10"].sort_values("list_rank")
        sig_names = set(top_sig["feature_name"].tolist())
        perm_names = set(top_perm["feature_name"].tolist())
        shared = [name for name in top_sig["feature_name"].tolist() if name in perm_names]
        perm_only = [name for name in top_perm["feature_name"].tolist() if name not in sig_names]
        sig_only = [name for name in top_sig["feature_name"].tolist() if name not in perm_names]

        lines.append(language)
        lines.append("-" * len(language))
        lines.append(
            f"Tested features: {int(summary['tested_features'])} | Bonferroni-significant: {int(summary['bonferroni_significant_count'])} | "
            f"FDR-significant: {int(summary['fdr_significant_count'])} | Shared top-10 overlap: {len(shared)}/10"
        )
        lines.append("Top significance features")
        for _, row in top_sig.iterrows():
            overlap = f"perm#{int(row['other_rank'])}" if pd.notna(row["other_rank"]) else "-"
            lines.append(
                f"{int(row['list_rank'])}. {row['feature_name']} [{row['feature_group']}] "
                f"p={row['p_value']:.2e} bonf={row['bonferroni_p']:.2e} fdr={row['fdr_bh_p']:.2e} overlap={overlap}"
            )
        lines.append(
            "Permutation-only top features: "
            + (", ".join(perm_only) if perm_only else "none")
        )
        lines.append(
            "Significance-only top features: "
            + (", ".join(sig_only) if sig_only else "none")
        )
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
    significance_rows = []
    comparison_rows = []

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

        sig_df = language_significance_frame(reconstructed["train_df"], reconstructed["feature_cols"])
        tested_features = len(sig_df)
        bonf_count = int(sig_df["significant_bonferroni"].sum())
        fdr_count = int(sig_df["significant_fdr"].sum())
        sig_df["language"] = meta["label"]
        sig_df["tested_features"] = tested_features
        sig_df["bonferroni_significant_count"] = bonf_count
        sig_df["fdr_significant_count"] = fdr_count
        significance_rows.append(
            sig_df[
                [
                    "language",
                    "significance_rank",
                    "feature_name",
                    "feature_group",
                    "h_statistic",
                    "p_value",
                    "bonferroni_p",
                    "fdr_bh_p",
                    "significant_bonferroni",
                    "significant_fdr",
                    "median_ad",
                    "median_hc",
                    "mean_ad",
                    "mean_hc",
                    "direction",
                    "tested_features",
                    "bonferroni_significant_count",
                    "fdr_significant_count",
                ]
            ]
        )

        top_sig = sig_df.sort_values(
            ["significant_bonferroni", "significant_fdr", "p_value", "h_statistic"],
            ascending=[False, False, True, False],
        ).head(10)
        top_perm = perm.sort_values("rank").head(10)
        perm_rank_map = {row["feature_name"]: int(row["rank"]) for _, row in top_perm.iterrows()}
        sig_rank_map = {row["feature_name"]: int(row["significance_rank"]) for _, row in top_sig.iterrows()}

        for _, row in top_sig.iterrows():
            comparison_rows.append(
                {
                    "language": meta["label"],
                    "list_name": "significance_top10",
                    "list_rank": int(row["significance_rank"]),
                    "feature_name": row["feature_name"],
                    "feature_group": row["feature_group"],
                    "p_value": float(row["p_value"]),
                    "bonferroni_p": float(row["bonferroni_p"]),
                    "fdr_bh_p": float(row["fdr_bh_p"]),
                    "h_statistic": float(row["h_statistic"]),
                    "other_rank": perm_rank_map.get(row["feature_name"]),
                }
            )

        sig_stats_map = sig_df.set_index("feature_name")
        for _, row in top_perm.iterrows():
            sig_row = sig_stats_map.loc[row["feature_name"]] if row["feature_name"] in sig_stats_map.index else None
            comparison_rows.append(
                {
                    "language": meta["label"],
                    "list_name": "permutation_top10",
                    "list_rank": int(row["rank"]),
                    "feature_name": row["feature_name"],
                    "feature_group": row["feature_group"],
                    "p_value": float(sig_row["p_value"]) if sig_row is not None else None,
                    "bonferroni_p": float(sig_row["bonferroni_p"]) if sig_row is not None else None,
                    "fdr_bh_p": float(sig_row["fdr_bh_p"]) if sig_row is not None else None,
                    "h_statistic": float(sig_row["h_statistic"]) if sig_row is not None else None,
                    "other_rank": sig_rank_map.get(row["feature_name"]),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    top_df = pd.concat(top_rows, ignore_index=True)
    significance_df = pd.concat(significance_rows, ignore_index=True)
    comparison_df = pd.DataFrame(comparison_rows)
    write_summary(summary_df)
    write_top_features(top_df)
    write_language_significance(significance_df, comparison_df)


if __name__ == "__main__":
    main()
