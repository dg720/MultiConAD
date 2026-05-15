from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "Experiments"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

experiments_pkg = types.ModuleType("experiments")
experiments_pkg.__path__ = [str(EXPERIMENTS_ROOT.resolve())]
sys.modules.setdefault("experiments", experiments_pkg)

import experiments.phase1.run_rich_sweep as p1


RESULT_DIR = p1.RICH_SWEEP_TABLES_ROOT
CSV_DIR = p1.RICH_SWEEP_RESULT_TABLES
SUMMARY_DIR = p1.RICH_SWEEP_SUMMARIES
TOP_KS: list[int | str] = [5, 10, 20, 50, 100, "all"]
LANGUAGE_SPECS = {
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
    manifest = pd.read_json(p1.MANIFEST_PATH, lines=True)
    features = pd.read_csv(p1.FEATURES_PATH)
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


def bh_fdr(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    previous = 1.0
    for rev_pos in range(len(indexed) - 1, -1, -1):
        original_idx, p_value = indexed[rev_pos]
        rank = rev_pos + 1
        candidate = min(previous, (p_value * len(p_values)) / rank, 1.0)
        adjusted[original_idx] = candidate
        previous = candidate
    return adjusted


def pooled_std(ad_vals: pd.Series, hc_vals: pd.Series) -> float:
    return math.sqrt((float(ad_vals.var(ddof=1)) + float(hc_vals.var(ddof=1))) / 2.0)


def welch_feature_ranking(train_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_cols:
        ad_vals = pd.to_numeric(train_df.loc[train_df["binary_label"] == 1, feature], errors="coerce").dropna()
        hc_vals = pd.to_numeric(train_df.loc[train_df["binary_label"] == 0, feature], errors="coerce").dropna()
        if len(ad_vals) < 3 or len(hc_vals) < 3:
            continue
        combined = pd.concat([ad_vals, hc_vals], ignore_index=True)
        if combined.nunique(dropna=True) <= 1:
            continue

        test = ttest_ind(ad_vals, hc_vals, equal_var=False, nan_policy="omit")
        spread = pooled_std(ad_vals, hc_vals)
        effect = (float(ad_vals.mean()) - float(hc_vals.mean())) / spread if spread else np.nan
        if not np.isfinite(test.pvalue) or not np.isfinite(effect):
            continue

        rows.append(
            {
                "feature_name": feature,
                "feature_group": feature.split("_", 1)[0],
                "t_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "mean_ad": float(ad_vals.mean()),
                "mean_hc": float(hc_vals.mean()),
                "median_ad": float(ad_vals.median()),
                "median_hc": float(hc_vals.median()),
                "cohens_d": float(effect),
                "cohens_d_abs": float(abs(effect)),
                "direction": "AD>HC" if float(ad_vals.mean()) > float(hc_vals.mean()) else "AD<HC",
            }
        )

    if not rows:
        raise RuntimeError("No usable Welch-ranked features were generated.")

    df = pd.DataFrame(rows)
    df["bonferroni_p"] = (df["p_value"] * len(df)).clip(upper=1.0)
    df["fdr_bh_p"] = bh_fdr(df["p_value"].tolist())
    df["significant_bonferroni"] = df["bonferroni_p"] < 0.05
    df["significant_fdr"] = df["fdr_bh_p"] < 0.05
    return df


def order_welch_ranking(ranking: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "welch":
        ordered = ranking.sort_values(
            ["significant_bonferroni", "cohens_d_abs", "p_value"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        description = "Welch t-test, Bonferroni-significant first, then absolute Cohen's d"
    elif method == "welch_bonferroni":
        ordered = ranking.sort_values(
            ["bonferroni_p", "p_value", "cohens_d_abs"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        description = "Welch t-test, ranked by Bonferroni-adjusted p-value"
    else:
        raise ValueError(f"Unknown Welch ranking method: {method}")
    ordered = ordered.copy()
    ordered["welch_rank"] = ordered.index + 1
    ordered["ranking_method"] = method
    ordered["ranking_description"] = description
    return ordered


def select_from_ranking(ranking: pd.DataFrame, top_k: int | str, all_cols: list[str]) -> list[str]:
    if top_k == "all":
        return all_cols
    return ranking["feature_name"].head(min(int(top_k), len(ranking))).tolist()


def evaluate_model(estimator, train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series):
    estimator.fit(train_x, train_y)
    pred = estimator.predict(test_x)
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(test_x)[:, 1]
    else:
        scores = estimator.decision_function(test_x)
    return p1.compute_metrics(test_y, pred, scores)


def best_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values(["accuracy", "auroc", "balanced_accuracy", "macro_f1"], ascending=False).iloc[0]


def pct(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.1f}"


def sci(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.2e}"


def fixed_table(rows: list[dict[str, object]], columns: list[tuple[str, str, int, str]]) -> list[str]:
    header = " ".join(label.ljust(width) if align == "left" else label.rjust(width) for _, label, width, align in columns)
    lines = [header, "-" * len(header)]
    for row in rows:
        parts = []
        for key, _, width, align in columns:
            value = str(row.get(key, ""))
            if len(value) > width:
                value = value[: width - 1] + "~"
            parts.append(value.ljust(width) if align == "left" else value.rjust(width))
        lines.append(" ".join(parts))
    return lines


def anova_rows_for_language(code: str, label: str) -> pd.DataFrame:
    path = CSV_DIR / f"language_{code}_all_rich_model_results.csv"
    df = pd.read_csv(path)
    df = df[df["subset"] == "all_universal"].copy()
    df["language_code"] = code
    df["language"] = label
    df["ranking_method"] = "anova"
    df["ranking_description"] = "ANOVA f_classif, ranked by f_score"
    df["top_k"] = df["top_k"].astype(str)
    return df


def run_welch_language(df: pd.DataFrame, code: str, meta: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_df = apply_filters(df, meta["filters"])
    train_df, test_df = p1.grouped_train_test_split(run_df, test_size=0.2, seed=p1.SEED)
    feature_cols = p1.feature_subset_columns(run_df, "all_universal")
    train_norm, test_norm = p1.normalize_with_fallback(train_df, test_df, feature_cols, meta["grouping_levels"])
    train_y = train_norm["binary_label"].astype(int)
    test_y = test_norm["binary_label"].astype(int)

    welch_base_ranking = welch_feature_ranking(train_df, feature_cols)
    welch_rankings = {
        "welch": order_welch_ranking(welch_base_ranking, "welch"),
        "welch_bonferroni": order_welch_ranking(welch_base_ranking, "welch_bonferroni"),
    }
    anova_all_cols, anova_ranking = p1.select_top_k(train_norm, train_y, feature_cols, "all")
    anova_ranking = anova_ranking.reset_index(drop=True).copy()
    anova_ranking["anova_rank"] = anova_ranking.index + 1

    rows = []
    for method, welch_ranking in welch_rankings.items():
        for top_k in TOP_KS:
            selected_cols = select_from_ranking(welch_ranking, top_k, anova_all_cols)
            train_x = train_norm[selected_cols]
            test_x = test_norm[selected_cols]
            for spec in p1.model_specs():
                estimator = clone(spec["estimator"])
                metrics = evaluate_model(estimator, train_x, train_y, test_x, test_y)
                rows.append(
                    {
                        "run_name": f"language_{code}_all_rich",
                        "language_code": code,
                        "language": meta["label"],
                        "ranking_method": method,
                        "ranking_description": welch_ranking["ranking_description"].iloc[0],
                        "subset": "all_universal",
                        "top_k": str(top_k),
                        "num_features": len(selected_cols),
                        "tested_features": len(welch_ranking),
                        "bonferroni_significant_features": int(welch_ranking["significant_bonferroni"].sum()),
                        "fdr_significant_features": int(welch_ranking["significant_fdr"].sum()),
                        "model_family": spec["model_family"],
                        "model_variant": spec["model_variant"],
                        **metrics,
                    }
                )

    overlap_rows = []
    for method, welch_ranking in welch_rankings.items():
        for top_k in [10, 25, 50, 100]:
            anova_top = set(anova_ranking["feature_name"].head(min(top_k, len(anova_ranking))))
            welch_top = set(welch_ranking["feature_name"].head(min(top_k, len(welch_ranking))))
            overlap_rows.append(
                {
                    "language_code": code,
                    "language": meta["label"],
                    "ranking_method": method,
                    "top_k": top_k,
                    "anova_features": len(anova_top),
                    "welch_features": len(welch_top),
                    "overlap": len(anova_top & welch_top),
                    "overlap_pct_of_welch": len(anova_top & welch_top) / len(welch_top) if welch_top else np.nan,
                }
            )

    ranking_frames = []
    for welch_ranking in welch_rankings.values():
        ranked = welch_ranking.copy()
        ranked.insert(0, "language_code", code)
        ranked.insert(1, "language", meta["label"])
        ranking_frames.append(ranked)
    return pd.DataFrame(rows), pd.concat(ranking_frames, ignore_index=True), pd.DataFrame(overlap_rows)


def write_text_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    lines = []
    lines.append("ANOVA vs Welch Top-k Feature Sweep")
    lines.append("==================================")
    lines.append("Scope: Phase 1 `all_universal` feature pool, same grouped train/test split and model grid as `language_*_all_rich`.")
    lines.append("ANOVA rows reuse the existing top-k sweep; Welch rows rerank features using the saliency-map rule: Welch t-test on the training split, Bonferroni-significant features first, then absolute Cohen's d.")
    lines.append("")
    header = (
        f"{'Language':<10} {'ANOVA Acc':>9} {'Welch Acc':>9} {'Delta':>8} "
        f"{'ANOVA Model':<12} {'ANOVA k':>7} {'Welch Model':<12} {'Welch k':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in summary_df.sort_values("language").iterrows():
        lines.append(
            f"{row['language']:<10} "
            f"{row['anova_accuracy'] * 100:>9.1f} "
            f"{row['welch_accuracy'] * 100:>9.1f} "
            f"{row['delta_welch_minus_anova_accuracy'] * 100:>8.1f} "
            f"{row['anova_model']:<12} "
            f"{row['anova_top_k']:>7} "
            f"{row['welch_model']:<12} "
            f"{row['welch_top_k']:>7}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_model_results_text(results_df: pd.DataFrame, output_path: Path, title: str) -> None:
    lines = [title, "=" * len(title), ""]
    working = results_df.copy()
    working["model"] = working["model_family"].astype(str) + ":" + working["model_variant"].astype(str)
    working = working.sort_values(
        ["language", "ranking_method", "accuracy", "auroc", "balanced_accuracy", "macro_f1"],
        ascending=[True, True, False, False, False, False],
    )
    for (language, method), group in working.groupby(["language", "ranking_method"], sort=True):
        lines.append(f"{language} - {method.upper()}")
        lines.append("-" * len(lines[-1]))
        rows = []
        for _, row in group.iterrows():
            rows.append(
                {
                    "top_k": str(row["top_k"]),
                    "n": str(int(row["num_features"])),
                    "model": row["model"],
                    "acc": pct(row["accuracy"]),
                    "bal_acc": pct(row["balanced_accuracy"]),
                    "auroc": pct(row["auroc"]),
                    "macro_f1": pct(row["macro_f1"]),
                }
            )
        lines.extend(
            fixed_table(
                rows,
                [
                    ("top_k", "k", 6, "right"),
                    ("n", "n", 5, "right"),
                    ("model", "model", 14, "left"),
                    ("acc", "acc", 7, "right"),
                    ("bal_acc", "bal", 7, "right"),
                    ("auroc", "auroc", 7, "right"),
                    ("macro_f1", "f1", 7, "right"),
                ],
            )
        )
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_overlap_text(overlap_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "ANOVA vs Welch Ranking Overlap",
        "================================",
        "Overlap compares the top-k ANOVA feature list with the top-k Welch feature list for each language.",
        "",
    ]
    rows = []
    for _, row in overlap_df.sort_values(["language", "top_k"]).iterrows():
        rows.append(
            {
                "language": row["language"],
                "top_k": str(row["top_k"]),
                "anova": str(row["anova_features"]),
                "welch": str(row["welch_features"]),
                "overlap": str(row["overlap"]),
                "pct": pct(row["overlap_pct_of_welch"]),
            }
        )
    lines.extend(
        fixed_table(
            rows,
            [
                ("language", "language", 10, "left"),
                ("top_k", "k", 5, "right"),
                ("anova", "anova", 7, "right"),
                ("welch", "welch", 7, "right"),
                ("overlap", "shared", 7, "right"),
                ("pct", "%welch", 8, "right"),
            ],
        )
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_welch_rankings_text(rankings_df: pd.DataFrame, output_path: Path, top_n: int = 30) -> None:
    lines = [
        f"Top {top_n} Welch-Ranked Features By Language",
        "=======================================",
        "Ranking rule: Bonferroni-significant features first, then absolute Cohen's d, then p-value.",
        "",
    ]
    for language, group in rankings_df.sort_values(["language", "welch_rank"]).groupby("language", sort=True):
        lines.append(language)
        lines.append("-" * len(language))
        rows = []
        for _, row in group.head(top_n).iterrows():
            rows.append(
                {
                    "rank": str(int(row["welch_rank"])),
                    "feature": row["feature_name"],
                    "group": row["feature_group"],
                    "p": sci(row["p_value"]),
                    "bonf": sci(row["bonferroni_p"]),
                    "d": f"{float(row['cohens_d_abs']):.2f}",
                    "dir": row["direction"],
                }
            )
        lines.extend(
            fixed_table(
                rows,
                [
                    ("rank", "#", 4, "right"),
                    ("feature", "feature", 58, "left"),
                    ("group", "grp", 6, "left"),
                    ("p", "p", 10, "right"),
                    ("bonf", "bonf", 10, "right"),
                    ("d", "|d|", 6, "right"),
                    ("dir", "dir", 6, "left"),
                ],
            )
        )
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_combined_report(
    summary_df: pd.DataFrame,
    anova_by_k: pd.DataFrame,
    welch_by_k: pd.DataFrame,
    welch_bonferroni_by_k: pd.DataFrame,
    ranking_overlap: pd.DataFrame,
    welch_rankings: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "ANOVA vs Welch Feature Ranking Report",
        "======================================",
        "Scope: Phase 1 `all_universal` feature pool, same grouped train/test split and model grid as `language_*_all_rich`.",
        "ANOVA ranking uses `f_classif` on the training split, ranked by f_score.",
        "Welch |d| ranking uses the saliency-map rule: Bonferroni-significant features first, then absolute Cohen's d.",
        "Welch Bonferroni ranking orders by Bonferroni-adjusted p-value first; within one language this is equivalent to Welch p-value order.",
        "For `k=all`, all ranking methods use the same canonical usable `all_universal` feature list and order, so the column is directly comparable.",
        "",
        "Table A. Best Overall Accuracy By Ranking Method",
        "------------------------------------------------",
    ]
    summary_rows = []
    for _, row in summary_df.sort_values("language").iterrows():
        summary_rows.append(
            {
                "language": row["language"],
                "anova": pct(row["anova_accuracy"]),
                "welch": pct(row["welch_accuracy"]),
                "bonf": pct(row["welch_bonferroni_accuracy"]),
                "d_delta": pct(row["delta_welch_minus_anova_accuracy"]),
                "b_delta": pct(row["delta_welch_bonferroni_minus_anova_accuracy"]),
                "anova_model": row["anova_model"],
                "anova_k": row["anova_top_k"],
                "welch_model": row["welch_model"],
                "welch_k": row["welch_top_k"],
                "bonf_model": row["welch_bonferroni_model"],
                "bonf_k": row["welch_bonferroni_top_k"],
            }
        )
    lines.extend(
        fixed_table(
            summary_rows,
            [
                ("language", "language", 10, "left"),
                ("anova", "anova", 8, "right"),
                ("welch", "welch|d|", 8, "right"),
                ("bonf", "welch-p", 8, "right"),
                ("d_delta", "|d|-A", 8, "right"),
                ("b_delta", "p-A", 8, "right"),
                ("anova_model", "anova model", 14, "left"),
                ("anova_k", "anova k", 8, "right"),
                ("welch_model", "welch model", 14, "left"),
                ("welch_k", "welch k", 8, "right"),
                ("bonf_model", "p model", 14, "left"),
                ("bonf_k", "p k", 8, "right"),
            ],
        )
    )
    lines.append("")

    def append_accuracy_matrix(title: str, table_df: pd.DataFrame) -> None:
        lines.append(title)
        lines.append("-" * len(title))
        rows = []
        for _, row in table_df.sort_values("language").iterrows():
            out = {"language": row["language"]}
            for top_k in TOP_KS:
                k_label = str(top_k)
                out[k_label] = pct(row[k_label])
            rows.append(out)
        lines.extend(
            fixed_table(
                rows,
                [
                    ("language", "language", 10, "left"),
                    ("5", "k=5", 8, "right"),
                    ("10", "k=10", 8, "right"),
                    ("20", "k=20", 8, "right"),
                    ("50", "k=50", 8, "right"),
                    ("100", "k=100", 8, "right"),
                    ("all", "k=all", 8, "right"),
                ],
            )
        )
        lines.append("")

    append_accuracy_matrix("Table B. ANOVA-Ranked Best Accuracy By k", anova_by_k)
    append_accuracy_matrix("Table C. Welch |d|-Ranked Best Accuracy By k", welch_by_k)
    append_accuracy_matrix("Table D. Welch Bonferroni-Ranked Best Accuracy By k", welch_bonferroni_by_k)

    lines.extend(
        [
            "Table E. Top-k Ranking Overlap",
            "------------------------------",
            "Overlap compares the top-k ANOVA feature list with each Welch feature list for each language.",
        ]
    )
    overlap_rows = []
    for _, row in ranking_overlap.sort_values(["ranking_method", "language", "top_k"]).iterrows():
        overlap_rows.append(
            {
                "method": row["ranking_method"],
                "language": row["language"],
                "top_k": str(row["top_k"]),
                "anova": str(row["anova_features"]),
                "welch": str(row["welch_features"]),
                "overlap": str(row["overlap"]),
                "pct": pct(row["overlap_pct_of_welch"]),
            }
        )
    lines.extend(
        fixed_table(
            overlap_rows,
            [
                ("method", "method", 17, "left"),
                ("language", "language", 10, "left"),
                ("top_k", "k", 5, "right"),
                ("anova", "anova", 7, "right"),
                ("welch", "welch", 7, "right"),
                ("overlap", "shared", 7, "right"),
                ("pct", "%welch", 8, "right"),
            ],
        )
    )
    lines.append("")
    lines.extend(["Table F. Top 10 Welch-Ranked Features", "--------------------------------------"])
    for method in ["welch", "welch_bonferroni"]:
        lines.append("")
        lines.append(method)
        method_df = welch_rankings[welch_rankings["ranking_method"] == method].copy()
        for language, group in method_df.sort_values(["language", "welch_rank"]).groupby("language", sort=True):
            lines.append("")
            lines.append(language)
            rows = []
            for _, row in group.head(10).iterrows():
                rows.append(
                    {
                        "rank": str(int(row["welch_rank"])),
                        "feature": row["feature_name"],
                        "p": sci(row["p_value"]),
                        "bonf": sci(row["bonferroni_p"]),
                        "d": f"{float(row['cohens_d_abs']):.2f}",
                        "dir": row["direction"],
                    }
                )
            lines.extend(
                fixed_table(
                    rows,
                    [
                        ("rank", "#", 4, "right"),
                        ("feature", "feature", 62, "left"),
                        ("p", "p", 10, "right"),
                        ("bonf", "bonf", 10, "right"),
                        ("d", "|d|", 6, "right"),
                        ("dir", "dir", 6, "left"),
                    ],
                )
            )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def best_accuracy_by_k(results_df: pd.DataFrame, ranking_method: str) -> pd.DataFrame:
    method_df = results_df[results_df["ranking_method"] == ranking_method].copy()
    rows = []
    for language, group in method_df.groupby("language", sort=True):
        row: dict[str, object] = {"language": language}
        for top_k in TOP_KS:
            k_label = str(top_k)
            subset = group[group["top_k"].astype(str) == k_label].copy()
            if subset.empty:
                row[k_label] = np.nan
                row[f"{k_label}_model"] = ""
                continue
            best = best_row(subset)
            row[k_label] = float(best["accuracy"])
            row[f"{k_label}_model"] = f"{best['model_family']}:{best['model_variant']}"
        rows.append(row)
    return pd.DataFrame(rows)


def write_best_accuracy_by_k_text(table_df: pd.DataFrame, output_path: Path, title: str, ranking_note: str) -> None:
    lines = [title, "=" * len(title), ranking_note, "Values are best held-out accuracies (%) across model families for each k.", ""]
    rows = []
    for _, row in table_df.sort_values("language").iterrows():
        out = {"language": row["language"]}
        for top_k in TOP_KS:
            k_label = str(top_k)
            out[k_label] = pct(row[k_label])
        rows.append(out)
    lines.extend(
        fixed_table(
            rows,
            [
                ("language", "language", 10, "left"),
                ("5", "k=5", 8, "right"),
                ("10", "k=10", 8, "right"),
                ("20", "k=20", 8, "right"),
                ("50", "k=50", 8, "right"),
                ("100", "k=100", 8, "right"),
                ("all", "k=all", 8, "right"),
            ],
        )
    )
    lines.append("")
    lines.append("Best model per cell")
    lines.append("-------------------")
    model_rows = []
    for _, row in table_df.sort_values("language").iterrows():
        out = {"language": row["language"]}
        for top_k in TOP_KS:
            k_label = str(top_k)
            out[k_label] = row[f"{k_label}_model"]
        model_rows.append(out)
    lines.extend(
        fixed_table(
            model_rows,
            [
                ("language", "language", 10, "left"),
                ("5", "k=5", 14, "left"),
                ("10", "k=10", 14, "left"),
                ("20", "k=20", 14, "left"),
                ("50", "k=50", 14, "left"),
                ("100", "k=100", 14, "left"),
                ("all", "k=all", 14, "left"),
            ],
        )
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_merged()

    welch_frames = []
    anova_frames = []
    ranking_frames = []
    overlap_frames = []
    summary_rows = []

    for code, meta in LANGUAGE_SPECS.items():
        welch_df, ranking_df, overlap_df = run_welch_language(merged, code, meta)
        anova_df = anova_rows_for_language(code, meta["label"])
        welch_frames.append(welch_df)
        anova_frames.append(anova_df)
        ranking_frames.append(ranking_df)
        overlap_frames.append(overlap_df)

        anova_best = best_row(anova_df)
        welch_best = best_row(welch_df[welch_df["ranking_method"] == "welch"])
        welch_bonferroni_best = best_row(welch_df[welch_df["ranking_method"] == "welch_bonferroni"])
        summary_rows.append(
            {
                "language_code": code,
                "language": meta["label"],
                "anova_accuracy": float(anova_best["accuracy"]),
                "anova_balanced_accuracy": float(anova_best["balanced_accuracy"]),
                "anova_auroc": float(anova_best["auroc"]),
                "anova_model": f"{anova_best['model_family']}:{anova_best['model_variant']}",
                "anova_top_k": str(anova_best["top_k"]),
                "anova_num_features": int(anova_best["num_features"]),
                "welch_accuracy": float(welch_best["accuracy"]),
                "welch_balanced_accuracy": float(welch_best["balanced_accuracy"]),
                "welch_auroc": float(welch_best["auroc"]),
                "welch_model": f"{welch_best['model_family']}:{welch_best['model_variant']}",
                "welch_top_k": str(welch_best["top_k"]),
                "welch_num_features": int(welch_best["num_features"]),
                "welch_tested_features": int(welch_best["tested_features"]),
                "welch_bonferroni_significant_features": int(welch_best["bonferroni_significant_features"]),
                "welch_fdr_significant_features": int(welch_best["fdr_significant_features"]),
                "welch_bonferroni_accuracy": float(welch_bonferroni_best["accuracy"]),
                "welch_bonferroni_balanced_accuracy": float(welch_bonferroni_best["balanced_accuracy"]),
                "welch_bonferroni_auroc": float(welch_bonferroni_best["auroc"]),
                "welch_bonferroni_model": (
                    f"{welch_bonferroni_best['model_family']}:{welch_bonferroni_best['model_variant']}"
                ),
                "welch_bonferroni_top_k": str(welch_bonferroni_best["top_k"]),
                "welch_bonferroni_num_features": int(welch_bonferroni_best["num_features"]),
                "delta_welch_minus_anova_accuracy": float(welch_best["accuracy"] - anova_best["accuracy"]),
                "delta_welch_bonferroni_minus_anova_accuracy": float(
                    welch_bonferroni_best["accuracy"] - anova_best["accuracy"]
                ),
                "delta_welch_minus_anova_balanced_accuracy": float(
                    welch_best["balanced_accuracy"] - anova_best["balanced_accuracy"]
                ),
                "delta_welch_bonferroni_minus_anova_balanced_accuracy": float(
                    welch_bonferroni_best["balanced_accuracy"] - anova_best["balanced_accuracy"]
                ),
                "delta_welch_minus_anova_auroc": float(welch_best["auroc"] - anova_best["auroc"]),
                "delta_welch_bonferroni_minus_anova_auroc": float(
                    welch_bonferroni_best["auroc"] - anova_best["auroc"]
                ),
            }
        )

    all_results = pd.concat([*anova_frames, *welch_frames], ignore_index=True, sort=False)
    welch_results = pd.concat(welch_frames, ignore_index=True)
    welch_rankings = pd.concat(ranking_frames, ignore_index=True)
    ranking_overlap = pd.concat(overlap_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    welch_results.to_csv(CSV_DIR / "all_universal_welch_topk_model_results.csv", index=False)
    all_results.to_csv(CSV_DIR / "all_universal_anova_vs_welch_topk_model_results.csv", index=False)
    welch_rankings.to_csv(CSV_DIR / "all_universal_welch_feature_rankings.csv", index=False)
    ranking_overlap.to_csv(CSV_DIR / "all_universal_anova_vs_welch_ranking_overlap.csv", index=False)
    summary_df.to_csv(CSV_DIR / "all_universal_anova_vs_welch_topk_summary.csv", index=False)
    anova_by_k = best_accuracy_by_k(all_results, "anova")
    welch_by_k = best_accuracy_by_k(all_results, "welch")
    welch_bonferroni_by_k = best_accuracy_by_k(all_results, "welch_bonferroni")
    anova_by_k.to_csv(CSV_DIR / "all_universal_anova_best_accuracy_by_k.csv", index=False)
    welch_by_k.to_csv(CSV_DIR / "all_universal_welch_best_accuracy_by_k.csv", index=False)
    welch_bonferroni_by_k.to_csv(CSV_DIR / "all_universal_welch_bonferroni_best_accuracy_by_k.csv", index=False)
    write_combined_report(
        summary_df,
        anova_by_k,
        welch_by_k,
        welch_bonferroni_by_k,
        ranking_overlap,
        welch_rankings,
        RESULT_DIR / "all_universal_anova_vs_welch_report.txt",
    )


if __name__ == "__main__":
    main()
