from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import generate_multiseed_grid_anova_welch_report as base


REPORT_PATH = base.RESULT_DIR / "all_universal_multiseed_grid_balanced_anova_welch_report.txt"
BEST_BY_SEED_PATH = base.CSV_DIR / "all_universal_multiseed_grid_balanced_best_by_seed.csv"
SUMMARY_PATH = base.CSV_DIR / "all_universal_multiseed_grid_balanced_accuracy_summary.csv"
COMPARISON_PATH = base.CSV_DIR / "all_universal_multiseed_grid_balanced_vs_standard_summary.csv"
STANDARD_SUMMARY_PATH = base.CSV_DIR / "all_universal_multiseed_grid_accuracy_summary.csv"


def classifier_specs(seed: int):
    return {
        "DT": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("clf", DecisionTreeClassifier(random_state=seed, class_weight="balanced")),
                ]
            ),
            {"clf__max_depth": [10, 20, 30]},
        ),
        "RF": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("clf", RandomForestClassifier(random_state=seed, class_weight="balanced", n_jobs=-1)),
                ]
            ),
            {"clf__n_estimators": [50, 100, 200]},
        ),
        "SVM": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", base.StandardScaler()),
                    ("clf", SVC(random_state=seed, class_weight="balanced", probability=False)),
                ]
            ),
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]},
        ),
        "LR": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", base.StandardScaler()),
                    ("clf", LogisticRegression(random_state=seed, class_weight="balanced", max_iter=1000)),
                ]
            ),
            {"clf__C": [0.1, 1, 10]},
        ),
    }


def best_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (language, method), group in summary_df.groupby(["language", "ranking_method"], sort=True):
        best = group.sort_values(["accuracy_mean", "accuracy_sd"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "language": language,
                "ranking_method": method,
                "best_top_k": str(best["top_k"]),
                "accuracy_mean": float(best["accuracy_mean"]),
                "accuracy_sd": float(best["accuracy_sd"]),
                "most_common_model": str(best["most_common_model"]),
                "most_common_model_count": int(best["most_common_model_count"]),
            }
        )
    return pd.DataFrame(rows)


def comparison_rows(balanced_summary: pd.DataFrame) -> pd.DataFrame:
    balanced_best = best_rows(balanced_summary).rename(
        columns={
            "best_top_k": "balanced_best_top_k",
            "accuracy_mean": "balanced_accuracy_mean",
            "accuracy_sd": "balanced_accuracy_sd",
            "most_common_model": "balanced_most_common_model",
            "most_common_model_count": "balanced_most_common_model_count",
        }
    )
    if not STANDARD_SUMMARY_PATH.exists():
        return balanced_best

    standard_best = best_rows(pd.read_csv(STANDARD_SUMMARY_PATH)).rename(
        columns={
            "best_top_k": "standard_best_top_k",
            "accuracy_mean": "standard_accuracy_mean",
            "accuracy_sd": "standard_accuracy_sd",
            "most_common_model": "standard_most_common_model",
            "most_common_model_count": "standard_most_common_model_count",
        }
    )
    comparison = balanced_best.merge(standard_best, on=["language", "ranking_method"], how="left")
    comparison["delta_balanced_minus_standard_accuracy"] = (
        comparison["balanced_accuracy_mean"] - comparison["standard_accuracy_mean"]
    )
    return comparison


def pct_mean_sd(mean_value: float, sd_value: float) -> str:
    return f"{mean_value * 100:.1f} +/- {sd_value * 100:.1f}"


def write_report(best_by_seed: pd.DataFrame, summary_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    seeds = sorted(best_by_seed["seed"].astype(int).unique().tolist())
    n_seeds = len(seeds)
    lines = [
        "Phase 1 Multiseed GridSearchCV Balanced-Classifier ANOVA/Welch Top-k Report",
        "==========================================================================",
        f"Scope: Phase 1 `all_universal` core feature pool, four language-specific runs, {n_seeds} seeds [{base.seed_range_label(seeds)}].",
        "Each cell reports mean +/- sd held-out accuracy (%) across seeds.",
        "Balanced setup: DT, RF, SVM, and LR all use class_weight=\"balanced\".",
        "GridSearchCV protocol: cv=5, scoring=accuracy; DT max_depth=[10,20,30], RF n_estimators=[50,100,200], SVM C=[0.1,1,10] kernel=[linear,rbf], LR C=[0.1,1,10].",
        "For `k=all`, all ranking methods use the same canonical usable `all_universal` feature list and order.",
        "",
    ]

    for title, method in [
        ("Table B. Balanced ANOVA-Ranked Accuracy By k", "anova"),
        ("Table C. Balanced Welch |d|-Ranked Accuracy By k", "welch"),
        ("Table D. Balanced Welch Bonferroni/p-Ranked Accuracy By k", "welch_bonferroni"),
    ]:
        lines.append(title)
        lines.append("-" * len(title))
        lines.extend(
            base.fixed_table(
                base.method_matrix(best_by_seed, method),
                [
                    ("language", "language", 10, "left"),
                    ("5", "k=5", 12, "right"),
                    ("10", "k=10", 12, "right"),
                    ("20", "k=20", 12, "right"),
                    ("50", "k=50", 12, "right"),
                    ("100", "k=100", 12, "right"),
                    ("all", "k=all", 12, "right"),
                ],
            )
        )
        lines.append("")

    lines.append("Best Mean Accuracy Per Language/Method")
    lines.append("--------------------------------------")
    rows = []
    for _, row in best_rows(summary_df).iterrows():
        rows.append(
            {
                "language": row["language"],
                "method": row["ranking_method"],
                "top_k": row["best_top_k"],
                "acc": pct_mean_sd(row["accuracy_mean"], row["accuracy_sd"]),
                "model": f"{row['most_common_model']} ({int(row['most_common_model_count'])}/{n_seeds})",
            }
        )
    lines.extend(
        base.fixed_table(
            rows,
            [
                ("language", "language", 10, "left"),
                ("method", "method", 17, "left"),
                ("top_k", "best k", 8, "right"),
                ("acc", "accuracy", 13, "right"),
                ("model", "common model", 14, "left"),
            ],
        )
    )
    lines.append("")

    if {"standard_accuracy_mean", "delta_balanced_minus_standard_accuracy"}.issubset(comparison_df.columns):
        lines.append("Best Balanced vs Standard Classifier Suite")
        lines.append("------------------------------------------")
        compare_rows = []
        for _, row in comparison_df.sort_values(["language", "ranking_method"]).iterrows():
            compare_rows.append(
                {
                    "language": row["language"],
                    "method": row["ranking_method"],
                    "balanced": pct_mean_sd(row["balanced_accuracy_mean"], row["balanced_accuracy_sd"]),
                    "standard": pct_mean_sd(row["standard_accuracy_mean"], row["standard_accuracy_sd"]),
                    "delta": f"{row['delta_balanced_minus_standard_accuracy'] * 100:+.1f}",
                    "balanced_k": row["balanced_best_top_k"],
                    "standard_k": row["standard_best_top_k"],
                }
            )
        lines.extend(
            base.fixed_table(
                compare_rows,
                [
                    ("language", "language", 10, "left"),
                    ("method", "method", 17, "left"),
                    ("balanced", "balanced", 13, "right"),
                    ("standard", "standard", 13, "right"),
                    ("delta", "delta", 7, "right"),
                    ("balanced_k", "bal k", 7, "right"),
                    ("standard_k", "std k", 7, "right"),
                ],
            )
        )
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    base.classifier_specs = classifier_specs
    base.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    base.CSV_DIR.mkdir(parents=True, exist_ok=True)
    merged = base.load_merged()
    rows = []
    completed: set[tuple[str, str, str, str]] = set()
    if BEST_BY_SEED_PATH.exists():
        existing = pd.read_csv(BEST_BY_SEED_PATH)
        rows = existing.to_dict("records")
        completed = {
            (str(row["seed"]), str(row["language"]), str(row["ranking_method"]), str(row["top_k"]))
            for _, row in existing.iterrows()
        }
        print(f"Resuming from {len(existing)} completed balanced seed/language/ranking/k rows", flush=True)

    for seed in base.FEATURE_SEEDS:
        for code, meta in base.LANGUAGE_SPECS.items():
            rows.extend(base.run_language_seed(merged, code, meta, seed, completed))
            completed = {
                (str(row["seed"]), str(row["language"]), str(row["ranking_method"]), str(row["top_k"]))
                for row in rows
            }
            pd.DataFrame(rows).to_csv(BEST_BY_SEED_PATH, index=False)

    best_by_seed = pd.DataFrame(rows)
    summary_df = base.summarize(best_by_seed)
    comparison_df = comparison_rows(summary_df)
    best_by_seed.to_csv(BEST_BY_SEED_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    write_report(best_by_seed, summary_df, comparison_df)


if __name__ == "__main__":
    main()
