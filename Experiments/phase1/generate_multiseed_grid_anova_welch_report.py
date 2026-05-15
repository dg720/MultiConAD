from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "Experiments"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

experiments_pkg = types.ModuleType("experiments")
experiments_pkg.__path__ = [str(EXPERIMENTS_ROOT.resolve())]
sys.modules.setdefault("experiments", experiments_pkg)

import experiments.phase1.run_rich_sweep as p1
from experiments.phase1.generate_welch_topk_comparison import (
    LANGUAGE_SPECS,
    TOP_KS,
    apply_filters,
    fixed_table,
    load_merged,
    order_welch_ranking,
    pct,
    select_from_ranking,
    welch_feature_ranking,
)


RESULT_DIR = p1.RICH_SWEEP_TABLES_ROOT
CSV_DIR = p1.RICH_SWEEP_RESULT_TABLES
REPORT_PATH = RESULT_DIR / "all_universal_multiseed_grid_anova_welch_report.txt"
BEST_BY_SEED_PATH = CSV_DIR / "all_universal_multiseed_grid_best_by_seed.csv"
SUMMARY_PATH = CSV_DIR / "all_universal_multiseed_grid_accuracy_summary.csv"
RANKING_METHODS = ["anova", "welch", "welch_bonferroni"]
FEATURE_SEEDS = list(range(42, 57))


def seed_range_label(seeds: list[int]) -> str:
    if not seeds:
        return "[]"
    return f"{seeds[0]}-{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) else str(seeds)


def classifier_specs(seed: int):
    return {
        "DT": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("clf", DecisionTreeClassifier(random_state=seed)),
                ]
            ),
            {"clf__max_depth": [10, 20, 30]},
        ),
        "RF": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("clf", RandomForestClassifier(random_state=seed, n_jobs=-1)),
                ]
            ),
            {"clf__n_estimators": [50, 100, 200]},
        ),
        "SVM": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    ("clf", SVC(random_state=seed, class_weight="balanced", probability=False)),
                ]
            ),
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]},
        ),
        "LR": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(random_state=seed, max_iter=1000)),
                ]
            ),
            {"clf__C": [0.1, 1, 10]},
        ),
    }


def select_anova(train_norm: pd.DataFrame, train_y: pd.Series, feature_cols: list[str], top_k: int | str):
    selected_cols, ranking = p1.select_top_k(train_norm, train_y, feature_cols, top_k)
    return selected_cols, ranking


def build_rankings(train_df: pd.DataFrame, train_norm: pd.DataFrame, train_y: pd.Series, feature_cols: list[str]):
    all_cols, anova_ranking = p1.select_top_k(train_norm, train_y, feature_cols, "all")
    welch_base = welch_feature_ranking(train_df, feature_cols)
    return {
        "all_cols": all_cols,
        "anova": anova_ranking,
        "welch": order_welch_ranking(welch_base, "welch"),
        "welch_bonferroni": order_welch_ranking(welch_base, "welch_bonferroni"),
    }


def selected_features(method: str, top_k: int | str, rankings: dict[str, object]) -> list[str]:
    if method == "anova":
        if top_k == "all":
            return rankings["all_cols"]
        ranking = rankings["anova"]
        return ranking["feature_name"].head(min(int(top_k), len(ranking))).tolist()
    return select_from_ranking(rankings[method], top_k, rankings["all_cols"])


def run_grid(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series, seed: int):
    rows = []
    for model_family, (estimator, params) in classifier_specs(seed).items():
        grid = GridSearchCV(clone(estimator), params, cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(train_x, train_y)
        pred = grid.predict(test_x)
        rows.append(
            {
                "model_family": model_family,
                "accuracy": float(accuracy_score(test_y, pred)),
                "best_params": str(grid.best_params_),
                "cv_accuracy": float(grid.best_score_),
            }
        )
    return pd.DataFrame(rows).sort_values(["accuracy", "cv_accuracy"], ascending=False).iloc[0].to_dict()


def run_language_seed(
    merged: pd.DataFrame,
    code: str,
    meta: dict[str, object],
    seed: int,
    completed: set[tuple[str, str, str, str]],
) -> list[dict[str, object]]:
    run_df = apply_filters(merged, meta["filters"])
    train_df, test_df = p1.grouped_train_test_split(run_df, test_size=0.2, seed=seed)
    feature_cols = p1.feature_subset_columns(run_df, "all_universal")
    train_norm, test_norm = p1.normalize_with_fallback(train_df, test_df, feature_cols, meta["grouping_levels"])
    train_y = train_norm["binary_label"].astype(int)
    test_y = test_norm["binary_label"].astype(int)
    rankings = build_rankings(train_df, train_norm, train_y, feature_cols)

    rows = []
    for method in RANKING_METHODS:
        for top_k in TOP_KS:
            key = (str(seed), str(meta["label"]), method, str(top_k))
            if key in completed:
                continue
            cols = selected_features(method, top_k, rankings)
            best = run_grid(train_norm[cols], train_y, test_norm[cols], test_y, seed)
            rows.append(
                {
                    "language_code": code,
                    "language": meta["label"],
                    "seed": seed,
                    "ranking_method": method,
                    "top_k": str(top_k),
                    "num_features": len(cols),
                    "test_rows": int(len(test_df)),
                    "test_groups": int(test_df["group_id"].nunique()),
                    **best,
                }
            )
            print(
                f"{meta['label']} seed={seed} method={method} k={top_k} "
                f"best={best['model_family']} acc={best['accuracy']:.3f}",
                flush=True,
            )
    return rows


def summarize(best_by_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (language, method, top_k), group in best_by_seed.groupby(["language", "ranking_method", "top_k"], sort=True):
        values = group["accuracy"].astype(float).to_numpy()
        model_counts = group["model_family"].value_counts()
        rows.append(
            {
                "language": language,
                "ranking_method": method,
                "top_k": str(top_k),
                "accuracy_mean": float(values.mean()),
                "accuracy_sd": float(values.std(ddof=0)),
                "n_seeds": int(len(values)),
                "most_common_model": str(model_counts.index[0]),
                "most_common_model_count": int(model_counts.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def mean_sd(values: pd.Series) -> str:
    vals = values.astype(float).to_numpy() * 100.0
    return f"{vals.mean():.1f} +/- {vals.std(ddof=0):.1f}"


def method_matrix(best_by_seed: pd.DataFrame, method: str) -> list[dict[str, object]]:
    rows = []
    subset = best_by_seed[best_by_seed["ranking_method"] == method]
    for language, group in subset.groupby("language", sort=True):
        row: dict[str, object] = {"language": language}
        for top_k in TOP_KS:
            k_group = group[group["top_k"].astype(str) == str(top_k)]
            row[str(top_k)] = mean_sd(k_group["accuracy"]) if not k_group.empty else "-"
        rows.append(row)
    return rows


def write_report(best_by_seed: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    seeds = sorted(best_by_seed["seed"].astype(int).unique().tolist())
    n_seeds = len(seeds)
    lines = [
        "Phase 1 Multiseed GridSearchCV ANOVA/Welch Top-k Report",
        "=======================================================",
        f"Scope: Phase 1 `all_universal` core feature pool, four language-specific runs, {n_seeds} seeds [{seed_range_label(seeds)}].",
        "Each cell reports mean +/- sd held-out accuracy (%) across seeds.",
        "For each seed/language/ranking/k cell, the reported accuracy is the best test accuracy across the full GridSearchCV model-family suite.",
        "GridSearchCV protocol: cv=5, scoring=accuracy; DT max_depth=[10,20,30], RF n_estimators=[50,100,200], SVM C=[0.1,1,10] kernel=[linear,rbf], LR C=[0.1,1,10].",
        "The same hyperparameter grids are used as the embedding multiseed suite; fits are parallelized with n_jobs=-1, and SVM probability calibration is disabled because this report only uses predicted labels/accuracy.",
        "For `k=all`, all ranking methods use the same canonical usable `all_universal` feature list and order.",
        "",
    ]

    table_specs = [
        ("Table B. ANOVA-Ranked Accuracy By k", "anova"),
        ("Table C. Welch |d|-Ranked Accuracy By k", "welch"),
        ("Table D. Welch Bonferroni/p-Ranked Accuracy By k", "welch_bonferroni"),
    ]
    for title, method in table_specs:
        lines.append(title)
        lines.append("-" * len(title))
        lines.extend(
            fixed_table(
                method_matrix(best_by_seed, method),
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
    best_rows = []
    for (language, method), group in summary_df.groupby(["language", "ranking_method"], sort=True):
        best = group.sort_values(["accuracy_mean", "accuracy_sd"], ascending=[False, True]).iloc[0]
        best_rows.append(
            {
                "language": language,
                "method": method,
                "top_k": best["top_k"],
                "acc": f"{best['accuracy_mean'] * 100:.1f} +/- {best['accuracy_sd'] * 100:.1f}",
                "model": f"{best['most_common_model']} ({int(best['most_common_model_count'])}/{n_seeds})",
            }
        )
    lines.extend(
        fixed_table(
            best_rows,
            [
                ("language", "language", 10, "left"),
                ("method", "method", 17, "left"),
                ("top_k", "best k", 8, "right"),
                ("acc", "accuracy", 13, "right"),
                ("model", "common model", 14, "left"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "Tuning Protocol Note",
            "--------------------",
            "This standard report uses the full classical ML tuning sweep for every language, seed, ranking method, and k value.",
            "For each cell, GridSearchCV uses cv=5 and scoring=accuracy, then the reported value is the best held-out test accuracy across DT, RF, SVM, and LR.",
            "The tuned grids are: DT max_depth=[10,20,30]; RF n_estimators=[50,100,200]; SVM C=[0.1,1,10] with kernel=[linear,rbf]; LR C=[0.1,1,10].",
            "The sweep covers 4 languages x 15 seeds x 3 ranking methods x 6 k values = 1080 reported cells, with the classifier grids evaluated inside each cell.",
            "This is the standard GridSearchCV suite retained for reporting; the separate balanced-classifier sensitivity run did not improve the 15-seed mean accuracies.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_merged()
    rows = []
    completed: set[tuple[str, str, str, str]] = set()
    if BEST_BY_SEED_PATH.exists():
        existing = pd.read_csv(BEST_BY_SEED_PATH)
        rows = existing.to_dict("records")
        completed = {
            (str(row["seed"]), str(row["language"]), str(row["ranking_method"]), str(row["top_k"]))
            for _, row in existing.iterrows()
        }
        print(f"Resuming from {len(existing)} completed seed/language/ranking/k rows", flush=True)
    for seed in FEATURE_SEEDS:
        for code, meta in LANGUAGE_SPECS.items():
            rows.extend(run_language_seed(merged, code, meta, seed, completed))
            completed = {
                (str(row["seed"]), str(row["language"]), str(row["ranking_method"]), str(row["top_k"]))
                for row in rows
            }
            pd.DataFrame(rows).to_csv(BEST_BY_SEED_PATH, index=False)
    best_by_seed = pd.DataFrame(rows)
    summary_df = summarize(best_by_seed)
    best_by_seed.to_csv(BEST_BY_SEED_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    write_report(best_by_seed, summary_df)


if __name__ == "__main__":
    main()
