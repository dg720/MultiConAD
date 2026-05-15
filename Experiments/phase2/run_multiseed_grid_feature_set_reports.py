from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.phase1.common import write_json
from processing.phase2.common import PHASE2_ROOT, TABLES_PHASE2_ROOT


FEATURES_PATH = PHASE2_ROOT / "phase2_features.csv"
METADATA_PATH = PHASE2_ROOT / "phase2_feature_metadata.csv"
CORE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "tables"
    / "03-ablation-translingual-language-specific"
    / "phase1-rich-sweep"
    / "result-tables"
    / "csv"
    / "all_universal_multiseed_grid_accuracy_summary.csv"
)
RUN_ROOT = TABLES_PHASE2_ROOT / "phase2-multiseed-grid-feature-sets"
CSV_DIR = RUN_ROOT / "result-tables" / "csv"
RESULT_DIR = RUN_ROOT / "result-tables"
SUMMARY_DIR = RUN_ROOT / "summaries"

SEEDS = list(range(42, 57))
TOP_KS: list[int | str] = [5, 10, 20, 50, 100, "all"]
RANKING_METHODS = ["anova", "welch_bonferroni"]
TASK_GROUPS = {"pd", "rd", "fc", "ft", "sr", "cmd", "rep", "ms", "na"}
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
FEATURE_SET_SPECS = {
    "full_feature_set": {
        "title": "Full Feature Set",
        "description": "All Phase 2 metadata features: core, richer non-task features, and task-specific features.",
        "report": RESULT_DIR / "full_feature_set_multiseed_grid_report.txt",
        "best_by_seed": CSV_DIR / "full_feature_set_multiseed_grid_best_by_seed.csv",
        "summary": CSV_DIR / "full_feature_set_multiseed_grid_accuracy_summary.csv",
    },
    "task_specific_only": {
        "title": "Task-Specific Features Only",
        "description": "Only task-specific Phase 2 groups: pd, rd, fc, ft, sr, cmd, rep, ms, and na.",
        "report": RESULT_DIR / "task_specific_only_multiseed_grid_report.txt",
        "best_by_seed": CSV_DIR / "task_specific_only_multiseed_grid_best_by_seed.csv",
        "summary": CSV_DIR / "task_specific_only_multiseed_grid_accuracy_summary.csv",
    },
    "extended_excluding_task_specific": {
        "title": "Extended Feature Set Excluding Task-Specific Features",
        "description": "All non-task-specific Phase 2 metadata features: Phase 1 core plus richer syntax, phrase, and paralinguistic features.",
        "report": RESULT_DIR / "extended_excluding_task_specific_multiseed_grid_report.txt",
        "best_by_seed": CSV_DIR / "extended_excluding_task_specific_multiseed_grid_best_by_seed.csv",
        "summary": CSV_DIR / "extended_excluding_task_specific_multiseed_grid_accuracy_summary.csv",
    },
}


def seed_range_label(seeds: list[int]) -> str:
    return f"{seeds[0]}-{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) else str(seeds)


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


def apply_filters(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for key, value in filters.items():
        out = out[out[key] == value].copy()
    return out


def grouped_train_test_split(df: pd.DataFrame, seed: int, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_labels = (
        df.groupby("group_id")["binary_label"]
        .agg(lambda values: int(pd.Series(values).mode().iloc[0]))
        .reset_index()
    )
    class_counts = group_labels["binary_label"].value_counts()
    if class_counts.min() < 2:
        raise RuntimeError(f"Not enough groups for stratified split: {class_counts.to_dict()}")
    train_groups, test_groups = train_test_split(
        group_labels["group_id"],
        test_size=test_size,
        random_state=seed,
        stratify=group_labels["binary_label"],
    )
    return (
        df[df["group_id"].isin(set(train_groups))].copy(),
        df[df["group_id"].isin(set(test_groups))].copy(),
    )


def normalize_with_fallback(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    grouping_levels: list[list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    test_out = test_df.copy()
    train_stats = {}
    for level in grouping_levels:
        key = tuple(level)
        if level:
            grouped = train_df.groupby(level)
            train_stats[key] = {
                "mean": grouped[feature_cols].mean(),
                "std": grouped[feature_cols].std().replace(0.0, np.nan),
            }
        else:
            train_stats[key] = {
                "mean": pd.DataFrame([train_df[feature_cols].mean()], index=[0]),
                "std": pd.DataFrame([train_df[feature_cols].std().replace(0.0, np.nan)], index=[0]),
            }

    def apply_frame(frame: pd.DataFrame) -> pd.DataFrame:
        normalized_rows = []
        for _, row in frame.iterrows():
            raw = row[feature_cols].astype(float)
            normalized = pd.Series(index=feature_cols, dtype=float)
            remaining = set(feature_cols)
            for level in grouping_levels:
                if not remaining:
                    break
                stats = train_stats[tuple(level)]
                if level:
                    group_key = tuple(row[col] for col in level)
                    if group_key not in stats["mean"].index:
                        continue
                    mean = stats["mean"].loc[group_key]
                    std = stats["std"].loc[group_key]
                else:
                    mean = stats["mean"].iloc[0]
                    std = stats["std"].iloc[0]
                fillable = [
                    col
                    for col in remaining
                    if pd.notna(raw[col]) and pd.notna(mean[col]) and pd.notna(std[col]) and std[col] != 0
                ]
                for col in fillable:
                    normalized[col] = (raw[col] - mean[col]) / std[col]
                remaining -= set(fillable)
            normalized_rows.append(normalized.reindex(feature_cols))
        return pd.DataFrame(normalized_rows, columns=feature_cols, index=frame.index)

    train_out[feature_cols] = apply_frame(train_df)
    test_out[feature_cols] = apply_frame(test_df)
    return train_out, test_out


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


def usable_feature_cols(train_norm: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    keep = []
    for col in feature_cols:
        series = pd.to_numeric(train_norm[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        filled = series.fillna(series.median())
        if filled.nunique(dropna=True) <= 1:
            continue
        keep.append(col)
    if not keep:
        raise RuntimeError("No usable feature columns remained after all-NaN/zero-variance filtering.")
    return keep


def anova_ranking(train_norm: pd.DataFrame, train_y: pd.Series, feature_cols: list[str]) -> tuple[list[str], pd.DataFrame]:
    keep = usable_feature_cols(train_norm, feature_cols)
    candidate = train_norm[keep].copy()
    for col in keep:
        series = pd.to_numeric(candidate[col], errors="coerce")
        candidate[col] = series.fillna(series.median())
    scores = f_classif(candidate[keep], train_y)
    ranking = pd.DataFrame(
        {"feature_name": keep, "f_score": scores[0], "p_value": scores[1], "rank_metric": scores[0]}
    ).sort_values(["f_score", "p_value"], ascending=[False, True])
    return keep, ranking.reset_index(drop=True)


def pooled_std(ad_vals: pd.Series, hc_vals: pd.Series) -> float:
    return math.sqrt((float(ad_vals.var(ddof=1)) + float(hc_vals.var(ddof=1))) / 2.0)


def welch_bonferroni_ranking(train_norm: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_cols:
        ad_vals = pd.to_numeric(train_norm.loc[train_norm["binary_label"] == 1, feature], errors="coerce").dropna()
        hc_vals = pd.to_numeric(train_norm.loc[train_norm["binary_label"] == 0, feature], errors="coerce").dropna()
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
                "bonferroni_p": np.nan,
                "cohens_d": float(effect),
                "cohens_d_abs": float(abs(effect)),
            }
        )
    if not rows:
        raise RuntimeError("No usable Welch-ranked features were generated.")
    ranking = pd.DataFrame(rows)
    ranking["bonferroni_p"] = (ranking["p_value"] * len(ranking)).clip(upper=1.0)
    ranking["significant_bonferroni"] = ranking["bonferroni_p"] < 0.05
    return ranking.sort_values(
        ["bonferroni_p", "p_value", "cohens_d_abs"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def select_features(method: str, top_k: int | str, all_cols: list[str], rankings: dict[str, pd.DataFrame]) -> list[str]:
    if top_k == "all":
        return all_cols
    ranking = rankings[method]
    return ranking["feature_name"].head(min(int(top_k), len(ranking))).tolist()


def run_grid(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    test_y: pd.Series,
    seed: int,
) -> dict[str, object]:
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


def load_feature_columns(feature_set: str, df: pd.DataFrame, metadata: pd.DataFrame) -> list[str]:
    meta = metadata.copy()
    meta["task_specific_bool"] = meta["task_specific"].astype(str).str.lower().isin({"1", "true"})
    if feature_set == "full_feature_set":
        names = meta["feature_name"].tolist()
    elif feature_set == "task_specific_only":
        names = meta.loc[meta["task_specific_bool"], "feature_name"].tolist()
    elif feature_set == "extended_excluding_task_specific":
        names = meta.loc[~meta["task_specific_bool"], "feature_name"].tolist()
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return [name for name in names if name in df.columns and pd.api.types.is_numeric_dtype(df[name])]


def run_language_seed(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_set: str,
    code: str,
    meta: dict[str, object],
    seed: int,
    completed: set[tuple[str, str, str, str, str]],
) -> list[dict[str, object]]:
    run_df = apply_filters(df, meta["filters"])
    run_df = run_df[run_df["binary_label"].isin([0, 1])].copy()
    train_df, test_df = grouped_train_test_split(run_df, seed=seed)
    feature_cols = load_feature_columns(feature_set, run_df, metadata)
    train_norm, test_norm = normalize_with_fallback(train_df, test_df, feature_cols, meta["grouping_levels"])
    train_y = train_norm["binary_label"].astype(int)
    test_y = test_norm["binary_label"].astype(int)
    all_cols, anova = anova_ranking(train_norm, train_y, feature_cols)
    welch = welch_bonferroni_ranking(train_norm, all_cols)
    rankings = {"anova": anova, "welch_bonferroni": welch}

    rows = []
    for method in RANKING_METHODS:
        for top_k in TOP_KS:
            key = (feature_set, str(seed), str(meta["label"]), method, str(top_k))
            if key in completed:
                continue
            selected_cols = select_features(method, top_k, all_cols, rankings)
            best = run_grid(train_norm[selected_cols], train_y, test_norm[selected_cols], test_y, seed)
            row = {
                "feature_set": feature_set,
                "language_code": code,
                "language": meta["label"],
                "seed": seed,
                "ranking_method": method,
                "top_k": str(top_k),
                "num_features": len(selected_cols),
                "pool_features": len(feature_cols),
                "usable_features": len(all_cols),
                "welch_bonferroni_significant_features": int(welch["significant_bonferroni"].sum()),
                "test_rows": int(len(test_df)),
                "test_groups": int(test_df["group_id"].nunique()),
                **best,
            }
            rows.append(row)
            print(
                f"{feature_set} {meta['label']} seed={seed} {method} k={top_k} "
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


def mean_sd(series: pd.Series) -> str:
    values = series.astype(float).to_numpy() * 100.0
    return f"{values.mean():.1f} +/- {values.std(ddof=0):.1f}"


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


def best_rows(summary_df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for (language, method), group in summary_df.groupby(["language", "ranking_method"], sort=True):
        best = group.sort_values(["accuracy_mean", "accuracy_sd"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "language": language,
                "method": method,
                "top_k": best["top_k"],
                "acc": f"{best['accuracy_mean'] * 100:.1f} +/- {best['accuracy_sd'] * 100:.1f}",
                "model": f"{best['most_common_model']} ({int(best['most_common_model_count'])}/{int(best['n_seeds'])})",
                "mean": float(best["accuracy_mean"]),
            }
        )
    return rows


def core_best_lookup() -> pd.DataFrame:
    if not CORE_SUMMARY_PATH.exists():
        return pd.DataFrame()
    core = pd.read_csv(CORE_SUMMARY_PATH)
    core = core[core["ranking_method"].isin(RANKING_METHODS)].copy()
    rows = []
    for (language, method), group in core.groupby(["language", "ranking_method"], sort=True):
        best = group.sort_values(["accuracy_mean", "accuracy_sd"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "language": language,
                "method": method,
                "core_best_k": str(best["top_k"]),
                "core_acc": f"{float(best['accuracy_mean']) * 100:.1f} +/- {float(best['accuracy_sd']) * 100:.1f}",
                "core_mean": float(best["accuracy_mean"]),
            }
        )
    return pd.DataFrame(rows)


def write_report(feature_set: str, best_by_seed: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    spec = FEATURE_SET_SPECS[feature_set]
    seeds = sorted(best_by_seed["seed"].astype(int).unique().tolist())
    lines = [
        f"{spec['title']} - 15-Seed Full GridSearchCV Top-k Report",
        "=" * (len(spec["title"]) + 47),
        f"Feature pool: {spec['description']}",
        f"Seeds: {seed_range_label(seeds)} ({len(seeds)} seeds).",
        "Rows: languages. Columns: k values. Cells: mean +/- sd held-out accuracy (%) across seeds.",
        "For each seed/language/ranking/k cell, the reported result is the best held-out accuracy across DT, RF, SVM, and LR after GridSearchCV.",
        "Tuning grid: DT max_depth=[10,20,30]; RF n_estimators=[50,100,200]; SVM C=[0.1,1,10] kernel=[linear,rbf]; LR C=[0.1,1,10].",
        "Ranking methods: ANOVA=f_classif by F score; Welch+Bonferroni=Welch t-test ranked by Bonferroni-adjusted p-value, then raw p-value, then |Cohen's d|.",
        "k values: 5, 10, 20, 50, 100, all.",
        "",
    ]
    for title, method in [("ANOVA-ranked top-k accuracy", "anova"), ("Welch t-test + Bonferroni-ranked top-k accuracy", "welch_bonferroni")]:
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

    current_best = pd.DataFrame(best_rows(summary_df))
    lines.append("Best setting by language and method")
    lines.append("-----------------------------------")
    lines.extend(
        fixed_table(
            current_best.to_dict("records"),
            [
                ("language", "language", 10, "left"),
                ("method", "method", 18, "left"),
                ("top_k", "best k", 8, "right"),
                ("acc", "accuracy", 14, "right"),
                ("model", "common model", 15, "left"),
            ],
        )
    )
    lines.append("")

    core = core_best_lookup()
    if not core.empty:
        comparison = current_best.merge(core, on=["language", "method"], how="left")
        comparison["delta"] = comparison.apply(
            lambda row: f"{(float(row['mean']) - float(row['core_mean'])) * 100:+.1f}" if pd.notna(row.get("core_mean")) else "-",
            axis=1,
        )
        comparison["current_acc"] = comparison["acc"]
        lines.append("Comparison with previous core feature-set run")
        lines.append("---------------------------------------------")
        lines.append("Delta is current best mean accuracy minus previous core-feature best mean accuracy, in percentage points.")
        lines.extend(
            fixed_table(
                comparison.to_dict("records"),
                [
                    ("language", "language", 10, "left"),
                    ("method", "method", 18, "left"),
                    ("current_acc", "current", 14, "right"),
                    ("top_k", "cur k", 7, "right"),
                    ("core_acc", "core", 14, "right"),
                    ("core_best_k", "core k", 7, "right"),
                    ("delta", "delta", 8, "right"),
                ],
            )
        )
        lines.append("")

    feature_counts = (
        best_by_seed.groupby("language")[["pool_features", "usable_features"]]
        .max()
        .reset_index()
        .to_dict("records")
    )
    lines.append("Feature availability")
    lines.append("--------------------")
    lines.extend(
        fixed_table(
            feature_counts,
            [
                ("language", "language", 10, "left"),
                ("pool_features", "pool", 8, "right"),
                ("usable_features", "usable", 8, "right"),
            ],
        )
    )
    lines.append("")
    lines.append("Interpretation note")
    lines.append("-------------------")
    lines.append("These runs are language-specific held-out evaluations with train-split-only normalization and train-split-only ranking.")
    lines.append("Task-specific features are sparse by design, so task-only results can be dominated by which task domains exist within each language.")
    lines.append("For k=all, ANOVA and Welch use the same usable feature pool; any difference would indicate a resume/version issue rather than ranking behavior.")
    spec["report"].write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_feature_set(feature_set: str) -> None:
    spec = FEATURE_SET_SPECS[feature_set]
    df = pd.read_csv(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    rows: list[dict[str, object]] = []
    completed: set[tuple[str, str, str, str, str]] = set()
    if spec["best_by_seed"].exists():
        existing = pd.read_csv(spec["best_by_seed"])
        rows = existing.to_dict("records")
        completed = {
            (
                str(row["feature_set"]),
                str(row["seed"]),
                str(row["language"]),
                str(row["ranking_method"]),
                str(row["top_k"]),
            )
            for _, row in existing.iterrows()
        }
        print(f"Resuming {feature_set}: {len(existing)} completed rows", flush=True)

    for seed in SEEDS:
        for code, meta in LANGUAGE_SPECS.items():
            rows.extend(run_language_seed(df, metadata, feature_set, code, meta, seed, completed))
            completed = {
                (
                    str(row["feature_set"]),
                    str(row["seed"]),
                    str(row["language"]),
                    str(row["ranking_method"]),
                    str(row["top_k"]),
                )
                for row in rows
            }
            pd.DataFrame(rows).to_csv(spec["best_by_seed"], index=False)

    best_by_seed = pd.DataFrame(rows)
    summary = summarize(best_by_seed)
    best_by_seed.to_csv(spec["best_by_seed"], index=False)
    summary.to_csv(spec["summary"], index=False)
    write_report(feature_set, best_by_seed, summary)


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    for feature_set in FEATURE_SET_SPECS:
        run_feature_set(feature_set)
    write_json(
        SUMMARY_DIR / "phase2_multiseed_grid_feature_set_run_index.json",
        {
            "feature_sets": list(FEATURE_SET_SPECS),
            "seeds": SEEDS,
            "top_ks": [str(value) for value in TOP_KS],
            "ranking_methods": RANKING_METHODS,
            "reports": {name: str(spec["report"]) for name, spec in FEATURE_SET_SPECS.items()},
        },
    )


if __name__ == "__main__":
    main()
