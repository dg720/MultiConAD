import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from processing.phase1.common import PHASE1_ROOT, TABLES_PHASE1_ROOT, make_logger


SEED = 42
FEATURES_PATH = PHASE1_ROOT / "phase1_features.csv"
MANIFEST_PATH = PHASE1_ROOT / "phase1_manifest.jsonl"


def grouped_train_test_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    group_labels = (
        df.groupby("group_id")["binary_label"]
        .agg(lambda values: int(pd.Series(values).mode().iloc[0]))
        .reset_index()
    )
    train_groups, test_groups = train_test_split(
        group_labels["group_id"],
        test_size=test_size,
        random_state=seed,
        stratify=group_labels["binary_label"],
    )
    train_df = df[df["group_id"].isin(set(train_groups))].copy()
    test_df = df[df["group_id"].isin(set(test_groups))].copy()
    return train_df, test_df


def normalize_with_fallback(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    grouping_levels: list[list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    test_out = test_df.copy()
    grouping_levels = grouping_levels or [
        ["language", "task_type", "dataset_name"],
        ["language", "task_type"],
        ["language"],
        [],
    ]

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
                key = tuple(level)
                if not remaining:
                    break
                stats = train_stats[key]
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


def feature_subset_columns(df: pd.DataFrame, subset_name: str) -> list[str]:
    allowed_prefixes = ("len_", "lex_", "pause_", "disc_", "syn_", "graph_", "ac_")
    core_feature_cols = [
        col
        for col in df.columns
        if col.startswith(allowed_prefixes) and pd.api.types.is_numeric_dtype(df[col])
    ]
    if subset_name == "all_universal":
        return core_feature_cols
    if subset_name == "text_only":
        return [
            col
            for col in core_feature_cols
            if not col.startswith("ac_")
            and not col.startswith("pause_")
            and col not in {"len_audio_duration", "len_speech_duration", "len_tokens_per_second", "len_syllables_per_second"}
        ]
    if subset_name == "acoustic_only":
        keep_prefixes = ("ac_", "pause_")
        keep_exact = {"len_audio_duration", "len_speech_duration", "len_tokens_per_second", "len_syllables_per_second"}
        return [col for col in core_feature_cols if col.startswith(keep_prefixes) or col in keep_exact]
    if subset_name == "text_plus_acoustic":
        return core_feature_cols
    raise ValueError(f"Unknown subset: {subset_name}")


def select_top_k(train_x: pd.DataFrame, train_y: pd.Series, feature_cols: list[str], k: int | str):
    candidate = train_x[feature_cols].copy()
    keep_cols = []
    for col in feature_cols:
        series = candidate[col]
        if series.notna().sum() == 0:
            continue
        filled = series.fillna(series.median())
        if filled.nunique(dropna=True) <= 1:
            continue
        candidate[col] = filled
        keep_cols.append(col)

    if not keep_cols:
        raise RuntimeError("No usable feature columns remained after dropping all-NaN and zero-variance features.")

    scores = f_classif(candidate[keep_cols], train_y)
    ranking = pd.DataFrame({"feature_name": keep_cols, "f_score": scores[0], "p_value": scores[1]}).sort_values(
        "f_score", ascending=False
    )
    if k == "all":
        return keep_cols, ranking
    selected = ranking["feature_name"].head(int(k)).tolist()
    return selected, ranking


def evaluate_model(model_name: str, train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series):
    if model_name == "logreg":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
            ]
        )
    elif model_name == "linear_svm":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=SEED)),
            ]
        )
    elif model_name == "rf":
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced")),
            ]
        )
    else:
        raise ValueError(model_name)

    estimator.fit(train_x, train_y)
    pred = estimator.predict(test_x)
    proba = estimator.predict_proba(test_x)[:, 1]

    result = {
        "balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
        "macro_f1": float(f1_score(test_y, pred, average="macro")),
        "auroc": float(roc_auc_score(test_y, proba)),
        "auprc": float(average_precision_score(test_y, proba)),
        "sensitivity": float(recall_score(test_y, pred, pos_label=1)),
        "specificity": float(recall_score(test_y, pred, pos_label=0)),
        "precision": float(precision_score(test_y, pred, pos_label=1)),
    }

    feature_importance = None
    if model_name in {"logreg", "linear_svm"}:
        coeffs = estimator.named_steps["clf"].coef_[0]
        names = estimator.named_steps["imputer"].get_feature_names_out(train_x.columns)
        feature_importance = pd.DataFrame(
            {
                "feature_name": names,
                "coefficient": coeffs,
                "abs_coefficient": np.abs(coeffs),
            }
        ).sort_values("abs_coefficient", ascending=False)
    elif model_name == "rf":
        names = estimator.named_steps["imputer"].get_feature_names_out(train_x.columns)
        feature_importance = pd.DataFrame(
            {
                "feature_name": names,
                "importance": estimator.named_steps["clf"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)
    return result, feature_importance


def aggregate_feature_groups(ranking: pd.DataFrame) -> pd.DataFrame:
    ranking = ranking.copy()
    ranking["feature_group"] = ranking["feature_name"].str.split("_").str[0]
    if "abs_coefficient" in ranking.columns:
        return ranking.groupby("feature_group", as_index=False)["abs_coefficient"].mean().sort_values("abs_coefficient", ascending=False)
    if "importance" in ranking.columns:
        return ranking.groupby("feature_group", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    if "f_score" in ranking.columns:
        return ranking.groupby("feature_group", as_index=False)["f_score"].mean().sort_values("f_score", ascending=False)
    raise ValueError("Unsupported ranking frame")


def write_importance_tables(prefix: str, importance_df: pd.DataFrame | None) -> None:
    if importance_df is None:
        return
    importance_path = TABLES_PHASE1_ROOT / f"{prefix}_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    aggregate_feature_groups(importance_df).to_csv(
        TABLES_PHASE1_ROOT / f"{prefix}_feature_group_importance.csv",
        index=False,
    )

    filtered = importance_df[~importance_df["feature_name"].str.startswith("missingindicator_")].copy()
    filtered.to_csv(TABLES_PHASE1_ROOT / f"{prefix}_feature_importance_no_missing_indicators.csv", index=False)


def grid_search_single_seed(
    df: pd.DataFrame,
    prefix: str,
    log,
    grouping_levels: list[list[str]] | None = None,
) -> dict[str, object]:
    train_df, test_df = grouped_train_test_split(df, test_size=0.2, seed=SEED)
    log(f"{prefix}: train rows/groups={len(train_df)}/{train_df['group_id'].nunique()} test rows/groups={len(test_df)}/{test_df['group_id'].nunique()}")

    subsets = ["all_universal", "text_only", "acoustic_only", "text_plus_acoustic"]
    ks = [10, 20, 50, "all"]
    models = ["logreg", "linear_svm", "rf"]

    results = []
    best_config = None
    best_score = -1.0
    stored_rankings = {}

    for subset in subsets:
        feature_cols = feature_subset_columns(df, subset)
        train_norm, test_norm = normalize_with_fallback(train_df, test_df, feature_cols, grouping_levels=grouping_levels)
        y_train = train_norm["binary_label"].astype(int)
        y_test = test_norm["binary_label"].astype(int)
        for k in ks:
            selected_cols, anova_ranking = select_top_k(train_norm, y_train, feature_cols, k)
            stored_rankings[(subset, str(k), "anova")] = anova_ranking
            x_train = train_norm[selected_cols]
            x_test = test_norm[selected_cols]
            for model_name in models:
                metrics, importance = evaluate_model(model_name, x_train, y_train, x_test, y_test)
                row = {
                    "subset": subset,
                    "top_k": k,
                    "model": model_name,
                    "num_features": len(selected_cols),
                    **metrics,
                }
                results.append(row)
                log(f"{prefix}: {subset} top_k={k} model={model_name} bal_acc={metrics['balanced_accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}")
                if metrics["balanced_accuracy"] > best_score:
                    best_score = metrics["balanced_accuracy"]
                    best_config = {
                        "subset": subset,
                        "top_k": str(k),
                        "model": model_name,
                        "selected_cols": selected_cols,
                        "importance": importance,
                    }

    results_df = pd.DataFrame(results).sort_values(["balanced_accuracy", "macro_f1"], ascending=False)
    results_df.to_csv(TABLES_PHASE1_ROOT / f"{prefix}_model_results.csv", index=False)

    if best_config is None:
        raise RuntimeError(f"No model configuration completed for {prefix}.")

    best_anova = stored_rankings[(best_config["subset"], best_config["top_k"], "anova")]
    best_anova.to_csv(TABLES_PHASE1_ROOT / f"{prefix}_anova_ranking.csv", index=False)
    aggregate_feature_groups(best_anova).to_csv(TABLES_PHASE1_ROOT / f"{prefix}_anova_feature_groups.csv", index=False)
    write_importance_tables(prefix, best_config["importance"])

    summary = {
        "seed": SEED,
        "rows": int(len(df)),
        "groups": int(df["group_id"].nunique()),
        "grouping_levels": grouping_levels
        or [["language", "task_type", "dataset_name"], ["language", "task_type"], ["language"], []],
        "best_config": {
            "subset": best_config["subset"],
            "top_k": best_config["top_k"],
            "model": best_config["model"],
            "num_features": len(best_config["selected_cols"]),
        },
        "best_result": results_df.iloc[0].to_dict(),
        "methodology": {
            "split": "single 80/20 grouped holdout by participant/group id with group-level label stratification",
            "normalization": "train-only z-scoring with hierarchical fallback across grouping levels",
            "feature_selection": "ANOVA f_classif fit on train only",
            "classifiers": ["logistic regression", "linear SVM", "random forest"],
            "imputation": "median imputation with missingness indicators",
        },
    }
    with (TABLES_PHASE1_ROOT / f"{prefix}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    log = make_logger("phase1_single_seed_experiment")
    log("Loading manifest and features")
    manifest = pd.read_json(MANIFEST_PATH, lines=True)
    features = pd.read_csv(FEATURES_PATH)
    df = manifest.merge(
        features,
        on=["sample_id", "group_id", "dataset_name", "language", "task_type", "diagnosis_mapped", "binary_label"],
    )

    eligible = df[df["binary_label"].isin([0, 1])].copy()
    log(f"Eligible AD_vs_HC rows: {len(eligible)}")
    log(f"Eligible groups: {eligible['group_id'].nunique()}")

    pooled_summary = grid_search_single_seed(
        eligible,
        prefix="single_seed",
        log=log,
        grouping_levels=[["language", "task_type", "dataset_name"], ["language", "task_type"], ["language"], []],
    )
    log(f"Pooled best config: {pooled_summary['best_config']}")

    pd_ctp = eligible[eligible["task_type"] == "PD_CTP"].copy()
    if len(pd_ctp) > 50 and pd_ctp["binary_label"].nunique() == 2:
        pd_summary = grid_search_single_seed(
            pd_ctp,
            prefix="pd_ctp_single_seed",
            log=log,
            grouping_levels=[["language", "dataset_name"], ["language"], []],
        )
        log(f"PD_CTP best config: {pd_summary['best_config']}")

    log("Finished single-seed phase 1 experiment run")


if __name__ == "__main__":
    main()
