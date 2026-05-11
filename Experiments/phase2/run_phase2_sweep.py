import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.phase1.common import make_logger, write_json
from processing.phase2.common import PHASE2_ROOT, TABLES_PHASE2_ROOT


SEED = 42
PERMUTATION_REPEATS = 5
FEATURES_PATH = PHASE2_ROOT / "phase2_features.csv"
RUN_ROOT = TABLES_PHASE2_ROOT / "rich_sweep"
RUN_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunSpec:
    name: str
    data_filter: dict[str, object]
    grouping_levels: list[list[str]]
    min_rows: int = 60
    min_groups: int = 20


def grouped_train_test_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED):
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


def normalize_with_fallback(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], grouping_levels: list[list[str]]):
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
                key = tuple(level)
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
                fillable = [col for col in remaining if pd.notna(raw[col]) and pd.notna(mean[col]) and pd.notna(std[col]) and std[col] != 0]
                for col in fillable:
                    normalized[col] = (raw[col] - mean[col]) / std[col]
                remaining -= set(fillable)
            normalized_rows.append(normalized.reindex(feature_cols))
        return pd.DataFrame(normalized_rows, columns=feature_cols, index=frame.index)

    train_out[feature_cols] = apply_frame(train_df)
    test_out[feature_cols] = apply_frame(test_df)
    return train_out, test_out


def core_feature_columns(df: pd.DataFrame) -> list[str]:
    allowed_prefixes = ("len_", "lex_", "pause_", "disc_", "syn_", "graph_", "ac_", "sx_", "par_", "pd_", "rd_", "fc_")
    return [col for col in df.columns if col.startswith(allowed_prefixes) and pd.api.types.is_numeric_dtype(df[col])]


def feature_subset_columns(df: pd.DataFrame, subset_name: str) -> list[str]:
    all_cols = core_feature_columns(df)
    phase1_universal = {col for col in all_cols if col.startswith(("len_", "lex_", "pause_", "disc_", "syn_", "graph_", "ac_"))}
    rich_syntax = {col for col in all_cols if col.startswith("sx_")}
    rich_acoustic = {col for col in all_cols if col.startswith("par_")}
    task_semantic = {col for col in all_cols if col.startswith(("pd_", "rd_", "fc_"))}
    pause_cols = {col for col in all_cols if col.startswith("pause_") or col in {"len_audio_duration", "len_speech_duration", "len_tokens_per_second", "len_syllables_per_second"}}
    rich_text = {col for col in all_cols if col.startswith(("len_", "lex_", "disc_", "syn_", "graph_", "sx_"))}
    acoustic_all = {col for col in all_cols if col.startswith(("ac_", "par_"))}

    subsets = {
        "phase1_universal": phase1_universal,
        "rich_universal": phase1_universal | rich_syntax | rich_acoustic,
        "task_specific_semantic": task_semantic,
        "semantic_plus_pause": task_semantic | pause_cols,
        "rich_text_only": rich_text,
        "rich_acoustic_only": acoustic_all | pause_cols,
        "all_phase2": set(all_cols),
    }
    if subset_name not in subsets:
        raise ValueError(f"Unknown subset: {subset_name}")
    return sorted(subsets[subset_name])


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
    ranking = pd.DataFrame({"feature_name": keep_cols, "f_score": scores[0], "p_value": scores[1], "rank_metric": scores[0]}).sort_values("f_score", ascending=False)
    selected = keep_cols if k == "all" else ranking["feature_name"].head(min(int(k), len(ranking))).tolist()
    return selected, ranking


def model_specs():
    return [
        {"model_family": "lr", "model_variant": "c1", "estimator": Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced", random_state=SEED))])},
        {"model_family": "dt", "model_variant": "depth20", "estimator": Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("clf", DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=SEED))])},
        {"model_family": "rf", "model_variant": "n200", "estimator": Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1))])},
        {"model_family": "svm", "model_variant": "linear_c1", "estimator": Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler()), ("clf", SVC(C=1.0, kernel="linear", probability=True, class_weight="balanced", random_state=SEED))])},
        {"model_family": "svm", "model_variant": "rbf_c1", "estimator": Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", StandardScaler()), ("clf", SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED))])},
    ]


def compute_metrics(y_true: pd.Series, pred: np.ndarray, score_values: np.ndarray) -> dict[str, float]:
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "sensitivity": float(recall_score(y_true, pred, pos_label=1)),
        "specificity": float(recall_score(y_true, pred, pos_label=0)),
        "precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
    }
    try:
        result["auroc"] = float(roc_auc_score(y_true, score_values))
    except Exception:
        result["auroc"] = np.nan
    try:
        result["auprc"] = float(average_precision_score(y_true, score_values))
    except Exception:
        result["auprc"] = np.nan
    return result


def apply_filters(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for key, value in filters.items():
        if isinstance(value, (list, tuple, set)):
            out = out[out[key].isin(list(value))]
        else:
            out = out[out[key] == value]
    return out


def importance_frame(pipeline, feature_names: list[str], test_x: pd.DataFrame, test_y: pd.Series):
    perm = permutation_importance(pipeline, test_x[feature_names], test_y, n_repeats=PERMUTATION_REPEATS, random_state=SEED, scoring="balanced_accuracy")
    frame = pd.DataFrame({"feature_name": feature_names, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    frame["feature_group"] = frame["feature_name"].str.split("_", n=1).str[0]
    return frame


def aggregate_group_importance(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("feature_group", dropna=False)[["importance_mean", "importance_std"]].sum().reset_index().sort_values("importance_mean", ascending=False)


def run_spec(df: pd.DataFrame, spec: RunSpec, log):
    filtered = apply_filters(df, spec.data_filter)
    filtered = filtered[filtered["binary_label"].isin([0, 1])].copy()
    filtered = filtered.dropna(subset=["group_id"])
    if len(filtered) < spec.min_rows or filtered["group_id"].nunique() < spec.min_groups:
        log(f"Skipping {spec.name}: rows={len(filtered)} groups={filtered['group_id'].nunique()}")
        return None

    train_df, test_df = grouped_train_test_split(filtered)
    results = []
    importance_tables = {}
    anova_tables = {}
    subset_names = ["phase1_universal", "rich_universal", "task_specific_semantic", "semantic_plus_pause", "all_phase2"]
    top_ks = [50, 100]
    for subset_name in subset_names:
        subset_cols = feature_subset_columns(filtered, subset_name)
        if not subset_cols:
            continue
        train_norm, test_norm = normalize_with_fallback(train_df, test_df, subset_cols, spec.grouping_levels)
        for top_k in top_ks:
            try:
                selected_cols, ranking = select_top_k(train_norm, train_norm["binary_label"], subset_cols, top_k)
            except RuntimeError:
                continue
            for model in model_specs():
                pipeline = clone(model["estimator"])
                pipeline.fit(train_norm[selected_cols], train_norm["binary_label"])
                pred = pipeline.predict(test_norm[selected_cols])
                try:
                    score_values = pipeline.predict_proba(test_norm[selected_cols])[:, 1]
                except Exception:
                    score_values = pipeline.decision_function(test_norm[selected_cols])
                metrics = compute_metrics(test_norm["binary_label"], pred, score_values)
                row = {"run_name": spec.name, "subset_name": subset_name, "top_k": top_k, "num_selected_features": len(selected_cols), "model_family": model["model_family"], "model_variant": model["model_variant"]}
                row.update(metrics)
                results.append(row)
                key = (subset_name, str(top_k), model["model_family"], model["model_variant"])
                importance_tables[key] = importance_frame(pipeline, selected_cols, test_norm, test_norm["binary_label"])
                anova_tables[key] = ranking

    if not results:
        return None
    result_df = pd.DataFrame(results).sort_values(["balanced_accuracy", "auroc", "macro_f1"], ascending=False)
    best = result_df.iloc[0].to_dict()
    key = (best["subset_name"], str(best["top_k"]), best["model_family"], best["model_variant"])
    importance_df = importance_tables[key]
    importance_group_df = aggregate_group_importance(importance_df)
    anova_df = anova_tables[key]
    anova_group_df = anova_df.assign(feature_group=anova_df["feature_name"].str.split("_", n=1).str[0]).groupby("feature_group", dropna=False)[["f_score"]].mean().reset_index().sort_values("f_score", ascending=False)
    stem = spec.name
    result_df.to_csv(RUN_ROOT / f"{stem}_model_results.csv", index=False)
    importance_df.to_csv(RUN_ROOT / f"{stem}_permutation_importance.csv", index=False)
    importance_group_df.to_csv(RUN_ROOT / f"{stem}_permutation_importance_feature_groups.csv", index=False)
    anova_df.to_csv(RUN_ROOT / f"{stem}_anova_ranking.csv", index=False)
    anova_group_df.to_csv(RUN_ROOT / f"{stem}_anova_feature_groups.csv", index=False)
    summary = {"run_name": spec.name, "n_rows": int(len(filtered)), "n_groups": int(filtered["group_id"].nunique()), "best_model": best}
    write_json(RUN_ROOT / f"{stem}_summary.json", summary)
    log(f"Completed {spec.name}: best={best['model_family']}:{best['model_variant']} subset={best['subset_name']} top_k={best['top_k']} bal_acc={best['balanced_accuracy']:.3f}")
    return summary


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(name="benchmark_wide_phase2", data_filter={}, grouping_levels=[["language", "dataset_name"], ["language"], []], min_rows=300, min_groups=100),
        RunSpec(name="pd_ctp_pooled_phase2", data_filter={"task_type": "PD_CTP"}, grouping_levels=[["language", "dataset_name"], ["language"], []], min_rows=300, min_groups=100),
        RunSpec(name="cookie_theft_pooled_phase2", data_filter={"task_type": "PD_CTP", "phase2_prompt_family": "cookie_theft"}, grouping_levels=[["language", "dataset_name"], ["language"], []], min_rows=150, min_groups=50),
        RunSpec(name="language_en_pd_ctp_phase2", data_filter={"language": "en", "task_type": "PD_CTP"}, grouping_levels=[["dataset_name"], []], min_rows=150, min_groups=50),
        RunSpec(name="language_zh_pd_ctp_phase2", data_filter={"language": "zh", "task_type": "PD_CTP"}, grouping_levels=[["dataset_name"], []], min_rows=80, min_groups=20),
        RunSpec(name="language_el_pd_ctp_phase2", data_filter={"language": "el", "task_type": "PD_CTP"}, grouping_levels=[["dataset_name"], []], min_rows=80, min_groups=20),
        RunSpec(name="language_en_cookie_theft_phase2", data_filter={"language": "en", "task_type": "PD_CTP", "phase2_prompt_family": "cookie_theft"}, grouping_levels=[["dataset_name"], []], min_rows=120, min_groups=40),
        RunSpec(name="language_zh_cookie_theft_phase2", data_filter={"language": "zh", "task_type": "PD_CTP", "phase2_prompt_family": "cookie_theft"}, grouping_levels=[["dataset_name"], []], min_rows=60, min_groups=20),
        RunSpec(name="language_es_reading_phase2", data_filter={"language": "es", "task_type": "READING"}, grouping_levels=[["dataset_name"], []], min_rows=100, min_groups=20),
    ]


def main():
    log = make_logger("phase2_rich_sweep")
    df = pd.read_csv(FEATURES_PATH)
    summaries = []
    skipped = []
    for spec in run_specs():
        try:
            summary = run_spec(df, spec, log)
            if summary is None:
                skipped.append({"run_name": spec.name, "reason": "min_rows_or_groups"})
            else:
                summaries.append(summary)
        except Exception as exc:
            skipped.append({"run_name": spec.name, "reason": str(exc)})
            log(f"Failed {spec.name}: {exc}")
    write_json(RUN_ROOT / "phase2_run_index.json", {"completed_runs": summaries, "skipped_runs": skipped})
    pd.DataFrame(skipped).to_csv(RUN_ROOT / "phase2_skipped_runs.csv", index=False)


if __name__ == "__main__":
    main()
