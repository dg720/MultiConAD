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

from processing.phase1.common import PHASE1_ROOT, TABLES_PHASE1_ROOT, make_logger, write_json


SEED = 42
PERMUTATION_REPEATS = 10
FEATURES_PATH = PHASE1_ROOT / "phase1_features.csv"
MANIFEST_PATH = PHASE1_ROOT / "phase1_manifest.jsonl"
RICH_SWEEP_ROOT = TABLES_PHASE1_ROOT / "rich_sweep"
RICH_SWEEP_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class RunSpec:
    name: str
    scope: str
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
    train_df = df[df["group_id"].isin(set(train_groups))].copy()
    test_df = df[df["group_id"].isin(set(test_groups))].copy()
    return train_df, test_df


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


def core_feature_columns(df: pd.DataFrame) -> list[str]:
    allowed_prefixes = ("len_", "lex_", "pause_", "disc_", "syn_", "graph_", "ac_")
    return [
        col
        for col in df.columns
        if col.startswith(allowed_prefixes) and pd.api.types.is_numeric_dtype(df[col])
    ]


def feature_subset_columns(df: pd.DataFrame, subset_name: str) -> list[str]:
    all_cols = core_feature_columns(df)
    text_core = {
        col
        for col in all_cols
        if (
            col.startswith(("len_", "lex_", "disc_", "syn_"))
            and col
            not in {"len_audio_duration", "len_speech_duration", "len_tokens_per_second", "len_syllables_per_second"}
        )
    }
    pause_cols = {
        col
        for col in all_cols
        if col.startswith("pause_")
        or col in {"len_audio_duration", "len_speech_duration", "len_tokens_per_second", "len_syllables_per_second"}
    }
    graph_cols = {col for col in all_cols if col.startswith("graph_")}
    acoustic_cols = {col for col in all_cols if col.startswith("ac_")}

    subsets = {
        "all_universal": set(all_cols),
        "text_only": text_core,
        "pause_only": pause_cols,
        "speech_graph_only": graph_cols,
        "acoustic_only": acoustic_cols,
        "text_plus_pause": text_core | pause_cols,
        "text_plus_pause_plus_graph": text_core | pause_cols | graph_cols,
        "text_plus_pause_plus_acoustic": text_core | pause_cols | acoustic_cols,
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
    ranking = pd.DataFrame(
        {"feature_name": keep_cols, "f_score": scores[0], "p_value": scores[1], "rank_metric": scores[0]}
    ).sort_values("f_score", ascending=False)
    if k == "all":
        selected = keep_cols
    else:
        selected = ranking["feature_name"].head(min(int(k), len(ranking))).tolist()
    return selected, ranking


def model_specs() -> list[dict[str, object]]:
    return [
        {
            "model_family": "lr",
            "model_variant": "c1",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced", random_state=SEED)),
                ]
            ),
        },
        {
            "model_family": "dt",
            "model_variant": "depth20",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("clf", DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=SEED)),
                ]
            ),
        },
        {
            "model_family": "rf",
            "model_variant": "n200",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=200,
                            class_weight="balanced",
                            random_state=SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        },
        {
            "model_family": "svm",
            "model_variant": "linear_c1",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    ("clf", SVC(C=1.0, kernel="linear", probability=True, class_weight="balanced", random_state=SEED)),
                ]
            ),
        },
        {
            "model_family": "svm",
            "model_variant": "rbf_c1",
            "estimator": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    ("clf", SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)),
                ]
            ),
        },
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
        result["auroc"] = float("nan")
    try:
        result["auprc"] = float(average_precision_score(y_true, score_values))
    except Exception:
        result["auprc"] = float("nan")
    return result


def native_importance_frame(model_family: str, estimator: Pipeline, feature_names: list[str]) -> tuple[pd.DataFrame | None, str | None]:
    names = estimator.named_steps["imputer"].get_feature_names_out(feature_names)
    clf = estimator.named_steps["clf"]
    if model_family in {"lr", "svm"} and hasattr(clf, "coef_"):
        coeffs = clf.coef_[0]
        frame = pd.DataFrame(
            {
                "feature_name": names,
                "native_importance": coeffs,
                "native_importance_abs": np.abs(coeffs),
                "native_importance_type": "coefficient",
                "is_missingness_indicator": [name.startswith("missingindicator_") for name in names],
            }
        ).sort_values("native_importance_abs", ascending=False)
        return frame, "coefficient"
    if model_family in {"dt", "rf"} and hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        frame = pd.DataFrame(
            {
                "feature_name": names,
                "native_importance": values,
                "native_importance_abs": np.abs(values),
                "native_importance_type": "impurity_importance",
                "is_missingness_indicator": [name.startswith("missingindicator_") for name in names],
            }
        ).sort_values("native_importance_abs", ascending=False)
        return frame, "impurity_importance"
    return None, None


def permutation_importance_frame(estimator: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    perm = permutation_importance(
        estimator,
        x_test,
        y_test,
        n_repeats=PERMUTATION_REPEATS,
        random_state=SEED,
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature_name": list(x_test.columns),
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
            "rank_metric": perm.importances_mean,
            "is_missingness_indicator": False,
        }
    ).sort_values("permutation_importance_mean", ascending=False)


def aggregate_feature_groups(ranking: pd.DataFrame, value_col: str) -> pd.DataFrame:
    frame = ranking.copy()
    frame["feature_group"] = frame["feature_name"].str.split("_").str[0]
    return (
        frame.groupby("feature_group", as_index=False)[value_col]
        .mean()
        .sort_values(value_col, ascending=False)
        .reset_index(drop=True)
    )


def write_importance_tables(output_dir: Path, prefix: str, ranking: pd.DataFrame, stem: str, value_col: str) -> None:
    ranking.to_csv(output_dir / f"{prefix}_{stem}.csv", index=False)
    aggregate_feature_groups(ranking, value_col).to_csv(output_dir / f"{prefix}_{stem}_feature_groups.csv", index=False)
    filtered = ranking[~ranking["feature_name"].str.startswith("missingindicator_")].copy()
    filtered.to_csv(output_dir / f"{prefix}_{stem}_no_missing_indicators.csv", index=False)
    aggregate_feature_groups(filtered, value_col).to_csv(
        output_dir / f"{prefix}_{stem}_feature_groups_no_missing_indicators.csv",
        index=False,
    )


def run_feature_sweep(
    df: pd.DataFrame,
    run_name: str,
    grouping_levels: list[list[str]],
    log,
    output_dir: Path,
) -> dict[str, object]:
    train_df, test_df = grouped_train_test_split(df, test_size=0.2, seed=SEED)
    log(
        f"{run_name}: train rows/groups={len(train_df)}/{train_df['group_id'].nunique()} "
        f"test rows/groups={len(test_df)}/{test_df['group_id'].nunique()}"
    )

    subsets = [
        "all_universal",
        "text_only",
        "pause_only",
        "speech_graph_only",
        "acoustic_only",
        "text_plus_pause",
        "text_plus_pause_plus_graph",
        "text_plus_pause_plus_acoustic",
    ]
    ks: list[int | str] = [5, 10, 20, 50, 100, "all"]
    results = []
    best_state: dict[str, object] | None = None
    best_score = -np.inf
    best_tiebreak = -np.inf
    best_anova: pd.DataFrame | None = None

    for subset in subsets:
        feature_cols = feature_subset_columns(df, subset)
        if not feature_cols:
            continue
        train_norm, test_norm = normalize_with_fallback(train_df, test_df, feature_cols, grouping_levels)
        y_train = train_norm["binary_label"].astype(int)
        y_test = test_norm["binary_label"].astype(int)
        for k in ks:
            try:
                selected_cols, anova_ranking = select_top_k(train_norm, y_train, feature_cols, k)
            except RuntimeError:
                log(f"{run_name}: subset={subset} top_k={k} skipped because no usable features remained")
                continue
            x_train = train_norm[selected_cols]
            x_test = test_norm[selected_cols]
            for spec in model_specs():
                estimator = clone(spec["estimator"])
                estimator.fit(x_train, y_train)
                pred = estimator.predict(x_test)
                if hasattr(estimator, "predict_proba"):
                    scores = estimator.predict_proba(x_test)[:, 1]
                else:
                    scores = estimator.decision_function(x_test)
                metrics = compute_metrics(y_test, pred, scores)
                row = {
                    "run_name": run_name,
                    "subset": subset,
                    "top_k": k,
                    "num_features": len(selected_cols),
                    "model_family": spec["model_family"],
                    "model_variant": spec["model_variant"],
                    **metrics,
                }
                results.append(row)
                log(
                    f"{run_name}: subset={subset} top_k={k} model={spec['model_family']}:{spec['model_variant']} "
                    f"bal_acc={metrics['balanced_accuracy']:.3f} auroc={metrics['auroc']:.3f}"
                )
                if (
                    metrics["balanced_accuracy"] > best_score
                    or (
                        metrics["balanced_accuracy"] == best_score
                        and metrics["auroc"] > best_tiebreak
                    )
                ):
                    best_score = metrics["balanced_accuracy"]
                    best_tiebreak = metrics["auroc"]
                    best_state = {
                        "subset": subset,
                        "top_k": str(k),
                        "selected_cols": selected_cols,
                        "model_family": spec["model_family"],
                        "model_variant": spec["model_variant"],
                        "estimator": estimator,
                        "x_test": x_test,
                        "y_test": y_test,
                        "metrics": metrics,
                    }
                    best_anova = anova_ranking.copy()

    if best_state is None or best_anova is None:
        raise RuntimeError(f"No completed sweep state for {run_name}")

    results_df = pd.DataFrame(results).sort_values(["balanced_accuracy", "auroc", "macro_f1"], ascending=False)
    results_df.to_csv(output_dir / f"{run_name}_model_results.csv", index=False)

    best_anova.to_csv(output_dir / f"{run_name}_anova_ranking.csv", index=False)
    aggregate_feature_groups(best_anova, "f_score").to_csv(output_dir / f"{run_name}_anova_feature_groups.csv", index=False)

    estimator = best_state["estimator"]
    x_test = best_state["x_test"]
    y_test = best_state["y_test"]
    selected_cols = best_state["selected_cols"]
    model_family = best_state["model_family"]

    permutation_df = permutation_importance_frame(estimator, x_test, y_test)
    write_importance_tables(output_dir, run_name, permutation_df, "permutation_importance", "permutation_importance_mean")

    native_df, native_type = native_importance_frame(model_family, estimator, selected_cols)
    if native_df is not None:
        write_importance_tables(output_dir, run_name, native_df, "native_importance", "native_importance_abs")

    summary = {
        "seed": SEED,
        "rows": int(len(df)),
        "groups": int(df["group_id"].nunique()),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_groups": int(train_df["group_id"].nunique()),
        "test_groups": int(test_df["group_id"].nunique()),
        "class_counts": df["binary_label"].value_counts().sort_index().to_dict(),
        "grouping_levels": grouping_levels,
        "best_config": {
            "subset": best_state["subset"],
            "top_k": best_state["top_k"],
            "num_features": len(best_state["selected_cols"]),
            "model_family": best_state["model_family"],
            "model_variant": best_state["model_variant"],
            "native_importance_type": native_type,
        },
        "best_result": results_df.iloc[0].to_dict(),
        "methodology": {
            "split": "single grouped 80/20 holdout by group_id with group-label stratification",
            "normalization": "train-only hierarchical z-scoring with fallback by grouping levels",
            "feature_selection": "ANOVA f_classif fit on training data only",
            "feature_subsets": subsets,
            "top_k_values": ks,
            "classifiers": [
                "logistic regression C=1",
                "decision tree max_depth=20",
                "random forest n_estimators=200",
                "SVM linear C=1",
                "SVM rbf C=1",
            ],
            "imputation": "median imputation with explicit missingness indicators",
            "importance_primary": "held-out permutation importance on the fitted best model using original selected input columns",
            "importance_secondary": "native coefficients or impurity importances on the transformed post-imputation design matrix when available",
        },
    }
    write_json(output_dir / f"{run_name}_summary.json", summary)
    return summary


def dataset_for_spec(df: pd.DataFrame, spec: RunSpec) -> pd.DataFrame:
    out = df.copy()
    for key, value in spec.data_filter.items():
        if isinstance(value, (list, tuple, set)):
            out = out[out[key].isin(value)].copy()
        else:
            out = out[out[key] == value].copy()
    return out


def rank_from_importance(df: pd.DataFrame, value_col: str, top_n: int = 30) -> pd.DataFrame:
    ranked = df.sort_values(value_col, ascending=False).reset_index(drop=True).copy()
    ranked["rank"] = ranked.index + 1
    return ranked.head(top_n)


def summarize_cross_lingual(output_dir: Path, run_names: list[str], summary_name: str) -> None:
    frames = []
    group_frames = []
    for run_name in run_names:
        path = output_dir / f"{run_name}_permutation_importance.csv"
        group_path = output_dir / f"{run_name}_permutation_importance_feature_groups.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        top_features = rank_from_importance(frame, "permutation_importance_mean", top_n=30)
        top_features["run_name"] = run_name
        frames.append(top_features)

        group_frame = pd.read_csv(group_path)
        top_groups = rank_from_importance(group_frame, "permutation_importance_mean", top_n=10)
        top_groups["run_name"] = run_name
        group_frames.append(top_groups)

    if not frames:
        return

    features = pd.concat(frames, ignore_index=True)
    features["language"] = features["run_name"].str.extract(r"language_([a-z]+)")
    feature_summary = (
        features.groupby("feature_name", as_index=False)
        .agg(
            languages_present=("language", lambda s: ",".join(sorted(set(str(v) for v in s if pd.notna(v))))),
            num_languages=("language", lambda s: len(set(v for v in s if pd.notna(v)))),
            mean_rank=("rank", "mean"),
            mean_importance=("permutation_importance_mean", "mean"),
            max_importance=("permutation_importance_mean", "max"),
        )
        .sort_values(["num_languages", "mean_rank", "mean_importance"], ascending=[False, True, False])
    )
    feature_summary.to_csv(output_dir / f"{summary_name}_feature_stability.csv", index=False)
    features.to_csv(output_dir / f"{summary_name}_feature_top30_by_language.csv", index=False)

    if group_frames:
        groups = pd.concat(group_frames, ignore_index=True)
        groups["language"] = groups["run_name"].str.extract(r"language_([a-z]+)")
        group_summary = (
            groups.groupby("feature_group", as_index=False)
            .agg(
                languages_present=("language", lambda s: ",".join(sorted(set(str(v) for v in s if pd.notna(v))))),
                num_languages=("language", lambda s: len(set(v for v in s if pd.notna(v)))),
                mean_rank=("rank", "mean"),
                mean_importance=("permutation_importance_mean", "mean"),
            )
            .sort_values(["num_languages", "mean_rank", "mean_importance"], ascending=[False, True, False])
        )
        group_summary.to_csv(output_dir / f"{summary_name}_feature_group_stability.csv", index=False)
        groups.to_csv(output_dir / f"{summary_name}_feature_group_top10_by_language.csv", index=False)


def validate_feature_table(features: pd.DataFrame, log) -> None:
    egemaps_cols = [col for col in features.columns if col.startswith("ac_egemaps_")]
    if not egemaps_cols:
        raise RuntimeError("No openSMILE eGeMAPS columns were found in phase1_features.csv")
    log(f"Validated rich acoustic block: {len(egemaps_cols)} openSMILE eGeMAPS columns detected")


def main() -> None:
    log = make_logger("phase1_rich_sweep")
    log("Loading manifest and rich feature table")
    manifest = pd.read_json(MANIFEST_PATH, lines=True)
    features = pd.read_csv(FEATURES_PATH)
    validate_feature_table(features, log)

    df = manifest.merge(
        features,
        on=["sample_id", "group_id", "dataset_name", "language", "task_type", "diagnosis_mapped", "binary_label"],
    )
    eligible = df[df["binary_label"].isin([0, 1])].copy()
    log(f"Eligible AD_vs_HC rows={len(eligible)} groups={eligible['group_id'].nunique()}")

    run_specs = [
        RunSpec(
            name="benchmark_wide_rich",
            scope="pooled",
            data_filter={},
            grouping_levels=[["language", "task_type", "dataset_name"], ["language", "task_type"], ["language"], []],
        ),
        RunSpec(
            name="pd_ctp_pooled_rich",
            scope="task_primary",
            data_filter={"task_type": "PD_CTP"},
            grouping_levels=[["language", "dataset_name"], ["language"], []],
        ),
        RunSpec(
            name="language_en_all_rich",
            scope="language_all",
            data_filter={"language": "en"},
            grouping_levels=[["task_type", "dataset_name"], ["task_type"], []],
        ),
        RunSpec(
            name="language_es_all_rich",
            scope="language_all",
            data_filter={"language": "es"},
            grouping_levels=[["task_type", "dataset_name"], ["task_type"], []],
        ),
        RunSpec(
            name="language_zh_all_rich",
            scope="language_all",
            data_filter={"language": "zh"},
            grouping_levels=[["task_type", "dataset_name"], ["task_type"], []],
        ),
        RunSpec(
            name="language_el_all_rich",
            scope="language_all",
            data_filter={"language": "el"},
            grouping_levels=[["task_type", "dataset_name"], ["task_type"], []],
        ),
        RunSpec(
            name="language_en_pd_ctp_rich",
            scope="language_pd_ctp",
            data_filter={"language": "en", "task_type": "PD_CTP"},
            grouping_levels=[["dataset_name"], []],
        ),
        RunSpec(
            name="language_zh_pd_ctp_rich",
            scope="language_pd_ctp",
            data_filter={"language": "zh", "task_type": "PD_CTP"},
            grouping_levels=[["dataset_name"], []],
        ),
        RunSpec(
            name="language_el_pd_ctp_rich",
            scope="language_pd_ctp",
            data_filter={"language": "el", "task_type": "PD_CTP"},
            grouping_levels=[["dataset_name"], []],
        ),
        RunSpec(
            name="language_es_reading_rich",
            scope="language_task_secondary",
            data_filter={"language": "es", "task_type": "READING"},
            grouping_levels=[["dataset_name"], []],
        ),
    ]

    completed = []
    skipped = []
    summaries = {}

    for spec in run_specs:
        run_df = dataset_for_spec(eligible, spec)
        if (
            len(run_df) < spec.min_rows
            or run_df["group_id"].nunique() < spec.min_groups
            or run_df["binary_label"].nunique() < 2
        ):
            skipped.append(
                {
                    "run_name": spec.name,
                    "rows": int(len(run_df)),
                    "groups": int(run_df["group_id"].nunique()),
                    "class_counts": run_df["binary_label"].value_counts().to_dict(),
                    "reason": "insufficient rows/groups/classes",
                }
            )
            log(f"Skipping {spec.name}: insufficient rows/groups/classes")
            continue
        log(
            f"Running {spec.name}: rows={len(run_df)} groups={run_df['group_id'].nunique()} "
            f"classes={run_df['binary_label'].value_counts().to_dict()}"
        )
        summaries[spec.name] = run_feature_sweep(
            run_df,
            run_name=spec.name,
            grouping_levels=spec.grouping_levels,
            log=log,
            output_dir=RICH_SWEEP_ROOT,
        )
        completed.append(spec.name)

    summarize_cross_lingual(
        RICH_SWEEP_ROOT,
        [name for name in completed if name.endswith("_all_rich") and name.startswith("language_")],
        "cross_lingual_all_tasks",
    )
    summarize_cross_lingual(
        RICH_SWEEP_ROOT,
        [name for name in completed if name.endswith("_pd_ctp_rich") and name.startswith("language_")],
        "cross_lingual_pd_ctp",
    )

    run_index = {
        "completed_runs": completed,
        "skipped_runs": skipped,
        "summaries": summaries,
    }
    write_json(RICH_SWEEP_ROOT / "rich_sweep_run_index.json", run_index)
    pd.DataFrame(skipped).to_csv(RICH_SWEEP_ROOT / "rich_sweep_skipped_runs.csv", index=False)
    log("Completed phase1 rich sweep")


if __name__ == "__main__":
    main()
