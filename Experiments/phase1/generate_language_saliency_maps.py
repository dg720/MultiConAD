from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.phase1.run_rich_sweep as p1


REPORT_ROOT = PROJECT_ROOT / "13.05-report"
LANGUAGE_SPECS = {
    "en": {
        "label": "English",
        "filters": {"language": "en"},
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
    "es": {
        "label": "Spanish",
        "filters": {"language": "es"},
        "grouping_levels": [["task_type", "dataset_name"], ["task_type"], []],
    },
}
PALETTE = {"AD": "#b24a4a", "HC": "#4f7f68"}


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


def get_model_estimator(model_family: str, model_variant: str):
    for spec in p1.model_specs():
        if spec["model_family"] == model_family and spec["model_variant"] == model_variant:
            return clone(spec["estimator"])
    raise KeyError((model_family, model_variant))


def best_row_for_run(run_name: str, subset: str = "all_universal") -> pd.Series:
    df = pd.read_csv(p1.RICH_SWEEP_ROOT / f"{run_name}_model_results.csv")
    return (
        df[df["subset"] == subset]
        .sort_values(["accuracy", "auroc", "balanced_accuracy", "macro_f1"], ascending=False)
        .iloc[0]
    )


def reconstruct_run(
    merged: pd.DataFrame,
    filters: dict[str, object],
    grouping_levels: list[list[str]],
    row: pd.Series,
) -> dict[str, object]:
    run_df = apply_filters(merged, filters)
    train_df, test_df = p1.grouped_train_test_split(run_df, test_size=0.2, seed=p1.SEED)
    feature_cols = p1.feature_subset_columns(run_df, "all_universal")
    train_norm, test_norm = p1.normalize_with_fallback(train_df, test_df, feature_cols, grouping_levels)
    y_train = train_norm["binary_label"].astype(int)
    y_test = test_norm["binary_label"].astype(int)
    selected_cols, _ = p1.select_top_k(train_norm, y_train, feature_cols, row["top_k"])
    x_train = train_norm[selected_cols]
    x_test = test_norm[selected_cols]
    estimator = get_model_estimator(row["model_family"], row["model_variant"])
    estimator.fit(x_train, y_train)
    pred = estimator.predict(x_test)
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(x_test)[:, 1]
    else:
        scores = estimator.decision_function(x_test)
    perm = permutation_importance(
        estimator,
        x_test,
        y_test,
        n_repeats=10,
        random_state=p1.SEED,
        scoring="accuracy",
        n_jobs=1,
    )
    perm_df = pd.DataFrame(
        {
            "feature_name": selected_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return {
        "run_df": run_df,
        "train_df": train_df,
        "test_df": test_df,
        "train_norm": train_norm,
        "test_norm": test_norm,
        "y_train": y_train,
        "y_test": y_test,
        "pred": pred,
        "scores": scores,
        "selected_cols": selected_cols,
        "row": row,
        "accuracy": float(accuracy_score(y_test, pred)),
        "auroc": float(roc_auc_score(y_test, scores)),
        "permutation": perm_df,
    }


def top_features_from_run(run_state: dict[str, object], top_n: int = 16) -> list[str]:
    return run_state["permutation"]["feature_name"].head(top_n).tolist()


def make_burden_frame(run_state: dict[str, object], feature_names: list[str]) -> pd.DataFrame:
    train_x = run_state["train_norm"][feature_names].copy()
    test_x = run_state["test_norm"][feature_names].copy()
    medians = train_x.median()
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)

    train_labels = run_state["train_norm"]["binary_label"].astype(int)
    train_ad = train_x.loc[train_labels == 1]
    train_hc = train_x.loc[train_labels == 0]
    direction = np.sign(train_ad.mean() - train_hc.mean()).replace(0, 1.0)

    oriented_train = train_x.mul(direction, axis=1)
    oriented_test = test_x.mul(direction, axis=1)

    lower = oriented_train.quantile(0.05)
    upper = oriented_train.quantile(0.95)
    scale = (upper - lower).replace(0, np.nan)
    scaled = oriented_test.sub(lower, axis=1).div(scale, axis=1).clip(0, 1).fillna(0.5) * 100.0

    meta = run_state["test_df"][
        ["sample_id", "group_id", "dataset_name", "language", "task_type", "diagnosis_mapped", "binary_label"]
    ].copy()
    meta["diagnosis"] = meta["binary_label"].map({0: "HC", 1: "AD"})
    meta["pred"] = run_state["pred"]
    meta["score"] = run_state["scores"]
    meta["mean_burden"] = scaled.mean(axis=1)

    long = scaled.copy()
    long["sample_id"] = meta["sample_id"].values
    long = long.melt(id_vars="sample_id", var_name="feature_name", value_name="burden_0_100")
    long = long.merge(meta, on="sample_id", how="left")
    long["feature_group"] = long["feature_name"].str.split("_").str[0]
    return long


def choose_pair(burden_long: pd.DataFrame) -> pd.DataFrame:
    meta = (
        burden_long[
            ["sample_id", "group_id", "dataset_name", "diagnosis", "mean_burden", "score", "pred", "binary_label"]
        ]
        .drop_duplicates()
        .copy()
    )
    correct = meta[meta["pred"].astype(int) == meta["binary_label"].astype(int)].copy()
    ad = correct[correct["diagnosis"] == "AD"].sort_values(["mean_burden", "score"], ascending=[False, False]).head(1)
    hc = correct[correct["diagnosis"] == "HC"].sort_values(["mean_burden", "score"], ascending=[True, True]).head(1)
    return pd.concat([ad, hc], ignore_index=True)


def radar_angles(n: int) -> list[float]:
    values = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    return values + values[:1]


def plot_overlay_grid(
    burden_by_language: dict[str, pd.DataFrame],
    feature_order_by_language: dict[str, list[str]],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, subplot_kw={"polar": True}, figsize=(15, 12))
    axes = axes.flatten()

    for ax, code in zip(axes, ["en", "zh", "el", "es"]):
        df = burden_by_language[code]
        features = feature_order_by_language[code]
        angles = radar_angles(len(features))

        for diagnosis in ["AD", "HC"]:
            subset = df[df["diagnosis"] == diagnosis]
            for sample_id, sample_df in subset.groupby("sample_id"):
                sample_df = sample_df.set_index("feature_name").reindex(features).reset_index()
                vals = sample_df["burden_0_100"].tolist()
                vals += vals[:1]
                ax.plot(angles, vals, color=PALETTE[diagnosis], alpha=0.07, linewidth=1.0)

            mean_df = (
                subset.groupby("feature_name", as_index=False)["burden_0_100"].mean()
                .set_index("feature_name")
                .reindex(features)
                .reset_index()
            )
            mean_vals = mean_df["burden_0_100"].tolist()
            mean_vals += mean_vals[:1]
            ax.plot(angles, mean_vals, color=PALETTE[diagnosis], linewidth=2.6, label=f"{diagnosis} mean")
            ax.fill(angles, mean_vals, color=PALETTE[diagnosis], alpha=0.10)

        ax.set_title(LANGUAGE_SPECS[code]["label"], fontsize=13, pad=20)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([feat.replace("_", "\n") for feat in features], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)
        ax.grid(alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=16, y=0.98)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.95))
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pair_grid(
    burden_by_language: dict[str, pd.DataFrame],
    feature_order_by_language: dict[str, list[str]],
    pair_meta: dict[str, pd.DataFrame],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, subplot_kw={"polar": True}, figsize=(15, 12))
    axes = axes.flatten()

    for ax, code in zip(axes, ["en", "zh", "el", "es"]):
        df = burden_by_language[code]
        features = feature_order_by_language[code]
        angles = radar_angles(len(features))
        pair = pair_meta[code]

        for diagnosis in ["AD", "HC"]:
            sample_id = pair[pair["diagnosis"] == diagnosis]["sample_id"].iloc[0]
            sample_df = (
                df[df["sample_id"] == sample_id]
                .set_index("feature_name")
                .reindex(features)
                .reset_index()
            )
            vals = sample_df["burden_0_100"].tolist()
            vals += vals[:1]
            label_row = pair[pair["diagnosis"] == diagnosis].iloc[0]
            label = f"{diagnosis} | {label_row['dataset_name']} | {label_row['group_id']}"
            ax.plot(angles, vals, color=PALETTE[diagnosis], linewidth=2.8, label=label)
            ax.fill(angles, vals, color=PALETTE[diagnosis], alpha=0.12)

        ax.set_title(LANGUAGE_SPECS[code]["label"], fontsize=13, pad=20)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([feat.replace("_", "\n") for feat in features], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)
        ax.grid(alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=16, y=0.98)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.95))
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_skew_summary(
    fixed_burden: dict[str, pd.DataFrame],
    fixed_features: list[str],
    output_path: Path,
) -> None:
    lines = []
    lines.append("Saliency Radar Interpretation")
    lines.append("============================")
    lines.append("Method")
    lines.append("------")
    lines.append("These radar maps use AD-oriented feature burden scores on a 0-100 scale.")
    lines.append("For each feature, the direction is set from the training split so that higher values are more AD-like.")
    lines.append("The fixed-feature plots use the same top 16 pooled all-universal features in every language.")
    lines.append("The local-feature plots use the top 16 features from the best all-universal model within each language.")
    lines.append("")
    lines.append("Fixed pooled feature set")
    lines.append("-----------------------")
    lines.append(", ".join(fixed_features))
    lines.append("")

    for code in ["en", "zh", "el", "es"]:
        df = fixed_burden[code]
        summary = (
            df.groupby(["diagnosis", "feature_group"], as_index=False)["burden_0_100"]
            .mean()
            .pivot(index="feature_group", columns="diagnosis", values="burden_0_100")
            .fillna(0.0)
        )
        summary["gap_ad_minus_hc"] = summary.get("AD", 0.0) - summary.get("HC", 0.0)
        top_groups = summary.sort_values("gap_ad_minus_hc", ascending=False).head(3)
        lines.append(LANGUAGE_SPECS[code]["label"])
        lines.append("-" * len(LANGUAGE_SPECS[code]["label"]))
        for group, row in top_groups.iterrows():
            lines.append(
                f"{group}: AD-HC gap {row['gap_ad_minus_hc']:.1f} "
                f"(AD mean {row.get('AD', 0.0):.1f}, HC mean {row.get('HC', 0.0):.1f})"
            )
        lines.append("")

    lines.append("Interpretation")
    lines.append("--------------")
    lines.append("The fixed pooled feature set does show language-specific skew rather than one universal saliency shape.")
    lines.append("English and Spanish lean more on lexical, syntactic, and discourse burden.")
    lines.append("Chinese shows stronger pause/rate and some acoustic burden on the pooled feature set.")
    lines.append("Greek remains the most acoustically skewed profile even when the feature set is fixed across languages.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    merged = load_merged()

    pooled_row = best_row_for_run("benchmark_wide_rich", subset="all_universal")
    pooled_state = reconstruct_run(
        merged,
        {},
        [["language", "task_type", "dataset_name"], ["language", "task_type"], ["language"], []],
        pooled_row,
    )
    fixed_features = top_features_from_run(pooled_state, top_n=16)

    fixed_burden = {}
    local_burden = {}
    fixed_pair_meta = {}
    local_pair_meta = {}
    local_feature_orders = {}
    fixed_feature_orders = {}
    pair_rows = []

    for code, meta in LANGUAGE_SPECS.items():
        row = best_row_for_run(f"language_{code}_all_rich", subset="all_universal")
        state = reconstruct_run(merged, meta["filters"], meta["grouping_levels"], row)

        local_features = top_features_from_run(state, top_n=16)
        fixed_df = make_burden_frame(state, fixed_features)
        local_df = make_burden_frame(state, local_features)

        fixed_pair = choose_pair(fixed_df)
        local_pair = choose_pair(local_df)

        fixed_burden[code] = fixed_df
        local_burden[code] = local_df
        fixed_pair_meta[code] = fixed_pair
        local_pair_meta[code] = local_pair
        fixed_feature_orders[code] = fixed_features
        local_feature_orders[code] = local_features

        for mode, pair in [("fixed_global16", fixed_pair), ("local_language16", local_pair)]:
            for _, row_meta in pair.iterrows():
                pair_rows.append(
                    {
                        "language": meta["label"],
                        "mode": mode,
                        "diagnosis": row_meta["diagnosis"],
                        "sample_id": row_meta["sample_id"],
                        "group_id": row_meta["group_id"],
                        "dataset_name": row_meta["dataset_name"],
                        "mean_burden": row_meta["mean_burden"],
                        "score": row_meta["score"],
                    }
                )

    plot_overlay_grid(
        fixed_burden,
        fixed_feature_orders,
        REPORT_ROOT / "06_saliency_overlay_fixed_global16_by_language.png",
        "Fixed pooled top-16 Phase 1 features: transparent AD/HC overlays by language",
    )
    plot_overlay_grid(
        local_burden,
        local_feature_orders,
        REPORT_ROOT / "07_saliency_overlay_local16_by_language.png",
        "Language-specific top-16 Phase 1 features: transparent AD/HC overlays by language",
    )
    plot_pair_grid(
        fixed_burden,
        fixed_feature_orders,
        fixed_pair_meta,
        REPORT_ROOT / "08_saliency_pair_fixed_global16_by_language.png",
        "Fixed pooled top-16 Phase 1 features: one high-burden AD vs one low-burden HC per language",
    )
    plot_pair_grid(
        local_burden,
        local_feature_orders,
        local_pair_meta,
        REPORT_ROOT / "09_saliency_pair_local16_by_language.png",
        "Language-specific top-16 Phase 1 features: one high-burden AD vs one low-burden HC per language",
    )

    pd.DataFrame(pair_rows).to_csv(REPORT_ROOT / "saliency_selected_examples.csv", index=False)
    build_skew_summary(fixed_burden, fixed_features, REPORT_ROOT / "10_saliency_language_skew_summary.txt")


if __name__ == "__main__":
    main()
