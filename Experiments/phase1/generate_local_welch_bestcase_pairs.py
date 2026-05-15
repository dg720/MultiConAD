from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "Experiments"
REPORT_ROOT = PROJECT_ROOT / "reports" / "13.05"
SELECTED_ROOT = REPORT_ROOT / "selected"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

experiments_pkg = types.ModuleType("experiments")
experiments_pkg.__path__ = [str(EXPERIMENTS_ROOT.resolve())]
sys.modules.setdefault("experiments", experiments_pkg)

import experiments.phase1.generate_language_saliency_maps as sal
import experiments.phase1.generate_welch_saliency_maps as welch
import experiments.phase1.run_rich_sweep as p1


LANGUAGE_ORDER = ["en", "zh", "el", "es"]


def choose_bestcase_pair(hc_oriented_burden: pd.DataFrame) -> pd.DataFrame:
    meta = (
        hc_oriented_burden[
            ["sample_id", "group_id", "dataset_name", "diagnosis", "mean_burden", "score", "pred", "binary_label"]
        ]
        .drop_duplicates()
        .copy()
    )
    correct = meta[meta["pred"].astype(int) == meta["binary_label"].astype(int)].copy()
    ad = correct[correct["diagnosis"] == "AD"].sort_values(["mean_burden", "score"], ascending=[True, False]).head(1)
    hc = correct[correct["diagnosis"] == "HC"].sort_values(["mean_burden", "score"], ascending=[False, True]).head(1)
    return pd.concat([ad, hc], ignore_index=True)


def smoothed_percentile_scores(train_values: pd.Series, test_values: pd.Series) -> np.ndarray:
    sorted_train = np.sort(train_values.astype(float).to_numpy())
    n = len(sorted_train)
    left = np.searchsorted(sorted_train, test_values.astype(float).to_numpy(), side="left")
    right = np.searchsorted(sorted_train, test_values.astype(float).to_numpy(), side="right")
    mid_rank = ((left + right) / 2.0) + 1.0
    return 100.0 * mid_rank / (n + 1.0)


def make_percentile_hc_burden_frame(run_state: dict[str, object], feature_names: list[str]) -> pd.DataFrame:
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

    hc_scores = pd.DataFrame(index=oriented_test.index, columns=feature_names, dtype=float)
    for feature in feature_names:
        ad_percentile = smoothed_percentile_scores(oriented_train[feature], oriented_test[feature])
        hc_scores[feature] = 100.0 - ad_percentile

    meta = run_state["test_df"][
        ["sample_id", "group_id", "dataset_name", "language", "task_type", "diagnosis_mapped", "binary_label"]
    ].copy()
    meta["diagnosis"] = meta["binary_label"].map({0: "HC", 1: "AD"})
    meta["pred"] = run_state["pred"]
    meta["score"] = run_state["scores"]
    meta["mean_burden"] = hc_scores.mean(axis=1)

    long = hc_scores.copy()
    long["sample_id"] = meta["sample_id"].values
    long = long.melt(id_vars="sample_id", var_name="feature_name", value_name="burden_0_100")
    long = long.merge(meta, on="sample_id", how="left")
    long["feature_group"] = long["feature_name"].str.split("_").str[0]
    return long


def build_note(pair_df: pd.DataFrame, feature_orders: dict[str, list[str]], output_path: Path) -> None:
    lines = []
    lines.append("Language-Specific Welch Best-Case Pairs")
    lines.append("======================================")
    lines.append("These examples use an HC-oriented local Welch saliency scale built from smoothed percentile ranks on the training split.")
    lines.append("Selection rule per language:")
    lines.append("- AD: correctly classified case with the lowest mean HC-oriented burden")
    lines.append("- HC: correctly classified case with the highest mean HC-oriented burden")
    lines.append("")
    lines.append("This is intended to show typical best-case near-empty AD and near-full HC profiles under the language-specific feature set.")
    lines.append("Unlike the canonical overlay plots, this case-level view does not use the 5th-95th clipped burden transform; it uses percentile-style scoring to avoid hard floor effects.")
    lines.append("")
    for code in LANGUAGE_ORDER:
        label = sal.LANGUAGE_SPECS[code]["label"]
        lines.append(label)
        lines.append("-" * len(label))
        subset = pair_df[pair_df["language_code"] == code].copy()
        for diagnosis in ["AD", "HC"]:
            row = subset[subset["diagnosis"] == diagnosis].iloc[0]
            lines.append(
                f"{diagnosis}: dataset={row['dataset_name']} | group={row['group_id']} | sample_id={row['sample_id']} | mean_hc_oriented_burden={row['mean_burden']:.1f}"
            )
        lines.append("Features: " + ", ".join(feature_orders[code]))
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    SELECTED_ROOT.mkdir(parents=True, exist_ok=True)
    merged = sal.load_merged()

    burden_by_language = {}
    pair_meta = {}
    feature_orders = {}
    pair_rows = []

    for code in LANGUAGE_ORDER:
        meta = sal.LANGUAGE_SPECS[code]
        row = sal.best_row_for_run(f"language_{code}_all_rich", subset="all_universal")
        state = sal.reconstruct_run(merged, meta["filters"], meta["grouping_levels"], row)
        feature_cols = p1.feature_subset_columns(state["run_df"], "all_universal")
        welch_df = welch.welch_significance_frame(state["train_df"], feature_cols, meta["label"])
        features = welch.top_welch_features(welch_df, meta["label"], top_n=16)

        hc_oriented = make_percentile_hc_burden_frame(state, features)
        pair = choose_bestcase_pair(hc_oriented)

        burden_by_language[code] = hc_oriented
        pair_meta[code] = pair
        feature_orders[code] = features

        for _, pair_row in pair.iterrows():
            pair_rows.append(
                {
                    "language_code": code,
                    "language": meta["label"],
                    "diagnosis": pair_row["diagnosis"],
                    "sample_id": pair_row["sample_id"],
                    "group_id": pair_row["group_id"],
                    "dataset_name": pair_row["dataset_name"],
                    "mean_burden": pair_row["mean_burden"],
                    "score": pair_row["score"],
                }
            )

    sal.plot_pair_grid(
        burden_by_language,
        feature_orders,
        pair_meta,
        SELECTED_ROOT / "32_saliency_pair_local_welch_hc_oriented_bestcase_by_language.png",
        "Language-specific Welch+Bonferroni+|d| Phase 1 features: best-case near-empty AD vs near-full HC by language",
    )

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(SELECTED_ROOT / "33_local_welch_bestcase_pairs.csv", index=False)
    build_note(pair_df, feature_orders, SELECTED_ROOT / "34_local_welch_bestcase_pairs_note.txt")


if __name__ == "__main__":
    main()
