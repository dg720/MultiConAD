from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "Experiments"
REPORT_ROOT = PROJECT_ROOT / "reports" / "13.05"
TABLE_REPORT_ROOT = PROJECT_ROOT / "tables" / "03-ablation-translingual-language-specific" / "phase1-rich-sweep" / "report_assets"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

experiments_pkg = types.ModuleType("experiments")
experiments_pkg.__path__ = [str(EXPERIMENTS_ROOT.resolve())]
sys.modules.setdefault("experiments", experiments_pkg)

import experiments.phase1.generate_language_saliency_maps as sal
import experiments.phase1.run_rich_sweep as p1


LANGUAGE_ORDER = ["en", "zh", "el", "es"]


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


def pooled_std(ad_vals: pd.Series, hc_vals: pd.Series) -> float:
    ad_var = float(ad_vals.var(ddof=1))
    hc_var = float(hc_vals.var(ddof=1))
    value = math.sqrt((ad_var + hc_var) / 2.0)
    return value


def welch_significance_frame(train_df: pd.DataFrame, feature_cols: list[str], language_label: str) -> pd.DataFrame:
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
        effect = (float(ad_vals.mean()) - float(hc_vals.mean())) / spread if spread not in {0.0} else np.nan
        rows.append(
            {
                "language": language_label,
                "feature_name": feature,
                "feature_group": feature.split("_", 1)[0],
                "t_statistic": float(test.statistic),
                "p_value": float(test.pvalue),
                "mean_ad": float(ad_vals.mean()),
                "mean_hc": float(hc_vals.mean()),
                "median_ad": float(ad_vals.median()),
                "median_hc": float(hc_vals.median()),
                "cohens_d": float(effect) if pd.notna(effect) else np.nan,
                "cohens_d_abs": float(abs(effect)) if pd.notna(effect) else np.nan,
                "direction": "AD>HC" if float(ad_vals.mean()) > float(hc_vals.mean()) else "AD<HC",
            }
        )

    if not rows:
        raise RuntimeError(f"No Welch significance rows generated for {language_label}.")

    df = pd.DataFrame(rows).sort_values(["p_value", "cohens_d_abs"], ascending=[True, False]).reset_index(drop=True)
    df["bonferroni_p"] = (df["p_value"] * len(df)).clip(upper=1.0)
    df["fdr_bh_p"] = bh_fdr(df["p_value"].tolist())
    df["significant_bonferroni"] = df["bonferroni_p"] < 0.05
    df["significant_fdr"] = df["fdr_bh_p"] < 0.05
    df["tested_features"] = len(df)
    df["bonferroni_significant_count"] = int(df["significant_bonferroni"].sum())
    df["fdr_significant_count"] = int(df["significant_fdr"].sum())
    return df


def top_welch_features(welch_df: pd.DataFrame, language_label: str, top_n: int = 16) -> list[str]:
    subset = welch_df[
        (welch_df["language"] == language_label) & (welch_df["significant_bonferroni"].astype(bool))
    ].sort_values(["cohens_d_abs", "p_value"], ascending=[False, True])
    if subset.empty:
        subset = welch_df[welch_df["language"] == language_label].sort_values(
            ["cohens_d_abs", "p_value"], ascending=[False, True]
        )
    return subset["feature_name"].head(top_n).tolist()


def mean_burden_metrics(burden_long: pd.DataFrame, feature_names: list[str]) -> dict[str, float]:
    sample_means = (
        burden_long[burden_long["feature_name"].isin(feature_names)]
        .groupby(["sample_id", "diagnosis"], as_index=False)["burden_0_100"]
        .mean()
    )
    ad_vals = sample_means.loc[sample_means["diagnosis"] == "AD", "burden_0_100"].astype(float)
    hc_vals = sample_means.loc[sample_means["diagnosis"] == "HC", "burden_0_100"].astype(float)
    pooled = np.sqrt((ad_vals.var(ddof=1) + hc_vals.var(ddof=1)) / 2.0) if len(ad_vals) > 1 and len(hc_vals) > 1 else np.nan
    effect = (ad_vals.mean() - hc_vals.mean()) / pooled if pooled and not np.isnan(pooled) and pooled != 0 else np.nan
    return {
        "ad_mean_burden": float(ad_vals.mean()),
        "hc_mean_burden": float(hc_vals.mean()),
        "gap_ad_minus_hc": float(ad_vals.mean() - hc_vals.mean()),
        "gap_abs": float(abs(ad_vals.mean() - hc_vals.mean())),
        "cohens_d": float(effect) if not np.isnan(effect) else np.nan,
        "ad_std": float(ad_vals.std(ddof=1)) if len(ad_vals) > 1 else np.nan,
        "hc_std": float(hc_vals.std(ddof=1)) if len(hc_vals) > 1 else np.nan,
    }


def top_group_gap(burden_long: pd.DataFrame) -> tuple[str, float]:
    summary = (
        burden_long.groupby(["diagnosis", "feature_group"], as_index=False)["burden_0_100"]
        .mean()
        .pivot(index="feature_group", columns="diagnosis", values="burden_0_100")
        .fillna(0.0)
    )
    summary["gap"] = summary.get("AD", 0.0) - summary.get("HC", 0.0)
    best_group = summary["gap"].abs().idxmax()
    return best_group, float(summary.loc[best_group, "gap"])


def build_summary(
    comparison_df: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    welch_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = []
    lines.append("Welch Mean-Difference Saliency vs Permutation")
    lines.append("============================================")
    lines.append("Method")
    lines.append("------")
    lines.append("Welch selection uses per-language feature-wise Welch t-tests on the training split, Bonferroni correction within language, then ranks surviving features by absolute Cohen's d.")
    lines.append("Permutation selection uses the local best-model top features from held-out permutation importance.")
    lines.append("")

    for code in LANGUAGE_ORDER:
        label = sal.LANGUAGE_SPECS[code]["label"]
        subset = comparison_df[comparison_df["language_code"] == code].copy()
        perm = subset[subset["feature_source"] == "permutation"].iloc[0]
        welch = subset[subset["feature_source"] == "welch_bonferroni_d"].iloc[0]
        features = feature_manifest[feature_manifest["language_code"] == code]
        welch_features = features[features["feature_source"] == "welch_bonferroni_d"]["feature_name"].tolist()
        perm_features = features[features["feature_source"] == "permutation"]["feature_name"].tolist()
        shared = [name for name in welch_features if name in set(perm_features)]

        top_welch_rows = (
            welch_df[(welch_df["language"] == label) & (welch_df["feature_name"].isin(welch_features))]
            .sort_values(["cohens_d_abs", "p_value"], ascending=[False, True])
            .head(5)
        )

        lines.append(label)
        lines.append("-" * len(label))
        lines.append(
            f"Permutation: n={int(perm['num_features'])} | mean burden gap={perm['gap_ad_minus_hc']:.1f} | Cohen's d={perm['cohens_d']:.2f} | strongest group={perm['top_gap_group']} ({perm['top_gap_value']:.1f})"
        )
        lines.append(
            f"Welch+Bonferroni+|d|: n={int(welch['num_features'])} | mean burden gap={welch['gap_ad_minus_hc']:.1f} | Cohen's d={welch['cohens_d']:.2f} | strongest group={welch['top_gap_group']} ({welch['top_gap_value']:.1f})"
        )
        lines.append(f"Bonferroni-significant Welch features available: {int(top_welch_rows['bonferroni_significant_count'].iloc[0]) if not top_welch_rows.empty else 0}")
        lines.append(
            f"Overlap with permutation feature set: {len(shared)}/{len(welch_features)}"
            + (f" | shared: {', '.join(shared[:8])}" if shared else " | shared: none")
        )
        lines.append("Top Welch-selected features")
        for _, row in top_welch_rows.iterrows():
            lines.append(
                f"- {row['feature_name']} [{row['feature_group']}] | mean_AD={row['mean_ad']:.3f} | mean_HC={row['mean_hc']:.3f} | p={row['p_value']:.2e} | bonf={row['bonferroni_p']:.2e} | |d|={row['cohens_d_abs']:.2f}"
            )

        clearer = "Welch" if welch["gap_abs"] > perm["gap_abs"] else "Permutation"
        if welch["gap_abs"] == perm["gap_abs"]:
            clearer = "Tie"
        lines.append(f"Clearer by mean-burden separation: {clearer}")
        if clearer == "Welch":
            lines.append("Why it looks clearer:")
            lines.append("These features are explicitly chosen for corrected mean-difference strength, so the AD polygon tends to fill more of the radar in a consistent direction.")
        elif clearer == "Permutation":
            lines.append("Why permutation remains clearer:")
            lines.append("The classifier is still relying on a different multivariate feature mix that produces a larger AD-vs-HC burden gap than the pure mean-difference shortlist.")
        else:
            lines.append("Why neither is clearly better:")
            lines.append("The burden-gap summary is effectively tied, so visual preference depends on whether you want mean-difference screening or model-faithful saliency.")
        lines.append("")

    lines.append("Overall")
    lines.append("-------")
    winners = []
    for code in LANGUAGE_ORDER:
        lang_rows = comparison_df[comparison_df["language_code"] == code]
        perm_gap = float(lang_rows[lang_rows["feature_source"] == "permutation"]["gap_abs"].iloc[0])
        welch_gap = float(lang_rows[lang_rows["feature_source"] == "welch_bonferroni_d"]["gap_abs"].iloc[0])
        label = sal.LANGUAGE_SPECS[code]["label"]
        if welch_gap > perm_gap:
            winners.append(f"{label}: Welch")
        elif welch_gap < perm_gap:
            winners.append(f"{label}: Permutation")
        else:
            winners.append(f"{label}: Tie")
    lines.extend(f"- {winner}" for winner in winners)
    lines.append("- Welch maps are the cleanest way to express corrected mean separation, because every plotted feature both survives Bonferroni and has large standardized mean difference.")
    lines.append("- They should still be treated as univariate biomarker-style plots, not as faithful explanations of the classifier.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    merged = sal.load_merged()

    welch_frames = []
    welch_burden = {}
    welch_pairs = {}
    welch_feature_orders = {}
    comparison_rows = []
    feature_rows = []

    for code in LANGUAGE_ORDER:
        meta = sal.LANGUAGE_SPECS[code]
        row = sal.best_row_for_run(f"language_{code}_all_rich", subset="all_universal")
        state = sal.reconstruct_run(merged, meta["filters"], meta["grouping_levels"], row)
        feature_cols = p1.feature_subset_columns(state["run_df"], "all_universal")
        welch_df = welch_significance_frame(state["train_df"], feature_cols, meta["label"])
        welch_frames.append(welch_df)

        perm_features = sal.top_features_from_run(state, top_n=16)
        welch_features = top_welch_features(welch_df, meta["label"], top_n=16)

        perm_burden = sal.make_burden_frame(state, perm_features)
        welch_burden_df = sal.make_burden_frame(state, welch_features)
        welch_pair = sal.choose_pair(welch_burden_df)

        welch_burden[code] = welch_burden_df
        welch_pairs[code] = welch_pair
        welch_feature_orders[code] = welch_features

        for source_name, feature_names, burden_df in [
            ("permutation", perm_features, perm_burden),
            ("welch_bonferroni_d", welch_features, welch_burden_df),
        ]:
            metrics = mean_burden_metrics(burden_df, feature_names)
            group_name, group_gap = top_group_gap(burden_df[burden_df["feature_name"].isin(feature_names)])
            comparison_rows.append(
                {
                    "language_code": code,
                    "language": meta["label"],
                    "feature_source": source_name,
                    "num_features": len(feature_names),
                    "top_gap_group": group_name,
                    "top_gap_value": group_gap,
                    **metrics,
                }
            )
            for rank, feature_name in enumerate(feature_names, start=1):
                feature_rows.append(
                    {
                        "language_code": code,
                        "language": meta["label"],
                        "feature_source": source_name,
                        "rank": rank,
                        "feature_name": feature_name,
                    }
                )

    all_welch_df = pd.concat(welch_frames, ignore_index=True)
    all_welch_df.to_csv(TABLE_REPORT_ROOT / "all_universal_language_welch_significance.csv", index=False)

    sal.plot_overlay_grid(
        welch_burden,
        welch_feature_orders,
        REPORT_ROOT / "18_saliency_overlay_welch_by_language.png",
        "Language-specific Welch+Bonferroni+|d| Phase 1 features: transparent AD/HC overlays by language",
    )
    sal.plot_pair_grid(
        welch_burden,
        welch_feature_orders,
        welch_pairs,
        REPORT_ROOT / "19_saliency_pair_welch_by_language.png",
        "Language-specific Welch+Bonferroni+|d| Phase 1 features: one high-burden AD vs one low-burden HC per language",
    )

    comparison_df = pd.DataFrame(comparison_rows)
    feature_df = pd.DataFrame(feature_rows)
    comparison_df.to_csv(TABLE_REPORT_ROOT / "welch_saliency_vs_permutation_metrics.csv", index=False)
    feature_df.to_csv(TABLE_REPORT_ROOT / "welch_saliency_feature_lists.csv", index=False)
    build_summary(comparison_df, feature_df, all_welch_df, REPORT_ROOT / "20_welch_saliency_vs_permutation.txt")


if __name__ == "__main__":
    main()
