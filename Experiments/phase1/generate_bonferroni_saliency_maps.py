from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

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


LANGUAGE_ORDER = ["en", "zh", "el", "es"]


def load_significance() -> pd.DataFrame:
    path = TABLE_REPORT_ROOT / "all_universal_language_significance.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing significance table: {path}")
    return pd.read_csv(path)


def top_bonferroni_features(significance_df: pd.DataFrame, language_label: str, top_n: int = 16) -> list[str]:
    subset = significance_df[
        (significance_df["language"] == language_label) & (significance_df["significant_bonferroni"].astype(bool))
    ].sort_values(["p_value", "h_statistic"], ascending=[True, False])
    if subset.empty:
        subset = significance_df[significance_df["language"] == language_label].sort_values(
            ["p_value", "h_statistic"], ascending=[True, False]
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
    pooled_std = np.sqrt((ad_vals.var(ddof=1) + hc_vals.var(ddof=1)) / 2.0) if len(ad_vals) > 1 and len(hc_vals) > 1 else np.nan
    effect = (ad_vals.mean() - hc_vals.mean()) / pooled_std if pooled_std and not np.isnan(pooled_std) and pooled_std != 0 else np.nan
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


def build_summary(comparison_df: pd.DataFrame, feature_manifest: pd.DataFrame, output_path: Path) -> None:
    lines = []
    lines.append("Bonferroni Saliency vs Permutation Saliency")
    lines.append("==========================================")
    lines.append("Method")
    lines.append("------")
    lines.append("Both saliency variants use the same Phase 1 language-specific best `all_universal` model per language.")
    lines.append("The only change is the feature list used to build the AD-oriented burden radar plots.")
    lines.append("- Permutation map: top local model features from held-out permutation importance.")
    lines.append("- Bonferroni map: top language-specific Bonferroni-significant features ranked by p-value.")
    lines.append("")

    for code in LANGUAGE_ORDER:
        label = sal.LANGUAGE_SPECS[code]["label"]
        subset = comparison_df[comparison_df["language_code"] == code].copy()
        perm = subset[subset["feature_source"] == "permutation"].iloc[0]
        bonf = subset[subset["feature_source"] == "bonferroni"].iloc[0]
        features = feature_manifest[feature_manifest["language_code"] == code]
        bonf_features = features[features["feature_source"] == "bonferroni"]["feature_name"].tolist()
        perm_features = features[features["feature_source"] == "permutation"]["feature_name"].tolist()
        shared = [name for name in bonf_features if name in set(perm_features)]

        lines.append(label)
        lines.append("-" * len(label))
        lines.append(
            f"Permutation: n={int(perm['num_features'])} | mean burden gap={perm['gap_ad_minus_hc']:.1f} | Cohen's d={perm['cohens_d']:.2f} | strongest group={perm['top_gap_group']} ({perm['top_gap_value']:.1f})"
        )
        lines.append(
            f"Bonferroni: n={int(bonf['num_features'])} | mean burden gap={bonf['gap_ad_minus_hc']:.1f} | Cohen's d={bonf['cohens_d']:.2f} | strongest group={bonf['top_gap_group']} ({bonf['top_gap_value']:.1f})"
        )
        lines.append(
            f"Overlap with permutation feature set: {len(shared)}/{len(bonf_features)}"
            + (f" | shared: {', '.join(shared[:8])}" if shared else " | shared: none")
        )

        clearer = "Bonferroni" if bonf["gap_abs"] > perm["gap_abs"] else "Permutation"
        if bonf["gap_abs"] == perm["gap_abs"]:
            clearer = "Tie"
        lines.append(f"Clearer by mean-burden separation: {clearer}")

        if clearer == "Bonferroni":
            lines.append("Why it looks clearer:")
            lines.append("Bonferroni features are more consistently AD-vs-HC separated one by one, so the radar shapes tend to open up more cleanly between classes.")
        elif clearer == "Permutation":
            lines.append("Why permutation remains clearer:")
            lines.append("The model-selected features create a larger class burden gap even when some individual features are only weakly significant on their own.")
        else:
            lines.append("Why neither is clearly better:")
            lines.append("The burden-gap summary is effectively tied, so visual preference depends more on whether you want univariate interpretability or model-faithful saliency.")
        lines.append("")

    lines.append("Overall")
    lines.append("-------")
    better = comparison_df.sort_values(["language_code", "feature_source"]).copy()
    winners = []
    for code in LANGUAGE_ORDER:
        lang_rows = better[better["language_code"] == code]
        perm_gap = float(lang_rows[lang_rows["feature_source"] == "permutation"]["gap_abs"].iloc[0])
        bonf_gap = float(lang_rows[lang_rows["feature_source"] == "bonferroni"]["gap_abs"].iloc[0])
        label = sal.LANGUAGE_SPECS[code]["label"]
        if bonf_gap > perm_gap:
            winners.append(f"{label}: Bonferroni")
        elif bonf_gap < perm_gap:
            winners.append(f"{label}: Permutation")
        else:
            winners.append(f"{label}: Tie")
    lines.extend(f"- {winner}" for winner in winners)
    lines.append("- Bonferroni maps are usually easier to justify as biomarker-style plots because every plotted feature passed a corrected univariate test.")
    lines.append("- Permutation maps remain better if the goal is fidelity to what the fitted classifier actually used.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    merged = sal.load_merged()
    significance_df = load_significance()

    bonf_burden = {}
    bonf_pairs = {}
    bonf_feature_orders = {}
    comparison_rows = []
    feature_rows = []

    for code in LANGUAGE_ORDER:
        meta = sal.LANGUAGE_SPECS[code]
        row = sal.best_row_for_run(f"language_{code}_all_rich", subset="all_universal")
        state = sal.reconstruct_run(merged, meta["filters"], meta["grouping_levels"], row)

        perm_features = sal.top_features_from_run(state, top_n=16)
        bonf_features = top_bonferroni_features(significance_df, meta["label"], top_n=16)

        perm_burden = sal.make_burden_frame(state, perm_features)
        bonf_df = sal.make_burden_frame(state, bonf_features)
        bonf_pair = sal.choose_pair(bonf_df)

        bonf_burden[code] = bonf_df
        bonf_pairs[code] = bonf_pair
        bonf_feature_orders[code] = bonf_features

        for source_name, feature_names, burden_df in [
            ("permutation", perm_features, perm_burden),
            ("bonferroni", bonf_features, bonf_df),
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

    sal.plot_overlay_grid(
        bonf_burden,
        bonf_feature_orders,
        REPORT_ROOT / "14_saliency_overlay_bonferroni_by_language.png",
        "Language-specific top Bonferroni Phase 1 features: transparent AD/HC overlays by language",
    )
    sal.plot_pair_grid(
        bonf_burden,
        bonf_feature_orders,
        bonf_pairs,
        REPORT_ROOT / "15_saliency_pair_bonferroni_by_language.png",
        "Language-specific top Bonferroni Phase 1 features: one high-burden AD vs one low-burden HC per language",
    )

    comparison_df = pd.DataFrame(comparison_rows)
    feature_df = pd.DataFrame(feature_rows)
    comparison_df.to_csv(TABLE_REPORT_ROOT / "bonferroni_saliency_vs_permutation_metrics.csv", index=False)
    feature_df.to_csv(TABLE_REPORT_ROOT / "bonferroni_saliency_feature_lists.csv", index=False)
    build_summary(comparison_df, feature_df, REPORT_ROOT / "16_bonferroni_saliency_vs_permutation.txt")


if __name__ == "__main__":
    main()
