from __future__ import annotations

import sys
import types
from pathlib import Path

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
import experiments.phase1.generate_welch_saliency_maps as welch
import experiments.phase1.run_rich_sweep as p1


LANGUAGE_ORDER = ["en", "zh", "el", "es"]
POOLED_GROUPING = [["language", "task_type", "dataset_name"], ["language", "task_type"], ["language"], []]
MIN_LANGUAGE_AVAILABILITY = 1.0


def universally_available_features(df: pd.DataFrame, feature_cols: list[str], min_share: float = MIN_LANGUAGE_AVAILABILITY) -> list[str]:
    keep = []
    for feature in feature_cols:
        shares = []
        for code in LANGUAGE_ORDER:
            sub = df[df["language"] == code]
            share = pd.to_numeric(sub[feature], errors="coerce").notna().mean()
            shares.append(float(share))
        if min(shares) >= min_share:
            keep.append(feature)
    return keep


def build_summary(
    pooled_features: list[str],
    comparison_df: pd.DataFrame,
    universal_feature_count: int,
    output_path: Path,
) -> None:
    lines = []
    lines.append("Pooled Welch Universal Features vs Language-Specific Welch Features")
    lines.append("===============================================================")
    lines.append("Method")
    lines.append("------")
    lines.append("The fixed universal set is built from a pooled multilingual Welch t-test on the benchmark-wide training split.")
    lines.append(f"Candidate pooled features are first restricted to the universally available Phase 1 subset with per-language availability >= {MIN_LANGUAGE_AVAILABILITY:.2f}.")
    lines.append("Features survive Bonferroni correction in the pooled training data and are ranked by absolute Cohen's d.")
    lines.append("The same fixed feature list is then projected into each language to test whether AD/HC separation weakens without local feature adaptation.")
    lines.append("")
    lines.append(f"Universally available candidate features: {universal_feature_count}")
    lines.append("")
    lines.append("Fixed pooled Welch feature set")
    lines.append("-----------------------------")
    lines.append(", ".join(pooled_features))
    lines.append("")

    lines.append("Per-language separation")
    lines.append("-----------------------")
    worse_languages = []
    for code in LANGUAGE_ORDER:
        label = sal.LANGUAGE_SPECS[code]["label"]
        subset = comparison_df[comparison_df["language_code"] == code]
        pooled_row = subset[subset["feature_source"] == "pooled_welch_fixed"].iloc[0]
        local_row = subset[subset["feature_source"] == "local_welch"].iloc[0]
        delta = float(local_row["gap_abs"] - pooled_row["gap_abs"])
        lines.append(
            f"{label}: pooled gap={pooled_row['gap_ad_minus_hc']:.1f} (d={pooled_row['cohens_d']:.2f}) | "
            f"local gap={local_row['gap_ad_minus_hc']:.1f} (d={local_row['cohens_d']:.2f}) | "
            f"delta local-pooled={delta:.1f}"
        )
        if delta > 0:
            worse_languages.append(label)
    lines.append("")

    lines.append("Interpretation")
    lines.append("--------------")
    if worse_languages:
        lines.append(
            "Languages that separate worse under the pooled universal Welch feature set and therefore appear to benefit from language-specific features: "
            + ", ".join(worse_languages)
        )
    else:
        lines.append("No language showed worse separation under the pooled universal Welch feature set.")
    lines.append("This is expected if some languages need feature axes that are locally diagnostic but not strong enough to survive a pooled multilingual significance screen.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    merged = sal.load_merged()

    pooled_row = sal.best_row_for_run("benchmark_wide_rich", subset="all_universal")
    pooled_state = sal.reconstruct_run(merged, {}, POOLED_GROUPING, pooled_row)
    pooled_feature_cols = p1.feature_subset_columns(pooled_state["run_df"], "all_universal")
    pooled_feature_cols = universally_available_features(pooled_state["run_df"], pooled_feature_cols, min_share=MIN_LANGUAGE_AVAILABILITY)
    pooled_welch_df = welch.welch_significance_frame(pooled_state["train_df"], pooled_feature_cols, "Pooled")
    pooled_features = welch.top_welch_features(pooled_welch_df, "Pooled", top_n=16)

    fixed_burden = {}
    fixed_feature_orders = {}
    comparison_rows = []

    for code in LANGUAGE_ORDER:
        meta = sal.LANGUAGE_SPECS[code]
        row = sal.best_row_for_run(f"language_{code}_all_rich", subset="all_universal")
        state = sal.reconstruct_run(merged, meta["filters"], meta["grouping_levels"], row)
        feature_cols = p1.feature_subset_columns(state["run_df"], "all_universal")
        local_welch_df = welch.welch_significance_frame(state["train_df"], feature_cols, meta["label"])
        local_features = welch.top_welch_features(local_welch_df, meta["label"], top_n=16)

        pooled_burden = sal.make_burden_frame(state, pooled_features)
        local_burden = sal.make_burden_frame(state, local_features)

        fixed_burden[code] = pooled_burden
        fixed_feature_orders[code] = pooled_features

        for source_name, feature_names, burden_df in [
            ("pooled_welch_fixed", pooled_features, pooled_burden),
            ("local_welch", local_features, local_burden),
        ]:
            metrics = welch.mean_burden_metrics(burden_df, feature_names)
            top_group, top_gap = welch.top_group_gap(burden_df[burden_df["feature_name"].isin(feature_names)])
            comparison_rows.append(
                {
                    "language_code": code,
                    "language": meta["label"],
                    "feature_source": source_name,
                    "num_features": len(feature_names),
                    "top_gap_group": top_group,
                    "top_gap_value": top_gap,
                    **metrics,
                }
            )

    sal.plot_overlay_grid(
        fixed_burden,
        fixed_feature_orders,
        REPORT_ROOT / "21_saliency_overlay_pooled_welch_by_language.png",
        "Fixed pooled Welch+Bonferroni+|d| Phase 1 features: transparent AD/HC overlays by language",
    )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(TABLE_REPORT_ROOT / "pooled_welch_vs_local_welch_metrics.csv", index=False)
    pooled_welch_df.to_csv(TABLE_REPORT_ROOT / "pooled_welch_universal_significance.csv", index=False)
    build_summary(pooled_features, comparison_df, len(pooled_feature_cols), REPORT_ROOT / "22_pooled_welch_vs_local_welch.txt")


if __name__ == "__main__":
    main()
