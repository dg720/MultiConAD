from __future__ import annotations

import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "Experiments"
REPORT_ROOT = PROJECT_ROOT / "reports" / "13.05"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

experiments_pkg = types.ModuleType("experiments")
experiments_pkg.__path__ = [str(EXPERIMENTS_ROOT.resolve())]
sys.modules.setdefault("experiments", experiments_pkg)

import experiments.phase1.generate_language_saliency_maps as sal
import experiments.phase1.generate_pooled_welch_overlay as pooled
import experiments.phase1.generate_welch_saliency_maps as welch
import experiments.phase1.run_rich_sweep as p1


PALETTE = {"AD": "#df4b4b", "HC": "#22a66f"}
IQR_ALPHA = {"AD": 0.34, "HC": 0.26}


def radar_angles(n: int) -> list[float]:
    values = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    return values + values[:1]


def summarize_iqr(burden_long: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    subset = burden_long[burden_long["feature_name"].isin(features)].copy()
    rows = []
    for diagnosis in ["AD", "HC"]:
        diag = subset[subset["diagnosis"] == diagnosis]
        for feature in features:
            vals = diag.loc[diag["feature_name"] == feature, "burden_0_100"].astype(float)
            rows.append(
                {
                    "diagnosis": diagnosis,
                    "feature_name": feature,
                    "median": float(vals.median()),
                    "q25": float(vals.quantile(0.25)),
                    "q75": float(vals.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def reverse_burden_scale(burden_long: pd.DataFrame) -> pd.DataFrame:
    out = burden_long.copy()
    out["burden_0_100"] = 100.0 - out["burden_0_100"].astype(float)
    sample_means = out.groupby("sample_id", as_index=False)["burden_0_100"].mean().rename(
        columns={"burden_0_100": "mean_burden"}
    )
    out = out.drop(columns=["mean_burden"], errors="ignore").merge(sample_means, on="sample_id", how="left")
    return out


def plot_iqr_overlay_grid(
    burden_by_language: dict[str, pd.DataFrame],
    feature_order_by_language: dict[str, list[str]],
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, subplot_kw={"polar": True}, figsize=(15, 12))
    axes = axes.flatten()

    for ax, code in zip(axes, ["en", "zh", "el", "es"]):
        features = feature_order_by_language[code]
        summary = summarize_iqr(burden_by_language[code], features)
        angles = radar_angles(len(features))

        for diagnosis in ["AD", "HC"]:
            diag = summary[summary["diagnosis"] == diagnosis].set_index("feature_name").reindex(features).reset_index()
            median_vals = diag["median"].tolist()
            q25_vals = diag["q25"].tolist()
            q75_vals = diag["q75"].tolist()
            median_vals += median_vals[:1]
            q25_vals += q25_vals[:1]
            q75_vals += q75_vals[:1]

            color = PALETTE[diagnosis]
            ax.plot(angles, median_vals, color=color, linewidth=2.8, label=f"{diagnosis} median")
            ax.fill_between(
                angles,
                q25_vals,
                q75_vals,
                color=color,
                alpha=IQR_ALPHA[diagnosis],
                label=f"{diagnosis} IQR",
            )

        ax.set_title(sal.LANGUAGE_SPECS[code]["label"], fontsize=13, pad=20)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([feat.replace("_", "\n") for feat in features], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)
        ax.grid(alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.suptitle(title, fontsize=16, y=0.98)
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.95))
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_note(features: list[str], output_path: Path) -> None:
    lines = []
    lines.append("Pooled Welch Median + IQR Overlay")
    lines.append("================================")
    lines.append("This plot uses the same fixed pooled universal Welch feature set as `21_saliency_overlay_pooled_welch_by_language.png`.")
    lines.append("The only difference is visualization:")
    lines.append("- solid line: per-language median burden for AD or HC")
    lines.append("- shaded band: interquartile range (25th to 75th percentile)")
    lines.append("- no individual transparent sample traces are drawn")
    lines.append("")
    lines.append("Fixed pooled Welch feature set")
    lines.append("-----------------------------")
    lines.append(", ".join(features))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_reverse_note(features: list[str], output_path: Path) -> None:
    lines = []
    lines.append("Pooled Welch Median + IQR Overlay (HC-Oriented Scale)")
    lines.append("====================================================")
    lines.append("This plot uses the same fixed pooled universal Welch feature set as the standard median/IQR version.")
    lines.append("The only change is score direction:")
    lines.append("- original burden scale: higher = more AD-like")
    lines.append("- reversed burden scale: higher = more HC-like")
    lines.append("- mathematically: `reversed_score = 100 - burden_0_100`")
    lines.append("")
    lines.append("Fixed pooled Welch feature set")
    lines.append("-----------------------------")
    lines.append(", ".join(features))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    merged = sal.load_merged()

    pooled_row = sal.best_row_for_run("benchmark_wide_rich", subset="all_universal")
    pooled_state = sal.reconstruct_run(merged, {}, pooled.POOLED_GROUPING, pooled_row)
    pooled_feature_cols = p1.feature_subset_columns(pooled_state["run_df"], "all_universal")
    pooled_feature_cols = pooled.universally_available_features(
        pooled_state["run_df"], pooled_feature_cols, min_share=pooled.MIN_LANGUAGE_AVAILABILITY
    )
    pooled_welch_df = welch.welch_significance_frame(pooled_state["train_df"], pooled_feature_cols, "Pooled")
    pooled_features = welch.top_welch_features(pooled_welch_df, "Pooled", top_n=16)

    burden_by_language = {}
    reversed_burden_by_language = {}
    feature_orders = {}
    for code in pooled.LANGUAGE_ORDER:
        meta = sal.LANGUAGE_SPECS[code]
        row = sal.best_row_for_run(f"language_{code}_all_rich", subset="all_universal")
        state = sal.reconstruct_run(merged, meta["filters"], meta["grouping_levels"], row)
        burden = sal.make_burden_frame(state, pooled_features)
        burden_by_language[code] = burden
        reversed_burden_by_language[code] = reverse_burden_scale(burden)
        feature_orders[code] = pooled_features

    plot_iqr_overlay_grid(
        burden_by_language,
        feature_orders,
        REPORT_ROOT / "23_saliency_overlay_pooled_welch_median_iqr_by_language.png",
        "Fixed pooled Welch+Bonferroni+|d| Phase 1 features: median and IQR by language",
    )
    plot_iqr_overlay_grid(
        reversed_burden_by_language,
        feature_orders,
        REPORT_ROOT / "25_saliency_overlay_pooled_welch_median_iqr_hc_oriented_by_language.png",
        "Fixed pooled Welch+Bonferroni+|d| Phase 1 features: median and IQR by language (higher = more HC-like)",
    )
    build_note(pooled_features, REPORT_ROOT / "24_pooled_welch_median_iqr_note.txt")
    build_reverse_note(pooled_features, REPORT_ROOT / "26_pooled_welch_median_iqr_hc_oriented_note.txt")


if __name__ == "__main__":
    main()
