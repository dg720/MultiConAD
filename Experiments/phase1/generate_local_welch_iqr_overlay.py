from __future__ import annotations

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "Experiments"
REPORT_ROOT = PROJECT_ROOT / "reports" / "13.05"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

experiments_pkg = types.ModuleType("experiments")
experiments_pkg.__path__ = [str(EXPERIMENTS_ROOT.resolve())]
sys.modules.setdefault("experiments", experiments_pkg)

import experiments.phase1.generate_language_saliency_maps as sal
import experiments.phase1.generate_pooled_welch_iqr_overlay as pooled_iqr
import experiments.phase1.generate_welch_saliency_maps as welch
import experiments.phase1.run_rich_sweep as p1


LANGUAGE_ORDER = ["en", "zh", "el", "es"]


def build_note(feature_orders: dict[str, list[str]], output_path: Path) -> None:
    lines = []
    lines.append("Language-Specific Welch Median + IQR Overlay")
    lines.append("===========================================")
    lines.append("This plot mirrors the pooled median/IQR radar style, but each language uses its own Welch+Bonferroni+|d| top feature list.")
    lines.append("The displayed burden scores remain AD-oriented on a 0-100 scale.")
    lines.append("- solid line: per-language median burden for AD or HC")
    lines.append("- shaded band: interquartile range (25th to 75th percentile)")
    lines.append("- no individual sample traces are drawn")
    lines.append("")
    lines.append("Language-specific Welch feature sets")
    lines.append("-----------------------------------")
    for code in LANGUAGE_ORDER:
        lines.append(f"{sal.LANGUAGE_SPECS[code]['label']}: " + ", ".join(feature_orders[code]))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_reverse_note(feature_orders: dict[str, list[str]], output_path: Path) -> None:
    lines = []
    lines.append("Language-Specific Welch Median + IQR Overlay (HC-Oriented Scale)")
    lines.append("==============================================================")
    lines.append("This plot uses the same language-specific Welch feature sets as the standard local median/IQR version.")
    lines.append("The only change is score direction:")
    lines.append("- original burden scale: higher = more AD-like")
    lines.append("- reversed burden scale: higher = more HC-like")
    lines.append("- mathematically: `reversed_score = 100 - burden_0_100`")
    lines.append("")
    lines.append("Language-specific Welch feature sets")
    lines.append("-----------------------------------")
    for code in LANGUAGE_ORDER:
        lines.append(f"{sal.LANGUAGE_SPECS[code]['label']}: " + ", ".join(feature_orders[code]))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    merged = sal.load_merged()

    burden_by_language = {}
    reversed_burden_by_language = {}
    feature_orders = {}

    for code in LANGUAGE_ORDER:
        meta = sal.LANGUAGE_SPECS[code]
        row = sal.best_row_for_run(f"language_{code}_all_rich", subset="all_universal")
        state = sal.reconstruct_run(merged, meta["filters"], meta["grouping_levels"], row)
        feature_cols = p1.feature_subset_columns(state["run_df"], "all_universal")
        welch_df = welch.welch_significance_frame(state["train_df"], feature_cols, meta["label"])
        features = welch.top_welch_features(welch_df, meta["label"], top_n=16)

        burden = sal.make_burden_frame(state, features)
        burden_by_language[code] = burden
        reversed_burden_by_language[code] = pooled_iqr.reverse_burden_scale(burden)
        feature_orders[code] = features

    pooled_iqr.plot_iqr_overlay_grid(
        burden_by_language,
        feature_orders,
        REPORT_ROOT / "27_saliency_overlay_local_welch_median_iqr_by_language.png",
        "Language-specific Welch+Bonferroni+|d| Phase 1 features: median and IQR by language",
    )
    pooled_iqr.plot_iqr_overlay_grid(
        reversed_burden_by_language,
        feature_orders,
        REPORT_ROOT / "29_saliency_overlay_local_welch_median_iqr_hc_oriented_by_language.png",
        "Language-specific Welch+Bonferroni+|d| Phase 1 features: median and IQR by language (higher = more HC-like)",
    )
    build_note(feature_orders, REPORT_ROOT / "28_local_welch_median_iqr_note.txt")
    build_reverse_note(feature_orders, REPORT_ROOT / "30_local_welch_median_iqr_hc_oriented_note.txt")


if __name__ == "__main__":
    main()
