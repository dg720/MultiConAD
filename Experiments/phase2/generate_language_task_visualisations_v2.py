from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

import generate_language_task_visualisations as base


V2_SUFFIX = "_v2"
ASSET_DIR = base.ASSET_DIR
RESULT_DIR = base.RESULT_DIR
CSV_DIR = base.CSV_DIR
SUMMARY_DIR = base.SUMMARY_DIR

CATEGORIES = [
    "translingual candidate",
    "language-sensitive candidate",
    "task-sensitive",
    "language-task confounded",
]

CATEGORY_TITLES = {
    "translingual candidate": "Translingual candidates",
    "language-sensitive candidate": "Language-sensitive candidates",
    "task-sensitive": "Task-sensitive candidates",
    "language-task confounded": "Language-task confounded",
}

CATEGORY_PREFIX = {
    "translingual candidate": "TL",
    "language-sensitive candidate": "LS",
    "task-sensitive": "TS",
    "language-task confounded": "LTC",
}

FAMILY_COLORS = {
    "ac": "#4e79a7",
    "par": "#76b7b2",
    "lex": "#59a14f",
    "syn": "#b07aa1",
    "sx": "#9c755f",
    "pr": "#8cd17d",
    "pause": "#e15759",
    "disc": "#f28e2b",
    "graph": "#ffbe7d",
    "len": "#edc948",
    "pd": "#af7aa1",
    "rd": "#bab0ac",
    "cmd": "#86bc86",
    "rep": "#c7c7c7",
    "sr": "#d4a6a6",
    "fc": "#8f9cbb",
    "ft": "#b6992d",
    "ms": "#499894",
    "na": "#79706e",
}


def ensure_dirs() -> None:
    for directory in [ASSET_DIR, RESULT_DIR, CSV_DIR, SUMMARY_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def save_png_pdf(fig: plt.Figure, stem: str) -> None:
    fig.savefig(ASSET_DIR / f"{stem}.png", dpi=220)
    try:
        fig.savefig(ASSET_DIR / f"{stem}.pdf")
    except PermissionError:
        fig.savefig(ASSET_DIR / f"{stem}_updated.pdf")


def load_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped, metadata, feature_cols = base.load_grouped_data()
    cell_meta = base.build_cell_metadata(grouped)
    effects = base.build_effects(grouped, metadata, feature_cols, cell_meta)
    summary = base.general_only_summary(base.build_feature_summary(effects))
    effects = effects[effects["feature_name"].isin(set(summary["feature_name"]))].copy()
    summary["display_name"] = summary["feature_name"].map(lambda name: base.short_feature_name(name, 38))
    summary["category_order"] = summary["interpretation_category"].map(base.CATEGORY_ORDER).fillna(9)
    return cell_meta, effects, summary


def selected_features_by_category(summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    selection_specs = {
        "translingual candidate": ("diagnostic_strength", 10),
        "language-sensitive candidate": ("language_specificity", 8),
        "task-sensitive": ("task_specificity_score", 8),
        "language-task confounded": ("diagnostic_strength", 10),
    }
    for category, (sort_col, limit) in selection_specs.items():
        sub = summary[summary["interpretation_category"] == category].copy()
        sub = sub.sort_values([sort_col, "diagnostic_strength", "stability_score"], ascending=[False, False, False]).head(limit)
        sub["category_label"] = CATEGORY_TITLES[category]
        sub["category_prefix"] = CATEGORY_PREFIX[category]
        frames.append(sub)
    selected = pd.concat(frames, ignore_index=True)
    selected["family_color"] = selected["feature_group"].map(FAMILY_COLORS).fillna("#999999")
    return selected


def ranked_feature_summary_plot(selected: pd.DataFrame) -> None:
    plot_df = selected.copy()
    plot_df = plot_df.sort_values(["category_order", "diagnostic_strength"], ascending=[True, False]).reset_index(drop=True)
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(13.0, max(9.0, len(plot_df) * 0.32 + 2.0)))
    ax.barh(
        y,
        plot_df["diagnostic_strength"],
        color=plot_df["family_color"],
        alpha=0.78,
        edgecolor="#333333",
        linewidth=0.35,
    )

    max_strength = max(float(plot_df["diagnostic_strength"].max()), 0.1)
    label_x = max_strength * 1.04

    def fmt_score(value: float) -> str:
        return "NA" if pd.isna(value) else f"{value:.2f}"

    for idx, row in plot_df.iterrows():
        ax.text(
            label_x,
            idx,
            f"{row['category_prefix']} | {row['feature_group']} | "
            f"L={fmt_score(row['language_specificity'])} "
            f"T={fmt_score(row['task_specificity_score'])} "
            f"dir={row['direction_consistency']:.2f} "
            f"cells={int(row['valid_cells'])}",
            va="center",
            fontsize=8.3,
            color="#222222",
        )

    prev_category = None
    for idx, row in plot_df.iterrows():
        if prev_category is not None and row["interpretation_category"] != prev_category:
            ax.axhline(idx - 0.5, color="#333333", linewidth=1.2)
        prev_category = row["interpretation_category"]

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_name"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Diagnostic strength: mean |Cohen's d| across valid language-task cells")
    ax.set_title("Ranked interpretable feature candidates by group", fontsize=15, pad=14)
    ax.set_xlim(0, max_strength * 1.8)
    ax.grid(axis="x", color="#dddddd", alpha=0.75)

    family_handles = [
        Patch(facecolor=color, edgecolor="#333333", label=family)
        for family, color in FAMILY_COLORS.items()
        if family in set(plot_df["feature_group"])
    ]
    ax.legend(
        handles=family_handles,
        title="Feature family",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    fig.text(
        0.01,
        0.012,
        "L=language specificity; T=task specificity; dir=direction consistency. This plot replaces the dense scatter as the main interpretive overview.",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 0.84, 1])
    save_png_pdf(fig, f"ranked_feature_groups{V2_SUFFIX}")
    plt.close(fig)


def heatmap_by_interpretation(selected: pd.DataFrame, effects: pd.DataFrame, cell_meta: pd.DataFrame) -> None:
    selected_features = selected["feature_name"].tolist()
    effect_subset = effects[effects["feature_name"].isin(selected_features)].copy()
    cell_order = (
        cell_meta[cell_meta["valid_cell_min5"] == 1]
        .sort_values(["task_type", "language"])
        ["cell_id"]
        .tolist()
    )
    cell_labels = cell_meta.set_index("cell_id").loc[cell_order, "cell_label"].tolist()
    abs_vals = np.abs(effect_subset["cohens_d"].to_numpy(dtype=float))
    vmax = float(np.nanpercentile(abs_vals, 95)) if np.isfinite(abs_vals).any() else 1.0
    vmax = max(vmax, 0.8)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d8d8d8")

    heights = []
    category_frames = []
    for category in CATEGORIES:
        cat_features = selected[selected["interpretation_category"] == category]["feature_name"].tolist()
        matrix = effect_subset[effect_subset["feature_name"].isin(cat_features)].pivot(
            index="feature_name", columns="cell_id", values="cohens_d"
        )
        matrix = matrix.reindex(index=cat_features, columns=cell_order)
        category_frames.append((category, cat_features, matrix))
        heights.append(max(1.4, 0.36 * len(cat_features) + 0.72))

    fig, axes = plt.subplots(
        len(CATEGORIES),
        1,
        figsize=(14.5, sum(heights) + 2.2),
        gridspec_kw={"height_ratios": heights},
        sharex=True,
    )
    if len(CATEGORIES) == 1:
        axes = [axes]

    effect_lookup = effects.set_index(["feature_name", "cell_id"])
    selected_lookup = selected.set_index("feature_name")
    image = None
    for ax, (category, cat_features, matrix) in zip(axes, category_frames):
        image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, norm=norm)
        labels = [
            f"{base.short_feature_name(feature, 31)} [{selected_lookup.loc[feature, 'feature_group']}]"
            for feature in cat_features
        ]
        ax.set_yticks(np.arange(len(cat_features)))
        ax.set_yticklabels(labels, fontsize=8.6)
        ax.tick_params(axis="y", colors=base.CATEGORY_COLORS.get(category, "#222222"))
        ax.set_title(CATEGORY_TITLES[category], loc="left", fontsize=11, color=base.CATEGORY_COLORS.get(category, "#222222"), pad=5)
        ax.set_xticks(np.arange(-0.5, len(cell_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(cat_features), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for y, feature in enumerate(cat_features):
            for x, cell_id in enumerate(cell_order):
                if (feature, cell_id) not in effect_lookup.index:
                    continue
                row = effect_lookup.loc[(feature, cell_id)]
                marker = ""
                if bool(row.get("welch_significant_bonferroni", False)):
                    marker = "*"
                elif pd.notna(row.get("welch_rank")) and float(row["welch_rank"]) <= 50:
                    marker = "."
                if marker:
                    ax.text(x, y, marker, ha="center", va="center", color="#111111", fontsize=10, fontweight="bold")

    axes[-1].set_xticks(np.arange(len(cell_order)))
    axes[-1].set_xticklabels(cell_labels, rotation=38, ha="right", fontsize=9)
    axes[-1].set_xlabel("Language-task cell: paper task label above live manifest label")
    fig.suptitle("Language-task AD/HC effects, split by interpretation group", fontsize=15, y=0.995)
    if image is not None:
        cax = fig.add_axes([0.91, 0.18, 0.014, 0.62])
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label("Signed Cohen's d (AD - HC)")
    fig.text(
        0.01,
        0.012,
        "* = Welch Bonferroni p < .05; . = Welch top-50 in that cell. Red = higher in AD; blue = lower in AD; grey = not applicable/insufficient.",
        fontsize=8.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.19, right=0.89, top=0.96, bottom=0.10, hspace=0.44)
    save_png_pdf(fig, f"language_task_heatmap_by_interpretation{V2_SUFFIX}")
    plt.close(fig)


def family_composition_plot(selected: pd.DataFrame) -> pd.DataFrame:
    counts = (
        selected.groupby(["interpretation_category", "feature_group"])
        .size()
        .reset_index(name="count")
    )
    wide = counts.pivot(index="interpretation_category", columns="feature_group", values="count").fillna(0)
    wide = wide.reindex(CATEGORIES).fillna(0)
    families = sorted(wide.columns.tolist())

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    left = np.zeros(len(wide))
    y = np.arange(len(wide))
    for family in families:
        values = wide[family].to_numpy()
        if values.sum() == 0:
            continue
        ax.barh(y, values, left=left, color=FAMILY_COLORS.get(family, "#999999"), edgecolor="white", label=family)
        for idx, value in enumerate(values):
            if value > 0:
                ax.text(
                    left[idx] + value / 2,
                    idx,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value >= 2 else "#222222",
                    fontweight="bold",
                )
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([CATEGORY_TITLES.get(cat, cat) for cat in wide.index])
    ax.invert_yaxis()
    ax.set_xlabel("Number of labelled features in curated v2 figures")
    ax.set_title("Feature-family composition of selected interpretation groups", fontsize=14, pad=12)
    ax.grid(axis="x", color="#dddddd", alpha=0.7)
    ax.legend(title="Feature family", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8, title_fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.84, 1])
    save_png_pdf(fig, f"feature_family_composition{V2_SUFFIX}")
    plt.close(fig)
    return counts


def write_v2_report(selected: pd.DataFrame, family_counts: pd.DataFrame, cell_meta: pd.DataFrame) -> None:
    lines = [
        "Language-Task Visualisations V2 Review",
        "======================================",
        "",
        "What changed",
        "------------",
        "The dense scatter is no longer the main interpretive figure. It is retained as a diagnostic appendix-style view, while V2 uses ranked grouped feature plots and small-multiple heatmaps.",
        "The V2 figures make interpretation group and feature family explicit in the row labels, colours, and composition summary.",
        "Task-specific feature families are excluded from these V2 mixed language/task figures. They are regenerated separately in the task-specific within-task outputs.",
        "",
        "Recommended main figures",
        "------------------------",
        "1. ranked_feature_groups_v2.png: the clearest overview of which features belong to each interpretation group.",
        "2. language_task_heatmap_by_interpretation_v2.png: the direct evidence view, split into translingual, language-sensitive, task-sensitive, and confounded panels.",
        "3. feature_family_composition_v2.png: compact summary of which feature families dominate each interpretation group.",
        "",
        "Interpretation",
        "--------------",
        "Translingual candidates are mostly syntactic, lexical, pause, graph, and acoustic features. They show moderate diagnostic strength and comparatively low variation across language/task summaries.",
        "Language-sensitive candidates are mainly lexical/discourse features, including MATTR/content-word ratio and utterance similarity. These are candidates only, because task coverage is still partially confounded.",
        "Task-sensitive candidates include acoustic/prosody and function-word/object-ratio features whose AD/HC separation changes more by task than by language.",
        "Language-task confounded features are dominated by acoustic/loudness/prosody features. Many have large effect sizes, but the heatmap shows strong cell dependence, so they should not be presented as language-specific biomarkers.",
        "",
        "Caveat",
        "------",
        "These figures use the simplified signed Cohen's d specificity method over valid language-task cells. They are general-feature descriptive visualisations over the Phase 2 matrix, not regression-interaction robustness checks.",
        "",
        "Curated feature rows",
        "--------------------",
    ]
    cols = [
        "category_prefix",
        "display_name",
        "feature_name",
        "feature_group",
        "diagnostic_strength",
        "language_specificity",
        "task_specificity_score",
        "direction_consistency",
        "valid_cells",
    ]
    report_df = selected[cols].copy()
    for col in ["diagnostic_strength", "language_specificity", "task_specificity_score", "direction_consistency"]:
        report_df[col] = report_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    lines.append(report_df.to_string(index=False))
    lines.extend(["", "Feature-family counts", "---------------------"])
    lines.append(family_counts.sort_values(["interpretation_category", "count"], ascending=[True, False]).to_string(index=False))
    lines.extend(
        [
            "",
            "Generated V2 files",
            "------------------",
            "- report_assets/ranked_feature_groups_v2.png/.pdf",
            "- report_assets/language_task_heatmap_by_interpretation_v2.png/.pdf",
            "- report_assets/feature_family_composition_v2.png/.pdf",
            "- result-tables/language_task_visualisation_v2_review.txt",
            "- result-tables/csv/language_task_visualisation_v2_selected_features.csv",
            "- result-tables/csv/language_task_visualisation_v2_family_counts.csv",
        ]
    )
    (RESULT_DIR / "language_task_visualisation_v2_review.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    cell_meta, effects, summary = load_analysis()
    selected = selected_features_by_category(summary)
    selected.to_csv(CSV_DIR / "language_task_visualisation_v2_selected_features.csv", index=False)

    ranked_feature_summary_plot(selected)
    heatmap_by_interpretation(selected, effects, cell_meta)
    family_counts = family_composition_plot(selected)
    family_counts.to_csv(CSV_DIR / "language_task_visualisation_v2_family_counts.csv", index=False)
    write_v2_report(selected, family_counts, cell_meta)

    run_summary = {
        "num_selected_features": int(len(selected)),
        "v2_outputs": [
            str(ASSET_DIR / f"ranked_feature_groups{V2_SUFFIX}.png"),
            str(ASSET_DIR / f"language_task_heatmap_by_interpretation{V2_SUFFIX}.png"),
            str(ASSET_DIR / f"feature_family_composition{V2_SUFFIX}.png"),
            str(RESULT_DIR / "language_task_visualisation_v2_review.txt"),
        ],
    }
    (SUMMARY_DIR / "language_task_visualisation_v2_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
