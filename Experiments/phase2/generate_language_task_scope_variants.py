from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

import generate_language_task_visualisations as base
import generate_language_task_visualisations_v2 as v2


ASSET_DIR = base.ASSET_DIR
RESULT_DIR = base.RESULT_DIR
CSV_DIR = base.CSV_DIR
SUMMARY_DIR = base.SUMMARY_DIR

SCOPES = {
    "core": {
        "title": "Core/general features only",
        "description": "General features excluding task-specific PD/FT/SR/FC/NA feature families.",
    },
    "task_specific": {
        "title": "Task-specific features only",
        "description": "Only features whose metadata marks them as task-specific.",
    },
}

SELECTION_SPECS = {
    "translingual candidate": ("diagnostic_strength", 10),
    "language-sensitive candidate": ("language_specificity", 8),
    "task-sensitive": ("task_specificity_score", 8),
    "language-task confounded": ("diagnostic_strength", 10),
}


def ensure_dirs() -> None:
    for directory in [ASSET_DIR, RESULT_DIR, CSV_DIR, SUMMARY_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def is_task_specific(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def scoped_summary(summary: pd.DataFrame, metadata: pd.DataFrame, scope: str) -> pd.DataFrame:
    task_mask = is_task_specific(metadata["task_specific"])
    if scope == "core":
        feature_names = set(metadata.loc[~task_mask, "feature_name"])
    elif scope == "task_specific":
        feature_names = set(metadata.loc[task_mask, "feature_name"])
    else:
        raise ValueError(f"Unknown scope: {scope}")
    return summary[summary["feature_name"].isin(feature_names)].copy()


def select_features(summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for category, (sort_col, limit) in SELECTION_SPECS.items():
        sub = summary[summary["interpretation_category"] == category].copy()
        if sub.empty:
            continue
        sub = sub.sort_values([sort_col, "diagnostic_strength", "stability_score"], ascending=[False, False, False]).head(limit)
        sub["category_label"] = v2.CATEGORY_TITLES[category]
        sub["category_prefix"] = v2.CATEGORY_PREFIX[category]
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=list(summary.columns) + ["category_label", "category_prefix", "family_color"])
    selected = pd.concat(frames, ignore_index=True)
    selected["family_color"] = selected["feature_group"].map(v2.FAMILY_COLORS).fillna("#999999")
    return selected


def fmt_score(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.2f}"


def plot_ranked(selected: pd.DataFrame, scope: str, title: str) -> Path | None:
    if selected.empty:
        return None
    plot_df = selected.sort_values(["category_order", "diagnostic_strength"], ascending=[True, False]).reset_index(drop=True)
    y = np.arange(len(plot_df))
    max_strength = max(float(plot_df["diagnostic_strength"].max()), 0.1)

    fig, ax = plt.subplots(figsize=(13.2, max(7.0, len(plot_df) * 0.34 + 2.0)))
    ax.barh(
        y,
        plot_df["diagnostic_strength"],
        color=plot_df["family_color"],
        alpha=0.78,
        edgecolor="#333333",
        linewidth=0.35,
    )

    label_x = max_strength * 1.04
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
            fontsize=8.2,
            color="#222222",
        )

    prev_category = None
    for idx, row in plot_df.iterrows():
        if prev_category is not None and row["interpretation_category"] != prev_category:
            ax.axhline(idx - 0.5, color="#333333", linewidth=1.1)
        prev_category = row["interpretation_category"]

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_name"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Diagnostic strength: mean |Cohen's d| across valid language-task cells")
    ax.set_title(f"{title}: ranked interpretation groups", fontsize=14, pad=12)
    ax.set_xlim(0, max_strength * 1.85)
    ax.grid(axis="x", color="#dddddd", alpha=0.75)
    handles = [
        Patch(facecolor=color, edgecolor="#333333", label=family)
        for family, color in v2.FAMILY_COLORS.items()
        if family in set(plot_df["feature_group"])
    ]
    ax.legend(handles=handles, title="Feature family", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8, title_fontsize=9)
    fig.text(0.01, 0.012, "L=language specificity; T=task specificity; dir=direction consistency.", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=[0, 0.04, 0.84, 1])
    out = ASSET_DIR / f"ranked_feature_groups_{scope}_v2.png"
    fig.savefig(out, dpi=220)
    fig.savefig(ASSET_DIR / f"ranked_feature_groups_{scope}_v2.pdf")
    plt.close(fig)
    return out


def plot_heatmap(selected: pd.DataFrame, effects: pd.DataFrame, cell_meta: pd.DataFrame, scope: str, title: str) -> Path | None:
    if selected.empty:
        return None
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

    category_frames = []
    heights = []
    for category in v2.CATEGORIES:
        cat_features = selected[selected["interpretation_category"] == category]["feature_name"].tolist()
        if not cat_features:
            continue
        matrix = effect_subset[effect_subset["feature_name"].isin(cat_features)].pivot(index="feature_name", columns="cell_id", values="cohens_d")
        matrix = matrix.reindex(index=cat_features, columns=cell_order)
        category_frames.append((category, cat_features, matrix))
        heights.append(max(1.35, 0.36 * len(cat_features) + 0.72))
    if not category_frames:
        return None

    fig, axes = plt.subplots(
        len(category_frames),
        1,
        figsize=(14.5, sum(heights) + 2.2),
        gridspec_kw={"height_ratios": heights},
        sharex=True,
    )
    if len(category_frames) == 1:
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
        ax.set_title(v2.CATEGORY_TITLES[category], loc="left", fontsize=11, color=base.CATEGORY_COLORS.get(category, "#222222"), pad=5)
        ax.set_xticks(np.arange(-0.5, len(cell_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(cat_features), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for y_idx, feature in enumerate(cat_features):
            for x_idx, cell_id in enumerate(cell_order):
                if (feature, cell_id) not in effect_lookup.index:
                    continue
                row = effect_lookup.loc[(feature, cell_id)]
                marker = ""
                if bool(row.get("welch_significant_bonferroni", False)):
                    marker = "*"
                elif pd.notna(row.get("welch_rank")) and float(row["welch_rank"]) <= 50:
                    marker = "."
                if marker:
                    ax.text(x_idx, y_idx, marker, ha="center", va="center", color="#111111", fontsize=10, fontweight="bold")

    axes[-1].set_xticks(np.arange(len(cell_order)))
    axes[-1].set_xticklabels(cell_labels, rotation=38, ha="right", fontsize=9)
    axes[-1].set_xlabel("Language-task cell: paper task label above live manifest label")
    fig.suptitle(f"{title}: signed AD/HC effects by interpretation group", fontsize=15, y=0.995)
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
    fig.subplots_adjust(left=0.19, right=0.89, top=0.94, bottom=0.12, hspace=0.44)
    out = ASSET_DIR / f"language_task_heatmap_by_interpretation_{scope}_v2.png"
    fig.savefig(out, dpi=220)
    fig.savefig(ASSET_DIR / f"language_task_heatmap_by_interpretation_{scope}_v2.pdf")
    plt.close(fig)
    return out


def plot_specificity_scatter(summary: pd.DataFrame, selected: pd.DataFrame, scope: str, title: str) -> Path | None:
    plot_df = summary[(summary["valid_cells"] >= 2) & summary["language_specificity"].notna()].copy()
    if plot_df.empty:
        return None
    plot_df["task_score_missing"] = plot_df["task_specificity_score"].isna()
    plot_df["task_specificity_plot"] = plot_df["task_specificity_score"].fillna(0.0)
    lang_threshold = float(plot_df["language_specificity_threshold"].dropna().iloc[0]) if plot_df["language_specificity_threshold"].notna().any() else 0.0
    task_threshold = float(plot_df["task_specificity_threshold"].dropna().iloc[0]) if plot_df["task_specificity_threshold"].notna().any() else 0.0
    xmax = max(plot_df["task_specificity_plot"].quantile(0.99) * 1.2, task_threshold * 1.3, 0.2)
    ymax = max(plot_df["language_specificity"].quantile(0.99) * 1.2, lang_threshold * 1.3, 0.2)

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for category in v2.CATEGORIES + ["unstable / insufficient evidence"]:
        sub = plot_df[plot_df["interpretation_category"] == category]
        if sub.empty:
            continue
        finite = sub[~sub["task_score_missing"]]
        missing = sub[sub["task_score_missing"]]
        color = base.CATEGORY_COLORS.get(category, "#999999")
        if not finite.empty:
            ax.scatter(finite["task_specificity_plot"], finite["language_specificity"], s=38, color=color, alpha=0.52, edgecolor="white", linewidth=0.35, label=category)
        if not missing.empty:
            ax.scatter(missing["task_specificity_plot"], missing["language_specificity"], marker="^", s=44, color=color, alpha=0.62, edgecolor="#333333", linewidth=0.35, label=f"{category} (single-task T=NA)")

    label_df = selected[selected["feature_name"].isin(set(plot_df["feature_name"]))].copy()
    label_df = (
        label_df.sort_values(["category_order", "diagnostic_strength"], ascending=[True, False])
        .groupby("interpretation_category", group_keys=False)
        .head(3)
        .head(10)
    )
    for _, row in label_df.iterrows():
        x = 0.0 if pd.isna(row["task_specificity_score"]) else float(row["task_specificity_score"])
        y = float(row["language_specificity"])
        ax.text(x + 0.012 * xmax, y + 0.012 * ymax, base.short_feature_name(row["feature_name"], 22), fontsize=7.5)

    ax.axvline(task_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axhline(lang_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_xlim(-0.03 * xmax, xmax)
    ax.set_ylim(-0.03 * ymax, ymax)
    ax.set_xlabel("Task specificity: std of signed task-averaged effects")
    ax.set_ylabel("Language specificity: std of signed language-averaged effects")
    ax.set_title(f"{title}: specificity scatter", fontsize=14, pad=12)
    ax.grid(color="#dddddd", alpha=0.65)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7.5)
    fig.text(0.01, 0.012, "Triangles have no task-specificity estimate because valid evidence spans only one task label.", fontsize=8.2, color="#444444")
    fig.tight_layout(rect=[0, 0.04, 0.78, 1])
    out = ASSET_DIR / f"specificity_scatter_fixed_{scope}_v2.png"
    fig.savefig(out, dpi=220)
    fig.savefig(ASSET_DIR / f"specificity_scatter_fixed_{scope}_v2.pdf")
    plt.close(fig)
    return out


def plot_strength_coverage(summary: pd.DataFrame, selected: pd.DataFrame, scope: str, title: str) -> Path | None:
    plot_df = summary[summary["valid_cells"] >= 1].copy()
    if plot_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    for category in v2.CATEGORIES + ["unstable / insufficient evidence"]:
        sub = plot_df[plot_df["interpretation_category"] == category]
        if sub.empty:
            continue
        ax.scatter(
            sub["valid_cells"],
            sub["diagnostic_strength"],
            s=36,
            color=base.CATEGORY_COLORS.get(category, "#999999"),
            alpha=0.55,
            edgecolor="white",
            linewidth=0.35,
            label=category,
        )
    label_df = (
        selected.sort_values(["category_order", "diagnostic_strength"], ascending=[True, False])
        .groupby("interpretation_category", group_keys=False)
        .head(3)
        .head(10)
    )
    for label_position, (_, row) in enumerate(label_df.iterrows()):
        x_offset = 0.06 + (label_position % 2) * 0.16
        y_offset = 0.025 + (label_position % 5) * 0.04
        ax.text(
            float(row["valid_cells"]) + x_offset,
            float(row["diagnostic_strength"]) + y_offset,
            base.short_feature_name(row["feature_name"], 22),
            fontsize=7.2,
        )
    ax.set_xlabel("Number of valid language-task cells")
    ax.set_ylabel("Diagnostic strength: mean |Cohen's d|")
    short_title = "Task-specific" if scope == "task_specific" else "Core/general"
    ax.set_title(f"{short_title}: strength vs evidence coverage", fontsize=14, pad=12)
    ax.set_xlim(0.5, max(8.5, float(plot_df["valid_cells"].max()) + 0.6))
    ymax = max(0.8, float(plot_df["diagnostic_strength"].quantile(0.99)) * 1.18)
    ax.set_ylim(0, ymax)
    ax.grid(color="#dddddd", alpha=0.65)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    out = ASSET_DIR / f"diagnostic_strength_coverage_{scope}_v2.png"
    fig.savefig(out, dpi=220)
    fig.savefig(ASSET_DIR / f"diagnostic_strength_coverage_{scope}_v2.pdf")
    plt.close(fig)
    return out


def family_counts(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(columns=["interpretation_category", "feature_group", "count"])
    return selected.groupby(["interpretation_category", "feature_group"]).size().reset_index(name="count")


def write_report(scope_rows: list[dict], selected_by_scope: dict[str, pd.DataFrame], generated: dict[str, list[str]]) -> None:
    lines = [
        "Language-Task Scope Variant Visualisation Review",
        "================================================",
        "",
        "Purpose",
        "-------",
        "This reruns the language/task specificity visual grammar separately for core/general features and task-specific features, without overwriting the full-feature V2 figures.",
        "",
        "Review",
        "------",
        "The core-only ranked plot and grouped heatmap are the most interpretable main figures. They remove sparse task-derived features and show cross-task behaviour for general lexical, syntactic, pause, discourse, acoustic, and paralinguistic features.",
        "The task-specific-only plots are useful as an audit of task-feature behaviour, but they should be treated as supporting figures. Most task-specific features are sparse or single-task by construction, so they cannot support clean translingual or language-specific claims.",
        "The fixed-size scatter variants are clearer than the original dense scatter after splitting by scope, but they are still secondary to the ranked plot and grouped heatmap. For task-specific features, the scatter mainly shows missing task-specificity estimates rather than a clean quadrant structure.",
        "The diagnostic-strength-vs-coverage variants are useful for explaining why some apparently strong task-specific effects are weak evidence: high effect size often comes with very few valid cells.",
        "",
        "Recommendation",
        "--------------",
        "Use ranked_feature_groups_core_v2 and language_task_heatmap_by_interpretation_core_v2 as the main interpretability figures.",
        "Use ranked_feature_groups_task_specific_v2 and diagnostic_strength_coverage_task_specific_v2 as supporting/audit figures.",
        "Keep the full-feature V2 plots as appendix-style comprehensive views.",
        "",
        "Scope summary",
        "-------------",
    ]
    scope_df = pd.DataFrame(scope_rows)
    lines.append(scope_df.to_string(index=False))

    for scope, selected in selected_by_scope.items():
        lines.extend(["", f"Selected rows: {scope}", "-" * (15 + len(scope))])
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
        lines.extend(["", "Generated figures"])
        lines.extend(f"- {Path(path).name}" for path in generated[scope])

    (RESULT_DIR / "language_task_scope_variant_review.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    cell_meta, effects, summary = v2.load_analysis()
    metadata = pd.read_csv(base.METADATA_PATH)

    generated: dict[str, list[str]] = {}
    selected_by_scope: dict[str, pd.DataFrame] = {}
    scope_rows: list[dict] = []

    for scope, scope_info in SCOPES.items():
        scoped = scoped_summary(summary, metadata, scope)
        selected = select_features(scoped)
        selected_by_scope[scope] = selected
        selected.to_csv(CSV_DIR / f"language_task_{scope}_v2_selected_features.csv", index=False)
        family_counts(selected).to_csv(CSV_DIR / f"language_task_{scope}_v2_family_counts.csv", index=False)

        category_counts = scoped["interpretation_category"].value_counts().to_dict()
        scope_rows.append(
            {
                "scope": scope,
                "features": int(len(scoped)),
                "selected": int(len(selected)),
                "translingual": int(category_counts.get("translingual candidate", 0)),
                "language_sensitive": int(category_counts.get("language-sensitive candidate", 0)),
                "task_sensitive": int(category_counts.get("task-sensitive", 0)),
                "language_task_confounded": int(category_counts.get("language-task confounded", 0)),
                "unstable": int(category_counts.get("unstable / insufficient evidence", 0)),
            }
        )

        title = scope_info["title"]
        paths = [
            plot_ranked(selected, scope, title),
            plot_heatmap(selected, effects, cell_meta, scope, title),
            plot_specificity_scatter(scoped, selected, scope, title),
            plot_strength_coverage(scoped, selected, scope, title),
        ]
        generated[scope] = [str(path) for path in paths if path is not None]

    pd.DataFrame(scope_rows).to_csv(CSV_DIR / "language_task_scope_variant_summary.csv", index=False)
    write_report(scope_rows, selected_by_scope, generated)

    run_summary = {
        "scopes": scope_rows,
        "generated": generated,
        "report": str(RESULT_DIR / "language_task_scope_variant_review.txt"),
    }
    (SUMMARY_DIR / "language_task_scope_variant_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
