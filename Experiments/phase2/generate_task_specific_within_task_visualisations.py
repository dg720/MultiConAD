from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import generate_language_task_visualisations as base


OUT_ROOT = base.OUT_ROOT
RESULT_DIR = base.RESULT_DIR
CSV_DIR = base.CSV_DIR
SUMMARY_DIR = base.SUMMARY_DIR
ASSET_DIR = OUT_ROOT / "report_assets" / "task-specific-within-task"

TASK_COLORS = {
    "PD_CTP": "#4e79a7",
    "PICTURE_RECALL": "#59a14f",
    "READING": "#f28e2b",
    "CONVERSATION": "#e15759",
    "COMMAND": "#76b7b2",
    "REPETITION": "#af7aa1",
    "MOTOR_SPEECH": "#edc948",
    "MIXED_PROTOCOL": "#8f9cbb",
    "OTHER": "#bab0ac",
}


def ensure_dirs() -> None:
    for directory in [ASSET_DIR, RESULT_DIR, CSV_DIR, SUMMARY_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def build_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped, metadata, feature_cols = base.load_grouped_data()
    cell_meta = base.build_cell_metadata(grouped)
    effects = base.build_effects(grouped, metadata, feature_cols, cell_meta)
    task_effects = effects[base.task_specific_mask(effects["task_specific"])].copy()
    return cell_meta, effects, task_effects


def within_task_summary(task_effects: pd.DataFrame) -> pd.DataFrame:
    valid = task_effects[task_effects["valid_effect"]].copy()
    rows: list[dict[str, object]] = []
    for (feature, task_type), group in valid.groupby(["feature_name", "task_type"], sort=True):
        first = group.iloc[0]
        lang_effects = group.groupby("language")["cohens_d"].mean()
        signs = np.sign(group["cohens_d"].astype(float))
        signs = signs[signs != 0]
        rows.append(
            {
                "feature_name": feature,
                "feature_label": first["feature_label"],
                "feature_group": first["feature_group"],
                "task_type": task_type,
                "paper_task_labels": "|".join(sorted(set("|".join(group["paper_task_labels"]).split("|")) - {""})),
                "diagnostic_strength_within_task": float(group["cohens_d_abs"].mean()),
                "language_specificity_within_task": float(lang_effects.std(ddof=0)) if len(lang_effects) >= 2 else np.nan,
                "direction_consistency": float(signs.value_counts(normalize=True).iloc[0]) if not signs.empty else np.nan,
                "valid_languages": int(group["language"].nunique()),
                "valid_cells": int(group["cell_id"].nunique()),
                "languages": "|".join(sorted(group["language"].unique())),
                "cell_ids": "|".join(sorted(group["cell_id"].unique())),
                "max_abs_effect": float(group["cohens_d_abs"].max()),
                "significant_cells": int(group["welch_significant_bonferroni"].fillna(False).sum()),
                "top50_cells": int((group["welch_rank"] <= 50).fillna(False).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["valid_languages", "language_specificity_within_task", "diagnostic_strength_within_task"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def select_rows(summary: pd.DataFrame, max_rows: int = 34) -> pd.DataFrame:
    comparable = summary[summary["valid_languages"] >= 2].copy()
    if comparable.empty:
        return summary.sort_values("diagnostic_strength_within_task", ascending=False).head(max_rows)
    selected = []
    for task_type, group in comparable.groupby("task_type", sort=True):
        selected.append(
            group.sort_values(
                ["language_specificity_within_task", "diagnostic_strength_within_task"],
                ascending=[False, False],
            ).head(10)
        )
    out = pd.concat(selected, ignore_index=True)
    out = out.sort_values(
        ["task_type", "language_specificity_within_task", "diagnostic_strength_within_task"],
        ascending=[True, False, False],
    )
    return out.head(max_rows)


def plot_ranked_language_specificity(summary: pd.DataFrame) -> Path | None:
    plot_df = select_rows(summary)
    plot_df = plot_df[plot_df["valid_languages"] >= 2].copy()
    if plot_df.empty:
        return None
    plot_df["row_label"] = plot_df.apply(
        lambda row: f"{base.short_feature_name(row['feature_name'], 30)} [{row['task_type']}]",
        axis=1,
    )
    plot_df = plot_df.sort_values(
        ["task_type", "language_specificity_within_task", "diagnostic_strength_within_task"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(12.4, max(7.2, 0.34 * len(plot_df) + 2.2)))
    colors = plot_df["task_type"].map(TASK_COLORS).fillna("#999999")
    ax.barh(y, plot_df["language_specificity_within_task"], color=colors, edgecolor="#333333", linewidth=0.35, alpha=0.82)
    for idx, row in plot_df.iterrows():
        ax.text(
            row["language_specificity_within_task"] + 0.015,
            idx,
            f"strength={row['diagnostic_strength_within_task']:.2f} langs={int(row['valid_languages'])} cells={int(row['valid_cells'])}",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["row_label"], fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlabel("Language specificity within task: SD of signed AD-HC effects across languages")
    ax.set_title("Task-Specific Features: Language Specificity Within Each Task", fontsize=14, pad=12)
    xmax = max(0.25, float(plot_df["language_specificity_within_task"].max()) * 1.55)
    ax.set_xlim(0, xmax)
    ax.grid(axis="x", color="#dddddd", alpha=0.7)
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=color, markeredgecolor="#333333", label=task, markersize=8)
        for task, color in TASK_COLORS.items()
        if task in set(plot_df["task_type"])
    ]
    ax.legend(handles=handles, title="Live task", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    fig.text(
        0.01,
        0.012,
        "Only feature-task pairs with at least two valid languages are shown. This avoids treating single-task applicability as task variance.",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 0.84, 1])
    out = ASSET_DIR / "task_specific_language_specificity_ranked.png"
    fig.savefig(out, dpi=220)
    fig.savefig(ASSET_DIR / "task_specific_language_specificity_ranked.pdf")
    plt.close(fig)
    return out


def plot_strength_scatter(summary: pd.DataFrame) -> Path | None:
    plot_df = summary[summary["valid_languages"] >= 2].copy()
    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10.4, 7.2))
    for task_type, sub in plot_df.groupby("task_type", sort=True):
        color = TASK_COLORS.get(task_type, "#999999")
        ax.scatter(
            sub["diagnostic_strength_within_task"],
            sub["language_specificity_within_task"],
            s=68,
            color=color,
            alpha=0.74,
            edgecolors="#222222",
            linewidth=0.45,
            label=task_type,
        )

    labels = plot_df.sort_values(
        ["language_specificity_within_task", "diagnostic_strength_within_task"],
        ascending=[False, False],
    ).head(12)
    xmax = max(0.5, float(plot_df["diagnostic_strength_within_task"].quantile(0.99)) * 1.22)
    ymax = max(0.25, float(plot_df["language_specificity_within_task"].quantile(0.99)) * 1.28)
    for idx, (_, row) in enumerate(labels.iterrows()):
        ax.annotate(
            base.short_feature_name(row["feature_name"], 24),
            (row["diagnostic_strength_within_task"], row["language_specificity_within_task"]),
            xytext=(5, 5 + (idx % 3) * 4),
            textcoords="offset points",
            fontsize=7.6,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.72},
        )
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.02 * ymax, ymax)
    ax.set_xlabel("Diagnostic strength within task: mean |Cohen's d|")
    ax.set_ylabel("Language specificity within task")
    ax.set_title("Task-Specific Features: Strength vs Within-Task Language Specificity", fontsize=14, pad=12)
    ax.grid(color="#dddddd", alpha=0.7)
    ax.legend(title="Live task", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.text(
        0.01,
        0.012,
        "Only feature-task pairs with at least two valid languages are plotted; sparse single-language effects are listed in the report.",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 0.80, 1])
    out = ASSET_DIR / "task_specific_strength_vs_language_specificity.png"
    fig.savefig(out, dpi=220)
    fig.savefig(ASSET_DIR / "task_specific_strength_vs_language_specificity.pdf")
    plt.close(fig)
    return out


def plot_effect_heatmap(task_effects: pd.DataFrame, summary: pd.DataFrame, cell_meta: pd.DataFrame) -> Path | None:
    selected = select_rows(summary, max_rows=36)
    if selected.empty:
        return None
    selected["row_id"] = selected["feature_name"] + "||" + selected["task_type"]
    valid = task_effects[task_effects["valid_effect"]].copy()
    valid["row_id"] = valid["feature_name"] + "||" + valid["task_type"]
    valid = valid[valid["row_id"].isin(set(selected["row_id"]))].copy()

    cells = (
        valid[["cell_id", "cell_label", "task_type", "language"]]
        .drop_duplicates()
        .sort_values(["task_type", "language"])
    )
    cell_order = cells["cell_id"].tolist()
    matrix = valid.pivot_table(index="row_id", columns="cell_id", values="cohens_d", aggfunc="first")
    row_order = selected["row_id"].tolist()
    matrix = matrix.reindex(index=row_order, columns=cell_order)
    row_labels = [
        f"{base.short_feature_name(row.feature_name, 28)} [{row.task_type}]"
        for row in selected.itertuples(index=False)
    ]
    col_labels = cells.set_index("cell_id").loc[cell_order, "cell_label"].map(lambda value: str(value).replace("\n", " ")).tolist()

    values = matrix.to_numpy(dtype=float)
    vmax = float(np.nanpercentile(np.abs(values), 95)) if np.isfinite(values).any() else 1.0
    vmax = max(vmax, 0.8)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d8d8d8")
    fig, ax = plt.subplots(figsize=(13.8, max(8.0, 0.31 * len(row_order) + 2.2)))
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.2)
    ax.set_xticks(np.arange(len(cell_order)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_title("Task-Specific Feature Effects by Language Within Task", fontsize=14, pad=12)
    ax.set_xlabel("Language-task cell")
    ax.set_ylabel("Task-specific feature")
    ax.set_xticks(np.arange(-0.5, len(cell_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    lookup = valid.set_index(["row_id", "cell_id"])
    for y, row_id in enumerate(row_order):
        for x, cell_id in enumerate(cell_order):
            if (row_id, cell_id) not in lookup.index:
                continue
            row = lookup.loc[(row_id, cell_id)]
            marker = ""
            if bool(row.get("welch_significant_bonferroni", False)):
                marker = "*"
            elif pd.notna(row.get("welch_rank")) and float(row["welch_rank"]) <= 50:
                marker = "."
            if marker:
                ax.text(x, y, marker, ha="center", va="center", fontsize=10, color="#111111", fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.024, pad=0.02)
    cbar.set_label("Signed Cohen's d (AD - HC)")
    fig.text(
        0.01,
        0.012,
        "* = Welch Bonferroni p < .05; . = Welch top-50. Grey means not applicable or insufficient in that language-task cell.",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = ASSET_DIR / "task_specific_within_task_effect_heatmap.png"
    fig.savefig(out, dpi=220)
    fig.savefig(ASSET_DIR / "task_specific_within_task_effect_heatmap.pdf")
    plt.close(fig)
    return out


def write_report(summary: pd.DataFrame, generated: list[Path]) -> None:
    comparable = summary[summary["valid_languages"] >= 2].copy()
    sparse = summary[summary["valid_languages"] < 2].copy()

    def table(df: pd.DataFrame, n: int = 20) -> str:
        if df.empty:
            return "(none)"
        cols = [
            "feature_label",
            "task_type",
            "diagnostic_strength_within_task",
            "language_specificity_within_task",
            "valid_languages",
            "valid_cells",
            "languages",
        ]
        view = df[cols].head(n).copy()
        for col in ["diagnostic_strength_within_task", "language_specificity_within_task"]:
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        return view.to_string(index=False)

    lines = [
        "Task-Specific Within-Task Visualisation Report",
        "==============================================",
        "",
        "Method",
        "------",
        "Task-specific features are not plotted on the general task-specificity axis. They are only valid within their task domain by construction.",
        "For each feature-task pair, language specificity is the standard deviation of signed AD-HC Cohen's d across valid languages inside that same task.",
        "Feature-task pairs with only one valid language have undefined language specificity and are listed separately as sparse evidence rather than plotted on the language-specificity scatter.",
        "",
        "Comparable feature-task pairs (at least two valid languages)",
        "-----------------------------------------------------------",
        table(comparable.sort_values(["language_specificity_within_task", "diagnostic_strength_within_task"], ascending=[False, False])),
        "",
        "Sparse single-language task-specific effects",
        "--------------------------------------------",
        table(sparse.sort_values("diagnostic_strength_within_task", ascending=False), n=20),
        "",
        "Generated files",
        "---------------",
    ]
    lines.extend(f"- {path}" for path in generated)
    lines.extend(
        [
            f"- {CSV_DIR / 'task_specific_within_task_summary.csv'}",
            f"- {CSV_DIR / 'task_specific_within_task_effects.csv'}",
        ]
    )
    (RESULT_DIR / "task_specific_within_task_visualisation_report.txt").write_text("\n".join(lines), encoding="utf-8")


def update_index() -> None:
    index_path = RESULT_DIR / "plot_asset_folder_index.txt"
    current = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
    entry = (
        "task-specific-within-task/\n"
        "Task-specific feature figures that compare language specificity only within the task where each feature is valid.\n"
        "- ranked within-task language specificity\n"
        "- strength vs within-task language specificity scatter\n"
        "- task-specific language-task effect heatmap\n"
    )
    if "task-specific-within-task/" not in current:
        index_path.write_text(current.rstrip() + "\n\n" + entry, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    cell_meta, _effects, task_effects = build_analysis()
    summary = within_task_summary(task_effects)
    summary.to_csv(CSV_DIR / "task_specific_within_task_summary.csv", index=False)
    task_effects.to_csv(CSV_DIR / "task_specific_within_task_effects.csv", index=False)

    generated = [
        path
        for path in [
            plot_ranked_language_specificity(summary),
            plot_strength_scatter(summary),
            plot_effect_heatmap(task_effects, summary, cell_meta),
        ]
        if path is not None
    ]
    write_report(summary, generated)
    update_index()

    run_summary = {
        "task_specific_feature_task_pairs": int(len(summary)),
        "comparable_pairs_with_at_least_two_languages": int((summary["valid_languages"] >= 2).sum()) if not summary.empty else 0,
        "generated": [str(path) for path in generated],
        "report": str(RESULT_DIR / "task_specific_within_task_visualisation_report.txt"),
    }
    (SUMMARY_DIR / "task_specific_within_task_visualisation_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
