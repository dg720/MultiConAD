from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tables" / "03-ablation-translingual-language-specific" / "language-task-visualisations"
CSV_DIR = OUT_ROOT / "result-tables" / "csv"
RESULT_DIR = OUT_ROOT / "result-tables"
ASSET_DIR = OUT_ROOT / "report_assets" / "tone-semitone-focus"

EFFECTS_PATH = CSV_DIR / "language_task_cell_effects.csv"
COVERAGE_PATH = CSV_DIR / "language_task_cell_coverage.csv"
SPECIFICITY_PATH = CSV_DIR / "language_vs_task_specificity_scores.csv"

TONE_FEATURES = [
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_amean",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_stddevnorm",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_percentile20_0",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_percentile50_0",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_percentile80_0",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_pctlrange0_2",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_meanrisingslope",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_stddevrisingslope",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_meanfallingslope",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_stddevfallingslope",
    "ac_f0_mean",
    "ac_f0_std",
    "ac_egemaps_logrelf0_h1_h2_sma3nz_amean",
    "ac_egemaps_logrelf0_h1_a3_sma3nz_amean",
]

LABEL_OVERRIDES = {
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_amean": "F0 semitone mean",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_stddevnorm": "F0 semitone variability",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_percentile20_0": "F0 semitone p20",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_percentile50_0": "F0 semitone median",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_percentile80_0": "F0 semitone p80",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_pctlrange0_2": "F0 semitone range",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_meanrisingslope": "F0 rising slope mean",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_stddevrisingslope": "F0 rising slope variability",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_meanfallingslope": "F0 falling slope mean",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_stddevfallingslope": "F0 falling slope variability",
    "ac_f0_mean": "F0 mean",
    "ac_f0_std": "F0 variability",
    "ac_egemaps_logrelf0_h1_h2_sma3nz_amean": "logRelF0 H1-H2",
    "ac_egemaps_logrelf0_h1_a3_sma3nz_amean": "logRelF0 H1-A3",
}

CELL_PRIORITY = [
    "zh|OTHER",
    "zh|PD_CTP",
    "en|PD_CTP",
    "el|PD_CTP",
    "en|COMMAND",
    "es|READING",
    "el|PICTURE_RECALL",
    "el|REPETITION",
]


def clean_label(value: str) -> str:
    return str(value).replace("\n", " ")


def ordered_cells(effects: pd.DataFrame, tone: pd.DataFrame) -> list[str]:
    valid_cells = (
        tone.loc[tone["valid_effect"].fillna(False), ["cell_id", "language", "task_type"]]
        .drop_duplicates()
        .copy()
    )
    if valid_cells.empty:
        valid_cells = effects[["cell_id", "language", "task_type"]].drop_duplicates().copy()
    all_cells = set(effects["cell_id"].unique())
    priority = [cell for cell in CELL_PRIORITY if cell in all_cells]
    remaining = sorted(cell for cell in valid_cells["cell_id"].unique() if cell not in priority)
    return priority + remaining


def ordered_features(tone: pd.DataFrame) -> list[str]:
    zh_other = tone[(tone["cell_id"] == "zh|OTHER") & tone["valid_effect"].fillna(False)]
    by_zh = (
        zh_other.assign(abs_d=zh_other["cohens_d"].abs())
        .sort_values(["welch_rank", "abs_d"], ascending=[True, False], na_position="last")
        ["feature_name"]
        .tolist()
    )
    remaining = [feature for feature in TONE_FEATURES if feature in set(tone["feature_name"]) and feature not in by_zh]
    return by_zh + remaining


def feature_labels(tone: pd.DataFrame) -> dict[str, str]:
    labels = {}
    for feature, group in tone.groupby("feature_name"):
        label = group["feature_label"].dropna().astype(str).iloc[0] if group["feature_label"].notna().any() else feature
        labels[feature] = LABEL_OVERRIDES.get(feature, label)
    return labels


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in (".png", ".pdf"):
        fig.savefig(ASSET_DIR / f"{stem}{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def plot_heatmap(tone: pd.DataFrame, cells: list[str], features: list[str], labels: dict[str, str]) -> None:
    selected = tone[tone["feature_name"].isin(features) & tone["cell_id"].isin(cells)].copy()
    matrix = selected.pivot_table(index="feature_name", columns="cell_id", values="cohens_d", aggfunc="first")
    matrix = matrix.reindex(index=features, columns=cells)
    display = matrix.to_numpy(dtype=float)

    cell_labels = {
        row["cell_id"]: clean_label(row["cell_label"])
        for _, row in selected[["cell_id", "cell_label"]].drop_duplicates().iterrows()
    }
    row_labels = [labels.get(feature, feature) for feature in features]
    col_labels = [cell_labels.get(cell, cell) for cell in cells]

    max_abs = np.nanmax(np.abs(display)) if np.isfinite(display).any() else 1.0
    max_abs = max(max_abs, 0.5)
    fig_w = max(12.0, 0.86 * len(cells) + 4.5)
    fig_h = max(7.5, 0.42 * len(features) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    image = ax.imshow(display, cmap=cmap, norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs), aspect="auto")

    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("Tone and Semitone Features Across Language-Task Cells", fontsize=13, pad=14)
    ax.set_xlabel("Language-task cell", labelpad=12)
    ax.set_ylabel("Forced-in tone/prosody feature")

    lookup = selected.set_index(["feature_name", "cell_id"])
    for y, feature in enumerate(features):
        for x, cell in enumerate(cells):
            value = matrix.loc[feature, cell]
            key = (feature, cell)
            if pd.isna(value):
                ax.text(x, y, "n/a", ha="center", va="center", fontsize=7, color="#777777")
                continue
            row = lookup.loc[key] if key in lookup.index else None
            suffix = ""
            if row is not None and bool(row.get("welch_significant_bonferroni", False)):
                suffix = "*"
            elif row is not None and bool(row.get("welch_top_50", False)):
                suffix = "."
            text_color = "white" if abs(float(value)) > max_abs * 0.55 else "#222222"
            ax.text(x, y, f"{float(value):.2f}{suffix}", ha="center", va="center", fontsize=7, color=text_color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cells), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(features), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("Signed Cohen's d, AD minus HC")
    fig.subplots_adjust(bottom=0.28)
    fig.text(
        0.01,
        0.035,
        "* = Welch Bonferroni significant; . = Welch top 50. Grey n/a means the feature was unavailable or the AD/HC cell was too small.",
        fontsize=8,
    )
    save_figure(fig, "tone_semitone_language_task_heatmap")


def plot_zh_other_bars(tone: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    zh = tone[(tone["cell_id"] == "zh|OTHER") & tone["valid_effect"].fillna(False)].copy()
    zh = zh[zh["feature_name"].isin(TONE_FEATURES)].copy()
    zh["feature_label_short"] = zh["feature_name"].map(labels)
    zh["abs_d"] = zh["cohens_d"].abs()
    zh = zh.sort_values(["welch_rank", "abs_d"], ascending=[True, False], na_position="last")

    fig_h = max(6.5, 0.42 * len(zh) + 1.4)
    fig, ax = plt.subplots(figsize=(11.8, fig_h))
    colors = np.where(zh["cohens_d"] >= 0, "#b74d4d", "#3f6fa8")
    y = np.arange(len(zh))
    ax.barh(y, zh["cohens_d"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(zh["feature_label_short"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Signed Cohen's d, AD minus HC")
    ax.set_title("Chinese NCMMSC/NCMMSE OTHER Tone and Semitone Effects", fontsize=13, pad=12)
    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-1.02, 1.04)
    text_x = 0.72
    for i, (_, row) in enumerate(zh.iterrows()):
        rank = int(row["welch_rank"]) if pd.notna(row["welch_rank"]) else -1
        p_adj = row["welch_bonferroni_p"]
        p_text = f"rank {rank}, p_adj={p_adj:.3g}" if pd.notna(p_adj) else f"rank {rank}"
        if bool(row["welch_significant_bonferroni"]):
            p_text += " *"
        ax.text(text_x, i, p_text, va="center", ha="left", fontsize=8)
    ax.text(text_x, -0.85, "Welch ranking", ha="left", va="bottom", fontsize=8, fontweight="bold")
    fig.text(
        0.01,
        0.035,
        "NCMMSC/NCMMSE is plotted under the live manifest task type OTHER because exact subtask boundaries are unavailable.",
        fontsize=8,
    )
    save_figure(fig, "tone_semitone_zh_other_ranked_effects")
    return zh


def plot_availability(tone: pd.DataFrame, cells: list[str]) -> pd.DataFrame:
    representative = "ac_egemaps_f0semitonefrom27_5hz_sma3nz_amean"
    availability = tone[tone["feature_name"] == representative].copy()
    if availability.empty:
        availability = tone.groupby(["cell_id", "cell_label"], as_index=False)[["n_ad", "n_hc"]].max()
    availability = availability.sort_values(
        "cell_id",
        key=lambda col: col.map({cell: i for i, cell in enumerate(cells)}).fillna(len(cells) + 1),
    )
    availability = availability.drop_duplicates("cell_id")

    labels = [clean_label(v) for v in availability["cell_label"]]
    x = np.arange(len(availability))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10.5, 0.85 * len(availability) + 3.5), 5.8))
    ax.bar(x - width / 2, availability["n_ad"], width=width, label="AD", color="#b74d4d")
    ax.bar(x + width / 2, availability["n_hc"], width=width, label="HC", color="#3f6fa8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Groups with non-missing representative semitone feature")
    ax.set_title("Tone/Semitone Availability by Language-Task Cell", fontsize=13, pad=12)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for i, (_, row) in enumerate(availability.iterrows()):
        ax.text(i - width / 2, row["n_ad"] + 1, str(int(row["n_ad"])), ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, row["n_hc"] + 1, str(int(row["n_hc"])), ha="center", va="bottom", fontsize=8)
    save_figure(fig, "tone_semitone_availability_by_cell")
    return availability


def plot_specificity_scatter(labels: dict[str, str]) -> pd.DataFrame:
    if not SPECIFICITY_PATH.exists():
        return pd.DataFrame()
    scores = pd.read_csv(SPECIFICITY_PATH)
    tone_scores = scores[scores["feature_name"].isin(TONE_FEATURES)].copy()
    if tone_scores.empty:
        return tone_scores

    fig, ax = plt.subplots(figsize=(10.5, 7.8))
    ax.scatter(
        scores["language_specificity"],
        scores["task_specificity_score"],
        s=22,
        color="#bdbdbd",
        alpha=0.25,
        linewidths=0,
        label="all features",
    )
    semitone_mask = tone_scores["feature_name"].str.contains("semitone", case=False, na=False)
    logrel_mask = tone_scores["feature_name"].str.contains("logrelf0", case=False, na=False)
    simple_f0_mask = ~(semitone_mask | logrel_mask)
    for mask, color, label in [
        (semitone_mask, "#d95f02", "semitone"),
        (logrel_mask, "#1b9e77", "logRelF0"),
        (simple_f0_mask, "#386cb0", "F0 summary"),
    ]:
        subset = tone_scores[mask]
        if subset.empty:
            continue
        ax.scatter(
            subset["language_specificity"],
            subset["task_specificity_score"],
            s=92,
            color=color,
            edgecolors="#222222",
            linewidths=0.6,
            alpha=0.95,
            label=label,
        )

    lang_thr = tone_scores["language_specificity_threshold"].dropna()
    task_thr = tone_scores["task_specificity_threshold"].dropna()
    if not lang_thr.empty:
        ax.axvline(float(lang_thr.iloc[0]), color="#777777", linestyle="--", linewidth=0.9)
    if not task_thr.empty:
        ax.axhline(float(task_thr.iloc[0]), color="#777777", linestyle="--", linewidth=0.9)

    labelled = (
        tone_scores.assign(label=tone_scores["feature_name"].map(labels))
        .sort_values(["diagnostic_strength", "language_specificity", "task_specificity_score"], ascending=False)
        .head(6)
    )
    offsets = [(6, 6), (6, -10), (8, 16), (-8, 10), (-8, -14), (10, 0)]
    for (offset_x, offset_y), (_, row) in zip(offsets, labelled.iterrows()):
        ax.annotate(
            row["label"],
            (row["language_specificity"], row["task_specificity_score"]),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8,
            ha="left" if offset_x >= 0 else "right",
            va="center",
            arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.6},
        )

    ax.set_xlabel("Language specificity: SD of signed AD-HC effect across languages")
    ax.set_ylabel("Task specificity: SD of signed AD-HC effect across task types")
    ax.set_title("Where Tone and Semitone Features Sit in Specificity Space", fontsize=13, pad=12)
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.text(
        0.01,
        0.025,
        "Points use fixed size. Tone features are forced and labelled over the full feature cloud.",
        fontsize=8,
    )
    save_figure(fig, "tone_semitone_specificity_scatter")

    numbered = tone_scores.assign(label=tone_scores["feature_name"].map(labels)).sort_values(
        ["diagnostic_strength", "language_specificity", "task_specificity_score"],
        ascending=False,
    )
    numbered["plot_id"] = np.arange(1, len(numbered) + 1)
    fig, ax = plt.subplots(figsize=(12.4, 7.8))
    ax.scatter(
        scores["language_specificity"],
        scores["task_specificity_score"],
        s=20,
        color="#c8c8c8",
        alpha=0.22,
        linewidths=0,
    )
    for mask, color, label in [
        (numbered["feature_name"].str.contains("semitone", case=False, na=False), "#d95f02", "semitone"),
        (numbered["feature_name"].str.contains("logrelf0", case=False, na=False), "#1b9e77", "logRelF0"),
        (
            ~numbered["feature_name"].str.contains("semitone|logrelf0", case=False, na=False),
            "#386cb0",
            "F0 summary",
        ),
    ]:
        subset = numbered[mask]
        if subset.empty:
            continue
        ax.scatter(
            subset["language_specificity"],
            subset["task_specificity_score"],
            s=115,
            color=color,
            edgecolors="#222222",
            linewidths=0.6,
            alpha=0.95,
            label=label,
        )
    for _, row in numbered.iterrows():
        ax.text(
            row["language_specificity"],
            row["task_specificity_score"],
            str(int(row["plot_id"])),
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
    if not lang_thr.empty:
        ax.axvline(float(lang_thr.iloc[0]), color="#777777", linestyle="--", linewidth=0.9)
    if not task_thr.empty:
        ax.axhline(float(task_thr.iloc[0]), color="#777777", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Language specificity: SD of signed AD-HC effect across languages")
    ax.set_ylabel("Task specificity: SD of signed AD-HC effect across task types")
    ax.set_title("Tone and Semitone Specificity Scatter With Numbered Key", fontsize=13, pad=12)
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    key_lines = ["Key, ordered by diagnostic strength"]
    for _, row in numbered.iterrows():
        key_lines.append(f"{int(row['plot_id'])}. {row['label']}")
    ax.text(
        1.02,
        0.98,
        "\n".join(key_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.45"},
    )
    fig.subplots_adjust(right=0.72)
    fig.text(0.01, 0.025, "Points use fixed size; color distinguishes semitone, logRelF0, and plain F0 summary features.", fontsize=8)
    save_figure(fig, "tone_semitone_specificity_scatter_numbered")
    return tone_scores


def write_outputs(
    tone: pd.DataFrame,
    zh: pd.DataFrame,
    availability: pd.DataFrame,
    tone_scores: pd.DataFrame,
    cells: list[str],
) -> None:
    export_cols = [
        "feature_name",
        "feature_label",
        "language",
        "task_type",
        "cell_id",
        "cell_label",
        "n_ad",
        "n_hc",
        "cohens_d",
        "welch_p",
        "welch_bonferroni_p",
        "welch_rank",
        "welch_significant_bonferroni",
        "welch_top_50",
    ]
    tone[export_cols].to_csv(CSV_DIR / "tone_semitone_focus_effects.csv", index=False)
    zh[export_cols + ["feature_label_short"]].to_csv(CSV_DIR / "tone_semitone_focus_zh_other_ranked.csv", index=False)
    availability.to_csv(CSV_DIR / "tone_semitone_focus_availability.csv", index=False)
    if not tone_scores.empty:
        tone_scores.to_csv(CSV_DIR / "tone_semitone_specificity_scores.csv", index=False)

    top_zh = zh.head(8).copy()
    lines = [
        "Tone/Semitone Focus Report",
        "===========================",
        "",
        "NCMMSC/NCMMSE task type",
        "----------------------",
        "In the paper, NCMMSE/NCMMSC is described as mixed-task audio: participants perform various tasks including picture descriptions and fluency exercises.",
        "In the live manifest, `NCMMSC2021_AD` is assigned `task_type = OTHER`, `task_type_source = lossy_long_audio_packaging`, and `task_type_confidence = low`.",
        "Interpretation: we should not treat the current NCMMSC acoustic/semitone signal as strict PD_CTP evidence. It is a mixed or unresolved Chinese source-level signal.",
        "",
        "Why semitone disappeared from some plots",
        "---------------------------------------",
        "The strict PD-only visualisations filter to `task_type == PD_CTP`. The strongest Chinese tone/prosody signal sits in `zh|OTHER`, not `zh|PD_CTP`.",
        "The tone-focused plots therefore force semitone/F0/logRelF0 features into a separate view so they can be interpreted with the source/task caveat visible.",
        "",
        "Chinese OTHER tone/prosody ranking",
        "----------------------------------",
    ]
    for _, row in top_zh.iterrows():
        star = " significant" if bool(row["welch_significant_bonferroni"]) else ""
        lines.append(
            f"- {row['feature_label_short']}: d={row['cohens_d']:.3f}, "
            f"Welch rank={int(row['welch_rank']) if pd.notna(row['welch_rank']) else 'n/a'}, "
            f"Bonferroni p={row['welch_bonferroni_p']:.4g}{star}."
        )
    lines += [
        "",
        "Generated files",
        "---------------",
        f"- {ASSET_DIR / 'tone_semitone_language_task_heatmap.png'}",
        f"- {ASSET_DIR / 'tone_semitone_zh_other_ranked_effects.png'}",
        f"- {ASSET_DIR / 'tone_semitone_availability_by_cell.png'}",
        f"- {ASSET_DIR / 'tone_semitone_specificity_scatter.png'}",
        f"- {ASSET_DIR / 'tone_semitone_specificity_scatter_numbered.png'}",
        f"- {CSV_DIR / 'tone_semitone_focus_effects.csv'}",
        f"- {CSV_DIR / 'tone_semitone_focus_zh_other_ranked.csv'}",
        f"- {CSV_DIR / 'tone_semitone_focus_availability.csv'}",
        f"- {CSV_DIR / 'tone_semitone_specificity_scores.csv'}",
        "",
        "Reading guide",
        "-------------",
        "The heatmap uses signed Cohen's d, AD minus HC. Blue means the feature is lower in AD; red means higher in AD.",
        "A star marks Welch Bonferroni significance and a dot marks Welch top-50 within the same language-task cell.",
        "The availability plot explains why Chinese PD_CTP is not comparable for these acoustic measures: the representative semitone feature has no usable AD groups there.",
        "",
        "Cell order used in heatmap",
        "--------------------------",
        ", ".join(cells),
        "",
    ]
    (RESULT_DIR / "tone_semitone_focus_report.txt").write_text("\n".join(lines), encoding="utf-8")


def update_index() -> None:
    index_path = RESULT_DIR / "plot_asset_folder_index.txt"
    current = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
    entry = dedent(
        """

        tone-semitone-focus/
        Forced-in tone, F0, and semitone plots:
        - all-task language-task heatmap
        - Chinese NCMMSC/NCMMSE OTHER ranked effects
        - semitone availability by language-task cell
        - fixed-point specificity scatter with tone features labelled
        """
    ).strip()
    if "tone-semitone-focus/" not in current:
        index_path.write_text(current.rstrip() + "\n\n" + entry + "\n", encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    effects = pd.read_csv(EFFECTS_PATH)
    tone = effects[effects["feature_name"].isin(TONE_FEATURES)].copy()
    if tone.empty:
        raise RuntimeError("No tone/semitone features found in language_task_cell_effects.csv")

    cells = ordered_cells(effects, tone)
    features = ordered_features(tone)
    labels = feature_labels(tone)

    plot_heatmap(tone, cells, features, labels)
    zh = plot_zh_other_bars(tone, labels)
    availability = plot_availability(tone, cells)
    tone_scores = plot_specificity_scatter(labels)
    write_outputs(tone, zh, availability, tone_scores, cells)
    update_index()

    print(f"Wrote tone/semitone focus plots to {ASSET_DIR}")
    print(f"Wrote report to {RESULT_DIR / 'tone_semitone_focus_report.txt'}")


if __name__ == "__main__":
    main()
