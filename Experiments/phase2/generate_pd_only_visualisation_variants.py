from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

import generate_language_task_visualisations as base
import generate_language_task_visualisations_v2 as v2


ASSET_DIR = base.ASSET_DIR / "pd-only-variants"
RESULT_DIR = base.RESULT_DIR
CSV_DIR = base.CSV_DIR
SUMMARY_DIR = base.SUMMARY_DIR

PD_TASK_TYPE = "PD_CTP"

SCOPES = {
    "pd_core": {
        "title": "PD-only core/general features",
        "description": "PD_CTP rows, excluding task-specific feature families.",
    },
    "pd_task_specific": {
        "title": "PD-only task-specific PD features",
        "description": "PD_CTP rows, using only PD-specific content/prompt features.",
    },
    "pd_full": {
        "title": "PD-only full feature set",
        "description": "PD_CTP rows, using all available general and task-specific features.",
    },
}

SELECTION_SPECS = {
    "translingual candidate": ("diagnostic_strength", 12),
    "language-sensitive candidate": ("language_specificity", 12),
    "task-sensitive": ("task_specificity_score", 8),
    "language-task confounded": ("diagnostic_strength", 8),
}

LANG_ORDER = ["en", "zh", "el", "es"]
LANG_LABELS = {"en": "English", "zh": "Chinese", "el": "Greek", "es": "Spanish"}


def ensure_dirs() -> None:
    for directory in [ASSET_DIR, RESULT_DIR, CSV_DIR, SUMMARY_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def metadata_with_helpers() -> pd.DataFrame:
    metadata = pd.read_csv(base.METADATA_PATH)
    metadata["task_specific_bool"] = metadata["task_specific"].map(base.boolish)
    metadata["valid_task_set"] = metadata["valid_task_types"].map(base.split_valid_tasks)
    return metadata


def load_pd_grouped(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    features = pd.read_csv(base.FEATURES_PATH)
    pd_all = features[features["task_type"] == PD_TASK_TYPE].copy()

    feature_cols = [name for name in metadata["feature_name"].tolist() if name in features.columns]
    binary = pd_all[pd_all["binary_label"].isin([0, 1])].copy()
    binary["binary_label"] = binary["binary_label"].astype(int)
    binary["diagnosis_mapped"] = binary["binary_label"].map({0: "HC", 1: "AD"})

    keys = ["group_id", "dataset_name", "language", "task_type", "binary_label", "diagnosis_mapped"]
    grouped = binary[keys + feature_cols].groupby(keys, dropna=False, as_index=False).mean(numeric_only=True)
    return grouped, pd_all, feature_cols, binary


def feature_names_for_scope(metadata: pd.DataFrame, feature_cols: list[str], scope: str) -> list[str]:
    available = metadata[metadata["feature_name"].isin(feature_cols)].copy()
    if scope == "pd_core":
        selected = available[~available["task_specific_bool"]]
    elif scope == "pd_task_specific":
        selected = available[(available["task_specific_bool"]) & (available["feature_group"] == "pd")]
    elif scope == "pd_full":
        selected = available
    else:
        raise ValueError(f"Unknown scope: {scope}")
    return selected["feature_name"].tolist()


def build_scope_analysis(
    grouped: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_meta = base.build_cell_metadata(grouped)
    effects = base.build_effects(grouped, metadata, feature_cols, cell_meta)
    summary = base.build_feature_summary(effects)
    summary["display_name"] = summary["feature_name"].map(lambda name: base.short_feature_name(name, 38))
    summary["category_order"] = summary["interpretation_category"].map(base.CATEGORY_ORDER).fillna(9)
    return effects, summary


def select_features(summary: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for category, (sort_col, limit) in SELECTION_SPECS.items():
        sub = summary[summary["interpretation_category"] == category].copy()
        if sub.empty:
            continue
        sub = sub.sort_values([sort_col, "diagnostic_strength", "stability_score"], ascending=[False, False, False]).head(limit)
        sub["category_label"] = v2.CATEGORY_TITLES.get(category, category)
        sub["category_prefix"] = v2.CATEGORY_PREFIX.get(category, "UN")
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=list(summary.columns) + ["category_label", "category_prefix", "family_color"])
    selected = pd.concat(frames, ignore_index=True)
    selected["family_color"] = selected["feature_group"].map(v2.FAMILY_COLORS).fillna("#999999")
    return selected


def fmt_score(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.2f}"


def language_counts(pd_all: pd.DataFrame, binary: pd.DataFrame, grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_diag = pd_all.groupby(["language", "diagnosis_mapped"]).agg(samples=("sample_id", "count"), groups=("group_id", "nunique")).reset_index()
    binary_diag = binary.groupby(["language", "diagnosis_mapped"]).agg(binary_samples=("sample_id", "count"), binary_groups=("group_id", "nunique")).reset_index()
    grouped_diag = grouped.groupby(["language", "diagnosis_mapped"]).agg(analysis_groups=("group_id", "nunique")).reset_index()
    for language in LANG_ORDER:
        row = {"language": language, "language_name": LANG_LABELS[language]}
        for diagnosis in ["AD", "HC", "MCI"]:
            match = all_diag[(all_diag["language"] == language) & (all_diag["diagnosis_mapped"] == diagnosis)]
            row[f"{diagnosis}_samples_all_dx"] = int(match["samples"].iloc[0]) if not match.empty else 0
            row[f"{diagnosis}_groups_all_dx"] = int(match["groups"].iloc[0]) if not match.empty else 0
        for diagnosis in ["AD", "HC"]:
            match = binary_diag[(binary_diag["language"] == language) & (binary_diag["diagnosis_mapped"] == diagnosis)]
            row[f"{diagnosis}_samples_used"] = int(match["binary_samples"].iloc[0]) if not match.empty else 0
            row[f"{diagnosis}_groups_used"] = int(match["binary_groups"].iloc[0]) if not match.empty else 0
            grouped_match = grouped_diag[(grouped_diag["language"] == language) & (grouped_diag["diagnosis_mapped"] == diagnosis)]
            row[f"{diagnosis}_analysis_groups"] = int(grouped_match["analysis_groups"].iloc[0]) if not grouped_match.empty else 0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_ranked(selected: pd.DataFrame, scope: str, title: str) -> Path | None:
    if selected.empty:
        return None
    plot_df = selected.sort_values(["category_order", "diagnostic_strength"], ascending=[True, False]).reset_index(drop=True)
    y = np.arange(len(plot_df))
    max_strength = max(float(plot_df["diagnostic_strength"].max()), 0.1)

    fig, ax = plt.subplots(figsize=(12.8, max(6.4, len(plot_df) * 0.34 + 1.8)))
    ax.barh(y, plot_df["diagnostic_strength"], color=plot_df["family_color"], alpha=0.78, edgecolor="#333333", linewidth=0.35)
    label_x = max_strength * 1.04
    for idx, row in plot_df.iterrows():
        ax.text(
            label_x,
            idx,
            f"{row['category_prefix']} | {row['feature_group']} | "
            f"L={fmt_score(row['language_specificity'])} "
            f"dir={row['direction_consistency']:.2f} "
            f"cells={int(row['valid_cells'])}",
            va="center",
            fontsize=8.1,
        )

    prev_category = None
    for idx, row in plot_df.iterrows():
        if prev_category is not None and row["interpretation_category"] != prev_category:
            ax.axhline(idx - 0.5, color="#333333", linewidth=1.1)
        prev_category = row["interpretation_category"]

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_name"], fontsize=8.8)
    ax.invert_yaxis()
    ax.set_xlabel("Diagnostic strength: mean |Cohen's d| across PD language cells")
    ax.set_title(f"{title}: ranked candidates", fontsize=14, pad=12)
    ax.set_xlim(0, max_strength * 1.75)
    ax.grid(axis="x", color="#dddddd", alpha=0.75)
    handles = [
        Patch(facecolor=color, edgecolor="#333333", label=family)
        for family, color in v2.FAMILY_COLORS.items()
        if family in set(plot_df["feature_group"])
    ]
    ax.legend(handles=handles, title="Feature family", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8, title_fontsize=9)
    fig.text(0.01, 0.012, "PD-only: task specificity is undefined because only PD_CTP cells are included.", fontsize=8.4, color="#444444")
    fig.tight_layout(rect=[0, 0.04, 0.84, 1])
    out = ASSET_DIR / f"ranked_candidates_{scope}.png"
    fig.savefig(out, dpi=220)
    fig.savefig(ASSET_DIR / f"ranked_candidates_{scope}.pdf")
    plt.close(fig)
    return out


def plot_language_heatmap(selected: pd.DataFrame, effects: pd.DataFrame, scope: str, title: str) -> Path | None:
    if selected.empty:
        return None
    cell_order = [f"{language}|{PD_TASK_TYPE}" for language in ["en", "zh", "el"]]
    cell_labels = ["English\nPD_CTP", "Chinese\nPD_CTP", "Greek\nPD_CTP"]
    effect_subset = effects[effects["feature_name"].isin(selected["feature_name"])].copy()
    abs_vals = np.abs(effect_subset["cohens_d"].to_numpy(dtype=float))
    vmax = float(np.nanpercentile(abs_vals, 95)) if np.isfinite(abs_vals).any() else 1.0
    vmax = max(vmax, 0.6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d8d8d8")

    frames = []
    heights = []
    for category in v2.CATEGORIES:
        cat_features = selected[selected["interpretation_category"] == category]["feature_name"].tolist()
        if not cat_features:
            continue
        matrix = effect_subset[effect_subset["feature_name"].isin(cat_features)].pivot(index="feature_name", columns="cell_id", values="cohens_d")
        matrix = matrix.reindex(index=cat_features, columns=cell_order)
        frames.append((category, cat_features, matrix))
        heights.append(max(1.3, 0.36 * len(cat_features) + 0.65))
    if not frames:
        return None

    fig, axes = plt.subplots(len(frames), 1, figsize=(8.8, sum(heights) + 1.9), gridspec_kw={"height_ratios": heights}, sharex=True)
    if len(frames) == 1:
        axes = [axes]
    effect_lookup = effects.set_index(["feature_name", "cell_id"])
    selected_lookup = selected.set_index("feature_name")
    image = None
    for ax, (category, features, matrix) in zip(axes, frames):
        image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, norm=norm)
        labels = [f"{base.short_feature_name(feature, 31)} [{selected_lookup.loc[feature, 'feature_group']}]" for feature in features]
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(labels, fontsize=8.6)
        ax.tick_params(axis="y", colors=base.CATEGORY_COLORS.get(category, "#222222"))
        ax.set_title(v2.CATEGORY_TITLES.get(category, category), loc="left", fontsize=11, color=base.CATEGORY_COLORS.get(category, "#222222"), pad=5)
        ax.set_xticks(np.arange(-0.5, len(cell_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(features), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for y_idx, feature in enumerate(features):
            for x_idx, cell_id in enumerate(cell_order):
                if (feature, cell_id) not in effect_lookup.index:
                    continue
                row = effect_lookup.loc[(feature, cell_id)]
                marker = "*" if bool(row.get("welch_significant_bonferroni", False)) else ("." if pd.notna(row.get("welch_rank")) and float(row["welch_rank"]) <= 50 else "")
                if marker:
                    ax.text(x_idx, y_idx, marker, ha="center", va="center", color="#111111", fontsize=10, fontweight="bold")

    axes[-1].set_xticks(np.arange(len(cell_order)))
    axes[-1].set_xticklabels(cell_labels, fontsize=9)
    fig.suptitle(f"{title}: signed AD/HC effects", fontsize=14, y=0.995)
    if image is not None:
        cax = fig.add_axes([0.89, 0.18, 0.018, 0.62])
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label("Signed Cohen's d (AD - HC)")
    fig.text(0.01, 0.012, "* = Welch Bonferroni p < .05; . = Welch top-50 in that PD cell. Grey = no valid PD cell/evidence.", fontsize=8.2, color="#444444")
    fig.subplots_adjust(left=0.32, right=0.86, top=0.94, bottom=0.12, hspace=0.42)
    out = ASSET_DIR / f"language_effect_heatmap_{scope}.png"
    fig.savefig(out, dpi=220)
    fig.savefig(ASSET_DIR / f"language_effect_heatmap_{scope}.pdf")
    plt.close(fig)
    return out


def plot_language_profiles(selected: pd.DataFrame, effects: pd.DataFrame, scope: str, title: str) -> Path | None:
    if selected.empty:
        return None
    plot_features = selected.sort_values(["category_order", "diagnostic_strength"], ascending=[True, False]).head(24)
    rows = []
    for _, feature_row in plot_features.iterrows():
        feature = feature_row["feature_name"]
        for language in ["en", "zh", "el"]:
            match = effects[(effects["feature_name"] == feature) & (effects["language"] == language) & (effects["task_type"] == PD_TASK_TYPE)]
            value = float(match["cohens_d"].iloc[0]) if not match.empty and pd.notna(match["cohens_d"].iloc[0]) else np.nan
            rows.append({"feature_name": feature, "display_name": feature_row["display_name"], "language": language, "cohens_d": value, "category": feature_row["interpretation_category"]})
    df = pd.DataFrame(rows)
    y_labels = plot_features["display_name"].tolist()
    y_pos = np.arange(len(y_labels))
    lang_offsets = {"en": -0.18, "zh": 0.0, "el": 0.18}
    lang_colors = {"en": "#4e79a7", "zh": "#f28e2b", "el": "#59a14f"}
    finite = df["cohens_d"].dropna()
    if finite.empty:
        return None
    xlim = max(abs(float(finite.quantile(0.05))), abs(float(finite.quantile(0.95))), 0.6) * 1.25

    fig, ax = plt.subplots(figsize=(10.4, max(7.0, len(y_labels) * 0.32 + 1.5)))
    for y_idx, feature in enumerate(plot_features["feature_name"]):
        sub = df[df["feature_name"] == feature]
        vals = sub.dropna(subset=["cohens_d"])
        if len(vals) >= 2:
            ax.plot(vals["cohens_d"], [y_idx + lang_offsets[lang] for lang in vals["language"]], color="#bbbbbb", linewidth=0.7, zorder=1)
        for _, row in vals.iterrows():
            ax.scatter(row["cohens_d"], y_idx + lang_offsets[row["language"]], color=lang_colors[row["language"]], s=42, edgecolor="white", linewidth=0.35, zorder=3, label=LANG_LABELS[row["language"]])
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Language")
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(-xlim, xlim)
    ax.set_xlabel("Signed Cohen's d (AD - HC)")
    ax.set_title(f"{title}: language effect profiles", fontsize=14, pad=12)
    ax.grid(axis="x", color="#dddddd", alpha=0.75)
    fig.text(0.01, 0.012, "Connected dots show whether the AD/HC effect direction and magnitude align across PD cells.", fontsize=8.2, color="#444444")
    fig.tight_layout(rect=[0, 0.04, 0.84, 1])
    out = ASSET_DIR / f"language_effect_profiles_{scope}.png"
    fig.savefig(out, dpi=220)
    fig.savefig(ASSET_DIR / f"language_effect_profiles_{scope}.pdf")
    plt.close(fig)
    return out


def plot_strength_language(summary: pd.DataFrame, selected: pd.DataFrame, scope: str, title: str) -> Path | None:
    plot_df = summary[(summary["valid_cells"] >= 1) & summary["diagnostic_strength"].notna()].copy()
    if plot_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    for category in v2.CATEGORIES + ["unstable / insufficient evidence"]:
        sub = plot_df[plot_df["interpretation_category"] == category]
        if sub.empty:
            continue
        ax.scatter(
            sub["language_specificity"].fillna(0.0),
            sub["diagnostic_strength"],
            s=38,
            color=base.CATEGORY_COLORS.get(category, "#999999"),
            alpha=0.55,
            edgecolor="white",
            linewidth=0.35,
            label=category,
        )
    label_df = selected.sort_values(["category_order", "diagnostic_strength"], ascending=[True, False]).groupby("interpretation_category", group_keys=False).head(3).head(10)
    for idx, (_, row) in enumerate(label_df.iterrows()):
        ax.text(
            (0.0 if pd.isna(row["language_specificity"]) else float(row["language_specificity"])) + 0.01,
            float(row["diagnostic_strength"]) + 0.015 + (idx % 3) * 0.012,
            base.short_feature_name(row["feature_name"], 22),
            fontsize=7.2,
        )
    ax.set_xlabel("Language specificity: std of signed language-averaged effects")
    ax.set_ylabel("Diagnostic strength: mean |Cohen's d|")
    ax.set_title(f"{title}: strength vs language specificity", fontsize=14, pad=12)
    ax.grid(color="#dddddd", alpha=0.65)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7.4)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    out = ASSET_DIR / f"strength_vs_language_specificity_{scope}.png"
    fig.savefig(out, dpi=220)
    fig.savefig(ASSET_DIR / f"strength_vs_language_specificity_{scope}.pdf")
    plt.close(fig)
    return out


def category_counts(summary: pd.DataFrame) -> dict[str, int]:
    counts = summary["interpretation_category"].value_counts().to_dict()
    return {
        "translingual": int(counts.get("translingual candidate", 0)),
        "language_sensitive": int(counts.get("language-sensitive candidate", 0)),
        "task_sensitive": int(counts.get("task-sensitive", 0)),
        "language_task_confounded": int(counts.get("language-task confounded", 0)),
        "unstable": int(counts.get("unstable / insufficient evidence", 0)),
    }


def write_report(
    count_df: pd.DataFrame,
    scope_rows: list[dict],
    selected_by_scope: dict[str, pd.DataFrame],
    generated: dict[str, list[str]],
) -> None:
    lines = [
        "PD-Only Visualisation Variant Review",
        "====================================",
        "",
        "Definition",
        "----------",
        "PD-only here means strict live manifest `task_type == PD_CTP`. This is more comparable than source-level PD labels because all plotted rows come from the same live task convention.",
        "Spanish has no PD_CTP rows in the current Phase 2 matrix, so it contributes zero samples/groups to these variants.",
        "",
        "Data points left",
        "----------------",
        "Rows are sample rows in `phase2_features.csv`. Analysis groups are unique group_id values after filtering to AD/HC binary labels.",
        count_df_for_report(count_df),
        "",
        "Plot review",
        "-----------",
        "The clearest separation comes from `language_effect_profiles_pd_core` and `language_effect_heatmap_pd_core`: with task fixed to PD_CTP, feature direction differences across English, Chinese, and Greek are immediately visible.",
        "The PD task-specific plots are clinically interpretable but narrow: selected rows are all PD-content/prompt features, and most have only 2-3 valid language cells. They are useful as a PD-feature audit, not as broad translingual evidence.",
        "The PD full-feature plots are useful as an appendix bridge between the core-only and full-feature analyses. They show the same acoustic/loudness confounding tendency but without cross-task variation.",
        "",
        "Recommendation",
        "--------------",
        "Use PD-only core/general as the main comparability figure if the claim is about language robustness under a fixed task.",
        "Use PD-only task-specific as a supporting audit of picture-description content features.",
        "Keep PD-only full feature set as appendix/context, because it mixes general and PD-specific signals.",
        "",
        "Scope summary",
        "-------------",
        pd.DataFrame(scope_rows).to_string(index=False),
    ]
    for scope, selected in selected_by_scope.items():
        lines.extend(["", f"Selected rows: {scope}", "-" * (15 + len(scope))])
        cols = [
            "category_prefix",
            "display_name",
            "feature_name",
            "feature_group",
            "diagnostic_strength",
            "language_specificity",
            "direction_consistency",
            "valid_cells",
        ]
        report_df = selected[cols].copy()
        for col in ["diagnostic_strength", "language_specificity", "direction_consistency"]:
            report_df[col] = report_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        lines.append(report_df.to_string(index=False))
        lines.extend(["", "Generated figures"])
        lines.extend(f"- report_assets/pd-only-variants/{Path(path).name}" for path in generated[scope])

    (RESULT_DIR / "pd_only_visualisation_variant_review.txt").write_text("\n".join(lines), encoding="utf-8")


def count_df_for_report(count_df: pd.DataFrame) -> str:
    display_cols = [
        "language_name",
        "AD_samples_used",
        "HC_samples_used",
        "AD_analysis_groups",
        "HC_analysis_groups",
        "MCI_samples_all_dx",
        "MCI_groups_all_dx",
    ]
    return count_df[display_cols].to_string(index=False)


def main() -> None:
    ensure_dirs()
    metadata = metadata_with_helpers()
    grouped, pd_all, feature_cols, binary = load_pd_grouped(metadata)
    count_df = language_counts(pd_all, binary, grouped)
    count_df.to_csv(CSV_DIR / "pd_only_language_counts.csv", index=False)

    generated: dict[str, list[str]] = {}
    selected_by_scope: dict[str, pd.DataFrame] = {}
    scope_rows: list[dict] = []

    for scope, info in SCOPES.items():
        scoped_features = feature_names_for_scope(metadata, feature_cols, scope)
        effects, summary = build_scope_analysis(grouped, metadata, scoped_features)
        selected = select_features(summary)
        selected_by_scope[scope] = selected
        effects.to_csv(CSV_DIR / f"pd_only_{scope}_cell_effects.csv", index=False)
        summary.to_csv(CSV_DIR / f"pd_only_{scope}_feature_summary.csv", index=False)
        selected.to_csv(CSV_DIR / f"pd_only_{scope}_selected_features.csv", index=False)

        counts = category_counts(summary)
        scope_rows.append(
            {
                "scope": scope,
                "features": int(len(summary)),
                "selected": int(len(selected)),
                **counts,
            }
        )
        title = info["title"]
        paths = [
            plot_ranked(selected, scope, title),
            plot_language_heatmap(selected, effects, scope, title),
            plot_language_profiles(selected, effects, scope, title),
            plot_strength_language(summary, selected, scope, title),
        ]
        generated[scope] = [str(path) for path in paths if path is not None]

    pd.DataFrame(scope_rows).to_csv(CSV_DIR / "pd_only_scope_variant_summary.csv", index=False)
    write_report(count_df, scope_rows, selected_by_scope, generated)
    run_summary = {
        "pd_task_type": PD_TASK_TYPE,
        "language_counts": count_df.to_dict(orient="records"),
        "scopes": scope_rows,
        "generated": generated,
        "report": str(RESULT_DIR / "pd_only_visualisation_variant_review.txt"),
    }
    (SUMMARY_DIR / "pd_only_visualisation_variant_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
