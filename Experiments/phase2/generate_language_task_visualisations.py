from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "phase2" / "phase2_features.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "phase2" / "phase2_feature_metadata.csv"
OUT_ROOT = (
    PROJECT_ROOT
    / "tables"
    / "03-ablation-translingual-language-specific"
    / "language-task-visualisations"
)
RESULT_DIR = OUT_ROOT / "result-tables"
CSV_DIR = RESULT_DIR / "csv"
ASSET_DIR = OUT_ROOT / "report_assets"
SUMMARY_DIR = OUT_ROOT / "summaries"

MIN_GROUPS_PER_CLASS = 5
ROBUST_GROUPS_PER_CLASS = 10
TOP_KS = [5, 10, 20, 50, 100]

ID_COLS = {
    "sample_id",
    "group_id",
    "dataset_name",
    "language",
    "task_type",
    "diagnosis_mapped",
    "binary_label",
}

LANG_LABELS = {"en": "EN", "es": "ES", "zh": "ZH", "el": "EL"}

SOURCE_TASK_LABELS = {
    "Pitt": ["PD"],
    "Lu": ["PD"],
    "VAS": ["FC"],
    "Baycrest": ["SR"],
    "Kempler": ["PD", "FT"],
    "WLS": ["PD", "FT"],
    "Delaware": ["PD", "NA"],
    "TAUKADIAL": ["PD"],
    "Ivanova": ["NA"],
    "PerLA": ["PD", "FT", "FC"],
    "NCMMSC2021_AD": ["PD", "FT", "FC"],
    "Predictive_Chinese_challenge_Chinese_2019": ["PD"],
    "ADReSS-M": ["PD"],
    "DS3": ["PD", "FT"],
    "DS5": ["PD", "FT"],
    "DS7": ["PD", "FT"],
}

MANIFEST_TASK_FALLBACK = {
    "PD_CTP": ["PD"],
    "READING": ["NA"],
    "COMMAND": ["FC"],
    "CONVERSATION": ["FC"],
    "PICTURE_RECALL": ["SR"],
    "REPETITION": ["FT"],
    "MOTOR_SPEECH": ["FT"],
    "PROCEDURAL": ["NA"],
    "MIXED_PROTOCOL": ["PD", "FT", "SR", "FC", "NA"],
    "OTHER": ["UNKNOWN"],
}

FEATURE_FAMILY_LABELS = {
    "len": "length/rate",
    "lex": "lexical",
    "pause": "pause/fluency",
    "disc": "discourse",
    "graph": "graph",
    "syn": "syntax",
    "sx": "syntax",
    "pr": "phrase",
    "ac": "acoustic",
    "par": "paralinguistic",
    "pd": "task:PD",
    "rd": "task:reading",
    "fc": "task:FC",
    "ft": "task:FT",
    "sr": "task:SR",
    "cmd": "task:command",
    "rep": "task:repetition",
    "ms": "task:motor",
    "na": "task:NA",
}

CLINICAL_GROUPS = {"pause", "lex", "syn", "sx", "pr", "ac", "par", "pd", "rd", "fc", "ft", "sr", "cmd", "rep", "ms", "na"}

TASK_LABEL_DISPLAY = {
    "PD": "PD",
    "FT": "FT",
    "SR": "SR",
    "FC": "FC",
    "NA": "Narrative (NA)",
    "UNKNOWN": "UNKNOWN",
}

DISPLAY_NAME_OVERRIDES = {
    "pause_short_count": "short pauses",
    "pause_medium_count": "medium pauses",
    "pause_long_count": "long pauses",
    "syn_noun_ratio": "noun ratio",
    "syn_noun_verb_ratio": "noun/verb ratio",
    "syn_subordination_ratio": "subordination ratio",
    "syn_obj_ratio": "object ratio",
    "lex_mattr_10": "MATTR 10",
    "lex_mattr_20": "MATTR 20",
    "lex_content_word_ratio": "content-word ratio",
    "lex_function_word_ratio": "function-word ratio",
    "lex_hapax_ratio": "hapax ratio",
    "disc_mean_pairwise_utt_similarity": "utterance similarity",
    "pd_unique_units_ratio": "PD unique units ratio",
    "pd_percentage_units_mentioned": "PD units mentioned",
    "pd_unique_units_count": "PD unique units count",
    "pd_num_unique_keywords": "PD unique keywords",
    "ac_f0_mean": "F0 mean",
    "ac_f0_std": "F0 variability",
    "par_f0_mean": "paralinguistic F0 mean",
    "par_f0_std": "paralinguistic F0 variability",
    "par_f0_iqr": "paralinguistic F0 IQR",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_amean": "F0 semitone mean",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_stddevnorm": "F0 semitone variability",
    "ac_egemaps_f0semitonefrom27_5hz_sma3nz_pctlrange0_2": "F0 semitone range",
    "ac_egemaps_logrelf0_h1_h2_sma3nz_amean": "logRelF0 H1-H2",
    "ac_egemaps_logrelf0_h1_a3_sma3nz_amean": "logRelF0 H1-A3",
    "ac_egemaps_jitterlocal_sma3nz_amean": "jitter",
    "ac_egemaps_shimmerlocaldb_sma3nz_amean": "shimmer",
    "ac_egemaps_loudness_sma3_percentile20_0": "loudness p20",
    "ac_egemaps_loudness_sma3_stddevrisingslope": "loudness rising slope",
}

TONAL_PROXY_PATTERNS = ("f0", "semitone", "logrelf0", "jitter", "shimmer", "loudness")
CATEGORY_ORDER = {
    "translingual candidate": 0,
    "language-sensitive candidate": 1,
    "task-sensitive": 2,
    "language-task confounded": 3,
    "unstable / insufficient evidence": 4,
}
CATEGORY_COLORS = {
    "translingual candidate": "#2b8cbe",
    "language-sensitive candidate": "#7b3294",
    "task-sensitive": "#d95f02",
    "language-task confounded": "#7570b3",
    "unstable / insufficient evidence": "#bdbdbd",
}


def ensure_dirs() -> None:
    for directory in [RESULT_DIR, CSV_DIR, ASSET_DIR, SUMMARY_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def split_valid_tasks(value: object) -> set[str]:
    text = str(value).strip()
    if not text or text.upper() == "ALL" or text.lower() == "nan":
        return {"ALL"}
    return {part.strip() for part in text.split("|") if part.strip()}


def task_labels_for_cell(datasets: list[str], task_type: str) -> list[str]:
    labels: set[str] = set()
    for dataset in datasets:
        labels.update(SOURCE_TASK_LABELS.get(dataset, []))
    if not labels:
        labels.update(MANIFEST_TASK_FALLBACK.get(task_type, ["UNKNOWN"]))
    ordered = ["PD", "FT", "SR", "FC", "NA", "UNKNOWN"]
    return [label for label in ordered if label in labels]


def short_feature_name(name: str, max_len: int = 34) -> str:
    if name in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[name]
    cleaned = name
    for prefix in ["len_", "lex_", "pause_", "disc_", "graph_", "syn_", "sx_", "pr_", "ac_", "par_", "pd_", "rd_", "fc_", "ft_", "sr_", "cmd_", "rep_", "ms_", "na_"]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    cleaned = cleaned.replace("_", " ")
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3].rstrip() + "..."
    return cleaned


def pooled_cohens_d(ad: np.ndarray, hc: np.ndarray) -> float:
    ad = ad[np.isfinite(ad)]
    hc = hc[np.isfinite(hc)]
    if len(ad) < MIN_GROUPS_PER_CLASS or len(hc) < MIN_GROUPS_PER_CLASS:
        return np.nan
    mean_diff = float(np.mean(ad) - np.mean(hc))
    var_ad = float(np.var(ad, ddof=1)) if len(ad) > 1 else 0.0
    var_hc = float(np.var(hc, ddof=1)) if len(hc) > 1 else 0.0
    denom_df = len(ad) + len(hc) - 2
    pooled_var = ((len(ad) - 1) * var_ad + (len(hc) - 1) * var_hc) / denom_df if denom_df > 0 else np.nan
    pooled_std = math.sqrt(max(pooled_var, 0.0)) if np.isfinite(pooled_var) else np.nan
    if not np.isfinite(pooled_std) or pooled_std <= 1e-12:
        return 0.0 if abs(mean_diff) <= 1e-12 else np.nan
    return mean_diff / pooled_std


def safe_welch(ad: np.ndarray, hc: np.ndarray) -> float:
    ad = ad[np.isfinite(ad)]
    hc = hc[np.isfinite(hc)]
    if len(ad) < MIN_GROUPS_PER_CLASS or len(hc) < MIN_GROUPS_PER_CLASS:
        return np.nan
    if np.nanstd(ad) <= 1e-12 and np.nanstd(hc) <= 1e-12:
        return 1.0 if abs(np.nanmean(ad) - np.nanmean(hc)) <= 1e-12 else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        p_value = stats.ttest_ind(ad, hc, equal_var=False, nan_policy="omit").pvalue
    return float(p_value) if np.isfinite(p_value) else np.nan


def safe_anova(ad: np.ndarray, hc: np.ndarray) -> float:
    ad = ad[np.isfinite(ad)]
    hc = hc[np.isfinite(hc)]
    if len(ad) < MIN_GROUPS_PER_CLASS or len(hc) < MIN_GROUPS_PER_CLASS:
        return np.nan
    if np.nanstd(ad) <= 1e-12 and np.nanstd(hc) <= 1e-12:
        return 1.0 if abs(np.nanmean(ad) - np.nanmean(hc)) <= 1e-12 else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        p_value = stats.f_oneway(ad, hc).pvalue
    return float(p_value) if np.isfinite(p_value) else np.nan


def load_grouped_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    features = pd.read_csv(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    metadata["task_specific_bool"] = metadata["task_specific"].map(boolish)
    metadata["valid_task_set"] = metadata["valid_task_types"].map(split_valid_tasks)

    feature_cols = [name for name in metadata["feature_name"].tolist() if name in features.columns]
    features = features[features["binary_label"].isin([0, 1])].copy()
    features["binary_label"] = features["binary_label"].astype(int)
    features["diagnosis_mapped"] = features["binary_label"].map({0: "HC", 1: "AD"})

    keys = ["group_id", "dataset_name", "language", "task_type", "binary_label", "diagnosis_mapped"]
    grouped = features[keys + feature_cols].groupby(keys, dropna=False, as_index=False).mean(numeric_only=True)
    return grouped, metadata, feature_cols


def build_cell_metadata(grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (language, task_type), cell in grouped.groupby(["language", "task_type"], sort=True):
        counts = cell.groupby("binary_label")["group_id"].nunique()
        datasets = sorted(cell["dataset_name"].dropna().astype(str).unique().tolist())
        paper_labels = task_labels_for_cell(datasets, task_type)
        lang_label = LANG_LABELS.get(str(language), str(language).upper())
        paper_short = "+".join(paper_labels)
        rows.append(
            {
                "language": language,
                "task_type": task_type,
                "cell_id": f"{language}|{task_type}",
                "cell_label": f"{lang_label} {paper_short}\n{task_type}",
                "paper_task_labels": "|".join(paper_labels),
                "paper_task_labels_display": "|".join(TASK_LABEL_DISPLAY.get(label, label) for label in paper_labels),
                "datasets": "|".join(datasets),
                "ad_groups": int(counts.get(1, 0)),
                "hc_groups": int(counts.get(0, 0)),
                "valid_cell_min5": int(counts.get(1, 0) >= MIN_GROUPS_PER_CLASS and counts.get(0, 0) >= MIN_GROUPS_PER_CLASS),
                "robust_cell_min10": int(counts.get(1, 0) >= ROBUST_GROUPS_PER_CLASS and counts.get(0, 0) >= ROBUST_GROUPS_PER_CLASS),
            }
        )
    return pd.DataFrame(rows)


def feature_applicable(meta_row: pd.Series, task_type: str) -> bool:
    valid = meta_row["valid_task_set"]
    return "ALL" in valid or task_type in valid


def task_specific_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def build_effects(grouped: pd.DataFrame, metadata: pd.DataFrame, feature_cols: list[str], cell_meta: pd.DataFrame) -> pd.DataFrame:
    meta_by_feature = metadata.set_index("feature_name")
    rows: list[dict[str, object]] = []

    for _, cell_row in cell_meta.iterrows():
        language = cell_row["language"]
        task_type = cell_row["task_type"]
        cell = grouped[(grouped["language"] == language) & (grouped["task_type"] == task_type)]
        ad_cell = cell[cell["binary_label"] == 1]
        hc_cell = cell[cell["binary_label"] == 0]

        for feature in feature_cols:
            meta_row = meta_by_feature.loc[feature]
            applicable = feature_applicable(meta_row, task_type)
            ad = pd.to_numeric(ad_cell[feature], errors="coerce").to_numpy(dtype=float) if applicable else np.array([])
            hc = pd.to_numeric(hc_cell[feature], errors="coerce").to_numpy(dtype=float) if applicable else np.array([])
            n_ad = int(np.isfinite(ad).sum())
            n_hc = int(np.isfinite(hc).sum())
            valid = applicable and n_ad >= MIN_GROUPS_PER_CLASS and n_hc >= MIN_GROUPS_PER_CLASS

            if valid:
                d_value = pooled_cohens_d(ad, hc)
                welch_p = safe_welch(ad, hc)
                anova_p = safe_anova(ad, hc)
                mean_ad = float(np.nanmean(ad))
                mean_hc = float(np.nanmean(hc))
            else:
                d_value = np.nan
                welch_p = np.nan
                anova_p = np.nan
                mean_ad = np.nan
                mean_hc = np.nan

            rows.append(
                {
                    "feature_name": feature,
                    "feature_label": short_feature_name(feature),
                    "feature_group": meta_row["feature_group"],
                    "feature_family": FEATURE_FAMILY_LABELS.get(str(meta_row["feature_group"]), str(meta_row["feature_group"])),
                    "task_specific": boolish(meta_row["task_specific"]),
                    "valid_task_types": meta_row["valid_task_types"],
                    "language": language,
                    "task_type": task_type,
                    "cell_id": cell_row["cell_id"],
                    "cell_label": cell_row["cell_label"],
                    "paper_task_labels": cell_row["paper_task_labels"],
                    "paper_task_labels_display": cell_row["paper_task_labels_display"],
                    "datasets": cell_row["datasets"],
                    "applicable": bool(applicable),
                    "valid_effect": bool(valid and np.isfinite(d_value)),
                    "min5_cell": bool(valid),
                    "robust_min10_cell": bool(cell_row["robust_cell_min10"]),
                    "n_ad": n_ad,
                    "n_hc": n_hc,
                    "mean_ad": mean_ad,
                    "mean_hc": mean_hc,
                    "cohens_d": float(d_value) if np.isfinite(d_value) else np.nan,
                    "cohens_d_abs": float(abs(d_value)) if np.isfinite(d_value) else np.nan,
                    "direction": "higher_in_AD" if np.isfinite(d_value) and d_value > 0 else ("lower_in_AD" if np.isfinite(d_value) and d_value < 0 else "none"),
                    "welch_p": welch_p,
                    "anova_p": anova_p,
                }
            )

    effects = pd.DataFrame(rows)
    effects = add_cell_rankings(effects)
    return effects


def add_cell_rankings(effects: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, cell in effects.groupby("cell_id", sort=False):
        cell = cell.copy()
        valid_welch = cell["valid_effect"] & cell["welch_p"].notna()
        valid_anova = cell["valid_effect"] & cell["anova_p"].notna()

        n_welch = int(valid_welch.sum())
        n_anova = int(valid_anova.sum())
        cell["welch_bonferroni_p"] = np.nan
        cell["anova_bonferroni_p"] = np.nan
        cell.loc[valid_welch, "welch_bonferroni_p"] = (cell.loc[valid_welch, "welch_p"] * max(n_welch, 1)).clip(upper=1.0)
        cell.loc[valid_anova, "anova_bonferroni_p"] = (cell.loc[valid_anova, "anova_p"] * max(n_anova, 1)).clip(upper=1.0)
        cell["welch_significant_bonferroni"] = cell["welch_bonferroni_p"] < 0.05
        cell["anova_significant_bonferroni"] = cell["anova_bonferroni_p"] < 0.05

        welch_order = cell[valid_welch].sort_values(
            ["welch_bonferroni_p", "welch_p", "cohens_d_abs"],
            ascending=[True, True, False],
            na_position="last",
        )
        anova_order = cell[valid_anova].sort_values(
            ["anova_p", "cohens_d_abs"],
            ascending=[True, False],
            na_position="last",
        )
        cell["welch_rank"] = np.nan
        cell["anova_rank"] = np.nan
        cell.loc[welch_order.index, "welch_rank"] = np.arange(1, len(welch_order) + 1)
        cell.loc[anova_order.index, "anova_rank"] = np.arange(1, len(anova_order) + 1)

        for top_k in TOP_KS:
            cell[f"welch_top_{top_k}"] = cell["welch_rank"] <= top_k
            cell[f"anova_top_{top_k}"] = cell["anova_rank"] <= top_k
        frames.append(cell)
    return pd.concat(frames, ignore_index=True)


def stability_from_rank(rank: float) -> int:
    if not np.isfinite(rank):
        return 0
    if rank <= 5:
        return 4
    if rank <= 10:
        return 3
    if rank <= 20:
        return 2
    if rank <= 50:
        return 1
    return 0


def classify_specificity_categories(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    if summary.empty:
        summary["interpretation_category"] = []
        summary["language_specificity_threshold"] = np.nan
        summary["task_specificity_threshold"] = np.nan
        return summary

    eligible = summary[(summary["valid_cells"] >= 2) & summary["direction_consistency"].fillna(0).ge(0.67)].copy()
    lang_threshold = float(eligible["language_specificity"].dropna().median()) if eligible["language_specificity"].notna().any() else 0.0
    task_threshold = float(eligible["task_specificity_score"].dropna().median()) if eligible["task_specificity_score"].notna().any() else 0.0

    categories = []
    for _, row in summary.iterrows():
        if row["valid_cells"] < 2 or row["valid_languages"] < 2 or not np.isfinite(row["direction_consistency"]) or row["direction_consistency"] < 0.67:
            categories.append("unstable / insufficient evidence")
            continue
        lang_high = float(row["language_specificity"]) > lang_threshold if np.isfinite(row["language_specificity"]) else False
        task_high = float(row["task_specificity_score"]) > task_threshold if np.isfinite(row["task_specificity_score"]) else False
        if not lang_high and not task_high:
            categories.append("translingual candidate")
        elif lang_high and not task_high:
            categories.append("language-sensitive candidate")
        elif not lang_high and task_high:
            categories.append("task-sensitive")
        else:
            categories.append("language-task confounded")

    summary["interpretation_category"] = categories
    summary["language_specificity_threshold"] = lang_threshold
    summary["task_specificity_threshold"] = task_threshold
    return summary


def build_feature_summary(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, group in effects.groupby("feature_name", sort=False):
        valid = group[group["valid_effect"]].copy()
        first = group.iloc[0]
        if valid.empty:
            rows.append(
                {
                    "feature_name": feature,
                    "feature_label": first["feature_label"],
                    "feature_group": first["feature_group"],
                    "feature_family": first["feature_family"],
                    "task_specific": first["task_specific"],
                    "diagnostic_strength": np.nan,
                    "direction_consistency": np.nan,
                    "language_specificity": np.nan,
                    "task_specificity_score": np.nan,
                    "valid_cells": 0,
                    "valid_languages": 0,
                    "valid_manifest_tasks": 0,
                    "valid_paper_task_labels": "",
                    "stability_score": 0,
                    "interpretation_category": "unstable / insufficient evidence",
                }
            )
            continue

        signs = np.sign(valid["cohens_d"].astype(float))
        signs = signs[signs != 0]
        direction_consistency = float(signs.value_counts(normalize=True).iloc[0]) if not signs.empty else np.nan
        lang_effects = valid.groupby("language")["cohens_d"].mean()
        task_effects = valid.groupby("task_type")["cohens_d"].mean()
        language_specificity = float(lang_effects.std(ddof=0)) if len(lang_effects) >= 2 else np.nan
        task_specificity = float(task_effects.std(ddof=0)) if len(task_effects) >= 2 else np.nan
        stability_score = int(valid["welch_rank"].map(stability_from_rank).sum())
        rows.append(
            {
                "feature_name": feature,
                "feature_label": first["feature_label"],
                "feature_group": first["feature_group"],
                "feature_family": first["feature_family"],
                "task_specific": first["task_specific"],
                "diagnostic_strength": float(valid["cohens_d_abs"].mean()),
                "direction_consistency": direction_consistency,
                "language_specificity": language_specificity,
                "task_specificity_score": task_specificity,
                "valid_cells": int(valid["cell_id"].nunique()),
                "valid_languages": int(valid["language"].nunique()),
                "valid_manifest_tasks": int(valid["task_type"].nunique()),
                "valid_paper_task_labels": "|".join(sorted(set("|".join(valid["paper_task_labels"]).split("|")) - {""})),
                "stability_score": stability_score,
            }
        )

    summary = classify_specificity_categories(pd.DataFrame(rows))
    return summary.sort_values(["stability_score", "diagnostic_strength"], ascending=[False, False]).reset_index(drop=True)


def general_only_summary(summary: pd.DataFrame) -> pd.DataFrame:
    general = summary[~task_specific_mask(summary["task_specific"])].copy()
    general = classify_specificity_categories(general)
    return general.sort_values(["stability_score", "diagnostic_strength"], ascending=[False, False]).reset_index(drop=True)


def plot_scatter(summary: pd.DataFrame) -> None:
    plot_df = summary[summary["valid_cells"] > 0].copy()
    plot_df["x"] = plot_df["task_specificity_score"].fillna(0.0)
    plot_df["y"] = plot_df["language_specificity"].fillna(0.0)
    plot_df["size"] = 35 + 360 * (plot_df["diagnostic_strength"].fillna(0) / max(plot_df["diagnostic_strength"].fillna(0).quantile(0.95), 1e-6)).clip(0, 1.2)
    plot_df["alpha"] = (0.28 + 0.62 * (plot_df["valid_cells"] / max(plot_df["valid_cells"].max(), 1))).clip(0.28, 0.9)

    groups = sorted(plot_df["feature_group"].dropna().unique().tolist())
    cmap = plt.get_cmap("tab20")
    colors = {group: cmap(i % 20) for i, group in enumerate(groups)}

    fig, ax = plt.subplots(figsize=(12.6, 8.4))
    for group, sub in plot_df.groupby("feature_group", sort=True):
        ax.scatter(
            sub["x"],
            sub["y"],
            s=sub["size"],
            c=[colors[group]],
            alpha=sub["alpha"].mean(),
            edgecolors=np.where(sub["direction_consistency"].fillna(0).ge(0.8), "#222222", "#ffffff"),
            linewidths=0.7,
            label=str(group),
        )

    lang_threshold = float(summary["language_specificity_threshold"].dropna().iloc[0])
    task_threshold = float(summary["task_specificity_threshold"].dropna().iloc[0])
    ax.axvline(task_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axhline(lang_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.5)

    xmax = max(plot_df["x"].quantile(0.99) * 1.25, task_threshold * 1.25, 0.2)
    ymax = max(plot_df["y"].quantile(0.99) * 1.25, lang_threshold * 1.25, 0.2)
    ax.set_xlim(-0.02 * xmax, xmax * 1.08)
    ax.set_ylim(-0.02 * ymax, ymax)

    region_box = {"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.58}
    ax.text(0.04 * xmax, 0.06 * ymax, "Translingual\ncandidates", fontsize=8.5, color="#555555", bbox=region_box)
    ax.text(task_threshold + 0.04 * xmax, 0.06 * ymax, "Task-sensitive", fontsize=8.5, color="#555555", bbox=region_box)
    ax.text(0.04 * xmax, lang_threshold + 0.06 * ymax, "Language-sensitive\ncandidates", fontsize=8.5, color="#555555", bbox=region_box)
    ax.text(task_threshold + 0.04 * xmax, lang_threshold + 0.06 * ymax, "Language-task\nconfounded", fontsize=8.5, color="#555555", bbox=region_box)

    label_candidates = pd.concat(
        [
            plot_df[plot_df["interpretation_category"] == "translingual candidate"].nlargest(3, "diagnostic_strength"),
            plot_df[plot_df["interpretation_category"] == "language-sensitive candidate"].nlargest(3, "language_specificity"),
            plot_df[plot_df["interpretation_category"] == "task-sensitive"].nlargest(3, "task_specificity_score"),
            plot_df[plot_df["interpretation_category"] == "language-task confounded"].nlargest(3, "diagnostic_strength"),
        ],
        ignore_index=True,
    ).drop_duplicates("feature_name")
    label_candidates = label_candidates[label_candidates["diagnostic_strength"] >= plot_df["diagnostic_strength"].quantile(0.65)]
    label_candidates = label_candidates.sort_values(["y", "x"], ascending=[False, True]).head(12)
    for idx, (_, row) in enumerate(label_candidates.iterrows()):
        x_offset = -8 if row["x"] > 0.82 * xmax else 5
        y_offset = -14 if row["y"] > 0.86 * ymax else 5 + (idx % 3) * 6
        ha = "right" if x_offset < 0 else "left"
        ax.annotate(
            row["feature_label"],
            (row["x"], row["y"]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=8,
            color="#222222",
            ha=ha,
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none", "alpha": 0.74},
        )

    ax.set_title("Language vs task specificity of AD/HC feature effects", fontsize=15, pad=14)
    ax.set_xlabel("Task specificity: std of signed AD/HC effects across manifest task labels")
    ax.set_ylabel("Language specificity: std of signed AD/HC effects across languages")
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Feature group", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8, title_fontsize=9)
    fig.text(
        0.01,
        0.01,
        "Point size = mean |Cohen's d|; alpha = evidence coverage; dark edge = >=80% direction consistency. "
        "Effects computed on group-aggregated AD vs HC cells with >=5 groups per class.",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 0.86, 1])
    fig.savefig(ASSET_DIR / "language_vs_task_specificity_scatter.png", dpi=220)
    fig.savefig(ASSET_DIR / "language_vs_task_specificity_scatter.pdf")
    plt.close(fig)


def scatter_common_frame(summary: pd.DataFrame) -> tuple[pd.DataFrame, float, float, float, float]:
    plot_df = summary[summary["valid_cells"] > 0].copy()
    plot_df["x"] = plot_df["task_specificity_score"].fillna(0.0)
    plot_df["y"] = plot_df["language_specificity"].fillna(0.0)
    lang_threshold = float(summary["language_specificity_threshold"].dropna().iloc[0])
    task_threshold = float(summary["task_specificity_threshold"].dropna().iloc[0])
    xmax = max(plot_df["x"].quantile(0.99) * 1.25, task_threshold * 1.25, 0.2)
    ymax = max(plot_df["y"].quantile(0.99) * 1.25, lang_threshold * 1.25, 0.2)
    return plot_df, task_threshold, lang_threshold, xmax, ymax


def annotate_points(ax, rows: pd.DataFrame, xmax: float, ymax: float, fontsize: float = 8.0) -> None:
    for idx, (_, row) in enumerate(rows.iterrows()):
        x_offset = -8 if row["x"] > 0.82 * xmax else 5
        y_offset = -14 if row["y"] > 0.86 * ymax else 5 + (idx % 3) * 5
        ax.annotate(
            row["feature_label"],
            (row["x"], row["y"]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="right" if x_offset < 0 else "left",
            fontsize=fontsize,
            color="#202020",
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.76},
        )


def plot_scatter_fixed_category(summary: pd.DataFrame) -> None:
    plot_df, task_threshold, lang_threshold, xmax, ymax = scatter_common_frame(summary)
    fig, ax = plt.subplots(figsize=(12.6, 8.2))
    for category, sub in plot_df.groupby("interpretation_category", sort=False):
        ax.scatter(
            sub["x"],
            sub["y"],
            s=52,
            c=CATEGORY_COLORS.get(category, "#777777"),
            alpha=0.58 if category != "unstable / insufficient evidence" else 0.22,
            edgecolors="#222222" if category != "unstable / insufficient evidence" else "none",
            linewidths=0.45,
            label=category,
        )
    ax.axvline(task_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.axhline(lang_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.set_xlim(-0.02 * xmax, xmax * 1.08)
    ax.set_ylim(-0.02 * ymax, ymax)

    curated = []
    for category in CATEGORY_ORDER:
        sub = plot_df[plot_df["interpretation_category"] == category].sort_values("diagnostic_strength", ascending=False).head(4)
        curated.append(sub)
    labels = pd.concat(curated, ignore_index=True).drop_duplicates("feature_name").head(18)
    annotate_points(ax, labels, xmax, ymax, fontsize=8.0)

    ax.set_title("Language vs task specificity: fixed-size feature points", fontsize=15, pad=14)
    ax.set_xlabel("Task specificity: std of signed AD/HC effects across manifest task labels")
    ax.set_ylabel("Language specificity: std of signed AD/HC effects across languages")
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.5, title="Interpretation")
    fig.text(
        0.01,
        0.012,
        "Fixed-size version: colour encodes interpretation category; labels show representative high-signal features in each region.",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 0.82, 1])
    fig.savefig(ASSET_DIR / "language_vs_task_specificity_scatter_fixed_category.png", dpi=220)
    fig.savefig(ASSET_DIR / "language_vs_task_specificity_scatter_fixed_category.pdf")
    plt.close(fig)


def plot_scatter_tonal_focus(summary: pd.DataFrame) -> None:
    plot_df, task_threshold, lang_threshold, xmax, ymax = scatter_common_frame(summary)
    plot_df["is_tonal_proxy"] = plot_df["feature_name"].str.contains("|".join(TONAL_PROXY_PATTERNS), case=False, regex=True)
    tonal = plot_df[plot_df["is_tonal_proxy"]].copy()

    fig, ax = plt.subplots(figsize=(12.4, 8.0))
    ax.scatter(plot_df["x"], plot_df["y"], s=34, c="#d0d0d0", alpha=0.28, edgecolors="none", label="other features")
    for category, sub in tonal.groupby("interpretation_category", sort=False):
        ax.scatter(
            sub["x"],
            sub["y"],
            s=78,
            c=CATEGORY_COLORS.get(category, "#777777"),
            alpha=0.82,
            edgecolors="#222222",
            linewidths=0.55,
            label=f"tonal/prosody: {category}",
        )
    ax.axvline(task_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.axhline(lang_threshold, color="#666666", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.set_xlim(-0.02 * xmax, xmax)
    ax.set_ylim(-0.02 * ymax, ymax)

    curated_tonal = [
        "ac_egemaps_loudness_sma3_percentile20_0",
        "ac_egemaps_loudness_sma3_stddevrisingslope",
        "ac_egemaps_logrelf0_h1_a3_sma3nz_amean",
        "ac_egemaps_logrelf0_h1_h2_sma3nz_amean",
        "ac_egemaps_f0semitonefrom27_5hz_sma3nz_amean",
        "ac_egemaps_f0semitonefrom27_5hz_sma3nz_stddevnorm",
        "ac_f0_mean",
        "ac_f0_std",
        "ac_egemaps_jitterlocal_sma3nz_amean",
        "ac_egemaps_shimmerlocaldb_sma3nz_amean",
    ]
    label_tonal = tonal[tonal["feature_name"].isin(curated_tonal)].copy()
    if len(label_tonal) < 9:
        label_tonal = pd.concat(
            [label_tonal, tonal.sort_values(["language_specificity", "diagnostic_strength"], ascending=[False, False]).head(9)],
            ignore_index=True,
        ).drop_duplicates("feature_name")
    label_tonal = label_tonal.sort_values(["language_specificity", "diagnostic_strength"], ascending=[False, False]).head(9)
    annotate_points(ax, label_tonal, xmax, ymax, fontsize=8.0)

    ax.set_title("Tonal/prosody proxy features in language-task specificity space", fontsize=15, pad=14)
    ax.set_xlabel("Task specificity")
    ax.set_ylabel("Language specificity")
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.0, title="Highlighted features")
    fig.text(
        0.01,
        0.012,
        "Highlighted tonal/prosody proxies include F0/semitone, logRelF0, jitter, shimmer, and loudness. They are not Chinese-only features in the current matrix.",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 0.78, 1])
    fig.savefig(ASSET_DIR / "language_vs_task_specificity_scatter_tonal_focus.png", dpi=220)
    fig.savefig(ASSET_DIR / "language_vs_task_specificity_scatter_tonal_focus.pdf")
    plt.close(fig)


def select_heatmap_features(effects: pd.DataFrame, summary: pd.DataFrame, max_features: int = 35) -> list[str]:
    valid = effects[effects["valid_effect"]].copy()
    top50_counts = valid[valid["welch_rank"] <= 50].groupby("feature_name")["cell_id"].nunique()
    top10_strong = valid[(valid["welch_rank"] <= 10) & (valid["cohens_d_abs"] > 0.5)]["feature_name"].unique().tolist()
    top100_clinical = valid[(valid["welch_rank"] <= 100) & (valid["feature_group"].isin(CLINICAL_GROUPS))]["feature_name"].unique().tolist()
    selected = set(top50_counts[top50_counts >= 2].index.tolist()) | set(top10_strong) | set(top100_clinical)

    ranked = summary[summary["feature_name"].isin(selected)].copy()
    ranked["category_order"] = ranked["interpretation_category"].map(CATEGORY_ORDER)
    ranked = ranked.sort_values(["category_order", "stability_score", "diagnostic_strength"], ascending=[True, False, False])

    # Keep the figure readable while preserving a mix of categories.
    keep: list[str] = []
    for category in ranked["interpretation_category"].drop_duplicates().tolist():
        keep.extend(ranked[ranked["interpretation_category"] == category].head(8)["feature_name"].tolist())
    for feature in ranked["feature_name"].tolist():
        if feature not in keep:
            keep.append(feature)
        if len(keep) >= max_features:
            break
    return keep[:max_features]


def plot_heatmap(effects: pd.DataFrame, summary: pd.DataFrame, cell_meta: pd.DataFrame) -> None:
    selected_features = select_heatmap_features(effects, summary)
    selected_summary = summary.set_index("feature_name").loc[selected_features]

    cell_order = (
        cell_meta[cell_meta["valid_cell_min5"] == 1]
        .sort_values(["task_type", "language"])
        ["cell_id"]
        .tolist()
    )
    cell_labels = cell_meta.set_index("cell_id").loc[cell_order, "cell_label"].tolist()

    matrix = effects[effects["feature_name"].isin(selected_features) & effects["cell_id"].isin(cell_order)].pivot(
        index="feature_name", columns="cell_id", values="cohens_d"
    )
    matrix = matrix.reindex(index=selected_features, columns=cell_order)

    category_prefix = {
        "translingual candidate": "TL",
        "language-sensitive candidate": "LS",
        "task-sensitive": "TS",
        "language-task confounded": "LTC",
        "unstable / insufficient evidence": "U",
    }
    row_categories = []
    y_labels = []
    for feature in selected_features:
        category = selected_summary.loc[feature, "interpretation_category"]
        row_categories.append(category)
        prefix = category_prefix.get(category, "?")
        y_labels.append(f"{prefix} | {short_feature_name(feature, 28)} [{selected_summary.loc[feature, 'feature_group']}]")

    abs_vals = np.abs(matrix.to_numpy(dtype=float))
    vmax = float(np.nanpercentile(abs_vals, 95)) if np.isfinite(abs_vals).any() else 1.0
    vmax = max(vmax, 0.8)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d8d8d8")

    height = max(8.5, 0.34 * len(selected_features) + 2.2)
    width = max(11.5, 1.08 * len(cell_order) + 5.0)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(cell_order)))
    ax.set_xticklabels(cell_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(selected_features)))
    ax.set_yticklabels(y_labels, fontsize=8.5)
    for y, category in enumerate(row_categories):
        ax.get_yticklabels()[y].set_color(CATEGORY_COLORS.get(category, "#222222"))

    effect_lookup = effects.set_index(["feature_name", "cell_id"])
    for y, feature in enumerate(selected_features):
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
                ax.text(x, y, marker, ha="center", va="center", color="#111111", fontsize=11, fontweight="bold")

    ax.set_title("Language-task recurrence of AD/HC feature effects, grouped by interpretation", fontsize=15, pad=14)
    ax.set_xlabel("Language and task cell: paper-style source label above live manifest task")
    ax.set_ylabel("Selected features")
    ax.set_xticks(np.arange(-0.5, len(cell_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(selected_features), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for y in range(1, len(row_categories)):
        if row_categories[y] != row_categories[y - 1]:
            ax.axhline(y - 0.5, color="#333333", linewidth=1.4)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Signed Cohen's d (AD - HC)")
    fig.text(
        0.01,
        0.012,
        "Row prefixes: TL=translingual, LS=language-sensitive, TS=task-sensitive, LTC=language-task confounded, U=unstable. "
        "* = Welch Bonferroni p < .05; . = Welch top-50. Red = higher in AD; blue = lower in AD.",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(ASSET_DIR / "language_task_feature_heatmap.png", dpi=220)
    fig.savefig(ASSET_DIR / "language_task_feature_heatmap.pdf")
    fig.savefig(ASSET_DIR / "language_task_feature_heatmap_category_ordered.png", dpi=220)
    fig.savefig(ASSET_DIR / "language_task_feature_heatmap_category_ordered.pdf")
    plt.close(fig)

    selected_summary.reset_index().to_csv(CSV_DIR / "language_task_heatmap_selected_features.csv", index=False)


def markdown_table(df: pd.DataFrame, cols: list[str], n: int = 12) -> str:
    if df.empty:
        return "(none)"
    view = df[cols].head(n).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    widths = {col: max(len(col), int(view[col].astype(str).map(len).max())) for col in view.columns}
    header = "  ".join(col.ljust(widths[col]) for col in view.columns)
    sep = "  ".join("-" * widths[col] for col in view.columns)
    lines = [header, sep]
    for _, row in view.iterrows():
        lines.append("  ".join(str(row[col]).ljust(widths[col]) for col in view.columns))
    return "\n".join(lines)


def write_report(cell_meta: pd.DataFrame, summary: pd.DataFrame, effects: pd.DataFrame) -> None:
    robust = cell_meta[cell_meta["robust_cell_min10"] == 1].copy()
    valid = cell_meta[cell_meta["valid_cell_min5"] == 1].copy()
    category_counts = summary["interpretation_category"].value_counts().rename_axis("category").reset_index(name="features")
    top_cols = [
        "feature_name",
        "feature_group",
        "diagnostic_strength",
        "language_specificity",
        "task_specificity_score",
        "direction_consistency",
        "valid_cells",
        "interpretation_category",
    ]
    tonal_summary = summary[
        summary["feature_name"].str.contains("|".join(TONAL_PROXY_PATTERNS), case=False, regex=True)
    ].sort_values(["language_specificity", "diagnostic_strength"], ascending=[False, False])

    lines = [
        "Language-Task Feature Visualisation Report",
        "==========================================",
        "",
        "Method",
        "------",
        "Feature values are group-aggregated before AD/HC comparisons so repeated rows do not overweight a participant/group.",
        f"Cell-level Cohen's d is computed only where at least {MIN_GROUPS_PER_CLASS} AD groups and {MIN_GROUPS_PER_CLASS} HC groups have non-missing values for the feature.",
        "Predictive top-k comparability is preserved separately: the visualisation records ANOVA ranks and Welch t-test + Bonferroni ranks using the same top-k checkpoints, but the figures are descriptive effect-size summaries.",
        "Paper-style task labels are retained for readability: PD=Picture Description, FT=Fluency Task, SR=Story Retelling, FC=Free Conversation, NA=Narrative. Live manifest labels are retained in parentheses/metadata.",
        "Pause features such as pause_short_count come from the Phase 1/2 audio pause extractor: RMS silence runs are counted as short pauses when duration is >=0.2 and <0.5 seconds.",
        "Tonal/prosody proxies in the current matrix are acoustic F0/semitone, logRelF0, jitter, shimmer, and loudness features. They are not encoded as Chinese-only task features.",
        "Task-specific feature families are excluded from these mixed language/task specificity figures. Their task specificity is an applicability property, not a cross-task effect-variance estimate.",
        "",
        "Valid live cells used in these figures (minimum 5 AD and 5 HC groups)",
        "------------------------------------------------------------------",
        markdown_table(valid, ["language", "task_type", "paper_task_labels_display", "ad_groups", "hc_groups", "datasets"], n=30),
        "",
        "More robust cells (minimum 10 AD and 10 HC groups)",
        "------------------------------------------------",
        markdown_table(robust, ["language", "task_type", "paper_task_labels_display", "ad_groups", "hc_groups", "datasets"], n=30),
        "",
        "Feature interpretation category counts",
        "--------------------------------------",
        markdown_table(category_counts, ["category", "features"], n=20),
        "",
        "Strongest translingual candidates",
        "--------------------------------",
        markdown_table(
            summary[summary["interpretation_category"] == "translingual candidate"].sort_values(
                ["diagnostic_strength", "stability_score"], ascending=[False, False]
            ),
            top_cols,
            n=15,
        ),
        "",
        "Strongest language-sensitive candidates",
        "---------------------------------------",
        markdown_table(
            summary[summary["interpretation_category"] == "language-sensitive candidate"].sort_values(
                ["language_specificity", "diagnostic_strength"], ascending=[False, False]
            ),
            top_cols,
            n=15,
        ),
        "",
        "Strongest task-sensitive candidates",
        "----------------------------------",
        markdown_table(
            summary[summary["interpretation_category"] == "task-sensitive"].sort_values(
                ["task_specificity_score", "diagnostic_strength"], ascending=[False, False]
            ),
            top_cols,
            n=15,
        ),
        "",
        "Most language-varying tonal/prosody proxy features",
        "-----------------------------------------------",
        markdown_table(tonal_summary, top_cols, n=15),
        "",
        "Important cautions",
        "------------------",
        "- Language and task are still partially confounded; a feature seen in only one language-task cell is not treated as language-sensitive.",
        "- Source-level coverage is broader than the live sample-level task labels; the plots show both where possible.",
        "- The heatmap is evidence-first: direction flips and one-cell signals should be interpreted as unstable/confounded rather than as biomarkers.",
        "- ASR-derived transcripts may contain combined participant/interviewer speech, and pause/acoustic features are not diarisation-aware.",
        "",
        "Generated files",
        "---------------",
        "- report_assets/language_vs_task_specificity_scatter.png",
        "- report_assets/language_vs_task_specificity_scatter.pdf",
        "- report_assets/language_vs_task_specificity_scatter_fixed_category.png",
        "- report_assets/language_vs_task_specificity_scatter_fixed_category.pdf",
        "- report_assets/language_vs_task_specificity_scatter_tonal_focus.png",
        "- report_assets/language_vs_task_specificity_scatter_tonal_focus.pdf",
        "- report_assets/language_task_feature_heatmap.png",
        "- report_assets/language_task_feature_heatmap.pdf",
        "- report_assets/language_task_feature_heatmap_category_ordered.png",
        "- report_assets/language_task_feature_heatmap_category_ordered.pdf",
        "- result-tables/csv/language_task_cell_effects.csv",
        "- result-tables/csv/language_vs_task_specificity_scores.csv",
        "- result-tables/csv/language_vs_task_specificity_scores_general_only.csv",
        "- result-tables/csv/language_task_cell_coverage.csv",
        "- result-tables/csv/language_task_heatmap_selected_features.csv",
    ]
    (RESULT_DIR / "language_task_visualisation_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    grouped, metadata, feature_cols = load_grouped_data()
    cell_meta = build_cell_metadata(grouped)
    effects = build_effects(grouped, metadata, feature_cols, cell_meta)
    summary = build_feature_summary(effects)
    plot_summary = general_only_summary(summary)
    plot_effects = effects[effects["feature_name"].isin(set(plot_summary["feature_name"]))].copy()

    cell_meta.to_csv(CSV_DIR / "language_task_cell_coverage.csv", index=False)
    effects.to_csv(CSV_DIR / "language_task_cell_effects.csv", index=False)
    summary.to_csv(CSV_DIR / "language_vs_task_specificity_scores.csv", index=False)
    plot_summary.to_csv(CSV_DIR / "language_vs_task_specificity_scores_general_only.csv", index=False)

    plot_scatter(plot_summary)
    plot_scatter_fixed_category(plot_summary)
    plot_scatter_tonal_focus(plot_summary)
    plot_heatmap(plot_effects, plot_summary, cell_meta)
    write_report(cell_meta, plot_summary, plot_effects)

    run_summary = {
        "features_path": str(FEATURES_PATH),
        "metadata_path": str(METADATA_PATH),
        "num_grouped_rows": int(len(grouped)),
        "num_features": int(len(feature_cols)),
        "num_general_features_plotted": int(len(plot_summary)),
        "num_task_specific_features_excluded_from_mixed_plots": int(task_specific_mask(summary["task_specific"]).sum()),
        "min_groups_per_class": MIN_GROUPS_PER_CLASS,
        "robust_groups_per_class": ROBUST_GROUPS_PER_CLASS,
        "valid_cells_min5": int(cell_meta["valid_cell_min5"].sum()),
        "robust_cells_min10": int(cell_meta["robust_cell_min10"].sum()),
        "outputs": {
            "report": str(RESULT_DIR / "language_task_visualisation_report.txt"),
            "scatter_png": str(ASSET_DIR / "language_vs_task_specificity_scatter.png"),
            "scatter_fixed_category_png": str(ASSET_DIR / "language_vs_task_specificity_scatter_fixed_category.png"),
            "scatter_tonal_focus_png": str(ASSET_DIR / "language_vs_task_specificity_scatter_tonal_focus.png"),
            "heatmap_png": str(ASSET_DIR / "language_task_feature_heatmap.png"),
            "heatmap_category_ordered_png": str(ASSET_DIR / "language_task_feature_heatmap_category_ordered.png"),
        },
    }
    (SUMMARY_DIR / "language_task_visualisation_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
