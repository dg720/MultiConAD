import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.phase1.common import make_logger
from processing.phase2.common import PHASE2_ROOT, TABLES_PHASE2_ROOT


RICH_SWEEP_ROOT = TABLES_PHASE2_ROOT / "phase2-rich-sweep"
RICH_SWEEP_RESULT_TABLES = RICH_SWEEP_ROOT / "result-tables" / "csv"
RICH_SWEEP_SUMMARIES = RICH_SWEEP_ROOT / "summaries"
REPORT_ROOT = RICH_SWEEP_ROOT / "report_assets"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

PALETTE = {"AD": "#c44e52", "Dementia": "#c44e52", "HC": "#4c9f70"}
MARKERS = {"AD": "o", "Dementia": "o", "HC": "s"}


LANGUAGE_LABELS = {
    "en": "English PD_CTP",
    "zh": "Chinese PD_CTP",
    "el": "Greek PD_CTP",
    "es": "Spanish READING",
}

SLICE_CONFIG = {
    "en": {
        "filters": {"language": "en", "task_type": "PD_CTP"},
        "global_run": "pd_ctp_pooled_phase2",
        "mono_run": "language_en_pd_ctp_phase2",
        "global_label": "Pooled multilingual PD_CTP",
        "mono_label": "Best English PD_CTP",
    },
    "zh": {
        "filters": {"language": "zh", "task_type": "PD_CTP"},
        "global_run": "pd_ctp_pooled_phase2",
        "mono_run": "language_zh_pd_ctp_phase2",
        "global_label": "Pooled multilingual PD_CTP",
        "mono_label": "Best Chinese PD_CTP",
    },
    "el": {
        "filters": {"language": "el", "task_type": "PD_CTP"},
        "global_run": "pd_ctp_pooled_phase2",
        "mono_run": "language_el_pd_ctp_phase2",
        "global_label": "Pooled multilingual PD_CTP",
        "mono_label": "Best Greek PD_CTP",
    },
    "es": {
        "filters": {"language": "es", "task_type": "READING"},
        "global_run": "benchmark_wide_phase2",
        "mono_run": "language_es_reading_phase2",
        "global_label": "Benchmark-wide multilingual",
        "mono_label": "Best Spanish READING",
    },
}


def load_summary(run_name: str) -> dict[str, object]:
    return json.loads((RICH_SWEEP_SUMMARIES / f"{run_name}_summary.json").read_text(encoding="utf-8"))


def load_best_feature_names(run_name: str) -> list[str]:
    summary = load_summary(run_name)
    best = summary["best_model"]
    top_k = int(best["top_k"])
    ranking = pd.read_csv(RICH_SWEEP_RESULT_TABLES / f"{run_name}_anova_ranking.csv")
    return ranking["feature_name"].head(top_k).tolist()


def apply_filters(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for key, value in filters.items():
        out = out[out[key] == value]
    return out


def compute_tsne(feature_frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    usable_cols = [col for col in feature_cols if feature_frame[col].notna().any()]
    if not usable_cols:
        raise RuntimeError("No usable non-missing features for projection.")
    prep = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x = prep.fit_transform(feature_frame[usable_cols])
    n_samples = len(feature_frame)
    perplexity = min(30, max(5, (n_samples - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    coords = tsne.fit_transform(x)
    out = feature_frame.copy()
    out["tsne_x"] = coords[:, 0]
    out["tsne_y"] = coords[:, 1]
    out["usable_feature_count"] = len(usable_cols)
    return out


def build_projection_rows(df: pd.DataFrame, language: str) -> list[pd.DataFrame]:
    cfg = SLICE_CONFIG[language]
    slice_df = apply_filters(df, cfg["filters"]).copy()
    rows = []
    for mode, run_name, feature_label in [
        ("global", cfg["global_run"], cfg["global_label"]),
        ("monolingual", cfg["mono_run"], cfg["mono_label"]),
    ]:
        feature_cols = [col for col in load_best_feature_names(run_name) if col in slice_df.columns]
        if not feature_cols:
            continue
        projected = compute_tsne(
            slice_df[
                [
                    "sample_id",
                    "group_id",
                    "dataset_name",
                    "language",
                    "task_type",
                    "diagnosis_mapped",
                    "binary_label",
                ]
                + feature_cols
            ].copy(),
            feature_cols,
        )
        projected["comparison_mode"] = mode
        projected["feature_source"] = feature_label
        projected["feature_count"] = len(feature_cols)
        projected["language_label"] = LANGUAGE_LABELS[language]
        rows.append(projected)
    return rows


def save_individual_plots(projection_df: pd.DataFrame) -> None:
    for language in sorted(projection_df["language"].unique()):
        subset = projection_df[projection_df["language"] == language].copy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, mode in zip(axes, ["global", "monolingual"]):
            panel = subset[subset["comparison_mode"] == mode]
            for diagnosis in sorted(panel["diagnosis_mapped"].dropna().unique()):
                diag_df = panel[panel["diagnosis_mapped"] == diagnosis]
                ax.scatter(
                    diag_df["tsne_x"],
                    diag_df["tsne_y"],
                    label=diagnosis,
                    color=PALETTE.get(diagnosis, "#666666"),
                    marker=MARKERS.get(diagnosis, "o"),
                    alpha=0.8,
                    s=28,
                    edgecolors="none",
                )
            title_label = panel["feature_source"].iloc[0] if not panel.empty else mode
            ax.set_title(title_label)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.suptitle(f"{LANGUAGE_LABELS[language]} t-SNE with updated Phase 2 features", fontsize=14)
        fig.savefig(REPORT_ROOT / f"phase2_tsne_{language}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def save_grid_plot(projection_df: pd.DataFrame) -> None:
    languages = ["en", "zh", "el", "es"]
    fig, axes = plt.subplots(len(languages), 2, figsize=(14, 18), constrained_layout=True)
    for row_idx, language in enumerate(languages):
        lang_df = projection_df[projection_df["language"] == language]
        for col_idx, mode in enumerate(["global", "monolingual"]):
            ax = axes[row_idx, col_idx]
            panel = lang_df[lang_df["comparison_mode"] == mode]
            for diagnosis in sorted(panel["diagnosis_mapped"].dropna().unique()):
                diag_df = panel[panel["diagnosis_mapped"] == diagnosis]
                ax.scatter(
                    diag_df["tsne_x"],
                    diag_df["tsne_y"],
                    label=diagnosis,
                    color=PALETTE.get(diagnosis, "#666666"),
                    marker=MARKERS.get(diagnosis, "o"),
                    alpha=0.8,
                    s=24,
                    edgecolors="none",
                )
            source = panel["feature_source"].iloc[0] if not panel.empty else mode
            feature_count = int(panel["feature_count"].iloc[0]) if not panel.empty else 0
            ax.set_title(f"{LANGUAGE_LABELS[language]}: {source} ({feature_count} features)")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            if not (row_idx == 0 and col_idx == 1):
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
    legend = axes[0, 1].legend(frameon=False)
    if legend is not None:
        legend.set_title("")
    fig.suptitle("Phase 2 t-SNE comparison: multilingual vs monolingual updated feature sets", fontsize=16)
    fig.savefig(REPORT_ROOT / "phase2_language_tsne_comparison_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    log = make_logger("phase2_projection_artifacts")
    df = pd.read_csv(PHASE2_ROOT / "phase2_features.csv")
    rows = []
    for language in ["en", "zh", "el", "es"]:
        log(f"Building Phase 2 t-SNE projections for language={language}")
        rows.extend(build_projection_rows(df, language))
    projection_df = pd.concat(rows, ignore_index=True)
    projection_df.to_csv(REPORT_ROOT / "phase2_language_projection_comparisons.csv", index=False)
    save_individual_plots(projection_df)
    save_grid_plot(projection_df)
    log("Finished Phase 2 projection artifacts")


if __name__ == "__main__":
    main()
