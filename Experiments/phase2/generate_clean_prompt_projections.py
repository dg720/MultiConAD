import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.phase1.common import make_logger
from processing.phase2.common import PHASE2_ROOT, TABLES_PHASE2_ROOT


RICH_ROOT = TABLES_PHASE2_ROOT / "rich_sweep"
CLEAN_ROOT = TABLES_PHASE2_ROOT / "clean_prompt_sweep"
REPORT_ROOT = CLEAN_ROOT / "report_assets"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

DIAG_COLORS = {"AD": "#c44e52", "HC": "#4c9f70"}
LANG_MARKERS = {"en": "o", "zh": "s", "el": "^", "es": "D"}
DATASET_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

SLICE_CONFIG = [
    {
        "slug": "en_cookie_theft",
        "title": "English Cookie Theft",
        "task_slug": "cookie_theft",
        "task_label": "Cookie Theft",
        "language": "en",
        "language_label": "English",
        "filters": {"language": "en", "phase2_prompt_family": "cookie_theft", "task_type": "PD_CTP"},
        "global_run_root": CLEAN_ROOT,
        "global_run_name": "cookie_theft_pooled_clean_phase2",
        "global_label": "Pooled multilingual Cookie Theft features",
        "local_run_root": CLEAN_ROOT,
        "local_run_name": "language_en_cookie_theft_clean_phase2",
        "local_label": "Best English Cookie Theft features",
    },
    {
        "slug": "zh_cookie_theft",
        "title": "Chinese Cookie Theft",
        "task_slug": "cookie_theft",
        "task_label": "Cookie Theft",
        "language": "zh",
        "language_label": "Chinese",
        "filters": {"language": "zh", "phase2_prompt_family": "cookie_theft", "task_type": "PD_CTP"},
        "global_run_root": CLEAN_ROOT,
        "global_run_name": "cookie_theft_pooled_clean_phase2",
        "global_label": "Pooled multilingual Cookie Theft features",
        "local_run_root": CLEAN_ROOT,
        "local_run_name": "language_zh_cookie_theft_clean_phase2",
        "local_label": "Best Chinese Cookie Theft features",
    },
    {
        "slug": "el_lion_scene",
        "title": "Greek Lion Scene",
        "task_slug": "lion_scene",
        "task_label": "Lion Scene",
        "language": "el",
        "language_label": "Greek",
        "filters": {"language": "el", "phase2_prompt_family": "lion_scene", "task_type": "PD_CTP"},
        "global_run_root": RICH_ROOT,
        "global_run_name": "language_el_pd_ctp_phase2",
        "global_label": "Broader Greek PD_CTP features",
        "local_run_root": CLEAN_ROOT,
        "local_run_name": "language_el_lion_scene_clean_phase2",
        "local_label": "Best Greek Lion Scene features",
    },
    {
        "slug": "es_reading",
        "title": "Spanish Reading",
        "task_slug": "reading",
        "task_label": "Reading",
        "language": "es",
        "language_label": "Spanish",
        "filters": {"language": "es", "task_type": "READING"},
        "global_run_root": RICH_ROOT,
        "global_run_name": "benchmark_wide_phase2",
        "global_label": "Benchmark-wide multilingual features",
        "local_run_root": CLEAN_ROOT,
        "local_run_name": "language_es_reading_clean_phase2",
        "local_label": "Best Spanish Reading features",
    },
]

TASK_POOLED_CONFIG = [
    {
        "slug": "cookie_theft_pooled",
        "title": "Pooled Cookie Theft",
        "task_slug": "cookie_theft",
        "filters": {"phase2_prompt_family": "cookie_theft", "task_type": "PD_CTP"},
        "run_root": CLEAN_ROOT,
        "run_name": "cookie_theft_pooled_clean_phase2",
        "label": "Pooled multilingual Cookie Theft features",
    }
]


def load_summary(root: Path, run_name: str) -> dict[str, object]:
    return json.loads((root / f"{run_name}_summary.json").read_text(encoding="utf-8"))


def load_feature_names(root: Path, run_name: str) -> tuple[list[str], dict[str, object]]:
    summary = load_summary(root, run_name)
    best = summary["best_model"] if "best_model" in summary else summary["best_result"]
    top_k = int(best["top_k"])
    ranking = pd.read_csv(root / f"{run_name}_anova_ranking.csv")
    names = ranking["feature_name"].head(top_k).tolist()
    meta = {
        "run_name": run_name,
        "subset_name": best["subset_name"],
        "top_k": top_k,
        "num_selected_features": int(best["num_selected_features"]),
        "balanced_accuracy": float(best["balanced_accuracy"]),
        "auroc": float(best["auroc"]),
        "model_family": best["model_family"],
        "model_variant": best["model_variant"],
    }
    return names, meta


def apply_filters(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for key, value in filters.items():
        out = out[out[key] == value]
    return out


def restrict_binary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["diagnosis_binary"] = out["diagnosis_mapped"].replace({"Dementia": "AD"})
    out = out[out["diagnosis_binary"].isin(["AD", "HC"])].copy()
    return out


def preprocess_features(feature_frame: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    usable_cols = [col for col in feature_cols if col in feature_frame.columns and feature_frame[col].notna().any()]
    prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    x = prep.fit_transform(feature_frame[usable_cols])
    if x.shape[1] > 2:
        pca_dims = min(20, x.shape[0] - 1, x.shape[1])
        if pca_dims >= 2:
            x = PCA(n_components=pca_dims, random_state=42).fit_transform(x)
    return x, usable_cols


def compute_tsne(feature_frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x, usable_cols = preprocess_features(feature_frame, feature_cols)
    n_rows = len(feature_frame)
    perplexity = min(25, max(5, (n_rows - 1) // 4))
    coords = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        early_exaggeration=8.0,
        init="pca",
        random_state=42,
    ).fit_transform(x)
    out = feature_frame.copy()
    out["tsne_x"] = coords[:, 0]
    out["tsne_y"] = coords[:, 1]
    out["usable_feature_count"] = len(usable_cols)
    out["tsne_perplexity"] = perplexity
    return out


def add_class_ellipse(ax, subset: pd.DataFrame, color: str):
    if len(subset) < 3:
        return
    coords = subset[["tsne_x", "tsne_y"]].to_numpy()
    cov = np.cov(coords.T)
    if not np.isfinite(cov).all():
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-6)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.5 * np.sqrt(vals)
    mean = coords.mean(axis=0)
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=theta,
        facecolor=color,
        edgecolor=color,
        alpha=0.08,
        linewidth=1.2,
    )
    ax.add_patch(ellipse)
    ax.scatter(mean[0], mean[1], marker="x", color=color, s=80, linewidths=1.6, zorder=4)


def draw_diagnosis_scatter(ax, panel: pd.DataFrame):
    for diagnosis in ["AD", "HC"]:
        diag_df = panel[panel["diagnosis_binary"] == diagnosis]
        if diag_df.empty:
            continue
        ax.scatter(
            diag_df["tsne_x"],
            diag_df["tsne_y"],
            label=f"{diagnosis} (n={len(diag_df)})",
            color=DIAG_COLORS[diagnosis],
            alpha=0.82,
            s=30,
            edgecolors="white",
            linewidths=0.2,
        )
        add_class_ellipse(ax, diag_df, DIAG_COLORS[diagnosis])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.15)


def draw_dataset_overlay(ax, panel: pd.DataFrame):
    datasets = sorted(panel["dataset_name"].dropna().unique())
    marker_map = {dataset: DATASET_MARKERS[idx % len(DATASET_MARKERS)] for idx, dataset in enumerate(datasets)}
    for dataset in datasets:
        for diagnosis in ["AD", "HC"]:
            subset = panel[(panel["dataset_name"] == dataset) & (panel["diagnosis_binary"] == diagnosis)]
            if subset.empty:
                continue
            ax.scatter(
                subset["tsne_x"],
                subset["tsne_y"],
                color=DIAG_COLORS[diagnosis],
                marker=marker_map[dataset],
                alpha=0.82,
                s=32,
                edgecolors="white",
                linewidths=0.2,
            )
    diag_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DIAG_COLORS[d], markeredgecolor="white", markersize=7, label=d)
        for d in ["AD", "HC"]
    ]
    dataset_handles = [
        Line2D([0], [0], marker=marker_map[d], color="#555555", linestyle="None", markersize=7, label=d)
        for d in datasets
    ]
    ax.legend(handles=diag_handles + dataset_handles, frameon=False, fontsize=8, ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.15)


def draw_language_overlay(ax, panel: pd.DataFrame):
    languages = sorted(panel["language"].dropna().unique())
    for language in languages:
        for diagnosis in ["AD", "HC"]:
            subset = panel[(panel["language"] == language) & (panel["diagnosis_binary"] == diagnosis)]
            if subset.empty:
                continue
            ax.scatter(
                subset["tsne_x"],
                subset["tsne_y"],
                color=DIAG_COLORS[diagnosis],
                marker=LANG_MARKERS.get(language, "o"),
                alpha=0.82,
                s=32,
                edgecolors="white",
                linewidths=0.2,
            )
    diag_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DIAG_COLORS[d], markeredgecolor="white", markersize=7, label=d)
        for d in ["AD", "HC"]
    ]
    lang_handles = [
        Line2D([0], [0], marker=LANG_MARKERS.get(lang, "o"), color="#555555", linestyle="None", markersize=7, label=lang)
        for lang in languages
    ]
    ax.legend(handles=diag_handles + lang_handles, frameon=False, fontsize=8, ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.15)


def projection_title(label: str, meta: dict[str, object], panel: pd.DataFrame) -> str:
    return (
        f"{label}\n"
        f"{meta['model_family']}:{meta['model_variant']} | {meta['subset_name']} | "
        f"k={meta['top_k']} | bal acc={meta['balanced_accuracy']:.3f} | "
        f"AD={int((panel['diagnosis_binary'] == 'AD').sum())}, HC={int((panel['diagnosis_binary'] == 'HC').sum())}"
    )


def save_single_projection(panel: pd.DataFrame, meta: dict[str, object], title_label: str, path: Path):
    fig, ax = plt.subplots(figsize=(6.6, 5.6), constrained_layout=True)
    draw_diagnosis_scatter(ax, panel)
    ax.set_title(projection_title(title_label, meta, panel), fontsize=11)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_dataset_projection(panel: pd.DataFrame, meta: dict[str, object], title_label: str, path: Path):
    if panel["dataset_name"].nunique() <= 1:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.8), constrained_layout=True)
    draw_dataset_overlay(ax, panel)
    ax.set_title(projection_title(title_label, meta, panel), fontsize=11)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_language_projection(panel: pd.DataFrame, meta: dict[str, object], title_label: str, path: Path):
    if panel["language"].nunique() <= 1:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.8), constrained_layout=True)
    draw_language_overlay(ax, panel)
    ax.set_title(projection_title(title_label, meta, panel), fontsize=11)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_comparison_panel(global_panel: pd.DataFrame, global_meta: dict[str, object], local_panel: pd.DataFrame, local_meta: dict[str, object], title: str, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), constrained_layout=True)
    draw_diagnosis_scatter(axes[0], global_panel)
    draw_diagnosis_scatter(axes[1], local_panel)
    axes[0].set_title(projection_title("Global / pooled feature view", global_meta, global_panel), fontsize=11)
    axes[1].set_title(projection_title("Local best feature view", local_meta, local_panel), fontsize=11)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_projection_csv(panel: pd.DataFrame, path: Path):
    cols = [
        "sample_id",
        "group_id",
        "dataset_name",
        "language",
        "task_type",
        "phase2_prompt_family",
        "diagnosis_binary",
        "tsne_x",
        "tsne_y",
        "usable_feature_count",
        "tsne_perplexity",
    ]
    panel[cols].to_csv(path, index=False)


def save_metadata(meta: dict[str, object], path: Path):
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build_projection(df: pd.DataFrame, filters: dict[str, object], root: Path, run_name: str) -> tuple[pd.DataFrame, dict[str, object]]:
    filtered = restrict_binary(apply_filters(df, filters))
    feature_names, meta = load_feature_names(root, run_name)
    feature_names = [name for name in feature_names if name in filtered.columns]
    projected = compute_tsne(
        filtered[
            [
                "sample_id",
                "group_id",
                "dataset_name",
                "language",
                "task_type",
                "phase2_prompt_family",
                "diagnosis_mapped",
                "diagnosis_binary",
            ]
            + feature_names
        ].copy(),
        feature_names,
    )
    meta["usable_feature_count"] = int(projected["usable_feature_count"].iloc[0])
    meta["n_rows"] = int(len(projected))
    meta["n_ad"] = int((projected["diagnosis_binary"] == "AD").sum())
    meta["n_hc"] = int((projected["diagnosis_binary"] == "HC").sum())
    return projected, meta


def ensure_dirs(*paths: Path):
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_slice_bundle(cfg: dict[str, object], global_panel: pd.DataFrame, global_meta: dict[str, object], local_panel: pd.DataFrame, local_meta: dict[str, object]):
    by_language_dir = REPORT_ROOT / "by_language" / str(cfg["language"]) / str(cfg["task_slug"])
    by_task_dir = REPORT_ROOT / "by_task" / str(cfg["task_slug"]) / str(cfg["language"])
    ensure_dirs(by_language_dir, by_task_dir)

    for out_dir in [by_language_dir, by_task_dir]:
        save_single_projection(global_panel, global_meta, str(cfg["global_label"]), out_dir / "tsne_global_features.png")
        save_single_projection(local_panel, local_meta, str(cfg["local_label"]), out_dir / "tsne_local_features.png")
        save_dataset_projection(global_panel, global_meta, str(cfg["global_label"]), out_dir / "tsne_global_dataset_overlay.png")
        save_dataset_projection(local_panel, local_meta, str(cfg["local_label"]), out_dir / "tsne_local_dataset_overlay.png")
        save_comparison_panel(global_panel, global_meta, local_panel, local_meta, str(cfg["title"]), out_dir / "tsne_global_vs_local.png")
        save_projection_csv(global_panel, out_dir / "projection_global.csv")
        save_projection_csv(local_panel, out_dir / "projection_local.csv")
        save_metadata(global_meta, out_dir / "projection_global_meta.json")
        save_metadata(local_meta, out_dir / "projection_local_meta.json")


def save_task_pooled_bundle(cfg: dict[str, object], panel: pd.DataFrame, meta: dict[str, object]):
    out_dir = REPORT_ROOT / "by_task" / str(cfg["task_slug"]) / "pooled_multilingual"
    ensure_dirs(out_dir)
    save_single_projection(panel, meta, str(cfg["label"]), out_dir / "tsne_diagnosis_only.png")
    save_language_projection(panel, meta, str(cfg["label"]), out_dir / "tsne_language_overlay.png")
    save_dataset_projection(panel, meta, str(cfg["label"]), out_dir / "tsne_dataset_overlay.png")
    save_projection_csv(panel, out_dir / "projection.csv")
    save_metadata(meta, out_dir / "projection_meta.json")


def write_summary_csv(rows: list[dict[str, object]]):
    pd.DataFrame(rows).to_csv(REPORT_ROOT / "clean_prompt_language_projection_comparisons.csv", index=False)


def main():
    log = make_logger("phase2_clean_prompt_projections")
    df = pd.read_csv(PHASE2_ROOT / "phase2_features.csv")
    summary_rows: list[dict[str, object]] = []

    log("Building AD/HC-only clean-slice t-SNE projections by language and by task")
    for cfg in SLICE_CONFIG:
        global_panel, global_meta = build_projection(df, cfg["filters"], cfg["global_run_root"], cfg["global_run_name"])
        local_panel, local_meta = build_projection(df, cfg["filters"], cfg["local_run_root"], cfg["local_run_name"])
        save_slice_bundle(cfg, global_panel, global_meta, local_panel, local_meta)
        summary_rows.extend(
            [
                {
                    "slice_slug": cfg["slug"],
                    "view_type": "global",
                    "task_slug": cfg["task_slug"],
                    "language": cfg["language"],
                    **global_meta,
                },
                {
                    "slice_slug": cfg["slug"],
                    "view_type": "local",
                    "task_slug": cfg["task_slug"],
                    "language": cfg["language"],
                    **local_meta,
                },
            ]
        )

    for cfg in TASK_POOLED_CONFIG:
        panel, meta = build_projection(df, cfg["filters"], cfg["run_root"], cfg["run_name"])
        save_task_pooled_bundle(cfg, panel, meta)
        summary_rows.append(
            {
                "slice_slug": cfg["slug"],
                "view_type": "task_pooled",
                "task_slug": cfg["task_slug"],
                "language": "multilingual",
                **meta,
            }
        )

    write_summary_csv(summary_rows)
    log("Finished clean prompt t-SNE artifacts")


if __name__ == "__main__":
    main()
