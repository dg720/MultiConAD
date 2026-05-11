import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, ttest_ind
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.phase1.run_rich_sweep import (  # noqa: E402
    FEATURES_PATH,
    MANIFEST_PATH,
    RICH_SWEEP_ROOT,
    grouped_train_test_split,
    model_specs,
    normalize_with_fallback,
    select_top_k,
)
from processing.phase1.common import make_logger  # noqa: E402
from processing.phase1.extract_features import (  # noqa: E402
    clean_text,
    get_stanza_pipeline,
    parsed_text_features,
)


REPORT_ROOT = RICH_SWEEP_ROOT / "report_assets"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    if path.exists():
        path.unlink()
    df.to_csv(path, index=False)


def load_merged() -> pd.DataFrame:
    manifest = pd.read_json(MANIFEST_PATH, lines=True)
    features = pd.read_csv(FEATURES_PATH)
    df = manifest.merge(
        features,
        on=["sample_id", "group_id", "dataset_name", "language", "task_type", "diagnosis_mapped", "binary_label"],
    )
    return df[df["binary_label"].isin([0, 1])].copy()


def get_model_spec(model_family: str, model_variant: str):
    for spec in model_specs():
        if spec["model_family"] == model_family and spec["model_variant"] == model_variant:
            return spec
    raise KeyError((model_family, model_variant))


def apply_filter(df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for key, value in filters.items():
        out = out[out[key] == value].copy()
    return out


def fit_run(df: pd.DataFrame, filters: dict[str, object], grouping_levels: list[list[str]], summary_name: str):
    summary = pd.read_json(RICH_SWEEP_ROOT / f"{summary_name}_summary.json", typ="series")
    best = summary["best_config"]
    run_df = apply_filter(df, filters)
    train_df, test_df = grouped_train_test_split(run_df, test_size=0.2, seed=42)

    # replicate the feature subset logic from the fitted run
    from experiments.phase1.run_rich_sweep import feature_subset_columns

    feature_cols = feature_subset_columns(run_df, best["subset"])
    train_norm, test_norm = normalize_with_fallback(train_df, test_df, feature_cols, grouping_levels)
    y_train = train_norm["binary_label"].astype(int)
    y_test = test_norm["binary_label"].astype(int)
    selected_cols, anova_ranking = select_top_k(train_norm, y_train, feature_cols, best["top_k"])
    x_train = train_norm[selected_cols]
    x_test = test_norm[selected_cols]

    spec = get_model_spec(best["model_family"], best["model_variant"])
    estimator = spec["estimator"]
    estimator.fit(x_train, y_train)
    pred = estimator.predict(x_test)
    if hasattr(estimator, "predict_proba"):
        score = estimator.predict_proba(x_test)[:, 1]
    else:
        score = estimator.decision_function(x_test)

    native_path = RICH_SWEEP_ROOT / f"{summary_name}_native_importance_no_missing_indicators.csv"
    native_importance = pd.read_csv(native_path) if native_path.exists() else None
    return {
        "summary": summary,
        "run_df": run_df,
        "train_df": train_df,
        "test_df": test_df,
        "train_norm": train_norm,
        "test_norm": test_norm,
        "y_train": y_train,
        "y_test": y_test,
        "x_train": x_train,
        "x_test": x_test,
        "selected_cols": selected_cols,
        "anova_ranking": anova_ranking,
        "estimator": estimator,
        "pred": pred,
        "score": score,
        "native_importance": native_importance,
    }


def choose_saliency_pair(run_state: dict[str, object]) -> pd.DataFrame:
    test_df = run_state["test_df"].copy()
    pred = pd.Series(run_state["pred"], index=test_df.index)
    score = pd.Series(run_state["score"], index=test_df.index)
    test_df["pred"] = pred
    test_df["score"] = score
    correct = test_df[test_df["binary_label"].astype(int) == test_df["pred"].astype(int)].copy()
    dataset_priority = ["Pitt", "Lu", "WLS", "ADReSS-M", "DS5", "DS7", "iFlytek"]
    for dataset_name in dataset_priority + sorted(correct["dataset_name"].unique().tolist()):
        subset = correct[correct["dataset_name"] == dataset_name].copy()
        if subset.empty or subset["binary_label"].nunique() < 2:
            continue
        ad_rows = subset[subset["binary_label"] == 1].sort_values("score", ascending=False)
        hc_rows = subset[subset["binary_label"] == 0].sort_values("score", ascending=True)
        if not ad_rows.empty and not hc_rows.empty:
            return pd.concat([ad_rows.head(1), hc_rows.head(1)], axis=0)
    raise RuntimeError("Could not find a well-classified AD/HC pair in the held-out test split.")


def make_saliency_plot(run_state: dict[str, object], pair: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    native = run_state["native_importance"]
    top_features = native.head(10)["feature_name"].tolist()

    estimator = run_state["estimator"]
    imputer = estimator.named_steps["imputer"]
    scaler = estimator.named_steps["scaler"]
    clf = estimator.named_steps["clf"]
    imputed_names = list(imputer.get_feature_names_out(run_state["selected_cols"]))

    pair_x = run_state["x_test"].loc[pair.index, run_state["selected_cols"]]
    imputed = imputer.transform(pair_x)
    scaled = scaler.transform(imputed)
    coef = clf.coef_[0]

    rows = []
    for sample_idx, (_, row) in enumerate(pair.iterrows()):
        for feature_name in top_features:
            feature_pos = imputed_names.index(feature_name)
            contribution = float(scaled[sample_idx, feature_pos] * coef[feature_pos])
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "dataset_name": row["dataset_name"],
                    "diagnosis": row["diagnosis_mapped"],
                    "feature_name": feature_name,
                    "contribution": contribution,
                    "saliency_abs": abs(contribution),
                    "ad_probability": float(row["score"]),
                }
            )

    saliency_df = pd.DataFrame(rows)
    max_val = max(float(saliency_df["saliency_abs"].max()), 1e-6)
    saliency_df["saliency_0_100"] = saliency_df["saliency_abs"] / max_val * 100.0
    safe_to_csv(saliency_df, REPORT_ROOT / "english_pd_ctp_saliency_values.csv")

    features = top_features
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    palette = {"AD": "#b33a3a", "HC": "#2d6a4f"}

    for diagnosis in ["AD", "HC"]:
        subset = saliency_df[saliency_df["diagnosis"] == diagnosis].copy()
        subset = subset.set_index("feature_name").reindex(features).reset_index()
        values = subset["saliency_0_100"].tolist()
        values += values[:1]
        label = f"{diagnosis} | {subset['dataset_name'].iloc[0]} | {subset['group_id'].iloc[0]}"
        ax.plot(angles, values, linewidth=2.5, label=label, color=palette[diagnosis])
        ax.fill(angles, values, alpha=0.15, color=palette[diagnosis])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [feature.replace("_", "\n") for feature in features],
        fontsize=9,
    )
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title("English PD_CTP Saliency Map\nTop linear-SVM feature contributions for one AD and one HC subject", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.15), frameon=False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return saliency_df


def example_feature_table(output_csv: Path) -> pd.DataFrame:
    example_text = (
        "The boy is taking a cookie from the jar. "
        "The mother is washing the dishes. "
        "The girl is watching them."
    )
    pipeline = get_stanza_pipeline("en")
    doc = pipeline(example_text)
    feature_values = parsed_text_features(example_text, "en", doc)
    chosen = [
        "len_token_count",
        "len_type_count",
        "len_tokens_per_utterance",
        "lex_lemma_type_token_ratio",
        "lex_function_word_ratio",
        "syn_determiner_ratio",
        "syn_aux_ratio",
        "syn_noun_ratio",
        "syn_content_function_ratio",
        "syn_mean_dependency_length",
    ]
    explanations = {
        "len_token_count": "Total number of word tokens in the response.",
        "len_type_count": "Number of distinct word forms used at least once.",
        "len_tokens_per_utterance": "Average number of tokens per utterance after splitting the response.",
        "lex_lemma_type_token_ratio": "Distinct lemmas divided by total tokens; reduces inflectional duplication.",
        "lex_function_word_ratio": "Share of tokens that are function words such as determiners, pronouns, auxiliaries, and conjunctions.",
        "syn_determiner_ratio": "Share of tokens tagged as determiners, e.g. 'the', 'a'.",
        "syn_aux_ratio": "Share of tokens tagged as auxiliary verbs, e.g. 'is', 'are'.",
        "syn_noun_ratio": "Share of tokens tagged as common nouns.",
        "syn_content_function_ratio": "Content-word count divided by function-word count.",
        "syn_mean_dependency_length": "Average absolute token distance between a word and its syntactic head.",
    }
    rows = []
    for feature in chosen:
        value = feature_values.get(feature, np.nan)
        if pd.isna(value):
            display = "NA"
        elif "ratio" in feature or "length" in feature:
            display = f"{value:.3f}"
        else:
            display = f"{int(value)}" if float(value).is_integer() else f"{value:.3f}"
        rows.append(
            {
                "feature_name": feature,
                "feature_family": feature.split("_", 1)[0],
                "explanation": explanations[feature],
                "example_value": display,
                "example_text": example_text,
            }
        )
    df = pd.DataFrame(rows)
    safe_to_csv(df, output_csv)
    return df


def summarize_feature_inventory(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    names = metadata["feature_name"].tolist()

    def count(pattern):
        return int(sum(1 for name in names if pattern(name)))

    lexical_rows = [
        ("Length and response volume", count(lambda n: n.startswith("len_")), "Counts, utterance length, and speech-rate derived length measures."),
        ("Lexical diversity and richness", count(lambda n: n.startswith("lex_")), "Type-token style diversity, lexical richness, repetition, and word-form measures."),
        ("Discourse repetition and coherence", count(lambda n: n.startswith("disc_")), "Repetition, local coherence, and utterance-to-utterance semantic similarity."),
        ("Syntactic composition", count(lambda n: n.startswith("syn_")), "POS ratios, dependency relations, subordination, and parse-complexity measures."),
        ("Speech graph topology", count(lambda n: n.startswith("graph_")), "Graph density, loops, components, and degree over token-transition graphs."),
    ]

    acoustic_rows = [
        ("Pause and silence timing", count(lambda n: n.startswith("pause_")), "Counts, durations, and timing ratios from silence and filled-pause behavior."),
        ("openSMILE eGeMAPS", count(lambda n: n.startswith("ac_egemaps_")), "Functionals from the eGeMAPSv02 descriptor set."),
        ("MFCC summary statistics", count(lambda n: n.startswith("ac_mfcc_")), "Mean and standard deviation of low-order MFCCs from librosa fallback extraction."),
        ("Energy and zero-crossing", count(lambda n: n.startswith("ac_rms_") or n.startswith("ac_zcr_")), "Signal-energy and zero-crossing descriptors."),
        ("Pitch and duration fallback", count(lambda n: n in {"ac_duration", "ac_f0_mean", "ac_f0_std"}), "Audio duration and coarse pitch descriptors used as fallback features."),
    ]

    semantic_rows = [
        ("Universal handcrafted semantics in Phase 1", 0, "No prompt-specific content-unit features are extracted in the current Phase 1 benchmark."),
        ("Planned PD_CTP semantic module", 9, "Phase 2 target block: content-unit coverage, density, object/action balance, and semantic similarity."),
        ("Planned FLUENCY semantic module", 12, "Phase 2 target block: valid items, intrusions, clustering, switching, and item rate."),
        ("Planned STORY_NARRATIVE semantic module", 7, "Phase 2 target block: proposition coverage, event ordering, and story-reference similarity."),
        ("Planned CONVERSATION semantic module", 7, "Phase 2 target block: topic coherence, topic switching, and named-entity / repetition dynamics."),
    ]

    return pd.DataFrame(lexical_rows, columns=["feature_type", "num_features", "brief_description"]), pd.DataFrame(
        acoustic_rows, columns=["feature_type", "num_features", "brief_description"]
    ), pd.DataFrame(semantic_rows, columns=["feature_type", "num_features", "brief_description"])


def bh_fdr(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    ranked = np.array(p_values)[order]
    adjusted = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * m / rank
        prev = min(prev, value)
        adjusted[i] = min(prev, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adjusted
    return out.tolist()


def feature_differentiation_table(run_state: dict[str, object], output_csv: Path) -> pd.DataFrame:
    native = pd.read_csv(RICH_SWEEP_ROOT / "pd_ctp_pooled_rich_native_importance_no_missing_indicators.csv")
    native = native.set_index("feature_name")
    selected = [feature for feature in run_state["selected_cols"] if feature in native.index]
    train_df = run_state["train_df"].copy()

    rows = []
    for feature in selected:
        ad_vals = pd.to_numeric(train_df.loc[train_df["binary_label"] == 1, feature], errors="coerce").dropna()
        hc_vals = pd.to_numeric(train_df.loc[train_df["binary_label"] == 0, feature], errors="coerce").dropna()
        if len(ad_vals) < 3 or len(hc_vals) < 3:
            continue
        t_stat, p_value = ttest_ind(ad_vals, hc_vals, equal_var=False, nan_policy="omit")
        pooled_std = math.sqrt(((ad_vals.std(ddof=1) ** 2) + (hc_vals.std(ddof=1) ** 2)) / 2.0)
        effect = (ad_vals.mean() - hc_vals.mean()) / pooled_std if pooled_std not in {0, 0.0} else np.nan
        merged_vals = pd.concat([ad_vals, hc_vals], axis=0)
        merged_labels = pd.Series([1] * len(ad_vals) + [0] * len(hc_vals))
        corr = pointbiserialr(merged_labels, merged_vals).correlation if merged_vals.nunique() > 1 else np.nan
        rows.append(
            {
                "feature_name": feature,
                "feature_type": feature.split("_", 1)[0],
                "mu_ad": float(ad_vals.mean()),
                "mu_hc": float(hc_vals.mean()),
                "cohens_d": float(effect) if pd.notna(effect) else np.nan,
                "label_correlation": float(corr) if pd.notna(corr) else np.nan,
                "svm_weight": float(native.loc[feature, "native_importance"]),
                "svm_weight_abs": float(native.loc[feature, "native_importance_abs"]),
                "p_value": float(p_value),
            }
        )

    df = pd.DataFrame(rows).sort_values("p_value", ascending=True).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No differentiation rows were generated.")
    df["bonferroni_p"] = (df["p_value"] * len(df)).clip(upper=1.0)
    df["fdr_bh_p"] = bh_fdr(df["p_value"].tolist())
    df["significant_bonferroni"] = df["bonferroni_p"] < 0.05
    df["significant_fdr"] = df["fdr_bh_p"] < 0.05
    report_df = df.sort_values(["significant_bonferroni", "significant_fdr", "p_value", "svm_weight_abs"], ascending=[False, False, True, False]).head(15)
    safe_to_csv(report_df, output_csv)
    return report_df


def projection_plots(run_state: dict[str, object], differentiation_df: pd.DataFrame) -> pd.DataFrame:
    import umap  # type: ignore

    top_features = differentiation_df["feature_name"].head(15).tolist()
    train_df = run_state["train_df"].copy()
    projection_frame = train_df[
        ["sample_id", "group_id", "dataset_name", "language", "diagnosis_mapped", "binary_label"] + top_features
    ].copy()

    prep = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x = prep.fit_transform(projection_frame[top_features])

    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
    tsne_coords = tsne.fit_transform(x)

    reducer = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.15, random_state=42)
    umap_coords = reducer.fit_transform(x)

    projection_frame["tsne_x"] = tsne_coords[:, 0]
    projection_frame["tsne_y"] = tsne_coords[:, 1]
    projection_frame["umap_x"] = umap_coords[:, 0]
    projection_frame["umap_y"] = umap_coords[:, 1]
    safe_to_csv(projection_frame, REPORT_ROOT / "pd_ctp_projection_embeddings.csv")

    palette = {"AD": "#b33a3a", "HC": "#2d6a4f"}
    markers = {"en": "o", "zh": "s", "el": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, xcol, ycol, title in [
        (axes[0], "tsne_x", "tsne_y", "t-SNE"),
        (axes[1], "umap_x", "umap_y", "UMAP"),
    ]:
        for language in sorted(projection_frame["language"].dropna().unique()):
            for diagnosis in ["AD", "HC"]:
                subset = projection_frame[
                    (projection_frame["language"] == language) & (projection_frame["diagnosis_mapped"] == diagnosis)
                ]
                if subset.empty:
                    continue
                ax.scatter(
                    subset[xcol],
                    subset[ycol],
                    c=palette.get(diagnosis, "#666666"),
                    marker=markers.get(language, "o"),
                    label=f"{diagnosis}-{language}",
                    alpha=0.72,
                    s=24,
                    edgecolors="none",
                )
        ax.set_title(f"{title} on top pooled PD_CTP differentiating features")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(alpha=0.2)

    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Pooled PD_CTP separability using top statistically differentiating features", y=1.08, fontsize=14)
    fig.tight_layout()
    fig.savefig(REPORT_ROOT / "pd_ctp_tsne_umap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return projection_frame


def compute_projection_coords(feature_frame: pd.DataFrame, feature_cols: list[str], method: str) -> pd.DataFrame:
    import umap  # type: ignore

    prep = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    x = prep.fit_transform(feature_frame[feature_cols])
    n_samples = len(feature_frame)
    if method == "tsne":
        perplexity = min(30, max(5, (n_samples - 1) // 3))
        reducer = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=42)
        coords = reducer.fit_transform(x)
        return pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    if method == "umap":
        n_neighbors = min(25, max(5, n_samples // 10))
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.15, random_state=42)
        coords = reducer.fit_transform(x)
        return pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    raise ValueError(method)


def language_projection_comparisons(
    df: pd.DataFrame,
    benchmark_state: dict[str, object],
    pd_ctp_state: dict[str, object],
    en_state: dict[str, object],
    zh_state: dict[str, object],
    el_state: dict[str, object],
    es_state: dict[str, object],
) -> pd.DataFrame:
    configs = [
        {
            "language": "English",
            "code": "en",
            "data_filter": {"language": "en", "task_type": "PD_CTP"},
            "global_features": pd_ctp_state["selected_cols"],
            "monolingual_features": en_state["selected_cols"],
            "slice_label": "PD_CTP",
        },
        {
            "language": "Chinese",
            "code": "zh",
            "data_filter": {"language": "zh", "task_type": "PD_CTP"},
            "global_features": pd_ctp_state["selected_cols"],
            "monolingual_features": zh_state["selected_cols"],
            "slice_label": "PD_CTP",
        },
        {
            "language": "Greek",
            "code": "el",
            "data_filter": {"language": "el", "task_type": "PD_CTP"},
            "global_features": pd_ctp_state["selected_cols"],
            "monolingual_features": el_state["selected_cols"],
            "slice_label": "PD_CTP",
        },
        {
            "language": "Spanish",
            "code": "es",
            "data_filter": {"language": "es", "task_type": "READING"},
            "global_features": benchmark_state["selected_cols"],
            "monolingual_features": es_state["selected_cols"],
            "slice_label": "READING",
        },
    ]

    all_rows = []
    for config in configs:
        subset = df.copy()
        for key, value in config["data_filter"].items():
            subset = subset[subset[key] == value].copy()
        subset = subset[subset["binary_label"].isin([0, 1])].copy()
        base_cols = ["sample_id", "group_id", "dataset_name", "language", "task_type", "diagnosis_mapped", "binary_label"]

        for feature_source, feature_cols in [("global", config["global_features"]), ("monolingual", config["monolingual_features"])]:
            available = [col for col in feature_cols if col in subset.columns]
            panel = subset[base_cols + available].copy()
            for method in ["tsne", "umap"]:
                coords = compute_projection_coords(panel, available, method)
                out = panel[base_cols].copy().reset_index(drop=True)
                out["projection_method"] = method
                out["feature_source"] = feature_source
                out["language_label"] = config["language"]
                out["slice_label"] = config["slice_label"]
                out["x"] = coords["x"]
                out["y"] = coords["y"]
                out["num_features"] = len(available)
                all_rows.append(out)

    projection_df = pd.concat(all_rows, ignore_index=True)
    safe_to_csv(projection_df, REPORT_ROOT / "language_projection_comparisons.csv")

    palette = {"AD": "#b33a3a", "HC": "#2d6a4f"}
    for method in ["tsne", "umap"]:
        fig, axes = plt.subplots(4, 2, figsize=(12, 18))
        method_df = projection_df[projection_df["projection_method"] == method].copy()
        for row_idx, language in enumerate(["English", "Chinese", "Greek", "Spanish"]):
            for col_idx, feature_source in enumerate(["global", "monolingual"]):
                ax = axes[row_idx, col_idx]
                subset = method_df[
                    (method_df["language_label"] == language) & (method_df["feature_source"] == feature_source)
                ].copy()
                for diagnosis in ["AD", "HC"]:
                    dx = subset[subset["diagnosis_mapped"] == diagnosis]
                    if dx.empty:
                        continue
                    ax.scatter(
                        dx["x"],
                        dx["y"],
                        c=palette[diagnosis],
                        label=diagnosis,
                        alpha=0.72,
                        s=22,
                        edgecolors="none",
                    )
                if row_idx == 0:
                    title_prefix = "Global multilingual features" if feature_source == "global" else "Best monolingual features"
                    ax.set_title(title_prefix)
                label = subset["slice_label"].iloc[0] if not subset.empty else ""
                ax.text(0.02, 0.97, f"{language} | {label}", transform=ax.transAxes, ha="left", va="top", fontsize=10)
                ax.grid(alpha=0.2)
                if row_idx == 3:
                    ax.set_xlabel("Component 1")
                if col_idx == 0:
                    ax.set_ylabel("Component 2")
                if row_idx == 0 and col_idx == 1:
                    ax.legend(frameon=False, loc="upper right")

        fig.suptitle(
            f"{method.upper()} separation by language: global multilingual vs best monolingual feature sets",
            y=0.995,
            fontsize=15,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        fig.savefig(REPORT_ROOT / f"language_{method}_comparison_grid.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    return projection_df


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    def fmt(value):
        if isinstance(value, float):
            if pd.isna(value):
                return "NA"
            return f"{value:.3f}"
        return str(value)

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def build_report(
    example_df: pd.DataFrame,
    lexical_df: pd.DataFrame,
    acoustic_df: pd.DataFrame,
    semantic_df: pd.DataFrame,
    differentiation_df: pd.DataFrame,
    benchmark_anova_df: pd.DataFrame,
    pd_ctp_anova_df: pd.DataFrame,
    benchmark_method_df: pd.DataFrame,
    pd_ctp_method_df: pd.DataFrame,
    projection_df: pd.DataFrame,
    en_pair: pd.DataFrame,
    saliency_df: pd.DataFrame,
) -> None:
    report_path = RICH_SWEEP_ROOT / "REPORT_RESULTS.md"
    ad_row = en_pair[en_pair["diagnosis_mapped"] == "AD"].iloc[0]
    hc_row = en_pair[en_pair["diagnosis_mapped"] == "HC"].iloc[0]

    model_summary = pd.DataFrame(
        [
            ["Benchmark-wide", "SVM (rbf)", "all_universal", 50, 0.749, 0.752, 0.818, 0.736],
            ["Pooled PD_CTP", "SVM (linear)", "all_universal", 100, 0.751, 0.752, 0.813, 0.727],
            ["English PD_CTP", "SVM (linear)", "text_only", 50, 0.756, 0.770, 0.825, 0.742],
            ["Chinese PD_CTP", "Decision Tree", "pause_only", 20, 0.986, 0.964, 0.986, 0.889],
            ["Greek PD_CTP", "SVM (linear)", "acoustic_only", 50, 0.925, 0.925, 0.944, 0.973],
        ],
        columns=["analysis", "best_model", "feature_subset", "top_k", "balanced_accuracy", "macro_f1", "auroc", "auprc"],
    )

    cross_lingual_df = pd.read_csv(RICH_SWEEP_ROOT / "cross_lingual_all_tasks_feature_stability.csv").head(10)
    pd_ctp_overlap_df = pd.read_csv(RICH_SWEEP_ROOT / "cross_lingual_pd_ctp_feature_group_stability.csv")

    md = []
    md.append("# Phase 1 Rich Sweep Report")
    md.append("")
    md.append("## How the cited papers present results")
    md.append("")
    md.append("The report structure here follows the presentation logic used in the three reference papers.")
    md.append("")
    md.append("- Lindsay, Tröger, and König (2021) use worked feature tables with short examples and plain-language explanations. That style is useful for prompt-specific or clinically interpretable features, so the feature-example table below mirrors that pattern.")
    md.append("- Balagopalan et al. (2021) separate three things cleanly: feature inventory tables, model-performance tables, and a feature-differentiation table with class means and model weights. The inventory and differentiation tables below follow that design.")
    md.append("- Laguarta and Subirana (2021) present subject-level saliency with a radar chart to support explainability and longitudinal monitoring. The patient-level saliency figure below adapts that visual idea for our best interpretable English PD_CTP model.")
    md.append("")
    md.append("## Main results")
    md.append("")
    md.append(markdown_table(model_summary, model_summary.columns.tolist()))
    md.append("")
    md.append("The main benchmark-wide winner is a nonlinear SVM on the full universal handcrafted set, but the more interpretable primary claim remains the pooled `PD_CTP` slice, where a linear SVM on the universal feature stack reaches balanced accuracy `0.751` and AUROC `0.813`.")
    md.append("")
    md.append("## Table 1. Worked examples for key report features")
    md.append("")
    md.append("Example text used for the calculations: \"The boy is taking a cookie from the jar. The mother is washing the dishes. The girl is watching them.\"")
    md.append("")
    md.append(markdown_table(example_df, ["feature_name", "feature_family", "explanation", "example_value"]))
    md.append("")
    md.append("## Table 2. Summary of extracted lexico-syntactic and discourse features")
    md.append("")
    md.append(markdown_table(lexical_df, lexical_df.columns.tolist()))
    md.append("")
    md.append("## Table 3. Summary of extracted acoustic and pause features")
    md.append("")
    md.append(markdown_table(acoustic_df, acoustic_df.columns.tolist()))
    md.append("")
    md.append("## Table 4. Status of task-specific semantic modules")
    md.append("")
    md.append(markdown_table(semantic_df, semantic_df.columns.tolist()))
    md.append("")
    md.append("The current rich sweep is still a Phase 1 universal-feature analysis. The Phase 2 task-specific semantic modules are planned but not yet extracted into the live benchmark table, which is why the prompt-specific content-unit block remains a next-step item rather than part of the current sweep.")
    md.append("")
    md.append("## Figure 1. Patient-level saliency map")
    md.append("")
    md.append(f"Selected held-out English `PD_CTP` pair from dataset `{ad_row['dataset_name']}`:")
    md.append("")
    md.append(f"- AD subject `{ad_row['group_id']}` with AD probability `{ad_row['score']:.3f}`")
    md.append(f"- HC subject `{hc_row['group_id']}` with AD probability `{hc_row['score']:.3f}`")
    md.append("")
    md.append(f"AD excerpt: \"{clean_text(ad_row['analysis_text'])[:240]}...\"")
    md.append("")
    md.append(f"HC excerpt: \"{clean_text(hc_row['analysis_text'])[:240]}...\"")
    md.append("")
    md.append("![English PD_CTP saliency map](report_assets/english_pd_ctp_saliency_map.png)")
    md.append("")
    md.append("The saliency figure uses the top non-missing linear-SVM features from the English `PD_CTP` winner and plots absolute per-feature contribution magnitudes after imputation and scaling. It is a contribution-style diagnostic view, not a causal claim.")
    md.append("")
    md.append("## Figure 2. Language-specific t-SNE comparison")
    md.append("")
    md.append("Each row is a language-specific slice. The left column uses the strongest pooled multilingual feature set available for that slice, while the right column uses the best monolingual feature set for that language.")
    md.append("")
    md.append("![Language t-SNE comparison](report_assets/language_tsne_comparison_grid.png)")
    md.append("")
    md.append("## Figure 3. Language-specific UMAP comparison")
    md.append("")
    md.append("The same comparison is shown with UMAP. In practice, the t-SNE panels are currently easier to interpret in this benchmark than the UMAP panels.")
    md.append("")
    md.append("![Language UMAP comparison](report_assets/language_umap_comparison_grid.png)")
    md.append("")
    md.append("Why English showed distinct clusters in the earlier pooled plot: English contains the largest number of samples and the widest mixture of corpora, transcript conventions, and prompt packaging. The projection was therefore reflecting residual dataset/task structure in addition to diagnosis. The language-specific panels are a cleaner way to inspect separation.")
    md.append("")
    md.append("## Table 5. Feature differentiation analysis for pooled PD_CTP top features")
    md.append("")
    md.append(markdown_table(differentiation_df, ["feature_name", "feature_type", "mu_ad", "mu_hc", "cohens_d", "label_correlation", "svm_weight", "bonferroni_p"]))
    md.append("")
    md.append("Interpretation of Table 5:")
    md.append("")
    md.append("- This table follows the Balagopalan et al. pattern: class means, association with the binary label, and model weight are shown together.")
    md.append("- The analysis is computed on the grouped training split for the pooled `PD_CTP` linear-SVM winner, using the selected features from that run.")
    md.append("- Because no MMSE target is available consistently across the benchmark, the correlation column reports point-biserial correlation with the AD label rather than severity correlation.")
    md.append("")
    md.append("## Table 6. Top ANOVA-ranked features")
    md.append("")
    md.append("### Benchmark-wide")
    md.append("")
    md.append(markdown_table(benchmark_anova_df, benchmark_anova_df.columns.tolist()))
    md.append("")
    md.append("### Pooled PD_CTP")
    md.append("")
    md.append(markdown_table(pd_ctp_anova_df, pd_ctp_anova_df.columns.tolist()))
    md.append("")
    md.append("These are train-only univariate screening scores. They are useful for understanding which single features separate classes most strongly before the final model is fit, but they are not the same thing as final model importance.")
    md.append("")
    md.append("## Table 7. Best held-out result per method, benchmark-wide")
    md.append("")
    md.append(markdown_table(benchmark_method_df, benchmark_method_df.columns.tolist()))
    md.append("")
    md.append("## Table 8. Best held-out result per method, pooled PD_CTP")
    md.append("")
    md.append(markdown_table(pd_ctp_method_df, pd_ctp_method_df.columns.tolist()))
    md.append("")
    md.append("These method tables follow the Balagopalan-style model comparison format more directly: each model family is shown with its best-performing feature subset and `top-k` setting on the same held-out split.")
    md.append("")
    md.append("## Cross-lingual discussion")
    md.append("")
    md.append("### All-task monolingual overlap")
    md.append("")
    md.append(markdown_table(cross_lingual_df, ["feature_name", "languages_present", "num_languages", "mean_rank", "mean_importance"]))
    md.append("")
    md.append("Across `en`, `es`, and `zh` all-task models, the most repeated overlap is lexical diversity and repetition structure rather than acoustics. `lex_mattr_20`, `lex_lemma_type_token_ratio`, and repetition-based discourse measures recur across languages. That is the clearest broad multilingual stability signal in the current benchmark.")
    md.append("")
    md.append("### PD_CTP cross-lingual overlap")
    md.append("")
    md.append(markdown_table(pd_ctp_overlap_df, pd_ctp_overlap_df.columns.tolist()))
    md.append("")
    md.append("Within `PD_CTP`, the feature picture is much less stable across languages. English is led by lexical and syntactic structure, Chinese by pause/rate measures, and Greek by acoustic descriptors. That means the current benchmark does not yet support a claim that one small universal top-feature set explains `PD_CTP` impairment consistently across languages.")
    md.append("")
    md.append("## Report takeaways")
    md.append("")
    md.append("1. The additive rich benchmark is now complete: proper `openSMILE` acoustics, archive-aligned model families, nonlinear SVM coverage, and held-out permutation importance for every best model.")
    md.append("2. The benchmark-wide winner benefits from nonlinear interactions, but the strongest interpretable result is still the pooled `PD_CTP` slice.")
    md.append("3. Cross-lingual stability is stronger for broad lexical/discourse patterns than for a single shared `PD_CTP` biomarker set.")
    md.append("4. The next scientifically meaningful step is Phase 2 task-specific semantic extraction, especially content-unit coverage for `PD_CTP` and separate semantic modules for `READING`, `FLUENCY`, and `CONVERSATION`.")
    md.append("")
    md.append("## What else from Balagopalan is still worth trying")
    md.append("")
    md.append("- `Gaussian NB` and the shallow `NN` baseline. Those were part of the original ADReSS comparison and would complete the classical-model family beyond the current `LR / DT / RF / SVM` set.")
    md.append("- `LOSO` or grouped leave-one-subject-out validation on a cleaner matched subset. Balagopalan reported both held-out and cross-validation settings; for small task-specific slices this could expose instability better than a single split.")
    md.append("- `BERT + handcrafted feature fusion` as a secondary comparison, not as the interpretability core.")
    md.append("")

    report_path.write_text("\n".join(md), encoding="utf-8")


def main():
    log = make_logger("phase1_report_artifacts")
    log("Loading merged phase1 data")
    df = load_merged()
    metadata = pd.read_csv("data/processed/phase1/phase1_feature_metadata.csv")

    log("Reconstructing pooled PD_CTP and English PD_CTP best runs")
    benchmark_state = fit_run(
        df,
        filters={},
        grouping_levels=[["language", "task_type", "dataset_name"], ["language", "task_type"], ["language"], []],
        summary_name="benchmark_wide_rich",
    )
    pd_ctp_state = fit_run(
        df,
        filters={"task_type": "PD_CTP"},
        grouping_levels=[["language", "dataset_name"], ["language"], []],
        summary_name="pd_ctp_pooled_rich",
    )
    en_pd_ctp_state = fit_run(
        df,
        filters={"language": "en", "task_type": "PD_CTP"},
        grouping_levels=[["dataset_name"], []],
        summary_name="language_en_pd_ctp_rich",
    )
    zh_state = fit_run(
        df,
        filters={"language": "zh", "task_type": "PD_CTP"},
        grouping_levels=[["dataset_name"], []],
        summary_name="language_zh_pd_ctp_rich",
    )
    el_state = fit_run(
        df,
        filters={"language": "el", "task_type": "PD_CTP"},
        grouping_levels=[["dataset_name"], []],
        summary_name="language_el_pd_ctp_rich",
    )
    es_state = fit_run(
        df,
        filters={"language": "es", "task_type": "READING"},
        grouping_levels=[["dataset_name"], []],
        summary_name="language_es_reading_rich",
    )

    log("Selecting a well-classified English PD_CTP AD/HC pair")
    pair = choose_saliency_pair(en_pd_ctp_state)
    safe_to_csv(pair, REPORT_ROOT / "english_pd_ctp_selected_pair.csv")

    log("Generating patient-level saliency figure")
    saliency_df = make_saliency_plot(en_pd_ctp_state, pair, REPORT_ROOT / "english_pd_ctp_saliency_map.png")

    log("Building worked example table")
    example_df = example_feature_table(REPORT_ROOT / "feature_example_table.csv")

    log("Building feature family summary tables")
    lexical_df, acoustic_df, semantic_df = summarize_feature_inventory(metadata)
    safe_to_csv(lexical_df, REPORT_ROOT / "feature_inventory_lexicosyntactic.csv")
    safe_to_csv(acoustic_df, REPORT_ROOT / "feature_inventory_acoustic.csv")
    safe_to_csv(semantic_df, REPORT_ROOT / "feature_inventory_semantic_status.csv")

    log("Building pooled PD_CTP feature differentiation table")
    differentiation_df = feature_differentiation_table(pd_ctp_state, REPORT_ROOT / "pd_ctp_feature_differentiation.csv")

    log("Building pooled PD_CTP t-SNE / UMAP projections")
    projection_df = projection_plots(pd_ctp_state, differentiation_df)

    log("Building language-specific projection comparisons")
    language_projection_df = language_projection_comparisons(
        df,
        benchmark_state,
        pd_ctp_state,
        en_pd_ctp_state,
        zh_state,
        el_state,
        es_state,
    )

    log("Building ANOVA summary tables")
    benchmark_anova_df = pd.read_csv(RICH_SWEEP_ROOT / "benchmark_wide_rich_anova_ranking.csv").head(15)[
        ["feature_name", "f_score", "p_value"]
    ]
    pd_ctp_anova_df = pd.read_csv(RICH_SWEEP_ROOT / "pd_ctp_pooled_rich_anova_ranking.csv").head(15)[
        ["feature_name", "f_score", "p_value"]
    ]
    safe_to_csv(benchmark_anova_df, REPORT_ROOT / "benchmark_wide_top_anova.csv")
    safe_to_csv(pd_ctp_anova_df, REPORT_ROOT / "pd_ctp_pooled_top_anova.csv")

    log("Building best-by-method result tables")
    benchmark_results = pd.read_csv(RICH_SWEEP_ROOT / "benchmark_wide_rich_model_results.csv")
    pd_ctp_results = pd.read_csv(RICH_SWEEP_ROOT / "pd_ctp_pooled_rich_model_results.csv")

    def best_by_method(df: pd.DataFrame) -> pd.DataFrame:
        out = (
            df.sort_values(["balanced_accuracy", "auroc", "macro_f1"], ascending=False)
            .groupby(["model_family", "model_variant"], as_index=False)
            .first()
        )
        out = out[
            [
                "model_family",
                "model_variant",
                "subset",
                "top_k",
                "num_features",
                "balanced_accuracy",
                "precision",
                "sensitivity",
                "specificity",
                "macro_f1",
                "auroc",
                "auprc",
            ]
        ].copy()
        out = out.rename(
            columns={
                "model_family": "model",
                "model_variant": "variant",
                "subset": "feature_subset",
                "sensitivity": "recall",
            }
        )
        return out

    benchmark_method_df = best_by_method(benchmark_results)
    pd_ctp_method_df = best_by_method(pd_ctp_results)
    safe_to_csv(benchmark_method_df, REPORT_ROOT / "benchmark_wide_best_by_method.csv")
    safe_to_csv(pd_ctp_method_df, REPORT_ROOT / "pd_ctp_pooled_best_by_method.csv")

    log("Writing report markdown")
    build_report(
        example_df,
        lexical_df,
        acoustic_df,
        semantic_df,
        differentiation_df,
        benchmark_anova_df,
        pd_ctp_anova_df,
        benchmark_method_df,
        pd_ctp_method_df,
        language_projection_df,
        pair,
        saliency_df,
    )
    log("Finished generating report artifacts")


if __name__ == "__main__":
    main()
