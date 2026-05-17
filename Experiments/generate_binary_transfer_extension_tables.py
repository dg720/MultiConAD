from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EMBED_SUMMARY = ROOT / "tables" / "01-baselines" / "embedding-baselines" / "multiseed-suite" / "summaries"
TRANSFER_CSV = (
    ROOT
    / "tables"
    / "01-baselines"
    / "transfer-learning-baselines"
    / "result-tables"
    / "csv"
    / "frozen_embedding_runs.csv"
)
OUT_PATH = ROOT / "reports" / "20.05" / "binary_transfer_extension_tables.txt"
PAPER_VS_OURS_PATH = (
    ROOT
    / "tables"
    / "01-baselines"
    / "embedding-baselines"
    / "multiseed-suite"
    / "result-tables"
    / "paper_vs_ours_3tables.txt"
)

LANGUAGES = ["Spanish", "Chinese", "Greek", "English"]
CLASSIFIERS = ["DT", "RF", "SVM", "LR"]
ENSEMBLE_COLS = [
    "DT+RF",
    "DT+SVM",
    "DT+LR",
    "RF+SVM",
    "RF+LR",
    "SVM+LR",
    "DT+RF+SVM",
    "DT+RF+LR",
    "DT+SVM+LR",
    "RF+SVM+LR",
    "DT+RF+SVM+LR",
]
TABLES = [
    ("Monolingual", "monolingual_binary", "mono", "no", "Table 9"),
    ("Combined-multilingual", "multilingual_combined_binary", "multi", "no", "Table 10"),
    ("Combined-translated", "translated_combined_binary", "multi", "yes", "Table 11"),
]


def parse_mean(cell: str) -> float:
    match = re.match(r"\s*([0-9.]+)", str(cell))
    if not match:
        raise ValueError(f"Cannot parse metric cell: {cell}")
    return float(match.group(1))


def best_ensemble(row: pd.Series) -> str:
    best_col = max(ENSEMBLE_COLS, key=lambda col: parse_mean(row[col]))
    return str(row[best_col])


def load_embedding_rows(method: str, stem: str) -> dict[str, dict[str, str]]:
    df = pd.read_csv(EMBED_SUMMARY / f"{method}_{stem}.csv")
    rows = {}
    for _, row in df.iterrows():
        rows[str(row["Language"])] = {
            "DT": str(row["DT"]),
            "RF": str(row["RF"]),
            "SVM": str(row["SVM"]),
            "LR": str(row["LR"]),
            "Best Ensemble": best_ensemble(row),
        }
    return rows


def load_paper_vs_ours_best_ensembles() -> dict[tuple[str, str, str], str]:
    text = PAPER_VS_OURS_PATH.read_text(encoding="utf-8")
    section = None
    out = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line in {"Monolingual", "Multilingual-Combined", "Translated-Combined"}:
            section = line
            continue
        if not section or not line.startswith("Binary"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 8:
            continue
        _, language, repr_name, dt, rf, svm, lr, best_ensemble, _best_combo = parts[:9]
        paper_values = []
        for cell in [dt, rf, svm, lr]:
            paper_values.append(float(cell.split("/", 1)[0].strip()))
        out[(section, language, repr_name)] = f"{max(paper_values):.1f} / {best_ensemble}"
    return out


def fmt_metric(mean: float, std: float) -> str:
    return f"{mean * 100.0:.1f} +/- {std * 100.0:.1f}"


def load_xlmr_rows(training: str, translated: str, pooling: str) -> dict[str, dict[str, str]]:
    df = pd.read_csv(TRANSFER_CSV)
    df = df[
        (df["smoke"].astype(str).str.lower() == "false")
        & (df["model"] == "xlm-roberta-base")
        & (df["pooling"] == pooling)
        & (df["length_mode"] == "truncate")
        & (df["task"] == "binary")
        & (df["training"] == training)
        & (df["translated"].astype(str) == translated)
    ].copy()
    if df.empty:
        raise RuntimeError(
            f"No XLM-R rows for training={training} translated={translated} pooling={pooling}"
        )

    rows = {}
    for language in LANGUAGES:
        part = df[df["language_label"] == language]
        if part["seed"].nunique() != 5:
            raise RuntimeError(
                f"Incomplete XLM-R rows for {language}, pooling={pooling}: "
                f"seeds={part['seed'].nunique()}"
            )
        out = {}
        for clf in CLASSIFIERS:
            vals = part[part["classifier"] == clf]["accuracy"].astype(float)
            out[clf] = fmt_metric(vals.mean(), vals.std(ddof=0))
        vals = part[part["classifier"] == "Best Ensemble"]["accuracy"].astype(float)
        out["Best Ensemble"] = fmt_metric(vals.mean(), vals.std(ddof=0))
        rows[language] = out
    return rows


def row_best_classifier(row: dict[str, str]) -> str:
    return max(CLASSIFIERS, key=lambda clf: parse_mean(row[clf]))


def widths_for(rows: list[dict[str, str]]) -> dict[str, int]:
    columns = ["Language", "Text Repr.", "DT", "RF", "SVM", "LR", "Best Ensemble"]
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row[col])))
    return widths


def render_table(title: str, stem: str, training: str, translated: str, number: str) -> str:
    sparse = load_embedding_rows("tfidf", stem)
    dense = load_embedding_rows("e5", stem)
    xlmr_cls = load_xlmr_rows(training, translated, "cls")
    xlmr_mean = load_xlmr_rows(training, translated, "mean")
    paper_vs_ours = load_paper_vs_ours_best_ensembles()
    section_name = {
        "Monolingual": "Monolingual",
        "Combined-multilingual": "Multilingual-Combined",
        "Combined-translated": "Translated-Combined",
    }[title]

    rows = []
    for language in LANGUAGES:
        sparse_row = sparse[language].copy()
        dense_row = dense[language].copy()
        sparse_row["Best Ensemble"] = paper_vs_ours[(section_name, language, "Sparse")]
        dense_row["Best Ensemble"] = paper_vs_ours[(section_name, language, "Dense")]
        rows.append({"Language": language, "Text Repr.": "Sparse", **sparse[language]})
        rows[-1]["Best Ensemble"] = sparse_row["Best Ensemble"]
        rows.append({"Language": "", "Text Repr.": "Dense", **dense_row})
        xlm_cls_row = xlmr_cls[language].copy()
        xlm_cls_row["Best Ensemble"] = f"-- / {xlm_cls_row['Best Ensemble']}"
        rows.append({"Language": "", "Text Repr.": "xlm-roberta-base CLS", **xlm_cls_row})
        xlm_mean_row = xlmr_mean[language].copy()
        xlm_mean_row["Best Ensemble"] = f"-- / {xlm_mean_row['Best Ensemble']}"
        rows.append({"Language": "", "Text Repr.": "xlm-roberta-base mean", **xlm_mean_row})

    columns = ["Language", "Text Repr.", "DT", "RF", "SVM", "LR", "Best Ensemble"]
    marked_rows = []
    for idx, language in enumerate(LANGUAGES):
        group = rows[idx * 4 : idx * 4 + 4]
        best = max(
            ((repr_row, row_best_classifier(repr_row)) for repr_row in group),
            key=lambda item: parse_mean(item[0][item[1]]),
        )
        for row in group:
            rendered = row.copy()
            best_clf = row_best_classifier(row)
            if row is best[0]:
                rendered[best_clf] = f"*{rendered[best_clf]}*"
            marked_rows.append(rendered)
        if idx < len(LANGUAGES) - 1:
            marked_rows.append({col: "" for col in columns})

    widths = widths_for(marked_rows)
    line_width = sum(widths[col] for col in columns) + 3 * (len(columns) - 1)
    lines = ["-" * line_width]
    lines.append(" | ".join(col.ljust(widths[col]) for col in columns))
    lines.append("-" * line_width)

    for row in marked_rows:
        if not any(row.values()):
            lines.append("")
            continue
        lines.append(" | ".join(str(row[col]).ljust(widths[col]) for col in columns))
    lines.append("-" * line_width)
    lines.append(
        f"{number}: {title} binary classification (AD vs. HC). "
        "Asterisks mark the best replication value in each language block; "
        "the final column shows best paper accuracy / our best ensemble mean +/- std "
        "for Sparse/Dense and our XLM-R ensemble mean +/- std for XLM-R rows."
    )
    return "\n".join(lines)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        render_table(
            title=title,
            stem=stem,
            training=training,
            translated=translated,
            number=number,
        )
        for title, stem, training, translated, number in TABLES
    ]
    OUT_PATH.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
