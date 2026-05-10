from __future__ import annotations

import csv
from pathlib import Path

from generate_accuracy_tables import RESULT_TABLES


SCRIPT_DIR = Path(__file__).resolve().parent
SUITE_DIR = SCRIPT_DIR.parent / "tables" / "experiment-results" / "multiseed-suite"
SUMMARY_DIR = SUITE_DIR / "summaries"
OUTPUT_PATH = SUITE_DIR / "paper_vs_ours_3tables.txt"

LANG_ORDER = ["Spanish", "Chinese", "Greek", "English"]
TASK_ORDER = [("binary", "Binary"), ("multiclass", "Multiclass")]
REPR_ORDER = [("Sparse", "tfidf"), ("Dense", "e5")]
SETTING_ORDER = [
    ("monolingual", "mono", "Monolingual"),
    ("multilingual_combined", "multi", "Multilingual-Combined"),
    ("translated_combined", "trans", "Translated-Combined"),
]
CLASSIFIERS = ["DT", "RF", "SVM", "LR"]
ENSEMBLE_COLUMNS = [
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
CLASSIFIER_ORDER = {name: idx for idx, name in enumerate(CLASSIFIERS)}


def load_summary_rows(method_key: str, setting_name: str, task_name: str) -> dict[str, dict[str, str]]:
    path = SUMMARY_DIR / f"{method_key}_{setting_name}_{task_name}.csv"
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["Language"]] = row
    return rows


def pct(value: float) -> str:
    return f"{value * 100:.1f}"


def fmt_pair(paper_value: float, ours_value: str) -> str:
    return f"{pct(paper_value)} / {ours_value}"


def best_paper_classifier(task_name: str, language: str, repr_name: str, mode_name: str) -> tuple[str, float]:
    values = RESULT_TABLES[task_name][language][repr_name][mode_name]
    scored = list(zip(CLASSIFIERS, values))
    best_name, (best_value, _) = max(scored, key=lambda item: item[1][0])
    return best_name, best_value


def best_ensemble(row: dict[str, str]) -> tuple[str, str]:
    best_name = ""
    best_score = -1.0
    best_value = ""
    for col in ENSEMBLE_COLUMNS:
        value = row[col]
        mean_str = value.split("+/-", 1)[0].strip()
        mean_val = float(mean_str)
        if mean_val > best_score:
            best_score = mean_val
            best_name = col
            best_value = value
    return best_name, best_value


def format_combo(combo_name: str) -> str:
    parts = combo_name.split("+")
    parts.sort(key=lambda part: CLASSIFIER_ORDER[part])
    return " + ".join(parts)


def build_table(setting_name: str, mode_name: str, title: str) -> str:
    blocks: list[str] = [title, "=" * len(title)]
    note = "Cells are `paper / ours`; paper values are single accuracies, ours are 5-seed mean +/- sd."
    blocks.append(note)
    blocks.append("")

    widths = {
        "Task": 10,
        "Language": 9,
        "Repr": 7,
        "DT": 22,
        "RF": 22,
        "SVM": 22,
        "LR": 22,
        "Best Ensemble": 24,
        "Best Combo": 28,
    }
    headers = list(widths.keys())
    header_line = " | ".join(name.ljust(widths[name]) for name in headers)
    sep_line = "-+-".join("-" * widths[name] for name in headers)
    blocks.append(header_line)
    blocks.append(sep_line)

    cached_rows: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for _, method_key in REPR_ORDER:
        for task_name, _ in TASK_ORDER:
            cached_rows[(method_key, task_name)] = load_summary_rows(method_key, setting_name, task_name)

    for task_name, task_label in TASK_ORDER:
        for language in LANG_ORDER:
            for repr_name, method_key in REPR_ORDER:
                ours_row = cached_rows[(method_key, task_name)][language]
                paper_cells = RESULT_TABLES[task_name][language][repr_name][mode_name]
                paper_map = {clf: val for clf, val in zip(CLASSIFIERS, paper_cells)}
                ours_best_combo, ours_best_ensemble = best_ensemble(ours_row)

                row = {
                    "Task": task_label,
                    "Language": language,
                    "Repr": repr_name,
                    "DT": fmt_pair(paper_map["DT"][0], ours_row["DT"]),
                    "RF": fmt_pair(paper_map["RF"][0], ours_row["RF"]),
                    "SVM": fmt_pair(paper_map["SVM"][0], ours_row["SVM"]),
                    "LR": fmt_pair(paper_map["LR"][0], ours_row["LR"]),
                    "Best Ensemble": ours_best_ensemble,
                    "Best Combo": format_combo(ours_best_combo),
                }
                blocks.append(" | ".join(row[name].ljust(widths[name]) for name in headers))
        blocks.append("")

    return "\n".join(blocks).rstrip() + "\n"


def main() -> None:
    sections = []
    for setting_name, mode_name, title in SETTING_ORDER:
        sections.append(build_table(setting_name, mode_name, title))
    OUTPUT_PATH.write_text("\n\n".join(sections), encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
