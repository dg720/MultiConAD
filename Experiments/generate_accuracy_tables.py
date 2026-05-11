"""Write the accuracy and delta tables used in the paper-comparison notes.

This script emits two outputs:
1. results/accuracy_tables.txt
   LaTeX tables for binary + multiclass accuracy and delta tables.
2. results/tfidf_comparison_tables.txt
   Plain-text table dump with the same values for quick inspection.

The values below are the reviewed table values to keep in sync with the
paper-comparison screenshots used in this repo.
"""

from __future__ import annotations

import csv
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "tables", "experiment-results")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "multiseed-suite", "summaries")

ACCURACY_TEX_PATH = os.path.join(RESULTS_DIR, "accuracy_tables.txt")
PLAIN_TXT_PATH = os.path.join(RESULTS_DIR, "tfidf_comparison_tables.txt")

LANGS = ["Spanish", "Chinese", "Greek", "English"]
REPRS = ["Sparse", "Dense"]
CLFS = ["DT", "RF", "SVM", "LR"]
MODES = ["mono", "multi", "trans"]
SUMMARY_METHODS = {"Sparse": "tfidf", "Dense": "e5"}
SUMMARY_SETTINGS = {
    "mono": "monolingual",
    "multi": "multilingual_combined",
    "trans": "translated_combined",
}


def cell(value: float, arrow: str = "") -> tuple[float, str]:
    return (value, arrow)


RESULT_TABLES = {
    "binary": {
        "Spanish": {
            "Sparse": {
                "mono":  [cell(0.73), cell(0.73), cell(0.78), cell(0.78)],
                "multi": [cell(0.80, "up"), cell(0.76, "up"), cell(0.66, "down"), cell(0.80, "up")],
                "trans": [cell(0.75, "up"), cell(0.80, "up"), cell(0.75, "down"), cell(0.73, "down")],
            },
            "Dense": {
                "mono":  [cell(0.71), cell(0.71), cell(0.80), cell(0.80)],
                "multi": [cell(0.66, "down"), cell(0.78, "up"), cell(0.66, "down"), cell(0.80)],
                "trans": [cell(0.59, "down"), cell(0.76, "up"), cell(0.78, "down"), cell(0.76, "down")],
            },
        },
        "Chinese": {
            "Sparse": {
                "mono":  [cell(0.67), cell(0.69), cell(0.70), cell(0.70)],
                "multi": [cell(0.67), cell(0.69), cell(0.67, "down"), cell(0.69, "down")],
                "trans": [cell(0.70, "up"), cell(0.90, "up"), cell(0.89, "up"), cell(0.84, "up")],
            },
            "Dense": {
                "mono":  [cell(0.69), cell(0.76), cell(0.83), cell(0.80)],
                "multi": [cell(0.76, "up"), cell(0.86, "up"), cell(0.67, "down"), cell(0.81, "up")],
                "trans": [cell(0.66, "down"), cell(0.84, "up"), cell(0.80, "down"), cell(0.84, "up")],
            },
        },
        "Greek": {
            "Sparse": {
                "mono":  [cell(0.68), cell(0.78), cell(0.77), cell(0.78)],
                "multi": [cell(0.68), cell(0.76, "down"), cell(0.60, "down"), cell(0.73, "down")],
                "trans": [cell(0.58, "down"), cell(0.67, "down"), cell(0.69, "down"), cell(0.51, "down")],
            },
            "Dense": {
                "mono":  [cell(0.65), cell(0.75), cell(0.78), cell(0.77)],
                "multi": [cell(0.64, "down"), cell(0.75), cell(0.75, "down"), cell(0.73, "down")],
                "trans": [cell(0.62, "down"), cell(0.66, "down"), cell(0.70, "down"), cell(0.64, "down")],
            },
        },
        "English": {
            "Sparse": {
                "mono":  [cell(0.73), cell(0.78), cell(0.77), cell(0.75)],
                "multi": [cell(0.67, "down"), cell(0.74, "down"), cell(0.58, "down"), cell(0.75)],
                "trans": [cell(0.73), cell(0.74, "down"), cell(0.76, "down"), cell(0.61, "down")],
            },
            "Dense": {
                "mono":  [cell(0.65), cell(0.75), cell(0.81), cell(0.79)],
                "multi": [cell(0.67, "up"), cell(0.77, "up"), cell(0.58, "down"), cell(0.67, "down")],
                "trans": [cell(0.71, "up"), cell(0.73, "down"), cell(0.83, "up"), cell(0.70, "down")],
            },
        },
    },
    "multiclass": {
        "Spanish": {
            "Sparse": {
                "mono":  [cell(0.61), cell(0.60), cell(0.61), cell(0.61)],
                "multi": [cell(0.51, "down"), cell(0.62, "up"), cell(0.51, "down"), cell(0.58, "down")],
                "trans": [cell(0.47, "down"), cell(0.58, "down"), cell(0.56, "down"), cell(0.56, "down")],
            },
            "Dense": {
                "mono":  [cell(0.52), cell(0.61), cell(0.61), cell(0.61)],
                "multi": [cell(0.47, "down"), cell(0.61), cell(0.61), cell(0.57, "down")],
                "trans": [cell(0.60, "up"), cell(0.58, "down"), cell(0.56, "down"), cell(0.51, "down")],
            },
        },
        "Chinese": {
            "Sparse": {
                "mono":  [cell(0.36), cell(0.35), cell(0.40), cell(0.39)],
                "multi": [cell(0.42, "up"), cell(0.39, "up"), cell(0.39, "down"), cell(0.40, "up")],
                "trans": [cell(0.45, "up"), cell(0.59, "up"), cell(0.68, "up"), cell(0.62, "up")],
            },
            "Dense": {
                "mono":  [cell(0.51), cell(0.58), cell(0.59), cell(0.56)],
                "multi": [cell(0.43, "down"), cell(0.62, "up"), cell(0.60, "up"), cell(0.60, "up")],
                "trans": [cell(0.43, "down"), cell(0.64, "up"), cell(0.60, "up"), cell(0.45, "down")],
            },
        },
        "Greek": {
            "Sparse": {
                "mono":  [cell(0.59), cell(0.74), cell(0.67), cell(0.71)],
                "multi": [cell(0.57, "down"), cell(0.71, "down"), cell(0.53, "down"), cell(0.66, "down")],
                "trans": [cell(0.64, "up"), cell(0.65, "down"), cell(0.69, "up"), cell(0.60, "down")],
            },
            "Dense": {
                "mono":  [cell(0.54), cell(0.66), cell(0.73), cell(0.73)],
                "multi": [cell(0.54), cell(0.66), cell(0.65, "down"), cell(0.67, "down")],
                "trans": [cell(0.62, "up"), cell(0.61, "down"), cell(0.60, "down"), cell(0.42, "down")],
            },
        },
        "English": {
            "Sparse": {
                "mono":  [cell(0.59), cell(0.62), cell(0.65), cell(0.65)],
                "multi": [cell(0.59), cell(0.58, "down"), cell(0.41, "down"), cell(0.66, "up")],
                "trans": [cell(0.50, "down"), cell(0.61, "down"), cell(0.66, "up"), cell(0.64, "down")],
            },
            "Dense": {
                "mono":  [cell(0.51), cell(0.62), cell(0.65), cell(0.63)],
                "multi": [cell(0.50, "down"), cell(0.62), cell(0.65), cell(0.63)],
                "trans": [cell(0.50, "down"), cell(0.57, "down"), cell(0.66, "up"), cell(0.41, "down")],
            },
        },
    },
}


def load_summary_rows(method_key: str, setting_key: str, task_key: str) -> dict[str, dict[str, str]]:
    path = os.path.join(SUMMARY_DIR, f"{method_key}_{SUMMARY_SETTINGS[setting_key]}_{task_key}.csv")
    rows: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["Language"]] = row
    return rows


def parse_mean_percent(value: str) -> float:
    return float(value.split("+/-", 1)[0].strip()) / 100.0


def build_delta_tables() -> dict[str, dict[str, dict[str, dict[str, list[float]]]]]:
    delta_tables: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    summary_cache: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}

    for task_key in RESULT_TABLES:
        delta_tables[task_key] = {}
        for repr_label in REPRS:
            method_key = SUMMARY_METHODS[repr_label]
            for mode_key in MODES:
                summary_cache[(task_key, repr_label, mode_key)] = load_summary_rows(method_key, mode_key, task_key)

        for lang in LANGS:
            delta_tables[task_key][lang] = {}
            for repr_label in REPRS:
                delta_tables[task_key][lang][repr_label] = {}
                for mode_key in MODES:
                    ours_row = summary_cache[(task_key, repr_label, mode_key)][lang]
                    paper_cells = RESULT_TABLES[task_key][lang][repr_label][mode_key]
                    deltas = [
                        round(parse_mean_percent(ours_row[clf]) - paper_cells[idx][0], 2)
                        for idx, clf in enumerate(CLFS)
                    ]
                    delta_tables[task_key][lang][repr_label][mode_key] = deltas

    return delta_tables


DELTA_TABLES = build_delta_tables()


PREAMBLE = r"""\usepackage{booktabs}
\usepackage{multirow}
\newcommand{\dpos}[1]{{\color{green!60!black}$+$#1}}
\newcommand{\dneg}[1]{{\color{red!70!black}$-$#1}}
\newcommand{\dzero}{{\color{gray}$\pm$0.00}}
"""


def arrow_tex(arrow: str) -> str:
    if arrow == "up":
        return r"$^{\uparrow}$"
    if arrow == "down":
        return r"$^{\downarrow}$"
    return ""


def arrow_txt(arrow: str) -> str:
    return "^" if arrow == "up" else ("v" if arrow == "down" else "")


def delta_tex(value: float) -> str:
    if round(value, 2) > 0.005:
        return r"\dpos{%.2f}" % round(value, 2)
    if round(value, 2) < -0.005:
        return r"\dneg{%.2f}" % abs(round(value, 2))
    return r"\dzero"


def format_result_cells_tex(cells: list[tuple[float, str]]) -> list[str]:
    max_val = max(value for value, _ in cells)
    parts = []
    for value, arrow in cells:
        rendered = f"{value:.2f}"
        if abs(value - max_val) < 1e-9:
            rendered = r"\textbf{" + rendered + "}"
        rendered += arrow_tex(arrow)
        parts.append(rendered)
    return parts


def format_result_cells_txt(cells: list[tuple[float, str]]) -> list[str]:
    max_val = max(value for value, _ in cells)
    parts = []
    for value, arrow in cells:
        rendered = f"{value:.2f}"
        if abs(value - max_val) < 1e-9:
            rendered = f"*{rendered}*"
        rendered += arrow_txt(arrow)
        parts.append(rendered)
    return parts


def make_latex_result_table(task: str, caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{ll cccc cccc cccc}",
        r"\toprule",
        r"& &",
        r"\multicolumn{4}{c}{\textbf{Monolingual}} &",
        r"\multicolumn{4}{c}{\textbf{Combined-Multilingual}} &",
        r"\multicolumn{4}{c}{\textbf{Combined-Translated}} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
        r"\textbf{Language} & \textbf{Repr.} &",
        r"DT & RF & SVM & LR &",
        r"DT & RF & SVM & LR &",
        r"DT & RF & SVM & LR \\",
        r"\midrule",
    ]
    for lang in LANGS:
        for idx, repr_label in enumerate(REPRS):
            lang_col = r"\multirow{2}{*}{" + lang + "}" if idx == 0 else ""
            mono = " & ".join(format_result_cells_tex(RESULT_TABLES[task][lang][repr_label]["mono"]))
            multi = " & ".join(format_result_cells_tex(RESULT_TABLES[task][lang][repr_label]["multi"]))
            trans = " & ".join(format_result_cells_tex(RESULT_TABLES[task][lang][repr_label]["trans"]))
            sep = r"\\[3pt]" if repr_label == "Dense" and lang != LANGS[-1] else r"\\"
            lines.append(
                f"{lang_col}\n& {repr_label}\n"
                f"  & {mono}\n"
                f"  & {multi}\n"
                f"  & {trans} {sep}"
            )
        lines.append("")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def make_latex_delta_table(task: str, caption: str, label: str) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{ll cccc cccc cccc}",
        r"\toprule",
        r"& &",
        r"\multicolumn{4}{c}{\textbf{Monolingual}} &",
        r"\multicolumn{4}{c}{\textbf{Combined-Multilingual}} &",
        r"\multicolumn{4}{c}{\textbf{Combined-Translated}} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}",
        r"\textbf{Language} & \textbf{Repr.} &",
        r"DT & RF & SVM & LR &",
        r"DT & RF & SVM & LR &",
        r"DT & RF & SVM & LR \\",
        r"\midrule",
    ]
    for lang in LANGS:
        for idx, repr_label in enumerate(REPRS):
            lang_col = r"\multirow{2}{*}{" + lang + "}" if idx == 0 else ""
            mono = " & ".join(delta_tex(v) for v in DELTA_TABLES[task][lang][repr_label]["mono"])
            multi = " & ".join(delta_tex(v) for v in DELTA_TABLES[task][lang][repr_label]["multi"])
            trans = " & ".join(delta_tex(v) for v in DELTA_TABLES[task][lang][repr_label]["trans"])
            sep = r"\\[3pt]" if repr_label == "Dense" and lang != LANGS[-1] else r"\\"
            lines.append(
                f"{lang_col}\n& {repr_label}\n"
                f"  & {mono}\n"
                f"  & {multi}\n"
                f"  & {trans} {sep}"
            )
        lines.append("")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def row_txt(parts: list[str], widths: list[int]) -> str:
    return " ".join(part.ljust(width) for part, width in zip(parts, widths))


def write_plain_text() -> None:
    widths = [10, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
    with open(PLAIN_TXT_PATH, "w", encoding="utf-8") as f:
        for task, table_no, title in [
            ("binary", "Table 6", "Binary MultiConAD accuracy replication, arrows compare against monolingual training."),
            ("multiclass", "Table 8", "Multiclass MultiConAD accuracy replication, arrows compare against monolingual training."),
        ]:
            f.write(f"{table_no}: {title}\n")
            f.write("=" * 118 + "\n")
            f.write(row_txt(["Language", "Repr.", "DT", "RF", "SVM", "LR", "DT", "RF", "SVM", "LR", "DT", "RF", "SVM", "LR"], widths) + "\n")
            f.write(row_txt(["", "", "Monolingual", "", "", "", "Combined-Multilingual", "", "", "", "Combined-Translated", "", "", ""], widths) + "\n")
            f.write("-" * 118 + "\n")
            for lang in LANGS:
                for idx, repr_label in enumerate(REPRS):
                    prefix = [lang if idx == 0 else "", repr_label]
                    parts = (
                        format_result_cells_txt(RESULT_TABLES[task][lang][repr_label]["mono"])
                        + format_result_cells_txt(RESULT_TABLES[task][lang][repr_label]["multi"])
                        + format_result_cells_txt(RESULT_TABLES[task][lang][repr_label]["trans"])
                    )
                    f.write(row_txt(prefix + parts, widths) + "\n")
                f.write("\n")
            f.write("\n")

            delta_title = "Binary accuracy deltas against Shakeri et al. green improves, red underperforms." if task == "binary" else "Multiclass accuracy deltas against Shakeri et al. green improves, red underperforms."
            delta_no = "Table 7" if task == "binary" else "Table 9"
            f.write(f"{delta_no}: {delta_title}\n")
            f.write("=" * 118 + "\n")
            f.write(row_txt(["Language", "Repr.", "DT", "RF", "SVM", "LR", "DT", "RF", "SVM", "LR", "DT", "RF", "SVM", "LR"], widths) + "\n")
            f.write(row_txt(["", "", "Monolingual", "", "", "", "Combined-Multilingual", "", "", "", "Combined-Translated", "", "", ""], widths) + "\n")
            f.write("-" * 118 + "\n")
            for lang in LANGS:
                for idx, repr_label in enumerate(REPRS):
                    prefix = [lang if idx == 0 else "", repr_label]
                    vals = DELTA_TABLES[task][lang][repr_label]["mono"] + DELTA_TABLES[task][lang][repr_label]["multi"] + DELTA_TABLES[task][lang][repr_label]["trans"]
                    parts = [f"{v:+.2f}" if abs(v) > 0.0001 else "0.00" for v in vals]
                    f.write(row_txt(prefix + parts, widths) + "\n")
                f.write("\n")
            f.write("\n\n")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(ACCURACY_TEX_PATH, "w", encoding="utf-8") as f:
        f.write("% ============================================================\n")
        f.write("% Preamble commands (add to your document once)\n")
        f.write("% ============================================================\n")
        f.write(PREAMBLE + "\n\n")
        f.write("% ============================================================\n")
        f.write("% TABLE 6\n")
        f.write("% ============================================================\n")
        f.write(
            make_latex_result_table(
                "binary",
                "Table 6: Binary MultiConAD accuracy replication, arrows compare against monolingual training.",
                "tab:multiconad_binary_acc",
            )
            + "\n\n"
        )
        f.write("% ============================================================\n")
        f.write("% TABLE 7\n")
        f.write("% ============================================================\n")
        f.write(
            make_latex_delta_table(
                "binary",
                "Table 7: Binary accuracy deltas against Shakeri et al. green improves, red underperforms.",
                "tab:multiconad_binary_acc_delta",
            )
            + "\n\n"
        )
        f.write("% ============================================================\n")
        f.write("% TABLE 8\n")
        f.write("% ============================================================\n")
        f.write(
            make_latex_result_table(
                "multiclass",
                "Table 8: Multiclass MultiConAD accuracy replication, arrows compare against monolingual training.",
                "tab:multiconad_multiclass_acc",
            )
            + "\n\n"
        )
        f.write("% ============================================================\n")
        f.write("% TABLE 9\n")
        f.write("% ============================================================\n")
        f.write(
            make_latex_delta_table(
                "multiclass",
                "Table 9: Multiclass accuracy deltas against Shakeri et al. green improves, red underperforms.",
                "tab:multiconad_multiclass_acc_delta",
            )
            + "\n"
        )
    write_plain_text()
    print(f"Written to {ACCURACY_TEX_PATH}")
    print(f"Written to {PLAIN_TXT_PATH}")


if __name__ == "__main__":
    main()
