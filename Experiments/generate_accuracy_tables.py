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

import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

ACCURACY_TEX_PATH = os.path.join(RESULTS_DIR, "accuracy_tables.txt")
PLAIN_TXT_PATH = os.path.join(RESULTS_DIR, "tfidf_comparison_tables.txt")

LANGS = ["Spanish", "Chinese", "Greek", "English"]
REPRS = ["Sparse", "Dense"]
CLFS = ["DT", "RF", "SVM", "LR"]
MODES = ["mono", "multi", "trans"]


def cell(value: float, arrow: str = "") -> tuple[float, str]:
    return (value, arrow)


RESULT_TABLES = {
    "binary": {
        "Spanish": {
            "Sparse": {
                "mono":  [cell(0.71), cell(0.71), cell(0.81), cell(0.76)],
                "multi": [cell(0.76, "up"), cell(0.75, "up"), cell(0.34, "down"), cell(0.80, "up")],
                "trans": [cell(0.71), cell(0.75, "up"), cell(0.69, "down"), cell(0.68, "down")],
            },
            "Dense": {
                "mono":  [cell(0.75), cell(0.71), cell(0.78), cell(0.76)],
                "multi": [cell(0.66, "down"), cell(0.78, "up"), cell(0.73, "down"), cell(0.66, "down")],
                "trans": [cell(0.66, "down"), cell(0.71), cell(0.34, "down"), cell(0.66, "down")],
            },
        },
        "Chinese": {
            "Sparse": {
                "mono":  [cell(0.78), cell(0.80), cell(0.82), cell(0.75)],
                "multi": [cell(0.75, "down"), cell(0.78, "down"), cell(0.23, "down"), cell(0.72, "down")],
                "trans": [cell(0.68, "down"), cell(0.88, "up"), cell(0.82), cell(0.78, "up")],
            },
            "Dense": {
                "mono":  [cell(0.85), cell(0.93), cell(0.88), cell(0.93)],
                "multi": [cell(0.80, "down"), cell(0.90, "down"), cell(0.85, "down"), cell(0.78, "down")],
                "trans": [cell(0.75, "down"), cell(0.85, "down"), cell(0.78, "down"), cell(0.78, "down")],
            },
        },
        "Greek": {
            "Sparse": {
                "mono":  [cell(0.77), cell(0.67), cell(0.74), cell(0.74)],
                "multi": [cell(0.65, "down"), cell(0.58, "down"), cell(0.60, "down"), cell(0.72, "down")],
                "trans": [cell(0.51, "down"), cell(0.74, "up"), cell(0.60, "down"), cell(0.40, "down")],
            },
            "Dense": {
                "mono":  [cell(0.60), cell(0.67), cell(0.79), cell(0.70)],
                "multi": [cell(0.79, "up"), cell(0.70, "up"), cell(0.70, "down"), cell(0.40, "down")],
                "trans": [cell(0.60), cell(0.72, "up"), cell(0.63, "down"), cell(0.40, "down")],
            },
        },
        "English": {
            "Sparse": {
                "mono":  [cell(0.77), cell(0.79), cell(0.86), cell(0.83)],
                "multi": [cell(0.76, "down"), cell(0.81, "up"), cell(0.82, "down"), cell(0.86, "up")],
                "trans": [cell(0.78, "up"), cell(0.77, "down"), cell(0.86), cell(0.69, "down")],
            },
            "Dense": {
                "mono":  [cell(0.75), cell(0.78), cell(0.84), cell(0.86)],
                "multi": [cell(0.77, "up"), cell(0.76, "down"), cell(0.86, "up"), cell(0.69, "down")],
                "trans": [cell(0.71, "down"), cell(0.78), cell(0.77, "down"), cell(0.69, "down")],
            },
        },
    },
    "multiclass": {
        "Spanish": {
            "Sparse": {
                "mono":  [cell(0.47), cell(0.55), cell(0.63), cell(0.59)],
                "multi": [cell(0.45, "down"), cell(0.51, "down"), cell(0.58, "down"), cell(0.62, "up")],
                "trans": [cell(0.51, "up"), cell(0.57, "up"), cell(0.43, "down"), cell(0.57, "down")],
            },
            "Dense": {
                "mono":  [cell(0.49), cell(0.61), cell(0.62), cell(0.61)],
                "multi": [cell(0.53, "up"), cell(0.57, "down"), cell(0.24, "down"), cell(0.57, "down")],
                "trans": [cell(0.51, "up"), cell(0.55, "down"), cell(0.30, "down"), cell(0.61)],
            },
        },
        "Chinese": {
            "Sparse": {
                "mono":  [cell(0.45), cell(0.43), cell(0.41), cell(0.41)],
                "multi": [cell(0.47, "up"), cell(0.41, "down"), cell(0.42, "up"), cell(0.39, "down")],
                "trans": [cell(0.54, "up"), cell(0.59, "up"), cell(0.46, "up"), cell(0.43, "up")],
            },
            "Dense": {
                "mono":  [cell(0.61), cell(0.49), cell(0.49), cell(0.51)],
                "multi": [cell(0.53, "down"), cell(0.49), cell(0.47, "down"), cell(0.62, "up")],
                "trans": [cell(0.38, "down"), cell(0.53, "up"), cell(0.47, "down"), cell(0.55, "up")],
            },
        },
        "Greek": {
            "Sparse": {
                "mono":  [cell(0.49), cell(0.57), cell(0.62), cell(0.60)],
                "multi": [cell(0.43, "down"), cell(0.53, "down"), cell(0.55, "down"), cell(0.58, "down")],
                "trans": [cell(0.40, "down"), cell(0.68, "up"), cell(0.53, "down"), cell(0.51, "down")],
            },
            "Dense": {
                "mono":  [cell(0.40), cell(0.57), cell(0.66), cell(0.55)],
                "multi": [cell(0.43, "up"), cell(0.55, "down"), cell(0.49, "down"), cell(0.62, "up")],
                "trans": [cell(0.45, "up"), cell(0.57), cell(0.55, "down"), cell(0.57, "up")],
            },
        },
        "English": {
            "Sparse": {
                "mono":  [cell(0.54), cell(0.63), cell(0.68), cell(0.67)],
                "multi": [cell(0.56, "up"), cell(0.61, "down"), cell(0.67, "up"), cell(0.66, "down")],
                "trans": [cell(0.59, "up"), cell(0.65, "up"), cell(0.67, "down"), cell(0.56, "down")],
            },
            "Dense": {
                "mono":  [cell(0.54), cell(0.59), cell(0.48), cell(0.66)],
                "multi": [cell(0.55, "up"), cell(0.61, "up"), cell(0.51, "up"), cell(0.67, "up")],
                "trans": [cell(0.60, "up"), cell(0.60, "up"), cell(0.53, "up"), cell(0.65, "down")],
            },
        },
    },
}


DELTA_TABLES = {
    "binary": {
        "Spanish": {
            "Sparse": {"mono": [-0.02, -0.02, +0.03, -0.02], "multi": [-0.04, -0.01, -0.32,  0.00], "trans": [-0.04, -0.05, -0.06, -0.05]},
            "Dense":  {"mono": [+0.04,  0.00, -0.02, -0.04], "multi": [ 0.00,  0.00, +0.07, -0.14], "trans": [+0.07, -0.05, -0.44, -0.10]},
        },
        "Chinese": {
            "Sparse": {"mono": [+0.11, +0.11, +0.12, +0.05], "multi": [+0.08, -0.09, -0.44, -0.03], "trans": [-0.02, +0.02, -0.07, -0.06]},
            "Dense":  {"mono": [+0.16, +0.23, +0.05, +0.13], "multi": [+0.14, +0.04, +0.04, -0.03], "trans": [+0.09, +0.01, -0.02, -0.06]},
        },
        "Greek": {
            "Sparse": {"mono": [+0.09, -0.11, -0.03, -0.04], "multi": [-0.03, -0.18,  0.00, -0.01], "trans": [-0.07, +0.07, -0.09, -0.11]},
            "Dense":  {"mono": [-0.08, -0.08, +0.01, -0.07], "multi": [+0.15, -0.05, -0.05, -0.33], "trans": [-0.02, +0.06, -0.07, -0.24]},
        },
        "English": {
            "Sparse": {"mono": [+0.04, -0.02, +0.09, +0.08], "multi": [+0.09, +0.07, +0.24, -0.11], "trans": [+0.05, +0.03, +0.10, +0.08]},
            "Dense":  {"mono": [+0.10, +0.03, +0.03, +0.07], "multi": [+0.10, -0.01, +0.28, +0.02], "trans": [ 0.00, +0.05, -0.06, -0.01]},
        },
    },
    "multiclass": {
        "Spanish": {
            "Sparse": {"mono": [-0.14, -0.05, +0.02, -0.02], "multi": [-0.06, -0.11, +0.07, +0.04], "trans": [+0.04, -0.01, -0.13, +0.01]},
            "Dense":  {"mono": [-0.03,  0.00, +0.01,  0.00], "multi": [+0.06, -0.04, -0.37,  0.00], "trans": [-0.09, -0.03, -0.26, +0.10]},
        },
        "Chinese": {
            "Sparse": {"mono": [+0.09, +0.08, +0.01, +0.02], "multi": [+0.05, +0.02, +0.03, -0.01], "trans": [+0.09,  0.00, -0.22, -0.19]},
            "Dense":  {"mono": [+0.10, -0.09, -0.10, -0.05], "multi": [+0.10, -0.13, -0.13, +0.02], "trans": [-0.05, -0.11, -0.13, +0.10]},
        },
        "Greek": {
            "Sparse": {"mono": [-0.10, -0.17, -0.05, -0.11], "multi": [-0.14, -0.18, +0.02, -0.08], "trans": [-0.24, +0.03, -0.16, -0.09]},
            "Dense":  {"mono": [-0.14, -0.09, -0.07, -0.18], "multi": [-0.11, -0.11, -0.16, -0.05], "trans": [-0.17, -0.04, -0.05, +0.15]},
        },
        "English": {
            "Sparse": {"mono": [-0.05, -0.01, +0.03, +0.02], "multi": [-0.03, +0.03, +0.26,  0.00], "trans": [+0.09, +0.04,  0.00, -0.08]},
            "Dense":  {"mono": [+0.03, -0.03, -0.17, +0.03], "multi": [+0.05, -0.01, -0.14, +0.04], "trans": [+0.10, +0.03, -0.13, +0.24]},
        },
    },
}


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
