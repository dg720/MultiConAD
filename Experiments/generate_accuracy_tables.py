"""
Generate accuracy-based LaTeX tables.

Strategy:
  - Parse results files for per-classifier accuracy AND macro-f1.
  - Existing delta LaTeX encodes (our_macro_f1 - paper_accuracy).
  - paper_accuracy = our_macro_f1 - old_delta
  - new_delta      = our_accuracy  - paper_accuracy
                   = our_accuracy  - (our_macro_f1 - old_delta)
                   = old_delta + (our_accuracy - our_macro_f1)
"""

import re, textwrap

# ---------------------------------------------------------------------------
# 1.  Parse a results file → {(repr, training, lang, task, translated, clf): (accuracy, macro_f1)}
# ---------------------------------------------------------------------------
CLF_MAP = {          # results-file name → LaTeX column name
    "Decision Tree":      "DT",
    "Random Forest":      "RF",
    "SVM":                "SVM",
    "Logistic Regression":"LR",
}

def parse_results(path):
    data = {}
    with open(path, encoding="utf-8") as f:
        content = f.read()
    blocks = re.split(r"={50,}", content)
    current = None
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = re.search(
            r'(TF-IDF|E5-large)\s*\|\s*training=(\w+)\s*\|\s*test=(\w+)\s*\|\s*task=(\w+)\s*\|\s*translated=(\w+)',
            block)
        if m:
            current = (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5))
            continue
        if current is None:
            continue
        parts = re.split(r'---\s+(.+?)\s+\(best params:', block)
        for i in range(1, len(parts), 2):
            clf_raw = parts[i].strip()
            body    = parts[i+1] if i+1 < len(parts) else ""
            if clf_raw not in CLF_MAP:
                continue
            acc_m  = re.search(r'\s+accuracy\s+([\d.]+)', body)
            f1_m   = re.search(r'macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)', body)
            if acc_m and f1_m:
                key = current + (CLF_MAP[clf_raw],)
                data[key] = (float(acc_m.group(1)), float(f1_m.group(1)))
    return data

TFIDF_PATH = r"C:\Users\Dhruv\Documents\00. Coding\MultiConAD\Experiments\results\tfidf_results.txt"
E5_PATH    = r"C:\Users\Dhruv\Documents\00. Coding\MultiConAD\Experiments\results\e5_results.txt"
tfidf = parse_results(TFIDF_PATH)
e5    = parse_results(E5_PATH)

def get(data, repr_t, training, lang, task, translated, clf):
    key = (repr_t, training, lang, task, translated, clf)
    return data.get(key)          # (accuracy, macro_f1) or None

# ---------------------------------------------------------------------------
# 2.  Old deltas from the existing macro-F1 LaTeX (our_f1 - paper_acc)
#     Structure: old_delta[task][lang][repr][training_mode][clf_abbr]
#     training_mode: "mono"/"multi"/"trans"
# ---------------------------------------------------------------------------
# Positive = we beat paper; negative = we under-perform paper.
# Copied verbatim from the provided LaTeX delta tables.

old_delta = {
 "binary": {
  "Spanish": {
   "Sparse": {
    "mono":  {"DT":-0.06,"RF":-0.10,"SVM":-0.01,"LR":-0.13},
    "multi": {"DT":-0.13,"RF":-0.10,"SVM":-0.41,"LR":-0.07},
    "trans": {"DT":-0.21,"RF":-0.14,"SVM":-0.12,"LR":-0.28},
   },
   "Dense": {
    "mono":  {"DT":+0.02,"RF":-0.07,"SVM":-0.11,"LR":-0.15},
    "multi": {"DT":-0.09,"RF":-0.09,"SVM":+0.04,"LR":-0.40},
    "trans": {"DT":-0.02,"RF":-0.15,"SVM":-0.53,"LR":-0.36},
   },
  },
  "Chinese": {
   "Sparse": {
    "mono":  {"DT":-0.08,"RF":-0.15,"SVM":+0.01,"LR":-0.19},
    "multi": {"DT":-0.24,"RF":-0.25,"SVM":-0.49,"LR":-0.20},
    "trans": {"DT":-0.12,"RF":-0.11,"SVM":-0.12,"LR":-0.40},
   },
   "Dense": {
    "mono":  {"DT":+0.11,"RF":+0.19,"SVM": 0.00,"LR":+0.08},
    "multi": {"DT":+0.05,"RF":-0.02,"SVM": 0.00,"LR":-0.37},
    "trans": {"DT":+0.01,"RF":-0.07,"SVM":-0.05,"LR":-0.40},
   },
  },
  "Greek": {
   "Sparse": {
    "mono":  {"DT":+0.07,"RF":-0.17,"SVM":-0.06,"LR":-0.08},
    "multi": {"DT":-0.03,"RF":-0.24,"SVM":-0.22,"LR":-0.05},
    "trans": {"DT":-0.07,"RF":+0.04,"SVM":-0.10,"LR":-0.23},
   },
   "Dense": {
    "mono":  {"DT":-0.08,"RF":-0.15,"SVM": 0.00,"LR":-0.14},
    "multi": {"DT":+0.13,"RF":-0.16,"SVM":-0.12,"LR":-0.45},
    "trans": {"DT":-0.02,"RF":+0.04,"SVM":-0.26,"LR":-0.36},
   },
  },
  "English": {
   "Sparse": {
    "mono":  {"DT":-0.02,"RF":-0.08,"SVM":+0.05,"LR":+0.03},
    "multi": {"DT": 0.00,"RF":-0.03,"SVM":+0.22,"LR":+0.07},
    "trans": {"DT":-0.02,"RF":-0.08,"SVM":+0.06,"LR":-0.17},
   },
   "Dense": {
    "mono":  {"DT":+0.02,"RF":-0.06,"SVM":+0.01,"LR":+0.03},
    "multi": {"DT":+0.04,"RF":-0.12,"SVM":+0.26,"LR":-0.26},
    "trans": {"DT":-0.07,"RF":-0.05,"SVM":-0.07,"LR":-0.29},
   },
  },
 },
 "multiclass": {
  "Spanish": {
   "Sparse": {
    "mono":  {"DT":-0.18,"RF":-0.16,"SVM":-0.10,"LR":-0.22},
    "multi": {"DT":-0.10,"RF":-0.22,"SVM":-0.02,"LR":-0.09},
    "trans": {"DT":-0.24,"RF":-0.15,"SVM":-0.17,"LR":-0.21},
   },
   "Dense": {
    "mono":  {"DT":-0.09,"RF":-0.13,"SVM":-0.08,"LR":-0.20},
    "multi": {"DT":-0.02,"RF":-0.22,"SVM":-0.46,"LR":-0.15},
    "trans": {"DT":-0.20,"RF":-0.18,"SVM":-0.28,"LR":-0.10},
   },
  },
  "Chinese": {
   "Sparse": {
    "mono":  {"DT":+0.01,"RF":-0.08,"SVM":-0.09,"LR":-0.11},
    "multi": {"DT":-0.15,"RF":-0.13,"SVM":-0.09,"LR":-0.16},
    "trans": {"DT":+0.04,"RF":-0.18,"SVM":-0.25,"LR":-0.31},
   },
   "Dense": {
    "mono":  {"DT":+0.09,"RF":-0.18,"SVM":-0.09,"LR":-0.20},
    "multi": {"DT":+0.08,"RF":-0.27,"SVM":-0.39,"LR":-0.06},
    "trans": {"DT":-0.06,"RF":-0.26,"SVM":-0.36,"LR": 0.00},
   },
  },
  "Greek": {
   "Sparse": {
    "mono":  {"DT":-0.18,"RF":-0.30,"SVM":-0.14,"LR":-0.23},
    "multi": {"DT":-0.18,"RF":-0.26,"SVM":-0.06,"LR":-0.20},
    "trans": {"DT":-0.38,"RF":-0.04,"SVM":-0.18,"LR":-0.17},
   },
   "Dense": {
    "mono":  {"DT":-0.18,"RF":-0.23,"SVM":-0.13,"LR":-0.31},
    "multi": {"DT":-0.14,"RF":-0.27,"SVM":-0.43,"LR":-0.09},
    "trans": {"DT":-0.18,"RF":-0.08,"SVM":-0.15,"LR":+0.10},
   },
  },
  "English": {
   "Sparse": {
    "mono":  {"DT":-0.07,"RF":-0.06,"SVM":+0.01,"LR":-0.02},
    "multi": {"DT":-0.04,"RF":-0.06,"SVM":+0.24,"LR":-0.02},
    "trans": {"DT":+0.08,"RF":-0.02,"SVM":-0.01,"LR":-0.23},
   },
   "Dense": {
    "mono":  {"DT":+0.01,"RF":-0.10,"SVM":-0.17,"LR": 0.00},
    "multi": {"DT":-0.01,"RF":-0.11,"SVM":-0.14,"LR":+0.01},
    "trans": {"DT":+0.07,"RF":-0.04,"SVM":-0.12,"LR":+0.21},
   },
  },
 },
}

# ---------------------------------------------------------------------------
# 3.  Helpers
# ---------------------------------------------------------------------------

LANG_MAP = {"Spanish":"spa","Chinese":"cha","Greek":"gr","English":"en"}
REPR_MAP  = {"Sparse":("TF-IDF","mono_translated=no"),
             "Dense": ("E5-large","mono_translated=no")}  # placeholder; overridden below

CLF_ORDER = ["DT","RF","SVM","LR"]
TRAINING_MODES = [("mono","no"),("multi","no"),("multi","yes")]  # (training, translated)
MODE_LABELS    = ["mono","multi","trans"]

def lookup_acc(task, lang_name, repr_label, training, translated, clf):
    lang  = LANG_MAP[lang_name]
    rtype = "TF-IDF" if repr_label == "Sparse" else "E5-large"
    data  = tfidf if repr_label == "Sparse" else e5
    v = get(data, rtype, training, lang, task, translated, clf)
    return v  # (accuracy, macro_f1) or None

def arrow(val, mono_val):
    if val is None or mono_val is None:
        return ""
    diff = round(val - mono_val, 4)
    if diff > 0.001:  return r"$^{\uparrow}$"
    if diff < -0.001: return r"$^{\downarrow}$"
    return ""

def fmt_cell(val, bold=False, arr=""):
    s = f"{val:.2f}"
    if bold:
        s = r"\textbf{" + s + "}"
    return s + arr

def delta_cmd(d):
    d = round(d, 2)
    if d > 0.005:
        return r"\dpos{%.2f}" % d
    if d < -0.005:
        return r"\dneg{%.2f}" % abs(d)
    return r"\dzero"

# ---------------------------------------------------------------------------
# 4.  Build row data: accuracy values + new deltas
# ---------------------------------------------------------------------------

def build_block(task, lang_name, repr_label):
    """Return list of 12 (accuracy, new_delta) pairs in LaTeX column order.
       Columns: Mono×4 | Multi×4 | Trans×4
    """
    cells = []
    mono_accs = {}
    # collect mono accuracy for arrow reference
    for clf in CLF_ORDER:
        v = lookup_acc(task, lang_name, repr_label, "mono", "no", clf)
        mono_accs[clf] = v[0] if v else None

    for (training, translated), mode_label in zip(TRAINING_MODES, MODE_LABELS):
        accs = {}
        for clf in CLF_ORDER:
            v = lookup_acc(task, lang_name, repr_label, training, translated, clf)
            accs[clf] = v[0] if v else None

        best_acc = max((a for a in accs.values() if a is not None), default=None)

        for clf in CLF_ORDER:
            acc = accs.get(clf)
            old_d = old_delta[task][lang_name][repr_label][mode_label][clf]
            if acc is not None:
                # new_delta = old_delta + (accuracy - macro_f1)
                v = lookup_acc(task, lang_name, repr_label, training, translated, clf)
                macro_f1 = v[1] if v else None
                if macro_f1 is not None:
                    new_d = round(old_d + (acc - macro_f1), 2)
                else:
                    new_d = old_d
                bold = (best_acc is not None and abs(acc - best_acc) < 0.001)
                arr  = arrow(acc, mono_accs[clf]) if mode_label != "mono" else ""
                cells.append((acc, new_d, bold, arr))
            else:
                cells.append((None, old_d, False, ""))
    return cells

# ---------------------------------------------------------------------------
# 5.  LaTeX templates
# ---------------------------------------------------------------------------

PREAMBLE = r"""\usepackage{booktabs}
\usepackage{multirow}
\newcommand{\dpos}[1]{{\color{green!60!black}$+$#1}}
\newcommand{\dneg}[1]{{\color{red!70!black}$-$#1}}
\newcommand{\dzero}{{\color{gray}$\pm$0.00}}
"""

TABLE_HEADER = r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\renewcommand{\arraystretch}{1.25}
\resizebox{\linewidth}{!}{%
\begin{tabular}{ll cccc cccc cccc}
\toprule
& &
\multicolumn{4}{c}{\textbf{Monolingual}} &
\multicolumn{4}{c}{\textbf{Combined-Multilingual}} &
\multicolumn{4}{c}{\textbf{Combined-Translated}} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}
\textbf{Language} & \textbf{Repr.} &
DT & RF & SVM & LR &
DT & RF & SVM & LR &
DT & RF & SVM & LR \\
\midrule"""

TABLE_FOOTER_RESULT = r"""\bottomrule
\end{tabular}
}
\caption{%s}
\label{%s}
\end{table}"""

TABLE_FOOTER_DELTA = r"""\bottomrule
\end{tabular}
}
\caption{%s}
\label{%s}
\end{table}"""

LANGS = ["Spanish","Chinese","Greek","English"]
REPRS = ["Sparse","Dense"]

def make_result_table(task):
    lines = [TABLE_HEADER]
    for lang in LANGS:
        for ri, repr_label in enumerate(REPRS):
            cells = build_block(task, lang, repr_label)
            # cells: 12 entries (mono×4, multi×4, trans×4)
            row_parts = []
            for i, (acc, _, bold, arr) in enumerate(cells):
                if acc is None:
                    row_parts.append("---")
                else:
                    row_parts.append(fmt_cell(acc, bold, arr))

            lang_col = r"\multirow{2}{*}{" + lang + "}" if ri == 0 else ""
            repr_col = repr_label

            # split into mono / multi / trans groups
            mono  = " & ".join(row_parts[0:4])
            multi = " & ".join(row_parts[4:8])
            trans = " & ".join(row_parts[8:12])

            sep = r"\\[3pt]" if repr_label == "Dense" and lang != LANGS[-1] else r"\\"
            lines.append(
                f"{lang_col}\n& {repr_col}\n"
                f"  & {mono}\n"
                f"  & {multi}\n"
                f"  & {trans} {sep}"
            )
        lines.append("")  # blank line between language groups

    task_label = "binary AD vs.\\ HC" if task == "binary" else "multiclass AD vs.\\ MCI vs.\\ HC"
    caption = (f"Replication of Table~5 ({task_label}), accuracy. "
               r"Bold marks the best classifier within each block; "
               r"arrows compare against Monolingual.")
    label = f"tab:multiconad_{task}_acc"

    lines.append(TABLE_FOOTER_RESULT % (caption, label))
    return "\n".join(lines)


def make_delta_table(task):
    lines = [TABLE_HEADER]
    for lang in LANGS:
        for ri, repr_label in enumerate(REPRS):
            cells = build_block(task, lang, repr_label)
            row_parts = []
            for acc, new_d, _, _ in cells:
                row_parts.append(delta_cmd(new_d))

            lang_col = r"\multirow{2}{*}{" + lang + "}" if ri == 0 else ""
            repr_col = repr_label

            mono  = " & ".join(row_parts[0:4])
            multi = " & ".join(row_parts[4:8])
            trans = " & ".join(row_parts[8:12])

            sep = r"\\[3pt]" if repr_label == "Dense" and lang != LANGS[-1] else r"\\"
            lines.append(
                f"{lang_col}\n& {repr_col}\n"
                f"  & {mono}\n"
                f"  & {multi}\n"
                f"  & {trans} {sep}"
            )
        lines.append("")

    task_label = "binary AD vs.\\ HC" if task == "binary" else "multiclass AD vs.\\ MCI vs.\\ HC"
    caption = (f"Delta vs.\\ Shakeri et al.~\\cite{{shakeri2025}} for {task_label}, accuracy. "
               r"Values are this replication minus the paper; "
               r"green~=~improvement, red~=~underperformance.")
    label = f"tab:multiconad_{task}_acc_delta"

    lines.append(TABLE_FOOTER_DELTA % (caption, label))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6.  Write output
# ---------------------------------------------------------------------------
OUT_PATH = r"C:\Users\Dhruv\Documents\00. Coding\MultiConAD\Experiments\results\accuracy_tables.txt"

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("% ============================================================\n")
    f.write("% Preamble commands (add to your document once)\n")
    f.write("% ============================================================\n")
    f.write(PREAMBLE + "\n\n")

    for task in ["binary", "multiclass"]:
        f.write(f"% ============================================================\n")
        f.write(f"% {task.upper()} RESULT TABLE\n")
        f.write(f"% ============================================================\n")
        f.write(make_result_table(task) + "\n\n")

        f.write(f"% ============================================================\n")
        f.write(f"% {task.upper()} DELTA TABLE\n")
        f.write(f"% ============================================================\n")
        f.write(make_delta_table(task) + "\n\n")

print(f"Written to {OUT_PATH}")
