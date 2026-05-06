"""Parse results files and emit accuracy-based comparison tables."""
import re, sys

PAPER = {
    # (task, lang): {mono_sparse, multi_sparse, trans_sparse, mono_dense, multi_dense, trans_dense}
    # From paper Tables 5 & 6 (best classifier per block, accuracy)
    # These need to be filled in by user — we just emit our values here
}

def parse_file(path):
    """Return dict keyed (training, test_lang, task, translated, classifier) -> accuracy."""
    results = {}
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Split on the separator blocks
    blocks = re.split(r"={60,}", content)

    current_config = None
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Try to find config header
        header_match = re.search(
            r'(TF-IDF|E5-large)\s*\|\s*training=(\w+)\s*\|\s*test=(\w+)\s*\|\s*task=(\w+)\s*\|\s*translated=(\w+)',
            block
        )
        if header_match:
            repr_type = header_match.group(1)
            training  = header_match.group(2)   # mono / multi
            test_lang = header_match.group(3)   # en / gr / cha / spa
            task      = header_match.group(4)   # binary / multiclass
            translated = header_match.group(5)  # yes / no
            current_config = (repr_type, training, test_lang, task, translated)
            continue

        if current_config is None:
            continue

        # Find classifier blocks within this block
        clf_sections = re.split(r'--- (.+?) \(best params:', block)
        # clf_sections[0] is pre-first-classifier; then alternates name, body
        for i in range(1, len(clf_sections), 2):
            clf_name = clf_sections[i].strip()
            body = clf_sections[i+1] if i+1 < len(clf_sections) else ""
            acc_match = re.search(r'\s+accuracy\s+([\d.]+)', body)
            if acc_match:
                acc = float(acc_match.group(1))
                key = current_config + (clf_name,)
                results[key] = acc

    return results

def best_acc(results, repr_type, training, test_lang, task, translated):
    """Return best accuracy across classifiers for a given config."""
    accs = []
    for key, acc in results.items():
        r, tr, tl, ta, trans, clf = key
        if r == repr_type and tr == training and tl == test_lang and ta == task and trans == translated:
            accs.append((acc, clf))
    if not accs:
        return None, None
    best = max(accs, key=lambda x: x[0])
    return best

def all_accs(results, repr_type, training, test_lang, task, translated):
    """Return dict {clf_name: acc} for a config."""
    out = {}
    for key, acc in results.items():
        r, tr, tl, ta, trans, clf = key
        if r == repr_type and tr == training and tl == test_lang and ta == task and trans == translated:
            out[clf] = acc
    return out

def fmt(val):
    return f"{val:.2f}" if val is not None else "  — "

def run():
    tfidf_path = r"C:\Users\Dhruv\Documents\00. Coding\MultiConAD\Experiments\results\tfidf_results.txt"
    e5_path    = r"C:\Users\Dhruv\Documents\00. Coding\MultiConAD\Experiments\results\e5_results.txt"

    tfidf = parse_file(tfidf_path)
    e5    = parse_file(e5_path)

    langs = [
        ("spa", "Spanish"),
        ("cha", "Chinese"),
        ("gr",  "Greek"),
        ("en",  "English"),
    ]

    for task, task_label in [("binary", "Binary"), ("multiclass", "Multiclass")]:
        print(f"\n{'='*110}")
        print(f"Table: {task_label} — Accuracy (best classifier per block)")
        print(f"{'='*110}")
        hdr = f"{'Language':<10} {'Repr':<8}  {'Mono':>6} {'Multi':>6} {'Trans':>6}    {'Mono':>6} {'Multi':>6} {'Trans':>6}"
        print(hdr)
        print(f"{'':10} {'':8}  {'--- Sparse (TF-IDF) ---':>20}    {'--- Dense (E5-large) ---':>22}")
        print("-"*110)

        for lang_code, lang_name in langs:
            for repr_label, results, repr_type in [
                ("Sparse", tfidf, "TF-IDF"),
                ("Dense",  e5,    "E5-large"),
            ]:
                mono_best,  mono_clf  = best_acc(results, repr_type, "mono",  lang_code, task, "no")
                multi_best, multi_clf = best_acc(results, repr_type, "multi", lang_code, task, "no")
                trans_best, trans_clf = best_acc(results, repr_type, "multi", lang_code, task, "yes")

                row = (f"{lang_name if repr_label=='Sparse' else '':<10} "
                       f"{repr_label:<8}  "
                       f"{fmt(mono_best):>6} {fmt(multi_best):>6} {fmt(trans_best):>6}")
                print(row)
            print()

    # Also print per-classifier detail for the monolingual sparse binary to validate
    print("\n\n=== Per-classifier accuracy: TF-IDF | mono | binary ===")
    tfidf_clfs = ["Decision Tree", "Random Forest", "Naive Bayes", "SVM", "Logistic Regression"]
    e5_clfs    = ["Decision Tree", "Random Forest", "SVM", "Logistic Regression"]

    print(f"\n{'Language':<10} {'Classifier':<22}  {'TF-IDF acc':>10}  {'E5 acc':>8}")
    print("-"*60)
    for lang_code, lang_name in langs:
        for clf in tfidf_clfs:
            key_tfidf = ("TF-IDF", "mono", lang_code, "binary", "no", clf)
            key_e5    = ("E5-large", "mono", lang_code, "binary", "no", clf)
            t = tfidf.get(key_tfidf)
            e = e5.get(key_e5)
            print(f"{lang_name:<10} {clf:<22}  {fmt(t):>10}  {fmt(e):>8}")
        print()

if __name__ == "__main__":
    run()
