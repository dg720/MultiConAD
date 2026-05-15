"""
Runs the full TF-IDF experiment matrix and logs results to tables/01-baselines/embedding-baselines/result-tables/tfidf_results.txt.
Covers: mono/multi x binary/multiclass x translated/original x all 4 test languages.
"""
import itertools
import os
import subprocess
import sys
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "tables", "01-baselines", "embedding-baselines", "result-tables")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "tfidf_results.txt")

languages = ["en", "gr", "cha", "spa"]
tasks = ["binary", "multiclass"]
translated = ["no", "yes"]
trainings = ["mono", "multi"]

combos = list(itertools.product(trainings, tasks, translated, languages))
total = len(combos)


def safe_print(*parts) -> None:
    text = " ".join(str(part) for part in parts)
    text = text.encode("ascii", "backslashreplace").decode("ascii")
    try:
        print(text, flush=True)
    except OSError:
        pass


safe_print(f"Running {total} TF-IDF experiments -> {OUT_FILE}\n")

with open(OUT_FILE, "w", encoding="utf-8") as log:
    log.write(f"TF-IDF Experiment Results - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    log.write("=" * 60 + "\n\n")

for i, (training, task, trans, lang) in enumerate(combos, 1):
    label = f"[{i}/{total}] training={training} task={task} translated={trans} test={lang}"
    safe_print(label)

    result = subprocess.run(
        [
            sys.executable,
            os.path.join(SCRIPT_DIR, "TF_IDF_classifier.py"),
            "--test_language",
            lang,
            "--task",
            task,
            "--translated",
            trans,
            "--training",
            training,
        ],
        capture_output=True,
        text=True,
    )

    with open(OUT_FILE, "a", encoding="utf-8") as log:
        log.write(result.stdout)
        if result.stderr:
            log.write(f"\n[STDERR]\n{result.stderr}\n")
        log.write("\n")

    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    for line in lines[-4:]:
        safe_print(" ", line)
    safe_print()

safe_print(f"\nDone. Results saved to {OUT_FILE}")
