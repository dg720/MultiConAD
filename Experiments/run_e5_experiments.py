"""
Runs the full E5-large experiment matrix and logs results to tables/01-baselines/embedding-baselines/result-tables/e5_results.txt.
Embedding caching means the model only encodes each unique (training-set, task, translated)
combination once; subsequent test languages reuse the cached vectors.
"""
import itertools
import os
import subprocess
import sys
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_ROOT = None
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_ROOT, "tables", "01-baselines", "embedding-baselines", "result-tables")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "e5_results.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "tables", "01-baselines", "embedding-baselines", "embedding-cache")

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


safe_print(f"Running {total} E5-large experiments -> {OUT_FILE}")
safe_print(f"Embedding cache: {CACHE_DIR}\n")

with open(OUT_FILE, "w", encoding="utf-8") as log:
    log.write(f"E5-large Experiment Results - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    log.write("=" * 60 + "\n\n")

for i, (training, task, trans, lang) in enumerate(combos, 1):
    label = f"[{i}/{total}] training={training} task={task} translated={trans} test={lang}"
    safe_print(label)

    result = subprocess.run(
        [
            sys.executable,
            os.path.join(SCRIPT_DIR, "e5_large_classifier.py"),
            "--test_language",
            lang,
            "--task",
            task,
            "--translated",
            trans,
            "--training",
            training,
            "--cache_dir",
            CACHE_DIR,
        ],
        capture_output=True,
        text=True,
    )

    with open(OUT_FILE, "a", encoding="utf-8") as log:
        log.write(result.stdout)
        if result.stderr.strip():
            err_lines = [
                line
                for line in result.stderr.splitlines()
                if "Error" in line or "error" in line or "Traceback" in line
            ]
            if err_lines:
                log.write("\n[STDERR]\n" + "\n".join(err_lines) + "\n")
        log.write("\n")

    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    for line in lines[-5:]:
        safe_print(" ", line)
    if result.returncode != 0:
        safe_print(f"  ERROR (exit {result.returncode})")
        safe_print(result.stderr[-500:])
    safe_print()

safe_print(f"Done. Results saved to {OUT_FILE}")
