"""
Runs the full TF-IDF experiment matrix and logs results to results/tfidf_results.txt.
Covers: mono/multi × binary/multiclass × translated/original × all 4 test languages.
"""
import subprocess
import sys
import os
import itertools
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "tfidf_results.txt")

languages   = ['en', 'gr', 'cha', 'spa']
tasks       = ['binary', 'multiclass']
translated  = ['no', 'yes']
trainings   = ['mono', 'multi']

combos = list(itertools.product(trainings, tasks, translated, languages))
total = len(combos)

print(f"Running {total} TF-IDF experiments → {OUT_FILE}\n")

with open(OUT_FILE, 'w', encoding='utf-8') as log:
    log.write(f"TF-IDF Experiment Results — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    log.write("="*60 + "\n\n")

for i, (training, task, trans, lang) in enumerate(combos, 1):
    label = f"[{i}/{total}] training={training} task={task} translated={trans} test={lang}"
    print(label)

    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "TF_IDF_classifier.py"),
         '--test_language', lang,
         '--task', task,
         '--translated', trans,
         '--training', training],
        capture_output=True, text=True
    )

    with open(OUT_FILE, 'a', encoding='utf-8') as log:
        log.write(result.stdout)
        if result.stderr:
            log.write(f"\n[STDERR]\n{result.stderr}\n")
        log.write("\n")

    # Print last few lines of output so we can see progress
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    for l in lines[-4:]:
        print(" ", l)
    print()

print(f"\nDone. Results saved to {OUT_FILE}")
