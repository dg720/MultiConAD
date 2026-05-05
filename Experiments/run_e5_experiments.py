"""
Runs the full E5-large experiment matrix and logs results to results/e5_results.txt.
Embedding caching means the model only encodes each unique (training-set, task, translated)
combination once — subsequent test languages reuse the cached vectors.
"""
import subprocess
import sys
import os
import itertools
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "e5_results.txt")
CACHE_DIR = os.path.join(SCRIPT_DIR, "embedding_cache")

languages  = ['en', 'gr', 'cha', 'spa']
tasks      = ['binary', 'multiclass']
translated = ['no', 'yes']
trainings  = ['mono', 'multi']

combos = list(itertools.product(trainings, tasks, translated, languages))
total = len(combos)

print(f"Running {total} E5-large experiments → {OUT_FILE}")
print(f"Embedding cache: {CACHE_DIR}\n")

with open(OUT_FILE, 'w', encoding='utf-8') as log:
    log.write(f"E5-large Experiment Results — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    log.write("="*60 + "\n\n")

for i, (training, task, trans, lang) in enumerate(combos, 1):
    label = f"[{i}/{total}] training={training} task={task} translated={trans} test={lang}"
    print(label)

    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "e5_large_classifier.py"),
         '--test_language', lang,
         '--task', task,
         '--translated', trans,
         '--training', training,
         '--cache_dir', CACHE_DIR],
        capture_output=True, text=True
    )

    with open(OUT_FILE, 'a', encoding='utf-8') as log:
        log.write(result.stdout)
        if result.stderr.strip():
            # Only log actual errors, not progress bars
            err_lines = [l for l in result.stderr.splitlines()
                         if 'Error' in l or 'error' in l or 'Traceback' in l]
            if err_lines:
                log.write(f"\n[STDERR]\n" + '\n'.join(err_lines) + "\n")
        log.write("\n")

    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    for l in lines[-5:]:
        print(" ", l)
    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode})")
        print(result.stderr[-500:])
    print()

print(f"Done. Results saved to {OUT_FILE}")
