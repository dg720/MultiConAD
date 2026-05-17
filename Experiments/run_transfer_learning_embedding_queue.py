from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "transfer_learning_embeddings.py"
OUT_ROOT = ROOT / "tables" / "01-baselines" / "transfer-learning-baselines"
QUEUE_DIR = OUT_ROOT / "queue"
JOB_LOG_DIR = OUT_ROOT / "job_logs"
SUMMARY_DIR = OUT_ROOT / "summaries"
RESULT_CSV = OUT_ROOT / "result-tables" / "csv" / "frozen_embedding_runs.csv"
STATE_PATH = QUEUE_DIR / "frozen_embedding_queue_state.json"
PLAN_PATH = QUEUE_DIR / "frozen_embedding_queue_plan.json"
QUEUE_LOG = QUEUE_DIR / "frozen_embedding_queue.log"

SEEDS = [42, 43, 44, 45, 46]
LANGS = ["en", "gr", "cha", "spa"]
TASKS = ["binary", "multiclass"]
SETTINGS = [
    {"training": "mono", "translated": "no"},
    {"training": "multi", "translated": "no"},
    {"training": "multi", "translated": "yes"},
]
MODEL_POOL_SEQUENCE = [
    {"model": "xlm-roberta-base", "pooling": "cls"},
    {"model": "xlm-roberta-base", "pooling": "mean"},
    {"model": "bert-base-multilingual-cased", "pooling": "cls"},
    {"model": "bert-base-multilingual-cased", "pooling": "mean"},
]
CLASSIFIERS = {"DT", "RF", "SVM", "LR", "Best Ensemble"}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")


def ensure_dirs() -> None:
    for path in [QUEUE_DIR, JOB_LOG_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    ensure_dirs()
    line = f"{now_iso()} {message}"
    print(line, flush=True)
    with QUEUE_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def job_id(job: dict) -> str:
    return (
        f"{safe_name(job['model'])}_{job['pooling']}_{job['length_mode']}_"
        f"seed{job['seed']}_{job['training']}_{job['translated']}_{job['task']}_{job['test_language']}"
    )


def make_job(model: str, pooling: str, seed: int, training: str, translated: str, task: str, lang: str) -> dict:
    job = {
        "model": model,
        "pooling": pooling,
        "length_mode": "truncate",
        "seed": seed,
        "training": training,
        "translated": translated,
        "task": task,
        "test_language": lang,
    }
    job["id"] = job_id(job)
    return job


def add_unique(jobs: list[dict], seen: set[str], job: dict) -> None:
    if job["id"] in seen:
        return
    seen.add(job["id"])
    jobs.append(job)


def build_jobs() -> list[dict]:
    jobs: list[dict] = []
    seen: set[str] = set()

    first = MODEL_POOL_SEQUENCE[0]
    model = first["model"]
    pooling = first["pooling"]

    for lang in LANGS:
        add_unique(jobs, seen, make_job(model, pooling, 42, "mono", "no", "binary", lang))

    for lang in LANGS:
        add_unique(jobs, seen, make_job(model, pooling, 42, "mono", "no", "multiclass", lang))

    for setting in SETTINGS[1:]:
        for task in TASKS:
            for lang in LANGS:
                add_unique(
                    jobs,
                    seen,
                    make_job(model, pooling, 42, setting["training"], setting["translated"], task, lang),
                )

    for seed in [43, 44, 45, 46]:
        for setting in SETTINGS:
            for task in TASKS:
                for lang in LANGS:
                    add_unique(
                        jobs,
                        seen,
                        make_job(model, pooling, seed, setting["training"], setting["translated"], task, lang),
                    )

    for item in MODEL_POOL_SEQUENCE[1:]:
        for seed in SEEDS:
            for setting in SETTINGS:
                for task in TASKS:
                    for lang in LANGS:
                        add_unique(
                            jobs,
                            seen,
                            make_job(
                                item["model"],
                                item["pooling"],
                                seed,
                                setting["training"],
                                setting["translated"],
                                task,
                                lang,
                            ),
                        )

    return jobs


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"created_at": now_iso(), "updated_at": now_iso(), "jobs": {}}
    with STATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    ensure_dirs()
    state["updated_at"] = now_iso()
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def result_exists(job: dict) -> bool:
    if not RESULT_CSV.exists():
        return False
    try:
        df = pd.read_csv(RESULT_CSV)
    except Exception:
        return False
    if df.empty:
        return False
    mask = (
        (df["smoke"].astype(str).str.lower() == "false")
        & (df["seed"].astype(int) == int(job["seed"]))
        & (df["model"] == job["model"])
        & (df["pooling"] == job["pooling"])
        & (df["length_mode"] == job["length_mode"])
        & (df["training"] == job["training"])
        & (df["translated"].astype(str) == str(job["translated"]))
        & (df["task"] == job["task"])
        & (df["test_language"] == job["test_language"])
    )
    classifiers = set(df.loc[mask, "classifier"].astype(str))
    return CLASSIFIERS.issubset(classifiers)


def command_for(job: dict, args) -> list[str]:
    return [
        sys.executable,
        str(RUNNER),
        "--model",
        job["model"],
        "--pooling",
        job["pooling"],
        "--length_mode",
        job["length_mode"],
        "--task",
        job["task"],
        "--training",
        job["training"],
        "--test_language",
        job["test_language"],
        "--translated",
        job["translated"],
        "--seed",
        str(job["seed"]),
        "--batch_size",
        str(args.batch_size),
    ]


def write_plan(jobs: list[dict]) -> None:
    ensure_dirs()
    payload = {
        "created_at": now_iso(),
        "description": "Frozen PLM embedding queue ordered by the staged transfer-learning plan.",
        "total_jobs": len(jobs),
        "model_pool_sequence": MODEL_POOL_SEQUENCE,
        "seeds": SEEDS,
        "settings": SETTINGS,
        "tasks": TASKS,
        "languages": LANGS,
        "jobs": jobs,
    }
    with PLAN_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_queue(args) -> None:
    ensure_dirs()
    jobs = build_jobs()
    write_plan(jobs)
    state = load_state()
    state.setdefault("jobs", {})
    log(f"queue start total_jobs={len(jobs)}")

    for index, job in enumerate(jobs, start=1):
        jid = job["id"]
        entry = state["jobs"].setdefault(jid, {})
        if entry.get("status") == "done" or result_exists(job):
            entry.update({"status": "done", "skipped_existing": True, "updated_at": now_iso()})
            save_state(state)
            log(f"[{index}/{len(jobs)}] skip done {jid}")
            continue

        job_log = JOB_LOG_DIR / f"{jid}.log"
        cmd = command_for(job, args)
        entry.update(
            {
                "status": "running",
                "started_at": now_iso(),
                "command": cmd,
                "job": job,
                "log": str(job_log),
            }
        )
        save_state(state)
        log(f"[{index}/{len(jobs)}] start {jid}")
        started = time.time()

        result = None
        for attempt in range(1, args.retries + 2):
            with job_log.open("a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 100}\n")
                f.write(f"{now_iso()} START {jid} attempt={attempt}/{args.retries + 1}\n")
                f.write("command: " + " ".join(cmd) + "\n\n")
                f.flush()
                result = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
            if result.returncode == 0:
                break
            log(f"[{index}/{len(jobs)}] attempt failed {jid} exit={result.returncode} attempt={attempt}")
            if attempt <= args.retries:
                time.sleep(args.retry_delay_seconds)

        elapsed = round(time.time() - started, 1)
        if result is None or result.returncode != 0:
            entry.update(
                {
                    "status": "failed",
                    "finished_at": now_iso(),
                    "elapsed_seconds": elapsed,
                    "returncode": None if result is None else result.returncode,
                }
            )
            save_state(state)
            log(f"[{index}/{len(jobs)}] failed {jid} exit={result.returncode} elapsed={elapsed}s")
            if args.fail_fast:
                raise SystemExit(result.returncode)
            continue

        if not result_exists(job):
            entry.update(
                {
                    "status": "failed_missing_results",
                    "finished_at": now_iso(),
                    "elapsed_seconds": elapsed,
                    "returncode": result.returncode,
                }
            )
            save_state(state)
            log(f"[{index}/{len(jobs)}] missing results {jid} elapsed={elapsed}s")
            if args.fail_fast:
                raise SystemExit(2)
            continue

        entry.update(
            {
                "status": "done",
                "finished_at": now_iso(),
                "elapsed_seconds": elapsed,
                "returncode": 0,
            }
        )
        save_state(state)
        log(f"[{index}/{len(jobs)}] done {jid} elapsed={elapsed}s")

    log("queue complete")


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential resumable queue for frozen PLM transfer-learning baselines.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry_delay_seconds", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    run_queue(parse_args())
