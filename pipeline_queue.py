"""
Repo-local sequential pipeline queue for MultiConAD.

Default order:
1. NCMMSC transcription
2. Data extraction / label attachment
3. Per-language preprocessing
4. TF-IDF benchmark rerun
5. E5 benchmark rerun
6. Accuracy table regeneration

The queue is resumable via tables/experiment-results/pipeline-queue/pipeline-state.json
and per-job logs in tables/experiment-results/pipeline-queue/.
It is intentionally single-worker and fail-fast because later stages depend on
the exact outputs of earlier stages.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = ROOT / "tables" / "experiment-results" / "pipeline-queue"
STATE_PATH = PIPELINE_DIR / "pipeline-state.json"
LOG_DIR = PIPELINE_DIR


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class Job:
    job_id: str
    description: str
    command: list[str]
    cwd: str
    outputs: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)


JOBS: list[Job] = [
    Job(
        job_id="asr_ncmmsc",
        description="Transcribe NCMMSC2021_AD long-audio dataset with Whisper large-v3",
        command=[sys.executable, "processing/transcription/ASR_audio_dataset.py", "--dataset", "ncmmsc"],
        cwd=".",
        outputs=["data/processed/transcriptions/ncmmsc_transcriptions.jsonl"],
    ),
    Job(
        job_id="extract_all",
        description="Run Step 2 extraction and attach labels across all datasets",
        command=[sys.executable, "processing/extraction/run_step2_collections.py"],
        cwd=".",
        outputs=[
            "data/processed/extracted/Chinese_NCMMSC_output.jsonl",
            "data/processed/extracted/ASR_taukadial_train_output.jsonl",
            "data/processed/extracted/English_WLS_output.jsonl",
        ],
        depends_on=["asr_ncmmsc"],
    ),
    Job(
        job_id="clean_english",
        description="Rebuild cleaned English train/test splits",
        command=[sys.executable, "processing/cleaning/text_cleaning_English.py"],
        cwd=".",
        outputs=[
            "data/processed/cleaned/train_english.jsonl",
            "data/processed/cleaned/test_english.jsonl",
        ],
        depends_on=["extract_all"],
    ),
    Job(
        job_id="clean_greek",
        description="Rebuild cleaned Greek train/test splits",
        command=[sys.executable, "processing/cleaning/text_cleaning_Greek.py"],
        cwd=".",
        outputs=[
            "data/processed/cleaned/train_greek.jsonl",
            "data/processed/cleaned/test_greek.jsonl",
        ],
        depends_on=["extract_all"],
    ),
    Job(
        job_id="clean_spanish",
        description="Rebuild cleaned Spanish train/test splits",
        command=[sys.executable, "processing/cleaning/text_cleaning_Spanish.py"],
        cwd=".",
        outputs=[
            "data/processed/cleaned/train_spanish.jsonl",
            "data/processed/cleaned/test_spanish.jsonl",
        ],
        depends_on=["extract_all"],
    ),
    Job(
        job_id="clean_chinese",
        description="Rebuild cleaned Chinese train/test splits",
        command=[sys.executable, "processing/cleaning/text_cleaning_Chinese.py"],
        cwd=".",
        outputs=[
            "data/processed/cleaned/train_chinese.jsonl",
            "data/processed/cleaned/test_chinese.jsonl",
        ],
        depends_on=["extract_all"],
    ),
    Job(
        job_id="benchmark_tfidf",
        description="Rerun the full TF-IDF benchmark matrix",
        command=[sys.executable, "experiments/run_tfidf_experiments.py"],
        cwd=".",
        outputs=["tables/experiment-results/tfidf_results.txt"],
        depends_on=["clean_english", "clean_greek", "clean_spanish", "clean_chinese"],
    ),
    Job(
        job_id="benchmark_e5",
        description="Rerun the full E5-large benchmark matrix",
        command=[sys.executable, "experiments/run_e5_experiments.py"],
        cwd=".",
        outputs=["tables/experiment-results/e5_results.txt"],
        depends_on=["clean_english", "clean_greek", "clean_spanish", "clean_chinese"],
    ),
    Job(
        job_id="tables_accuracy",
        description="Regenerate accuracy comparison tables from benchmark outputs",
        command=[sys.executable, "experiments/generate_accuracy_tables.py"],
        cwd=".",
        outputs=["tables/experiment-results/accuracy_tables.txt"],
        depends_on=["benchmark_tfidf", "benchmark_e5"],
    ),
]

JOB_MAP = {job.job_id: job for job in JOBS}


def abs_path(rel_path: str) -> Path:
    return (ROOT / rel_path).resolve()


def outputs_exist(job: Job) -> bool:
    if not job.outputs:
        return True
    for rel_path in job.outputs:
        path = abs_path(rel_path)
        if not path.exists():
            return False
        if path.is_file() and path.stat().st_size == 0:
            return False
    return True


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"created_at": now_iso(), "updated_at": now_iso(), "jobs": {}}
    with STATE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    state["updated_at"] = now_iso()
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def state_entry(state: dict, job_id: str) -> dict:
    return state.setdefault("jobs", {}).setdefault(job_id, {})


def is_done(state: dict, job: Job) -> bool:
    entry = state.get("jobs", {}).get(job.job_id, {})
    return entry.get("status") == "done" and outputs_exist(job)


def dependency_satisfied(state: dict, dep_id: str) -> bool:
    dep_job = JOB_MAP[dep_id]
    return is_done(state, dep_job) or outputs_exist(dep_job)


def topological_subset(job_ids: Iterable[str]) -> list[Job]:
    selected = set(job_ids)
    ordered: list[Job] = []
    for job in JOBS:
        if job.job_id in selected:
            ordered.append(job)
    return ordered


def dependency_closure(job_id: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    def visit(current: str) -> None:
        if current in seen:
            return
        seen.add(current)
        for dep in JOB_MAP[current].depends_on:
            visit(dep)
        result.append(current)

    visit(job_id)
    return result


def append_log_header(log_path: Path, job: Job) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 80}\n")
        log.write(f"{now_iso()}  START  {job.job_id}\n")
        log.write(f"description: {job.description}\n")
        log.write(f"cwd: {abs_path(job.cwd)}\n")
        log.write(f"command: {job.command}\n")
        if job.outputs:
            log.write(f"outputs: {job.outputs}\n")
        log.write(f"{'=' * 80}\n\n")


def append_log_footer(log_path: Path, returncode: int, duration_sec: float) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n{'-' * 80}\n")
        log.write(f"{now_iso()}  END  returncode={returncode}  duration_sec={duration_sec:.1f}\n")
        log.write(f"{'-' * 80}\n")


def run_job(state: dict, job: Job) -> None:
    for dep in job.depends_on:
        if not dependency_satisfied(state, dep):
            raise RuntimeError(f"Dependency '{dep}' is not satisfied for job '{job.job_id}'")

    entry = state_entry(state, job.job_id)
    log_path = LOG_DIR / f"{job.job_id}.log"
    entry.update(
        {
            "status": "running",
            "description": job.description,
            "command": job.command,
            "cwd": str(abs_path(job.cwd)),
            "outputs": job.outputs,
            "depends_on": job.depends_on,
            "last_started_at": now_iso(),
            "log_path": str(log_path),
        }
    )
    save_state(state)
    append_log_header(log_path, job)

    start = time.time()
    env = os.environ.copy()
    # Force UTF-8 for child Python processes so dataset scripts can print
    # Unicode status markers on Windows without crashing on console encoding.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            job.command,
            cwd=abs_path(job.cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    duration = time.time() - start
    append_log_footer(log_path, process.returncode, duration)

    entry["returncode"] = process.returncode
    entry["last_finished_at"] = now_iso()
    entry["duration_sec"] = round(duration, 1)

    if process.returncode != 0:
        entry["status"] = "failed"
        save_state(state)
        raise RuntimeError(f"Job '{job.job_id}' failed with exit code {process.returncode}")

    if not outputs_exist(job):
        entry["status"] = "failed"
        entry["returncode"] = -1
        save_state(state)
        raise RuntimeError(f"Job '{job.job_id}' completed but expected outputs were not found")

    entry["status"] = "done"
    save_state(state)


def list_jobs() -> None:
    for idx, job in enumerate(JOBS, 1):
        print(f"[{idx}] {job.job_id}")
        print(f"    {job.description}")
        if job.depends_on:
            print(f"    depends_on: {', '.join(job.depends_on)}")
        if job.outputs:
            print(f"    outputs: {', '.join(job.outputs)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential MultiConAD pipeline queue")
    parser.add_argument("--list", action="store_true", help="List jobs and exit")
    parser.add_argument("--resume", action="store_true", help="Skip jobs already marked done with outputs present")
    parser.add_argument("--from-job", dest="from_job", choices=JOB_MAP.keys(), help="Start from this job and include all later jobs")
    parser.add_argument("--only", choices=JOB_MAP.keys(), help="Run one job plus its dependency closure")
    parser.add_argument("--status", action="store_true", help="Print queue state and exit")
    return parser.parse_args()


def print_status(state: dict) -> None:
    for job in JOBS:
        entry = state.get("jobs", {}).get(job.job_id, {})
        status = entry.get("status", "pending")
        print(f"{job.job_id:16} {status:8} outputs={'yes' if outputs_exist(job) else 'no'}")


def selected_jobs_from_args(args: argparse.Namespace) -> list[Job]:
    if args.only:
        return topological_subset(dependency_closure(args.only))
    if args.from_job:
        start_seen = False
        subset: list[str] = []
        for job in JOBS:
            if job.job_id == args.from_job:
                start_seen = True
            if start_seen:
                subset.append(job.job_id)
        return topological_subset(subset)
    return JOBS


def main() -> int:
    args = parse_args()

    if args.list:
        list_jobs()
        return 0

    state = load_state()

    if args.status:
        print_status(state)
        return 0

    jobs_to_run = selected_jobs_from_args(args)
    print(f"Queue root: {ROOT}")
    print(f"State file: {STATE_PATH}")
    print(f"Log dir   : {LOG_DIR}")
    print("Jobs:")
    for job in jobs_to_run:
        print(f"  - {job.job_id}")
    print()

    for job in jobs_to_run:
        if args.resume and is_done(state, job):
            print(f"[skip] {job.job_id} already done")
            continue
        print(f"[run ] {job.job_id} - {job.description}")
        run_job(state, job)
        print(f"[done] {job.job_id}")

    print("\nQueue completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
