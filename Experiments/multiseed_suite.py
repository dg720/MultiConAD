import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


SEEDS = [42, 43, 44, 45, 46]
LANG_CODES = ["en", "gr", "cha", "spa"]
LANG_NAMES = {"en": "english", "gr": "greek", "cha": "chinese", "spa": "spanish"}
LANG_LABELS = {"en": "English", "gr": "Greek", "cha": "Chinese", "spa": "Spanish"}
TASKS = ["binary", "multiclass"]
METHODS = ["tfidf", "e5"]
SETTINGS = [
    {"name": "monolingual", "training": "mono", "translated": "no"},
    {"name": "multilingual_combined", "training": "multi", "translated": "no"},
    {"name": "translated_combined", "training": "multi", "translated": "yes"},
]
CLASSIFIERS = ["DT", "RF", "SVM", "LR"]
FUSION_COMBOS = [
    combo
    for size in range(2, len(CLASSIFIERS) + 1)
    for combo in itertools.combinations(CLASSIFIERS, size)
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def pct_str(values: list[float]) -> str:
    values = np.array(values, dtype=float) * 100.0
    return f"{values.mean():.1f} +/- {values.std(ddof=0):.1f}"


def safe_print(*parts) -> None:
    text = " ".join(str(part) for part in parts)
    text = text.encode("ascii", "backslashreplace").decode("ascii")
    try:
        print(text, flush=True)
    except OSError:
        pass


def json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def record_base_key(row: pd.Series) -> str:
    key_parts = [
        str(row.get("Dataset", "")),
        str(row.get("File_ID", "")),
        str(row.get("Diagnosis", "")),
        str(row.get("Languages", "")),
        str(row.get("Text_interviewer_participant", "")),
    ]
    return hashlib.sha256("||".join(key_parts).encode("utf-8")).hexdigest()


def attach_row_uids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    counts = defaultdict(int)
    row_ids = []
    for _, row in df.iterrows():
        base = record_base_key(row)
        idx = counts[base]
        counts[base] += 1
        row_ids.append(f"{base}:{idx}")
    df["_row_uid"] = row_ids
    return df


def load_full_corpora(project_root: Path):
    cleaned_dir = project_root / "data" / "processed" / "cleaned"
    translated_dir = project_root / "data" / "processed" / "translated"

    full_clean = {}
    full_translated = {}

    for code, lang in LANG_NAMES.items():
        train_df = read_jsonl(cleaned_dir / f"train_{lang}.jsonl")
        test_df = read_jsonl(cleaned_dir / f"test_{lang}.jsonl")
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        full_df = attach_row_uids(full_df)
        full_clean[code] = full_df

        if code == "en":
            trans_df = full_df.copy()
            if "translated" not in trans_df.columns:
                trans_df["translated"] = trans_df["Text_interviewer_participant"]
            full_translated[code] = trans_df
            continue

        trans_train = read_jsonl(translated_dir / f"train_{lang}_translated.jsonl")
        trans_test = read_jsonl(translated_dir / f"test_{lang}_translated.jsonl")
        trans_full = pd.concat([trans_train, trans_test], ignore_index=True)
        trans_full = attach_row_uids(trans_full)

        clean_ids = Counter(full_df["_row_uid"])
        trans_ids = Counter(trans_full["_row_uid"])
        if clean_ids != trans_ids:
            raise RuntimeError(f"Translated corpus mismatch for {lang}.")
        full_translated[code] = trans_full

    return full_clean, full_translated


def prepare_seed_data(seed: int, suite_dir: Path, full_clean: dict, full_translated: dict, logger) -> dict:
    seed_root = suite_dir / "seed_data" / f"seed_{seed}"
    cleaned_out = seed_root / "cleaned"
    translated_out = seed_root / "translated"
    cleaned_out.mkdir(parents=True, exist_ok=True)
    translated_out.mkdir(parents=True, exist_ok=True)

    logger(f"Preparing seed {seed} data under {seed_root}")

    for code, lang in LANG_NAMES.items():
        full_df = full_clean[code].copy()
        full_trans_df = full_translated[code].copy()

        if code == "en":
            wls_df = full_df[full_df["Dataset"] == "WLS"].copy()
            base_df = full_df[full_df["Dataset"] != "WLS"].copy()
            train_base, test_df = train_test_split(
                base_df,
                test_size=0.2,
                stratify=base_df["Diagnosis"],
                random_state=seed,
            )
            train_df = pd.concat([train_base, wls_df], ignore_index=True)
        else:
            train_df, test_df = train_test_split(
                full_df,
                test_size=0.2,
                stratify=full_df["Diagnosis"],
                random_state=seed,
            )

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_ids = set(train_df["_row_uid"])
        test_ids = set(test_df["_row_uid"])

        trans_train_df = full_trans_df[full_trans_df["_row_uid"].isin(train_ids)].copy().reset_index(drop=True)
        trans_test_df = full_trans_df[full_trans_df["_row_uid"].isin(test_ids)].copy().reset_index(drop=True)

        if len(trans_train_df) != len(train_df) or len(trans_test_df) != len(test_df):
            raise RuntimeError(f"Translated split size mismatch for {lang} seed {seed}.")

        write_jsonl(train_df.drop(columns=["_row_uid"]), cleaned_out / f"train_{lang}.jsonl")
        write_jsonl(test_df.drop(columns=["_row_uid"]), cleaned_out / f"test_{lang}.jsonl")
        write_jsonl(trans_train_df.drop(columns=["_row_uid"]), translated_out / f"train_{lang}_translated.jsonl")
        write_jsonl(trans_test_df.drop(columns=["_row_uid"]), translated_out / f"test_{lang}_translated.jsonl")

        logger(
            f"Seed {seed} {lang}: train={len(train_df)} test={len(test_df)} "
            f"translated_train={len(trans_train_df)} translated_test={len(trans_test_df)}"
        )

    return {
        "seed_root": str(seed_root),
        "cleaned_dir": str(cleaned_out),
        "translated_dir": str(translated_out),
    }


_E5_MODEL = None


def load_project_env(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip()
    if "HF_TOKEN" not in os.environ and "HUGGINGFACE_HUB_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_HUB_TOKEN"]


def get_e5_model():
    global _E5_MODEL
    if _E5_MODEL is None:
        _E5_MODEL = SentenceTransformer("intfloat/multilingual-e5-large").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return _E5_MODEL


def aligned_proba(proba: np.ndarray, model_classes: np.ndarray, all_classes: list[str]) -> np.ndarray:
    out = np.zeros((proba.shape[0], len(all_classes)), dtype=float)
    idx_map = {cls: i for i, cls in enumerate(all_classes)}
    for src_idx, cls in enumerate(model_classes):
        out[:, idx_map[str(cls)]] = proba[:, src_idx]
    return out


def scores_to_proba(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    denom = exp_scores.sum(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return exp_scores / denom


def estimator_proba(estimator, x_test, all_classes: list[str]) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return aligned_proba(estimator.predict_proba(x_test), estimator.classes_, all_classes)

    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x_test)
        return aligned_proba(scores_to_proba(scores), estimator.classes_, all_classes)

    raise RuntimeError(f"Estimator {estimator.__class__.__name__} does not support probability fallback.")


def should_use_fast_svm_fallback(job: dict, clf_name: str) -> bool:
    return (
        clf_name == "SVM"
        and job["method"] == "tfidf"
        and job["seed"] == 46
        and job["training"] == "multi"
        and job["translated"] == "yes"
        and job["task"] == "multiclass"
        and job["test_language"] == "spa"
    )


def build_classifier_specs(seed: int):
    return {
        "DT": (
            DecisionTreeClassifier(random_state=seed),
            {"max_depth": [10, 20, 30]},
        ),
        "RF": (
            RandomForestClassifier(random_state=seed),
            {"n_estimators": [50, 100, 200]},
        ),
        "SVM": (
            SVC(random_state=seed, class_weight="balanced", probability=True),
            {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        ),
        "LR": (
            LogisticRegression(random_state=seed, max_iter=1000),
            {"C": [0.1, 1, 10]},
        ),
    }


def get_e5_embeddings(df: pd.DataFrame, text_col: str, cache_path: Path, logger) -> np.ndarray:
    if cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape[0] == len(df):
            logger(f"Loading cached embeddings: {cache_path.name}")
            return arr
        logger(
            f"Cache row mismatch for {cache_path.name}: cache={arr.shape[0]} current={len(df)}. Rebuilding."
        )

    logger(f"Encoding {len(df)} texts for {cache_path.name}")
    model = get_e5_model()
    texts = ["passage: " + str(t) for t in df[text_col].fillna("").tolist()]
    arr = model.encode(
        texts,
        normalize_embeddings=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32,
        show_progress_bar=False,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, arr)
    return arr


def prepare_features(
    method: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    cache_dir: Path,
    cache_stem: str,
    logger,
):
    if method == "tfidf":
        vec = TfidfVectorizer()
        x_train = vec.fit_transform(train_df[text_col].fillna("").astype(str))
        x_test = vec.transform(test_df[text_col].fillna("").astype(str))
        return x_train, x_test

    train_cache = cache_dir / f"{cache_stem}_train.npy"
    test_cache = cache_dir / f"{cache_stem}_test.npy"
    x_train = get_e5_embeddings(train_df, text_col, train_cache, logger)
    x_test = get_e5_embeddings(test_df, text_col, test_cache, logger)
    return x_train, x_test


def majority_vote_with_proba(
    combo: tuple[str, ...],
    pred_labels: dict[str, np.ndarray],
    pred_probas: dict[str, np.ndarray],
    all_classes: list[str],
) -> np.ndarray:
    out = []
    for i in range(len(next(iter(pred_labels.values())))):
        votes = [pred_labels[name][i] for name in combo]
        counts = Counter(votes)
        max_votes = max(counts.values())
        top = [label for label, count in counts.items() if count == max_votes]
        if len(top) == 1:
            out.append(top[0])
            continue
        score = {label: 0.0 for label in top}
        for name in combo:
            for class_idx, label in enumerate(all_classes):
                if label in score:
                    score[label] += float(pred_probas[name][i, class_idx])
        out.append(max(score.items(), key=lambda kv: (kv[1], kv[0]))[0])
    return np.array(out)


def load_split(seed_root: Path, split: str, code: str, translated: bool) -> pd.DataFrame:
    lang = LANG_NAMES[code]
    if translated:
        return read_jsonl(seed_root / "translated" / f"{split}_{lang}_translated.jsonl")
    return read_jsonl(seed_root / "cleaned" / f"{split}_{lang}.jsonl")


def run_experiment_job(job: dict, suite_dir: Path, logger) -> dict:
    seed = job["seed"]
    seed_root = suite_dir / "seed_data" / f"seed_{seed}"
    translated = job["translated"] == "yes"
    method = job["method"]
    training = job["training"]
    task = job["task"]
    test_code = job["test_language"]

    text_col = "translated" if translated else "Text_interviewer_participant"

    train_parts = {code: load_split(seed_root, "train", code, translated) for code in LANG_CODES}
    test_parts = {code: load_split(seed_root, "test", code, translated) for code in LANG_CODES}

    if training == "mono":
        train_df = train_parts[test_code].copy()
    else:
        train_df = pd.concat([train_parts[code] for code in LANG_CODES], ignore_index=True)

    test_df = test_parts[test_code].copy()

    train_df["Diagnosis"] = train_df["Diagnosis"].replace({"AD": "Dementia"})
    test_df["Diagnosis"] = test_df["Diagnosis"].replace({"AD": "Dementia"})

    if task == "binary":
        train_df = train_df[train_df["Diagnosis"] != "MCI"].reset_index(drop=True)
        test_df = test_df[test_df["Diagnosis"] != "MCI"].reset_index(drop=True)

    y_train = train_df["Diagnosis"].astype(str).to_numpy()
    y_test = test_df["Diagnosis"].astype(str).to_numpy()
    all_classes = sorted(pd.unique(pd.Series(y_train)))

    logger(
        f"{job['id']}: method={method} seed={seed} training={training} translated={job['translated']} "
        f"task={task} test={test_code} train={len(train_df)} test={len(test_df)}"
    )

    cache_dir = suite_dir / "cache" / method
    cache_stem = f"seed{seed}_{training}_{task}_{job['translated']}_{test_code}"
    x_train, x_test = prepare_features(
        method=method,
        train_df=train_df,
        test_df=test_df,
        text_col=text_col,
        cache_dir=cache_dir,
        cache_stem=cache_stem,
        logger=logger,
    )

    classifier_specs = build_classifier_specs(seed)
    classifier_results = {}
    pred_labels = {}
    pred_probas = {}

    for clf_name, (clf, params) in classifier_specs.items():
        if should_use_fast_svm_fallback(job, clf_name):
            clf = SVC(random_state=seed, class_weight="balanced", probability=False)
            logger(f"{job['id']}: using fast SVM probability fallback for this job")
        logger(f"{job['id']}: fitting {clf_name}")
        grid = GridSearchCV(clf, params, cv=5, scoring="accuracy", n_jobs=1)
        grid.fit(x_train, y_train)
        pred = grid.predict(x_test)
        proba = estimator_proba(grid.best_estimator_, x_test, all_classes)
        acc = float(accuracy_score(y_test, pred))

        classifier_results[clf_name] = {
            "accuracy": acc,
            "best_params": grid.best_params_,
        }
        pred_labels[clf_name] = pred
        pred_probas[clf_name] = proba
        logger(f"{job['id']}: {clf_name} accuracy={acc:.4f} params={grid.best_params_}")

    fusion_results = {}
    for combo in FUSION_COMBOS:
        combo_name = "+".join(combo)
        fused = majority_vote_with_proba(combo, pred_labels, pred_probas, all_classes)
        acc = float(accuracy_score(y_test, fused))
        fusion_results[combo_name] = {"accuracy": acc}
        logger(f"{job['id']}: fusion {combo_name} accuracy={acc:.4f}")

    return {
        "job": job,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "train_label_dist": {k: int(v) for k, v in pd.Series(y_train).value_counts().to_dict().items()},
        "test_label_dist": {k: int(v) for k, v in pd.Series(y_test).value_counts().to_dict().items()},
        "classifiers": classifier_results,
        "fusion": fusion_results,
    }


def build_jobs(seeds: list[int], smoke: bool = False) -> list[dict]:
    jobs = []
    for seed in seeds:
        jobs.append({"id": f"prepare_seed_{seed}", "kind": "prepare_seed", "seed": seed})

    if smoke:
        jobs.extend(
            [
                {
                    "id": "smoke_tfidf_seed42_cha_multiclass_translated_combined",
                    "kind": "experiment",
                    "seed": 42,
                    "method": "tfidf",
                    "training": "multi",
                    "translated": "yes",
                    "task": "multiclass",
                    "test_language": "cha",
                    "setting_name": "translated_combined",
                },
                {
                    "id": "smoke_e5_seed42_en_binary_monolingual",
                    "kind": "experiment",
                    "seed": 42,
                    "method": "e5",
                    "training": "mono",
                    "translated": "no",
                    "task": "binary",
                    "test_language": "en",
                    "setting_name": "monolingual",
                },
            ]
        )
        jobs.append({"id": "aggregate_smoke", "kind": "aggregate", "smoke": True})
        return jobs

    for seed in seeds:
        for method in METHODS:
            for setting in SETTINGS:
                for task in TASKS:
                    for test_language in LANG_CODES:
                        jobs.append(
                            {
                                "id": f"{method}_seed{seed}_{setting['name']}_{task}_{test_language}",
                                "kind": "experiment",
                                "seed": seed,
                                "method": method,
                                "training": setting["training"],
                                "translated": setting["translated"],
                                "task": task,
                                "test_language": test_language,
                                "setting_name": setting["name"],
                            }
                        )
    jobs.append({"id": "aggregate_full", "kind": "aggregate", "smoke": False})
    return jobs


def aggregate_results(suite_dir: Path, smoke: bool, logger) -> None:
    raw_dir = suite_dir / "raw_results"
    files = sorted(raw_dir.glob("*.json"))
    rows = [load_json(path) for path in files]

    if smoke:
        summary_path = suite_dir / "summaries" / "smoke_summary.json"
        json_dump(summary_path, rows)
        logger(f"Wrote smoke summary: {summary_path}")
        return

    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        job = row["job"]
        key = (job["method"], job["setting_name"], job["task"], job["test_language"])
        for clf_name, payload in row["classifiers"].items():
            grouped[key][clf_name].append(payload["accuracy"])
        for combo_name, payload in row["fusion"].items():
            grouped[key][combo_name].append(payload["accuracy"])

    out_dir = suite_dir / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    for method in METHODS:
        for setting in [cfg["name"] for cfg in SETTINGS]:
            for task in TASKS:
                table_rows = []
                for code in LANG_CODES:
                    key = (method, setting, task, code)
                    metrics = grouped.get(key)
                    if not metrics:
                        continue
                    row = {
                        "Language": LANG_LABELS[code],
                        "Task": task,
                    }
                    for name in CLASSIFIERS + ["+".join(combo) for combo in FUSION_COMBOS]:
                        vals = metrics.get(name, [])
                        row[name] = pct_str(vals) if vals else ""
                    table_rows.append(row)

                if not table_rows:
                    continue

                df = pd.DataFrame(table_rows)
                stem = f"{method}_{setting}_{task}"
                csv_path = out_dir / f"{stem}.csv"
                md_path = out_dir / f"{stem}.md"
                txt_path = out_dir / f"{stem}.txt"
                df.to_csv(csv_path, index=False)
                txt_path.write_text(df.to_string(index=False), encoding="utf-8")
                try:
                    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
                    logger(f"Wrote summary tables: {csv_path.name}, {md_path.name}, {txt_path.name}")
                except ImportError:
                    md_path.write_text(df.to_string(index=False), encoding="utf-8")
                    logger(
                        f"Wrote summary tables without tabulate markdown dependency: "
                        f"{csv_path.name}, {md_path.name}, {txt_path.name}"
                    )


def default_logger_factory(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"[{now()}] {message}"
        safe_print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log


def run_queue(args) -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_project_env(project_root)
    suite_dir = project_root / "tables" / "01-baselines" / "embedding-baselines" / "multiseed-suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    state_path = suite_dir / "summaries" / ("smoke_state.json" if args.command == "smoke" else "state.json")
    log_path = suite_dir / "logs" / ("smoke.log" if args.command == "smoke" else "run.log")
    logger = default_logger_factory(log_path)

    jobs = build_jobs(SEEDS if not args.seeds else args.seeds, smoke=(args.command == "smoke"))
    job_map = {job["id"]: job for job in jobs}

    if state_path.exists() and args.resume:
        state = load_json(state_path)
    else:
        state = {"created_at": now(), "jobs": {}}

    full_clean, full_translated = load_full_corpora(project_root)

    for job in jobs:
        job_state = state["jobs"].get(job["id"], {})
        if job_state.get("status") in {"done", "skipped"}:
            continue

        state["jobs"][job["id"]] = {"status": "running", "started_at": now(), "job": job}
        json_dump(state_path, state)

        logger(f"Starting job {job['id']}")
        job_log = default_logger_factory((suite_dir / "job_logs" / f"{job['id']}.log"))

        try:
            if job["kind"] == "prepare_seed":
                payload = prepare_seed_data(job["seed"], suite_dir, full_clean, full_translated, job_log)
                state["jobs"][job["id"]]["result"] = payload
            elif job["kind"] == "experiment":
                payload = run_experiment_job(job, suite_dir, job_log)
                raw_path = suite_dir / "raw_results" / f"{job['id']}.json"
                json_dump(raw_path, payload)
                state["jobs"][job["id"]]["result_path"] = str(raw_path)
            elif job["kind"] == "aggregate":
                aggregate_results(suite_dir, smoke=job.get("smoke", False), logger=job_log)
            else:
                raise RuntimeError(f"Unknown job kind: {job['kind']}")
            state["jobs"][job["id"]]["status"] = "done"
            state["jobs"][job["id"]]["finished_at"] = now()
            json_dump(state_path, state)
            logger(f"Finished job {job['id']}")
        except Exception as exc:
            state["jobs"][job["id"]]["status"] = "failed"
            state["jobs"][job["id"]]["finished_at"] = now()
            state["jobs"][job["id"]]["error"] = f"{exc.__class__.__name__}: {exc}"
            json_dump(state_path, state)
            logger(f"Job failed {job['id']}: {exc.__class__.__name__}: {exc}")
            raise

    logger("All jobs complete.")


def show_status(args) -> None:
    project_root = Path(__file__).resolve().parents[1]
    suite_dir = project_root / "tables" / "01-baselines" / "embedding-baselines" / "multiseed-suite"
    state_path = suite_dir / "summaries" / ("smoke_state.json" if args.smoke else "state.json")
    if not state_path.exists():
        safe_print(f"No state file at {state_path}")
        return
    state = load_json(state_path)
    counts = Counter(entry["status"] for entry in state["jobs"].values())
    safe_print(f"State file: {state_path}")
    for status in ["done", "running", "failed"]:
        safe_print(f"{status}: {counts.get(status, 0)}")
    safe_print(f"skipped: {counts.get('skipped', 0)}")
    pending = max(0, len(build_jobs(SEEDS if not args.seeds else args.seeds, smoke=args.smoke)) - len(state["jobs"]))
    safe_print(f"pending: {pending}")


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--seeds", nargs="*", type=int)

    run = sub.add_parser("run")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--seeds", nargs="*", type=int)

    status = sub.add_parser("status")
    status.add_argument("--smoke", action="store_true")
    status.add_argument("--seeds", nargs="*", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command in {"smoke", "run"}:
        run_queue(args)
    elif args.command == "status":
        show_status(args)
