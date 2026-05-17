from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_SUITE_DIR = ROOT / "tables" / "01-baselines" / "embedding-baselines" / "multiseed-suite"
OUT_ROOT = ROOT / "tables" / "01-baselines" / "transfer-learning-baselines"
RESULT_DIR = OUT_ROOT / "result-tables"
CSV_DIR = RESULT_DIR / "csv"
SUMMARY_DIR = OUT_ROOT / "summaries"
LOG_DIR = OUT_ROOT / "logs"
CACHE_DIR = OUT_ROOT / "cache"

LANG_CODES = ["en", "gr", "cha", "spa"]
LANG_NAMES = {"en": "english", "gr": "greek", "cha": "chinese", "spa": "spanish"}
LANG_LABELS = {"en": "English", "gr": "Greek", "cha": "Chinese", "spa": "Spanish"}
TASK_LABELS = {"binary": "Binary", "multiclass": "Multiclass"}
SETTING_LABELS = {
    "mono:no": "Monolingual",
    "multi:no": "Multilingual-Combined",
    "multi:yes": "Translated-Combined",
}
CLASSIFIERS = ["DT", "RF", "SVM", "LR"]
FUSION_COMBOS = [
    combo
    for size in range(2, len(CLASSIFIERS) + 1)
    for combo in itertools.combinations(CLASSIFIERS, size)
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_print(*parts) -> None:
    text = " ".join(str(part) for part in parts)
    text = text.encode("ascii", "backslashreplace").decode("ascii")
    print(text, flush=True)


def ensure_dirs() -> None:
    for path in [RESULT_DIR, CSV_DIR, SUMMARY_DIR, LOG_DIR, CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def append_log(message: str) -> None:
    ensure_dirs()
    with (LOG_DIR / "transfer_learning_embeddings.log").open("a", encoding="utf-8") as f:
        f.write(f"{now()} {message}\n")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def seed_root(seed: int) -> Path:
    path = EMBEDDING_SUITE_DIR / "seed_data" / f"seed_{seed}"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing seed split data: {path}. Run experiments/multiseed_suite.py prepare first."
        )
    return path


def load_split(seed: int, split: str, lang_code: str, translated: bool) -> pd.DataFrame:
    root = seed_root(seed)
    lang = LANG_NAMES[lang_code]
    if translated:
        return read_jsonl(root / "translated" / f"{split}_{lang}_translated.jsonl")
    return read_jsonl(root / "cleaned" / f"{split}_{lang}.jsonl")


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Diagnosis"] = out["Diagnosis"].replace({"AD": "Dementia"})
    return out


def balanced_cap(df: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    if cap is None or len(df) <= cap:
        return df.reset_index(drop=True)
    groups = list(df.groupby("Diagnosis", sort=True))
    per_group = max(1, cap // max(1, len(groups)))
    pieces = []
    sampled_indices = []
    for _, group in groups:
        piece = group.sample(n=min(len(group), per_group), random_state=seed)
        pieces.append(piece)
        sampled_indices.extend(piece.index.tolist())
    sampled = pd.concat(pieces, ignore_index=False)
    remaining = cap - len(sampled)
    if remaining > 0:
        rest = df.drop(index=sampled_indices, errors="ignore")
        if len(rest) > 0:
            sampled = pd.concat(
                [sampled, rest.sample(n=min(remaining, len(rest)), random_state=seed)],
                ignore_index=False,
            )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_data(args) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    translated = args.translated == "yes"
    train_parts = {code: load_split(args.seed, "train", code, translated) for code in LANG_CODES}
    test_parts = {code: load_split(args.seed, "test", code, translated) for code in LANG_CODES}

    if args.training == "mono":
        train_df = train_parts[args.test_language].copy()
    else:
        train_df = pd.concat([train_parts[code] for code in LANG_CODES], ignore_index=True)
    test_df = test_parts[args.test_language].copy()

    train_df = normalize_labels(train_df)
    test_df = normalize_labels(test_df)

    if args.task == "binary":
        train_df = train_df[train_df["Diagnosis"] != "MCI"].reset_index(drop=True)
        test_df = test_df[test_df["Diagnosis"] != "MCI"].reset_index(drop=True)

    if args.smoke:
        train_df = balanced_cap(train_df, args.max_train or 96, args.seed)
        test_df = balanced_cap(test_df, args.max_test or 48, args.seed)
    else:
        train_df = balanced_cap(train_df, args.max_train, args.seed)
        test_df = balanced_cap(test_df, args.max_test, args.seed)

    text_col = "translated" if translated else "Text_interviewer_participant"
    return train_df, test_df, text_col


def text_fingerprint(df: pd.DataFrame, text_col: str) -> str:
    raw = "||".join(df[text_col].fillna("").astype(str).tolist())
    import hashlib

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def cache_path(args, split: str, df: pd.DataFrame, text_col: str) -> Path:
    model_tag = slug(args.model)
    fp = text_fingerprint(df, text_col)
    lang_tag = args.test_language
    if split == "train" and args.training == "multi":
        lang_tag = "all"
    name = (
        f"seed{args.seed}_{args.training}_{args.translated}_{args.task}_{lang_tag}_"
        f"{split}_{args.pooling}_{args.length_mode}_{fp}.npy"
    )
    return CACHE_DIR / model_tag / args.pooling / args.length_mode / name


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def encode_batch(model, tokenizer, texts: list[str], pooling: str, device: str, max_length: int) -> np.ndarray:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    if pooling == "cls":
        arr = outputs.last_hidden_state[:, 0, :]
    elif pooling == "mean":
        arr = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
    else:
        raise ValueError(f"Unsupported pooling mode: {pooling}")
    return arr.detach().cpu().numpy()


def encode_chunked_text(model, tokenizer, text: str, pooling: str, device: str, max_length: int) -> np.ndarray:
    token_ids = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    chunk_size = max_length - 2
    if not token_ids:
        token_ids = []
    chunks = [token_ids[i : i + chunk_size] for i in range(0, len(token_ids), chunk_size)] or [[]]
    vectors = []
    for chunk in chunks:
        encoded = tokenizer.prepare_for_model(
            chunk,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        if pooling == "cls":
            vec = outputs.last_hidden_state[:, 0, :]
        else:
            vec = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        vectors.append(vec.detach().cpu().numpy()[0])
    return np.mean(np.vstack(vectors), axis=0)


def encode_embeddings(df: pd.DataFrame, text_col: str, args, split: str) -> np.ndarray:
    path = cache_path(args, split, df, text_col)
    if path.exists():
        arr = np.load(path)
        if arr.shape[0] == len(df):
            append_log(f"cache hit {path}")
            return arr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    append_log(f"loading model={args.model} device={device}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModel.from_pretrained(args.model, local_files_only=True).to(device)
    except Exception as exc:
        append_log(f"local model load missed for {args.model}; falling back to remote load: {exc}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    texts = df[text_col].fillna("").astype(str).tolist()
    vectors = []
    append_log(f"encoding split={split} rows={len(texts)} pooling={args.pooling} length_mode={args.length_mode}")
    if args.length_mode == "truncate":
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            vectors.append(encode_batch(model, tokenizer, batch, args.pooling, device, args.max_length))
    elif args.length_mode == "chunk_mean":
        for text in texts:
            vectors.append(encode_chunked_text(model, tokenizer, text, args.pooling, device, args.max_length)[None, :])
    else:
        raise ValueError(f"Unsupported length mode: {args.length_mode}")

    arr = np.vstack(vectors)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arr


def classifier_specs(seed: int, smoke: bool):
    if smoke:
        return {
            "DT": (DecisionTreeClassifier(random_state=seed), {"max_depth": [10]}),
            "RF": (RandomForestClassifier(random_state=seed), {"n_estimators": [50]}),
            "SVM": (SVC(random_state=seed, class_weight="balanced", probability=True), {"C": [1], "kernel": ["linear"]}),
            "LR": (LogisticRegression(random_state=seed, max_iter=1000), {"C": [1]}),
        }
    return {
        "DT": (DecisionTreeClassifier(random_state=seed), {"max_depth": [10, 20, 30]}),
        "RF": (RandomForestClassifier(random_state=seed), {"n_estimators": [50, 100, 200]}),
        "SVM": (
            SVC(random_state=seed, class_weight="balanced", probability=True),
            {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        ),
        "LR": (LogisticRegression(random_state=seed, max_iter=1000), {"C": [0.1, 1, 10]}),
    }


def aligned_proba(proba: np.ndarray, model_classes: np.ndarray, all_classes: list[str]) -> np.ndarray:
    out = np.zeros((proba.shape[0], len(all_classes)), dtype=float)
    idx_map = {label: i for i, label in enumerate(all_classes)}
    for src_idx, cls in enumerate(model_classes):
        out[:, idx_map[str(cls)]] = proba[:, src_idx]
    return out


def majority_vote(combo: tuple[str, ...], predictions: dict[str, np.ndarray], probas: dict[str, np.ndarray], classes: list[str]):
    out = []
    for row_idx in range(len(next(iter(predictions.values())))):
        votes = [predictions[name][row_idx] for name in combo]
        counts = Counter(votes)
        max_votes = max(counts.values())
        top = [label for label, count in counts.items() if count == max_votes]
        if len(top) == 1:
            out.append(top[0])
            continue
        scores = {label: 0.0 for label in top}
        for name in combo:
            for class_idx, label in enumerate(classes):
                if label in scores:
                    scores[label] += float(probas[name][row_idx, class_idx])
        out.append(max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0])
    return np.array(out)


def upsert_rows(path: Path, rows: list[dict], key_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if path.exists():
        old_df = pd.read_csv(path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.sort_values("timestamp").drop_duplicates(subset=key_columns, keep="last")
    combined.to_csv(path, index=False)


def upsert_confusion_rows(path: Path, rows: list[dict]) -> None:
    field_names = [
        "timestamp",
        "smoke",
        "seed",
        "model",
        "pooling",
        "length_mode",
        "training",
        "translated",
        "task",
        "test_language",
        "classifier",
        "actual",
        "predicted",
        "count",
    ]
    new_df = pd.DataFrame(rows, columns=field_names)
    if path.exists():
        old_df = pd.read_csv(path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    keys = [
        "smoke",
        "seed",
        "model",
        "pooling",
        "length_mode",
        "training",
        "translated",
        "task",
        "test_language",
        "classifier",
        "actual",
        "predicted",
    ]
    combined = combined.sort_values("timestamp").drop_duplicates(subset=keys, keep="last")
    combined.to_csv(path, index=False)


def fmt_pct(value: float) -> str:
    return f"{value * 100.0:.1f}"


def regenerate_readable_table(csv_path: Path, out_path: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    df = df[df["classifier"].isin(CLASSIFIERS + ["Best Ensemble"])].copy()
    if df.empty:
        return

    lines = [
        "Frozen PLM Transfer-Learning Baselines",
        "======================================",
        "Cells are accuracies as mean +/- sd over completed seeds.",
        "The layout mirrors paper_vs_ours_3tables.txt for transfer-learning models.",
        "Smoke rows are implementation checks and should not be used as final results.",
        "",
    ]

    for setting in ["Monolingual", "Multilingual-Combined", "Translated-Combined"]:
        part = df[df["setting"] == setting]
        if part.empty:
            continue
        lines.extend([setting, "=" * len(setting)])
        columns = ["Task", "Language", "Model", "Pool", "Length", "DT", "RF", "SVM", "LR", "Best Ensemble", "Best Combo"]
        rows = []
        keys = ["task", "test_language", "model", "pooling", "length_mode", "smoke"]
        for key, group in part.groupby(keys, dropna=False):
            task, lang, model, pooling, length_mode, smoke = key
            row = {
                "Task": TASK_LABELS.get(task, task),
                "Language": LANG_LABELS.get(lang, lang),
                "Model": model,
                "Pool": pooling,
                "Length": length_mode,
                "DT": "",
                "RF": "",
                "SVM": "",
                "LR": "",
                "Best Ensemble": "",
                "Best Combo": "",
            }
            class_rows = group[group["classifier"].isin(CLASSIFIERS)]
            for clf_name in CLASSIFIERS:
                values = class_rows[class_rows["classifier"] == clf_name]["accuracy"].astype(float).to_numpy()
                if len(values):
                    row[clf_name] = f"{values.mean() * 100.0:.1f} +/- {values.std(ddof=0) * 100.0:.1f}"
            ensemble = group[group["classifier"] == "Best Ensemble"]
            if not ensemble.empty:
                values = ensemble["accuracy"].astype(float).to_numpy()
                row["Best Ensemble"] = f"{values.mean() * 100.0:.1f} +/- {values.std(ddof=0) * 100.0:.1f}"
                combo_counts = ensemble["best_params"].astype(str).value_counts()
                row["Best Combo"] = str(combo_counts.index[0])
            if bool(smoke):
                row["Model"] = f"{row['Model']} [smoke]"
            rows.append(row)

        widths = {
            col: max(len(col), *(len(str(row[col])) for row in rows))
            for col in columns
        }
        lines.append(" | ".join(col.ljust(widths[col]) for col in columns))
        lines.append("-+-".join("-" * widths[col] for col in columns))
        for row in rows:
            lines.append(" | ".join(str(row[col]).ljust(widths[col]) for col in columns))
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run(args) -> dict:
    ensure_dirs()
    train_df, test_df, text_col = build_data(args)
    y_train = train_df["Diagnosis"].astype(str).to_numpy()
    y_test = test_df["Diagnosis"].astype(str).to_numpy()
    classes = sorted(pd.unique(pd.Series(y_train)))

    min_class = min(Counter(y_train).values())
    cv = min(5, min_class)
    if cv < 2:
        raise RuntimeError(f"Not enough training rows per class for CV: {Counter(y_train)}")

    setting = SETTING_LABELS[f"{args.training}:{args.translated}"]
    append_log(
        f"run start model={args.model} seed={args.seed} setting={setting} task={args.task} "
        f"test={args.test_language} train={len(train_df)} test={len(test_df)} smoke={args.smoke}"
    )

    x_train = encode_embeddings(train_df, text_col, args, "train")
    x_test = encode_embeddings(test_df, text_col, args, "test")

    timestamp = now()
    metric_rows = []
    confusion_rows = []
    predictions = {}
    probas = {}
    reports = {}

    for clf_name, (clf, params) in classifier_specs(args.seed, args.smoke).items():
        append_log(f"fitting {clf_name} cv={cv}")
        grid = GridSearchCV(clf, params, cv=cv, scoring="accuracy", n_jobs=1)
        grid.fit(x_train, y_train)
        pred = grid.predict(x_test)
        proba = aligned_proba(grid.best_estimator_.predict_proba(x_test), grid.best_estimator_.classes_, classes)
        predictions[clf_name] = pred
        probas[clf_name] = proba
        acc = float(accuracy_score(y_test, pred))
        macro = float(f1_score(y_test, pred, average="macro", zero_division=0))
        reports[clf_name] = classification_report(y_test, pred, zero_division=0)
        metric_rows.append(
            {
                "timestamp": timestamp,
                "smoke": args.smoke,
                "seed": args.seed,
                "model": args.model,
                "pooling": args.pooling,
                "length_mode": args.length_mode,
                "training": args.training,
                "translated": args.translated,
                "setting": setting,
                "task": args.task,
                "test_language": args.test_language,
                "language_label": LANG_LABELS[args.test_language],
                "text_col": text_col,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "train_label_dist": json.dumps(Counter(y_train), sort_keys=True),
                "test_label_dist": json.dumps(Counter(y_test), sort_keys=True),
                "classifier": clf_name,
                "accuracy": acc,
                "macro_f1": macro,
                "best_params": json.dumps(grid.best_params_, sort_keys=True),
            }
        )
        cm = confusion_matrix(y_test, pred, labels=classes)
        for actual_idx, actual in enumerate(classes):
            for pred_idx, predicted in enumerate(classes):
                confusion_rows.append(
                    {
                        "timestamp": timestamp,
                        "smoke": args.smoke,
                        "seed": args.seed,
                        "model": args.model,
                        "pooling": args.pooling,
                        "length_mode": args.length_mode,
                        "training": args.training,
                        "translated": args.translated,
                        "task": args.task,
                        "test_language": args.test_language,
                        "classifier": clf_name,
                        "actual": actual,
                        "predicted": predicted,
                        "count": int(cm[actual_idx, pred_idx]),
                    }
                )
        safe_print(f"{clf_name}: accuracy={acc:.4f} macro_f1={macro:.4f} params={grid.best_params_}")

    best_combo = None
    best_combo_acc = -1.0
    for combo in FUSION_COMBOS:
        fused = majority_vote(combo, predictions, probas, classes)
        acc = float(accuracy_score(y_test, fused))
        if acc > best_combo_acc:
            best_combo_acc = acc
            best_combo = " + ".join(combo)
    metric_rows.append(
        {
            "timestamp": timestamp,
            "smoke": args.smoke,
            "seed": args.seed,
            "model": args.model,
            "pooling": args.pooling,
            "length_mode": args.length_mode,
            "training": args.training,
            "translated": args.translated,
            "setting": setting,
            "task": args.task,
            "test_language": args.test_language,
            "language_label": LANG_LABELS[args.test_language],
            "text_col": text_col,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_label_dist": json.dumps(Counter(y_train), sort_keys=True),
            "test_label_dist": json.dumps(Counter(y_test), sort_keys=True),
            "classifier": "Best Ensemble",
            "accuracy": best_combo_acc,
            "macro_f1": "",
            "best_params": best_combo,
        }
    )
    safe_print(f"Best Ensemble: accuracy={best_combo_acc:.4f} combo={best_combo}")

    metrics_path = CSV_DIR / "frozen_embedding_runs.csv"
    confusion_path = CSV_DIR / "frozen_embedding_confusion_matrices.csv"
    upsert_rows(
        metrics_path,
        metric_rows,
        [
            "smoke",
            "seed",
            "model",
            "pooling",
            "length_mode",
            "training",
            "translated",
            "task",
            "test_language",
            "classifier",
        ],
    )
    upsert_confusion_rows(confusion_path, confusion_rows)
    regenerate_readable_table(metrics_path, RESULT_DIR / "transfer_learning_results.txt")

    summary = {
        "timestamp": timestamp,
        "script": "experiments/transfer_learning_embeddings.py",
        "smoke": args.smoke,
        "seed": args.seed,
        "model": args.model,
        "pooling": args.pooling,
        "length_mode": args.length_mode,
        "training": args.training,
        "translated": args.translated,
        "setting": setting,
        "task": args.task,
        "test_language": args.test_language,
        "text_col": text_col,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_label_dist": Counter(y_train),
        "test_label_dist": Counter(y_test),
        "results_csv": str(metrics_path),
        "confusion_csv": str(confusion_path),
        "readable_results": str(RESULT_DIR / "transfer_learning_results.txt"),
        "classification_reports": reports,
    }
    with (SUMMARY_DIR / "frozen_embedding_latest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    append_log("run complete")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen BERT/RoBERTa embedding baselines for MultiConAD.")
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--length_mode", choices=["truncate", "chunk_mean"], default="truncate")
    parser.add_argument("--task", choices=["binary", "multiclass"], required=True)
    parser.add_argument("--training", choices=["mono", "multi"], required=True)
    parser.add_argument("--test_language", choices=LANG_CODES, required=True)
    parser.add_argument("--translated", choices=["yes", "no"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    summary = run(args)
    safe_print(f"Results: {summary['readable_results']}")
    safe_print(f"CSV: {summary['results_csv']}")


if __name__ == "__main__":
    main()
